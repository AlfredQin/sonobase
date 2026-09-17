"""B2 standalone — iterative-correction sweep with cached image-encoder.

Why this script exists
----------------------
Both ``test_sam2.py`` (benchmark recipe) and ``test.py`` (per-dataset
sonobase test recipe) accept a single ``num_correction_pt_per_frame_val``
value per run. To produce metrics across an iteration sweep
``[0, 1, 3, 5, 7]`` (the click-efficiency / human-correction curves used
in the A3 analysis) the **default ("B1") workflow** simply invokes the
existing recipe N times (one process per iteration count), then folds the
N ``test_metrics.json`` outputs into a single long-format CSV via
``aggregate_iterations.py``.

This standalone script implements an alternative ("B2") workflow: it
runs the **image-encoder once per batch** and then loops the prompt
preparation + tracking head over the same iteration set, mutating
``num_correction_pt_per_frame_val`` between iterations and reusing the
cached encoder features. Because the image encoder dominates inference
cost, B2 cuts wall time roughly proportional to len(iterations) — for
N=5 iterations it is ~3-4× faster than B1 while producing the *same*
long-format CSV schema.

We keep B1 as the default for production sweeps because it exercises the
exact same code path as a single-iteration test run (no risk of state
leakage between iterations, identical RNG semantics) and because the
per-rank wall time of a single test run is already small. B2 is provided
for future verification: by running both pipelines on a small dataset
(e.g. ``busi_camus``) and diffing the resulting per-iteration CSVs we
can confirm B2 introduces no systematic bias before promoting it to a
default.

Important difference from B1
----------------------------
For B2 to genuinely cache the encoder we MUST set
``model.forward_backbone_per_frame_for_eval=false`` so that
``forward_image`` is called upfront on the whole flat image batch. The
default test config ships with this flag set to ``true`` (per-frame
on-demand), and the B2 config (``configs/test_sam2/iterative_eval.yaml``)
overrides it to ``false``. This is the only architectural difference
between the two paths; the actual encoder weights and the math performed
are unchanged.

Numerical equivalence (or lack thereof)
---------------------------------------
B1 and B2 sample prompts via ``self.rng`` inside
``prepare_prompt_inputs``. B1 starts each run with a fresh per-rank seed
(see ``ScopedRNG(seed=1, ranked=True)`` in ``test_sam2.py``), so the RNG
sequence consumed for each iteration count is independent. B2 advances
the same RNG sequence across iterations (iter-0 prompts are sampled
first, then iter-1 prompts, etc.). Per-sample predictions therefore
differ between B1 and B2, but per-dataset aggregate metrics over hundreds
of samples should agree to within Monte-Carlo noise (~1% relative). The
``compare_b1_vs_b2.sh`` helper performs exactly this comparison.

CLI
---
``CHECKPOINT_DIR`` must be set in env (e.g. ``export CHECKPOINT_DIR=$WORK/Checkpoints``)::

    uv run torchrun --nproc-per-node=N -m nemo_cv.recipes.benchmarks.iterative_eval \\
        -c ./configs/test_sam2 -cn iterative_eval \\
        experiment=test_sam2_no_ft_on_busi_camus \\
        ckpt_path=${CHECKPOINT_DIR}/SAM2/sam2.1_hiera_base_plus.pt
"""

import csv
import gc
import json
import logging
import pathlib
import time
from typing import Dict, List, Optional

import torch
from hydra.utils import instantiate as hydra_instantiate
from omegaconf import OmegaConf

from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.loggers.metric_logger import MetricsSample

from nemo_cv.components.training.utils import (
    CORE_LOSS_KEY,
    barrier,
    unwrap_ddp_if_wrapped,
)
from nemo_cv.recipes.benchmarks.test_sam2 import (
    TestSam2BenchmarkRecipe,
    _build_autocast,
    _select_test_data,
)
try:
    from nemo_automodel.components.training.rng import ScopedRNG
except Exception:  # pragma: no cover -- nemo_automodel optional
    ScopedRNG = None  # type: ignore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Recipe
# ---------------------------------------------------------------------------


class IterativeEvalRecipe(TestSam2BenchmarkRecipe):
    """Encoder-cached iterative-correction sweep ("B2").

    Subclasses :class:`TestSam2BenchmarkRecipe` to reuse model loading,
    DDP setup, dataset construction and metric instantiation. Only the
    eval loop is replaced — see :meth:`run_test_loop`.

    Required config additions vs ``test.yaml``:

    - ``iterations``: list of integer ``num_correction_pt_per_frame_val``
      values to sweep (e.g. ``[0, 1, 3, 5, 7]``).
    - ``model.forward_backbone_per_frame_for_eval``: must be ``false``
      so that the image encoder runs upfront and can be cached. The
      sibling ``configs/test_sam2/iterative_eval.yaml`` enforces this.
    """

    # ------------------ build phase ------------------
    def setup(self):
        super().setup()
        iterations_cfg = self.hydra_cfg.get("iterations", None)
        if iterations_cfg is None:
            raise ValueError(
                "iterative_eval requires `iterations: [int, ...]` at top level "
                "of the config (e.g. iterations: [0, 1, 3, 5, 7]). For a single "
                "iteration count use the standard test_sam2 recipe instead."
            )
        self.iterations: List[int] = sorted({int(x) for x in iterations_cfg})
        if not self.iterations:
            raise ValueError("`iterations` must contain at least one integer.")

        unwrapped = unwrap_ddp_if_wrapped(self.model)
        if getattr(unwrapped, "forward_backbone_per_frame_for_eval", False):
            raise ValueError(
                "iterative_eval requires `model.forward_backbone_per_frame_for_eval=false` "
                "so that the image encoder runs upfront and can be cached across "
                "iterations. Use the bundled `configs/test_sam2/iterative_eval.yaml` "
                "(which sets this) or override on the CLI."
            )

        # Replace the single test_metrics dict (per-dataset → metric_name → instance)
        # with an iteration-aware dict (per-dataset → iter → metric_name → instance).
        # The instances built by the parent are reused for iter==self.iterations[0]
        # to avoid throwing them away.
        val_metrics_cfg = self.hydra_cfg.get("val_metrics", None)
        if val_metrics_cfg is None:
            raise ValueError("`val_metrics` config is required (e.g. miou, dice).")
        self.iter_metrics: Dict[str, Dict[int, Dict[str, object]]] = {}
        for ds_name in self.test_datasets:
            self.iter_metrics[ds_name] = {}
            for n_corr in self.iterations:
                self.iter_metrics[ds_name][n_corr] = {
                    name: hydra_instantiate(val_metrics_cfg[name]).to(self.device)
                    for name in val_metrics_cfg
                }
        # Drop the parent's per-dataset metric dict — we own metric lifecycle now.
        self.test_metrics = {}

        if self.dist_env.is_main:
            logger.info(
                f"IterativeEvalRecipe ready — iterations={self.iterations}, "
                f"datasets={list(self.test_datasets)}, encoder_caching=ON"
            )

    # ------------------ eval loop ------------------
    @torch.no_grad()
    def run_test_loop(self):
        """Encoder-cached iterative-correction loop.

        Per batch we call ``forward_image`` exactly once and then re-enter
        ``prepare_prompt_inputs`` + ``forward_tracking`` for each
        iteration count, mutating ``num_correction_pt_per_frame_val`` in
        between. ``prepare_prompt_inputs`` only adds new keys to the
        backbone dict (it does not mutate ``backbone_fpn`` /
        ``vision_pos_enc`` tensors), so a shallow ``dict(...)`` copy is
        sufficient to keep the cached features intact.
        """
        self.model.eval()
        unwrapped = unwrap_ddp_if_wrapped(self.model)
        autocast_factory = lambda: _build_autocast(self.hydra_cfg.get("optim", None))

        # records[(ds_name, n_corr)] -> dict with n_samples + per-metric scalars + loss
        records: Dict[tuple, Dict[str, float]] = {}
        wall_per_ds: Dict[str, float] = {}

        for ds_name, dataset in self.test_datasets.items():
            barrier()
            if self.dist_env.is_main:
                logger.info(
                    f"--- Iterative eval on {ds_name} (iterations={self.iterations}) ---"
                )

            rng_ctx = (
                ScopedRNG(seed=1, ranked=True)
                if ScopedRNG is not None
                else _NullCtx()
            )
            with rng_ctx:
                loader = dataset.get_loader(epoch=0)

                # Per-iteration accumulators (loss + sample count). Metrics live
                # in self.iter_metrics and aggregate via torchmetrics.
                loss_sum_local: Dict[int, float] = {n: 0.0 for n in self.iterations}
                n_samples_local = 0
                n_iters_local = 0
                t_start = time.perf_counter()

                for data_iter, batch in enumerate(loader):
                    batch = batch.to(self.device, non_blocking=True)
                    with autocast_factory():
                        # ----- ENCODER (cached across iterations) -----
                        backbone_out_base = unwrapped.forward_image(batch.flat_img_batch)

                        for n_corr in self.iterations:
                            unwrapped.num_correction_pt_per_frame_val = int(n_corr)

                            # Shallow copy: prepare_prompt_inputs only adds keys.
                            bbout = dict(backbone_out_base)
                            bbout = unwrapped.prepare_prompt_inputs(bbout, batch)
                            outputs = unwrapped.forward_tracking(bbout, batch)

                            targets = batch.masks
                            if self.test_loss_fn is not None:
                                key = batch.dict_key
                                loss = self.test_loss_fn[key](outputs, targets)
                                if isinstance(loss, dict):
                                    loss = loss[CORE_LOSS_KEY]
                                loss_sum_local[n_corr] += float(loss.item()) * batch.num_videos

                            for m in self.iter_metrics[ds_name][n_corr].values():
                                m.update(outputs, batch)

                    n_samples_local += batch.num_videos
                    n_iters_local += 1

                    if self.dist_env.is_main and (data_iter % self.log_freq == 0):
                        elapsed = time.perf_counter() - t_start
                        logger.info(
                            f"[{ds_name}] iter {data_iter} | "
                            f"local_samples {n_samples_local} | "
                            f"iters_per_batch {len(self.iterations)} | "
                            f"elapsed {elapsed:.1f}s"
                        )
                    if data_iter % 10 == 0:
                        barrier()

                del loader
                gc.collect()

            # Aggregate scalars across DDP ranks (per iteration count)
            n_t = torch.tensor([n_samples_local], dtype=torch.long, device=self.device)
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(n_t)
            n_samples = int(n_t.item())

            ds_loss_global: Dict[int, Optional[float]] = {}
            for n_corr in self.iterations:
                if self.test_loss_fn is None:
                    ds_loss_global[n_corr] = None
                    continue
                lt = torch.tensor(
                    [loss_sum_local[n_corr]], dtype=torch.float64, device=self.device
                )
                if torch.distributed.is_initialized():
                    torch.distributed.all_reduce(lt)
                ds_loss_global[n_corr] = float(lt.item()) / max(n_samples, 1)

            # Compute per-iteration torchmetrics (sync internally) + record
            for n_corr in self.iterations:
                metric_values: Dict[str, float] = {}
                for name, m in self.iter_metrics[ds_name][n_corr].items():
                    metric_values[name] = float(m.compute().item())
                    m.reset()
                rec: Dict[str, float] = {
                    "n_samples": n_samples,
                    **metric_values,
                }
                if ds_loss_global[n_corr] is not None:
                    rec["test_loss"] = round(ds_loss_global[n_corr], 6)
                records[(ds_name, n_corr)] = rec

            elapsed = time.perf_counter() - t_start
            wall_per_ds[ds_name] = elapsed

            if self.dist_env.is_main:
                # Pretty-print one row per iteration count
                logger.info(
                    f"[{ds_name}] DONE | n_samples={n_samples} | "
                    f"elapsed={elapsed:.1f}s | per-iteration breakdown:"
                )
                for n_corr in self.iterations:
                    rec = records[(ds_name, n_corr)]
                    metric_str = " | ".join(
                        f"{k}={v:.4f}"
                        for k, v in rec.items()
                        if k not in ("n_samples", "test_loss")
                    )
                    loss_str = (
                        f" | test_loss={rec['test_loss']:.4f}"
                        if "test_loss" in rec
                        else ""
                    )
                    logger.info(
                        f"  iter={n_corr:>2d} | {metric_str}{loss_str}"
                    )
                    self.metric_logger_test.log(
                        MetricsSample(
                            step=int(n_corr),
                            epoch=0,
                            metrics={"dataset": ds_name, "iteration": int(n_corr), **rec},
                        )
                    )

        if self.dist_env.is_main:
            self._write_iteration_results(records, wall_per_ds)
            self.metric_logger_test.close()

        barrier()

    # ------------------ output writers ------------------
    def _write_iteration_results(
        self,
        records: Dict[tuple, Dict[str, float]],
        wall_per_ds: Dict[str, float],
    ):
        """Write the long-format ``test_metrics_iterations.{json,csv}``.

        The CSV is the canonical input to ``a3_click_efficiency.py`` and
        is what the ``aggregate_iterations.py`` helper produces from B1
        outputs. Both pipelines therefore share a single downstream
        consumer.

        Schema (long format):
            dataset,iteration,n_samples,<metric_1>,<metric_2>,...,test_loss
            ...
            AGGREGATE_MACRO,<iter>,<total_n>,...
            AGGREGATE_MICRO,<iter>,<total_n>,...
        """
        ordered_dsets = list(self.test_datasets.keys())
        bookkeeping = {"n_samples"}
        first_rec = records[(ordered_dsets[0], self.iterations[0])]
        metric_names = [k for k in first_rec.keys() if k not in bookkeeping and k != "test_loss"]
        has_loss = any("test_loss" in records[(ds, it)] for ds in ordered_dsets for it in self.iterations)

        # Per-iteration aggregates (macro = unweighted across datasets,
        # micro = sample-weighted)
        macro: Dict[int, Dict[str, float]] = {n: {} for n in self.iterations}
        micro: Dict[int, Dict[str, float]] = {n: {} for n in self.iterations}
        total_samples_per_iter: Dict[int, int] = {}
        for n_corr in self.iterations:
            valid = [(records[(ds, n_corr)], records[(ds, n_corr)]["n_samples"]) for ds in ordered_dsets]
            total_samples_per_iter[n_corr] = sum(n for _, n in valid)
            for name in metric_names:
                vals = [(r[name], n) for r, n in valid if r.get(name) is not None]
                if not vals:
                    continue
                macro[n_corr][name] = sum(v for v, _ in vals) / len(vals)
                denom = sum(n for _, n in vals)
                micro[n_corr][name] = sum(v * n for v, n in vals) / max(denom, 1)
            if has_loss:
                losses = [(r["test_loss"], n) for r, n in valid if "test_loss" in r]
                if losses:
                    macro[n_corr]["test_loss"] = sum(v for v, _ in losses) / len(losses)
                    denom = sum(n for _, n in losses)
                    micro[n_corr]["test_loss"] = sum(v * n for v, n in losses) / max(denom, 1)

        # JSON dump (machine-readable; nested by dataset)
        out = {
            "checkpoint": self.checkpoint_path,
            "iterations": self.iterations,
            "n_datasets": len(ordered_dsets),
            "datasets": {
                ds: {
                    str(n_corr): records[(ds, n_corr)] for n_corr in self.iterations
                }
                for ds in ordered_dsets
            },
            "aggregate_macro": {
                str(n_corr): macro[n_corr] for n_corr in self.iterations
            },
            "aggregate_micro": {
                str(n_corr): micro[n_corr] for n_corr in self.iterations
            },
            "elapsed_sec": {ds: round(wall_per_ds[ds], 2) for ds in ordered_dsets},
        }
        json_path = self.results_dir / "test_metrics_iterations.json"
        with json_path.open("w") as f:
            json.dump(out, f, indent=2)
        logger.info(f"Wrote {json_path}")

        # CSV dump (long format — A3 consumer expects this layout)
        csv_path = self.results_dir / "test_metrics_iterations.csv"
        loss_col = ["test_loss"] if has_loss else []
        with csv_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["dataset", "iteration", "n_samples"] + metric_names + loss_col)
            for ds in ordered_dsets:
                for n_corr in self.iterations:
                    r = records[(ds, n_corr)]
                    row = [ds, n_corr, r["n_samples"]] + [
                        round(r[m], 6) if m in r and r[m] is not None else ""
                        for m in metric_names
                    ]
                    if has_loss:
                        row.append(round(r["test_loss"], 6) if "test_loss" in r else "")
                    w.writerow(row)
            w.writerow([])
            for n_corr in self.iterations:
                row = ["AGGREGATE_MACRO", n_corr, total_samples_per_iter[n_corr]] + [
                    round(macro[n_corr].get(m, float("nan")), 6) for m in metric_names
                ]
                if has_loss:
                    row.append(round(macro[n_corr].get("test_loss", float("nan")), 6))
                w.writerow(row)
            for n_corr in self.iterations:
                row = ["AGGREGATE_MICRO", n_corr, total_samples_per_iter[n_corr]] + [
                    round(micro[n_corr].get(m, float("nan")), 6) for m in metric_names
                ]
                if has_loss:
                    row.append(round(micro[n_corr].get("test_loss", float("nan")), 6))
                w.writerow(row)
        logger.info(f"Wrote {csv_path}")

        # Pretty table (one block per iteration)
        col_w = max(24, max(len(n) for n in ordered_dsets) + 2) if ordered_dsets else 24
        for n_corr in self.iterations:
            header = (
                f"{'dataset':<{col_w}} {'n_samples':>10} "
                + " ".join(f"{n:>10}" for n in metric_names)
                + (f" {'test_loss':>10}" if has_loss else "")
            )
            sep = "-" * len(header)
            logger.info("=" * len(header))
            logger.info(
                f"ITERATION = {n_corr}  (checkpoint: {self.checkpoint_path})"
            )
            logger.info("=" * len(header))
            logger.info(header)
            logger.info(sep)
            for ds in ordered_dsets:
                r = records[(ds, n_corr)]
                row = (
                    f"{ds:<{col_w}} {r['n_samples']:>10} "
                    + " ".join(
                        f"{r.get(m, float('nan')):>10.4f}" for m in metric_names
                    )
                )
                if has_loss:
                    row += f" {r.get('test_loss', float('nan')):>10.4f}"
                logger.info(row)
            logger.info(sep)
            macro_row = (
                f"{'AGGREGATE_MACRO':<{col_w}} {total_samples_per_iter[n_corr]:>10} "
                + " ".join(f"{macro[n_corr].get(m, float('nan')):>10.4f}" for m in metric_names)
            )
            micro_row = (
                f"{'AGGREGATE_MICRO':<{col_w}} {total_samples_per_iter[n_corr]:>10} "
                + " ".join(f"{micro[n_corr].get(m, float('nan')):>10.4f}" for m in metric_names)
            )
            if has_loss:
                macro_row += f" {macro[n_corr].get('test_loss', float('nan')):>10.4f}"
                micro_row += f" {micro[n_corr].get('test_loss', float('nan')):>10.4f}"
            logger.info(macro_row)
            logger.info(micro_row)
            logger.info("=" * len(header))


class _NullCtx:
    """Tiny no-op context manager used when ``ScopedRNG`` is unavailable."""

    def __enter__(self):
        return self

    def __exit__(self, *a, **kw):
        return False


# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------


def main():
    """Entry point for the standalone iterative-eval recipe (B2)."""
    import sys

    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    config_dir = None
    config_name = "iterative_eval"
    overrides: List[str] = []
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] in ("-c", "--config-dir") and i + 1 < len(args):
            config_dir = str(pathlib.Path(args[i + 1]).resolve())
            i += 2
        elif args[i] in ("-cn", "--config-name") and i + 1 < len(args):
            config_name = args[i + 1]
            i += 2
        elif "=" in args[i] or args[i].startswith("+") or args[i].startswith("~"):
            overrides.append(args[i])
            i += 1
        else:
            i += 1

    if config_dir is None:
        config_dir = str(pathlib.Path(__file__).parent.resolve() / "configs")

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=config_dir, version_base=None)
    hydra_cfg = compose(config_name=config_name, overrides=overrides)
    OmegaConf.resolve(hydra_cfg)

    cfg_dict = OmegaConf.to_container(hydra_cfg, resolve=True)
    cfg_node = ConfigNode(cfg_dict)

    recipe = IterativeEvalRecipe(cfg_node, hydra_cfg)
    recipe.setup()
    recipe.run_test_loop()


if __name__ == "__main__":
    main()
