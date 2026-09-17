"""Benchmark test recipe for SAM2 / MedSAM2 / Sonobase comparison.

Loads a single-file `.pt` checkpoint (in SAM2's `{"model": state_dict}`
convention) into a SAM2-architecture model, then evaluates it on one or more
held-out datasets defined under `cfg.data.test` (or `cfg.data.val` as a
fallback). For each dataset it computes per-dataset segmentation metrics
(mIoU, Dice) and an optional test loss; a final summary records both
macro (unweighted) and micro (sample-weighted) aggregates over datasets.

Outputs are written to `${checkpoint.save_dir}/`:
    results/test_metrics.json   machine-readable per-dataset record
    results/test_metrics.csv    paper-friendly flat table
    test.jsonl                  per-dataset JSON-lines log

Why this recipe expects a `.pt` file (not DCP):
    SAM2 and MedSAM2 are released as single `.pt` files. Sonobase training
    saves DCP-sharded directories (one shard per DDP rank). To compare all
    three with the SAME load path, we run the sonobase DCP checkpoint
    through `convert_sonobase_dcp_to_pt.py` first, which produces a
    `.pt` file in the same layout as Meta's SAM2 release. See the README
    in this directory for the full workflow.
"""

import csv
import gc
import json
import logging
import os
import pathlib
import time
from typing import Dict, Optional

import torch
import torch.nn as nn
from hydra.utils import instantiate as hydra_instantiate
from omegaconf import DictConfig, OmegaConf

from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.distributed.init_utils import DistInfo
from nemo_automodel.components.loggers.log_utils import setup_logging
from nemo_automodel.components.loggers.metric_logger import MetricsSample, build_metric_logger
from nemo_automodel.components.training.rng import ScopedRNG
from nemo_automodel.recipes.base_recipe import BaseRecipe

from nemo_cv.components.training.utils import (
    CORE_LOSS_KEY,
    barrier,
    build_autocast_context,
    unwrap_ddp_if_wrapped,
)
from nemo_cv.recipes.sonobase.pretrain import (
    build_distributed,
    build_loss_fn,
    build_model,
    setup_ddp,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _resolve_checkpoint_file(ckpt_path: str) -> str:
    """Validate and resolve a single-file `.pt` checkpoint path.

    Accepts the SAM2 / MedSAM2 release format (a `.pt` file containing
    `{"model": state_dict}`), and the equivalent format produced by
    `convert_sonobase_dcp_to_pt.py` for sonobase pretraining checkpoints.
    """
    if ckpt_path is None or ckpt_path == "":
        raise ValueError(
            "`ckpt_path` is required for the benchmark test recipe. "
            "Pass it via CLI (set CHECKPOINT_DIR in your env first — e.g. `export CHECKPOINT_DIR=$WORK/Checkpoints`):\n"
            "  ckpt_path=${CHECKPOINT_DIR}/SAM2/sam2.1_hiera_base_plus.pt\n"
            "  ckpt_path=${CHECKPOINT_DIR}/MedSAM2/MedSAM2_latest.pt\n"
            "  ckpt_path=./experiments/.../LATEST.pt   # converted sonobase checkpoint"
        )
    ckpt_path = str(pathlib.Path(ckpt_path).expanduser().resolve())
    if not os.path.isfile(ckpt_path):
        if os.path.isdir(ckpt_path):
            raise FileNotFoundError(
                f"`ckpt_path` is a directory, not a file: {ckpt_path}\n"
                "This recipe expects a single `.pt` file. For a DCP-format "
                "sonobase pretrain checkpoint, run "
                "`convert_sonobase_dcp_to_pt.py` first to produce a `.pt`."
            )
        raise FileNotFoundError(f"Checkpoint file does not exist: {ckpt_path}")
    return ckpt_path


def _select_test_data(hcfg_data: DictConfig) -> DictConfig:
    """Select `cfg.data.test` if present, else `cfg.data.val`.

    Some data configs in this tree use `test:` (semantically correct for
    held-out evaluation), others use `val:` (carried over from the pretrain
    config tree). This recipe accepts either, preferring `test`.
    """
    test_cfg = OmegaConf.select(hcfg_data, "test", default=None)
    if test_cfg is not None:
        return test_cfg
    val_cfg = OmegaConf.select(hcfg_data, "val", default=None)
    if val_cfg is not None:
        return val_cfg
    raise ValueError(
        "data config has neither `test:` nor `val:` block. The benchmark "
        "recipe expects one or more datasets keyed by name under one of "
        "those top-level blocks."
    )


def _build_autocast(optim_cfg: Optional[DictConfig]):
    """Build an autocast context, falling back to nullcontext when optim is absent."""
    if optim_cfg is None or "amp" not in optim_cfg:
        from contextlib import nullcontext
        return nullcontext()
    return build_autocast_context(optim_cfg)


# ---------------------------------------------------------------------------
#  Recipe
# ---------------------------------------------------------------------------


class TestSam2BenchmarkRecipe(BaseRecipe):
    """Test/eval recipe for the SAM2 / MedSAM2 / Sonobase benchmark.

    Differs from the sonobase test recipe in two ways:

    1. **Checkpoint format**: loads a single-file `.pt` (`{"model": state_dict}`)
       via `model.load_state_dict(sd, strict=False)`, instead of DCP shards.
       This makes it compatible with Meta's SAM2 release format directly,
       and with sonobase via the DCP→.pt converter.

    2. **Data key**: tolerant of either `cfg.data.test` or `cfg.data.val`,
       since the test_sam2 config tree mixes both conventions.

    Notes on `BaseRecipe.__setattr__`:
    Attributes whose names contain `"test"`, `"val"`, `"eval"`, or `"loss"`
    are skipped from `__state_tracked`. We use those substrings everywhere
    in this recipe, so nothing accidentally triggers checkpoint writes.
    """

    def __init__(self, cfg: ConfigNode, hydra_cfg: DictConfig):
        self.cfg = cfg
        self.hydra_cfg = hydra_cfg

    # ------------------ build phase ------------------
    def setup(self):
        hcfg = self.hydra_cfg

        torch.cuda.reset_peak_memory_stats()
        self.dist_env: DistInfo = build_distributed(hcfg.get("dist_env", {}))
        setup_logging()

        self.seed = int(hcfg.get("seed", 42))
        self.device = self.dist_env.device

        self._log_experiment_details()
        self._log_library_versions()

        cuda_cfg = hcfg.get("cuda", {})
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = cuda_cfg.get("cudnn_deterministic", True)
            torch.backends.cudnn.benchmark = cuda_cfg.get("cudnn_benchmark", False)
            allow_tf32 = cuda_cfg.get("allow_tf32", False)
            torch.backends.cuda.matmul.allow_tf32 = cuda_cfg.get("matmul_allow_tf32", allow_tf32)
            torch.backends.cudnn.allow_tf32 = cuda_cfg.get("cudnn_allow_tf32", allow_tf32)

        # Loss is optional during test. If provided, we additionally report
        # per-dataset test_loss for parity with validation.jsonl.
        loss_cfg = hcfg.get("loss_fn", None)
        self.test_loss_fn: Optional[nn.ModuleDict] = build_loss_fn(loss_cfg) if loss_cfg is not None else None
        if self.test_loss_fn is not None:
            self.test_loss_fn.to(self.device)

        # Model: build with NO pretrained_ckpt_path (we load the benchmark
        # checkpoint below). The `_target_` and sub-component configs come
        # from cfg.model — sam2-no-ft, medsam2, sonobase share the same
        # top-level SAM2Train target and differ only in image_encoder.
        model = build_model(hcfg.model, self.device, seed=self.seed, pretrained_ckpt_path=None)

        # DDP wrap; eval-only so find_unused_parameters defaults to False.
        dist_cfg = hcfg.get("distributed", {})
        from omegaconf import OmegaConf as _OC
        dist_cfg_local = _OC.create(_OC.to_container(dist_cfg, resolve=True) if dist_cfg else {})
        if "find_unused_parameters" not in dist_cfg_local:
            dist_cfg_local.find_unused_parameters = False
        self.model = setup_ddp(model, self.dist_env, dist_cfg_local)

        # Load the .pt checkpoint with strict=False. SAM2 / MedSAM2 / converted
        # sonobase all share the SAM2 key prefixes (image_encoder.*,
        # memory_attention.*, memory_encoder.*, sam_mask_decoder.*,
        # prompt_encoder.*). image_encoder.trunk.* sub-keys differ between
        # variants (Hiera-B+, Hiera-T, TriBranchTrunk) but each variant's
        # checkpoint matches its model config, so within a single experiment
        # there should be no missing trunk keys.
        self.checkpoint_path = _resolve_checkpoint_file(hcfg.get("ckpt_path", None))
        if self.dist_env.is_main:
            logger.info(f"Loading model weights from: {self.checkpoint_path}")
        sd = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        if "model" in sd:
            sd = sd["model"]
        unwrapped = unwrap_ddp_if_wrapped(self.model)
        missing, unexpected = unwrapped.load_state_dict(sd, strict=False)
        if self.dist_env.is_main:
            logger.info(
                f"Loaded weights — missing keys: {len(missing)}, "
                f"unexpected keys: {len(unexpected)}"
            )
            if len(missing) > 0:
                logger.info(f"  first 5 missing: {missing[:5]}")
            if len(unexpected) > 0:
                logger.info(f"  first 5 unexpected: {unexpected[:5]}")
        del sd
        barrier()

        # Build the test datasets. Each value under data.test (or data.val)
        # is a separate TorchTrainMixedDataset → one DataLoader per named
        # dataset → per-dataset metrics.
        test_data_cfg = _select_test_data(hcfg.data)
        self.test_datasets: Dict[str, object] = {}
        for ds_name, ds_cfg in test_data_cfg.items():
            self.test_datasets[ds_name] = hydra_instantiate(ds_cfg, _recursive_=True, _convert_="all")
        if not self.test_datasets:
            raise ValueError("No datasets found under hcfg.data.test (or .val).")
        if self.dist_env.is_main:
            logger.info(f"Loaded {len(self.test_datasets)} test dataset(s): {list(self.test_datasets)}")

        # Per-dataset metric instances (torchmetrics objects auto-sync across DDP)
        val_metrics_cfg = hcfg.get("val_metrics", None)
        if val_metrics_cfg is None:
            raise ValueError("`val_metrics` config is required (e.g. miou, dice).")
        self.test_metrics: Dict[str, Dict[str, object]] = {}
        for ds_name in self.test_datasets:
            self.test_metrics[ds_name] = {
                name: hydra_instantiate(val_metrics_cfg[name]).to(self.device)
                for name in val_metrics_cfg
            }

        # Output directory: ./experiments/<scratch.experiment_name>/
        save_dir = hcfg.get("checkpoint", {}).get("save_dir", "./test_outputs")
        self.output_dir = pathlib.Path(save_dir)
        self.results_dir = self.output_dir / "results"
        if self.dist_env.is_main:
            self.results_dir.mkdir(parents=True, exist_ok=True)
            self.metric_logger_test = build_metric_logger(str(self.output_dir / "test.jsonl"))
            # Match the buffer-size-1 fix applied in pretrain.py — small
            # number of records per run, and we want them on disk eagerly.
            self.metric_logger_test.buffer_size = 1
        else:
            self.metric_logger_test = None

        self.log_freq = int(hcfg.get("logging", {}).get("log_freq", 10))

        barrier()
        if self.dist_env.is_main:
            logger.info(
                f"TestSam2BenchmarkRecipe setup complete. "
                f"Output dir: {self.output_dir}"
            )

    # ------------------ test loop ------------------
    @torch.no_grad()
    def run_test_loop(self):
        """Run inference on each test dataset and write aggregate results."""
        self.model.eval()
        all_results: Dict[str, Dict[str, float]] = {}
        autocast_factory = lambda: _build_autocast(self.hydra_cfg.get("optim", None))

        for ds_name, dataset in self.test_datasets.items():
            barrier()
            if self.dist_env.is_main:
                logger.info(f"--- Testing on {ds_name} ---")

            with ScopedRNG(seed=1, ranked=True):
                loader = dataset.get_loader(epoch=0)

                ds_loss_sum_local = 0.0
                ds_n_samples_local = 0
                ds_n_iters_local = 0
                t_start = time.perf_counter()

                for data_iter, batch in enumerate(loader):
                    batch = batch.to(self.device, non_blocking=True)
                    with autocast_factory():
                        outputs = self.model(batch)
                        targets = batch.masks
                        batch_size = batch.num_videos

                        if self.test_loss_fn is not None:
                            key = batch.dict_key
                            loss = self.test_loss_fn[key](outputs, targets)
                            if isinstance(loss, dict):
                                loss = loss[CORE_LOSS_KEY]
                            ds_loss_sum_local += float(loss.item()) * batch_size

                    for m in self.test_metrics[ds_name].values():
                        m.update(outputs, batch)

                    ds_n_samples_local += batch_size
                    ds_n_iters_local += 1

                    if self.dist_env.is_main and (data_iter % self.log_freq == 0):
                        elapsed = time.perf_counter() - t_start
                        logger.info(
                            f"[{ds_name}] iter {data_iter} | "
                            f"local_samples {ds_n_samples_local} | elapsed {elapsed:.1f}s"
                        )
                    if data_iter % 10 == 0:
                        barrier()

                del loader
                gc.collect()

            # Aggregate scalars across DDP ranks
            loss_t = torch.tensor([ds_loss_sum_local], dtype=torch.float64, device=self.device)
            n_t = torch.tensor([ds_n_samples_local], dtype=torch.long, device=self.device)
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(loss_t)
                torch.distributed.all_reduce(n_t)

            n_samples = int(n_t.item())
            test_loss: Optional[float] = (
                loss_t.item() / max(n_samples, 1) if self.test_loss_fn is not None else None
            )

            # Compute per-dataset metrics (torchmetrics auto-syncs internally)
            ds_metric_values: Dict[str, float] = {}
            for name, m in self.test_metrics[ds_name].items():
                ds_metric_values[name] = float(m.compute().item())
                m.reset()

            elapsed = time.perf_counter() - t_start
            ds_record: Dict[str, float] = {
                "n_samples": n_samples,
                "n_iters_per_rank": ds_n_iters_local,
                "elapsed_sec": round(elapsed, 2),
                **ds_metric_values,
            }
            if test_loss is not None:
                ds_record["test_loss"] = round(test_loss, 6)
            all_results[ds_name] = ds_record

            if self.dist_env.is_main:
                metric_str = " | ".join(f"{k}={v:.4f}" for k, v in ds_metric_values.items())
                loss_str = f" | test_loss={test_loss:.4f}" if test_loss is not None else ""
                logger.info(
                    f"[{ds_name}] DONE | n_samples={n_samples} | {metric_str}{loss_str} | "
                    f"elapsed={elapsed:.1f}s"
                )
                self.metric_logger_test.log(
                    MetricsSample(
                        step=0,
                        epoch=0,
                        metrics={"dataset": ds_name, **ds_record},
                    )
                )

        if self.dist_env.is_main:
            self._write_results(all_results)
            self.metric_logger_test.close()

        barrier()

    # ------------------ output writers ------------------
    def _write_results(self, all_results: Dict[str, Dict[str, float]]):
        """Write JSON + CSV + a pretty table to the log."""
        ordered_dsets = list(all_results.keys())
        bookkeeping = {"n_samples", "n_iters_per_rank", "elapsed_sec"}
        first = next(iter(all_results.values()))
        metric_names = [k for k in first.keys() if k not in bookkeeping]

        total_samples = sum(r["n_samples"] for r in all_results.values())
        macro: Dict[str, float] = {}
        micro: Dict[str, float] = {}
        for name in metric_names:
            valid = [(r[name], r["n_samples"]) for r in all_results.values() if r.get(name) is not None]
            if not valid:
                continue
            macro[name] = sum(v for v, _ in valid) / len(valid)
            denom = sum(n for _, n in valid)
            micro[name] = sum(v * n for v, n in valid) / max(denom, 1)

        out = {
            "checkpoint": self.checkpoint_path,
            "n_datasets": len(all_results),
            "datasets": all_results,
            "aggregate_macro": macro,
            "aggregate_micro": micro,
        }

        json_path = self.results_dir / "test_metrics.json"
        with json_path.open("w") as f:
            json.dump(out, f, indent=2)
        logger.info(f"Wrote {json_path}")

        csv_path = self.results_dir / "test_metrics.csv"
        with csv_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["dataset", "n_samples"] + metric_names)
            for ds_name in ordered_dsets:
                r = all_results[ds_name]
                w.writerow([ds_name, r["n_samples"]] + [r.get(n, "") for n in metric_names])
            w.writerow([])
            w.writerow(["AGGREGATE_MACRO", total_samples] + [round(macro.get(n, float("nan")), 6) for n in metric_names])
            w.writerow(["AGGREGATE_MICRO", total_samples] + [round(micro.get(n, float("nan")), 6) for n in metric_names])
        logger.info(f"Wrote {csv_path}")

        col_w = max(24, max(len(n) for n in ordered_dsets) + 2) if ordered_dsets else 24
        header = f"{'dataset':<{col_w}} {'n_samples':>10} " + " ".join(f"{n:>10}" for n in metric_names)
        sep = "-" * len(header)
        logger.info("=" * len(header))
        logger.info(f"BENCHMARK RESULTS  (checkpoint: {self.checkpoint_path})")
        logger.info("=" * len(header))
        logger.info(header)
        logger.info(sep)
        for ds_name in ordered_dsets:
            r = all_results[ds_name]
            row = f"{ds_name:<{col_w}} {r['n_samples']:>10} " + " ".join(
                f"{r.get(n, float('nan')):>10.4f}" for n in metric_names
            )
            logger.info(row)
        logger.info(sep)
        macro_row = f"{'AGGREGATE_MACRO':<{col_w}} {total_samples:>10} " + " ".join(
            f"{macro.get(n, float('nan')):>10.4f}" for n in metric_names
        )
        micro_row = f"{'AGGREGATE_MICRO':<{col_w}} {total_samples:>10} " + " ".join(
            f"{micro.get(n, float('nan')):>10.4f}" for n in metric_names
        )
        logger.info(macro_row)
        logger.info(micro_row)
        logger.info("=" * len(header))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    """Entry point for the benchmark test recipe.

    CLI:
        uv run torchrun --nproc-per-node=N -m nemo_cv.recipes.benchmarks.test_sam2 \
            -c <config_dir> -cn test \
            experiment=<exp_name> \
            ckpt_path=<path_to_pt_file>
    """
    import sys

    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    config_dir = None
    config_name = "test"
    overrides = []
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

    recipe = TestSam2BenchmarkRecipe(cfg_node, hydra_cfg)
    recipe.setup()
    recipe.run_test_loop()


if __name__ == "__main__":
    main()
