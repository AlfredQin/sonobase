"""Analysis A3 — Click-efficiency (interactive correction convergence).

Stage-2 analysis. Consumes the **long-format** ``test_metrics_iterations.csv``
artifacts produced by the benchmark iterative-correction sweep — one CSV per
``(model, prompt)`` combination — and renders convergence curves +
clicks-to-target summaries + a JSON report.

Background
----------
Earlier drafts of this analysis ran a per-iteration prediction sweep via
``save_predictions.py`` against HC18 only and computed mean-IoU curves
from the per-sample CSV. That design was abandoned in favor of consuming
the benchmark sweep directly: the benchmark already evaluates each
``(model × prompt × dataset)`` cell at ``num_correction_pt_per_frame_val
∈ {0, 1, 3, 5, 7}``, so duplicating that work in A3 was wasteful and the
two pipelines could diverge over time. A3 is now a pure CSV-consumer
that reads the benchmark output and produces the click-efficiency
figures and tables.

Both the production B1 path
(``aggregate_iterations.py`` over multiple ``test_metrics.json``) and
the standalone B2 path (``iterative_eval.py`` in a single process)
emit the exact same long-format CSV schema, so this analysis is
agnostic to which pipeline produced its inputs:

::

    dataset,iteration,n_samples,miou,dice[,test_loss]
    BUSI,0,131,0.4237,0.5236,1.0597
    BUSI,1,131,0.6512,0.7341,...
    ...
    AGGREGATE_MACRO,0,...
    AGGREGATE_MICRO,0,...

Outputs
-------
For each ``(prompt, tier)`` facet — where ``tier`` is one of
``benchmark``, ``external``, or freeform via ``--dataset-tiers``:

  * ``convergence_<prompt>_<tier>.{pdf,png}`` — single-panel curve
    (one line per model). Y-axis is mean mIoU%.
  * ``convergence_grid.{pdf,png}`` — 2x2 (or larger) grid of all facets.

Per-dataset detail (one panel per dataset):

  * ``per_dataset/<dataset>__<prompt>.{pdf,png}``

Summary table (PDF + JSON):

  * ``summary_table.{pdf,png}`` — per-(model, prompt, tier) row showing
    mIoU at iter 0, asymptote (max iter), Δ, clicks-to-target,
    seconds-to-target.
  * ``analysis_report.json`` — machine-readable record of all curves +
    summaries + asymptote estimates.

Time conversion
---------------
Clinician time per prompt:

    1 correction click  = 2 s of clinician time
    1 box prompt        = 4 s of clinician time
    Initial point click = 2 s

CLI
---
::

    uv run python -m nemo_cv.recipes.analysis.a3_click_efficiency \\
        --inputs \\
            sam2_no_ft:point=./experiments/benchmarks/8_bm_7_ext/sam2/b1_point/test_metrics_iterations.csv \\
            sam2_no_ft:box  =./experiments/benchmarks/8_bm_7_ext/sam2/b1_box/test_metrics_iterations.csv \\
            medsam2:point   =./experiments/benchmarks/8_bm_7_ext/medsam2/b1_point/test_metrics_iterations.csv \\
            medsam2:box     =./experiments/benchmarks/8_bm_7_ext/medsam2/b1_box/test_metrics_iterations.csv \\
            sonobase:point  =./experiments/benchmarks/8_bm_7_ext/hiera_b_conv_s_conv_t/b1_point/test_metrics_iterations.csv \\
            sonobase:box    =./experiments/benchmarks/8_bm_7_ext/hiera_b_conv_s_conv_t/b1_box/test_metrics_iterations.csv \\
        --output-dir ./experiments/analysis/a3_click_efficiency
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import pathlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from nemo_cv.components.analysis.plots.convergence_curves import (
    convergence_curves,
    convergence_curves_grid,
)
from nemo_cv.components.analysis.plots.style import display_name
from nemo_cv.components.analysis.plots.summary_table import summary_table_figure

logger = logging.getLogger(__name__)


SECONDS_PER_POINT = 2.0
SECONDS_PER_BOX = 4.0
TARGET_MIOU_PCT = 80.0

# Default dataset-tier labelling — matches the 8_bm_7_ext data config.
# Override via `--dataset-tiers` to extend or relabel.
DEFAULT_BENCHMARK_DATASETS = {
    "BUSI", "Brachial-Plexus", "C-TRUS", "CAMUS",
    "HC18", "PFUS", "RegPro", "TG3K",
}
DEFAULT_EXTERNAL_DATASETS = {
    "ACOUSLIC", "BUS-BRA", "DDTI", "FUGC",
    "KidneyUS", "LUMINOUS", "MMOTU-3d",
}


# ---------------------------------------------------------------------------
#  Data containers
# ---------------------------------------------------------------------------


@dataclass
class _CsvInput:
    """One ``(model, prompt) -> long-format CSV`` mapping."""
    model_label: str
    prompt: str                    # "point" | "box"
    csv_path: pathlib.Path


@dataclass
class _DatasetCurve:
    """Per-iteration metric values for one (model, prompt, dataset)."""
    iterations: List[int] = field(default_factory=list)
    miou_pct: Dict[int, float] = field(default_factory=dict)
    dice_pct: Dict[int, float] = field(default_factory=dict)
    n_samples: Dict[int, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
#  Parsing
# ---------------------------------------------------------------------------


def _parse_inputs(items: Sequence[str]) -> List[_CsvInput]:
    """Parse CLI ``--inputs`` entries: ``<model>:<prompt>=<csv_path>``."""
    out: List[_CsvInput] = []
    for s in items:
        if "=" not in s:
            raise ValueError(
                f"--inputs entry must be `<model>:<prompt>=<csv_path>`, got: {s!r}"
            )
        spec, path = s.split("=", 1)
        parts = spec.split(":")
        if len(parts) != 2:
            raise ValueError(
                f"--inputs spec must have 2 colon-separated parts (model:prompt), "
                f"got: {spec!r}"
            )
        model, prompt = parts[0].strip(), parts[1].strip()
        if prompt not in ("point", "box"):
            raise ValueError(f"prompt must be 'point' or 'box', got: {prompt!r}")
        p = pathlib.Path(path.strip()).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"CSV not found: {p}")
        out.append(_CsvInput(model, prompt, p))
    return out


def _parse_dataset_tiers(arg: Optional[str]) -> Dict[str, str]:
    """Parse a ``DATASET=TIER,DATASET=TIER,...`` mapping.

    Falls back to :data:`DEFAULT_BENCHMARK_DATASETS` /
    :data:`DEFAULT_EXTERNAL_DATASETS` when ``arg`` is None. When ``arg``
    is given it MERGES on top of the defaults so callers only need to
    label new datasets.
    """
    out: Dict[str, str] = {}
    for ds in DEFAULT_BENCHMARK_DATASETS:
        out[ds] = "benchmark"
    for ds in DEFAULT_EXTERNAL_DATASETS:
        out[ds] = "external"
    if arg:
        for token in arg.split(","):
            token = token.strip()
            if not token:
                continue
            if "=" not in token:
                raise ValueError(
                    f"--dataset-tiers entry must be `DATASET=TIER`, got: {token!r}"
                )
            ds, tier = token.split("=", 1)
            out[ds.strip()] = tier.strip()
    return out


def _load_csv(csv_path: pathlib.Path) -> Dict[str, _DatasetCurve]:
    """Load a long-format ``test_metrics_iterations.csv`` into per-dataset curves."""
    by_ds: Dict[str, _DatasetCurve] = defaultdict(_DatasetCurve)
    with csv_path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            ds = (row.get("dataset") or "").strip()
            if not ds or ds.startswith("AGGREGATE"):
                continue
            try:
                it = int(row["iteration"])
            except (KeyError, ValueError):
                continue
            curve = by_ds[ds]
            curve.iterations.append(it)
            try:
                curve.miou_pct[it] = float(row["miou"]) * 100.0
            except (KeyError, ValueError, TypeError):
                pass
            try:
                curve.dice_pct[it] = float(row["dice"]) * 100.0
            except (KeyError, ValueError, TypeError):
                pass
            try:
                curve.n_samples[it] = int(row["n_samples"])
            except (KeyError, ValueError, TypeError):
                pass
    for ds, c in by_ds.items():
        c.iterations = sorted(set(c.iterations))
    return dict(by_ds)


# ---------------------------------------------------------------------------
#  Aggregation helpers
# ---------------------------------------------------------------------------


def _macro_curve(
    per_ds: Dict[str, _DatasetCurve],
    datasets: Sequence[str],
    metric: str = "miou_pct",
) -> Dict[int, float]:
    """Unweighted mean across the given datasets, per iteration."""
    if not datasets:
        return {}
    iter_set = sorted({
        it for ds in datasets if ds in per_ds for it in per_ds[ds].iterations
    })
    out: Dict[int, float] = {}
    for it in iter_set:
        vals = [
            getattr(per_ds[ds], metric).get(it)
            for ds in datasets
            if ds in per_ds and getattr(per_ds[ds], metric).get(it) is not None
        ]
        if vals:
            out[it] = float(np.mean(vals))
    return out


def _micro_curve(
    per_ds: Dict[str, _DatasetCurve],
    datasets: Sequence[str],
    metric: str = "miou_pct",
) -> Dict[int, float]:
    """Sample-weighted mean across the given datasets, per iteration."""
    if not datasets:
        return {}
    iter_set = sorted({
        it for ds in datasets if ds in per_ds for it in per_ds[ds].iterations
    })
    out: Dict[int, float] = {}
    for it in iter_set:
        num = 0.0
        den = 0
        for ds in datasets:
            if ds not in per_ds:
                continue
            v = getattr(per_ds[ds], metric).get(it)
            n = per_ds[ds].n_samples.get(it, 0)
            if v is None or n <= 0:
                continue
            num += v * n
            den += n
        if den > 0:
            out[it] = num / den
    return out


def _clicks_to_target(curve: Dict[int, float], target_pct: float) -> Optional[int]:
    """Smallest iteration count where ``curve[it] >= target_pct``."""
    for it in sorted(curve.keys()):
        v = curve[it]
        if v is not None and not math.isnan(v) and v >= target_pct:
            return it
    return None


def _seconds_to_target(prompt: str, n_clicks: Optional[int]) -> Optional[float]:
    """Convert iteration count to total clinician seconds."""
    if n_clicks is None:
        return None
    init_s = SECONDS_PER_BOX if prompt == "box" else SECONDS_PER_POINT
    return init_s + n_clicks * SECONDS_PER_POINT


def _asymptote(curve: Dict[int, float]) -> Optional[Tuple[int, float]]:
    """Return ``(iteration, mIoU%)`` of the curve maximum (highest iter wins ties)."""
    if not curve:
        return None
    best_iter = max(curve.keys(), key=lambda it: (curve[it], it))
    return best_iter, curve[best_iter]


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--inputs", nargs="+", required=True,
        help=(
            "List of `<model>:<prompt>=<csv_path>` entries. Each <csv_path> "
            "is a long-format `test_metrics_iterations.csv` (B1-merged or "
            "B2 standalone) for the (model, prompt) cell."
        ),
    )
    p.add_argument(
        "--output-dir", required=True, type=pathlib.Path,
        help="Output directory for figures + analysis_report.json.",
    )
    p.add_argument(
        "--target-miou-pct", type=float, default=TARGET_MIOU_PCT,
        help=f"Target mIoU percent for clicks-to-target. Default {TARGET_MIOU_PCT}.",
    )
    p.add_argument(
        "--dataset-tiers", type=str, default=None,
        help=(
            "Comma-separated `DATASET=TIER` mapping merged on top of the "
            "default benchmark/external split. Example: "
            "'KidneyUS=external,FUGC=external'."
        ),
    )
    p.add_argument(
        "--per-dataset", action="store_true",
        help="Also write a per-dataset curve panel under per_dataset/.",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    args = _build_argparser().parse_args(argv)

    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs = _parse_inputs(args.inputs)
    tiers = _parse_dataset_tiers(args.dataset_tiers)

    # data[model][prompt] = {dataset_name: _DatasetCurve}
    data: Dict[str, Dict[str, Dict[str, _DatasetCurve]]] = defaultdict(dict)
    for ci in inputs:
        per_ds = _load_csv(ci.csv_path)
        if not per_ds:
            logger.warning(
                f"No per-dataset rows in {ci.csv_path} — skipping "
                f"({ci.model_label}/{ci.prompt})."
            )
            continue
        data[ci.model_label][ci.prompt] = per_ds
        logger.info(
            f"[{ci.model_label}/{ci.prompt}] loaded {len(per_ds)} datasets, "
            f"iters={sorted({it for c in per_ds.values() for it in c.iterations})}"
        )

    if not data:
        logger.error("No data loaded — refusing to write empty report.")
        return 1

    # Group every dataset that appeared anywhere by tier
    all_datasets = sorted({
        ds for prompts in data.values()
        for per_ds in prompts.values()
        for ds in per_ds
    })
    tier_to_datasets: Dict[str, List[str]] = defaultdict(list)
    for ds in all_datasets:
        tier = tiers.get(ds, "all")
        tier_to_datasets[tier].append(ds)
    logger.info(
        f"Tier assignment: "
        + ", ".join(f"{t}={len(d)}" for t, d in sorted(tier_to_datasets.items()))
    )

    # ----- per-(prompt, tier) facet macro curves & summary records -----
    facet_curves: Dict[Tuple[str, str], Dict[str, Dict[int, float]]] = defaultdict(dict)
    summaries: Dict[str, Dict] = {}

    for model, prompts_map in data.items():
        for prompt, per_ds in prompts_map.items():
            for tier, ds_list in tier_to_datasets.items():
                ds_in_run = [ds for ds in ds_list if ds in per_ds]
                if not ds_in_run:
                    continue
                macro = _macro_curve(per_ds, ds_in_run)
                micro = _micro_curve(per_ds, ds_in_run)
                facet_curves[(prompt, tier)][model] = macro

                n_clicks_macro = _clicks_to_target(macro, args.target_miou_pct)
                seconds_macro = _seconds_to_target(prompt, n_clicks_macro)
                asym = _asymptote(macro)

                summaries[f"{model}__{prompt}__{tier}"] = {
                    "model": model,
                    "prompt": prompt,
                    "tier": tier,
                    "datasets_in_facet": ds_in_run,
                    "iterations": sorted(macro.keys()),
                    "miou_macro_pct": {str(k): macro[k] for k in sorted(macro)},
                    "miou_micro_pct": {str(k): micro[k] for k in sorted(micro)},
                    "asymptote_iter": asym[0] if asym else None,
                    "asymptote_miou_pct": asym[1] if asym else None,
                    "iter0_miou_pct": macro.get(0),
                    "delta_iter0_to_asym_pct": (
                        (asym[1] - macro[0]) if (asym and 0 in macro) else None
                    ),
                    f"clicks_to_{int(args.target_miou_pct)}": n_clicks_macro,
                    f"seconds_to_{int(args.target_miou_pct)}": seconds_macro,
                }

                logger.info(
                    f"[{model} | {prompt} | {tier}] "
                    f"macro: iter0={macro.get(0)}  asym={asym}  "
                    f"clicks_to_{int(args.target_miou_pct)}={n_clicks_macro}"
                )

    # ----- render convergence panels -----
    panels: List[Dict] = []
    for (prompt, tier), model_curves in sorted(facet_curves.items()):
        all_iters = sorted({it for c in model_curves.values() for it in c.keys()})
        panel = {
            "title": f"{prompt.capitalize()} init — {tier}",
            "iterations": all_iters,
            "model_curves": {
                lbl: [model_curves[lbl].get(it, float("nan")) for it in all_iters]
                for lbl in model_curves
            },
        }
        panels.append(panel)
        convergence_curves(
            iterations=all_iters,
            model_curves=panel["model_curves"],
            output_path=out_dir / f"convergence_{prompt}_{tier}.pdf",
            title=panel["title"],
        )

    if panels:
        n_cols = min(2, max(1, len(panels)))
        convergence_curves_grid(
            panels, output_path=out_dir / "convergence_grid.pdf", n_cols=n_cols
        )

    # ----- per-dataset detail panels -----
    if args.per_dataset:
        per_ds_dir = out_dir / "per_dataset"
        per_ds_dir.mkdir(exist_ok=True)
        for ds in all_datasets:
            for prompt in ("point", "box"):
                model_curves: Dict[str, Dict[int, float]] = {}
                for model, prompts_map in data.items():
                    per_ds_map = prompts_map.get(prompt, {})
                    if ds not in per_ds_map:
                        continue
                    model_curves[model] = per_ds_map[ds].miou_pct
                if not model_curves:
                    continue
                all_iters = sorted({it for c in model_curves.values() for it in c.keys()})
                convergence_curves(
                    iterations=all_iters,
                    model_curves={
                        lbl: [model_curves[lbl].get(it, float("nan")) for it in all_iters]
                        for lbl in model_curves
                    },
                    output_path=per_ds_dir / f"{ds}__{prompt}.pdf",
                    title=f"{ds} — {prompt} init",
                )

    # ----- summary table -----
    target_int = int(args.target_miou_pct)
    table_rows: List[List[str]] = [[
        "Model", "Prompt", "Tier", "iter 0 mIoU%",
        "Asymptote mIoU%", "Δ (asym - 0)",
        f"Clicks → {target_int}%", "Time (s)",
    ]]
    for key in sorted(summaries):
        s = summaries[key]
        nc = s[f"clicks_to_{target_int}"]
        sc = s[f"seconds_to_{target_int}"]
        iter0 = s["iter0_miou_pct"]
        asym = s["asymptote_miou_pct"]
        delta = s["delta_iter0_to_asym_pct"]
        table_rows.append([
            display_name(s["model"]),
            s["prompt"],
            s["tier"],
            "—" if iter0 is None else f"{iter0:.2f}",
            "—" if asym is None else f"{asym:.2f}",
            "—" if delta is None else f"{delta:+.2f}",
            "—" if nc is None else str(nc),
            "—" if sc is None else f"{sc:.1f}",
        ])
    summary_table_figure(
        table_rows, out_dir / "summary_table.pdf",
        title=f"Click efficiency — A3 summary (target {target_int}% mIoU)",
    )

    # ----- JSON report -----
    with (out_dir / "analysis_report.json").open("w") as f:
        json.dump({
            "analysis": "A3",
            "title": "Click efficiency / iterative-correction convergence",
            "spec_section": "A3",
            "target_miou_pct": args.target_miou_pct,
            "seconds_per_point": SECONDS_PER_POINT,
            "seconds_per_box": SECONDS_PER_BOX,
            "tier_assignment": {ds: tiers.get(ds, "all") for ds in all_datasets},
            "facet_summaries": summaries,
            "n_models": len(data),
            "n_datasets": len(all_datasets),
        }, f, indent=2)

    logger.info(f"A3 analysis complete. Outputs at: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
