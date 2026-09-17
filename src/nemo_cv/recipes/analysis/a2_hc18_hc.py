"""Analysis A2 — HC18 head circumference measurement error.

Stage-2 analysis. Consumes Stage-1 prediction artifacts (one or more
`predictions/<run_name>/` directories produced by `save_predictions.py`)
and produces the HC18 head-circumference comparison: per-sample CSV,
analysis report JSON, scatter + Bland-Altman plots, and summary table.

CPU only (no model, no GPU). Uses the shared utilities under
`nemo_cv.components.analysis.{measurements,stats,plots,metadata_loaders}`.

CLI:
    cd src
    uv run python -m nemo_cv.recipes.analysis.a2_hc18_hc \\
        --runs sam2_no_ft=./experiments/analysis/predictions/sam2_no_ft_HC18_point_0corr \\
               medsam2=./experiments/analysis/predictions/medsam2_HC18_point_0corr \\
               sonobase=./experiments/analysis/predictions/sonobase_HC18_point_0corr \\
        --output-dir ./experiments/analysis/a2_hc18_hc/HC18_point_0corr \\
        --hc18-zip "$HC18_ZIP" \\        # or set HC18_ZIP env var (e.g. $WORK/Dataset/UltraSound/Raw/HC.zip)
        [--include-gt-fit]   # also report the GT-mask ellipse-fit baseline

The output directory layout is:

    <output_dir>/
    ├── analysis_report.json         full metrics dict
    ├── per_sample.csv               augmented per-sample CSV (adds pred_clinical_value, abs_error_mm, rel_error_pct)
    ├── summary_table.{pdf,png}      summary metrics rendered as a figure
    ├── scatter_<model>.{pdf,png}    per-model predicted vs GT HC scatter
    ├── scatter_comparison.{pdf,png} side-by-side scatter for all models
    └── bland_altman_<model>.{pdf,png}
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import pathlib
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from nemo_cv.components.analysis import prediction_io as pio
from nemo_cv.components.analysis import record_io
from nemo_cv.components.analysis.measurements.ellipse_fit import head_circumference_mm
from nemo_cv.components.analysis.metadata_loaders.default import ClinicalMeta
from nemo_cv.components.analysis.metadata_loaders.hc18 import load_hc18_metadata
from nemo_cv.components.analysis.plots.bland_altman import bland_altman_panel
from nemo_cv.components.analysis.plots.scatter import scatter_with_identity
from nemo_cv.components.analysis.plots.style import (
    color_for,
    display_name,
    apply_nm_defaults,
)
from nemo_cv.components.analysis.plots.summary_table import summary_table_figure
from nemo_cv.components.analysis.stats.tests import (
    bland_altman_stats,
    pearson_r,
    percent_within,
)

logger = logging.getLogger(__name__)

# Clinical thresholds
INTEROBSERVER_THRESHOLD_MM = 3.0   # within human-expert agreement
CLINICAL_TOLERANCE_MM = 5.0        # within clinical decision tolerance


# ---------------------------------------------------------------------------
# Per-model loading + measurement
# ---------------------------------------------------------------------------


@dataclass
class ModelHCResult:
    """One model's HC measurement results across the HC18 test set."""

    model_label: str
    sample_ids: List[str]
    gt_hc_mm: np.ndarray          # shape (N,)
    pred_hc_mm: np.ndarray        # shape (N,) — np.nan where ellipse fit failed
    iou: np.ndarray               # shape (N,) — segmentation IoU per sample
    dice: np.ndarray              # shape (N,)
    n_failed_ellipse: int


def _load_model_run(
    run_dir: pathlib.Path,
    model_label: str,
    metadata: Dict[str, ClinicalMeta],
) -> ModelHCResult:
    """Read Stage-1 outputs for one (model, prompt) run; compute predicted HC mm.

    For each row in ``per_sample_metrics.csv``:
      1. Look up the per-image pixel spacing + GT HC from the metadata table.
      2. Load the predicted mask PNG.
      3. Fit an ellipse and compute predicted HC via Ramanujan's formula.
      4. Record (gt_mm, pred_mm) for the per-sample CSV and aggregations.

    Samples missing from the HC18 metadata (rare; usually IDs that are in
    the SaUS test split but not in the original challenge CSV) are skipped
    with a warning.
    """
    rows = pio.read_per_sample_csv(run_dir)
    sample_ids: List[str] = []
    gt_arr: List[float] = []
    pred_arr: List[float] = []
    iou_arr: List[float] = []
    dice_arr: List[float] = []
    n_failed = 0

    for row in rows:
        sid = row["sample_id"]
        meta = metadata.get(sid)
        if meta is None:
            logger.warning(
                f"[{model_label}] sample {sid!r} missing from HC18 metadata CSV; skipping"
            )
            continue

        # Load predicted mask (relative path is stored relative to run_dir)
        pred_mask_path = run_dir / row["pred_mask_path"]
        if not pred_mask_path.is_file():
            logger.warning(f"[{model_label}] pred mask missing: {pred_mask_path}")
            continue
        pred_mask = np.array(Image.open(pred_mask_path).convert("L")) > 0

        pred_hc = head_circumference_mm(pred_mask, meta.pixel_spacing_mm)
        if pred_hc is None:
            n_failed += 1
            pred_hc = float("nan")

        sample_ids.append(sid)
        gt_arr.append(float(meta.gt_clinical_value))
        pred_arr.append(float(pred_hc))
        iou_arr.append(float(row["iou"]))
        dice_arr.append(float(row["dice"]))

    return ModelHCResult(
        model_label=model_label,
        sample_ids=sample_ids,
        gt_hc_mm=np.asarray(gt_arr, dtype=float),
        pred_hc_mm=np.asarray(pred_arr, dtype=float),
        iou=np.asarray(iou_arr, dtype=float),
        dice=np.asarray(dice_arr, dtype=float),
        n_failed_ellipse=n_failed,
    )


def _gt_ellipse_fit_baseline(
    run_dir: pathlib.Path, metadata: Dict[str, ClinicalMeta]
) -> ModelHCResult:
    """Compute the "GT ellipse fit" baseline — same pipeline as a model, but on GT masks.

    Establishes the systematic floor of the ellipse-fit + Ramanujan
    pipeline on the GT data itself (~1.4 mm MAE). Uses any model run's `gt_mask_path` columns as the source of
    GT mask paths (all model runs save the GT alongside the prediction).
    """
    rows = pio.read_per_sample_csv(run_dir)
    sample_ids, gt_arr, pred_arr = [], [], []
    n_failed = 0

    for row in rows:
        sid = row["sample_id"]
        meta = metadata.get(sid)
        if meta is None:
            continue
        gt_mask_path = run_dir / row["gt_mask_path"]
        if not gt_mask_path.is_file():
            continue
        gt_mask = np.array(Image.open(gt_mask_path).convert("L")) > 0
        pred_hc = head_circumference_mm(gt_mask, meta.pixel_spacing_mm)
        if pred_hc is None:
            n_failed += 1
            pred_hc = float("nan")
        sample_ids.append(sid)
        gt_arr.append(float(meta.gt_clinical_value))
        pred_arr.append(float(pred_hc))

    n = len(sample_ids)
    return ModelHCResult(
        model_label="GT (ellipse fit)",
        sample_ids=sample_ids,
        gt_hc_mm=np.asarray(gt_arr, dtype=float),
        pred_hc_mm=np.asarray(pred_arr, dtype=float),
        iou=np.full(n, 1.0),
        dice=np.full(n, 1.0),
        n_failed_ellipse=n_failed,
    )


# ---------------------------------------------------------------------------
# Aggregations
# ---------------------------------------------------------------------------


def _summarize(result: ModelHCResult) -> Dict[str, float]:
    """Compute the per-model summary metrics (MAE, %within, r, BA stats, IoU/Dice)."""
    gt = result.gt_hc_mm
    pred = result.pred_hc_mm
    valid = np.isfinite(gt) & np.isfinite(pred)
    gt_v, pred_v = gt[valid], pred[valid]

    if gt_v.size == 0:
        return {
            "n": 0,
            "n_failed_ellipse": int(result.n_failed_ellipse),
            "mae_mm": float("nan"),
            "std_mm": float("nan"),
            "rel_err_pct": float("nan"),
            "pct_within_3mm": float("nan"),
            "pct_within_5mm": float("nan"),
            "pearson_r": float("nan"),
            "bias_mm": float("nan"),
            "loa_lower_mm": float("nan"),
            "loa_upper_mm": float("nan"),
            "mean_iou": float("nan"),
            "mean_dice": float("nan"),
        }

    abs_err = np.abs(pred_v - gt_v)
    rel_err = abs_err / np.maximum(np.abs(gt_v), 1e-9) * 100.0

    ba = bland_altman_stats(gt_v, pred_v)

    # IoU/Dice across the valid subset (all samples, including ellipse-fit failures)
    iou_valid = result.iou[np.isfinite(result.iou)]
    dice_valid = result.dice[np.isfinite(result.dice)]

    return {
        "n": int(gt_v.size),
        "n_failed_ellipse": int(result.n_failed_ellipse),
        "mae_mm": float(abs_err.mean()),
        "std_mm": float(abs_err.std(ddof=1)) if gt_v.size > 1 else 0.0,
        "rel_err_pct": float(rel_err.mean()),
        "pct_within_3mm": percent_within(abs_err, INTEROBSERVER_THRESHOLD_MM),
        "pct_within_5mm": percent_within(abs_err, CLINICAL_TOLERANCE_MM),
        "pearson_r": pearson_r(gt_v, pred_v),
        "bias_mm": ba.bias,
        "loa_lower_mm": ba.loa_lower,
        "loa_upper_mm": ba.loa_upper,
        "mean_iou": float(iou_valid.mean()) if iou_valid.size else float("nan"),
        "mean_dice": float(dice_valid.mean()) if dice_valid.size else float("nan"),
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def _write_per_sample_csv(
    output_dir: pathlib.Path, results: List[ModelHCResult]
) -> None:
    """Write a wide per-sample CSV: one row per sample × N model columns."""
    # Build sample-id × model matrix
    sids = sorted({sid for r in results for sid in r.sample_ids})

    # Maps for fast lookup
    gt_by_sid: Dict[str, float] = {}
    pred_by_model_sid: Dict[Tuple[str, str], float] = {}
    for r in results:
        for i, sid in enumerate(r.sample_ids):
            if sid not in gt_by_sid and np.isfinite(r.gt_hc_mm[i]):
                gt_by_sid[sid] = float(r.gt_hc_mm[i])
            pred_by_model_sid[(r.model_label, sid)] = float(r.pred_hc_mm[i])

    headers = ["sample_id", "gt_hc_mm"]
    for r in results:
        headers.append(f"pred_hc_mm__{r.model_label}")
        headers.append(f"abs_err_mm__{r.model_label}")

    out_path = output_dir / "per_sample.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for sid in sids:
            gt = gt_by_sid.get(sid)
            row = [sid, "" if gt is None else f"{gt:.4f}"]
            for r in results:
                pred = pred_by_model_sid.get((r.model_label, sid))
                row.append("" if pred is None or not math.isfinite(pred) else f"{pred:.4f}")
                if pred is None or gt is None or not math.isfinite(pred):
                    row.append("")
                else:
                    row.append(f"{abs(pred - gt):.4f}")
            w.writerow(row)
    logger.info(f"Wrote {out_path}")


def _write_report(
    output_dir: pathlib.Path,
    summaries: Dict[str, Dict[str, float]],
    inputs: Dict[str, str],
) -> None:
    out = {
        "analysis": "A2",
        "title": "HC18 head circumference measurement error",
        "spec_section": "A2",
        "inputs": inputs,
        "thresholds": {
            "interobserver_mm": INTEROBSERVER_THRESHOLD_MM,
            "clinical_tolerance_mm": CLINICAL_TOLERANCE_MM,
        },
        "models": summaries,
        # mean_iou / mean_dice are intentionally unchecked: per_sample.csv
        # carries the measurement error columns but no per-row IoU/Dice, so a
        # checker will list them as uncovered rather than pass them by default.
        "verification": record_io.declare("per_sample.csv", [
            record_io.check(["models", "$model", "mae_mm"], "abs_err_mm__{model}"),
            record_io.check(["models", "$model", "std_mm"], "abs_err_mm__{model}", agg="std"),
            record_io.check(["models", "$model", "n"], "abs_err_mm__{model}", agg="count"),
        ]),
    }
    path = output_dir / "analysis_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"Wrote {path}")


def _render_summary_table(
    output_dir: pathlib.Path, summaries: Dict[str, Dict[str, float]]
) -> None:
    headers = ["Model", "n", "MAE (mm)", "Rel-Err (%)", "<3 mm (%)",
               "<5 mm (%)", "r", "Bias (mm)", "LoA (mm)"]
    rows = [headers]
    for label, s in summaries.items():
        loa = f"[{s['loa_lower_mm']:.2f}, {s['loa_upper_mm']:.2f}]"
        rows.append([
            display_name(label) if label not in ("GT (ellipse fit)",) else label,
            f"{s['n']:d}",
            f"{s['mae_mm']:.2f} ± {s['std_mm']:.2f}",
            f"{s['rel_err_pct']:.2f}",
            f"{s['pct_within_3mm']:.1f}",
            f"{s['pct_within_5mm']:.1f}",
            f"{s['pearson_r']:.3f}",
            f"{s['bias_mm']:+.2f}",
            loa,
        ])
    summary_table_figure(rows, output_dir / "summary_table.pdf",
                        title="HC18 Head Circumference — A2 summary")


def _render_per_model_plots(
    output_dir: pathlib.Path, results: List[ModelHCResult]
) -> None:
    apply_nm_defaults()

    # Per-model scatter + Bland-Altman
    for r in results:
        valid = np.isfinite(r.gt_hc_mm) & np.isfinite(r.pred_hc_mm)
        gt, pred = r.gt_hc_mm[valid], r.pred_hc_mm[valid]
        color = color_for(r.model_label)
        label = display_name(r.model_label) if r.model_label not in ("GT (ellipse fit)",) else r.model_label

        # Scatter
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        scatter_with_identity(ax, gt, pred, title=f"{label} — HC predicted vs GT",
                              xlabel="Ground-truth HC (mm)",
                              ylabel="Predicted HC (mm)",
                              point_color=color)
        out = output_dir / f"scatter_{r.model_label}.pdf"
        fig.tight_layout()
        fig.savefig(out)
        fig.savefig(out.with_suffix(".png"))
        plt.close(fig)
        logger.info(f"Wrote {out}")

        # Bland-Altman
        fig, ax = plt.subplots(figsize=(5.0, 4.0))
        bland_altman_panel(ax, gt, pred,
                           title=f"{label} — HC Bland-Altman",
                           ylabel="HC difference (pred − GT, mm)",
                           point_color=color)
        out = output_dir / f"bland_altman_{r.model_label}.pdf"
        fig.tight_layout()
        fig.savefig(out)
        fig.savefig(out.with_suffix(".png"))
        plt.close(fig)
        logger.info(f"Wrote {out}")

    # Composite scatter
    panels = []
    for r in results:
        valid = np.isfinite(r.gt_hc_mm) & np.isfinite(r.pred_hc_mm)
        label = display_name(r.model_label) if r.model_label not in ("GT (ellipse fit)",) else r.model_label
        panels.append((label, r.gt_hc_mm[valid], r.pred_hc_mm[valid], color_for(r.model_label)))

    from nemo_cv.components.analysis.plots.scatter import scatter_comparison
    scatter_comparison(panels, xlabel="Ground-truth HC (mm)",
                       ylabel="Predicted HC (mm)",
                       output_path=output_dir / "scatter_comparison.pdf",
                       n_cols=min(3, len(panels)))
    logger.info(f"Wrote {output_dir / 'scatter_comparison.pdf'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_runs_arg(items: List[str]) -> List[Tuple[str, pathlib.Path]]:
    out = []
    for s in items:
        if "=" not in s:
            raise ValueError(f"--runs entry must be `label=path`, got: {s!r}")
        label, path = s.split("=", 1)
        out.append((label.strip(), pathlib.Path(path.strip())))
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(
        description="HC18 head circumference measurement error analysis (A2)."
    )
    parser.add_argument(
        "--runs", nargs="+", required=True,
        help="Space-separated `label=path` entries pointing at Stage-1 prediction dirs."
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Output directory for all A2 artifacts (per_sample.csv, plots, report)."
    )
    _hc18_default = os.environ.get("HC18_ZIP")
    parser.add_argument(
        "--hc18-zip",
        default=_hc18_default,
        required=(_hc18_default is None),
        help="Path to HC18 raw zip (default: $HC18_ZIP env var; required if unset).",
    )
    parser.add_argument(
        "--include-gt-fit", action="store_true",
        help="Also include 'GT (ellipse fit)' as a baseline model in the report.",
    )
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = _parse_runs_arg(args.runs)
    metadata = load_hc18_metadata(args.hc18_zip)
    logger.info(f"HC18 metadata: {len(metadata)} samples")

    # ---- Load + measure each model run ----
    results: List[ModelHCResult] = []
    for label, run_dir in runs:
        logger.info(f"Loading {label}: {run_dir}")
        results.append(_load_model_run(run_dir, label, metadata))

    if args.include_gt_fit:
        # Use the first run's GT mask paths (every run saves the same GT)
        first_run_dir = runs[0][1]
        results.insert(0, _gt_ellipse_fit_baseline(first_run_dir, metadata))

    # ---- Summarize ----
    summaries: Dict[str, Dict[str, float]] = {}
    for r in results:
        s = _summarize(r)
        summaries[r.model_label] = s
        logger.info(
            f"[{r.model_label}] n={s['n']} MAE={s['mae_mm']:.2f}±{s['std_mm']:.2f}mm "
            f"<3mm={s['pct_within_3mm']:.1f}% <5mm={s['pct_within_5mm']:.1f}% "
            f"r={s['pearson_r']:.3f} bias={s['bias_mm']:+.2f}mm "
            f"failed_ellipse={s['n_failed_ellipse']}"
        )

    # ---- Write outputs ----
    _write_report(
        output_dir,
        summaries,
        inputs={label: str(rd) for label, rd in runs},
    )
    _write_per_sample_csv(output_dir, results)
    _render_summary_table(output_dir, summaries)
    _render_per_model_plots(output_dir, results)

    logger.info(f"A2 analysis complete. Outputs at: {output_dir}")


if __name__ == "__main__":
    main()
