"""Analysis T2.3 — FGR screening simulation (conditional).

Strict-conditional analysis. **Skips with a clear message** if ACOUSLIC
GT metadata doesn't include per-video gestational age — the FGR
classification can't be computed without GA.

When data is available, classifies each video as FGR-positive iff
predicted (or GT) AC < INTERGROWTH-21st 10th percentile at that GA.
Reports sensitivity, specificity, PPV, NPV, Cohen's κ.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import pathlib
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from nemo_cv.components.analysis import prediction_io as pio
from nemo_cv.components.analysis import record_io
from nemo_cv.components.analysis.measurements.ellipse_fit import abdominal_circumference_mm
from nemo_cv.components.analysis.measurements.intergrowth_21st import (
    ac_10th_percentile_mm, is_fgr,
)
from nemo_cv.components.analysis.metadata_loaders.acouslic import load_acouslic_metadata
from nemo_cv.components.analysis.plots.style import display_name
from nemo_cv.components.analysis.plots.summary_table import summary_table_figure
from nemo_cv.components.analysis.stats.tests import cohens_kappa

logger = logging.getLogger(__name__)


def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Sensitivity / specificity / PPV / NPV / κ from binary arrays."""
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    kappa = cohens_kappa(y_true, y_pred) if y_true.size >= 2 else float("nan")
    return {
        "n": int(y_true.size), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "sensitivity": sens, "specificity": spec, "ppv": ppv, "npv": npv,
        "kappa": kappa,
    }


def _per_video_pred_ac(run_dir: pathlib.Path, sample_id: str,
                       pixel_spacing_mm: float) -> Optional[float]:
    """Mean predicted AC across annotated frames for one video."""
    sd = run_dir / "ACOUSLIC" / sample_id
    if not sd.is_dir():
        return None
    acs: List[float] = []
    for path in sorted(sd.glob("pred_*_obj_0.png")):
        # Skip frames marked no_gt — but the per_sample CSV already excludes
        # them via `notes` and we re-validate below by checking the matching
        # GT mask exists and is non-empty.
        f_idx = int(path.stem.split("_")[1])
        gt_path = sd / f"gt_{f_idx:05d}_obj_0.png"
        if not gt_path.is_file():
            continue
        gt = np.array(Image.open(gt_path).convert("L")) > 0
        if not gt.any():
            continue
        pred = np.array(Image.open(path).convert("L")) > 0
        ac = abdominal_circumference_mm(pred, pixel_spacing_mm)
        if ac is not None:
            acs.append(float(ac))
    if not acs:
        return None
    return float(np.mean(acs))


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    p = argparse.ArgumentParser(description="FGR screening simulation (T2.3).")
    p.add_argument("--runs", nargs="+", required=True,
                   help="`label=path` Stage-1 ACOUSLIC prediction dirs.")
    p.add_argument("--output-dir", required=True)
    _acouslic_default = os.environ.get("ACOUSLIC_CSV")
    p.add_argument("--acouslic-csv", default=_acouslic_default,
                   required=(_acouslic_default is None),
                   help="Per-sweep GT AC CSV path (default: $ACOUSLIC_CSV env var; required if unset).")
    p.add_argument("--pixel-spacing-mm", type=float, default=None,
                   help="Override the per-image pixel spacing (sensitivity checks only; "
                        "default 0.28 mm/px from the MHA header).")
    args = p.parse_args()

    out_dir = pathlib.Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_acouslic_metadata(args.acouslic_csv)
    if not metadata:
        logger.error(
            "ACOUSLIC metadata is empty — cannot run T2.3. "
            "Set --acouslic-csv to the per-sweep CSV path."
        )
        with (out_dir / "analysis_report.json").open("w") as f:
            json.dump({
                "analysis": "T2.3",
                "title": "FGR screening simulation",
                "skipped": True,
                "reason": "ACOUSLIC GT CSV not found",
                "verification": record_io.declare(
                    None, [], skipped_reason="ACOUSLIC GT CSV not found — no statistics produced"),
            }, f, indent=2)
        return

    have_ga = sum(1 for m in metadata.values() if m.gestational_age_weeks is not None)
    if have_ga == 0:
        logger.error(
            "ACOUSLIC metadata has no gestational_age_weeks values — T2.3 cannot run. "
            "This analysis is conditional on gestational-age metadata."
        )
        with (out_dir / "analysis_report.json").open("w") as f:
            json.dump({
                "analysis": "T2.3",
                "title": "FGR screening simulation",
                "skipped": True,
                "reason": "ACOUSLIC GA metadata absent — conditional analysis not applicable",
                "n_videos_in_metadata": len(metadata),
                "verification": record_io.declare(
                    None, [],
                    skipped_reason="ACOUSLIC GA metadata absent — no statistics produced"),
            }, f, indent=2)
        logger.info(f"T2.3 skipped (no GA metadata). Stub written at {out_dir}")
        return

    logger.info(f"ACOUSLIC metadata: {len(metadata)} videos, {have_ga} with GA")

    runs = []
    for s in args.runs:
        if "=" not in s:
            raise ValueError(f"--runs entry must be `label=path`, got: {s!r}")
        label, path = s.split("=", 1)
        runs.append((label.strip(), pathlib.Path(path.strip())))

    summaries: Dict[str, Dict] = {}
    per_sample_rows: List[Dict] = []

    for label, run_dir in runs:
        y_true: List[int] = []
        y_pred: List[int] = []
        for sid, meta in metadata.items():
            if meta.gestational_age_weeks is None or meta.gt_clinical_value is None:
                continue
            spacing = args.pixel_spacing_mm or meta.pixel_spacing_mm
            if spacing is None:
                continue

            gt_fgr = is_fgr(meta.gt_clinical_value, meta.gestational_age_weeks)
            if gt_fgr is None:
                continue   # GA out of supported range

            pred_ac = _per_video_pred_ac(run_dir, sid, spacing)
            if pred_ac is None or not math.isfinite(pred_ac):
                continue
            pred_fgr = is_fgr(pred_ac, meta.gestational_age_weeks)
            if pred_fgr is None:
                continue
            y_true.append(int(gt_fgr))
            y_pred.append(int(pred_fgr))
            per_sample_rows.append({
                "model": label,
                "sample_id": sid,
                "ga_weeks": meta.gestational_age_weeks,
                "gt_ac_mm": meta.gt_clinical_value,
                "pred_ac_mm": pred_ac,
                "ac_p10_mm": ac_10th_percentile_mm(meta.gestational_age_weeks),
                "gt_fgr": int(gt_fgr),
                "pred_fgr": int(pred_fgr),
            })

        cm = _confusion(np.array(y_true), np.array(y_pred))
        summaries[label] = cm
        logger.info(
            f"[{label}] n={cm['n']} sens={cm['sensitivity']:.3f} spec={cm['specificity']:.3f} "
            f"PPV={cm['ppv']:.3f} NPV={cm['npv']:.3f} κ={cm['kappa']:.3f}"
        )

    # Per-sample CSV
    csv_path = out_dir / "per_sample.csv"
    if per_sample_rows:
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per_sample_rows[0].keys()))
            w.writeheader()
            w.writerows(per_sample_rows)
        logger.info(f"Wrote {csv_path}")

    headers = ["Model", "n", "Sens", "Spec", "PPV", "NPV", "κ", "TP/FP/TN/FN"]
    rows = [headers]
    for label, s in summaries.items():
        if s["n"] == 0:
            rows.append([display_name(label), "0"] + ["n/a"] * 6)
            continue
        rows.append([
            display_name(label), f"{s['n']:d}",
            f"{s['sensitivity']:.3f}", f"{s['specificity']:.3f}",
            f"{s['ppv']:.3f}", f"{s['npv']:.3f}",
            f"{s['kappa']:.3f}",
            f"{s['tp']}/{s['fp']}/{s['tn']}/{s['fn']}",
        ])
    summary_table_figure(rows, out_dir / "summary_table.pdf",
                         title="FGR screening simulation — T2.3")

    with (out_dir / "analysis_report.json").open("w") as f:
        json.dump({
            "analysis": "T2.3",
            "title": "FGR screening simulation",
            "spec_section": "T2.3",
            "skipped": False,
            "inputs": {label: str(rd) for label, rd in runs},
            "models": summaries,
            # Only `n` is an aggregation of a column; sensitivity / specificity
            # / PPV / NPV / κ derive from the confusion counts, so a checker
            # reports them as uncovered instead of assuming they are verified.
            "verification": record_io.declare("per_sample.csv", [
                record_io.check(["models", "$model", "n"], "gt_fgr", agg="count"),
            ]),
        }, f, indent=2)

    logger.info(f"T2.3 analysis complete. Outputs at: {out_dir}")


if __name__ == "__main__":
    main()
