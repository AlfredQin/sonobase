"""Analysis T2.2 — EF gray zone (35-45 %) reclassification.

Stage-2 analysis. Filters the A1 per-patient table to the gray zone
(35 ≤ GT_EF ≤ 45) and recomputes MAE + reclassification rates within
that subset. If `n_gray < 15`, automatically expand to (30, 50).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import pathlib
from typing import Dict, List, Optional, Tuple

import numpy as np

from nemo_cv.components.analysis import record_io
from nemo_cv.components.analysis.plots.style import display_name
from nemo_cv.components.analysis.plots.summary_table import summary_table_figure
from nemo_cv.components.analysis.stats.tests import bland_altman_stats, cohens_kappa, pearson_r

logger = logging.getLogger(__name__)


HFREF_THRESHOLD = 40.0
ICD_THRESHOLD = 35.0


def _read_a1_per_patient(a1_csv: pathlib.Path) -> List[Dict]:
    out = []
    with a1_csv.open() as f:
        for row in csv.DictReader(f):
            out.append(row)
    return out


def _filter_gray_zone(rows: List[Dict], lo: float, hi: float) -> List[Dict]:
    out = []
    for r in rows:
        try:
            gt = float(r["gt_ef"])
        except (KeyError, ValueError):
            continue
        if lo <= gt <= hi:
            out.append(r)
    return out


def _arrays_for_model(rows: List[Dict], model_label: str,
                      ef_field: str = "pred_ef_biplane",
                      kept: Optional[List[Dict]] = None,
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Paired (GT, predicted) EF for one model within the already-filtered rows.

    ``ef_field`` selects the EF convention. A1 emits more than one — the
    view-averaged ``pred_ef_biplane`` and the ASE disc-pairing
    ``pred_ef_biplane_ase`` — and which one a table used is not recoverable
    from the numbers afterwards, so it is recorded in the report.
    """
    g, p = [], []
    for r in rows:
        try:
            gt = float(r["gt_ef"])
            pred = float(r[f"{ef_field}__{model_label}"])
        except (KeyError, ValueError):
            continue
        if not math.isfinite(pred):
            continue
        g.append(gt); p.append(pred)
        if kept is not None:
            kept.append({
                "model": model_label,
                "patient_id": r.get("patient_id") or r.get("sample_id", ""),
                "gt_ef": gt,
                "pred_ef": pred,
                "abs_err_pct": abs(pred - gt),
                "ef_field": ef_field,
            })
    return np.asarray(g), np.asarray(p)


def _summarize(g: np.ndarray, p: np.ndarray) -> Dict:
    if g.size == 0:
        return {"n": 0}
    err = np.abs(p - g)
    ba = bland_altman_stats(g, p)
    gt_hfref = (g <= HFREF_THRESHOLD).astype(int)
    pr_hfref = (p <= HFREF_THRESHOLD).astype(int)
    return {
        "n": int(g.size),
        "mae_pct": float(err.mean()),
        "std_pct": float(err.std(ddof=1)) if g.size > 1 else 0.0,
        "pearson_r": pearson_r(g, p),
        "bias_pct": ba.bias,
        "loa_lower_pct": ba.loa_lower,
        "loa_upper_pct": ba.loa_upper,
        "reclass_at_40_pct": float((gt_hfref != pr_hfref).mean() * 100),
        "kappa_at_40": cohens_kappa(gt_hfref, pr_hfref) if g.size >= 2 else float("nan"),
    }


def _model_labels(rows: List[Dict], ef_field: str = "pred_ef_biplane") -> List[str]:
    if not rows:
        return []
    prefix = f"{ef_field}__"
    # The trailing `__` already separates field from label, so a shorter
    # ef_field can't bleed into a longer one's columns. The extra guard keeps
    # that true if ef_field is ever passed a value that stops mid-name.
    return [k[len(prefix):] for k in rows[0].keys()
            if k.startswith(prefix) and "__" not in k[len(prefix):]]


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    p = argparse.ArgumentParser(description="EF gray-zone reclassification (T2.2).")
    p.add_argument("--a1-per-patient-csv", required=True,
                   help="Path to A1's per_patient.csv output.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--gray-lo", type=float, default=35.0)
    p.add_argument("--gray-hi", type=float, default=45.0)
    p.add_argument("--gray-lo-expanded", type=float, default=30.0)
    p.add_argument("--gray-hi-expanded", type=float, default=50.0)
    p.add_argument("--min-n", type=int, default=15)
    p.add_argument("--ef-field", default="pred_ef_biplane",
                   choices=["pred_ef_biplane", "pred_ef_biplane_ase"],
                   help="Which EF convention from A1 to analyse. Default is the "
                        "view-averaged biplane; `pred_ef_biplane_ase` is the ASE "
                        "disc-pairing variant. Recorded in the report either way.")
    args = p.parse_args()

    out_dir = pathlib.Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_a1_per_patient(pathlib.Path(args.a1_per_patient_csv))
    labels = _model_labels(rows, args.ef_field)
    logger.info(f"Loaded {len(rows)} patients; EF field {args.ef_field}; models: {labels}")

    gray = _filter_gray_zone(rows, args.gray_lo, args.gray_hi)
    used_lo, used_hi = args.gray_lo, args.gray_hi
    if len(gray) < args.min_n:
        logger.info(
            f"Gray-zone ({args.gray_lo}, {args.gray_hi}) has only {len(gray)} patients (< {args.min_n}); "
            f"expanding to ({args.gray_lo_expanded}, {args.gray_hi_expanded})."
        )
        gray = _filter_gray_zone(rows, args.gray_lo_expanded, args.gray_hi_expanded)
        used_lo, used_hi = args.gray_lo_expanded, args.gray_hi_expanded
    logger.info(f"Gray-zone n = {len(gray)} (range {used_lo}-{used_hi})")

    summaries: Dict[str, Dict] = {}
    kept: List[Dict] = []
    for label in labels:
        g, p = _arrays_for_model(gray, label, args.ef_field, kept)
        s = _summarize(g, p)
        summaries[label] = s
        if s["n"]:
            logger.info(
                f"[{label}] gray n={s['n']} MAE={s['mae_pct']:.2f}% "
                f"reclass@40={s['reclass_at_40_pct']:.1f}% κ@40={s['kappa_at_40']:.3f}"
            )

    headers = ["Model", "n_gray", "MAE (%)", "Reclass @40 (%)", "κ@40", "Bias (%)"]
    table_rows = [headers]
    for label, s in summaries.items():
        if s["n"] == 0:
            table_rows.append([display_name(label), "0", "n/a", "n/a", "n/a", "n/a"])
            continue
        table_rows.append([
            display_name(label), f"{s['n']:d}",
            f"{s['mae_pct']:.2f} ± {s['std_pct']:.2f}",
            f"{s['reclass_at_40_pct']:.1f}",
            f"{s['kappa_at_40']:.3f}",
            f"{s['bias_pct']:+.2f}",
        ])
    summary_table_figure(table_rows, out_dir / "summary_table.pdf",
                         title=f"EF gray zone ({used_lo:.0f}–{used_hi:.0f}%) — T2.2")

    record_io.write_dump(out_dir, kept)

    # `n_in_gray` counts patients inside the EF window; each model's `n` counts
    # those that also have a computable EF, and the two differ sharply (147 vs 23
    # on CAMUS point). The cause is upstream of this analysis: A1 writes a row for
    # every patient in the CAMUS metadata (500), but the evaluation split covers
    # only ~100 of them, so the rest have no prediction and hence no EF from any
    # view. Verified against the A1 table: of the 147 gray-zone patients, 23 have
    # a 4CH EF, 23 have a 2CH EF, and it is the same 23 -- none is lost to having
    # only one of the two views. The drop hits the GT control identically, so it
    # is a property of the split rather than of any model -- but nothing in the
    # record said so, which made the analysed cohort look 6x larger than it is.
    n_analysed = {m: s.get("n", 0) for m, s in summaries.items()}
    with (out_dir / "analysis_report.json").open("w") as f:
        json.dump({
            "analysis": "T2.2",
            "title": "EF gray-zone reclassification",
            "spec_section": "T2.2",
            "input_csv": str(args.a1_per_patient_csv),
            "ef_field": args.ef_field,
            "gray_range": [used_lo, used_hi],
            "n_in_gray": len(gray),
            "n_analysed_per_model": n_analysed,
            "n_dropped_no_biplane_ef": {m: len(gray) - n for m, n in n_analysed.items()},
            "models": summaries,
            "verification": record_io.declare("per_sample.csv", [
                record_io.check(["models", "$model", "mae_pct"], "abs_err_pct"),
                record_io.check(["models", "$model", "std_pct"], "abs_err_pct", agg="std"),
                record_io.check(["models", "$model", "n"], "abs_err_pct", agg="count"),
            ]),
        }, f, indent=2)

    logger.info(f"T2.2 analysis complete. Outputs at: {out_dir}")


if __name__ == "__main__":
    main()
