"""Cross-analysis aggregator: scan every `analysis_report.json` under a
directory tree and emit a master CSV summarizing the **primary endpoint**
of each analysis × model.

Output: `<output_dir>/cross_analysis_summary.csv` with columns
    analysis | analysis_title | model | metric_name | value | unit | n

Pre-defined per-analysis primary endpoints:

    A1  → MAE (%)              [Ejection fraction, Simpson's biplane]
    A2  → MAE (mm)             [HC18 head circumference]
    A3  → seconds_to_80         [Click efficiency]
    A4  → MAE (mm)             [ACOUSLIC abdominal circumference]
    B1  → mean IoU (per-structure mean across structures)
    B2  → # failures rescued
    B3  → mean per-class IoU
    C2  → MAE (mL)             [Prostate volume]
    T1.1 → # tests significant after BH-FDR
    T1.2 → (figures only — no scalar)
    T2.1 → GA MAE (days)
    T2.2 → reclass_at_40 (gray-zone)
    T2.3 → Cohen's κ
    T3.1 → mean inter-frame IoU
    T3.2 → (qualitative — no scalar)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import pathlib
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Per-analysis primary endpoint extractor: returns list of
# (model_label, metric_name, value, unit) tuples to write.
PrimaryEndpoint = List[Tuple[str, str, float, str]]


def _models_dict(report: dict) -> dict:
    return report.get("models", {})


def _extract_a1(r: dict) -> PrimaryEndpoint:
    out = []
    for m, s in _models_dict(r).items():
        if s.get("n", 0):
            out.append((m, "MAE", float(s["mae_pct"]), "%"))
    return out


def _extract_mae_mm(r: dict, key: str = "mae_mm") -> PrimaryEndpoint:
    out = []
    for m, s in _models_dict(r).items():
        if s.get("n", 0) and s.get(key) is not None:
            out.append((m, "MAE", float(s[key]), "mm"))
    return out


def _extract_a3(r: dict) -> PrimaryEndpoint:
    out = []
    target_pct = int(r.get("target_miou_pct", 80))
    for run_key, s in r.get("runs", {}).items():
        sec = s.get(f"seconds_to_{target_pct}")
        if sec is None:
            continue
        # Encode prompt_init / tier into the model label for clarity
        label = f"{s.get('model')}({s.get('prompt_init')},{s.get('tier')})"
        out.append((label, f"Time-to-{target_pct}", float(sec), "s"))
    return out


def _extract_b1(r: dict) -> PrimaryEndpoint:
    out = []
    for m, classes in _models_dict(r).items():
        if not classes:
            continue
        ious = [v["mean_iou"] for v in classes.values() if v.get("mean_iou") is not None]
        if ious:
            out.append((m, "mean per-structure IoU",
                        float(sum(ious) / len(ious)), ""))
    return out


def _extract_b2(r: dict) -> PrimaryEndpoint:
    return [("(SonoBase rescues)", "# failures resolved",
             float(r.get("n_rows", 0)), "")]


def _extract_b3(r: dict) -> PrimaryEndpoint:
    out = []
    for m, classes in _models_dict(r).items():
        if not classes:
            continue
        ious = [v["mean_iou"] for v in classes.values() if v.get("mean_iou") is not None]
        if ious:
            out.append((m, "mean per-class IoU",
                        float(sum(ious) / len(ious)), ""))
    return out


def _extract_c2(r: dict) -> PrimaryEndpoint:
    out = []
    for m, s in _models_dict(r).items():
        if s.get("n", 0) and s.get("mae_ml") is not None:
            out.append((m, "MAE", float(s["mae_ml"]), "mL"))
    return out


def _extract_t1_1(r: dict) -> PrimaryEndpoint:
    n_sig = sum(1 for t in r.get("wilcoxon_tests", []) if t.get("sig"))
    n_total = len(r.get("wilcoxon_tests", []))
    return [("(family)", "# significant (BH q<α)",
             float(n_sig), f"of {n_total}")]


def _extract_t2_1(r: dict) -> PrimaryEndpoint:
    out = []
    for m, s in _models_dict(r).items():
        if s.get("n", 0):
            out.append((m, "GA MAE", float(s["mae_days"]), "days"))
    return out


def _extract_t2_2(r: dict) -> PrimaryEndpoint:
    out = []
    for m, s in _models_dict(r).items():
        if s.get("n", 0):
            out.append((m, "Reclass @40%", float(s["reclass_at_40_pct"]), "%"))
    return out


def _extract_t2_3(r: dict) -> PrimaryEndpoint:
    if r.get("skipped"):
        return [("(skipped)", "kappa", float("nan"), "")]
    out = []
    for m, s in _models_dict(r).items():
        if s.get("n", 0):
            out.append((m, "Kappa", float(s["kappa"]), ""))
    return out


def _extract_t3_1(r: dict) -> PrimaryEndpoint:
    out = []
    for m, s in _models_dict(r).items():
        if s.get("n_sequences", 0):
            out.append((m, "Mean inter-frame IoU",
                        float(s["mean_inter_frame_iou"]), ""))
    return out


_EXTRACTORS = {
    "A1": _extract_a1,
    "A2": _extract_mae_mm,
    "A3": _extract_a3,
    "A4": _extract_mae_mm,
    "B1": _extract_b1,
    "B2": _extract_b2,
    "B3": _extract_b3,
    "C2": _extract_c2,
    "T1.1": _extract_t1_1,
    "T2.1": _extract_t2_1,
    "T2.2": _extract_t2_2,
    "T2.3": _extract_t2_3,
    "T3.1": _extract_t3_1,
}


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    p = argparse.ArgumentParser(description="Cross-analysis aggregator (compare_models).")
    p.add_argument("--analyses-root", required=True,
                   help="Directory under which to scan for analysis_report.json files "
                        "(typically src/experiments/analysis/).")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    analyses_root = pathlib.Path(args.analyses_root).expanduser().resolve()
    out_dir = pathlib.Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for report_path in sorted(analyses_root.rglob("analysis_report.json")):
        try:
            with report_path.open() as f:
                r = json.load(f)
        except Exception as e:
            logger.warning(f"Skipping unreadable report {report_path}: {e}")
            continue
        aid = r.get("analysis", "?")
        title = r.get("title", "")
        extractor = _EXTRACTORS.get(aid)
        if extractor is None:
            logger.info(f"No primary endpoint extractor for analysis={aid!r} "
                        f"(report at {report_path}); skipping.")
            continue
        try:
            entries = extractor(r)
        except Exception as e:
            logger.warning(f"Extractor failed for {report_path}: {e}")
            continue
        for model, metric_name, value, unit in entries:
            rows.append({
                "analysis": aid,
                "analysis_title": title,
                "model": model,
                "metric": metric_name,
                "value": f"{value:.4f}",
                "unit": unit,
                "report_path": str(report_path.relative_to(analyses_root)),
            })

    if not rows:
        logger.warning("No analysis reports found; the aggregator output will be empty.")

    out_csv = out_dir / "cross_analysis_summary.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "analysis", "analysis_title", "model", "metric", "value", "unit", "report_path",
        ])
        w.writeheader()
        for row in rows:
            w.writerow(row)
    logger.info(f"Wrote {out_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
