"""Analysis T2.1 — HC → Gestational Age (Hadlock 1984).

Stage-2 analysis. Consumes the Stage-1 HC18 prediction directories (same
inputs as A2) and reports per-model gestational-age error in days,
together with the clinically meaningful "% within 3 days / 7 days / >14
days" thresholds.

CPU only. Uses `measurements/ga_dating.py` (Hadlock formula) which is
already implemented and ships with its citation header.

CLI mirror of A2 (same `--runs` / `--output-dir` / `--hc18-zip`):

    python -m nemo_cv.recipes.analysis.t2_1_hc_ga \\
        --runs sam2_no_ft=<dir> medsam2=<dir> sonobase=<dir> \\
        --output-dir <output> \\
        --hc18-zip "$HC18_ZIP" \\        # or set HC18_ZIP env var
        [--include-gt-fit]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import pathlib
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

from nemo_cv.components.analysis import prediction_io as pio
from nemo_cv.components.analysis import record_io
from nemo_cv.components.analysis.measurements.ellipse_fit import head_circumference_mm
from nemo_cv.components.analysis.measurements.ga_dating import hc_to_ga_days
from nemo_cv.components.analysis.metadata_loaders.default import ClinicalMeta
from nemo_cv.components.analysis.metadata_loaders.hc18 import load_hc18_metadata
from nemo_cv.components.analysis.plots.style import display_name
from nemo_cv.components.analysis.plots.summary_table import summary_table_figure
from nemo_cv.components.analysis.stats.tests import bootstrap_ci_mean, percent_within

logger = logging.getLogger(__name__)


@dataclass
class _Result:
    model_label: str
    sample_ids: List[str]
    gt_ga_days: np.ndarray
    pred_ga_days: np.ndarray


def _load_run(run_dir: pathlib.Path, model_label: str,
              meta: Dict[str, ClinicalMeta]) -> _Result:
    """For each sample: read pred mask → predicted HC mm → predicted GA days."""
    rows = pio.read_per_sample_csv(run_dir)
    sids: List[str] = []
    gt_arr: List[float] = []
    pred_arr: List[float] = []

    for row in rows:
        sid = row["sample_id"]
        m = meta.get(sid)
        if m is None or m.gt_clinical_value is None:
            continue
        pred_mask_path = run_dir / row["pred_mask_path"]
        if not pred_mask_path.is_file():
            continue
        pred_mask = np.array(Image.open(pred_mask_path).convert("L")) > 0
        pred_hc = head_circumference_mm(pred_mask, m.pixel_spacing_mm)

        sids.append(sid)
        gt_arr.append(hc_to_ga_days(float(m.gt_clinical_value)))
        pred_arr.append(hc_to_ga_days(float(pred_hc)) if pred_hc is not None else float("nan"))

    return _Result(model_label, sids,
                   np.asarray(gt_arr, dtype=float),
                   np.asarray(pred_arr, dtype=float))


def _summarize(r: _Result) -> Dict[str, float]:
    valid = np.isfinite(r.gt_ga_days) & np.isfinite(r.pred_ga_days)
    g, p = r.gt_ga_days[valid], r.pred_ga_days[valid]
    if g.size == 0:
        return {"n": 0, "mae_days": float("nan")}
    err = np.abs(p - g)
    mean_mae, lo, hi = bootstrap_ci_mean(err, n_bootstrap=10_000, seed=42)
    return {
        "n": int(g.size),
        "mae_days": float(err.mean()),
        "std_days": float(err.std(ddof=1)) if g.size > 1 else 0.0,
        "mae_days_ci95": [lo, hi],
        "pct_within_3d": percent_within(err, 3.0),
        "pct_within_7d": percent_within(err, 7.0),
        "pct_above_14d": float((err > 14).mean() * 100),
        "bias_days": float((p - g).mean()),
    }


def _write_per_sample_csv(out_dir: pathlib.Path, results: List[_Result]) -> None:
    sids = sorted({s for r in results for s in r.sample_ids})
    headers = ["sample_id", "gt_ga_days"]
    for r in results:
        headers += [f"pred_ga_days__{r.model_label}", f"abs_err_days__{r.model_label}"]
    gt_lookup: Dict[str, float] = {}
    pred_lookup: Dict[Tuple[str, str], float] = {}
    for r in results:
        for i, sid in enumerate(r.sample_ids):
            if sid not in gt_lookup and np.isfinite(r.gt_ga_days[i]):
                gt_lookup[sid] = float(r.gt_ga_days[i])
            pred_lookup[(r.model_label, sid)] = float(r.pred_ga_days[i])
    path = out_dir / "per_sample.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for sid in sids:
            gt = gt_lookup.get(sid)
            row = [sid, "" if gt is None else f"{gt:.4f}"]
            for r in results:
                pr = pred_lookup.get((r.model_label, sid))
                row.append("" if pr is None or not math.isfinite(pr) else f"{pr:.4f}")
                if pr is None or gt is None or not math.isfinite(pr):
                    row.append("")
                else:
                    row.append(f"{abs(pr - gt):.4f}")
            w.writerow(row)
    logger.info(f"Wrote {path}")


def _render_table(out_dir: pathlib.Path, summaries: Dict[str, Dict]) -> None:
    headers = ["Model", "n", "GA MAE (days)", "≤3 days (%)", "≤7 days (%)", ">14 days (%)", "Bias (days)"]
    rows = [headers]
    for label, s in summaries.items():
        if s["n"] == 0:
            continue
        rows.append([
            display_name(label),
            f"{s['n']:d}",
            f"{s['mae_days']:.2f} ± {s['std_days']:.2f}",
            f"{s['pct_within_3d']:.1f}",
            f"{s['pct_within_7d']:.1f}",
            f"{s['pct_above_14d']:.1f}",
            f"{s['bias_days']:+.2f}",
        ])
    summary_table_figure(rows, out_dir / "summary_table.pdf",
                         title="HC → Gestational Age (Hadlock 1984) — T2.1 summary")


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    p = argparse.ArgumentParser(description="HC → Gestational Age error (T2.1).")
    p.add_argument("--runs", nargs="+", required=True,
                   help="Space-separated `label=path` Stage-1 prediction dirs.")
    p.add_argument("--output-dir", required=True)
    _hc18_default = os.environ.get("HC18_ZIP")
    p.add_argument("--hc18-zip", default=_hc18_default,
                   required=(_hc18_default is None),
                   help="Path to HC18 raw zip (default: $HC18_ZIP env var; required if unset).")
    args = p.parse_args()

    out_dir = pathlib.Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for s in args.runs:
        if "=" not in s:
            raise ValueError(f"--runs entry must be `label=path`, got: {s!r}")
        label, path = s.split("=", 1)
        runs.append((label.strip(), pathlib.Path(path.strip())))

    meta = load_hc18_metadata(args.hc18_zip)

    results = [_load_run(rd, lbl, meta) for lbl, rd in runs]
    summaries = {r.model_label: _summarize(r) for r in results}
    for lbl, s in summaries.items():
        if s["n"]:
            logger.info(
                f"[{lbl}] n={s['n']} GA-MAE={s['mae_days']:.2f}±{s['std_days']:.2f} days "
                f"<=3d={s['pct_within_3d']:.1f}% <=7d={s['pct_within_7d']:.1f}% "
                f">14d={s['pct_above_14d']:.1f}%"
            )

    with (out_dir / "analysis_report.json").open("w") as f:
        json.dump({
            "analysis": "T2.1",
            "title": "HC → Gestational Age (Hadlock 1984)",
            "spec_section": "T2.1",
            "inputs": {label: str(rd) for label, rd in runs},
            "models": summaries,
            "verification": record_io.declare("per_sample.csv", [
                record_io.check(["models", "$model", "mae_days"], "abs_err_days__{model}"),
                record_io.check(["models", "$model", "std_days"], "abs_err_days__{model}", agg="std"),
                record_io.check(["models", "$model", "n"], "abs_err_days__{model}", agg="count"),
            ]),
        }, f, indent=2)

    _write_per_sample_csv(out_dir, results)
    _render_table(out_dir, summaries)
    logger.info(f"T2.1 analysis complete. Outputs at: {out_dir}")


if __name__ == "__main__":
    main()
