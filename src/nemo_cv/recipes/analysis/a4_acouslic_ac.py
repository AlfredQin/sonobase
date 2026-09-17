"""Analysis A4 — ACOUSLIC abdominal circumference measurement error.

Stage-2 analysis. Consumes the Stage-1 video prediction directories for
ACOUSLIC (one per model × prompt) and reports per-video predicted AC vs.
GT AC (mm), with the same suite of outputs as A2 (per_sample CSV +
analysis_report.json + scatter / Bland-Altman / summary table).

Per-video prediction:
  * For each annotated frame the predicted mask is loaded and an ellipse
    is fit + Ramanujan-2 perimeter computed → per-frame AC in mm.
  * The video's predicted AC is the mean over annotated frames.
  * Per-frame ellipse fits that fail (mask too small, fewer than 5
    contour points) are dropped; if all frames fail, the video is
    reported with a null prediction.

Pixel spacing handling:
  * Default = the metadata loader's value (0.28 mm/px, from the MHA
    header). Ground truth must be the ACOUSLIC v1.1 circumference file;
    the loader rejects the 2x-inflated v1.0 file.
  * `--pixel-spacing-mm <float>` exists for sensitivity checks only. It
    must never be used to "calibrate" predictions to a ground-truth file.
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

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from nemo_cv.components.analysis import prediction_io as pio
from nemo_cv.components.analysis import record_io
from nemo_cv.components.analysis.measurements.ellipse_fit import abdominal_circumference_mm
from nemo_cv.components.analysis.metadata_loaders.acouslic import load_acouslic_metadata
from nemo_cv.components.analysis.metadata_loaders.default import ClinicalMeta
from nemo_cv.components.analysis.plots.bland_altman import bland_altman_panel
from nemo_cv.components.analysis.plots.scatter import scatter_with_identity, scatter_comparison
from nemo_cv.components.analysis.plots.style import (
    apply_nm_defaults, color_for, display_name,
)
from nemo_cv.components.analysis.plots.summary_table import summary_table_figure
from nemo_cv.components.analysis.stats.tests import (
    bland_altman_stats, pearson_r,
)

logger = logging.getLogger(__name__)


@dataclass
class _Result:
    model_label: str
    sample_ids: List[str]
    gt_ac_mm: np.ndarray         # one per video
    pred_ac_mm: np.ndarray       # one per video; np.nan when all frames failed
    n_frames_used: List[int]
    n_videos_failed: int


def _load_run(
    run_dir: pathlib.Path,
    model_label: str,
    metadata: Dict[str, ClinicalMeta],
    pixel_spacing_mm_override: float | None,
) -> _Result:
    rows = pio.read_per_sample_csv(run_dir)
    # Group rows by video (sample_id) → list of (frame_idx, pred_mask_path)
    by_video: Dict[str, List[dict]] = {}
    for r in rows:
        by_video.setdefault(r["sample_id"], []).append(r)

    sids: List[str] = []
    gt_arr: List[float] = []
    pred_arr: List[float] = []
    n_frames_used_list: List[int] = []
    n_failed = 0

    for sid, video_rows in by_video.items():
        meta = metadata.get(sid)
        if meta is None or meta.gt_clinical_value is None:
            continue
        spacing = pixel_spacing_mm_override or meta.pixel_spacing_mm
        if spacing is None:
            continue

        per_frame_ac: List[float] = []
        for row in video_rows:
            # only frames that have GT (avoids the implicit zero pred we save
            # for un-annotated frames, which would skew the per-video mean).
            if (row.get("notes") or "") == "no_gt_this_frame":
                continue
            pred_mask_path = run_dir / row["pred_mask_path"]
            if not pred_mask_path.is_file():
                continue
            pred_mask = np.array(Image.open(pred_mask_path).convert("L")) > 0
            ac = abdominal_circumference_mm(pred_mask, spacing)
            if ac is not None:
                per_frame_ac.append(float(ac))

        sids.append(sid)
        gt_arr.append(float(meta.gt_clinical_value))
        if per_frame_ac:
            pred_arr.append(float(np.mean(per_frame_ac)))
        else:
            pred_arr.append(float("nan"))
            n_failed += 1
        n_frames_used_list.append(len(per_frame_ac))

    return _Result(
        model_label, sids,
        np.asarray(gt_arr, dtype=float),
        np.asarray(pred_arr, dtype=float),
        n_frames_used_list, n_failed,
    )


def _summarize(r: _Result) -> Dict[str, float]:
    valid = np.isfinite(r.gt_ac_mm) & np.isfinite(r.pred_ac_mm)
    g, p = r.gt_ac_mm[valid], r.pred_ac_mm[valid]
    if g.size == 0:
        return {"n": 0, "n_failed": int(r.n_videos_failed), "mae_mm": float("nan")}
    err = np.abs(p - g)
    rel = err / np.maximum(np.abs(g), 1e-9) * 100.0
    ba = bland_altman_stats(g, p)
    return {
        "n": int(g.size),
        "n_failed": int(r.n_videos_failed),
        "mae_mm": float(err.mean()),
        "std_mm": float(err.std(ddof=1)) if g.size > 1 else 0.0,
        "rel_err_pct": float(rel.mean()),
        "pearson_r": pearson_r(g, p),
        "bias_mm": ba.bias,
        "loa_lower_mm": ba.loa_lower,
        "loa_upper_mm": ba.loa_upper,
        "mean_frames_per_video": float(np.mean(r.n_frames_used)) if r.n_frames_used else 0.0,
    }


def _write_per_sample_csv(out_dir: pathlib.Path, results: List[_Result]) -> None:
    sids = sorted({s for r in results for s in r.sample_ids})
    headers = ["sample_id", "gt_ac_mm"]
    for r in results:
        headers += [f"pred_ac_mm__{r.model_label}", f"abs_err_mm__{r.model_label}"]
    gt_lookup: Dict[str, float] = {}
    pred_lookup: Dict[Tuple[str, str], float] = {}
    for r in results:
        for i, sid in enumerate(r.sample_ids):
            if sid not in gt_lookup and np.isfinite(r.gt_ac_mm[i]):
                gt_lookup[sid] = float(r.gt_ac_mm[i])
            pred_lookup[(r.model_label, sid)] = float(r.pred_ac_mm[i])
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
    headers = ["Model", "n", "MAE (mm)", "Rel-Err (%)", "r", "Bias (mm)", "LoA (mm)", "Frames/video"]
    rows = [headers]
    for label, s in summaries.items():
        if s["n"] == 0:
            rows.append([display_name(label), "0", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a"])
            continue
        loa = f"[{s['loa_lower_mm']:.2f}, {s['loa_upper_mm']:.2f}]"
        rows.append([
            display_name(label), f"{s['n']:d}",
            f"{s['mae_mm']:.2f} ± {s['std_mm']:.2f}",
            f"{s['rel_err_pct']:.2f}",
            f"{s['pearson_r']:.3f}",
            f"{s['bias_mm']:+.2f}",
            loa,
            f"{s['mean_frames_per_video']:.1f}",
        ])
    summary_table_figure(rows, out_dir / "summary_table.pdf",
                         title="ACOUSLIC Abdominal Circumference — A4 summary")


def _render_plots(out_dir: pathlib.Path, results: List[_Result]) -> None:
    apply_nm_defaults()
    panels = []
    for r in results:
        valid = np.isfinite(r.gt_ac_mm) & np.isfinite(r.pred_ac_mm)
        g, p = r.gt_ac_mm[valid], r.pred_ac_mm[valid]
        if g.size == 0:
            continue
        color = color_for(r.model_label)
        label = display_name(r.model_label)

        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        scatter_with_identity(ax, g, p, title=f"{label} — AC predicted vs GT",
                              xlabel="Ground-truth AC (mm)", ylabel="Predicted AC (mm)",
                              point_color=color)
        out = out_dir / f"scatter_{r.model_label}.pdf"
        fig.tight_layout(); fig.savefig(out); fig.savefig(out.with_suffix(".png"))
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5.0, 4.0))
        bland_altman_panel(ax, g, p,
                           title=f"{label} — AC Bland-Altman",
                           ylabel="AC difference (pred − GT, mm)",
                           point_color=color)
        out = out_dir / f"bland_altman_{r.model_label}.pdf"
        fig.tight_layout(); fig.savefig(out); fig.savefig(out.with_suffix(".png"))
        plt.close(fig)

        panels.append((label, g, p, color))

    if panels:
        scatter_comparison(panels, xlabel="Ground-truth AC (mm)", ylabel="Predicted AC (mm)",
                           output_path=out_dir / "scatter_comparison.pdf",
                           n_cols=min(3, len(panels)))


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    p = argparse.ArgumentParser(description="ACOUSLIC abdominal circumference (A4).")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--output-dir", required=True)
    _acouslic_default = os.environ.get("ACOUSLIC_CSV")
    p.add_argument("--acouslic-csv", default=_acouslic_default,
                   required=(_acouslic_default is None),
                   help="Per-sweep GT AC CSV path (default: $ACOUSLIC_CSV env var; required if unset).")
    p.add_argument("--acouslic-dir",
                   default=None,
                   help="ACOUSLIC root (used to search for the CSV if --acouslic-csv is missing).")
    p.add_argument("--pixel-spacing-mm", type=float, default=None,
                   help="Override per-image pixel spacing for sensitivity checks only "
                        "(default: metadata loader value, 0.28 mm/px from the MHA header). "
                        "Never use this to fit predictions to a ground-truth file.")
    args = p.parse_args()

    out_dir = pathlib.Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for s in args.runs:
        if "=" not in s:
            raise ValueError(f"--runs entry must be `label=path`, got: {s!r}")
        label, path = s.split("=", 1)
        runs.append((label.strip(), pathlib.Path(path.strip())))

    meta = load_acouslic_metadata(args.acouslic_csv, args.acouslic_dir)
    if not meta:
        logger.error(
            "ACOUSLIC metadata is empty — cannot run A4. Set --acouslic-csv "
            "to the per-sweep CSV path."
        )
        return

    results = [_load_run(rd, lbl, meta, args.pixel_spacing_mm) for lbl, rd in runs]
    summaries = {r.model_label: _summarize(r) for r in results}
    for lbl, s in summaries.items():
        if s["n"]:
            logger.info(
                f"[{lbl}] n={s['n']} MAE={s['mae_mm']:.2f}±{s['std_mm']:.2f}mm "
                f"r={s['pearson_r']:.3f} bias={s['bias_mm']:+.2f}mm "
                f"failed={s['n_failed']}"
            )

    with (out_dir / "analysis_report.json").open("w") as f:
        json.dump({
            "analysis": "A4",
            "title": "ACOUSLIC Abdominal Circumference",
            "spec_section": "A4",
            "inputs": {label: str(rd) for label, rd in runs},
            "pixel_spacing_mm_override": args.pixel_spacing_mm,
            "models": summaries,
            "verification": record_io.declare("per_sample.csv", [
                record_io.check(["models", "$model", "mae_mm"], "abs_err_mm__{model}"),
                record_io.check(["models", "$model", "std_mm"], "abs_err_mm__{model}", agg="std"),
                record_io.check(["models", "$model", "n"], "abs_err_mm__{model}", agg="count"),
            ]),
        }, f, indent=2)

    _write_per_sample_csv(out_dir, results)
    _render_table(out_dir, summaries)
    _render_plots(out_dir, results)
    logger.info(f"A4 analysis complete. Outputs at: {out_dir}")


if __name__ == "__main__":
    main()
