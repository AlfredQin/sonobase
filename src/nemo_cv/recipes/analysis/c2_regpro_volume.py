"""Analysis C2 — RegPro prostate volume.

Stage-2 analysis. Uses the per-slice predicted masks from the Stage-1
RegPro run (one PNG per slice per case) and computes per-case prostate
volume in mL via voxel summation × NIfTI voxel-spacing.

Pipeline:
  1. For each case, enumerate per-slice predicted-mask PNGs from Stage 1.
  2. Sum foreground voxel count across all slices.
  3. Multiply by voxel volume (sx × sy × sz, mm³) read from the raw
     NIfTI header (loaded lazily by `metadata_loaders.regpro`).
  4. Convert mm³ → mL.
  5. Compare to GT volume computed the same way from GT masks.

Inter-observer reference: ±11.4 % (Tong 1998).
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
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from nemo_cv.components.analysis import prediction_io as pio
from nemo_cv.components.analysis import record_io
from nemo_cv.components.analysis.measurements.volumetric import voxel_volume_mL
from nemo_cv.components.analysis.metadata_loaders.regpro import get_metadata as regpro_meta
from nemo_cv.components.analysis.plots.bland_altman import bland_altman_panel
from nemo_cv.components.analysis.plots.scatter import scatter_with_identity, scatter_comparison
from nemo_cv.components.analysis.plots.style import (
    apply_nm_defaults, color_for, display_name,
)
from nemo_cv.components.analysis.plots.summary_table import summary_table_figure
from nemo_cv.components.analysis.stats.tests import bland_altman_stats, pearson_r

logger = logging.getLogger(__name__)


PROSTATE_OBJ_ID = 0   # RegPro single-object dataset (per its dataset_info.json)


@dataclass
class _Result:
    model_label: str
    sample_ids: List[str]
    gt_vol_ml: np.ndarray
    pred_vol_ml: np.ndarray
    voxel_mm3: List[float]


def _stack_for(run_dir: pathlib.Path, sample_id: str, kind: str) -> List[np.ndarray]:
    """Load every per-slice mask PNG for a sample, sorted by frame_idx.

    `kind` is 'pred' or 'gt' — selects the file prefix.
    """
    out: List[np.ndarray] = []
    sd = run_dir / "RegPro" / sample_id
    if not sd.is_dir():
        return out
    for path in sorted(sd.glob(f"{kind}_*_obj_{PROSTATE_OBJ_ID}.png")):
        out.append(np.array(Image.open(path).convert("L")) > 0)
    return out


def _load_run(run_dir: pathlib.Path, model_label: str,
              regpro_zip_path: str) -> _Result:
    rows = pio.read_per_sample_csv(run_dir)
    sids = sorted({r["sample_id"] for r in rows})

    gt_arr: List[float] = []
    pred_arr: List[float] = []
    voxel_list: List[float] = []
    keep_sids: List[str] = []

    for sid in sids:
        meta = regpro_meta(sid, regpro_zip_path)
        if meta is None or meta.voxel_spacing_mm3 is None:
            logger.warning(f"RegPro voxel spacing missing for {sid}; skipping")
            continue
        v_mm3 = float(meta.voxel_spacing_mm3)

        pred_stack = _stack_for(run_dir, sid, "pred")
        gt_stack = _stack_for(run_dir, sid, "gt")
        if not gt_stack:
            continue

        keep_sids.append(sid)
        voxel_list.append(v_mm3)
        gt_arr.append(voxel_volume_mL(gt_stack, v_mm3))
        pred_arr.append(voxel_volume_mL(pred_stack, v_mm3) if pred_stack else float("nan"))

    return _Result(model_label, keep_sids,
                   np.asarray(gt_arr, dtype=float),
                   np.asarray(pred_arr, dtype=float),
                   voxel_list)


def _summarize(r: _Result) -> Dict[str, float]:
    valid = np.isfinite(r.gt_vol_ml) & np.isfinite(r.pred_vol_ml)
    g, p = r.gt_vol_ml[valid], r.pred_vol_ml[valid]
    if g.size == 0:
        return {"n": 0}
    err = np.abs(p - g)
    rel = err / np.maximum(np.abs(g), 1e-9) * 100.0
    ba = bland_altman_stats(g, p)
    return {
        "n": int(g.size),
        "mae_ml": float(err.mean()),
        "std_ml": float(err.std(ddof=1)) if g.size > 1 else 0.0,
        "rel_err_pct": float(rel.mean()),
        "pearson_r": pearson_r(g, p),
        "bias_ml": ba.bias,
        "loa_lower_ml": ba.loa_lower,
        "loa_upper_ml": ba.loa_upper,
    }


def _write_per_sample_csv(out_dir: pathlib.Path, results: List[_Result]) -> None:
    sids = sorted({s for r in results for s in r.sample_ids})
    headers = ["sample_id", "gt_vol_ml"]
    for r in results:
        headers += [f"pred_vol_ml__{r.model_label}", f"abs_err_ml__{r.model_label}"]
    gt_lookup: Dict[str, float] = {}
    pred_lookup: Dict[Tuple[str, str], float] = {}
    for r in results:
        for i, sid in enumerate(r.sample_ids):
            if sid not in gt_lookup and np.isfinite(r.gt_vol_ml[i]):
                gt_lookup[sid] = float(r.gt_vol_ml[i])
            pred_lookup[(r.model_label, sid)] = float(r.pred_vol_ml[i])
    path = out_dir / "per_sample.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(headers)
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
    headers = ["Model", "n", "MAE (mL)", "Rel-Err (%)", "r", "Bias (mL)", "LoA (mL)"]
    rows = [headers]
    for label, s in summaries.items():
        if s["n"] == 0:
            rows.append([display_name(label), "0"] + ["n/a"] * 5)
            continue
        loa = f"[{s['loa_lower_ml']:.2f}, {s['loa_upper_ml']:.2f}]"
        rows.append([
            display_name(label), f"{s['n']:d}",
            f"{s['mae_ml']:.2f} ± {s['std_ml']:.2f}",
            f"{s['rel_err_pct']:.2f}",
            f"{s['pearson_r']:.3f}",
            f"{s['bias_ml']:+.2f}", loa,
        ])
    summary_table_figure(rows, out_dir / "summary_table.pdf",
                         title="RegPro Prostate Volume — C2 summary")


def _render_plots(out_dir: pathlib.Path, results: List[_Result]) -> None:
    apply_nm_defaults()
    panels = []
    for r in results:
        valid = np.isfinite(r.gt_vol_ml) & np.isfinite(r.pred_vol_ml)
        g, p = r.gt_vol_ml[valid], r.pred_vol_ml[valid]
        if g.size == 0:
            continue
        color = color_for(r.model_label); label = display_name(r.model_label)

        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        scatter_with_identity(ax, g, p, title=f"{label} — Volume predicted vs GT",
                              xlabel="Ground-truth volume (mL)",
                              ylabel="Predicted volume (mL)", point_color=color)
        out = out_dir / f"scatter_{r.model_label}.pdf"
        fig.tight_layout(); fig.savefig(out); fig.savefig(out.with_suffix(".png"))
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5.0, 4.0))
        bland_altman_panel(ax, g, p, title=f"{label} — Volume Bland-Altman",
                           ylabel="Volume diff (pred − GT, mL)", point_color=color)
        out = out_dir / f"bland_altman_{r.model_label}.pdf"
        fig.tight_layout(); fig.savefig(out); fig.savefig(out.with_suffix(".png"))
        plt.close(fig)
        panels.append((label, g, p, color))

    if panels:
        scatter_comparison(panels, xlabel="Ground-truth volume (mL)",
                           ylabel="Predicted volume (mL)",
                           output_path=out_dir / "scatter_comparison.pdf",
                           n_cols=min(3, len(panels)))


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    p = argparse.ArgumentParser(description="RegPro prostate volume (C2).")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--output-dir", required=True)
    _regpro_default = os.environ.get("REGPRO_ZIP")
    p.add_argument("--regpro-zip", default=_regpro_default,
                   required=(_regpro_default is None),
                   help="Path to RegPro raw zip (default: $REGPRO_ZIP env var; required if unset).")
    args = p.parse_args()

    out_dir = pathlib.Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for s in args.runs:
        if "=" not in s:
            raise ValueError(f"--runs entry must be `label=path`, got: {s!r}")
        label, path = s.split("=", 1)
        runs.append((label.strip(), pathlib.Path(path.strip())))

    results = [_load_run(rd, lbl, args.regpro_zip) for lbl, rd in runs]
    summaries = {r.model_label: _summarize(r) for r in results}
    for lbl, s in summaries.items():
        if s["n"]:
            logger.info(
                f"[{lbl}] n={s['n']} MAE={s['mae_ml']:.2f}±{s['std_ml']:.2f}mL "
                f"r={s['pearson_r']:.3f} bias={s['bias_ml']:+.2f}mL"
            )

    with (out_dir / "analysis_report.json").open("w") as f:
        json.dump({
            "analysis": "C2",
            "title": "RegPro Prostate Volume",
            "spec_section": "C2",
            "inputs": {label: str(rd) for label, rd in runs},
            "models": summaries,
            "verification": record_io.declare("per_sample.csv", [
                record_io.check(["models", "$model", "mae_ml"], "abs_err_ml__{model}"),
                record_io.check(["models", "$model", "std_ml"], "abs_err_ml__{model}", agg="std"),
                record_io.check(["models", "$model", "n"], "abs_err_ml__{model}", agg="count"),
            ]),
        }, f, indent=2)

    _write_per_sample_csv(out_dir, results)
    _render_table(out_dir, summaries)
    _render_plots(out_dir, results)
    logger.info(f"C2 analysis complete. Outputs at: {out_dir}")


if __name__ == "__main__":
    main()
