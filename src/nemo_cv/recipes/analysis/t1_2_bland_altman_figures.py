"""Analysis T1.2 — Composite Bland-Altman figures.

Stage-2 analysis. Renders publication-grade Bland-Altman figures from the
per-sample CSVs of A1 / A2 / A4 / C2:

  * **Supplementary composite** (4 rows × 3 cols): rows = (EF, HC, AC,
    Volume); cols = (SonoBase, MedSAM2, SAM2 no-ft). Y-axis limits are
    shared within each row so the columns are visually comparable.
  * **Main paper figure** (1 × 2): SonoBase EF Bland-Altman for point and
    box prompts. Pass two A1 per_patient.csv paths via
    `--main-sonobase-point` and `--main-sonobase-box`.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import pathlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from nemo_cv.components.analysis.plots.bland_altman import (
    bland_altman_panel, bland_altman_composite,
)
from nemo_cv.components.analysis.plots.style import (
    apply_nm_defaults, color_for, display_name,
)

logger = logging.getLogger(__name__)


@dataclass
class _Spec:
    label: str        # row label for the supplementary composite
    csv_path: pathlib.Path
    pred_col_prefix: str
    gt_col: str
    ylabel: str       # e.g. "EF difference (%)", "HC difference (mm)"


# Same registry as T1.1 — same CSV column conventions for A1 / A2 / A4 / C2.
_ANALYSIS_SPECS: Dict[str, Dict] = {
    "A1": {"label": "EF (%)", "pred": "pred_ef_biplane__", "gt": "gt_ef",
           "ylabel": "EF difference (%)"},
    "A2": {"label": "HC (mm)", "pred": "pred_hc_mm__", "gt": "gt_hc_mm",
           "ylabel": "HC difference (mm)"},
    "A4": {"label": "AC (mm)", "pred": "pred_ac_mm__", "gt": "gt_ac_mm",
           "ylabel": "AC difference (mm)"},
    "C2": {"label": "Volume (mL)", "pred": "pred_vol_ml__", "gt": "gt_vol_ml",
           "ylabel": "Volume difference (mL)"},
}


def _read_per_sample(csv_path: pathlib.Path) -> List[Dict[str, str]]:
    with csv_path.open() as f:
        return list(csv.DictReader(f))


def _arrays_for_model(rows: List[Dict[str, str]], pred_col: str, gt_col: str
                      ) -> Tuple[np.ndarray, np.ndarray]:
    g, p = [], []
    for r in rows:
        try:
            gt = float(r[gt_col]); pred = float(r[pred_col])
        except (KeyError, ValueError):
            continue
        if not math.isfinite(pred):
            continue
        g.append(gt); p.append(pred)
    return np.asarray(g), np.asarray(p)


def _render_supplementary_composite(
    inputs: Dict[str, _Spec],
    models_in_order: List[str],
    output_path: pathlib.Path,
    *,
    cell_size_in: float = 4.0,
) -> None:
    """4 × N grid (one row per analysis, one column per model)."""
    apply_nm_defaults()
    n_rows = len(inputs)
    n_cols = len(models_in_order)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(cell_size_in * n_cols, cell_size_in * n_rows * 0.9),
        squeeze=False,
    )

    for r_idx, (aid, spec) in enumerate(inputs.items()):
        rows = _read_per_sample(spec.csv_path)

        # Compute the row's max range across all models for shared y-axis
        per_model_arrays: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        all_diffs = []
        for m in models_in_order:
            g, p = _arrays_for_model(rows, f"{spec.pred_col_prefix}{m}", spec.gt_col)
            if g.size:
                per_model_arrays[m] = (g, p)
                all_diffs.extend((p - g).tolist())

        # Render each column (model)
        for c_idx, m in enumerate(models_in_order):
            ax = axes[r_idx][c_idx]
            ga, pa = per_model_arrays.get(m, (np.array([]), np.array([])))
            if ga.size == 0:
                ax.axis("off"); continue
            bland_altman_panel(
                ax, ga, pa,
                title=f"{display_name(m)} — {spec.label}" if r_idx == 0 else display_name(m),
                ylabel=spec.ylabel if c_idx == 0 else "",
                xlabel="Mean of GT and predicted",
                point_color=color_for(m),
            )

        # Apply shared y-limits per row
        if all_diffs:
            lo = float(np.min(all_diffs)); hi = float(np.max(all_diffs))
            pad = 0.05 * (hi - lo) if hi > lo else 1.0
            for c_idx in range(n_cols):
                ax = axes[r_idx][c_idx]
                if ax.has_data():
                    ax.set_ylim(lo - pad, hi + pad)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix(".png"))
    plt.close(fig)
    logger.info(f"Wrote {output_path}")


def _render_main_sonobase(
    point_csv: Optional[pathlib.Path],
    box_csv: Optional[pathlib.Path],
    output_path: pathlib.Path,
) -> None:
    """1 × 2 main paper figure: SonoBase EF BA at point vs box prompt."""
    apply_nm_defaults()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), squeeze=False)
    titles = ["SonoBase EF — point prompt", "SonoBase EF — box prompt"]
    for col, (csv_path, title) in enumerate(zip([point_csv, box_csv], titles)):
        ax = axes[0][col]
        if csv_path is None or not csv_path.is_file():
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.axis("off"); continue
        rows = _read_per_sample(csv_path)
        g, p = _arrays_for_model(rows, "pred_ef_biplane__sonobase", "gt_ef")
        if g.size == 0:
            ax.text(0.5, 0.5, "No SonoBase data", ha="center", va="center")
            ax.axis("off"); continue
        bland_altman_panel(ax, g, p, title=title, ylabel="EF difference (%)",
                           xlabel="Mean of GT and predicted",
                           point_color=color_for("sonobase"))

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix(".png"))
    plt.close(fig)
    logger.info(f"Wrote {output_path}")


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    p = argparse.ArgumentParser(description="Composite Bland-Altman figures (T1.2).")
    p.add_argument("--inputs", nargs="+", required=True,
                   help="`<analysis_id>=<per_sample.csv>` per analysis (e.g. A1=path/per_patient.csv).")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--models", nargs="+",
                   default=["sonobase", "medsam2", "sam2_no_ft"],
                   help="Column order in the supplementary composite.")
    p.add_argument("--main-sonobase-point", default=None,
                   help="Path to A1 per_patient.csv from a sonobase × point Stage-1 run "
                        "(used for the main paper figure left panel).")
    p.add_argument("--main-sonobase-box", default=None,
                   help="Path to A1 per_patient.csv from a sonobase × box Stage-1 run "
                        "(used for the main paper figure right panel).")
    args = p.parse_args()

    out_dir = pathlib.Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs: Dict[str, _Spec] = {}
    for s in args.inputs:
        aid, path = s.split("=", 1)
        aid = aid.strip().upper()
        meta = _ANALYSIS_SPECS.get(aid)
        if meta is None:
            raise ValueError(f"Unknown analysis id {aid!r}; expected {list(_ANALYSIS_SPECS)}")
        inputs[aid] = _Spec(
            label=meta["label"], csv_path=pathlib.Path(path.strip()),
            pred_col_prefix=meta["pred"], gt_col=meta["gt"], ylabel=meta["ylabel"],
        )

    _render_supplementary_composite(
        inputs, args.models, out_dir / "bland_altman_supplementary.pdf",
    )

    point_csv = pathlib.Path(args.main_sonobase_point) if args.main_sonobase_point else None
    box_csv = pathlib.Path(args.main_sonobase_box) if args.main_sonobase_box else None
    if point_csv or box_csv:
        _render_main_sonobase(
            point_csv, box_csv,
            out_dir / "bland_altman_main_sonobase.pdf",
        )

    logger.info(f"T1.2 analysis complete. Outputs at: {out_dir}")


if __name__ == "__main__":
    main()
