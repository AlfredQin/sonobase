"""Render the few-shot publication figures.

Produces TWO files (each PDF + PNG, 300 DPI, Arial 8 pt — journal figure
formatting via `plots/style.py`):

  1. `fig_fewshot_segmentation.{pdf,png}` — 3 cols × 2 rows = 6 panels.
        rows: prompt ∈ {point, box}
        cols: dataset ∈ {ACOUSLIC, DDTI, FUGC}
        each panel: mIoU vs N, log-scale x, ±1 std bands across 3 seeds,
                    one line per model (sonobase / medsam2 / sam2_no_ft).

  2. `fig_fewshot_acouslic_clinical.{pdf,png}` — 1 × 2 panels.
        cols: prompt ∈ {point, box}
        each panel: AC MAE (mm) vs N, lower is better.
        + horizontal dashed line at the GT-fit floor (configurable via
          `--gt-fit-floor-mm`, default 7.39 mm — the ellipse-fit floor on the
          ACOUSLIC ground-truth masks, v1.1 ground truth at 0.28 mm/px).
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import pathlib
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from nemo_cv.components.analysis.plots.style import (
    apply_nm_defaults, color_for, display_name, linestyle_for, marker_for,
)

logger = logging.getLogger(__name__)


DATASETS = ("ACOUSLIC", "DDTI", "FUGC")
PROMPTS = ("point", "box")
MODELS = ("sonobase", "medsam2", "sam2_no_ft")


def _read_summary(path: pathlib.Path) -> List[Dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _curve(rows: List[Dict[str, str]], dataset: str, prompt: str, model: str,
           value_col: str, std_col: str
           ) -> Tuple[List[int], List[float], List[float]]:
    """Sorted (N_list, mean_list, std_list) for one (dataset, prompt, model)."""
    by_n: Dict[int, Tuple[float, float]] = {}
    for r in rows:
        if r.get("dataset") != dataset or r.get("prompt") != prompt or r.get("model") != model:
            continue
        try:
            N = int(r["N"])
            v = float(r[value_col]) if r[value_col] else float("nan")
            s = float(r[std_col]) if r[std_col] else 0.0
        except (KeyError, ValueError):
            continue
        if math.isfinite(v):
            by_n[N] = (v, s)
    if not by_n:
        return [], [], []
    Ns = sorted(by_n)
    means = [by_n[n][0] for n in Ns]
    stds = [by_n[n][1] for n in Ns]
    return Ns, means, stds


def _plot_panel(ax: plt.Axes, rows: List[Dict[str, str]], dataset: str, prompt: str,
                value_col: str, std_col: str, ylabel: str, scale_pct: bool = True,
                horizontal_line: Optional[float] = None) -> None:
    apply_nm_defaults()
    plotted_any = False
    for model in MODELS:
        Ns, means, stds = _curve(rows, dataset, prompt, model, value_col, std_col)
        if not Ns:
            continue
        # mIoU/Dice in the CSV are stored as fractions (0..1); convert to %
        # for display (y-axis 0–100).
        if scale_pct:
            means_disp = [m * 100 for m in means]
            stds_disp = [s * 100 for s in stds]
        else:
            means_disp, stds_disp = means, stds

        # x=0 doesn't fit on log scale; offset to 0.5 for plotting.
        xs_plot = [n if n > 0 else 0.5 for n in Ns]

        ax.plot(xs_plot, means_disp,
                color=color_for(model), marker=marker_for(model),
                linestyle=linestyle_for(model),
                linewidth=1.4, markersize=5, label=display_name(model))
        ax.fill_between(
            xs_plot,
            [m - s for m, s in zip(means_disp, stds_disp)],
            [m + s for m, s in zip(means_disp, stds_disp)],
            color=color_for(model), alpha=0.20, linewidth=0,
        )
        plotted_any = True

    if horizontal_line is not None:
        ax.axhline(horizontal_line, linestyle="--", color="black",
                   linewidth=0.6, label=f"Floor: {horizontal_line:.2f}")

    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 5, 10, 20, 30])
    ax.set_xticklabels(["0", "1", "2", "5", "10", "20", "30"])
    ax.set_xlabel("N (training samples per model)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3, linewidth=0.4)
    if plotted_any:
        ax.legend(loc="best", fontsize=6, framealpha=0.85)


def render_segmentation_figure(combined_rows: List[Dict[str, str]],
                               output_path: pathlib.Path) -> None:
    apply_nm_defaults()
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.5), squeeze=False)
    for r_idx, prompt in enumerate(PROMPTS):
        for c_idx, dataset in enumerate(DATASETS):
            ax = axes[r_idx][c_idx]
            _plot_panel(ax, combined_rows, dataset, prompt,
                        value_col="mIoU_mean", std_col="mIoU_std",
                        ylabel="mIoU (%)", scale_pct=True)
            if r_idx == 0:
                ax.set_title(dataset)
            if c_idx == 0:
                ax.set_ylabel(("Point prompt — " if r_idx == 0 else "Box prompt — ") + "mIoU (%)")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path); fig.savefig(output_path.with_suffix(".png"))
    plt.close(fig)
    logger.info(f"Wrote {output_path}")


def render_acouslic_clinical_figure(combined_rows: List[Dict[str, str]],
                                    output_path: pathlib.Path,
                                    gt_fit_floor_mm: Optional[float]) -> None:
    apply_nm_defaults()
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.0), squeeze=False)
    for c_idx, prompt in enumerate(PROMPTS):
        ax = axes[0][c_idx]
        _plot_panel(ax, combined_rows, "ACOUSLIC", prompt,
                    value_col="AC_MAE_mean", std_col="AC_MAE_std",
                    ylabel="AC MAE (mm)  [lower is better]",
                    scale_pct=False, horizontal_line=gt_fit_floor_mm)
        ax.set_title(f"ACOUSLIC AC — {prompt} prompt")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path); fig.savefig(output_path.with_suffix(".png"))
    plt.close(fig)
    logger.info(f"Wrote {output_path}")


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    p = argparse.ArgumentParser(description="Render few-shot publication figures.")
    p.add_argument("--results-dir", required=True,
                   help="Directory holding ACOUSLIC_summary.csv / DDTI_summary.csv / FUGC_summary.csv "
                        "(typically the same one apply_fdr_correction.py wrote into).")
    p.add_argument("--output-dir", default=None,
                   help="Defaults to --results-dir.")
    p.add_argument("--gt-fit-floor-mm", type=float, default=7.39,
                   help="Horizontal floor for the ACOUSLIC clinical figure (default = 7.39 mm, "
                        "the ellipse-fit floor on the ground-truth masks).")
    args = p.parse_args()

    results_dir = pathlib.Path(args.results_dir).expanduser().resolve()
    output_dir = pathlib.Path(args.output_dir).expanduser().resolve() if args.output_dir else results_dir

    combined = results_dir / "combined_summary.csv"
    if not combined.is_file():
        logger.error(
            f"combined_summary.csv not found at {combined}. Run "
            "apply_fdr_correction.py first (it writes the combined file)."
        )
        return
    rows = _read_summary(combined)

    render_segmentation_figure(rows, output_dir / "fig_fewshot_segmentation.pdf")
    render_acouslic_clinical_figure(
        rows, output_dir / "fig_fewshot_acouslic_clinical.pdf",
        gt_fit_floor_mm=args.gt_fit_floor_mm,
    )
    logger.info(f"Figures written to {output_dir}")


if __name__ == "__main__":
    main()
