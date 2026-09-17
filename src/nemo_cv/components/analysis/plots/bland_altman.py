"""Bland-Altman plotting (per-axis panel + multi-panel composite).

The single-panel function is reusable: A1 (EF), A2 (HC), A4 (AC), C2
(prostate volume) all call it with their own arrays. The composite figure
(used by T1.2) just stacks several single panels with shared y-axes.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from nemo_cv.components.analysis.plots.style import apply_nm_defaults
from nemo_cv.components.analysis.stats.tests import bland_altman_stats


def bland_altman_panel(
    ax: plt.Axes,
    gt: np.ndarray,
    pred: np.ndarray,
    title: str,
    ylabel: str,
    xlabel: str = "Mean of GT and predicted",
    point_color: str = "steelblue",
    point_alpha: float = 0.5,
) -> Tuple[float, float, float, float]:
    """Render a single Bland-Altman panel onto an existing matplotlib axis.

    Plots:
      * a scatter of (mean(gt, pred), pred − gt) for each sample
      * a horizontal red bias line (mean of differences)
      * two horizontal grey ±1.96 SD limit-of-agreement lines
      * a thin black zero-line for visual reference
      * a legend with numeric bias and LoA values

    Returns ``(bias, std_diff, loa_lower, loa_upper)`` so callers can also
    feed those into the per-analysis report JSON.
    """
    apply_nm_defaults()

    gt = np.asarray(gt, dtype=float).ravel()
    pred = np.asarray(pred, dtype=float).ravel()
    if gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch: gt {gt.shape} vs pred {pred.shape}")

    diff = pred - gt
    mean = (pred + gt) / 2.0
    stats = bland_altman_stats(gt, pred)

    ax.scatter(mean, diff, s=14, alpha=point_alpha, color=point_color, edgecolors="none")
    ax.axhline(stats.bias, color="red", linewidth=1.2,
               label=f"Bias: {stats.bias:.2f}")
    ax.axhline(stats.loa_upper, color="gray", linestyle="--", linewidth=0.8,
               label=f"+1.96 SD: {stats.loa_upper:.2f}")
    ax.axhline(stats.loa_lower, color="gray", linestyle="--", linewidth=0.8,
               label=f"-1.96 SD: {stats.loa_lower:.2f}")
    ax.axhline(0, color="black", linestyle=":", linewidth=0.4)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right", fontsize=6, framealpha=0.85)

    return stats.bias, stats.std_diff, stats.loa_lower, stats.loa_upper


def bland_altman_composite(
    panels: List[Tuple[str, np.ndarray, np.ndarray]],
    ylabel: str,
    output_path,
    n_cols: int = 3,
    figsize_per_panel: Tuple[float, float] = (4.0, 3.5),
    share_y: bool = True,
) -> None:
    """Render a multi-panel Bland-Altman figure (e.g. one model per column).

    Args:
        panels: list of ``(panel_title, gt_array, pred_array)`` tuples in
            the order they should appear left-to-right, top-to-bottom.
        ylabel: shared y-axis label (e.g. ``"EF difference (%)"``).
        output_path: PDF path. A sibling PNG is also written.
        n_cols: panels per row.
        share_y: if True, all panels use the global min/max y range — makes
            cross-panel visual comparison meaningful.
    """
    apply_nm_defaults()
    n = len(panels)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
        squeeze=False,
    )

    for idx, (title, gt, pred) in enumerate(panels):
        r, c = divmod(idx, n_cols)
        bland_altman_panel(axes[r][c], gt, pred, title=title, ylabel=ylabel)

    # Hide unused axes
    for idx in range(len(panels), n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].axis("off")

    if share_y:
        ylims = [ax.get_ylim() for row in axes for ax in row if ax.has_data()]
        if ylims:
            y_min = min(yl[0] for yl in ylims)
            y_max = max(yl[1] for yl in ylims)
            for row in axes:
                for ax in row:
                    if ax.has_data():
                        ax.set_ylim(y_min, y_max)

    fig.tight_layout()
    import pathlib as _p
    out = _p.Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
