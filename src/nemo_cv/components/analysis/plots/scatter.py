"""Scatter plot (predicted vs ground truth) with identity + regression lines.

Used by A1 (EF correlation), A2 (HC), A4 (AC), C2 (prostate volume) to show
how strongly predicted clinical measurements track ground truth on a
per-sample basis.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from nemo_cv.components.analysis.plots.style import apply_nm_defaults
from nemo_cv.components.analysis.stats.tests import pearson_r


def scatter_with_identity(
    ax: plt.Axes,
    gt: np.ndarray,
    pred: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    point_color: str = "steelblue",
    fit_line: bool = True,
) -> float:
    """Scatter `pred` vs `gt` with x=y identity line and optional regression line.

    Returns the Pearson r of the (gt, pred) pair (also annotated on the plot).
    """
    apply_nm_defaults()
    gt = np.asarray(gt, dtype=float).ravel()
    pred = np.asarray(pred, dtype=float).ravel()
    if gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch: gt {gt.shape} vs pred {pred.shape}")

    ax.scatter(gt, pred, s=14, alpha=0.5, color=point_color, edgecolors="none")

    # Identity line: full extent of both arrays
    finite = np.isfinite(gt) & np.isfinite(pred)
    if finite.sum() > 0:
        lo = float(min(gt[finite].min(), pred[finite].min()))
        hi = float(max(gt[finite].max(), pred[finite].max()))
        pad = 0.02 * (hi - lo) if hi > lo else 1.0
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                linestyle=":", color="black", linewidth=0.6, label="y = x")

    # Linear regression line (least-squares)
    r = pearson_r(gt, pred)
    if fit_line and finite.sum() >= 2:
        slope, intercept = np.polyfit(gt[finite], pred[finite], 1)
        xs = np.linspace(gt[finite].min(), gt[finite].max(), 50)
        ax.plot(xs, slope * xs + intercept, color="red", linewidth=1.0,
                label=f"r = {r:.3f}")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", fontsize=6, framealpha=0.85)
    ax.set_aspect("equal", adjustable="datalim")
    return r


def scatter_comparison(
    panels: List[Tuple[str, np.ndarray, np.ndarray, str]],   # (title, gt, pred, color)
    xlabel: str,
    ylabel: str,
    output_path,
    n_cols: int = 3,
    figsize_per_panel: Tuple[float, float] = (4.0, 4.0),
) -> None:
    """Render side-by-side scatters for multiple models.

    Each panel plots one model's predicted-vs-GT scatter; same axes scale
    for visual comparison. Saves PDF + PNG.
    """
    apply_nm_defaults()
    n = len(panels)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
        squeeze=False,
    )

    # Common axis range across panels
    all_vals = []
    for _, gt, pred, _ in panels:
        all_vals.extend(np.asarray(gt).ravel().tolist())
        all_vals.extend(np.asarray(pred).ravel().tolist())
    all_vals = [v for v in all_vals if np.isfinite(v)]
    if all_vals:
        lo = float(np.min(all_vals)); hi = float(np.max(all_vals))
        pad = 0.05 * (hi - lo) if hi > lo else 1.0
        xlim = (lo - pad, hi + pad)
    else:
        xlim = None

    for idx, (title, gt, pred, color) in enumerate(panels):
        r, c = divmod(idx, n_cols)
        ax = axes[r][c]
        scatter_with_identity(ax, gt, pred, title=title, xlabel=xlabel,
                              ylabel=ylabel, point_color=color)
        if xlim is not None:
            ax.set_xlim(*xlim)
            ax.set_ylim(*xlim)

    for idx in range(len(panels), n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].axis("off")

    fig.tight_layout()
    import pathlib as _p
    out = _p.Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
