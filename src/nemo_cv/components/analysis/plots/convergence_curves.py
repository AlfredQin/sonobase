"""Convergence curves for A3 click-efficiency.

Plot mean mIoU as a function of correction-click count for each model,
optionally faceted by (prompt_init × dataset_tier). Saves PDF + PNG.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np

from nemo_cv.components.analysis.plots.style import (
    apply_nm_defaults,
    color_for,
    display_name,
    linestyle_for,
    marker_for,
)


def convergence_curves(
    iterations: Sequence[int],
    model_curves: Dict[str, Sequence[float]],
    output_path,
    *,
    title: str = "Click-efficiency convergence",
    xlabel: str = "Correction clicks",
    ylabel: str = "Mean IoU (%)",
    xscale: str = "linear",
    figsize: tuple = (5.0, 4.0),
) -> None:
    """Render a single convergence curves panel (one line per model).

    Args:
        iterations: x-axis values, e.g. [0, 1, 3, 5, 7].
        model_curves: ``{model_label: [mean_iou_at_each_iter, ...]}``. Each
            list must be the same length as ``iterations``.
        output_path: PDF path. PNG also written.
        xscale: 'linear' or 'log' (use 'log' if iterations has 0 only when
            you've replaced 0→0.5 to allow log scale; otherwise 'linear').
    """
    apply_nm_defaults()
    fig, ax = plt.subplots(figsize=figsize)
    xs = np.asarray(iterations, dtype=float)

    for label, curve in model_curves.items():
        ys = np.asarray(curve, dtype=float)
        ax.plot(
            xs, ys,
            color=color_for(label),
            marker=marker_for(label),
            linestyle=linestyle_for(label),
            linewidth=1.4,
            markersize=5,
            label=display_name(label),
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xscale == "log":
        ax.set_xscale("log")
    ax.grid(True, alpha=0.3, linewidth=0.4)
    ax.legend(loc="lower right", fontsize=7, framealpha=0.85)

    fig.tight_layout()
    import pathlib as _p
    out = _p.Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)


def convergence_curves_grid(
    panels: List[Dict],   # {"title": str, "iterations": [...], "model_curves": {...}}
    output_path,
    *,
    n_cols: int = 2,
    xlabel: str = "Correction clicks",
    ylabel: str = "Mean IoU (%)",
    figsize_per_panel: tuple = (4.5, 3.5),
) -> None:
    """Render a grid of convergence panels (one per facet)."""
    apply_nm_defaults()
    n = len(panels)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
        squeeze=False, sharex=False, sharey=True,
    )
    for idx, p in enumerate(panels):
        r, c = divmod(idx, n_cols)
        ax = axes[r][c]
        xs = np.asarray(p["iterations"], dtype=float)
        for label, curve in p["model_curves"].items():
            ys = np.asarray(curve, dtype=float)
            ax.plot(
                xs, ys, color=color_for(label),
                marker=marker_for(label), linestyle=linestyle_for(label),
                linewidth=1.4, markersize=5, label=display_name(label),
            )
        ax.set_title(p.get("title", ""))
        ax.set_xlabel(xlabel)
        if c == 0:
            ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3, linewidth=0.4)
        ax.legend(loc="lower right", fontsize=6, framealpha=0.85)

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
