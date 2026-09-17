"""Render a metrics dictionary as a publication-grade table figure (PDF + PNG)."""

from __future__ import annotations

from typing import Iterable, List

import matplotlib.pyplot as plt

from nemo_cv.components.analysis.plots.style import apply_nm_defaults


def summary_table_figure(
    rows: List[List[str]],
    output_path,
    col_widths: Iterable[float] | None = None,
    title: str | None = None,
    figsize: tuple = (8.5, None),
) -> None:
    """Render `rows` (incl. header as `rows[0]`) as a PDF + PNG table.

    Args:
        rows: list of rows; each row is a list of cell strings. The first
            row is treated as the header (rendered bold).
        output_path: PDF destination. PNG is written alongside.
        col_widths: relative column widths (sum will be normalized).
        title: optional title above the table.
        figsize: (width, height) in inches; height auto-computed from row
            count if the second element is None.
    """
    apply_nm_defaults()
    n_rows = len(rows)
    if figsize[1] is None:
        height = 0.4 * n_rows + (0.6 if title else 0.0)
        figsize = (figsize[0], height)

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    if title is not None:
        ax.set_title(title, pad=10)

    # Guard: matplotlib's ax.table does len(cellText[0]) and crashes on an
    # empty body; render a single placeholder row when there are no data rows.
    body = rows[1:] if len(rows) > 1 else [["(none)"] + [""] * (len(rows[0]) - 1)]
    table = ax.table(
        cellText=body,
        colLabels=rows[0],
        loc="center",
        cellLoc="center",
        colWidths=list(col_widths) if col_widths is not None else None,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.4)

    # Bold header
    n_cols = len(rows[0])
    for c in range(n_cols):
        cell = table[(0, c)]
        cell.set_text_props(weight="bold")
        cell.set_facecolor("#E8E8E8")

    fig.tight_layout()
    import pathlib as _p
    out = _p.Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
