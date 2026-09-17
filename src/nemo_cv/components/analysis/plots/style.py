"""Centralized matplotlib defaults for the analysis suite.

Journal figure requirements:

  * PDF format (vector); PNG only as preview
  * DPI ≥ 300
  * Arial, ≥ 8 pt everywhere
  * SonoBase = green #2ca02c, MedSAM2 = red #d62728, SAM2 (no-ft) = gray #7f7f7f

Importing this module applies the global rcParams. Per-figure helpers also
re-apply them so a one-off plot doesn't depend on import order.
"""

from __future__ import annotations

import matplotlib

# Per-model styling — used everywhere
MODEL_COLORS = {
    "sonobase": "#2ca02c",     # green
    "medsam2": "#d62728",      # red
    "sam2_no_ft": "#7f7f7f",   # gray
}
MODEL_MARKERS = {
    "sonobase": "o",
    "medsam2": "s",
    "sam2_no_ft": "^",
}
MODEL_LINESTYLES = {
    "sonobase": "-",
    "medsam2": "-",
    "sam2_no_ft": "--",
}
MODEL_DISPLAY = {
    "sonobase": "SonoBase",
    "medsam2": "MedSAM2",
    "sam2_no_ft": "SAM2 (no-ft)",
}


def apply_nm_defaults() -> None:
    """Set matplotlib rcParams to satisfy the journal figure requirements."""
    rc = matplotlib.rcParams
    # Try Arial first; fall back to Liberation Sans (Linux default Arial-alike).
    rc["font.family"] = ["Arial", "Liberation Sans", "DejaVu Sans"]
    rc["font.size"] = 8
    rc["axes.linewidth"] = 0.5
    rc["axes.titlesize"] = 9
    rc["axes.labelsize"] = 8
    rc["xtick.labelsize"] = 7
    rc["ytick.labelsize"] = 7
    rc["legend.fontsize"] = 7
    rc["figure.titlesize"] = 10
    rc["savefig.dpi"] = 300
    rc["savefig.bbox"] = "tight"
    rc["pdf.fonttype"] = 42         # embed as TrueType (editable in vector tools)
    rc["ps.fonttype"] = 42


# Apply at import time so any module that imports plots/style gets it.
apply_nm_defaults()


def color_for(model_label: str) -> str:
    return MODEL_COLORS.get(model_label, "#444444")


def marker_for(model_label: str) -> str:
    return MODEL_MARKERS.get(model_label, "o")


def linestyle_for(model_label: str) -> str:
    return MODEL_LINESTYLES.get(model_label, "-")


def display_name(model_label: str) -> str:
    return MODEL_DISPLAY.get(model_label, model_label)
