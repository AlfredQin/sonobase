"""Side-by-side qualitative failure-catalog figures (B2, T3.2).

Layout per sample: 5 columns left to right
    [Input | GT | SAM2 (no-ft) | MedSAM2 | SonoBase]
each annotated with its IoU. The image strip is small (~3 inches per cell,
0.5 inch caption row); each catalog page can show 6–10 samples.
"""

from __future__ import annotations

import pathlib
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from nemo_cv.components.analysis.plots.style import (
    apply_nm_defaults,
    color_for,
    display_name,
)


def _to_rgb(arr: np.ndarray) -> np.ndarray:
    """Coerce any image / mask to a uint8 RGB array."""
    a = np.asarray(arr)
    if a.ndim == 2:
        a = (a > 0).astype(np.uint8) * 255 if a.max() <= 1 else a
        return np.stack([a] * 3, axis=-1).astype(np.uint8)
    if a.ndim == 3 and a.shape[2] == 4:
        return a[..., :3].astype(np.uint8)
    return a.astype(np.uint8)


def _blend(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int],
           alpha: float = 0.45) -> np.ndarray:
    out = _to_rgb(image).copy()
    m = (np.asarray(mask) > 0)
    if not m.any():
        return out
    for c in range(3):
        out[m, c] = (out[m, c] * (1 - alpha) + color[c] * alpha).astype(np.uint8)
    return out


def failure_catalog_figure(
    samples: List[Dict],   # {"image": np, "gt": np, "preds": {model: np}, "ious": {model: float}, "title": str}
    model_order: Sequence[str],
    output_path,
    *,
    title: Optional[str] = None,
    cell_size_in: float = 2.5,
) -> None:
    """Render one page of the failure catalog: rows = samples, cols = views.

    `samples`: list of dicts with keys
        image (HxWx3 or HxW), gt, preds (dict[label, mask]),
        ious (dict[label, float]), title (str).
    `model_order`: model labels to show in the prediction columns
        (typically ["sam2_no_ft", "medsam2", "sonobase"]).
    """
    apply_nm_defaults()
    if not samples:
        raise ValueError("failure_catalog_figure: no samples provided")

    n_cols = 2 + len(model_order)   # input + GT + 1 per model
    n_rows = len(samples)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(cell_size_in * n_cols, cell_size_in * n_rows),
        squeeze=False,
    )

    if title is not None:
        fig.suptitle(title, y=1.0, fontsize=10)

    for r, s in enumerate(samples):
        image_rgb = _to_rgb(s["image"])
        gt_overlay = _blend(image_rgb, s["gt"], (0, 220, 0), alpha=0.4)

        # Column 0: input image
        axes[r][0].imshow(image_rgb)
        axes[r][0].set_xticks([]); axes[r][0].set_yticks([])
        if r == 0:
            axes[r][0].set_title("Input", fontsize=8)
        axes[r][0].set_ylabel(s.get("title", ""), fontsize=7)

        # Column 1: GT overlay
        axes[r][1].imshow(gt_overlay)
        axes[r][1].set_xticks([]); axes[r][1].set_yticks([])
        if r == 0:
            axes[r][1].set_title("GT", fontsize=8)

        # Per-model prediction overlays
        for c, label in enumerate(model_order, start=2):
            pred = s.get("preds", {}).get(label)
            iou = s.get("ious", {}).get(label)
            if pred is None:
                axes[r][c].imshow(image_rgb)
            else:
                axes[r][c].imshow(_blend(image_rgb, pred, (255, 140, 0), alpha=0.45))
            axes[r][c].set_xticks([]); axes[r][c].set_yticks([])
            if r == 0:
                axes[r][c].set_title(display_name(label), fontsize=8,
                                     color=color_for(label))
            cap = "—" if iou is None else f"IoU={iou:.2f}"
            axes[r][c].text(
                0.02, 0.98, cap, transform=axes[r][c].transAxes,
                fontsize=6, color="white", verticalalignment="top",
                bbox=dict(facecolor="black", alpha=0.6, pad=1.5, edgecolor="none"),
            )

    fig.tight_layout()
    out = pathlib.Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)


def load_image_or_mask(path: pathlib.Path) -> np.ndarray:
    """Helper: load any PNG/JPG into a numpy array (RGB or single-channel)."""
    img = Image.open(path)
    if img.mode in ("L", "1"):
        return np.array(img.convert("L"))
    return np.array(img.convert("RGB"))
