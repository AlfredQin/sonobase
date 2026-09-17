"""Derive prompts (point / box / correction-click sequences) from GT masks.

These functions are pure (no model dependency), deterministic given a seed,
and produce the exact prompt formats SAM2's `add_new_points_or_box` /
`SAM2ImagePredictor.predict` expect.

The "centre click" implementation uses the RITM-style sampling from the
SAM2 codebase: pick the point in the error region that is farthest from
its boundary (via a 2-D distance transform). See
`nemo_cv.components.models.sam2.modeling.sam2_utils.sample_one_point_from_error_center`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch

from nemo_cv.components.models.sam2.modeling.sam2_utils import (
    sample_one_point_from_error_center,
)


# ---------------------------------------------------------------------------
# Records (what the recipe writes into per-sample meta.json)
# ---------------------------------------------------------------------------


@dataclass
class PromptRecord:
    """One initial prompt + its correction sequence for a single (frame, obj)."""

    frame_idx: int
    obj_id: int
    type: str                                # "point" | "box"
    points: List[List[float]] = field(default_factory=list)   # [[x, y], ...]
    labels: List[int] = field(default_factory=list)           # [1=positive, 0=negative]
    box: Optional[List[float]] = None                          # [x0, y0, x1, y1]

    def to_dict(self) -> dict:
        d = {
            "frame_idx": self.frame_idx,
            "obj_id": self.obj_id,
            "type": self.type,
        }
        if self.points:
            d["points"] = [[float(x), float(y)] for x, y in self.points]
            d["labels"] = [int(l) for l in self.labels]
        if self.box is not None:
            d["box"] = [float(v) for v in self.box]
        return d


# ---------------------------------------------------------------------------
# Geometric helpers
# ---------------------------------------------------------------------------


def mask_to_box(mask: np.ndarray, padding: int = 0) -> np.ndarray:
    """Tight bounding box around a binary mask, optionally padded.

    Args:
        mask: 2-D `np.ndarray`, foreground != 0.
        padding: pixels to add on each side (clamped to image bounds).

    Returns:
        `np.array([x0, y0, x1, y1], dtype=float32)` in image coordinates,
        with `x` indexing columns and `y` indexing rows.

    Raises:
        ValueError: if `mask` has no foreground pixels.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise ValueError("mask_to_box: input mask has no foreground pixels.")
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())

    if padding > 0:
        H, W = mask.shape
        x0 = max(0, x0 - padding)
        y0 = max(0, y0 - padding)
        x1 = min(W - 1, x1 + padding)
        y1 = min(H - 1, y1 + padding)

    return np.array([x0, y0, x1, y1], dtype=np.float32)


def jitter_box(
    box: np.ndarray,
    jitter_pct: float,
    image_shape: Tuple[int, int],
    rng: np.random.Generator,
    max_offset_px: float = 20.0,
) -> np.ndarray:
    """Perturb each box corner to simulate an imprecise human-drawn box.

    Mirrors the noise model SAM2's own training/eval sampler uses
    (`sam2_utils.sample_box_points`, noise=0.1 / noise_bound=20): each of the
    four coordinates gets an independent uniform offset in `[-d, +d]`, where
    `d = min(side * jitter_pct, max_offset_px)` — width for the x coordinates,
    height for the y. The result is clamped to the image.

    Unlike that sampler we also guarantee a non-degenerate box: SAM2 clamps to
    the image bounds only, which can collapse a box whose side is a couple of
    pixels long, and `SAM2ImagePredictor.predict` cannot consume a zero-area box.

    Args:
        box: `[x0, y0, x1, y1]` as returned by `mask_to_box`.
        jitter_pct: offset bound as a fraction of the box side. `0` returns
            the box unchanged **without drawing from `rng`**, so enabling
            jitter never shifts the random stream of an unjittered run.
        image_shape: `(H, W)` of the image the box lives in.
        rng: seeded generator — required, so a jittered run is reproducible.
        max_offset_px: absolute cap on the offset, in pixels.

    Returns:
        `np.array([x0, y0, x1, y1], dtype=float32)`.
    """
    if jitter_pct <= 0:
        return np.asarray(box, dtype=np.float32)

    H, W = image_shape
    x0, y0, x1, y1 = (float(v) for v in box)
    dx = min((x1 - x0) * jitter_pct, max_offset_px)
    dy = min((y1 - y0) * jitter_pct, max_offset_px)

    offsets = rng.uniform(-1.0, 1.0, size=4) * np.array([dx, dy, dx, dy])
    x0, y0, x1, y1 = np.array([x0, y0, x1, y1]) + offsets

    x0, x1 = sorted((float(np.clip(x0, 0, W - 1)), float(np.clip(x1, 0, W - 1))))
    y0, y1 = sorted((float(np.clip(y0, 0, H - 1)), float(np.clip(y1, 0, H - 1))))
    if x1 - x0 < 1.0:
        x0, x1 = max(0.0, min(x0, W - 2.0)), min(W - 1.0, max(x1, x0 + 1.0))
    if y1 - y0 < 1.0:
        y0, y1 = max(0.0, min(y0, H - 2.0)), min(H - 1.0, max(y1, y0 + 1.0))

    return np.array([x0, y0, x1, y1], dtype=np.float32)


# ---------------------------------------------------------------------------
# Initial prompts
# ---------------------------------------------------------------------------


def initial_point_from_gt(
    gt_mask: np.ndarray,
    method: str = "center",
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, int]:
    """Sample the initial positive click for a GT mask.

    Two samplers, matching the two `pt_sampling_for_eval` modes SAM2 itself
    exposes (`sam2_train.py`, `get_next_point`):

    ``center`` (default, every published run)
        RITM-style: the foreground pixel farthest from the mask boundary.
        Deterministic — one call of `sample_one_point_from_error_center`
        against an empty prediction.

    ``uniform``
        One pixel drawn uniformly from the GT foreground. This is what SAM2
        *trains* with (`method="uniform" if self.training`), so it is the
        train/eval-matched choice, and it is a better model of a clinician
        clicking somewhere inside a structure than a distance-transform
        maximum is.

        We draw with `rng.integers` over the foreground index list rather
        than calling SAM2's `sample_random_points_from_errors`. That function
        allocates a `[B, 1, H, W, 2]` uniform tensor on the GPU and takes a
        region-masked argmax; masking uniform noise to a region and taking
        the argmax *is* a uniform draw from that region, so the two are
        distributionally identical. Ours keeps prompt randomness on one
        seeded numpy stream (no CUDA RNG state, no dependence on image size)
        and costs O(|fg|) instead of O(H*W). `test_prompts.py` asserts the
        equivalence empirically rather than taking it on trust.

    Returns:
        `(point, label)` where `point` is `np.array([x, y], dtype=float32)`
        and `label` is `1` (positive click — always positive on the first
        prompt because the empty prediction makes every fg pixel a false-negative).
    """
    if gt_mask.sum() == 0:
        raise ValueError("initial_point_from_gt: input GT mask is empty.")

    if method == "uniform":
        if rng is None:
            raise ValueError(
                "point_sampling='uniform' requires a seeded `rng` so the run "
                "is reproducible."
            )
        ys, xs = np.nonzero(gt_mask)
        i = int(rng.integers(len(xs)))
        return np.array([xs[i], ys[i]], dtype=np.float32), 1

    if method != "center":
        raise ValueError(
            f"unknown point_sampling {method!r}; expected 'center' or 'uniform'."
        )

    gt_bool = torch.from_numpy(gt_mask.astype(bool))
    empty_pred = torch.zeros_like(gt_bool)
    point_t, label_t = sample_one_point_from_error_center(
        gt_bool[None, None],          # [B=1, 1, H, W]
        empty_pred[None, None],
    )
    point = point_t.squeeze().cpu().numpy().astype(np.float32)   # [x, y]
    label = int(label_t.item())
    return point, label


def initial_box_from_gt(gt_mask: np.ndarray, padding: int = 0) -> np.ndarray:
    """Bounding-box prompt derived from a GT mask. Thin wrapper over `mask_to_box`."""
    return mask_to_box(gt_mask, padding=padding)


# ---------------------------------------------------------------------------
# Correction clicks (used when num_correction_clicks > 0)
# ---------------------------------------------------------------------------


def next_correction_point(
    gt_mask: np.ndarray, current_pred: np.ndarray
) -> Tuple[np.ndarray, int]:
    """Sample the next correction click given the current prediction.

    Picks the centre of the largest error region (FP or FN). Returns
    `(point, label)` with `label=1` when the click sits in a false-negative
    region (need to add foreground there) or `label=0` for a false-positive
    region (need to remove foreground).
    """
    gt_bool = torch.from_numpy(gt_mask.astype(bool))
    pred_bool = torch.from_numpy(current_pred.astype(bool))
    point_t, label_t = sample_one_point_from_error_center(
        gt_bool[None, None],
        pred_bool[None, None],
    )
    point = point_t.squeeze().cpu().numpy().astype(np.float32)
    label = int(label_t.item())
    return point, label
