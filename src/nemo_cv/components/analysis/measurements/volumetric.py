"""Voxel-counting volumes for 3-D segmentation masks (C2 — prostate volume).

Pure NumPy. The C2 analysis loads per-slice 2-D masks (one PNG per slice
per object) from Stage-1 outputs and uses these helpers to convert
voxel counts into mL.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def voxel_volume_mm3(mask_stack: Iterable[np.ndarray], voxel_size_mm3: float) -> float:
    """Return the total volume (mm³) of a stack of binary masks.

    Args:
        mask_stack: Iterable of 2-D numpy arrays (any non-zero pixel is fg).
        voxel_size_mm3: The volume of a single voxel in mm³ (= product of
            the three spacings, sx × sy × sz, from the NIfTI header).

    Returns:
        Total volume in mm³. Returns 0.0 if the stack is empty.
    """
    n_voxels = 0
    for m in mask_stack:
        n_voxels += int(np.count_nonzero(np.asarray(m) > 0))
    return float(n_voxels) * float(voxel_size_mm3)


def voxel_volume_mL(mask_stack: Iterable[np.ndarray], voxel_size_mm3: float) -> float:
    """Return total volume in mL (= cm³ = mm³ / 1000)."""
    return voxel_volume_mm3(mask_stack, voxel_size_mm3) / 1000.0
