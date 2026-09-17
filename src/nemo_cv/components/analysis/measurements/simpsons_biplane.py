"""Simpson's biplane LV-volume + ejection-fraction measurement (A1).

Implements the disc-summation method recommended by:

    Lang RM, et al. "Recommendations for Cardiac Chamber Quantification by
    Echocardiography in Adults: An Update from the American Society of
    Echocardiography and the European Association of Cardiovascular Imaging."
    J Am Soc Echocardiogr. 2015;28(1):1-39. PMID 25559473.

Pure NumPy + OpenCV; no model dependency. Easy to unit-test.

Pipeline (analysis A1):

  1. Detect the LV long axis on each frame:
     a. Find the LV-endocardium contour.
     b. Apex = topmost contour point (smallest y).
     c. Base midpoint = mean of the two widest-apart contour points in
        the bottom 10 % of the mask.
     d. Long axis = base_midpoint → apex.
  2. Divide the long axis into ``num_discs`` (default 20) equal segments.
  3. For each disc, scan a perpendicular line across the mask and measure
     the chord length at that level → diameter ``d_i``.
  4. Disc volume: ``V_i = π × (d_i / 2)² × h`` where ``h = L / num_discs``.
  5. Sum the disc volumes → LV cavity volume in pixel-volume units. To
     convert to mL, multiply by ``pixel_spacing_mm² × pixel_spacing_mm``
     and divide by 1000 (mL = cm³).
  6. Biplane average: ``EDV = (EDV_A4C + EDV_A2C) / 2``, similarly ESV.
     EF = ``(EDV − ESV) / EDV × 100``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

DEFAULT_NUM_DISCS = 20
BASE_BAND_FRACTION = 0.10   # bottom 10 % of mask used to find the base midpoint


@dataclass
class LongAxis:
    apex: Tuple[float, float]            # (x, y) in pixels
    base_midpoint: Tuple[float, float]
    length_px: float

    @property
    def vector(self) -> np.ndarray:
        ax, ay = self.apex
        bx, by = self.base_midpoint
        return np.array([ax - bx, ay - by], dtype=np.float64)


@dataclass
class SimpsonsResult:
    """Single-view Simpson's volume measurement."""

    volume_px: float                     # raw volume in pixel-volume units
    long_axis: LongAxis
    disc_diameters_px: List[float]
    notes: str = ""

    def volume_mm3(self, pixel_spacing_mm: float) -> float:
        # Pixel diameter d_i (in px) becomes d_i × pixel_spacing_mm (mm).
        # Disc thickness h = (L × pixel_spacing_mm) / num_discs
        # V_total_mm³ = sum_i π × (d_i_mm / 2)² × h_mm
        #             = volume_px × pixel_spacing_mm³
        return float(self.volume_px) * (pixel_spacing_mm ** 3)

    def volume_mL(self, pixel_spacing_mm: float) -> float:
        # 1 mL = 1 cm³ = 1000 mm³
        return self.volume_mm3(pixel_spacing_mm) / 1000.0


# ---------------------------------------------------------------------------
# Long-axis detection
# ---------------------------------------------------------------------------


def _largest_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return None
    return max(contours, key=cv2.contourArea).reshape(-1, 2)


def detect_long_axis(mask: np.ndarray) -> Optional[LongAxis]:
    """Detect the LV long axis from a binary endocardial mask.

    Convention: image y increases downward, so the apex (most superior point
    in the image) has the **smallest** y. The base sits at the bottom of
    the mask (largest y).

    Returns None if the mask is empty or has fewer than 5 contour points.
    """
    contour = _largest_contour(mask)
    if contour is None or len(contour) < 5:
        return None

    xs, ys = contour[:, 0], contour[:, 1]
    apex_idx = int(np.argmin(ys))
    apex = (float(xs[apex_idx]), float(ys[apex_idx]))

    y_min, y_max = float(ys.min()), float(ys.max())
    h = y_max - y_min
    if h < 1.0:
        return None

    # Base band: bottom BASE_BAND_FRACTION of the mask
    base_band_thresh = y_max - BASE_BAND_FRACTION * h
    band_mask = ys >= base_band_thresh
    band_xs = xs[band_mask]
    band_ys = ys[band_mask]
    if band_xs.size < 2:
        return None

    # Two widest-apart points in the band
    band_idx_left = int(np.argmin(band_xs))
    band_idx_right = int(np.argmax(band_xs))
    p_left = (float(band_xs[band_idx_left]), float(band_ys[band_idx_left]))
    p_right = (float(band_xs[band_idx_right]), float(band_ys[band_idx_right]))
    base_midpoint = (
        (p_left[0] + p_right[0]) / 2.0,
        (p_left[1] + p_right[1]) / 2.0,
    )

    length_px = math.hypot(apex[0] - base_midpoint[0], apex[1] - base_midpoint[1])
    if length_px < 5.0:
        return None
    return LongAxis(apex=apex, base_midpoint=base_midpoint, length_px=length_px)


# ---------------------------------------------------------------------------
# Disc summation
# ---------------------------------------------------------------------------


def _line_pixels(p0: np.ndarray, p1: np.ndarray, n: int = 1024) -> np.ndarray:
    """Sample `n` equally spaced points along the segment p0-p1; return int (x,y)."""
    t = np.linspace(0.0, 1.0, n)
    pts = p0[None, :] * (1 - t)[:, None] + p1[None, :] * t[:, None]
    return np.round(pts).astype(np.int32)


def _measure_chord(mask: np.ndarray, disc_center: np.ndarray, perp: np.ndarray,
                   max_radius_px: float) -> float:
    """Length of the largest contiguous mask-true run along a perpendicular through disc_center.

    Returns chord length in pixels; 0 if the disc misses the mask entirely.
    """
    H, W = mask.shape
    far0 = disc_center - perp * max_radius_px
    far1 = disc_center + perp * max_radius_px
    n_samples = int(2 * max_radius_px) + 4
    pts = _line_pixels(far0, far1, n=n_samples)
    valid = (
        (pts[:, 0] >= 0) & (pts[:, 0] < W)
        & (pts[:, 1] >= 0) & (pts[:, 1] < H)
    )
    pts = pts[valid]
    if pts.size == 0:
        return 0.0
    samples = mask[pts[:, 1], pts[:, 0]].astype(bool)
    if not samples.any():
        return 0.0

    # Largest contiguous True run
    best_run, run = 0, 0
    for s in samples:
        if s:
            run += 1
            best_run = max(best_run, run)
        else:
            run = 0
    if best_run <= 1:
        return float(best_run)

    # Convert "n consecutive samples" → physical chord length.
    # Derive the spacing from the ORIGINAL sample count, not the post-clip
    # count. Clipping drops samples but does not change the spacing of the
    # contiguous survivors; dividing the full unclipped length by the clipped
    # count inflates `step` ~1.7-2.6x → per-disc volume ~3-7x too high.
    step = float(np.linalg.norm(far1 - far0)) / max(n_samples - 1, 1)
    return float(best_run) * step


def lv_volume_simpsons(
    mask: np.ndarray,
    num_discs: int = DEFAULT_NUM_DISCS,
) -> Optional[SimpsonsResult]:
    """Simpson's monoplane LV cavity volume from a single endocardial mask.

    Returns None if no long axis can be detected.

    Output volume is in **pixel-volume units**. Convert to mm³ / mL via
    `SimpsonsResult.volume_mL(pixel_spacing_mm)`.
    """
    long_axis = detect_long_axis(mask)
    if long_axis is None:
        return None

    apex = np.array(long_axis.apex)
    base = np.array(long_axis.base_midpoint)
    axis_vec = apex - base
    L = float(np.linalg.norm(axis_vec))
    if L <= 0:
        return None
    h = L / num_discs
    axis_unit = axis_vec / L
    perp = np.array([-axis_unit[1], axis_unit[0]])

    # Maximum chord length we can possibly measure: max(H, W).
    max_radius = float(max(mask.shape))

    diameters: List[float] = []
    volume_px = 0.0
    for i in range(num_discs):
        # Disc midpoint at (i + 0.5) * h along axis from base
        disc_center = base + axis_unit * h * (i + 0.5)
        d = _measure_chord(mask, disc_center, perp, max_radius_px=max_radius)
        diameters.append(d)
        # V_i = π × (d/2)² × h (in pixel-volume units)
        volume_px += math.pi * (d / 2.0) ** 2 * h

    return SimpsonsResult(
        volume_px=volume_px,
        long_axis=long_axis,
        disc_diameters_px=diameters,
    )


def biplane_ef(
    edv_a4c: float, esv_a4c: float, edv_a2c: float, esv_a2c: float
) -> float:
    """Biplane EF (%) — "view-averaged monoplane" formula (sensitivity comparator).

    EDV = (EDV_A4C + EDV_A2C)/2, similarly ESV; EF = (EDV-ESV)/EDV*100.

    NOTE: this is NOT the ASE disc-pairing standard. It averages the two
    monoplane *volumes* (mean-of-squares of per-disc diameters), whereas ASE
    pairs per-disc *diameters* across views (product). By AM-GM the two agree
    only when the views' per-disc diameters match; otherwise this over-estimates
    volume. Retained as the sensitivity-analysis comparator alongside
    `lv_volume_biplane_ase_mL` / `biplane_ef_ase` (the reported primary).
    """
    edv = (edv_a4c + edv_a2c) / 2.0
    esv = (esv_a4c + esv_a2c) / 2.0
    if edv <= 0:
        return float("nan")
    return (edv - esv) / edv * 100.0


def lv_volume_biplane_ase_mL(
    res_a4c: SimpsonsResult,
    res_a2c: SimpsonsResult,
    pixel_spacing_mm: float,
) -> Optional[float]:
    """ASE biplane disc-summation LV volume (mL) — Lang RM et al., JASE 2015.

    V = Σ_i  π/4 · d_i_A4C · d_i_A2C · h     (per-disc paired-ellipse area × h)

    where d_i are per-disc chord diameters (one per view, same disc index i)
    and h is the disc height. Both views use ``DEFAULT_NUM_DISCS`` discs, so
    the diameter lists align disc-for-disc. Per ASE, the disc height is taken
    from the **longer** of the two long axes (h = max(L_A4C, L_A2C)/num_discs);
    the views are imaged at the same cardiac phase so their long axes should be
    close, and taking the max avoids under-counting apical discs.

    All lengths converted to mm via ``pixel_spacing_mm`` before summation;
    result divided by 1000 → mL (= cm³). Returns None if either view lacks
    usable discs.
    """
    d4 = res_a4c.disc_diameters_px
    d2 = res_a2c.disc_diameters_px
    if not d4 or not d2 or len(d4) != len(d2):
        return None
    num_discs = len(d4)
    # Per-view long-axis length in px → disc height h (px); take the longer axis.
    L4 = res_a4c.long_axis.length_px
    L2 = res_a2c.long_axis.length_px
    L = max(L4, L2)
    if L <= 0:
        return None
    h_px = L / num_discs
    s = pixel_spacing_mm
    # V_mm3 = Σ π/4 · (d4·s) · (d2·s) · (h·s) = (π/4) s³ h_px Σ d4_i d2_i
    pair_sum = sum(a * b for a, b in zip(d4, d2))
    volume_mm3 = (math.pi / 4.0) * (s ** 3) * h_px * pair_sum
    return volume_mm3 / 1000.0


def biplane_ef_ase(
    edv_ase: float, esv_ase: float
) -> float:
    """EF (%) from ASE biplane disc-paired ED/ES volumes (matched units)."""
    if edv_ase <= 0:
        return float("nan")
    return (edv_ase - esv_ase) / edv_ase * 100.0
