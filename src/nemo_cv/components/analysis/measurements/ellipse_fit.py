"""Ellipse fitting and ellipse-perimeter (head/abdomen circumference) utilities.

Used by the HC18 (A2), ACOUSLIC (A4), and BUSI/BUS-BRA (C1) analyses.
Pure NumPy + OpenCV; no model dependency. Easy to unit-test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

# cv2.fitEllipse needs at least 5 contour points
MIN_CONTOUR_POINTS_FOR_ELLIPSE = 5


@dataclass
class EllipseFit:
    """Result of fitting an ellipse to a binary mask.

    Axis lengths are FULL axes in pixels (matching `cv2.fitEllipse`'s
    convention — what gets returned in the second tuple of its output).
    """

    center_xy: tuple                # (x, y), pixels
    major_axis_px: float            # full length of the major axis
    minor_axis_px: float            # full length of the minor axis
    angle_deg: float                # rotation angle of the major axis (cv2 convention)
    contour_n_points: int

    def major_mm(self, pixel_spacing_mm: float) -> float:
        return self.major_axis_px * pixel_spacing_mm

    def minor_mm(self, pixel_spacing_mm: float) -> float:
        return self.minor_axis_px * pixel_spacing_mm


def fit_ellipse(mask: np.ndarray) -> Optional[EllipseFit]:
    """Fit an ellipse to the largest connected component of a binary mask.

    Returns ``None`` if the mask is empty or the largest contour has fewer
    than 5 points (`cv2.fitEllipse` requirement).

    Args:
        mask: 2-D `np.ndarray`. Non-zero values are treated as foreground.

    Returns:
        `EllipseFit` (full-axis lengths in pixels) or `None`.
    """
    mask_u8 = (np.asarray(mask) > 0).astype(np.uint8)
    if mask_u8.sum() == 0:
        return None

    contours, _ = cv2.findContours(
        mask_u8, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return None

    # Pick the largest contour by area
    largest = max(contours, key=cv2.contourArea)
    if len(largest) < MIN_CONTOUR_POINTS_FOR_ELLIPSE:
        return None

    (cx, cy), (ax1, ax2), angle = cv2.fitEllipse(largest)
    # cv2.fitEllipse returns (axis1, axis2) where axis1 is along the angle
    # direction and axis2 is perpendicular. We take major = max, minor = min.
    major = float(max(ax1, ax2))
    minor = float(min(ax1, ax2))
    return EllipseFit(
        center_xy=(float(cx), float(cy)),
        major_axis_px=major,
        minor_axis_px=minor,
        angle_deg=float(angle),
        contour_n_points=int(len(largest)),
    )


def ellipse_perimeter_rms(major_mm: float, minor_mm: float) -> float:
    """Compute ellipse perimeter using the root-mean-square (Euler) approximation.

    This is the formula referenced by the HC18 challenge ground-truth:

        P = pi * sqrt( (a^2 + b^2) / 2 )

    where ``a`` and ``b`` are the **full axis lengths** (NOT semi-axes) in
    millimetres. Returns the perimeter in the same units as the inputs.

    Notes:
        - This is ~1% off from the true ellipse perimeter at typical
          fetal-skull eccentricities (~0.3–0.5). The HC18 ground-truth HC
          values appear to use the same approximation, so when fitting
          ellipses to GT masks we recover GT HC to ~1.4 mm MAE — the
          systematic floor of the method (analysis A2).
        - For abdominal circumference (ACOUSLIC, A4) the same formula
          applies — the only difference is the input mask is the abdomen
          contour, not the skull.
    """
    return math.pi * math.sqrt((major_mm ** 2 + minor_mm ** 2) / 2.0)


def head_circumference_mm(
    mask: np.ndarray, pixel_spacing_mm: float
) -> Optional[float]:
    """Convenience wrapper: ellipse-fit + Ramanujan perimeter, in mm.

    Returns ``None`` if no ellipse can be fit (mask too small, fewer than 5
    contour points). Used by both A2 (HC18 head circumference) and A4
    (ACOUSLIC abdominal circumference) — same math, different anatomy.
    """
    fit = fit_ellipse(mask)
    if fit is None:
        return None
    return ellipse_perimeter_rms(
        fit.major_mm(pixel_spacing_mm),
        fit.minor_mm(pixel_spacing_mm),
    )


# Alias — same operation, semantically labelled for AC analyses
abdominal_circumference_mm = head_circumference_mm
