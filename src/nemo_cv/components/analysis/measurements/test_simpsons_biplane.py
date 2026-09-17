"""Regression tests for Simpson's-biplane disc measurement.

**What these guard.** ``_measure_chord`` turns a contiguous run of mask-true
samples into a physical length by multiplying the run count by a per-sample
spacing. Deriving that spacing from the *post-clip* sample count inflates it
~1.7-2.6x whenever the measurement line is clipped by the image border — which
happens for **every** disc, because ``lv_volume_simpsons`` samples with
``max_radius_px = max(H, W)`` — and per-disc volumes then come out 3-7x too
high. The spacing must come from the ORIGINAL (unclipped) sample count.

**Ground truth for the test.** A chord through the centre of a filled disc of
radius ``R`` must measure its diameter, ``2R`` (``chord == 2 * radius``),
regardless of the chord angle or how far the measurement line runs off the
image. A post-clip spacing violates this by the inflation factor above.

Run standalone (``python test_simpsons_biplane.py``) or under pytest.
"""
from __future__ import annotations

import math

import numpy as np

from nemo_cv.components.analysis.measurements.simpsons_biplane import _measure_chord

# A chord is measured on an integer pixel grid, so allow ~2 px of discretisation
# slack. The spacing defect inflates the chord to 1.7-2.6x the diameter (e.g. 136-208 px
# for a 80 px diameter), so this tolerance still fails loudly on a regression.
_TOL_PX = 2.0


def _filled_disc(h: int, w: int, cy: float, cx: float, r: float) -> np.ndarray:
    yy, xx = np.ogrid[:h, :w]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= r ** 2


def test_chord_through_centre_equals_diameter():
    """chord through a disc centre == 2R (± ~2 px), at several angles.

    Uses the production ``max_radius_px = max(H, W)`` so the sampling line runs
    off the image on both ends and IS clipped — the exact condition that
    triggered the pre-fix spacing inflation.
    """
    h = w = 200
    r = 40.0
    cx = cy = 100.0
    mask = _filled_disc(h, w, cy, cx, r)
    centre = np.array([cx, cy])              # module convention: (x, y)
    max_radius = float(max(h, w))            # forces clipping, as in lv_volume_simpsons
    for deg in (0.0, 30.0, 45.0, 60.0, 90.0, 135.0):
        th = math.radians(deg)
        perp = np.array([math.cos(th), math.sin(th)])
        chord = _measure_chord(mask, centre, perp, max_radius_px=max_radius)
        assert abs(chord - 2 * r) <= _TOL_PX, (
            f"angle {deg}deg: chord {chord:.2f} px vs diameter {2 * r:.1f} px "
            f"(regression: spacing inflation would blow this up to >=1.7x)"
        )


def test_offcentre_disc_against_border_not_inflated():
    """A disc pushed into a corner still measures ~2R and is never inflated.

    Heavier clipping (the line is cut close to the disc on one side) is exactly
    where the pre-fix spacing error was largest.
    """
    h = w = 160
    r = 30.0
    cy, cx = 30.0, 30.0                       # near the top-left corner
    mask = _filled_disc(h, w, cy, cx, r)
    centre = np.array([cx, cy])
    max_radius = float(max(h, w))
    chord = _measure_chord(mask, centre, np.array([0.0, 1.0]), max_radius_px=max_radius)
    assert abs(chord - 2 * r) <= _TOL_PX, f"chord {chord:.2f} px vs {2 * r:.1f} px"
    assert chord <= 2 * r + _TOL_PX, "chord inflated beyond the diameter (regression)"


def test_disc_miss_returns_zero():
    """A disc whose perpendicular never crosses the mask returns 0 (guard path)."""
    mask = _filled_disc(120, 120, 60.0, 60.0, 20.0)
    far_centre = np.array([10.0, 10.0])       # outside the disc
    chord = _measure_chord(mask, far_centre, np.array([1.0, 0.0]), max_radius_px=5.0)
    assert chord == 0.0


if __name__ == "__main__":
    test_chord_through_centre_equals_diameter()
    test_offcentre_disc_against_border_not_inflated()
    test_disc_miss_returns_zero()
    print("simpsons_biplane disc regression tests passed")
