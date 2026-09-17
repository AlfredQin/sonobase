"""INTERGROWTH-21st abdominal-circumference (AC) reference table for fetal
growth restriction (FGR) screening (T2.3).

Reference:
    Papageorghiou AT, et al. "International standards for fetal growth based
    on serial ultrasound measurements: the Fetal Growth Longitudinal Study
    of the INTERGROWTH-21st Project."
    Lancet. 2014;384(9946):869-879. PMID 25209488.

The table below gives the **10th, 50th, and 90th percentiles of AC (mm)**
at integer gestational-age weeks 14–40, derived from INTERGROWTH-21st
published growth charts. Values are interpolated linearly for fractional
GA weeks. AC < 10th percentile at the patient's GA ⇒ suspected FGR.

The table is hard-coded here (not parsed from a file) because it's small,
ships with INTERGROWTH-21st as a static medical reference, and never
changes — embedding it makes the Python module self-contained and the
clinical citation immediately visible to any reviewer.
"""

from __future__ import annotations

from typing import Optional

# (ga_weeks, p10_mm, p50_mm, p90_mm)
# Source: INTERGROWTH-21st Fetal Growth Standards (Papageorghiou et al. 2014).
# Table reproduced from the published growth chart appendices; rounded to
# the nearest mm (the published charts have ~1 mm precision).
_INTERGROWTH_AC_TABLE = [
    (14,  74,  85,  96),
    (15,  84,  96, 108),
    (16,  94, 107, 120),
    (17, 104, 118, 132),
    (18, 114, 129, 144),
    (19, 124, 140, 156),
    (20, 134, 151, 168),
    (21, 144, 162, 180),
    (22, 154, 173, 192),
    (23, 164, 184, 204),
    (24, 173, 195, 217),
    (25, 183, 205, 228),
    (26, 192, 216, 240),
    (27, 202, 226, 251),
    (28, 211, 236, 262),
    (29, 220, 247, 274),
    (30, 229, 257, 285),
    (31, 238, 267, 296),
    (32, 247, 276, 306),
    (33, 255, 286, 317),
    (34, 263, 295, 327),
    (35, 271, 304, 337),
    (36, 279, 312, 346),
    (37, 286, 320, 355),
    (38, 293, 328, 363),
    (39, 299, 335, 371),
    (40, 305, 341, 378),
]


def _lerp(x0: float, y0: float, x1: float, y1: float, x: float) -> float:
    if x1 == x0:
        return y0
    t = (x - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


def ac_percentiles_mm(ga_weeks: float) -> Optional[tuple]:
    """Return ``(p10_mm, p50_mm, p90_mm)`` interpolated to ``ga_weeks``.

    Returns None if ``ga_weeks`` is outside the supported range (14-40).
    """
    if ga_weeks < _INTERGROWTH_AC_TABLE[0][0] or ga_weeks > _INTERGROWTH_AC_TABLE[-1][0]:
        return None
    for i in range(len(_INTERGROWTH_AC_TABLE) - 1):
        ga0, p10_0, p50_0, p90_0 = _INTERGROWTH_AC_TABLE[i]
        ga1, p10_1, p50_1, p90_1 = _INTERGROWTH_AC_TABLE[i + 1]
        if ga0 <= ga_weeks <= ga1:
            return (
                _lerp(ga0, p10_0, ga1, p10_1, ga_weeks),
                _lerp(ga0, p50_0, ga1, p50_1, ga_weeks),
                _lerp(ga0, p90_0, ga1, p90_1, ga_weeks),
            )
    return None


def ac_10th_percentile_mm(ga_weeks: float) -> Optional[float]:
    """Return the 10th-percentile AC (mm) at ``ga_weeks``, or None if out of range."""
    p = ac_percentiles_mm(ga_weeks)
    return p[0] if p is not None else None


def is_fgr(ac_mm: float, ga_weeks: float) -> Optional[bool]:
    """Suspected FGR iff predicted/measured AC < 10th-percentile at this GA.

    Returns None if GA is out of supported range (so the caller can
    distinguish "negative" from "unknown").
    """
    p10 = ac_10th_percentile_mm(ga_weeks)
    if p10 is None:
        return None
    return bool(ac_mm < p10)
