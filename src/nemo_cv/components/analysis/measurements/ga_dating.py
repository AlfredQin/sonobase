"""Gestational-age dating from head circumference (Hadlock 1984).

Used by analysis T2.1: report HC measurement error in days of GA error
rather than mm of HC error. ±1 day GA error is an immediately interpretable
clinical number; "±2 mm HC error" is not.

Reference:
    Hadlock FP, Deter RL, Harrist RB, Park SK.
    "Estimating fetal age: computer-assisted analysis of multiple
    fetal growth parameters."
    Radiology. 1984;152(2):497-501. PMID: 6739822.
    Standard error of estimate: 1.23 weeks.

Terminology note:
    The original paper calls the output "menstrual age" (MA). In modern
    clinical practice MA = GA (gestational age) — both count from the
    first day of the last menstrual period. We use "GA" throughout.
"""

from __future__ import annotations


def hc_to_ga_weeks(hc_mm: float) -> float:
    """Convert head circumference (mm) to gestational age (weeks).

    The Hadlock formula expects HC in **centimetres**:

        GA (weeks) = 8.96 + 0.540 * HC_cm + 0.0003 * HC_cm^3

    We convert mm -> cm internally so callers can pass mm directly.
    """
    hc_cm = hc_mm / 10.0
    return 8.96 + 0.540 * hc_cm + 0.0003 * (hc_cm ** 3)


def hc_to_ga_days(hc_mm: float) -> float:
    """Convert HC (mm) to GA in days. Useful for reporting "±N days" errors."""
    return hc_to_ga_weeks(hc_mm) * 7.0
