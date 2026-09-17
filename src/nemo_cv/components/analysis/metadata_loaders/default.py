"""Default metadata loader — returns empty `ClinicalMeta` for any sample.

Used by analyses on datasets that have no per-image metadata to enrich the
benchmark per_sample CSV (B2 catastrophic-failure catalog, B3 porcine, …).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ClinicalMeta:
    """All fields optional; analyses pull what they need."""
    sample_id: str
    pixel_spacing_mm: Optional[float] = None
    voxel_spacing_mm3: Optional[float] = None
    gt_clinical_value: Optional[float] = None
    gt_clinical_unit: Optional[str] = None
    image_quality: Optional[str] = None
    ed_frame_idx: Optional[int] = None
    es_frame_idx: Optional[int] = None
    view: Optional[str] = None
    gestational_age_weeks: Optional[float] = None
    notes: Optional[str] = None


def get_metadata(sample_id: str, **kwargs) -> ClinicalMeta:
    return ClinicalMeta(sample_id=sample_id)
