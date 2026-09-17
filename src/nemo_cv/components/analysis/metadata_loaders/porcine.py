"""Porcine spinal cord metadata loader.

The dataset has 5 semantic classes (per its `dataset_info.json`):
    1 = spinal cord
    2 = dura mater
    3 = epidural space
    4 = vertebral bone
    5 = intervertebral space

There's no per-image clinical metadata; this loader exists so that the B3
analysis can use the same metadata-loader interface as A1/A2/A4 without a
special case. It returns the class table as `notes`.
"""

from __future__ import annotations

import json
import logging
import pathlib
from typing import Dict, Optional

from .default import ClinicalMeta

logger = logging.getLogger(__name__)


def load_porcine_categories(saus_dir: str) -> Dict[int, str]:
    info = pathlib.Path(saus_dir).expanduser() / "dataset_info.json"
    if not info.is_file():
        logger.warning(f"PorcineSpinalCord dataset_info.json not found at {info}")
        return {}
    with info.open() as f:
        data = json.load(f)
    return {int(c["id"]): c["name"] for c in data.get("categories", [])}


def get_metadata(sample_id: str, **_kwargs) -> ClinicalMeta:
    return ClinicalMeta(sample_id=sample_id)
