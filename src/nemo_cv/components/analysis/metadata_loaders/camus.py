"""CAMUS per-patient-view metadata loader.

The CAMUS challenge ships per-patient-view `Info_<view>.cfg` files inside
`CAMUS.zip` (the raw dataset archive) containing:

    ED: 1                    # 1-indexed end-diastole frame
    ES: 18                   # 1-indexed end-systole frame
    NbFrame: 18
    Sex: F
    Age: 56
    ImageQuality: Good       # Good / Medium / Poor (used for stratification)
    EF: 54                   # ground-truth ejection fraction (%)
    FrameRate: 48.4

We parse all of them once and cache by `sample_id` (the SaUS video-id, e.g.
``patient0004_2CH``).
"""

from __future__ import annotations

import io
import logging
import pathlib
import zipfile
from typing import Dict, Optional

from .default import ClinicalMeta

logger = logging.getLogger(__name__)

# In-process cache keyed by zip path.
_CACHE: Dict[str, Dict[str, ClinicalMeta]] = {}


def _parse_cfg(text: str) -> dict:
    """Parse a colon-separated key/value config (one per line)."""
    out = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        out[k.strip()] = v.strip()
    return out


def load_camus_metadata(camus_zip_path: str) -> Dict[str, ClinicalMeta]:
    """Load all CAMUS per-patient-view metadata from the raw zip.

    Returns a dict keyed by ``"patientXXXX_YCH"`` (e.g. ``patient0001_2CH``)
    matching the SaUS video-id convention.
    """
    cached = _CACHE.get(camus_zip_path)
    if cached is not None:
        return cached

    zip_path = pathlib.Path(camus_zip_path).expanduser().resolve()
    if not zip_path.is_file():
        raise FileNotFoundError(
            f"CAMUS raw zip not found: {zip_path}. "
            "Set scratch.camus_raw_zip in the analysis config to the correct path."
        )

    table: Dict[str, ClinicalMeta] = {}
    with zipfile.ZipFile(zip_path) as zf:
        cfg_names = [n for n in zf.namelist() if n.endswith(".cfg") and "Info_" in n]
        for name in cfg_names:
            with zf.open(name) as f:
                text = io.TextIOWrapper(f, encoding="utf-8").read()
            kv = _parse_cfg(text)

            # name = ".../patientXXXX/Info_2CH.cfg" -> patientXXXX_2CH
            parts = pathlib.PurePosixPath(name).parts
            patient = next((p for p in parts if p.startswith("patient")), None)
            view_file = pathlib.PurePosixPath(name).stem  # "Info_2CH"
            if patient is None or "_" not in view_file:
                continue
            view = view_file.split("_", 1)[1]   # "2CH" / "4CH"
            sample_id = f"{patient}_{view}"

            try:
                ef = float(kv.get("EF", "")) if kv.get("EF", "") else None
            except ValueError:
                ef = None
            # CFG file is 1-indexed; convert to 0-indexed for our pipeline.
            try:
                ed = int(kv.get("ED", "")) - 1 if kv.get("ED", "") else None
            except ValueError:
                ed = None
            try:
                es = int(kv.get("ES", "")) - 1 if kv.get("ES", "") else None
            except ValueError:
                es = None

            quality = kv.get("ImageQuality") or None
            if quality is not None:
                quality = quality.strip().capitalize()

            table[sample_id] = ClinicalMeta(
                sample_id=sample_id,
                gt_clinical_value=ef,
                gt_clinical_unit="%" if ef is not None else None,
                image_quality=quality,
                ed_frame_idx=ed,
                es_frame_idx=es,
                view=view,
            )

    _CACHE[camus_zip_path] = table
    logger.info(f"Loaded CAMUS metadata: {len(table)} patient-views from {zip_path}")
    return table


def get_metadata(sample_id: str, camus_zip_path: str) -> Optional[ClinicalMeta]:
    return load_camus_metadata(camus_zip_path).get(sample_id)
