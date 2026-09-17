"""RegPro per-volume voxel-spacing loader.

RegPro ships per-case 3-D NIfTI volumes inside `RegPro.zip`. The voxel
spacing lives in each NIfTI header (3 floats; mm per voxel along x/y/z).
We extract the volume's voxel-volume in mm³ for use by the C2 prostate-
volume analysis.

We **lazy-load** per case (NIfTI files are large). SaUS sample IDs look like
``val_case000065`` (``<split>_<case>``); RegPro.zip is a zip-of-zips, so the
raw NIfTI lives at ``<split>/us_images/<case>.nii.gz`` inside the nested
``<split>.zip``. If a case can't be found the loader returns None and C2
logs the missing samples.
"""

from __future__ import annotations

import io
import logging
import pathlib
import tempfile
import zipfile
from typing import Dict, Optional

from .default import ClinicalMeta

logger = logging.getLogger(__name__)


_CACHE: Dict[str, ClinicalMeta] = {}


def _read_nifti_voxel_spacing(file_bytes: bytes) -> Optional[tuple]:
    """Read the 3 voxel-spacing values (mm) from a NIfTI header.

    Uses SimpleITK (pinned in ``pyproject.toml``) to decode the gzipped
    NIfTI bytes via a temp file. Returns ``(sx, sy, sz)`` in mm, or
    ``None`` on error.
    """
    import SimpleITK as sitk
    try:
        with tempfile.NamedTemporaryFile(suffix=".nii.gz") as tmp:
            tmp.write(file_bytes)
            tmp.flush()
            img = sitk.ReadImage(tmp.name)
            spacing = img.GetSpacing()
            if len(spacing) < 3:
                return None
            return tuple(float(s) for s in spacing[:3])
    except Exception as e:
        logger.warning(f"NIfTI voxel-spacing read failed: {e}")
        return None


def get_metadata(
    sample_id: str,
    regpro_zip_path: str,
) -> Optional[ClinicalMeta]:
    """Lookup per-case voxel spacing from the raw RegPro zip.

    `sample_id` is the SaUS video id (e.g. ``val_case000065``); it is split
    into ``<split>_<case>`` to locate ``<split>/us_images/<case>.nii[.gz]``
    inside the nested ``<split>.zip`` within RegPro.zip.
    """
    cache_key = f"{regpro_zip_path}::{sample_id}"
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    zip_path = pathlib.Path(regpro_zip_path).expanduser().resolve()
    if not zip_path.is_file():
        logger.warning(f"RegPro raw zip not found: {zip_path}")
        return None

    # SaUS sample ids are "<split>_<case>" (e.g. "val_case000065"). RegPro.zip
    # is a zip-of-zips: the raw NIfTI lives at "<split>/us_images/<case>.nii[.gz]"
    # inside the nested "<split>.zip". Recurse one level to find it.
    split, _, case = sample_id.partition("_")
    if not case:
        return None
    inner_zip_name = f"{split}.zip"
    candidates = (f"us_images/{case}.nii.gz", f"us_images/{case}.nii")

    with zipfile.ZipFile(zip_path) as zf:
        if inner_zip_name not in zf.namelist():
            return None
        inner_bytes = zf.read(inner_zip_name)
    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as izf:
        match = None
        for name in izf.namelist():
            if any(name.endswith(c) for c in candidates):
                match = name
                break
        if match is None:
            return None
        data = izf.read(match)

    spacing = _read_nifti_voxel_spacing(data)
    if spacing is None:
        return None
    sx, sy, sz = spacing
    voxel_mm3 = sx * sy * sz
    meta = ClinicalMeta(
        sample_id=sample_id,
        voxel_spacing_mm3=voxel_mm3,
        notes=f"voxel_spacing_mm=({sx:.4f},{sy:.4f},{sz:.4f})",
    )
    _CACHE[cache_key] = meta
    return meta
