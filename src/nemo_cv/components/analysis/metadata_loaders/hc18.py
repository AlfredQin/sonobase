"""HC18 per-image metadata loader.

The HC18 challenge ships per-image pixel spacing (mm/px) and ground-truth
head circumference (mm) in `training_set_pixel_size_and_HC.csv` inside
`HC.zip` (the raw dataset archive). The SaUS-format conversion does NOT
preserve this metadata, so the analysis layer reads it directly from the
zip the first time it's asked — caching the parsed table in-process.

The CSV schema (verified from the raw zip):

    filename,pixel size,head circumference (mm)
    1_HC.png,0.0691358041432,44.3
    2_HC.png,0.0896585200514,56.81
    ...

Note: the CSV uses `.png` filenames; the SaUS conversion uses `.jpg` ids
(e.g. `103_HC`). We strip the extension from the CSV filename and use the
stem as the join key.
"""

from __future__ import annotations

import csv
import io
import logging
import pathlib
import zipfile
from typing import Dict, Optional

from .default import ClinicalMeta

logger = logging.getLogger(__name__)


# The CSV ships inside the raw HC.zip. Two known names — the older
# challenge release uses "training_set_pixel_size_and_HC.csv".
_KNOWN_CSV_NAMES = (
    "training_set_pixel_size_and_HC.csv",
)


# Backwards-compatible alias — older imports referred to `HC18Sample`.
# `ClinicalMeta` is now the unified shape across every metadata loader.
HC18Sample = ClinicalMeta


# In-process cache so we don't re-open the zip on every call.
_CACHE: Dict[str, Dict[str, HC18Sample]] = {}


def load_hc18_metadata(hc18_zip_path: str) -> Dict[str, ClinicalMeta]:
    """Load the per-image HC18 metadata table.

    Args:
        hc18_zip_path: Absolute path to the raw `HC.zip` shipped with the
            HC18 challenge. Must contain `training_set_pixel_size_and_HC.csv`
            at its root.

    Returns:
        Dict mapping `sample_id` (file stem, e.g. ``103_HC``) to
        `ClinicalMeta` (`pixel_spacing_mm`, `gt_clinical_value` = HC mm,
        `gt_clinical_unit` = "mm").
    """
    cached = _CACHE.get(hc18_zip_path)
    if cached is not None:
        return cached

    zip_path = pathlib.Path(hc18_zip_path).expanduser().resolve()
    if not zip_path.is_file():
        raise FileNotFoundError(
            f"HC18 raw zip not found: {zip_path}. "
            "Set scratch.hc18_raw_zip in the analysis config to the correct path."
        )

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        csv_name = next((n for n in _KNOWN_CSV_NAMES if n in names), None)
        if csv_name is None:
            for n in names:
                if n.endswith(_KNOWN_CSV_NAMES[0]):
                    csv_name = n
                    break
        if csv_name is None:
            raise FileNotFoundError(
                f"Could not find {_KNOWN_CSV_NAMES[0]} inside {zip_path}. "
                f"Available files: {names[:5]}..."
            )
        with zf.open(csv_name) as f:
            text = io.TextIOWrapper(f, encoding="utf-8").read()

    table: Dict[str, ClinicalMeta] = {}
    reader = csv.reader(io.StringIO(text))
    header = next(reader, None)
    if header is None or len(header) < 3:
        raise ValueError(f"Unexpected HC18 CSV header: {header!r}")

    for row in reader:
        if len(row) < 3 or not row[0].strip():
            continue
        filename = row[0].strip()
        stem = pathlib.Path(filename).stem
        try:
            pixel_spacing = float(row[1])
            gt_hc = float(row[2])
        except ValueError:
            logger.warning(f"Skipping malformed row: {row}")
            continue
        table[stem] = ClinicalMeta(
            sample_id=stem,
            pixel_spacing_mm=pixel_spacing,
            gt_clinical_value=gt_hc,
            gt_clinical_unit="mm",
        )

    _CACHE[hc18_zip_path] = table
    logger.info(f"Loaded HC18 metadata: {len(table)} samples from {zip_path}")
    return table


def get_metadata(sample_id: str, hc18_zip_path: str) -> Optional[ClinicalMeta]:
    """Return per-sample metadata or ``None`` if the sample isn't in the CSV."""
    return load_hc18_metadata(hc18_zip_path).get(sample_id)
