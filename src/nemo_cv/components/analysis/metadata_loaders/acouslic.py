"""ACOUSLIC per-video metadata loader.

ACOUSLIC ships `circumferences/fetal_abdominal_circumferences_per_sweep.csv`
with one row per video and one AC measurement (mm) per sweep. We aggregate
to one row per video by taking the **mean AC across the available sweeps**
(analysis A4).

Ground-truth version: use the Zenodo **v1.1** release (record 12697994,
2024-07-09). The v1.0 file (2024-04-18) reports every circumference 2x too
large — the organisers used the full axes as semi-axes — while images and
masks are identical between the two releases. `_parse_acouslic_csv` refuses
a file whose median AC exceeds `MAX_PLAUSIBLE_AC_MM`, so a v1.0 copy cannot
be loaded silently.

Pixel spacing: 0.28 mm/px isotropic, as recorded in the MHA headers
(`ElementSpacing`). Never add a spacing override to make predictions match a
ground-truth file — check the ground truth instead.

Optional: gestational age. If the dataset ships per-video GA in the same
CSV (or a sibling file), we read it; otherwise `gestational_age_weeks` is
None and the T2.3 FGR analysis short-circuits with a "data unavailable"
message.
"""

from __future__ import annotations

import csv
import io
import logging
import pathlib
import statistics
import zipfile
from typing import Dict, List, Optional

from .default import ClinicalMeta

logger = logging.getLogger(__name__)

_CACHE: Dict[str, Dict[str, ClinicalMeta]] = {}

DEFAULT_PIXEL_SPACING_MM = 0.28          # ACOUSLIC MHA header `ElementSpacing` (isotropic)
# A term fetus does not exceed ~370 mm AC; the v1.0 circumference file has a
# median of ~480 mm because its values are 2x too large. Used as a guard only.
MAX_PLAUSIBLE_AC_MM = 370.0


def _parse_acouslic_csv(text: str) -> Dict[str, ClinicalMeta]:
    """Parse the ACOUSLIC per-video AC CSV; aggregate to one ClinicalMeta per video.

    The published schema (verified against the dataset) has one row per
    video with 6 sweep-level AC columns:

        uuid, subject_id, sweep_1_ac_mm, sweep_2_ac_mm, ..., sweep_6_ac_mm

    The video's GT AC is the mean of the 6 per-sweep measurements. We also
    tolerate older release schemas with a single
    `ac_mm` / `circumference_mm` column for backwards compatibility.

    Raises `ValueError` if the median GT AC exceeds `MAX_PLAUSIBLE_AC_MM`,
    which identifies the Zenodo v1.0 file (values 2x too large).
    """
    table: Dict[str, ClinicalMeta] = {}

    reader = csv.DictReader(io.StringIO(text))
    field_lower = {k: k.lower() for k in (reader.fieldnames or [])}
    inverse = {v: k for k, v in field_lower.items()}

    uuid_col = next((c for c in (
        "uuid", "video_id", "case", "subject_uuid", "case_uuid"
    ) if c in inverse), None)

    sweep_cols_lower = sorted([
        c for c in inverse if c.startswith("sweep_") and c.endswith("_ac_mm")
    ])
    legacy_ac_col = next((c for c in (
        "circumference_mm", "abdominal_circumference_mm", "ac_mm", "circumference"
    ) if c in inverse), None)

    ga_col = next((c for c in (
        "gestational_age_weeks", "ga_weeks", "ga"
    ) if c in inverse), None)

    if uuid_col is None or (not sweep_cols_lower and legacy_ac_col is None):
        logger.warning(
            f"ACOUSLIC CSV: could not locate expected columns "
            f"(uuid={uuid_col!r}, sweeps={sweep_cols_lower!r}, "
            f"legacy_ac={legacy_ac_col!r}). Got: {reader.fieldnames!r}"
        )
        return {}

    for row in reader:
        uuid = row[inverse[uuid_col]].strip()
        if not uuid:
            continue

        # Mean of valid sweep measurements; fall back to legacy single column.
        ac_values: List[float] = []
        if sweep_cols_lower:
            for c_lower in sweep_cols_lower:
                raw = row.get(inverse[c_lower], "").strip()
                if raw:
                    try:
                        ac_values.append(float(raw))
                    except ValueError:
                        continue
        elif legacy_ac_col is not None:
            try:
                ac_values = [float(row[inverse[legacy_ac_col]])]
            except (KeyError, ValueError):
                pass

        if not ac_values:
            continue
        gt_ac = float(sum(ac_values) / len(ac_values))

        ga = None
        if ga_col:
            try:
                ga = float(row[inverse[ga_col]])
            except (KeyError, ValueError):
                ga = None

        table[uuid] = ClinicalMeta(
            sample_id=uuid,
            pixel_spacing_mm=DEFAULT_PIXEL_SPACING_MM,
            gt_clinical_value=gt_ac,
            gt_clinical_unit="mm",
            gestational_age_weeks=ga,
            notes=f"GT AC = mean of {len(ac_values)} sweep measurements",
        )

    if table:
        median_ac = statistics.median(m.gt_clinical_value for m in table.values())
        if median_ac > MAX_PLAUSIBLE_AC_MM:
            raise ValueError(
                f"ACOUSLIC GT AC median is {median_ac:.0f} mm (> {MAX_PLAUSIBLE_AC_MM:.0f} mm): "
                "this is the Zenodo v1.0 circumference file, whose values are 2x too large. "
                "Use the v1.1 release (record 12697994)."
            )
    return table


def load_acouslic_metadata(
    acouslic_csv_path: Optional[str] = None,
    acouslic_dir: Optional[str] = None,
) -> Dict[str, ClinicalMeta]:
    """Load per-video AC metadata.

    Pass either a direct CSV path (preferred) or the ACOUSLIC base
    directory (we'll search for the CSV under it). Returns empty dict if
    nothing usable is found — callers should treat that as "skip the
    AC-dependent analysis with a clear log line".
    """
    key = str(acouslic_csv_path or acouslic_dir or "")
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    csv_path: Optional[pathlib.Path] = None
    if acouslic_csv_path:
        p = pathlib.Path(acouslic_csv_path).expanduser()
        if p.is_file():
            csv_path = p
    elif acouslic_dir:
        d = pathlib.Path(acouslic_dir).expanduser()
        candidates = list(d.rglob("fetal_abdominal_circumferences_per_sweep.csv"))
        if candidates:
            csv_path = candidates[0]

    if csv_path is None:
        logger.warning(
            "ACOUSLIC GT CSV not found. Set scratch.acouslic_csv_path "
            "in the analysis config (or ensure raw ACOUSLIC dir is on disk)."
        )
        _CACHE[key] = {}
        return {}

    text = csv_path.read_text()
    table = _parse_acouslic_csv(text)
    _CACHE[key] = table
    logger.info(f"Loaded ACOUSLIC metadata: {len(table)} videos from {csv_path}")
    return table


def get_metadata(
    sample_id: str,
    acouslic_csv_path: Optional[str] = None,
    acouslic_dir: Optional[str] = None,
) -> Optional[ClinicalMeta]:
    return load_acouslic_metadata(acouslic_csv_path, acouslic_dir).get(sample_id)
