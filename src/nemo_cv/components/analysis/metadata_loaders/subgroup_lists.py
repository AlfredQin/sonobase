"""Load per-subgroup sample-id lists for re-slicing benchmark predictions.

Used by the subgroup-stratification analyses (S1, S3) — the canonical
project annotation tree at `${ANNOTATION_DIR}` (canonical
`ANNOTATION_DIR=$WORK/Dataset/SaUS_Annotation/...`) already ships
pre-computed per-subgroup test lists for KidneyUS (manufacturer) and
BUSI (pathology).
This module just enumerates them and returns
`Dict[subgroup_name, Set[sample_id]]`.

Filename convention (verified against the on-disk layout):
    <annotation_root>/<DATASET>/<prefix>_<subgroup>_list.txt

Examples:
    KidneyUS / `prefix=test_mfr` →
        test_mfr_acuson_list.txt   (n=17,  → "acuson")
        test_mfr_ge_list.txt       (n=51,  → "ge")
        test_mfr_philips_list.txt  (n=260, → "philips")
        test_mfr_siemens_list.txt  (n=47,  → "siemens")
        test_mfr_toshiba_list.txt  (n=47,  → "toshiba")
    BUSI / `prefix=test_pathology` →
        test_pathology_benign_list.txt    (n=88,  → "benign")
        test_pathology_malignant_list.txt (n=42,  → "malignant")

For CAMUS quality stratification we don't use this loader — the
metadata is in the per-patient `Info_*.cfg` already parsed by
`metadata_loaders.camus.load_camus_metadata`. See
`recipes/analysis/s2_camus_quality.py` for that path.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Dict, List, Set

logger = logging.getLogger(__name__)


def load_sublist_groups(
    annotation_root: str,
    dataset: str,
    prefix: str,
) -> Dict[str, Set[str]]:
    """Read every `<annotation_root>/<dataset>/<prefix>_<group>_list.txt`.

    Args:
        annotation_root: e.g. `${ANNOTATION_DIR}` (canonical `$WORK/Dataset/SaUS_Annotation/38_pt_8_bm_7_ext`).
        dataset: e.g. ``"KidneyUS"`` or ``"BUSI"``.
        prefix: filename prefix; ``"test_mfr"`` for KidneyUS, ``"test_pathology"`` for BUSI.

    Returns:
        Dict mapping the subgroup name (the substring between `<prefix>_`
        and `_list.txt`) to the set of sample ids in that subgroup.
        Empty dict if no matching files are found.

    Example:
        >>> load_sublist_groups(
        ...     os.environ["ANNOTATION_DIR"],   # canonical $WORK/Dataset/SaUS_Annotation/38_pt_8_bm_7_ext
        ...     "KidneyUS", "test_mfr",
        ... )
        {"acuson": {...}, "ge": {...}, "philips": {...}, ...}
    """
    base = pathlib.Path(annotation_root).expanduser() / dataset
    if not base.is_dir():
        raise FileNotFoundError(
            f"Annotation directory not found: {base}. "
            "Check scratch.annotation_dir in your config."
        )

    glob_pattern = f"{prefix}_*_list.txt"
    files = sorted(base.glob(glob_pattern))
    if not files:
        logger.warning(
            f"load_sublist_groups: no files matching '{glob_pattern}' under {base}. "
            "Subgroup analysis will see an empty group set."
        )
        return {}

    out: Dict[str, Set[str]] = {}
    for path in files:
        # path.stem looks like "test_mfr_philips_list" — strip the prefix
        # and the trailing "_list" to extract the group name.
        stem = path.stem
        if not stem.startswith(prefix + "_") or not stem.endswith("_list"):
            logger.warning(f"Skipping unexpected filename: {path.name}")
            continue
        group = stem[len(prefix) + 1: -len("_list")]
        ids = {line.strip() for line in path.read_text().splitlines() if line.strip()}
        if not ids:
            logger.warning(f"Empty subgroup file: {path.name}")
            continue
        out[group] = ids
        logger.info(f"  {dataset}/{prefix}/{group}: {len(ids)} samples")

    return out


def assert_partition(groups: Dict[str, Set[str]],
                     expected_total: int = None,
                     allow_overlap: bool = False) -> None:
    """Sanity-check helper: subgroups should partition the test set.

    For KidneyUS and BUSI the on-disk subgroups DO partition the full test
    list (verified: 17+51+260+47+47=422 for KidneyUS; 88+42=130 for BUSI
    minus the unsegmentable 'normal' subset). Catching overlap or coverage
    drift early avoids confusing per-subgroup counts in the supplement table.

    Raises:
        AssertionError if subgroups overlap when allow_overlap=False, or
            if the union doesn't match expected_total when supplied.
    """
    if not allow_overlap:
        seen: Set[str] = set()
        for name, ids in groups.items():
            dup = seen & ids
            if dup:
                raise AssertionError(
                    f"Subgroup '{name}' overlaps with previously-seen subgroups "
                    f"on {len(dup)} ids (e.g. {sorted(list(dup))[:3]})."
                )
            seen |= ids

    if expected_total is not None:
        total = sum(len(ids) for ids in groups.values())
        if total != expected_total:
            raise AssertionError(
                f"Subgroup partition size {total} != expected {expected_total}."
            )


# Convenience subgroup-spec registry (used by the recipes)
SUBGROUP_SPECS = {
    "kidneyus_mfr": dict(dataset="KidneyUS", prefix="test_mfr",
                         display_name="Manufacturer"),
    "busi_pathology": dict(dataset="BUSI", prefix="test_pathology",
                           display_name="Pathology"),
}


def load_spec(spec_key: str, annotation_root: str) -> Dict[str, Set[str]]:
    """Convenience wrapper: load by spec key (e.g. "kidneyus_mfr")."""
    if spec_key not in SUBGROUP_SPECS:
        raise KeyError(
            f"Unknown subgroup spec '{spec_key}'. "
            f"Known: {sorted(SUBGROUP_SPECS)}."
        )
    spec = SUBGROUP_SPECS[spec_key]
    return load_sublist_groups(annotation_root, spec["dataset"], spec["prefix"])
