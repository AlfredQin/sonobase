"""Analysis S3 — BUSI stratified by pathology.

Stage-2 analysis. Reads 6 Stage-1 BUSI prediction directories (3 models ×
2 prompts) and reports per-pathology (benign / malignant) mean IoU/Dice
+ paired Wilcoxon SonoBase vs each baseline.

BUSI 'normal' images are excluded —
they have no lesion to segment so per-image IoU/Dice are undefined.

CLI:
    cd src
    uv run python -m nemo_cv.recipes.analysis.s3_busi_pathology \\
      --runs sonobase_point=<...> sonobase_box=<...> \\
             medsam2_point=<...>  medsam2_box=<...> \\
             sam2_no_ft_point=<...> sam2_no_ft_box=<...> \\
      --output-dir ./experiments/analysis/s3_busi_pathology/BUSI_0corr \\
      --annotation-dir "$ANNOTATION_DIR"     # or set ANNOTATION_DIR env var
"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib

from nemo_cv.components.analysis.metadata_loaders.subgroup_lists import (
    assert_partition, load_sublist_groups,
)
from nemo_cv.recipes.analysis._subgroup_common import run_subgroup_analysis

logger = logging.getLogger(__name__)


def _parse_runs(items):
    out = {}
    for s in items:
        if "=" not in s:
            raise ValueError(f"--runs entry must be `<model>_<prompt>=<dir>`, got: {s!r}")
        key, path = s.split("=", 1)
        parts = key.rsplit("_", 1)
        if len(parts) != 2 or parts[1] not in ("point", "box"):
            raise ValueError(f"Expected `<model>_<point|box>`, got: {key!r}")
        model, prompt = parts[0], parts[1]
        out[(model, prompt)] = pathlib.Path(path).expanduser().resolve()
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    p = argparse.ArgumentParser(description="BUSI × Pathology (S3).")
    p.add_argument("--runs", nargs="+", required=True,
                   help="`<model>_<prompt>=<predictions_dir>` per run; expect 6.")
    p.add_argument("--output-dir", required=True)
    _annotation_default = os.environ.get("ANNOTATION_DIR")
    p.add_argument("--annotation-dir", default=_annotation_default,
                   required=(_annotation_default is None),
                   help="Annotation root (default: $ANNOTATION_DIR env var; required if unset).")
    args = p.parse_args()

    runs = _parse_runs(args.runs)
    subgroups = load_sublist_groups(args.annotation_dir, "BUSI", "test_pathology")
    # benign + malignant over the BUSI test split (88 + 42); the unsegmentable
    # "normal" class carries no mask and is excluded. Update if the split changes.
    assert_partition(subgroups, expected_total=130)
    logger.info(f"BUSI pathology subgroups: {[(k, len(v)) for k, v in subgroups.items()]}")

    run_subgroup_analysis(
        analysis_id="S3",
        title="BUSI by Pathology Class",
        spec_section="S3",
        runs=runs,
        subgroups=subgroups,
        output_dir=pathlib.Path(args.output_dir).expanduser().resolve(),
        subgroup_axis_name="pathology",
        subgroup_source=str(args.annotation_dir),
    )


if __name__ == "__main__":
    main()
