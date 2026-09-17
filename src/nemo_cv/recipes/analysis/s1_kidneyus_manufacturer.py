"""Analysis S1 — KidneyUS stratified by scanner manufacturer.

Stage-2 analysis. Reads 6 Stage-1 KidneyUS prediction directories
(3 models × 2 prompts) and reports per-manufacturer mean IoU/Dice +
paired Wilcoxon SonoBase vs each baseline.

CPU-only. Uses the existing analysis stack:
  * `prediction_io.read_per_sample_csv` — Stage-1 outputs.
  * `metadata_loaders.subgroup_lists.load_sublist_groups` — manufacturer membership.
  * `_subgroup_common.run_subgroup_analysis` — shared driver.

CLI:
    cd src
    uv run python -m nemo_cv.recipes.analysis.s1_kidneyus_manufacturer \\
      --runs sonobase_point=<...> sonobase_box=<...> \\
             medsam2_point=<...>  medsam2_box=<...> \\
             sam2_no_ft_point=<...> sam2_no_ft_box=<...> \\
      --output-dir ./experiments/analysis/s1_kidneyus_manufacturer/KidneyUS_0corr \\
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
    p = argparse.ArgumentParser(description="KidneyUS × Manufacturer (S1).")
    p.add_argument("--runs", nargs="+", required=True,
                   help="`<model>_<prompt>=<predictions_dir>` per run; expect 6 (3 models × 2 prompts).")
    p.add_argument("--output-dir", required=True)
    _annotation_default = os.environ.get("ANNOTATION_DIR")
    p.add_argument("--annotation-dir", default=_annotation_default,
                   required=(_annotation_default is None),
                   help="Annotation root (default: $ANNOTATION_DIR env var; required if unset).")
    args = p.parse_args()

    runs = _parse_runs(args.runs)
    subgroups = load_sublist_groups(args.annotation_dir, "KidneyUS", "test_mfr")
    # Full KidneyUS test set, re-derived from the transducer spreadsheet by
    # data.utils.gen_kidneyus_mfr_lists (17+51+260+47+47). Update this if the
    # KidneyUS test split changes.
    assert_partition(subgroups, expected_total=422)
    logger.info(f"KidneyUS subgroups: {[(k, len(v)) for k, v in subgroups.items()]}")

    run_subgroup_analysis(
        analysis_id="S1",
        title="KidneyUS by Scanner Manufacturer",
        spec_section="S1",
        runs=runs,
        subgroups=subgroups,
        output_dir=pathlib.Path(args.output_dir).expanduser().resolve(),
        subgroup_axis_name="manufacturer",
        subgroup_source=str(args.annotation_dir),
    )


if __name__ == "__main__":
    main()
