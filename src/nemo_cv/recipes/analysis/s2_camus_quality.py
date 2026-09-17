"""Analysis S2 — CAMUS stratified by image quality.

Stage-2 analysis. Reads 6 Stage-1 CAMUS prediction directories (3 models
× 2 prompts) and reports per-quality (Good / Medium / Poor) mean IoU/Dice
+ paired Wilcoxon SonoBase vs each baseline.

Per-patient quality is the WORST quality across the patient's two views
(`<patient>_2CH` and `<patient>_4CH`) — same convention as A1 EF analysis.

CLI:
    cd src
    uv run python -m nemo_cv.recipes.analysis.s2_camus_quality \\
      --runs sonobase_point=<...> sonobase_box=<...> \\
             medsam2_point=<...>  medsam2_box=<...> \\
             sam2_no_ft_point=<...> sam2_no_ft_box=<...> \\
      --output-dir ./experiments/analysis/s2_camus_quality/CAMUS_0corr \\
      --camus-zip "$CAMUS_ZIP"     # or set CAMUS_ZIP env var
"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib
from collections import defaultdict
from typing import Dict, Set

from nemo_cv.components.analysis.metadata_loaders.camus import load_camus_metadata
from nemo_cv.recipes.analysis._subgroup_common import run_subgroup_analysis

logger = logging.getLogger(__name__)


_QUALITY_RANK = {"Good": 0, "Medium": 1, "Poor": 2}


def _build_quality_subgroups(camus_zip_path: str) -> Dict[str, Set[str]]:
    """Map each CAMUS sample_id (= `<patient>_<view>`) to its patient's
    worst-of-2-views quality. Returns Dict[quality, Set[sample_id]] with
    keys in {Good, Medium, Poor}."""
    metadata = load_camus_metadata(camus_zip_path)
    by_patient_views: Dict[str, list] = defaultdict(list)
    for sample_id, meta in metadata.items():
        if "_" not in sample_id:
            continue
        patient = sample_id.rsplit("_", 1)[0]
        if meta.image_quality:
            by_patient_views[patient].append((sample_id, meta.image_quality))

    out: Dict[str, Set[str]] = defaultdict(set)
    for patient, view_pairs in by_patient_views.items():
        # Worst quality across this patient's views (max rank → worst).
        worst = max(
            (q for _, q in view_pairs if q in _QUALITY_RANK),
            key=_QUALITY_RANK.__getitem__,
            default=None,
        )
        if worst is None:
            continue
        for sample_id, _ in view_pairs:
            out[worst].add(sample_id)
    return dict(out)


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
    p = argparse.ArgumentParser(description="CAMUS × Image Quality (S2).")
    p.add_argument("--runs", nargs="+", required=True,
                   help="`<model>_<prompt>=<predictions_dir>` per run; expect 6.")
    p.add_argument("--output-dir", required=True)
    _camus_default = os.environ.get("CAMUS_ZIP")
    p.add_argument("--camus-zip", default=_camus_default,
                   required=(_camus_default is None),
                   help="Path to CAMUS raw zip (default: $CAMUS_ZIP env var; required if unset).")
    args = p.parse_args()

    runs = _parse_runs(args.runs)
    subgroups = _build_quality_subgroups(args.camus_zip)
    logger.info(f"CAMUS quality subgroups: {[(k, len(v)) for k, v in subgroups.items()]}")

    run_subgroup_analysis(
        analysis_id="S2",
        title="CAMUS by Image Quality",
        spec_section="S2",
        runs=runs,
        subgroups=subgroups,
        output_dir=pathlib.Path(args.output_dir).expanduser().resolve(),
        subgroup_axis_name="quality",
        subgroup_source=str(args.camus_zip),
    )


if __name__ == "__main__":
    main()
