"""Analysis T3.2 — SonoBase failure analysis (worst-N qualitative).

Mirror of B2, but inverted: find samples where **SonoBase** fails (lowest
IoU). Used for the paper's "where SonoBase still struggles" qualitative
discussion. The resulting figures are categorised manually, each failure
into one of:
    A — low contrast / acoustic shadowing
    B — unusual anatomy / pathology
    C — motion blur (videos only)
    D — very small target
    E — out-of-distribution image type (Doppler / elastography / scanline / …)
    F — other
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
from collections import Counter, defaultdict
from typing import Dict

from nemo_cv.components.analysis import prediction_io as pio
from nemo_cv.components.analysis.plots.summary_table import summary_table_figure
from nemo_cv.recipes.analysis._failure_common import (
    cross_join, render_catalog_pages, select_worst_rows, write_catalog_json,
)

logger = logging.getLogger(__name__)


SONOBASE_IOU_MAX = 0.70    # Only consider sub-70 IoU as a failure worth showing


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    p = argparse.ArgumentParser(description="SonoBase failure analysis (T3.2).")
    p.add_argument("--sam2", required=True)
    p.add_argument("--medsam2", required=True)
    p.add_argument("--sonobase", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--per-dataset-top-k", type=int, default=5,
                   help="Worst-K per dataset (where SonoBase fails).")
    p.add_argument("--samples-per-page", type=int, default=6)
    args = p.parse_args()

    out_dir = pathlib.Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = {
        "sam2_no_ft": pathlib.Path(args.sam2).resolve(),
        "medsam2": pathlib.Path(args.medsam2).resolve(),
        "sonobase": pathlib.Path(args.sonobase).resolve(),
    }
    rows = cross_join(runs)
    logger.info(f"Cross-joined {len(rows)} per-(sample,frame,obj) rows")

    def predicate(ious: Dict[str, float]) -> bool:
        return ious["sonobase"] < SONOBASE_IOU_MAX

    # Per-dataset top-K (where SonoBase fails)
    by_dataset = defaultdict(list)
    for r in rows:
        if predicate(r.ious):
            by_dataset[r.key.dataset].append(r)
    selected = []
    for ds, ds_rows in by_dataset.items():
        ds_rows.sort(key=lambda r: r.ious["sonobase"])
        selected.extend(ds_rows[:args.per_dataset_top_k])

    counts = Counter(r.key.dataset for r in selected)
    table_rows = [["Dataset", "# SonoBase failures shown",
                   "Total <" + str(SONOBASE_IOU_MAX)]]
    for ds, ds_rows in by_dataset.items():
        table_rows.append([ds, str(counts.get(ds, 0)), str(len(ds_rows))])
    summary_table_figure(table_rows, out_dir / "count_table.pdf",
                         title="SonoBase failure cases — T3.2 (per-dataset)")

    figures_dir = out_dir / "figures"
    pages = render_catalog_pages(
        selected,
        model_order=["sam2_no_ft", "medsam2", "sonobase"],
        output_dir=figures_dir,
        title_prefix="T3.2 — SonoBase failures: ",
        samples_per_page=args.samples_per_page,
    )

    write_catalog_json(
        selected, out_dir / "failure_catalog.json",
        analysis="T3.2",
        title="SonoBase failure analysis (worst-N qualitative)",
        thresholds={"sonobase_iou_max": SONOBASE_IOU_MAX},
        per_dataset_counts=dict(counts),
        per_dataset_total_failures={k: len(v) for k, v in by_dataset.items()},
        figure_pages=[str(p) for p in pages],
    )

    logger.info(f"T3.2 analysis complete. Outputs at: {out_dir}")


if __name__ == "__main__":
    main()
