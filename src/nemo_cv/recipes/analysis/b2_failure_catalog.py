"""Analysis B2 — Catastrophic-failure catalog.

Find samples where SAM2 (no-ft) AND/OR MedSAM2 fail catastrophically
(IoU < 10) but SonoBase succeeds (IoU > 50). These are the "SonoBase
saves the day" cases that are most useful for the paper's qualitative
discussion.

Pipeline:
  1. Cross-join the three Stage-1 prediction CSVs per (sample × frame × obj).
  2. Predicate:
        catastrophic = (sam2_iou < 0.10  OR  medsam2_iou < 0.10)
                       AND  sonobase_iou > 0.50
  3. Per-dataset count → table.
  4. Worst-K (rank by min(sam2, medsam2) ascending) → side-by-side figure.
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
from collections import Counter
from typing import Dict, List

from nemo_cv.components.analysis import prediction_io as pio
from nemo_cv.components.analysis.plots.style import display_name
from nemo_cv.components.analysis.plots.summary_table import summary_table_figure
from nemo_cv.recipes.analysis._failure_common import (
    cross_join, render_catalog_pages, select_worst_rows, write_catalog_json,
)

logger = logging.getLogger(__name__)


SAM2_IOU_THRESHOLD = 0.10
MEDSAM2_IOU_THRESHOLD = 0.10
SONOBASE_IOU_THRESHOLD = 0.50


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    p = argparse.ArgumentParser(description="Catastrophic failure catalog (B2).")
    p.add_argument("--sam2", required=True, help="Stage-1 dir for SAM2 (no-ft).")
    p.add_argument("--medsam2", required=True, help="Stage-1 dir for MedSAM2.")
    p.add_argument("--sonobase", required=True, help="Stage-1 dir for SonoBase.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--top-k", type=int, default=24,
                   help="Top-K worst per-baseline failures resolved by SonoBase.")
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
        return (
            (ious["sam2_no_ft"] < SAM2_IOU_THRESHOLD
             or ious["medsam2"] < MEDSAM2_IOU_THRESHOLD)
            and ious["sonobase"] > SONOBASE_IOU_THRESHOLD
        )

    matching = [r for r in rows if predicate(r.ious)]
    logger.info(f"Catastrophic-failure-resolved rows: {len(matching)}")

    # Per-dataset counts
    counts = Counter(r.key.dataset for r in matching)
    table_rows = [["Dataset", "# rescued"]]
    for ds, n in counts.most_common():
        table_rows.append([ds, str(n)])
    summary_table_figure(table_rows, out_dir / "count_table.pdf",
                         title="Catastrophic failures resolved by SonoBase — B2")

    # Worst-K (rank by the smaller of the two baseline IoUs, ascending)
    worst = select_worst_rows(
        rows,
        predicate=predicate,
        sort_key=lambda iou: min(iou["sam2_no_ft"], iou["medsam2"]),
        top_k=args.top_k,
    )

    figures_dir = out_dir / "figures"
    pages = render_catalog_pages(
        worst,
        model_order=["sam2_no_ft", "medsam2", "sonobase"],
        output_dir=figures_dir,
        title_prefix="B2 — SonoBase saves: ",
        samples_per_page=args.samples_per_page,
    )

    write_catalog_json(
        worst, out_dir / "failure_catalog.json",
        analysis="B2",
        title="Catastrophic failures resolved by SonoBase",
        thresholds={
            "sam2_iou_max": SAM2_IOU_THRESHOLD,
            "medsam2_iou_max": MEDSAM2_IOU_THRESHOLD,
            "sonobase_iou_min": SONOBASE_IOU_THRESHOLD,
        },
        per_dataset_counts=dict(counts),
        figure_pages=[str(p) for p in pages],
    )

    logger.info(f"B2 analysis complete. Outputs at: {out_dir}")


if __name__ == "__main__":
    main()
