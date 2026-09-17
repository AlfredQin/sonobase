"""Analysis B1 — CAMUS per-structure mIoU/Dice.

A trivial filter on Stage-1 CAMUS outputs: read the per-sample CSV,
group rows by ``category_id`` (sourced from CAMUS's SaUS
`masklet_category_id` array at Stage 1), and report mean IoU/Dice per
structure per model.

The CAMUS SaUS dataset declares 3 categories per its `dataset_info.json`:
    1 = endocardium
    2 = epicardium
    3 = atrium_wall

Output: a 3-row × N-model table for the paper.
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from nemo_cv.components.analysis import prediction_io as pio
from nemo_cv.components.analysis import record_io
from nemo_cv.components.analysis.plots.style import display_name
from nemo_cv.components.analysis.plots.summary_table import summary_table_figure

logger = logging.getLogger(__name__)


def _per_structure_metrics(
    run_dir: pathlib.Path, label: str = "", kept: Optional[List[Dict]] = None,
) -> Tuple[Dict[int, Dict[str, float]], Dict[int, str]]:
    """Group per-sample rows by `category_id`; also return a {cat_id: name} map.

    Skips rows tagged `no_gt_this_frame` (don't reward predicting nothing on
    un-annotated frames). Errors loudly if no row carries a `category_id` —
    means Stage 1 was run pre-schema-update or the dataset has no categories.

    Every row that actually enters a mean is appended to ``kept`` (when given),
    so the report can ship the rows it was computed from. Dumping the rows
    *after* this filtering is the point: a checker then reproduces each stored
    mean exactly, instead of having to re-derive which rows were dropped.
    """
    rows = pio.read_per_sample_csv(run_dir)
    by_cat: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    names: Dict[int, str] = {}
    n_total = 0
    n_with_cat = 0
    for r in rows:
        if (r.get("notes") or "") == "no_gt_this_frame":
            continue
        n_total += 1
        raw_cat = r.get("category_id")
        if raw_cat in (None, "", "None"):
            continue
        cat_id = int(raw_cat)
        n_with_cat += 1
        by_cat[cat_id]["iou"].append(float(r["iou"]))
        by_cat[cat_id]["dice"].append(float(r["dice"]))
        nm = r.get("category_name")
        if kept is not None:
            kept.append({
                "model": label,
                "sample_id": r.get("sample_id", ""),
                "frame_idx": r.get("frame_idx", ""),
                "obj_id": r.get("obj_id", ""),
                "category_id": cat_id,
                "category_name": nm or "",
                "iou": float(r["iou"]),
                "dice": float(r["dice"]),
            })
        if nm and cat_id not in names:
            names[cat_id] = str(nm)

    if n_total > 0 and n_with_cat == 0:
        raise RuntimeError(
            f"B1: no rows in {run_dir} carry a category_id. Re-run Stage 1 "
            f"with the updated schema (PerSampleRecord must include "
            f"category_id / category_name) before running this analysis."
        )

    out: Dict[int, Dict[str, float]] = {}
    for cid, m in by_cat.items():
        out[cid] = {
            "n": int(len(m["iou"])),
            "mean_iou": float(np.mean(m["iou"])) if m["iou"] else float("nan"),
            "mean_dice": float(np.mean(m["dice"])) if m["dice"] else float("nan"),
        }
    return out, names


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    p = argparse.ArgumentParser(description="CAMUS per-structure mIoU/Dice (B1).")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    out_dir = pathlib.Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for s in args.runs:
        if "=" not in s:
            raise ValueError(f"--runs entry must be `label=path`, got: {s!r}")
        label, path = s.split("=", 1)
        runs.append((label.strip(), pathlib.Path(path.strip())))

    per_model: Dict[str, Dict[int, Dict[str, float]]] = {}
    name_map: Dict[int, str] = {}
    kept: List[Dict] = []
    for label, run_dir in runs:
        m, names = _per_structure_metrics(run_dir, label, kept)
        per_model[label] = m
        for cid, nm in names.items():
            name_map.setdefault(cid, nm)
        for cid in sorted(m.keys()):
            met = m[cid]
            label_name = name_map.get(cid, f"cat_{cid}")
            logger.info(
                f"[{label}] {label_name} (cat={cid}): "
                f"n={met['n']} IoU={met['mean_iou']:.3f} Dice={met['mean_dice']:.3f}"
            )

    # 3-row × N-model table per metric (one each for IoU and Dice)
    cat_ids = sorted({cid for m in per_model.values() for cid in m})
    headers = ["Structure"] + [display_name(lbl) for lbl, _ in runs]

    for metric, decimals in [("mean_iou", 3), ("mean_dice", 3)]:
        rows = [headers]
        for cid in cat_ids:
            rows.append([name_map.get(cid, f"cat_{cid}")] + [
                ("n/a" if not np.isfinite(v := per_model.get(label, {}).get(cid, {}).get(metric, float("nan")))
                 else f"{v:.{decimals}f}")
                for label, _ in runs
            ])
        summary_table_figure(
            rows, out_dir / f"summary_table_{metric}.pdf",
            title=f"CAMUS per-structure {metric.replace('mean_', '').capitalize()} — B1",
        )

    record_io.write_dump(out_dir, kept)

    with (out_dir / "analysis_report.json").open("w") as f:
        json.dump({
            "analysis": "B1",
            "title": "CAMUS per-structure mIoU/Dice",
            "spec_section": "B1",
            "grouping_key": "category_id",
            "categories": name_map,
            "inputs": {label: str(rd) for label, rd in runs},
            "models": per_model,
            "verification": record_io.declare("per_sample.csv", [
                record_io.check(["models", "$model", "$category_id", "mean_iou"], "iou"),
                record_io.check(["models", "$model", "$category_id", "mean_dice"], "dice"),
                record_io.check(["models", "$model", "$category_id", "n"], "iou", agg="count"),
            ]),
        }, f, indent=2)

    logger.info(f"B1 analysis complete. Outputs at: {out_dir}")


if __name__ == "__main__":
    main()
