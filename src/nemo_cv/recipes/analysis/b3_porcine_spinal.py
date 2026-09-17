"""Analysis B3 — Porcine spinal cord per-class mIoU/Dice.

Stage-2 analysis. The PorcineSpinalCord SaUS dataset has **9 semantic
classes** (per its `dataset_info.json`):

    1 = dura                  6 = hematoma           (pathology)
    2 = pia                   7 = ventral_space
    3 = csf                   8 = dura_ventral_complex
    4 = spinal_cord           9 = dura_pia_complex
    5 = dorsal_space

Each Stage-1 sample has one or more objects (one per visible class).
**Group rows by `category_id`**, not `obj_id` — the SaUS `obj_idx`
within a sample doesn't reliably correspond to a fixed canonical
class when classes are sparsely present.

Categories and names are written into `per_sample_metrics.csv` at
Stage 1 (sourced from each SaUS COCO annotation's `category_id` /
`category_name` fields); this analysis reads them directly and does
not need to consult `dataset_info.json`.
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


def _per_class_metrics(
    run_dir: pathlib.Path, label: str = "", kept: Optional[List[Dict]] = None,
) -> Tuple[Dict[int, Dict[str, float]], Dict[int, str]]:
    """Group per-sample rows by `category_id`; collect a {cat_id: name} map.

    Returns ``(by_cat, names)``:
      - by_cat: ``{category_id: {"n": int, "mean_iou": float, "mean_dice": float}}``
      - names:  ``{category_id: category_name}`` (first non-null seen per id)

    Errors loudly if no row carries a `category_id` — that means Stage 1
    was run before this column was added to the schema, or the dataset
    lacks per-object categories. Either case is a data issue, not a
    silent-fallback case.
    """
    rows = pio.read_per_sample_csv(run_dir)
    by_cat: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    names: Dict[int, str] = {}
    n_total = 0
    n_with_cat = 0
    for r in rows:
        # Skip frames where the GT object isn't visible. Stage 1 still writes
        # a row for these (with notes='no_gt_this_frame', iou/dice=1.0 by
        # pio.compute_iou's "(empty, empty) = perfect" convention) so that
        # T3.1 temporal consistency sees every frame. For per-class means
        # those iou=1.0 rows would inflate sparse-class results — drop them.
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
        # Dump the post-filter rows — the ones that actually entered a mean.
        # The `no_gt_this_frame` rows dropped above would otherwise look like
        # unexplained discrepancies to anything recomputing these numbers.
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
            f"B3: no rows in {run_dir} carry a category_id. Re-run Stage 1 "
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
    p = argparse.ArgumentParser(description="Porcine spinal cord per-class mIoU (B3).")
    p.add_argument("--runs", nargs="+", required=True,
                   help="label=run_dir entries (one per model under comparison).")
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
        m, names = _per_class_metrics(run_dir, label, kept)
        per_model[label] = m
        # Merge name maps across runs (all runs of the same dataset should
        # agree, but use first-seen to be tolerant).
        for cid, nm in names.items():
            name_map.setdefault(cid, nm)

    logger.info(f"Categories observed across runs: {dict(sorted(name_map.items()))}")

    cat_ids = sorted({cid for m in per_model.values() for cid in m})
    headers = ["Class (category_id)"] + [display_name(lbl) for lbl, _ in runs]

    def _row_label(cid: int) -> str:
        return f"{name_map.get(cid, f'cat_{cid}')} ({cid})"

    for metric in ("mean_iou", "mean_dice"):
        rows = [headers]
        for cid in cat_ids:
            row = [_row_label(cid)]
            for label, _ in runs:
                v = per_model.get(label, {}).get(cid, {}).get(metric, float("nan"))
                row.append("n/a" if not np.isfinite(v) else f"{v:.3f}")
            rows.append(row)
        summary_table_figure(
            rows, out_dir / f"summary_table_{metric}.pdf",
            title=f"Porcine spinal cord — per-class {metric.replace('mean_', '').capitalize()} (B3)",
        )

    record_io.write_dump(out_dir, kept)

    with (out_dir / "analysis_report.json").open("w") as f:
        json.dump({
            "analysis": "B3",
            "title": "Porcine spinal cord per-class mIoU/Dice",
            "spec_section": "B3",
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

    for label, m in per_model.items():
        for cid, met in sorted(m.items()):
            nm = name_map.get(cid, f"cat_{cid}")
            logger.info(
                f"[{label}] {nm} (cat={cid}): "
                f"n={met['n']} IoU={met['mean_iou']:.3f} Dice={met['mean_dice']:.3f}"
            )

    logger.info(f"B3 analysis complete. Outputs at: {out_dir}")


if __name__ == "__main__":
    main()
