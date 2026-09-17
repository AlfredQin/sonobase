"""Analysis T3.1 — Temporal consistency (CAMUS only).

Per video sequence (per patient × per view × per object), compute the
inter-frame IoU between consecutive predicted masks and count "breaks"
(IoU < 0.5 between consecutive frames — a sign that the model lost track
of the structure). For EF, breaks between ED and ES corrupt the volume
estimate.

Outputs per model: mean per-video temporal consistency, total breaks,
breaks-per-sequence histogram values.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import pathlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from PIL import Image

from nemo_cv.components.analysis import prediction_io as pio
from nemo_cv.components.analysis import record_io
from nemo_cv.components.analysis.plots.style import display_name
from nemo_cv.components.analysis.plots.summary_table import summary_table_figure

logger = logging.getLogger(__name__)


BREAK_IOU_THRESHOLD = 0.5


def compute_temporal_consistency(frame_masks: List[np.ndarray]
                                 ) -> tuple:
    """Compute (mean_inter_frame_IoU, n_breaks, per_pair_IoUs).

    """
    per_pair: List[float] = []
    n_breaks = 0
    for i in range(len(frame_masks) - 1):
        m_t = frame_masks[i].astype(bool)
        m_t1 = frame_masks[i + 1].astype(bool)
        inter = np.logical_and(m_t, m_t1).sum()
        union = np.logical_or(m_t, m_t1).sum()
        iou = 1.0 if union == 0 else float(inter / union)
        per_pair.append(iou)
        if iou < BREAK_IOU_THRESHOLD:
            n_breaks += 1
    if not per_pair:
        return float("nan"), 0, []
    return float(np.mean(per_pair)), n_breaks, per_pair


@dataclass
class _SeqResult:
    sample_id: str
    obj_id: int
    n_frames: int
    mean_iou: float
    n_breaks: int


def _seq_results_for_run(run_dir: pathlib.Path) -> List[_SeqResult]:
    """Build per-(video, obj) consistency stats by streaming the run's CSV."""
    rows = pio.read_per_sample_csv(run_dir)
    # Group rows by (sample_id, obj_id) → ordered list of pred_mask_paths
    grouped: Dict[tuple, List[dict]] = defaultdict(list)
    for r in rows:
        grouped[(r["sample_id"], int(r["obj_id"]))].append(r)

    results: List[_SeqResult] = []
    for (sid, obj_id), seq_rows in grouped.items():
        seq_rows.sort(key=lambda r: int(r["frame_idx"]))
        # Skip frames with notes='no_gt_this_frame' is not valid here — for
        # temporal consistency we explicitly want every consecutive pair.
        masks: List[np.ndarray] = []
        for r in seq_rows:
            p = run_dir / r["pred_mask_path"]
            if not p.is_file():
                continue
            masks.append(np.array(Image.open(p).convert("L")) > 0)
        if len(masks) < 2:
            continue
        mean_iou, n_breaks, _ = compute_temporal_consistency(masks)
        results.append(_SeqResult(
            sample_id=sid, obj_id=obj_id,
            n_frames=len(masks),
            mean_iou=mean_iou, n_breaks=n_breaks,
        ))
    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    p = argparse.ArgumentParser(description="Temporal consistency (T3.1).")
    p.add_argument("--runs", nargs="+", required=True,
                   help="`label=path` Stage-1 prediction dirs (CAMUS).")
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

    summaries: Dict[str, Dict] = {}
    per_sequence: Dict[str, List[_SeqResult]] = {}

    for label, run_dir in runs:
        seq_results = _seq_results_for_run(run_dir)
        per_sequence[label] = seq_results
        if not seq_results:
            summaries[label] = {"n_sequences": 0}
            continue
        ious = np.array([s.mean_iou for s in seq_results], dtype=float)
        breaks = np.array([s.n_breaks for s in seq_results], dtype=int)
        summaries[label] = {
            "n_sequences": int(len(seq_results)),
            "mean_inter_frame_iou": float(np.mean(ious)),
            "median_inter_frame_iou": float(np.median(ious)),
            "total_breaks": int(breaks.sum()),
            "mean_breaks_per_seq": float(breaks.mean()),
            "p50_breaks_per_seq": float(np.median(breaks)),
            "p90_breaks_per_seq": float(np.percentile(breaks, 90)),
            "n_seq_with_zero_breaks": int((breaks == 0).sum()),
        }
        logger.info(
            f"[{label}] n={len(seq_results)} mean_consistency={summaries[label]['mean_inter_frame_iou']:.3f} "
            f"total_breaks={summaries[label]['total_breaks']} "
            f"zero_breaks={summaries[label]['n_seq_with_zero_breaks']}"
        )

    # Per-sequence CSV (long format)
    csv_path = out_dir / "per_sequence.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "sample_id", "obj_id", "n_frames",
                    "mean_inter_frame_iou", "n_breaks"])
        for label, seqs in per_sequence.items():
            for s in seqs:
                w.writerow([label, s.sample_id, s.obj_id, s.n_frames,
                            f"{s.mean_iou:.4f}", s.n_breaks])
    logger.info(f"Wrote {csv_path}")

    # Summary table figure
    headers = ["Model", "n", "Mean inter-frame IoU",
               "Total breaks", "Mean / 90-pct breaks per seq", "Sequences w/ 0 breaks"]
    rows = [headers]
    for label, s in summaries.items():
        if s["n_sequences"] == 0:
            rows.append([display_name(label), "0", "n/a", "n/a", "n/a", "n/a"])
            continue
        rows.append([
            display_name(label), f"{s['n_sequences']:d}",
            f"{s['mean_inter_frame_iou']:.3f}",
            f"{s['total_breaks']:d}",
            f"{s['mean_breaks_per_seq']:.2f} / {s['p90_breaks_per_seq']:.0f}",
            f"{s['n_seq_with_zero_breaks']:d}",
        ])
    summary_table_figure(rows, out_dir / "summary_table.pdf",
                         title="CAMUS Temporal Consistency — T3.1 summary")

    with (out_dir / "analysis_report.json").open("w") as f:
        json.dump({
            "analysis": "T3.1",
            "title": "Temporal consistency (CAMUS sequences)",
            "spec_section": "T3.1",
            "break_iou_threshold": BREAK_IOU_THRESHOLD,
            "inputs": {label: str(rd) for label, rd in runs},
            "models": summaries,
            # The per-sequence dump was always written; nothing declared it as
            # the evidence for these numbers, so a checker looking only for a
            # file named per_sample.csv skipped this report entirely.
            "verification": record_io.declare("per_sequence.csv", [
                record_io.check(["models", "$model", "mean_inter_frame_iou"],
                                "mean_inter_frame_iou"),
                record_io.check(["models", "$model", "n_sequences"],
                                "mean_inter_frame_iou", agg="count"),
                record_io.check(["models", "$model", "total_breaks"],
                                "n_breaks", agg="sum"),
            ]),
        }, f, indent=2)

    logger.info(f"T3.1 analysis complete. Outputs at: {out_dir}")


if __name__ == "__main__":
    main()
