"""Shared helpers for the subgroup-stratification analyses (S1, S2, S3).

All three analyses share the same shape:
  1. Read per-image IoU/Dice from 6 Stage-1 runs (3 models × 2 prompts).
  2. Group sample_ids into subgroups (manufacturer / quality / pathology).
  3. Per (subgroup × model × prompt): mean per-image IoU / Dice over
     intersection of run sample_ids and subgroup sample_ids.
  4. (Optional) paired Wilcoxon SonoBase vs each baseline within each subgroup.
  5. Render the supplement table (12 metric columns + N + Wilcoxon p-values).
"""

from __future__ import annotations

import csv
import json
import logging
import math
import pathlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from nemo_cv.components.analysis import prediction_io as pio
from nemo_cv.components.analysis import record_io
from nemo_cv.components.analysis.plots.style import display_name
from nemo_cv.components.analysis.plots.summary_table import summary_table_figure
from nemo_cv.components.analysis.stats.tests import paired_wilcoxon

logger = logging.getLogger(__name__)


MODEL_ORDER = ("sonobase", "medsam2", "sam2_no_ft")
PROMPT_ORDER = ("point", "box")
TARGET_MODEL = "sonobase"
BASELINE_MODELS = ("medsam2", "sam2_no_ft")


# ---------------------------------------------------------------------------
# Per-(model, prompt) prediction loading
# ---------------------------------------------------------------------------


@dataclass
class _RunInputs:
    """User-supplied: where to read predictions from for each (model, prompt)."""
    runs: Dict[Tuple[str, str], pathlib.Path]   # {(model_label, prompt): predictions_dir}


def _per_image_iou_dice(run_dir: pathlib.Path
                        ) -> Dict[str, Tuple[float, float]]:
    """Average per-image IoU/Dice across all (frame, obj) rows for that image.

    Returns ``{sample_id: (mean_iou, mean_dice)}``. Skips frames marked
    `notes='no_gt_this_frame'` (video runs save those even when there's no
    GT on that frame).
    """
    rows = pio.read_per_sample_csv(run_dir)
    by_image_iou: Dict[str, List[float]] = defaultdict(list)
    by_image_dice: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        if (r.get("notes") or "") == "no_gt_this_frame":
            continue
        sid = r["sample_id"]
        by_image_iou[sid].append(float(r["iou"]))
        by_image_dice[sid].append(float(r["dice"]))
    return {
        sid: (float(np.mean(by_image_iou[sid])), float(np.mean(by_image_dice[sid])))
        for sid in by_image_iou
    }


# ---------------------------------------------------------------------------
# Per-subgroup × model × prompt aggregation
# ---------------------------------------------------------------------------


@dataclass
class _CellMetric:
    n: int = 0
    mean_iou: float = float("nan")
    mean_dice: float = float("nan")


@dataclass
class _CellWilcoxon:
    """Per-subgroup Wilcoxon for SonoBase vs one baseline at one prompt."""
    n_paired: int
    raw_p: float
    statistic: float


def _cell_for(scores: Dict[str, Tuple[float, float]],
              ids: Set[str]) -> _CellMetric:
    inter = ids & set(scores)
    if not inter:
        return _CellMetric()
    ious = np.array([scores[s][0] for s in inter])
    dices = np.array([scores[s][1] for s in inter])
    return _CellMetric(
        n=len(inter),
        mean_iou=float(np.mean(ious)),
        mean_dice=float(np.mean(dices)),
    )


def aggregate_per_subgroup(
    scores: Dict[Tuple[str, str], Dict[str, Tuple[float, float]]],
    subgroups: Dict[str, Set[str]],
) -> Dict[str, Dict[Tuple[str, str], _CellMetric]]:
    """Build {subgroup_name: {(model, prompt): _CellMetric}}.

    `scores` maps (model, prompt) → per-image (iou, dice) dict.
    `subgroups` maps subgroup_name → set of sample ids.
    """
    out: Dict[str, Dict[Tuple[str, str], _CellMetric]] = {}
    for sg_name, sg_ids in subgroups.items():
        out[sg_name] = {
            (m, p): _cell_for(scores.get((m, p), {}), sg_ids)
            for m in MODEL_ORDER for p in PROMPT_ORDER
        }
    return out


def per_image_rows(
    scores: Dict[Tuple[str, str], Dict[str, Tuple[float, float]]],
    subgroups: Dict[str, Set[str]],
) -> List[Dict]:
    """The per-image rows behind every cell, in long format.

    `per_subgroup.csv` holds the *aggregated* cells, so recomputing a cell mean
    from it only proves the CSV and the JSON were written from the same
    variable — it cannot catch a wrong aggregation. These rows are the actual
    inputs, so a checker recomputes each cell from the images that formed it.

    `cell` duplicates `model`/`prompt` joined the way the report keys its cells,
    so a checker can bind the report key directly instead of parsing it.
    """
    rows: List[Dict] = []
    for sg_name, sg_ids in sorted(subgroups.items()):
        for m in MODEL_ORDER:
            for p in PROMPT_ORDER:
                per_image = scores.get((m, p), {})
                for sid in sorted(sg_ids & set(per_image)):
                    iou, dice = per_image[sid]
                    rows.append({
                        "subgroup": sg_name,
                        "cell": f"{m}__{p}",
                        "model": m,
                        "prompt": p,
                        "sample_id": sid,
                        "iou": iou,
                        "dice": dice,
                    })
    return rows


def per_subgroup_wilcoxon(
    scores: Dict[Tuple[str, str], Dict[str, Tuple[float, float]]],
    subgroups: Dict[str, Set[str]],
) -> Dict[Tuple[str, str, str], _CellWilcoxon]:
    """Paired Wilcoxon SonoBase vs each baseline within each subgroup × prompt.

    Returns ``{(subgroup_name, baseline_model, prompt): _CellWilcoxon}``.
    Pairs by sample_id, intersected with the subgroup, intersected with
    both runs' sample_ids. Per-image IoU is the test statistic.
    """
    out: Dict[Tuple[str, str, str], _CellWilcoxon] = {}
    for sg_name, sg_ids in subgroups.items():
        for prompt in PROMPT_ORDER:
            target = scores.get((TARGET_MODEL, prompt), {})
            for baseline in BASELINE_MODELS:
                base = scores.get((baseline, prompt), {})
                paired = sorted(sg_ids & set(target) & set(base))
                if len(paired) < 2:
                    out[(sg_name, baseline, prompt)] = _CellWilcoxon(
                        n_paired=len(paired), raw_p=float("nan"), statistic=float("nan"),
                    )
                    continue
                x = np.array([target[s][0] for s in paired])
                y = np.array([base[s][0] for s in paired])
                stat, pval = paired_wilcoxon(x, y)
                out[(sg_name, baseline, prompt)] = _CellWilcoxon(
                    n_paired=len(paired), raw_p=float(pval), statistic=float(stat),
                )
    return out


# ---------------------------------------------------------------------------
# CSV / table writers
# ---------------------------------------------------------------------------


def _fmt_metric(v: float, digits: int = 3) -> str:
    return "n/a" if not math.isfinite(v) else f"{v:.{digits}f}"


def write_per_subgroup_csv(
    output_dir: pathlib.Path,
    cells: Dict[str, Dict[Tuple[str, str], _CellMetric]],
    subgroup_axis_name: str,
    wilcoxon: Optional[Dict[Tuple[str, str, str], _CellWilcoxon]] = None,
) -> pathlib.Path:
    """Long-format CSV: one row per (subgroup × model × prompt)."""
    headers = [subgroup_axis_name, "model", "prompt", "n",
               "mean_iou", "mean_dice"]
    if wilcoxon is not None:
        headers += ["raw_p_vs_target", "wilcoxon_W"]

    path = output_dir / "per_subgroup.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(headers)
        for sg_name, cell_map in cells.items():
            for m in MODEL_ORDER:
                for p in PROMPT_ORDER:
                    cell = cell_map[(m, p)]
                    row = [sg_name, m, p, cell.n,
                           _fmt_metric(cell.mean_iou),
                           _fmt_metric(cell.mean_dice)]
                    if wilcoxon is not None:
                        if m == TARGET_MODEL:
                            row += ["", ""]
                        else:
                            wcell = wilcoxon.get((sg_name, m, p))
                            row += [
                                "" if (wcell is None or not math.isfinite(wcell.raw_p)) else f"{wcell.raw_p:.6e}",
                                "" if (wcell is None or not math.isfinite(wcell.statistic)) else f"{wcell.statistic:.4f}",
                            ]
                    w.writerow(row)
    logger.info(f"Wrote {path}")
    return path


def render_summary_table(
    output_dir: pathlib.Path,
    cells: Dict[str, Dict[Tuple[str, str], _CellMetric]],
    subgroup_axis_name: str,
    metric: str = "iou",
    title_suffix: str = "",
) -> pathlib.Path:
    """Summary table: 1 column per (model × prompt) + N images + subgroup name."""
    headers = [subgroup_axis_name, "n images"]
    headers += [f"{display_name(m)} ({p})" for m in MODEL_ORDER for p in PROMPT_ORDER]
    rows = [headers]

    def _avg_n(ns_list):
        """Mean cell-n rounded to nearest int (per the per-subgroup docstring).

        Subgroups partition the test set, so every (m,p) cell *should* see the
        same subgroup-N. In practice some runs miss samples, so we average
        across (m,p) cells rather than report max — max would overstate the
        true population by the size of the most-complete cell. Drop zero
        entries from the mean (missing cells, not "subgroup is empty").
        """
        nonzero = [n for n in ns_list if n > 0]
        return int(round(sum(nonzero) / len(nonzero))) if nonzero else 0

    def _cell_metric(cell):
        return cell.mean_iou if metric == "iou" else cell.mean_dice

    # Per-subgroup row
    for sg_name, cell_map in cells.items():
        ns = [cell_map[(m, p)].n for m in MODEL_ORDER for p in PROMPT_ORDER]
        row = [sg_name, str(_avg_n(ns))]
        for m in MODEL_ORDER:
            for p in PROMPT_ORDER:
                row.append(_fmt_metric(_cell_metric(cell_map[(m, p)])))
        rows.append(row)

    # ALL row (= union of all subgroups for each cell)
    all_ids_per_cell: Dict[Tuple[str, str], int] = defaultdict(int)
    for cell_map in cells.values():
        for k, c in cell_map.items():
            all_ids_per_cell[k] += c.n   # sums per-cell ns; subgroups partition the test set
    all_row = ["ALL", str(_avg_n(list(all_ids_per_cell.values())))]
    # mean over subgroups (weighted by n) for the ALL row. Gate on the metric
    # currently being computed (Dice and IoU are jointly NaN in this codebase,
    # but pin the gate to the right field so a future change doesn't bite).
    for m in MODEL_ORDER:
        for p in PROMPT_ORDER:
            num = den = 0.0
            for cell_map in cells.values():
                cell = cell_map[(m, p)]
                v = _cell_metric(cell)
                if math.isfinite(v):
                    num += v * cell.n
                    den += cell.n
            all_row.append("n/a" if den == 0 else f"{num / den:.3f}")
    rows.append(all_row)

    path = output_dir / f"summary_table_{metric}.pdf"
    title = f"Subgroup × {' '.join(title_suffix.split())} ({metric.upper()})" if title_suffix else f"Subgroup ({metric.upper()})"
    summary_table_figure(rows, path, title=title)
    return path


def write_report(
    output_dir: pathlib.Path,
    analysis_id: str,
    title: str,
    spec_section: str,
    inputs: Dict[Tuple[str, str], pathlib.Path],
    subgroup_axis_name: str,
    cells: Dict[str, Dict[Tuple[str, str], _CellMetric]],
    wilcoxon: Optional[Dict[Tuple[str, str, str], _CellWilcoxon]] = None,
    subgroup_source: Optional[str] = None,
) -> pathlib.Path:
    """Machine-readable analysis_report.json. Same shape as the other analysis recipes."""
    cell_payload: Dict[str, Dict[str, Dict]] = {}
    for sg_name, cell_map in cells.items():
        cell_payload[sg_name] = {}
        for (m, p), cell in cell_map.items():
            cell_payload[sg_name][f"{m}__{p}"] = {
                "n": cell.n,
                "mean_iou": cell.mean_iou,
                "mean_dice": cell.mean_dice,
            }

    wilcoxon_payload: Dict[str, Dict[str, Dict]] = {}
    if wilcoxon is not None:
        for (sg, baseline, prompt), wcell in wilcoxon.items():
            wilcoxon_payload.setdefault(sg, {})[f"{baseline}__{prompt}"] = {
                "n_paired": wcell.n_paired,
                "raw_p": wcell.raw_p,
                "statistic": wcell.statistic,
            }

    report = {
        "analysis": analysis_id,
        "title": title,
        "spec_section": spec_section,
        "inputs": {f"{m}__{p}": str(d) for (m, p), d in inputs.items()},
        "subgroup_axis": subgroup_axis_name,
        # Which file the subgroup labels came from. Without it a cell cannot be
        # rechecked at all: the per-image scores are reproducible, but the
        # assignment of images to subgroups is not recoverable from the report.
        "subgroup_source": subgroup_source,
        "cells": cell_payload,
        "wilcoxon": wilcoxon_payload,
        # `wilcoxon` is not covered: p-values are not an aggregation of a
        # column, so a checker lists them as uncovered rather than assuming.
        "verification": record_io.declare("per_image.csv", [
            record_io.check(["cells", "$subgroup", "$cell", "mean_iou"], "iou"),
            record_io.check(["cells", "$subgroup", "$cell", "mean_dice"], "dice"),
            record_io.check(["cells", "$subgroup", "$cell", "n"], "iou", agg="count"),
        ]),
    }
    path = output_dir / "analysis_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Wrote {path}")
    return path


# ---------------------------------------------------------------------------
# Top-level helper that ties it all together
# ---------------------------------------------------------------------------


def run_subgroup_analysis(
    *,
    analysis_id: str,
    title: str,
    spec_section: str,
    runs: Dict[Tuple[str, str], pathlib.Path],
    subgroups: Dict[str, Set[str]],
    output_dir: pathlib.Path,
    subgroup_axis_name: str,
    subgroup_source: Optional[str] = None,
) -> None:
    """Drive the full S1/S2/S3 pipeline for one (analysis × dataset)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: load per-image scores per (model, prompt)
    scores: Dict[Tuple[str, str], Dict[str, Tuple[float, float]]] = {}
    for (model, prompt), run_dir in runs.items():
        if not run_dir.is_dir():
            logger.warning(f"Run dir missing: {run_dir} (model={model} prompt={prompt})")
            scores[(model, prompt)] = {}
            continue
        scores[(model, prompt)] = _per_image_iou_dice(run_dir)
        logger.info(f"  loaded {model}/{prompt}: {len(scores[(model, prompt)])} images "
                    f"from {run_dir.name}")

    # Step 2-3: aggregate per subgroup
    cells = aggregate_per_subgroup(scores, subgroups)

    # Step 4: per-subgroup Wilcoxon
    wilcoxon = per_subgroup_wilcoxon(scores, subgroups)

    # Step 5: write outputs
    write_report(output_dir, analysis_id, title, spec_section, runs,
                 subgroup_axis_name, cells, wilcoxon, subgroup_source)
    write_per_subgroup_csv(output_dir, cells, subgroup_axis_name, wilcoxon)
    record_io.write_dump(output_dir, per_image_rows(scores, subgroups),
                         filename="per_image.csv")
    render_summary_table(output_dir, cells, subgroup_axis_name,
                         metric="iou", title_suffix=title)
    render_summary_table(output_dir, cells, subgroup_axis_name,
                         metric="dice", title_suffix=title)
    logger.info(f"{analysis_id} analysis complete. Outputs at: {output_dir}")
