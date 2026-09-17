"""Build the per-dataset few-shot summary CSVs from Stage-1 prediction dirs.

Reads the on-disk artifacts produced by:
  * `nemo_cv.recipes.few_shot.finetune` — checkpoints under
    `experiments/few_shot/checkpoints/<exp>/` (we don't actually read these
    here; we rely on the per_sample_metrics.csv produced by the eval step).
  * `nemo_cv.recipes.analysis.save_predictions` — predictions under
    `experiments/few_shot/predictions/<exp>_<prompt>_0corr/`.
  * (ACOUSLIC only) ACOUSLIC GT CSV via the existing
    `metadata_loaders.acouslic` for AC measurement.
  * (ACOUSLIC only) `measurements.ellipse_fit` for AC mm.

Writes:
  * `<DATASET>_raw.csv`     one row per (model, N, seed, prompt, image)
                            — raw per-image numbers.
  * `<DATASET>_summary.csv` mean ± std across 3 seeds per (model, N, prompt),
                            plus paired Wilcoxon raw p-values for SonoBase
                            vs each baseline. FDR columns (`fdr_q_*`,
                            `sig_*`) start as empty; populated globally
                            by `apply_fdr_correction.py` once all 3
                            dataset summaries exist.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import pathlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from nemo_cv.components.analysis import prediction_io as pio
from nemo_cv.components.analysis.measurements.ellipse_fit import abdominal_circumference_mm
from nemo_cv.components.analysis.metadata_loaders.acouslic import load_acouslic_metadata
from nemo_cv.components.analysis.stats.tests import paired_wilcoxon, pearson_r

logger = logging.getLogger(__name__)


N_VALUES = (1, 2, 5, 10, 20, 30)
SEEDS = (42, 123, 456)
MODELS = ("sonobase", "medsam2", "sam2_no_ft")
PROMPTS = ("point", "box")
TARGET_MODEL = "sonobase"   # paired tests are SonoBase vs each baseline
BASELINE_MODELS = ("medsam2", "sam2_no_ft")


# ---------------------------------------------------------------------------
# Per-image extraction from Stage-1 outputs
# ---------------------------------------------------------------------------


def _run_dir(predictions_root: pathlib.Path, dataset: str, model: str,
             N: int, seed: int, prompt: str) -> pathlib.Path:
    return predictions_root / f"{dataset}_{model}_N{N}_seed{seed}_{prompt}_0corr"


def _per_image_iou_dice(run_dir: pathlib.Path) -> List[Tuple[str, int, int, float, float]]:
    """Return list of (sample_id, frame_idx, obj_id, iou, dice).

    Skips frames marked `notes='no_gt_this_frame'` (video runs save those
    even when there's no GT — we don't want them weighted in the average).
    """
    out = []
    rows = pio.read_per_sample_csv(run_dir)
    for r in rows:
        if (r.get("notes") or "") == "no_gt_this_frame":
            continue
        out.append((
            r["sample_id"], int(r["frame_idx"]), int(r["obj_id"]),
            float(r["iou"]), float(r["dice"]),
        ))
    return out


def _per_video_ac_mm(
    run_dir: pathlib.Path,
    pixel_spacing_mm_lookup: Dict[str, float],
    pixel_spacing_override: Optional[float],
) -> Dict[str, float]:
    """For ACOUSLIC: per-video predicted AC (mean across annotated frames).

    Mirrors the existing A4 analysis logic exactly so few-shot AC numbers
    are directly comparable with A4 (zero-shot) numbers.
    """
    rows = pio.read_per_sample_csv(run_dir)
    by_vid: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        if (r.get("notes") or "") == "no_gt_this_frame":
            continue
        sid = r["sample_id"]
        spacing = pixel_spacing_override if pixel_spacing_override else pixel_spacing_mm_lookup.get(sid)
        if spacing is None:
            continue
        pred_path = run_dir / r["pred_mask_path"]
        if not pred_path.is_file():
            continue
        pred_mask = np.array(Image.open(pred_path).convert("L")) > 0
        ac = abdominal_circumference_mm(pred_mask, spacing)
        if ac is not None:
            by_vid[sid].append(float(ac))
    return {sid: float(np.mean(acs)) for sid, acs in by_vid.items() if acs}


# ---------------------------------------------------------------------------
# Raw CSV (one row per image × model × N × seed × prompt)
# ---------------------------------------------------------------------------


def _write_raw_csv(
    out_dir: pathlib.Path, dataset: str,
    predictions_root: pathlib.Path,
    acouslic_meta: Optional[Dict],
    pixel_spacing_override: Optional[float],
) -> pathlib.Path:
    is_acouslic = (dataset == "ACOUSLIC")
    pixel_spacing_lookup = {}
    gt_ac_lookup = {}
    if is_acouslic and acouslic_meta:
        for sid, m in acouslic_meta.items():
            if m.pixel_spacing_mm is not None:
                pixel_spacing_lookup[sid] = float(m.pixel_spacing_mm)
            if m.gt_clinical_value is not None:
                gt_ac_lookup[sid] = float(m.gt_clinical_value)

    headers = ["dataset", "model", "N", "seed", "prompt",
               "sample_id", "frame_idx", "obj_id",
               "iou", "dice"]
    if is_acouslic:
        headers += ["pred_ac_mm", "gt_ac_mm", "ac_abs_err_mm"]

    path = out_dir / f"{dataset}_raw.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    with path.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(headers)
        for prompt in PROMPTS:
            for seed in SEEDS:
                for N in N_VALUES:
                    for model in MODELS:
                        run = _run_dir(predictions_root, dataset, model, N, seed, prompt)
                        if not run.is_dir():
                            continue
                        ious = _per_image_iou_dice(run)
                        per_video_ac = (
                            _per_video_ac_mm(run, pixel_spacing_lookup, pixel_spacing_override)
                            if is_acouslic else {}
                        )
                        for sid, fidx, oid, iou, dice in ious:
                            row = [dataset, model, N, seed, prompt,
                                   sid, fidx, oid, f"{iou:.6f}", f"{dice:.6f}"]
                            if is_acouslic:
                                pred_ac = per_video_ac.get(sid)
                                gt_ac = gt_ac_lookup.get(sid)
                                row += [
                                    "" if pred_ac is None else f"{pred_ac:.4f}",
                                    "" if gt_ac is None else f"{gt_ac:.4f}",
                                    ("" if (pred_ac is None or gt_ac is None)
                                     else f"{abs(pred_ac - gt_ac):.4f}"),
                                ]
                            w.writerow(row); n_rows += 1
    logger.info(f"Wrote {path} ({n_rows} rows)")
    return path


# ---------------------------------------------------------------------------
# Summary CSV (mean ± std per (model, N, prompt) + paired p-values)
# ---------------------------------------------------------------------------


@dataclass
class _ZeroShotRow:
    miou: float
    dice: float
    ac_mae: Optional[float] = None


def _zero_shot_lookup(
    zero_shot_csv: Optional[pathlib.Path],
    dataset: str,
) -> Dict[Tuple[str, str], _ZeroShotRow]:
    """Optionally read existing zero-shot per-(model, prompt) numbers for the N=0 rows.

    If absent, N=0 rows are emitted with empty values and a note so the
    paper figure logic can still know to plot them as "missing zero-shot".
    """
    if zero_shot_csv is None or not zero_shot_csv.is_file():
        return {}
    out: Dict[Tuple[str, str], _ZeroShotRow] = {}
    with zero_shot_csv.open() as f:
        for r in csv.DictReader(f):
            if r.get("dataset") != dataset:
                continue
            try:
                miou = float(r["mIoU"])
                dice = float(r["Dice"])
            except (KeyError, ValueError):
                continue
            ac_mae = None
            ac_str = r.get("AC_MAE_mm", "")
            if ac_str:
                try: ac_mae = float(ac_str)
                except ValueError: pass
            out[(r["model"], r.get("prompt", ""))] = _ZeroShotRow(miou=miou, dice=dice, ac_mae=ac_mae)
    return out


def _aggregate(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    """Return (mean, std). Returns (None, None) if the list is empty so the
    CSV writer can render an empty cell instead of the string 'nan'."""
    if not values:
        return None, None
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), (float(arr.std(ddof=1)) if arr.size > 1 else 0.0)


def _fmt(value: Optional[float], digits: int = 4) -> str:
    """Render a float as fixed-precision string; empty for None / NaN."""
    if value is None or not math.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def _per_seed_per_image_means(
    rows: List[Tuple[str, int, int, float, float]],
    metric_idx: int,    # 3 for iou, 4 for dice
) -> Dict[str, float]:
    """Average each (sample_id) over all its (frame, obj) rows. Used as the
    paired-test unit so different N values are comparable per image."""
    bucket: Dict[str, List[float]] = defaultdict(list)
    for sid, _, _, iou, dice in rows:
        v = iou if metric_idx == 3 else dice
        bucket[sid].append(float(v))
    return {sid: float(np.mean(vs)) for sid, vs in bucket.items() if vs}


def _wilcoxon_paired_by_image(
    target_by_seed: Dict[int, Dict[str, float]],
    baseline_by_seed: Dict[int, Dict[str, float]],
) -> Tuple[float, float]:
    """Average a per-(seed, sample_id) scalar across seeds for each model, pair by
    sample_id, and run the paired Wilcoxon.

    Metric-agnostic: the caller supplies per-image IoU (DDTI / FUGC) or per-video
    AC-MAE (ACOUSLIC) as ``{seed: {sample_id: scalar}}``. This is the test statistic
    for the dataset's *primary* endpoint: the p-value must match the metric
    reported as primary, not always IoU.
    """
    target_per_image: Dict[str, List[float]] = defaultdict(list)
    base_per_image: Dict[str, List[float]] = defaultdict(list)
    for _seed, means in target_by_seed.items():
        for sid, v in means.items():
            target_per_image[sid].append(v)
    for _seed, means in baseline_by_seed.items():
        for sid, v in means.items():
            base_per_image[sid].append(v)
    common = sorted(set(target_per_image) & set(base_per_image))
    if len(common) < 2:
        return float("nan"), float("nan")
    x = np.asarray([np.mean(target_per_image[s]) for s in common])
    y = np.asarray([np.mean(base_per_image[s]) for s in common])
    return paired_wilcoxon(x, y)


def _iou_by_seed(
    cache: Dict[Tuple[int, int, str, str], List],
    N: int,
    prompt: str,
    model: str,
) -> Dict[int, Dict[str, float]]:
    """Per-seed ``{sample_id -> mean per-image IoU}`` for one (N, prompt, model).

    The primary endpoint for DDTI / FUGC (and the secondary for ACOUSLIC).
    """
    return {
        seed: _per_seed_per_image_means(cache.get((N, seed, prompt, model), []), 3)
        for seed in SEEDS
    }


def _ac_err_by_seed(
    ac_per_run: Dict[Tuple[int, int, str, str], Dict[str, float]],
    gt_ac_lookup: Dict[str, float],
    N: int,
    prompt: str,
    model: str,
) -> Dict[int, Dict[str, float]]:
    """Per-seed ``{video_id -> |pred_AC - GT_AC|}`` for one (N, prompt, model).

    Reuses the exact AC-error expression that feeds the displayed ``AC_MAE_mean``
    column, so the Wilcoxon tests the same per-video numbers. ACOUSLIC's primary
    endpoint is AC-MAE, not IoU.
    """
    out: Dict[int, Dict[str, float]] = {}
    for seed in SEEDS:
        per_video = ac_per_run.get((N, seed, prompt, model), {})
        errs = {
            v: abs(per_video[v] - gt_ac_lookup[v])
            for v in per_video
            if v in gt_ac_lookup
        }
        if errs:
            out[seed] = errs
    return out


def _write_summary_csv(
    out_dir: pathlib.Path, dataset: str,
    predictions_root: pathlib.Path,
    acouslic_meta: Optional[Dict],
    pixel_spacing_override: Optional[float],
    zero_shot_csv: Optional[pathlib.Path],
) -> pathlib.Path:
    is_acouslic = (dataset == "ACOUSLIC")
    zs_lookup = _zero_shot_lookup(zero_shot_csv, dataset)

    headers = ["dataset", "model", "N", "prompt",
               "mIoU_mean", "mIoU_std", "Dice_mean", "Dice_std"]
    if is_acouslic:
        headers += ["AC_MAE_mean", "AC_MAE_std"]
    headers += ["raw_p_vs_medsam2", "raw_p_vs_sam2_no_ft",
                "fdr_q_vs_medsam2", "fdr_q_vs_sam2_no_ft",
                "sig_vs_medsam2", "sig_vs_sam2_no_ft"]

    # Pre-cache per-(N, seed, prompt, model) metrics to avoid re-reading
    # large prediction dirs N×S×M×P times.
    cache: Dict[Tuple[int, int, str, str], List] = {}
    for prompt in PROMPTS:
        for seed in SEEDS:
            for N in N_VALUES:
                for model in MODELS:
                    run = _run_dir(predictions_root, dataset, model, N, seed, prompt)
                    if run.is_dir():
                        cache[(N, seed, prompt, model)] = _per_image_iou_dice(run)

    # AC per-video lookup (ACOUSLIC only)
    pixel_spacing_lookup = {}
    gt_ac_lookup = {}
    if is_acouslic and acouslic_meta:
        for sid, m in acouslic_meta.items():
            if m.pixel_spacing_mm is not None:
                pixel_spacing_lookup[sid] = float(m.pixel_spacing_mm)
            if m.gt_clinical_value is not None:
                gt_ac_lookup[sid] = float(m.gt_clinical_value)

    ac_per_run: Dict[Tuple[int, int, str, str], Dict[str, float]] = {}
    if is_acouslic:
        for prompt in PROMPTS:
            for seed in SEEDS:
                for N in N_VALUES:
                    for model in MODELS:
                        run = _run_dir(predictions_root, dataset, model, N, seed, prompt)
                        if run.is_dir():
                            ac_per_run[(N, seed, prompt, model)] = _per_video_ac_mm(
                                run, pixel_spacing_lookup, pixel_spacing_override
                            )

    path = out_dir / f"{dataset}_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(headers)

        for prompt in PROMPTS:
            # ---- N=0 row from zero-shot ----
            for model in MODELS:
                zs = zs_lookup.get((model, prompt))
                row = [dataset, model, 0, prompt,
                       "" if zs is None else f"{zs.miou:.4f}", "0.0",
                       "" if zs is None else f"{zs.dice:.4f}", "0.0"]
                if is_acouslic:
                    row += ["" if (zs is None or zs.ac_mae is None) else f"{zs.ac_mae:.4f}", "0.0"]
                row += ["", "", "", "", "", ""]   # no p-values for N=0
                w.writerow(row)

            # ---- N=1..30 rows (mean ± std across seeds) ----
            for N in N_VALUES:
                # Per-seed scalar metrics: mean per-image IoU/Dice across the run.
                per_seed_iou: Dict[str, List[float]] = defaultdict(list)
                per_seed_dice: Dict[str, List[float]] = defaultdict(list)
                per_seed_ac_mae: Dict[str, List[float]] = defaultdict(list)
                for model in MODELS:
                    for seed in SEEDS:
                        rows = cache.get((N, seed, prompt, model), [])
                        if not rows:
                            continue
                        ious = [t[3] for t in rows]
                        dices = [t[4] for t in rows]
                        per_seed_iou[model].append(float(np.mean(ious)))
                        per_seed_dice[model].append(float(np.mean(dices)))
                        if is_acouslic:
                            per_video = ac_per_run.get((N, seed, prompt, model), {})
                            errs = [
                                abs(per_video[v] - gt_ac_lookup[v])
                                for v in per_video
                                if v in gt_ac_lookup
                            ]
                            if errs:
                                per_seed_ac_mae[model].append(float(np.mean(errs)))

                # Now write one row per model
                for model in MODELS:
                    miou_m, miou_s = _aggregate(per_seed_iou.get(model, []))
                    dice_m, dice_s = _aggregate(per_seed_dice.get(model, []))
                    row = [dataset, model, N, prompt,
                           _fmt(miou_m), _fmt(miou_s),
                           _fmt(dice_m), _fmt(dice_s)]
                    if is_acouslic:
                        ac_m, ac_s = _aggregate(per_seed_ac_mae.get(model, []))
                        row += [_fmt(ac_m), _fmt(ac_s)]

                    # Paired Wilcoxon (SonoBase vs each baseline) on the dataset's
                    # PRIMARY endpoint: per-video AC-MAE for ACOUSLIC, per-image IoU
                    # for DDTI / FUGC.
                    raw_p_strs: List[str] = ["", ""]
                    if model == TARGET_MODEL:
                        if is_acouslic:
                            target_by_seed = _ac_err_by_seed(ac_per_run, gt_ac_lookup, N, prompt, TARGET_MODEL)
                        else:
                            target_by_seed = _iou_by_seed(cache, N, prompt, TARGET_MODEL)
                        for j, baseline in enumerate(BASELINE_MODELS):
                            if is_acouslic:
                                base_by_seed = _ac_err_by_seed(ac_per_run, gt_ac_lookup, N, prompt, baseline)
                            else:
                                base_by_seed = _iou_by_seed(cache, N, prompt, baseline)
                            stat, pval = _wilcoxon_paired_by_image(target_by_seed, base_by_seed)
                            if math.isfinite(pval):
                                raw_p_strs[j] = f"{pval:.6e}"
                    row += [
                        raw_p_strs[0], raw_p_strs[1],
                        "", "",   # FDR q-values populated by apply_fdr_correction.py
                        "", "",   # sig_*
                    ]
                    w.writerow(row)

    logger.info(f"Wrote {path}")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    p = argparse.ArgumentParser(description="Few-shot per-dataset aggregator.")
    p.add_argument("--dataset", required=True, choices=("ACOUSLIC", "DDTI", "FUGC"))
    p.add_argument("--predictions-root",
                   default="./experiments/few_shot/predictions")
    p.add_argument("--output-dir",
                   default="./experiments/few_shot/few_shot_results")
    _acouslic_default = os.environ.get("ACOUSLIC_CSV")
    p.add_argument("--acouslic-csv", default=_acouslic_default,
                   required=(_acouslic_default is None),
                   help="Per-sweep GT AC CSV path (default: $ACOUSLIC_CSV env var; required if unset).")
    p.add_argument("--pixel-spacing-mm", type=float, default=None,
                   help="ACOUSLIC pixel-spacing override (sensitivity checks only; "
                        "default 0.28 mm/px from the MHA header).")
    p.add_argument("--zero-shot-csv", default=None,
                   help="Optional CSV with N=0 rows; columns: dataset, model, prompt, mIoU, Dice, AC_MAE_mm.")
    args = p.parse_args()

    out_dir = pathlib.Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_root = pathlib.Path(args.predictions_root).expanduser().resolve()

    acouslic_meta = None
    if args.dataset == "ACOUSLIC":
        acouslic_meta = load_acouslic_metadata(args.acouslic_csv) or None
        if acouslic_meta is None:
            logger.warning("ACOUSLIC GT CSV not found; AC columns will be empty in raw + summary.")

    _write_raw_csv(out_dir, args.dataset, pred_root, acouslic_meta, args.pixel_spacing_mm)
    _write_summary_csv(
        out_dir, args.dataset, pred_root, acouslic_meta,
        args.pixel_spacing_mm,
        pathlib.Path(args.zero_shot_csv) if args.zero_shot_csv else None,
    )
    logger.info(f"Aggregation complete for {args.dataset}.")


if __name__ == "__main__":
    main()
