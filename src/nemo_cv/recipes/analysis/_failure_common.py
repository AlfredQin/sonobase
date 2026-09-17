"""Shared helpers for B2 (catastrophic-failure catalog) and T3.2 (SonoBase
failure analysis).

Both analyses:
  1. Walk one Stage-1 prediction directory per model (sharing dataset).
  2. Cross-join per (sample_id, frame_idx, obj_id) to get per-row per-model
     IoU on the same item.
  3. Apply an analysis-specific predicate to identify "interesting" rows.
  4. Save the worst-K rows and a side-by-side qualitative figure.
"""

from __future__ import annotations

import json
import logging
import pathlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from nemo_cv.components.analysis import prediction_io as pio
from nemo_cv.components.analysis.plots.failure_catalog import failure_catalog_figure

logger = logging.getLogger(__name__)


@dataclass
class _RowKey:
    dataset: str
    sample_id: str
    frame_idx: int
    obj_id: int


@dataclass
class _CrossJoinedRow:
    key: _RowKey
    ious: Dict[str, float]                # model_label → IoU on this item
    pred_paths: Dict[str, pathlib.Path]   # model_label → absolute pred mask path
    gt_path: Optional[pathlib.Path] = None
    image_path: Optional[pathlib.Path] = None    # only known for image datasets


def cross_join(
    runs: Dict[str, pathlib.Path],   # model_label → run_dir
    dataset_filter: Optional[str] = None,
) -> List[_CrossJoinedRow]:
    """Read each run's per_sample_metrics.csv and join by (dataset, sample_id, frame_idx, obj_id).

    Only rows present in ALL runs are kept (paired comparison).

    The image_path is recovered from the manifest's `dataset.saus_dir` (no
    image_path is in the per-sample CSV — this is OK because image overlays
    are needed only for the qualitative figure step at the end).
    """
    per_run_rows: Dict[str, Dict[Tuple, dict]] = {}
    saus_dirs: Dict[str, str] = {}

    for label, run_dir in runs.items():
        manifest = pio.read_manifest(run_dir)
        saus_dirs[manifest.get("dataset", {}).get("name", "")] = manifest.get("dataset", {}).get("saus_dir", "")
        rows = pio.read_per_sample_csv(run_dir)
        per_row: Dict[Tuple, dict] = {}
        for r in rows:
            if dataset_filter is not None and r["dataset"] != dataset_filter:
                continue
            key = (r["dataset"], r["sample_id"], int(r["frame_idx"]), int(r["obj_id"]))
            per_row[key] = r
        per_run_rows[label] = per_row

    common_keys = set.intersection(*(set(d.keys()) for d in per_run_rows.values()))

    out: List[_CrossJoinedRow] = []
    for key in common_keys:
        # Find a run whose (gt_mask_path, run_dir) pair points at an actual
        # GT file on disk. Stage 1 today writes the GT mask alongside each
        # prediction in every run dir, so the first match wins; iterating
        # explicitly avoids the silent dict-order dependency that would
        # bite if a future refactor saves GT once per dataset OR varies
        # the `gt_mask_path` value across runs.
        gt_path = None
        gt_rel = None
        for label, run_dir in runs.items():
            candidate_rel = per_run_rows[label][key]["gt_mask_path"]
            candidate = run_dir / candidate_rel
            if candidate.is_file():
                gt_path = candidate
                gt_rel = candidate_rel
                break
        if gt_path is None:
            raise FileNotFoundError(
                f"GT mask not found in any run dir for key={key}; "
                f"searched: {[str(rd) for rd in runs.values()]}"
            )
        ious = {label: float(per_run_rows[label][key]["iou"]) for label in runs}
        pred_paths = {
            label: runs[label] / per_run_rows[label][key]["pred_mask_path"]
            for label in runs
        }
        # Recover the original image path from the SaUS dir
        ds_name, sid, f_idx, _ = key
        saus_dir = saus_dirs.get(ds_name)
        image_path = _resolve_image_path(saus_dir, ds_name, sid, f_idx)

        out.append(_CrossJoinedRow(
            key=_RowKey(*key),
            ious=ious,
            pred_paths=pred_paths,
            gt_path=gt_path,
            image_path=image_path,
        ))
    return out


def _resolve_image_path(saus_dir: Optional[str], ds_name: str, sample_id: str,
                        frame_idx: int) -> Optional[pathlib.Path]:
    """Recover the input image path for a row (for qualitative figures).

    Image datasets store one .jpg per sample; video datasets have a directory
    per video with `<frame:05d>.jpg`.
    """
    if not saus_dir:
        return None
    base = pathlib.Path(saus_dir).expanduser()
    # Try video layout first
    video_dir = base / "images" / sample_id
    if video_dir.is_dir():
        return video_dir / f"{frame_idx:05d}.jpg"
    # Image layout fallback
    for ext in (".jpg", ".png", ".jpeg"):
        cand = base / "images" / f"{sample_id}{ext}"
        if cand.is_file():
            return cand
    return None


def select_worst_rows(
    rows: List[_CrossJoinedRow],
    predicate: Callable[[Dict[str, float]], bool],
    sort_key: Callable[[Dict[str, float]], float],
    top_k: int,
) -> List[_CrossJoinedRow]:
    """Keep rows matching `predicate`, sorted ascending by `sort_key`, take top-K."""
    matching = [r for r in rows if predicate(r.ious)]
    matching.sort(key=lambda r: sort_key(r.ious))
    return matching[:top_k]


def render_catalog_pages(
    rows: List[_CrossJoinedRow],
    model_order: List[str],
    output_dir: pathlib.Path,
    *,
    title_prefix: str = "",
    samples_per_page: int = 6,
) -> List[pathlib.Path]:
    """Save N qualitative pages with `samples_per_page` rows each."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: List[pathlib.Path] = []

    for page_idx in range(0, len(rows), samples_per_page):
        page = rows[page_idx:page_idx + samples_per_page]
        samples = []
        for row in page:
            try:
                image = (np.array(Image.open(row.image_path).convert("RGB"))
                         if row.image_path and row.image_path.is_file()
                         else _placeholder(row.gt_path))
                gt = (np.array(Image.open(row.gt_path).convert("L")) > 0
                      if row.gt_path and row.gt_path.is_file() else np.zeros(image.shape[:2], dtype=bool))
                preds = {}
                for label in model_order:
                    p = row.pred_paths.get(label)
                    if p and p.is_file():
                        preds[label] = np.array(Image.open(p).convert("L")) > 0
            except Exception as e:
                logger.warning(f"Failed to load row {row.key}: {e}")
                continue
            samples.append({
                "image": image, "gt": gt, "preds": preds,
                "ious": row.ious,
                "title": f"{row.key.dataset}/{row.key.sample_id}\nf{row.key.frame_idx} obj{row.key.obj_id}",
            })
        if not samples:
            continue
        out = output_dir / f"failures_page_{page_idx // samples_per_page + 1:02d}.pdf"
        failure_catalog_figure(samples, model_order, out,
                               title=f"{title_prefix}page {page_idx // samples_per_page + 1}")
        saved.append(out)
        logger.info(f"Wrote {out}")
    return saved


def _placeholder(reference: Optional[pathlib.Path]) -> np.ndarray:
    """Generate an all-black RGB image matching the GT mask shape (fallback)."""
    if reference and reference.is_file():
        h, w = np.array(Image.open(reference).convert("L")).shape
    else:
        h, w = 256, 256
    return np.zeros((h, w, 3), dtype=np.uint8)


def write_catalog_json(rows: List[_CrossJoinedRow], output_path: pathlib.Path,
                       analysis: str, title: str, **extra) -> None:
    payload = {
        "analysis": analysis,
        "title": title,
        "n_rows": len(rows),
        **extra,
        "rows": [
            {
                "dataset": r.key.dataset,
                "sample_id": r.key.sample_id,
                "frame_idx": r.key.frame_idx,
                "obj_id": r.key.obj_id,
                "ious": r.ious,
            }
            for r in rows
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Wrote {output_path}")
