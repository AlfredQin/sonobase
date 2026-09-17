"""Read/write utilities for Stage-1 prediction artifacts.

This module defines the on-disk layout and is the
**only** place that touches that layout — both Stage 1 (`save_predictions`)
and Stage 2 (per-analysis modules) call into here so that any future change
to the layout has exactly one place to update.

Per-(model × dataset × prompt) layout:

    <output_root>/
    ├── manifest.json                    run-level metadata
    ├── per_sample_metrics.csv           canonical join table — see column spec below
    └── <dataset>/<sample_id>/
        ├── meta.json                    per-sample metadata
        ├── pred_<frame>_obj_<obj>.png   binary mask PNG (255=fg)
        ├── gt_<frame>_obj_<obj>.png     GT mask PNG (mirrors pred for diffing)
        ├── overlay_<frame>_obj_<obj>.png   [optional] image + GT + pred + prompts
        └── prompts.json                 prompts used (in image-space coords)
"""

from __future__ import annotations

import csv
import json
import logging
import os
import pathlib
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


# Columns of `per_sample_metrics.csv` — the canonical Stage-2 join table.
# Order matters; downstream code uses `csv.DictWriter(fieldnames=PER_SAMPLE_FIELDS)`.
PER_SAMPLE_FIELDS: List[str] = [
    "dataset",
    "sample_id",
    "frame_idx",
    "obj_id",
    "category_id",            # nullable; semantic class id from SaUS dataset_info (1-indexed)
    "category_name",          # nullable; human-readable class name from SaUS dataset_info
    "iteration",              # nullable; populated only by A3 click-efficiency runs
    "model",
    "prompt_protocol",
    "iou",
    "dice",
    "pred_area_px",
    "gt_area_px",
    "pixel_spacing_mm",       # nullable
    "voxel_spacing_mm3",      # nullable
    "gt_clinical_value",      # nullable; populated by Stage-2 from metadata loaders
    "gt_clinical_unit",       # nullable; populated by Stage-2 from metadata loaders
    "pred_clinical_value",    # nullable; populated by Stage-2 analyses in-place
    "image_quality",          # nullable
    "pred_mask_path",         # relative to output_root
    "gt_mask_path",           # relative to output_root
    "notes",                  # freeform diagnostic
    # Scoring-grid ablation. Empty unless `score_square_grid` is on. The
    # same 256x256 logits are upsampled to a square 1024x1024 instead of to the
    # native frame, and the GT is resized to match -- which is the grid the
    # pretraining harness scores on. Appended last, and readers key by column
    # name, so existing records and consumers are unaffected.
    "iou_sq1024",             # nullable
    "dice_sq1024",            # nullable
]


@dataclass
class PerSampleRecord:
    """One row of `per_sample_metrics.csv`. All numeric fields are CSV-serializable."""

    dataset: str
    sample_id: str
    frame_idx: int
    obj_id: int
    model: str
    prompt_protocol: str
    iou: float
    dice: float
    pred_area_px: int
    gt_area_px: int
    pred_mask_path: str
    gt_mask_path: str
    category_id: Optional[int] = None
    category_name: Optional[str] = None
    iteration: Optional[int] = None
    pixel_spacing_mm: Optional[float] = None
    voxel_spacing_mm3: Optional[float] = None
    gt_clinical_value: Optional[float] = None
    gt_clinical_unit: Optional[str] = None
    pred_clinical_value: Optional[float] = None
    image_quality: Optional[str] = None
    notes: str = ""
    iou_sq1024: Optional[float] = None
    dice_sq1024: Optional[float] = None

    def to_csv_row(self) -> Dict[str, Any]:
        return {f: getattr(self, f) for f in PER_SAMPLE_FIELDS}


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def sample_dir(output_root: pathlib.Path, dataset: str, sample_id: str) -> pathlib.Path:
    return output_root / dataset / sample_id


def pred_mask_path(
    sample_dir: pathlib.Path,
    frame_idx: int,
    obj_id: int,
    iteration: Optional[int] = None,
) -> pathlib.Path:
    """Path of a per-sample predicted mask PNG.

    For A3 click-efficiency runs (`iteration is not None`) the file is
    suffixed with `_iter<N>` so multiple iterations can coexist in the
    same sample directory. For everything else the simpler name is used
    (preserves backwards compatibility with A2's existing artifacts).
    """
    if iteration is None:
        return sample_dir / f"pred_{frame_idx:05d}_obj_{obj_id}.png"
    return sample_dir / f"pred_{frame_idx:05d}_obj_{obj_id}_iter{iteration}.png"


def gt_mask_path(sample_dir: pathlib.Path, frame_idx: int, obj_id: int) -> pathlib.Path:
    return sample_dir / f"gt_{frame_idx:05d}_obj_{obj_id}.png"


def overlay_path(
    sample_dir: pathlib.Path,
    frame_idx: int,
    obj_id: int,
    iteration: Optional[int] = None,
) -> pathlib.Path:
    if iteration is None:
        return sample_dir / f"overlay_{frame_idx:05d}_obj_{obj_id}.png"
    return sample_dir / f"overlay_{frame_idx:05d}_obj_{obj_id}_iter{iteration}.png"


def meta_json_path(sample_dir: pathlib.Path) -> pathlib.Path:
    return sample_dir / "meta.json"


def prompts_json_path(sample_dir: pathlib.Path) -> pathlib.Path:
    return sample_dir / "prompts.json"


def manifest_path(output_root: pathlib.Path) -> pathlib.Path:
    return output_root / "manifest.json"


def per_sample_csv_path(output_root: pathlib.Path) -> pathlib.Path:
    return output_root / "per_sample_metrics.csv"


def run_signature_path(output_root: pathlib.Path, dataset: str) -> pathlib.Path:
    """Per-dataset run signature file: <output_root>/<dataset>/run_signature.json.

    Records which (model, prompt_protocol, ckpt) wrote into this directory so
    re-running with different knobs fails loudly instead of silently mixing.
    """
    return output_root / dataset / "run_signature.json"


def check_run_signature(
    output_root: pathlib.Path,
    dataset: str,
    signature: Dict[str, Any],
) -> None:
    """Verify (or initialise) the per-dataset run signature.

    Behaviour:
      - File absent  → write `signature` and return.
      - File present and equals `signature` → return.
      - File present and differs → raise RuntimeError with both signatures
        and a one-line resolver pointing at manual deletion.

    Signature contents are caller-defined; canonical keys are
    `model_label`, `prompt_protocol`, `ckpt_path`, `dataset`. Avoid keys
    whose value drifts run-to-run (timestamps, env vars) — they'd flap.
    """
    import json
    path = run_signature_path(output_root, dataset)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        with path.open("w") as f:
            json.dump(signature, f, indent=2, sort_keys=True)
        return

    with path.open() as f:
        existing = json.load(f)
    if existing == signature:
        return

    raise RuntimeError(
        f"Run signature mismatch at {path}.\n"
        f"  existing: {json.dumps(existing, sort_keys=True)}\n"
        f"  current : {json.dumps(signature, sort_keys=True)}\n"
        f"This output_dir was previously written with different "
        f"(model, prompt_protocol, ckpt) — mixing them would corrupt the "
        f"per_sample_metrics.csv columns. Either point this run at a fresh "
        f"output_dir, or delete the existing dataset directory ("
        f"{path.parent}) to start over deliberately."
    )


# ---------------------------------------------------------------------------
# Mask I/O
# ---------------------------------------------------------------------------


def save_mask_png(path: pathlib.Path, mask: np.ndarray) -> None:
    """Save a binary mask (any non-zero -> 255) as a single-channel PNG.

    The output is a uint8 image with values in {0, 255}. PNG is chosen over
    RLE because it's human-inspectable in any image viewer and its file size
    is acceptable for our scale.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (np.asarray(mask) > 0).astype(np.uint8) * 255
    Image.fromarray(arr, mode="L").save(path)


def load_mask_png(path: pathlib.Path) -> np.ndarray:
    """Load a PNG mask back to a `bool` array."""
    arr = np.array(Image.open(path).convert("L"))
    return arr > 0


def save_overlay_png(path: pathlib.Path, overlay: np.ndarray) -> None:
    """Save an RGB overlay image (uint8, [H, W, 3])."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay.astype(np.uint8)).save(path)


# ---------------------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------------------


def write_json(path: pathlib.Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write so a kill mid-flush doesn't leave a half-written file.
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(data, f, indent=2, default=_json_default)
    os.replace(tmp, path)


def read_json(path: pathlib.Path) -> Dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _json_default(o: Any) -> Any:
    """Serializer fallback for numpy types and pathlib objects."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, pathlib.PurePath):
        return str(o)
    raise TypeError(f"Object of type {type(o)} is not JSON-serializable")


# ---------------------------------------------------------------------------
# Per-sample CSV
# ---------------------------------------------------------------------------


def write_per_sample_csv(
    output_root: pathlib.Path, rows: Iterable[PerSampleRecord]
) -> pathlib.Path:
    """Write `per_sample_metrics.csv` from a list of records.

    Overwrites any existing file. For incremental/resumable runs, the recipe
    re-collects all rows from disk and re-writes the whole file at the end —
    this keeps the schema consistent and is fast at our scale.
    """
    path = per_sample_csv_path(output_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PER_SAMPLE_FIELDS)
        w.writeheader()
        for r in rows:
            row = r.to_csv_row()
            # CSV-friendly null rendering: empty string for None
            row = {k: ("" if v is None else v) for k, v in row.items()}
            w.writerow(row)
    return path


def read_per_sample_csv(output_root: pathlib.Path) -> List[Dict[str, Any]]:
    """Read `per_sample_metrics.csv` back into a list of dicts.

    Numeric columns are coerced to floats; null columns (empty string in CSV)
    become `None`. Stage-2 analyses convert further as needed (e.g. to a
    pandas DataFrame).
    """
    path = per_sample_csv_path(output_root)
    if not path.exists():
        raise FileNotFoundError(f"per_sample_metrics.csv not found: {path}")

    numeric_cols = {
        "frame_idx", "obj_id",
        "iou", "dice",
        "pred_area_px", "gt_area_px",
        "pixel_spacing_mm", "voxel_spacing_mm3",
        "gt_clinical_value", "pred_clinical_value",
    }
    int_cols = {"frame_idx", "obj_id", "pred_area_px", "gt_area_px"}

    out = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k in list(row.keys()):
                v = row[k]
                if k in numeric_cols:
                    row[k] = None if v == "" else (int(float(v)) if k in int_cols else float(v))
                elif v == "":
                    row[k] = None
            out.append(row)
    return out


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def write_manifest(
    output_root: pathlib.Path,
    *,
    run_name: str,
    model: Dict[str, Any],
    dataset: Dict[str, Any],
    prompt_protocol: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> pathlib.Path:
    """Write `manifest.json` for a Stage-1 run."""
    import socket
    import subprocess
    import time

    import torch

    def _git_hash() -> str:
        try:
            out = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            ).decode().strip()
            return out
        except Exception:
            return ""

    manifest = {
        "run_name": run_name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": socket.gethostname(),
        "git_hash": _git_hash(),
        "torch_version": str(torch.__version__),
        "model": model,
        "dataset": dataset,
        "prompt_protocol": prompt_protocol,
    }
    if extra:
        manifest.update(extra)

    path = manifest_path(output_root)
    write_json(path, manifest)
    return path


def read_manifest(output_root: pathlib.Path) -> Dict[str, Any]:
    return read_json(manifest_path(output_root))


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------


def is_sample_complete(output_root: pathlib.Path, dataset: str, sample_id: str) -> bool:
    """Check whether a sample has been fully processed (atomic test)."""
    return meta_json_path(sample_dir(output_root, dataset, sample_id)).exists()


# ---------------------------------------------------------------------------
# Mask metrics (used by Stage 1 to populate the CSV)
# ---------------------------------------------------------------------------


def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Intersection-over-union between two boolean masks.

    **Convention — `(empty pred, empty gt) → 1.0`** ("perfect agreement on
    nothing present"). This is consistent with treating an unannotated
    frame plus an empty prediction as a true negative. The trade-off:
    downstream per-class / per-sample mean-IoU aggregators MUST filter
    these rows out (e.g. by `notes == "no_gt_this_frame"`) or the means
    are inflated by 1.0s from frames where neither GT nor pred had any
    foreground. Stage-1 records carry `notes` for exactly this purpose;
    see B3 / B1 / S1-3 / few-shot aggregators for the canonical filter.

    `notes == "no_gt_this_frame"` is the load-bearing flag — don't drop
    it from the Stage-1 row schema without auditing every downstream
    aggregator first.
    """
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    inter = np.logical_and(pred_b, gt_b).sum()
    union = np.logical_or(pred_b, gt_b).sum()
    if union == 0:
        return 1.0  # both empty → perfect agreement; see docstring
    return float(inter / union)


def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Sørensen-Dice between two boolean masks.

    **Convention — `(empty pred, empty gt) → 1.0`**. Same caveat as
    `compute_iou`: aggregators that average across frames must drop
    rows tagged `notes == "no_gt_this_frame"` or sparse-class means
    are biased upward.
    """
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    inter = np.logical_and(pred_b, gt_b).sum()
    denom = pred_b.sum() + gt_b.sum()
    if denom == 0:
        return 1.0
    return float(2 * inter / denom)
