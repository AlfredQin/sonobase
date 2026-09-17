"""Stage 1 of the analysis suite: run a single model on a dataset and save
per-sample prediction artifacts to disk.

The output layout is produced via the
helpers in `nemo_cv.components.analysis.prediction_io`. Every Stage-2
analysis (HC measurement, EF, Bland-Altman, etc.) consumes those artifacts
without re-running inference.

Design notes:

- **Single-GPU only.** No `torchrun`, no DDP, no rank-aware logic. The
  `__main__` block deliberately calls into `main()` once.
- **One model × one dataset × one prompt protocol per run.** The recipe
  instantiates the model once and iterates the dataset's test split.
- **Resumable.** A sample is "done" iff its `meta.json` exists. Re-running
  the recipe on the same `output_dir` skips done samples.
- **Image AND video datasets.** Dispatches on `data.kind`:
    - `image` → `SAM2ImagePredictor.predict()` per sample (HC18, BUSI, …).
    - `video` → `SAM2VideoPredictor.init_state` + prompt frame 0 +
      `propagate_in_video` (CAMUS, ACOUSLIC, RegPro).
  Both paths share the same prompt-derivation utilities (point /
  box / correction-click sequences) and write to the same per-sample
  schema. Only the inference call shape differs.
- **Click-efficiency mode (A3).** When `prompt_protocol.iterations` is
  set (e.g. `[0, 1, 3, 5, 7]`), the recipe saves predictions at each
  iteration milestone within a single sample run, suffixing per-iter
  mask filenames with `_iter<N>` and emitting one CSV row per
  (sample × frame × obj × iteration). For non-A3 runs the field is
  left null and the legacy file naming applies.
- **Inference path (image).** Uses `SAM2ImagePredictor` (wraps any
  `SAM2Base` subclass including `SAM2Train`) with explicit point/box
  prompts derived from the GT mask. Same protocol as the old
  `visualize_predictions.py` and gives us full prompt control
  (vs. the training-style val pass that `test_sam2.py` runs).
- **Inference path (video).** Uses `SAM2VideoPredictor.init_state`
  pointing at `<saus_dir>/images/<video_id>/` (a directory of frame
  JPEGs), prompts on a configurable frame (default 0), then propagates
  through all frames. The first frame on which a given object has a
  non-empty GT mask is used as the prompt frame.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from pycocotools import mask as coco_mask

from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.loggers.log_utils import setup_logging

from nemo_cv.components.analysis import prediction_io as pio
from nemo_cv.components.analysis.prompts import (
    PromptRecord,
    initial_box_from_gt,
    initial_point_from_gt,
    jitter_box,
    next_correction_point,
)
from nemo_cv.components.models.sam2.sam2_image_predictor import SAM2ImagePredictor
from nemo_cv.components.models.sam2.sam2_video_predictor import SAM2VideoPredictor
from nemo_cv.components.training.utils import build_autocast_context
from nemo_cv.recipes.sonobase.pretrain import build_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sample loading (image + video datasets)
# ---------------------------------------------------------------------------


@dataclass
class ImageSample:
    """One sample of an image dataset (one image, one or more GT objects).

    For HC18 each sample has exactly one object (the fetal head). For
    multi-object image datasets (e.g. CAMUS still frames with LV-endo +
    LV-epi + LA), the GT JSON has multiple annotations and we iterate them
    inside the recipe.
    """

    sample_id: str
    image_path: pathlib.Path
    gt_json_path: pathlib.Path


def _read_split_list(split_list_txt: str) -> List[str]:
    """Read sample IDs (one per line) from an explicit split-list file.

    The caller is responsible for providing the absolute path. We don't
    infer it from `saus_dir` + a `split` token — that lookup hard-coded
    a "raw SaUS layout" assumption that's wrong for any dataset where the
    project-canonical splits live outside the SaUS tree (e.g. under
    `SaUS_Annotation/.../<DATASET>/test_list.txt`).
    """
    list_path = pathlib.Path(split_list_txt).expanduser()
    if not list_path.is_file():
        raise FileNotFoundError(
            f"Split-list file not found: {list_path}. "
            "Set `split_list_txt` in the data config to the absolute path "
            "of the file you want to evaluate (it's the dataset config's "
            "responsibility, not this recipe's, to know where the list lives)."
        )
    with list_path.open() as f:
        return [line.strip() for line in f if line.strip()]


def iter_image_samples(data_cfg: DictConfig) -> Iterator[ImageSample]:
    """Iterate (image, gt) pairs for a SaUS-format image dataset.

    Expects `data_cfg` with at least: `name`, `saus_dir`, `split_list_txt`
    (absolute path), `image_ext`, `gt_ext`. See
    `src/configs/analysis/data/HC18.yaml` for a worked example.
    """
    saus_dir = pathlib.Path(data_cfg.saus_dir)
    image_ext = data_cfg.image_ext
    gt_ext = data_cfg.gt_ext
    image_dir = saus_dir / "images"
    gt_dir = saus_dir / "gt"

    for sid in _read_split_list(data_cfg.split_list_txt):
        yield ImageSample(
            sample_id=sid,
            image_path=image_dir / f"{sid}{image_ext}",
            gt_json_path=gt_dir / f"{sid}{gt_ext}",
        )


def load_image_rgb(path: pathlib.Path) -> np.ndarray:
    """Load an image as a uint8 [H, W, 3] RGB numpy array."""
    return np.array(Image.open(path).convert("RGB"))


@dataclass
class GtObject:
    obj_id: int
    mask: np.ndarray   # bool [H, W]
    category_id: Optional[int] = None
    category_name: Optional[str] = None


def load_gt_objects_sa1b(json_path: pathlib.Path) -> List[GtObject]:
    """Decode all objects from a SA-1B style GT JSON.

    The annotation format is `{"image": {...}, "annotations": [{...}, ...]}`
    where each annotation has a COCO-RLE-encoded `segmentation`. For HC18
    there is exactly one annotation per sample (the fetal head).
    """
    with json_path.open() as f:
        ann = json.load(f)

    objects: List[GtObject] = []
    for i, a in enumerate(ann.get("annotations", [])):
        seg = a.get("segmentation")
        if seg is None:
            continue
        mask = coco_mask.decode(seg).astype(bool)
        # Some SaUS converters squeeze the 3rd dim; some don't. Normalize.
        if mask.ndim == 3:
            mask = mask[..., 0]
        objects.append(
            GtObject(
                obj_id=i,
                mask=mask,
                category_id=a.get("category_id"),
                category_name=a.get("category_name"),
            )
        )
    return objects


# ---------------------------------------------------------------------------
# Video-dataset loaders (CAMUS / ACOUSLIC / RegPro / EchoNet etc.)
# ---------------------------------------------------------------------------


@dataclass
class VideoSample:
    """One sample of a video / 3-D dataset (one video, one or more GT objects).

    SaUS video layout (per CAMUS / ACOUSLIC / RegPro inspection):
        <saus_dir>/images/<sample_id>/<frame_idx:05d>.jpg     # frame stack
        <saus_dir>/gt/<sample_id>_manual.json                 # per-frame masklet

    The GT JSON's `masklet` field is **frame-major**: `masklet[frame_idx][obj_idx]`
    is the COCO-RLE dict for object ``obj_idx`` at frame ``frame_idx`` — see
    ``load_gt_objects_video`` for the canonical reader and per-object category
    plumbing (`masklet_category_id` / `masklet_category_name` parallel arrays
    indexed by obj_idx). For CAMUS that's 18 frames × 3 objects (endocardium
    / epicardium / atrium_wall — in that obj_idx order).
    """

    sample_id: str
    images_dir: pathlib.Path
    gt_json_path: pathlib.Path


def iter_video_samples(data_cfg: DictConfig) -> Iterator[VideoSample]:
    """Iterate (images_dir, gt_json) pairs for a SaUS-format video dataset.

    Expects `data_cfg` with: `name`, `kind=video`, `saus_dir`, `split`,
    `gt_suffix` (default ``_manual``), `gt_ext` (default ``.json``).
    """
    saus_dir = pathlib.Path(data_cfg.saus_dir)
    image_dir = saus_dir / "images"
    gt_dir = saus_dir / "gt"
    gt_suffix = data_cfg.get("gt_suffix", "_manual")
    gt_ext = data_cfg.get("gt_ext", ".json")

    for sid in _read_split_list(data_cfg.split_list_txt):
        yield VideoSample(
            sample_id=sid,
            images_dir=image_dir / sid,
            gt_json_path=gt_dir / f"{sid}{gt_suffix}{gt_ext}",
        )


@dataclass
class VideoGtObject:
    obj_id: int
    # one bool mask per frame; missing/empty frames have `None` in the dict
    per_frame_masks: dict
    n_frames: int
    height: int
    width: int
    category_id: Optional[int] = None
    category_name: Optional[str] = None


def load_gt_objects_video(json_path: pathlib.Path) -> Tuple[List[VideoGtObject], int, Tuple[int, int]]:
    """Decode a SaUS-style video GT JSON.

    Returns `(objects, n_frames, (height, width))`.

    The `masklet` field in the SaUS / SA-V format is **frame-major**:
    `masklet[frame_idx][obj_idx]` is the COCO-RLE dict for object
    ``obj_idx`` at frame ``frame_idx``. Object indices are stable across
    frames (object 0 at frame 0 = object 0 at frame 17). Frames where an
    object isn't visible may carry an RLE that decodes to all zeros — we
    treat those as "no GT this frame".

    For CAMUS this means 19 frames × 3 objects (endocardium / epicardium
    / atrium_wall, in that order — preserved from the original SA-V conversion).
    """
    with json_path.open() as f:
        ann = json.load(f)

    n_frames = int(ann.get("video_frame_count", 0))
    h = int(ann.get("video_height", 0))
    w = int(ann.get("video_width", 0))
    masklet = ann.get("masklet") or []
    if not masklet:
        return [], n_frames, (h, w)

    n_objs = max(len(frame_rles) for frame_rles in masklet) if masklet else 0
    per_obj: List[dict] = [dict() for _ in range(n_objs)]
    for f_idx, frame_rles in enumerate(masklet):
        for obj_idx, rle in enumerate(frame_rles):
            if rle is None:
                continue
            try:
                m = coco_mask.decode(rle).astype(bool)
            except Exception:
                continue
            if m.ndim == 3:
                m = m[..., 0]
            if m.any():
                per_obj[obj_idx][f_idx] = m

    # Per-object category metadata (SaUS schema: parallel arrays at top level).
    # CAMUS JSONs have `masklet_category_id: [1, 2, 3]` and
    # `masklet_category_name: ["endocardium", "epicardium", "atrium_wall"]`,
    # indexed by obj_idx. Older / non-SaUS-conformant JSONs may lack these —
    # default each entry to None so downstream code degrades gracefully.
    cat_ids = ann.get("masklet_category_id") or []
    cat_names = ann.get("masklet_category_name") or []

    objects: List[VideoGtObject] = []
    for obj_idx in range(n_objs):
        objects.append(
            VideoGtObject(
                obj_id=obj_idx,
                per_frame_masks=per_obj[obj_idx],
                n_frames=n_frames,
                height=h,
                width=w,
                category_id=cat_ids[obj_idx] if obj_idx < len(cat_ids) else None,
                category_name=cat_names[obj_idx] if obj_idx < len(cat_names) else None,
            )
        )
    return objects, n_frames, (h, w)


# ---------------------------------------------------------------------------
# Prompted inference for one (image, GT object)
# ---------------------------------------------------------------------------


def score_on_square_grid(low_res_logits: np.ndarray, gt_mask: np.ndarray,
                         size: int = 1024) -> Tuple[float, float]:
    """IoU/Dice computed on a square `size`x`size` grid instead of the native frame.

    This is the grid the pretraining harness scores on: its val transform applies
    `RandomResizeAPI(sizes=1024, square=True)`, which stretches the ground-truth
    mask to a square before the metric sees it.

    Both sides are put on that grid the same way the model itself would:
    the 256x256 logits are bilinearly upsampled and thresholded at 0 (exactly
    what `SAM2Transforms.postprocess_masks` does, only to a square target rather
    than to `orig_hw`), and the GT is resized with NEAREST. Starting from the
    logits rather than from the already-native mask is what keeps this free of a
    resampling round-trip -- native and square scores then differ only in the
    grid, which is the whole point of the ablation.
    """
    lr = torch.as_tensor(low_res_logits, dtype=torch.float32)[None, None]
    pred_sq = F.interpolate(lr, (size, size), mode="bilinear", align_corners=False)
    pred_sq = (pred_sq[0, 0] > 0.0).numpy()

    gt = torch.as_tensor(gt_mask.astype(np.float32))[None, None]
    gt_sq = F.interpolate(gt, (size, size), mode="nearest")[0, 0].numpy() > 0.5

    return pio.compute_iou(pred_sq, gt_sq), pio.compute_dice(pred_sq, gt_sq)


def run_one_object_image(
    img_predictor: SAM2ImagePredictor,
    image: np.ndarray,
    gt_mask: np.ndarray,
    obj_id: int,
    prompt_type: str,
    num_correction_clicks: int,
    box_jitter_pct: float = 0.0,
    rng: Optional[np.random.Generator] = None,
    multimask: bool = True,
    return_low_res: bool = False,
    point_sampling: str = "center",
):
    """Run prompted inference on a single (image, GT object) pair.

    Returns the final binary predicted mask and the `PromptRecord` describing
    every prompt issued (initial + corrections).

    Protocol:
      1. Initial prompt = centre point (RITM-style) OR GT bounding box,
         optionally perturbed by `box_jitter_pct` (see `prompts.jitter_box`).
      2. Run `predict(multimask_output=multimask)`; with three candidates,
         pick the highest-score mask.
      3. For each of `num_correction_clicks` extra clicks, sample the next
         click from the largest current error region and re-predict
         (`multimask_output=False`).

    `box_jitter_pct > 0` requires `rng`: a jittered box is a random draw, and
    a benchmark whose prompts are unseeded draws is not reproducible. The
    prompt actually issued — jittered or not — is what lands in `record.box`.

    `multimask=False` asks the decoder for one mask instead of three, removing
    the best-of-3 selection (ablation). The pretraining harness effectively
    runs this way for box prompts: it encodes a box as two points, and its
    `multimask_max_pt_num` of 1 then gates multimask off.

    `return_low_res=True` appends the selected 256x256 logits to the returned
    tuple. Those are what the scoring-grid ablation needs — upsampling the same
    logits to a square 1024 rather than to the native frame isolates the metric
    grid with no resampling round-trip.
    """
    if box_jitter_pct > 0:
        if prompt_type != "box":
            raise ValueError(
                f"box_jitter_pct={box_jitter_pct} is meaningless for "
                f"prompt_type={prompt_type!r}; it applies to box prompts only."
            )
        if rng is None:
            raise ValueError(
                "box_jitter_pct > 0 requires a seeded `rng` so the run is reproducible."
            )
    if point_sampling == "uniform" and rng is None:
        raise ValueError(
            "point_sampling='uniform' requires a seeded `rng` so the run is reproducible."
        )
    if gt_mask.sum() == 0:
        raise ValueError("run_one_object_image: GT mask is empty")

    # Prompt-side bookkeeping
    record = PromptRecord(frame_idx=0, obj_id=obj_id, type=prompt_type)
    box: Optional[np.ndarray] = None

    # Embed the image (recomputes vision-encoder features)
    img_predictor.set_image(image)

    # ---- Step 1: initial prompt + first prediction ----
    if prompt_type == "box":
        box = initial_box_from_gt(gt_mask)
        if box_jitter_pct > 0:
            box = jitter_box(box, box_jitter_pct, gt_mask.shape, rng)
        record.box = box.tolist()
        masks, scores, low_res = img_predictor.predict(
            box=box,
            multimask_output=multimask,
        )
    elif prompt_type == "point":
        point, label = initial_point_from_gt(gt_mask, method=point_sampling, rng=rng)
        record.points.append([float(point[0]), float(point[1])])
        record.labels.append(int(label))
        masks, scores, low_res = img_predictor.predict(
            point_coords=np.asarray([point], dtype=np.float32),
            point_labels=np.asarray([label], dtype=np.int32),
            multimask_output=multimask,
        )
    else:
        raise ValueError(f"Unknown prompt_type: {prompt_type!r}")

    # With multimask_output=True SAM2 returns 3 candidates plus iou-prediction
    # scores and we take the best; with False there is a single mask and argmax
    # over the length-1 score array selects it, so the same line serves both.
    best = int(np.argmax(scores))
    current_pred = masks[best].astype(np.uint8)
    low_res_sel = low_res[best]

    # ---- Step 2: correction clicks ----
    for _ in range(num_correction_clicks):
        pt, lbl = next_correction_point(gt_mask, current_pred.astype(bool))
        record.points.append([float(pt[0]), float(pt[1])])
        record.labels.append(int(lbl))

        masks, scores, low_res = img_predictor.predict(
            point_coords=np.asarray(record.points, dtype=np.float32),
            point_labels=np.asarray(record.labels, dtype=np.int32),
            box=box,                                # may be None
            multimask_output=False,                  # single refined output
        )
        current_pred = masks[0].astype(np.uint8)
        low_res_sel = low_res[0]

    if return_low_res:
        return current_pred, record, low_res_sel
    return current_pred, record


# ---------------------------------------------------------------------------
# Prompted inference for one video object (used by CAMUS / ACOUSLIC / RegPro)
# ---------------------------------------------------------------------------


def _pick_prompt_frame(gt_obj: VideoGtObject, requested: int) -> int:
    """Pick the frame on which to prompt this object.

    If the requested frame index has a non-empty GT for this object, use it.
    Otherwise fall back to the lowest frame index that has a GT (so we
    always issue the prompt where the object is actually visible). Returns
    -1 if no frame has a GT — the caller skips the object.
    """
    if requested in gt_obj.per_frame_masks:
        return requested
    if not gt_obj.per_frame_masks:
        return -1
    return min(gt_obj.per_frame_masks.keys())


def run_one_object_video(
    video_predictor: SAM2VideoPredictor,
    state: dict,
    gt_obj: VideoGtObject,
    prompt_type: str,
    num_correction_clicks: int,
    requested_prompt_frame: int = 0,
    box_jitter_pct: float = 0.0,
    rng: Optional[np.random.Generator] = None,
    point_sampling: str = "center",
) -> Tuple[dict, PromptRecord]:
    """Run prompted inference on one (video, GT object).

    Returns:
        (per_frame_pred, prompt_record)

        `per_frame_pred` is a dict ``{frame_idx: bool mask}`` for every
        frame the predictor emitted output for. Frames without GT for this
        object are still propagated and recorded — the analysis layer
        decides what to do with them (e.g. EF needs both ED and ES;
        temporal consistency wants every frame).

    Protocol on the prompt frame (frame `requested_prompt_frame` if it has
    GT, otherwise the first frame with GT):

      1. Initial prompt = centre point (RITM) OR GT bounding box.
      2. For `num_correction_clicks` extra clicks, sample from the
         current prediction's largest error region on the same frame
         (so corrections refine the prompt frame, not subsequent frames).
      3. Propagate forward through the rest of the video.

    `box_jitter_pct` perturbs the box on the prompt frame only, and carries the
    same seeded-`rng` requirement as `run_one_object_image`.
    """
    if box_jitter_pct > 0:
        if prompt_type != "box":
            raise ValueError(
                f"box_jitter_pct={box_jitter_pct} is meaningless for "
                f"prompt_type={prompt_type!r}; it applies to box prompts only."
            )
        if rng is None:
            raise ValueError(
                "box_jitter_pct > 0 requires a seeded `rng` so the run is reproducible."
            )
    if point_sampling == "uniform" and rng is None:
        raise ValueError(
            "point_sampling='uniform' requires a seeded `rng` so the run is reproducible."
        )

    prompt_frame = _pick_prompt_frame(gt_obj, requested_prompt_frame)
    if prompt_frame < 0:
        # No GT anywhere — caller will skip this object.
        return {}, PromptRecord(frame_idx=-1, obj_id=gt_obj.obj_id, type=prompt_type)

    record = PromptRecord(frame_idx=prompt_frame, obj_id=gt_obj.obj_id, type=prompt_type)
    box: Optional[np.ndarray] = None
    gt_mask = gt_obj.per_frame_masks[prompt_frame]

    # ---- Step 1: initial prompt ----
    if prompt_type == "box":
        box = initial_box_from_gt(gt_mask)
        if box_jitter_pct > 0:
            box = jitter_box(box, box_jitter_pct, gt_mask.shape, rng)
        record.box = box.tolist()
        _, _, mask_logits = video_predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=prompt_frame,
            obj_id=gt_obj.obj_id,
            box=box,
        )
    elif prompt_type == "point":
        point, label = initial_point_from_gt(gt_mask, method=point_sampling, rng=rng)
        record.points.append([float(point[0]), float(point[1])])
        record.labels.append(int(label))
        _, _, mask_logits = video_predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=prompt_frame,
            obj_id=gt_obj.obj_id,
            points=np.asarray([point], dtype=np.float32),
            labels=np.asarray([label], dtype=np.int32),
        )
    else:
        raise ValueError(f"Unknown prompt_type: {prompt_type!r}")

    current_pred = (mask_logits[0] > 0.0).cpu().numpy().squeeze().astype(bool)

    # ---- Step 2: correction clicks (still on the prompt frame) ----
    for _ in range(num_correction_clicks):
        pt, lbl = next_correction_point(gt_mask, current_pred)
        record.points.append([float(pt[0]), float(pt[1])])
        record.labels.append(int(lbl))

        _, _, mask_logits = video_predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=prompt_frame,
            obj_id=gt_obj.obj_id,
            points=np.asarray(record.points, dtype=np.float32),
            labels=np.asarray(record.labels, dtype=np.int32),
            box=box,
        )
        current_pred = (mask_logits[0] > 0.0).cpu().numpy().squeeze().astype(bool)

    return {prompt_frame: current_pred}, record


def collect_video_predictions(
    video_predictor: SAM2VideoPredictor,
    state: dict,
    obj_ids: List[int],
    square_size: Optional[int] = None,
) -> dict:
    """After all prompts are added, propagate and collect per-(frame, obj) masks.

    Returns ``{(frame_idx, obj_id): bool mask}``, or with `square_size` set,
    ``{(frame_idx, obj_id): (native_mask, square_mask)}`` for the scoring-grid
    ablation.

    The square mask is obtained by interpolating the *logits* and thresholding,
    not by resampling the thresholded mask. `propagate_in_video` has already
    upsampled the 256x256 logits to video resolution and only yields those, and
    reaching the low-res tensor would mean editing shared SAM2 code. Resizing a
    continuous field one extra time costs far less than resampling a binary one,
    so this stays close to the image path's logits-based scoring -- but it is
    one interpolation further from the source, which the image path is not.
    """
    out = {}
    for f_idx, returned_obj_ids, mask_logits in video_predictor.propagate_in_video(state):
        for i, oid in enumerate(returned_obj_ids):
            if oid not in obj_ids:
                continue
            logits = mask_logits[i]
            mask = (logits > 0.0).cpu().numpy().squeeze().astype(bool)
            if square_size is None:
                out[(int(f_idx), int(oid))] = mask
            else:
                lg = logits.detach().float()
                while lg.dim() < 4:
                    lg = lg.unsqueeze(0)
                sq = F.interpolate(lg, (square_size, square_size),
                                   mode="bilinear", align_corners=False)
                sq = (sq[0, 0] > 0.0).cpu().numpy().astype(bool)
                out[(int(f_idx), int(oid))] = (mask, sq)
    return out


# ---------------------------------------------------------------------------
# Overlay rendering (viz only; behind a config flag)
# ---------------------------------------------------------------------------


def _blend_mask(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int],
                alpha: float = 0.45) -> np.ndarray:
    """Alpha-blend a binary mask onto an RGB image."""
    out = image.copy()
    if mask.any():
        m = mask.astype(bool)
        for c in range(3):
            out[m, c] = (out[m, c] * (1 - alpha) + color[c] * alpha).astype(np.uint8)
    return out


def render_overlay(
    image: np.ndarray, gt_mask: np.ndarray, pred_mask: np.ndarray,
    prompt: PromptRecord, iou: float, dice: float,
) -> np.ndarray:
    """Render image + GT (green) + pred (orange) + prompt markers + caption."""
    import cv2

    img = _blend_mask(image, gt_mask.astype(bool), color=(0, 220, 0), alpha=0.35)
    img = _blend_mask(img, pred_mask.astype(bool), color=(255, 140, 0), alpha=0.45)

    # Draw prompts
    if prompt.box is not None:
        x0, y0, x1, y1 = [int(v) for v in prompt.box]
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 255), 2)
    for (x, y), lab in zip(prompt.points, prompt.labels):
        color = (0, 255, 0) if lab == 1 else (255, 0, 0)
        cv2.circle(img, (int(x), int(y)), 6, color, -1)
        cv2.circle(img, (int(x), int(y)), 6, (0, 0, 0), 1)

    # IoU/Dice caption (top-left)
    text = f"IoU={iou:.3f}  Dice={dice:.3f}"
    cv2.rectangle(img, (5, 5), (5 + 11 * len(text), 30), (0, 0, 0), -1)
    cv2.putText(img, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, lineType=cv2.LINE_AA)

    return img


# ---------------------------------------------------------------------------
# Main recipe
# ---------------------------------------------------------------------------


class SavePredictionsRecipe:
    """Single-GPU prediction-saving recipe."""

    def __init__(self, cfg: ConfigNode, hydra_cfg: DictConfig):
        self.cfg = cfg
        self.hydra_cfg = hydra_cfg

    def setup(self):
        hcfg = self.hydra_cfg
        setup_logging()

        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_str)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        logger.info(f"Device: {self.device}")

        cuda_cfg = hcfg.get("cuda", {})
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = cuda_cfg.get("cudnn_deterministic", True)
            torch.backends.cudnn.benchmark = cuda_cfg.get("cudnn_benchmark", False)
            allow_tf32 = cuda_cfg.get("allow_tf32", False)
            torch.backends.cuda.matmul.allow_tf32 = cuda_cfg.get("matmul_allow_tf32", allow_tf32)
            torch.backends.cudnn.allow_tf32 = cuda_cfg.get("cudnn_allow_tf32", allow_tf32)

        # ---- Build the model ----
        # Always instantiate as SAM2Train (the released model config target)
        # so all SAM2Train-only kwargs in the YAML compose cleanly. For
        # video runs we class-swap to SAM2VideoPredictor afterwards so we
        # get init_state / add_new_points_or_box / propagate_in_video.
        # Both classes inherit from SAM2Base and share the same parameter
        # set, so this is purely a method-table swap (no parameter copy).
        seed = int(hcfg.get("seed", 42))
        # Prompt-side randomness lives in its own stream, so turning box jitter
        # on cannot perturb anything else that draws from the global seed.
        self._box_jitter_pct = float(hcfg.prompt_protocol.get("box_jitter_pct", 0.0))
        # Ablation knobs. Both default to the shipped behaviour, so an
        # unmodified config reproduces every existing record exactly.
        self._multimask = bool(hcfg.prompt_protocol.get("multimask_output", True))
        self._score_square_grid = bool(hcfg.get("score_square_grid", False))
        # "center" (RITM, every published run) | "uniform" (SAM2's training-time
        # sampler). See `prompts.initial_point_from_gt`.
        self._point_sampling = str(hcfg.prompt_protocol.get("point_sampling", "center"))
        if self._point_sampling not in ("center", "uniform"):
            raise ValueError(
                f"prompt_protocol.point_sampling={self._point_sampling!r}; "
                "expected 'center' or 'uniform'."
            )
        # A dedicated prompt seed lets a multi-seed prompt-noise study vary the
        # prompts while holding everything else (model build included) fixed.
        # Unset falls back to the global seed, so every existing run reproduces.
        prompt_seed = hcfg.prompt_protocol.get("seed", None)
        prompt_seed = seed if prompt_seed is None else int(prompt_seed)
        self._prompt_rng = np.random.default_rng(prompt_seed)
        kind = hcfg.data.get("kind", "image")
        model = build_model(hcfg.model, self.device, seed=seed, pretrained_ckpt_path=None)
        if kind == "video":
            self._reclass_as_video_predictor(model)

        ckpt_path = str(pathlib.Path(hcfg.ckpt_path).expanduser().resolve())
        logger.info(f"Loading model weights from: {ckpt_path}")
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if "model" in sd:
            sd = sd["model"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        logger.info(
            f"Loaded weights — missing={len(missing)}, unexpected={len(unexpected)}"
        )
        if len(unexpected) > 0:
            logger.info(f"  first 3 unexpected: {unexpected[:3]}")
        if len(missing) > 0:
            logger.info(f"  first 3 missing: {missing[:3]}")
        del sd
        model.eval()
        self.model = model

        # ---- Build the inference wrapper appropriate for this dataset kind ----
        if kind == "video":
            # The model IS the video predictor for video runs.
            self.video_predictor = self.model
            self.img_predictor = None
        else:
            # SAM2ImagePredictor accepts any SAM2Base subclass including SAM2Train.
            self.img_predictor = SAM2ImagePredictor(sam_model=self.model)
            self.img_predictor.model.eval()
            self.video_predictor = None

        # ---- Output dir ----
        self.output_root = pathlib.Path(hcfg.output_dir).expanduser().resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output root: {self.output_root}")

        self.run_name = hcfg.get("run_name", "unnamed_run")
        self.model_label = hcfg.get("model_label", "unknown")
        self.prompt_protocol_str = self._format_prompt_protocol(hcfg.prompt_protocol)

        # Run signature gate FIRST — before any file writes — so a re-run
        # with mismatched (model, prompt, ckpt) raises before clobbering the
        # existing root-level manifest.json. Scope is per-dataset
        # (<output_root>/<dataset>/run_signature.json), so the same
        # output_root can host multiple datasets but each is locked to one
        # signature.
        self._run_signature: Dict[str, Any] = {
            "dataset": hcfg.data.name,
            "model_label": self.model_label,
            "prompt_protocol": self.prompt_protocol_str,
            "ckpt_path": ckpt_path,
        }
        pio.check_run_signature(self.output_root, hcfg.data.name, self._run_signature)

        # ---- Manifest ----
        pio.write_manifest(
            self.output_root,
            run_name=self.run_name,
            model={
                "label": self.model_label,
                "image_encoder_target": OmegaConf.select(hcfg, "model.image_encoder.trunk._target_"),
                "ckpt_path": ckpt_path,
            },
            dataset={
                "name": hcfg.data.name,
                "kind": hcfg.data.kind,
                "split_list_txt": str(hcfg.data.split_list_txt),
                "saus_dir": str(hcfg.data.saus_dir),
            },
            prompt_protocol=OmegaConf.to_container(hcfg.prompt_protocol, resolve=True),
            extra={
                "save_overlays": bool(hcfg.get("save_overlays", True)),
                "max_samples": hcfg.get("max_samples", None),
            },
        )

        # AMP context factory
        self._autocast = self._build_autocast(hcfg.get("optim", None))

        logger.info(f"SavePredictionsRecipe setup complete (run_name={self.run_name})")

    @staticmethod
    def _reclass_as_video_predictor(model) -> None:
        """In-place: change model's class to SAM2VideoPredictor.

        Both `SAM2Train` and `SAM2VideoPredictor` extend `SAM2Base` and
        register the same nn.Modules and parameters (image_encoder,
        memory_attention, memory_encoder, mask_decoder, prompt_encoder,
        learned no-mem/no-obj embeddings). Their differences live in
        non-parameter attributes and method tables, both of which we can
        substitute by overwriting `__class__` and assigning the
        `SAM2VideoPredictor`-only attributes (these have safe defaults
        from the predictor's __init__ signature).

        This avoids the brittle alternative of trying to whitelist which
        SAM2Train-only YAML kwargs to drop before re-instantiation.
        """
        model.__class__ = SAM2VideoPredictor
        # Defaults from SAM2VideoPredictor.__init__ signature:
        model.fill_hole_area = 0
        model.non_overlap_masks = False
        model.clear_non_cond_mem_around_input = False
        model.add_all_frames_to_correct_as_cond = False

    @staticmethod
    def _format_prompt_protocol(p: DictConfig) -> str:
        n = int(p.get("num_correction_clicks", 0))
        iters = p.get("iterations", None)
        if iters:
            # A3 click-efficiency protocol — encode the snapshot iterations.
            return f"{p.type}_iters_{'_'.join(str(int(x)) for x in iters)}"
        # Randomised-prompt knobs append only when non-default, so every run
        # made before they existed keeps the exact string it already wrote.
        # This column is what makes a per_sample_metrics.csv self-describing:
        # without it a jittered run and an exact one are indistinguishable
        # outside the run name.
        s = f"{p.type}_{n}corr"
        jit = float(p.get("box_jitter_pct", 0.0) or 0.0)
        if p.type == "box" and jit > 0:
            s += f"_jit{int(round(jit * 100)):02d}"
        if p.type == "point" and p.get("point_sampling", "center") == "uniform":
            s += "_unif"
        if p.get("seed", None) is not None and (jit > 0 or p.get("point_sampling") == "uniform"):
            s += f"_s{int(p.seed)}"
        return s

    @staticmethod
    def _build_autocast(optim_cfg: Optional[DictConfig]):
        if optim_cfg is None or "amp" not in optim_cfg:
            return nullcontext
        amp_enabled = bool(optim_cfg.amp.get("enabled", False))
        if not amp_enabled:
            return nullcontext
        return lambda: build_autocast_context(optim_cfg)

    # ------------------ main loop ------------------
    @torch.no_grad()
    def run(self):
        kind = self.hydra_cfg.data.get("kind", "image")
        if kind == "image":
            self._run_image()
        elif kind == "video":
            self._run_video()
        else:
            raise ValueError(f"Unknown data.kind={kind!r}; expected 'image' or 'video'.")

    # ------------------ image-dataset loop ------------------
    @torch.no_grad()
    def _run_image(self):
        hcfg = self.hydra_cfg
        data_cfg = hcfg.data
        prompt_cfg = hcfg.prompt_protocol
        save_overlays = bool(hcfg.get("save_overlays", True))
        max_samples = hcfg.get("max_samples", None)
        log_freq = int(hcfg.get("logging", {}).get("log_freq", 20))

        # A3 click-efficiency mode: snapshot iterations [0, 1, 3, 5, ...].
        # Default behaviour (None) is "single shot at num_correction_clicks".
        iter_snapshots: Optional[List[int]] = None
        raw_iters = prompt_cfg.get("iterations", None)
        if raw_iters:
            iter_snapshots = sorted({int(x) for x in raw_iters})

        samples = list(iter_image_samples(data_cfg))
        if max_samples is not None:
            samples = samples[: int(max_samples)]
        logger.info(f"Processing {len(samples)} samples from {data_cfg.name} "
                    f"(list: {pathlib.Path(data_cfg.split_list_txt).name})")

        all_records: List[pio.PerSampleRecord] = []
        n_done = 0
        n_skipped_done = 0
        n_skipped_empty = 0
        t_start = time.perf_counter()

        for i, sample in enumerate(samples):
            sample_dir = pio.sample_dir(self.output_root, data_cfg.name, sample.sample_id)

            if pio.is_sample_complete(self.output_root, data_cfg.name, sample.sample_id):
                try:
                    meta = pio.read_json(pio.meta_json_path(sample_dir))
                    for rec_dict in meta.get("records", []):
                        all_records.append(pio.PerSampleRecord(**rec_dict))
                    n_skipped_done += 1
                    continue
                except Exception as e:
                    logger.warning(
                        f"Sample {sample.sample_id} marked complete but meta.json "
                        f"unreadable ({e}); reprocessing."
                    )

            try:
                image = load_image_rgb(sample.image_path)
            except FileNotFoundError:
                logger.warning(f"Image not found: {sample.image_path}; skipping")
                continue

            try:
                gt_objects = load_gt_objects_sa1b(sample.gt_json_path)
            except FileNotFoundError:
                logger.warning(f"GT not found: {sample.gt_json_path}; skipping")
                continue

            if not gt_objects:
                n_skipped_empty += 1
                continue

            sample_records: List[pio.PerSampleRecord] = []
            sample_prompts: List[PromptRecord] = []

            for obj in gt_objects:
                if obj.mask.sum() == 0:
                    n_skipped_empty += 1
                    continue

                with self._autocast():
                    if iter_snapshots is None:
                        out = run_one_object_image(
                            self.img_predictor,
                            image,
                            obj.mask,
                            obj_id=obj.obj_id,
                            prompt_type=prompt_cfg.type,
                            num_correction_clicks=int(prompt_cfg.num_correction_clicks),
                            box_jitter_pct=self._box_jitter_pct,
                            rng=self._prompt_rng,
                            multimask=self._multimask,
                            return_low_res=self._score_square_grid,
                            point_sampling=self._point_sampling,
                        )
                        if self._score_square_grid:
                            pred_mask, prompt_rec, low_res_sel = out
                            sq = score_on_square_grid(low_res_sel, obj.mask)
                        else:
                            pred_mask, prompt_rec = out
                            sq = None
                        sample_records.extend(self._save_image_record(
                            sample_dir, sample, image, obj, pred_mask, prompt_rec,
                            iteration=None, save_overlays=save_overlays, square=sq,
                        ))
                        sample_prompts.append(prompt_rec)
                    else:
                        # A3 mode: run the longest iteration and snapshot in between.
                        records_iter, prompts_iter = self._run_image_with_snapshots(
                            sample_dir, sample, image, obj,
                            iter_snapshots=iter_snapshots,
                            prompt_type=prompt_cfg.type,
                            save_overlays=save_overlays,
                        )
                        sample_records.extend(records_iter)
                        sample_prompts.extend(prompts_iter)

            pio.write_json(
                pio.prompts_json_path(sample_dir),
                {"prompts": [r.to_dict() for r in sample_prompts]},
            )
            pio.write_json(
                pio.meta_json_path(sample_dir),
                {
                    "sample_id": sample.sample_id,
                    "dataset": data_cfg.name,
                    "n_frames": 1,
                    "n_objects": len(gt_objects),
                    "image_path": str(sample.image_path),
                    "image_size_hw": list(image.shape[:2]),
                    "records": [r.to_csv_row() for r in sample_records],
                },
            )

            all_records.extend(sample_records)
            n_done += 1

            if (i + 1) % log_freq == 0 or (i + 1) == len(samples):
                elapsed = time.perf_counter() - t_start
                rate = (i + 1) / max(elapsed, 1e-6)
                logger.info(
                    f"[{i + 1:4d}/{len(samples)}] done={n_done} skipped_done={n_skipped_done} "
                    f"skipped_empty={n_skipped_empty} | {rate:.2f} samples/s"
                )

        pio.write_per_sample_csv(self.output_root, all_records)
        elapsed = time.perf_counter() - t_start
        logger.info(
            f"Stage 1 complete: {len(all_records)} records | done={n_done} "
            f"skipped_done={n_skipped_done} skipped_empty={n_skipped_empty} | "
            f"elapsed={elapsed:.1f}s | output_root={self.output_root}"
        )

    def _save_image_record(
        self,
        sample_dir: pathlib.Path,
        sample: ImageSample,
        image: np.ndarray,
        obj: GtObject,
        pred_mask: np.ndarray,
        prompt_rec: PromptRecord,
        iteration: Optional[int],
        save_overlays: bool,
        square: Optional[Tuple[float, float]] = None,
    ) -> List[pio.PerSampleRecord]:
        """Write per-iteration mask + (optional) overlay; return list with one record.

        `square` carries (iou, dice) recomputed on the square 1024 grid when the
        scoring-grid ablation is on; None leaves those columns empty.
        """
        iou = pio.compute_iou(pred_mask, obj.mask)
        dice = pio.compute_dice(pred_mask, obj.mask)

        pred_path = pio.pred_mask_path(sample_dir, 0, obj.obj_id, iteration=iteration)
        gt_path = pio.gt_mask_path(sample_dir, 0, obj.obj_id)
        pio.save_mask_png(pred_path, pred_mask)
        if not gt_path.is_file():
            pio.save_mask_png(gt_path, obj.mask)

        if save_overlays:
            overlay = render_overlay(image, obj.mask, pred_mask, prompt_rec, iou, dice)
            pio.save_overlay_png(
                pio.overlay_path(sample_dir, 0, obj.obj_id, iteration=iteration),
                overlay,
            )

        return [pio.PerSampleRecord(
            dataset=self.hydra_cfg.data.name,
            sample_id=sample.sample_id,
            frame_idx=0,
            obj_id=obj.obj_id,
            category_id=obj.category_id,
            category_name=obj.category_name,
            iteration=iteration,
            model=self.model_label,
            prompt_protocol=self.prompt_protocol_str,
            iou=iou, dice=dice,
            pred_area_px=int(pred_mask.astype(bool).sum()),
            gt_area_px=int(obj.mask.astype(bool).sum()),
            pred_mask_path=str(pred_path.relative_to(self.output_root)),
            gt_mask_path=str(gt_path.relative_to(self.output_root)),
            iou_sq1024=(square[0] if square else None),
            dice_sq1024=(square[1] if square else None),
        )]

    def _run_image_with_snapshots(
        self,
        sample_dir: pathlib.Path,
        sample: ImageSample,
        image: np.ndarray,
        obj: GtObject,
        iter_snapshots: List[int],
        prompt_type: str,
        save_overlays: bool,
    ) -> Tuple[List[pio.PerSampleRecord], List[PromptRecord]]:
        """A3 click-efficiency mode for image datasets.

        Walks `iter_snapshots = [0, 1, 3, 5, 7]` and at each milestone runs
        the model with that many correction clicks accumulated, snapshots
        the predicted mask, and records iou/dice/etc. per iteration.

        Returns (records, prompt_snapshots). Each prompt snapshot mirrors
        the record at the same iteration so we can read prompts.json and
        learn exactly which clicks were issued at each step.
        """
        records: List[pio.PerSampleRecord] = []
        prompt_snapshots: List[PromptRecord] = []
        max_iter = max(iter_snapshots)

        # Seed the inference state with the initial prompt
        self.img_predictor.set_image(image)
        prompt_rec = PromptRecord(frame_idx=0, obj_id=obj.obj_id, type=prompt_type)
        box: Optional[np.ndarray] = None

        if prompt_type == "box":
            box = initial_box_from_gt(obj.mask)
            if self._box_jitter_pct > 0:
                box = jitter_box(box, self._box_jitter_pct, obj.mask.shape, self._prompt_rng)
            prompt_rec.box = box.tolist()
            masks, scores, _ = self.img_predictor.predict(box=box, multimask_output=True)
        else:
            point, label = initial_point_from_gt(
                obj.mask, method=self._point_sampling, rng=self._prompt_rng
            )
            prompt_rec.points.append([float(point[0]), float(point[1])])
            prompt_rec.labels.append(int(label))
            masks, scores, _ = self.img_predictor.predict(
                point_coords=np.asarray([point], dtype=np.float32),
                point_labels=np.asarray([label], dtype=np.int32),
                multimask_output=True,
            )
        best = int(np.argmax(scores))
        current_pred = masks[best].astype(np.uint8)

        if 0 in iter_snapshots:
            records.extend(self._save_image_record(
                sample_dir, sample, image, obj, current_pred,
                _clone_prompt(prompt_rec), iteration=0,
                save_overlays=save_overlays,
            ))
            prompt_snapshots.append(_clone_prompt(prompt_rec, mark_iter=0))

        for n_iter in range(1, max_iter + 1):
            pt, lbl = next_correction_point(obj.mask, current_pred.astype(bool))
            prompt_rec.points.append([float(pt[0]), float(pt[1])])
            prompt_rec.labels.append(int(lbl))
            masks, scores, _ = self.img_predictor.predict(
                point_coords=np.asarray(prompt_rec.points, dtype=np.float32),
                point_labels=np.asarray(prompt_rec.labels, dtype=np.int32),
                box=box,
                multimask_output=False,
            )
            current_pred = masks[0].astype(np.uint8)

            if n_iter in iter_snapshots:
                records.extend(self._save_image_record(
                    sample_dir, sample, image, obj, current_pred,
                    _clone_prompt(prompt_rec), iteration=n_iter,
                    save_overlays=save_overlays,
                ))
                prompt_snapshots.append(_clone_prompt(prompt_rec, mark_iter=n_iter))

        return records, prompt_snapshots

    # ------------------ video-dataset loop ------------------
    @torch.no_grad()
    def _run_video(self):
        hcfg = self.hydra_cfg
        data_cfg = hcfg.data
        prompt_cfg = hcfg.prompt_protocol
        save_overlays = bool(hcfg.get("save_overlays", False))   # videos: many frames; off by default
        max_samples = hcfg.get("max_samples", None)
        log_freq = int(hcfg.get("logging", {}).get("log_freq", 5))
        prompt_frame = int(prompt_cfg.get("prompt_frame", 0))

        if prompt_cfg.get("iterations", None):
            raise NotImplementedError(
                "Per-iteration snapshotting (`prompt_protocol.iterations=[0,1,3,5,7]`, "
                "i.e. saving the predicted mask after every k correction clicks within "
                "ONE recipe call) is image-only.\n"
                "\n"
                "Note this is NOT about supporting correction clicks at all on video — "
                "the standard single-N path via `prompt_protocol.num_correction_clicks=N` "
                "IS supported here (see `run_one_object_video` step 2). Use that for any "
                "fixed correction-click count; loop the recipe externally if you need a "
                "sweep across multiple Ns.\n"
                "\n"
                "Why image-only: SAM2VideoPredictor's per-frame masks come from "
                "`propagate_in_video()`, which walks every frame via memory attention. "
                "Snapshotting at K iteration milestones would require K full re-propagations "
                "per video (the cost dominates). The current analysis pipeline doesn't "
                "need this anyway — A3 click-efficiency consumes the long-format "
                "test_metrics_iterations.csv from the iterative-correction benchmark sweep "
                "(scripts/benchmarks/test_sam2/iterative/)."
            )

        samples = list(iter_video_samples(data_cfg))
        if max_samples is not None:
            samples = samples[: int(max_samples)]
        logger.info(f"Processing {len(samples)} video samples from {data_cfg.name} "
                    f"(list: {pathlib.Path(data_cfg.split_list_txt).name})")

        all_records: List[pio.PerSampleRecord] = []
        n_done = 0
        n_skipped_done = 0
        n_skipped_empty = 0
        t_start = time.perf_counter()

        for i, sample in enumerate(samples):
            sample_dir = pio.sample_dir(self.output_root, data_cfg.name, sample.sample_id)

            if pio.is_sample_complete(self.output_root, data_cfg.name, sample.sample_id):
                try:
                    meta = pio.read_json(pio.meta_json_path(sample_dir))
                    for rec_dict in meta.get("records", []):
                        all_records.append(pio.PerSampleRecord(**rec_dict))
                    n_skipped_done += 1
                    continue
                except Exception as e:
                    logger.warning(
                        f"Sample {sample.sample_id} marked complete but meta.json "
                        f"unreadable ({e}); reprocessing."
                    )

            if not sample.images_dir.is_dir():
                logger.warning(f"Frames dir not found: {sample.images_dir}; skipping")
                continue
            if not sample.gt_json_path.is_file():
                logger.warning(f"GT not found: {sample.gt_json_path}; skipping")
                continue

            try:
                gt_objects, n_frames, (h, w) = load_gt_objects_video(sample.gt_json_path)
            except Exception as e:
                logger.warning(f"GT decode failed for {sample.sample_id}: {e}; skipping")
                continue

            non_empty = [o for o in gt_objects if o.per_frame_masks]
            if not non_empty:
                n_skipped_empty += 1
                continue

            # Init the SAM2 video predictor on this clip
            with self._autocast():
                state = self.video_predictor.init_state(video_path=str(sample.images_dir))

            sample_prompts: List[PromptRecord] = []
            with self._autocast():
                for obj in non_empty:
                    _, prompt_rec = run_one_object_video(
                        self.video_predictor,
                        state,
                        obj,
                        prompt_type=prompt_cfg.type,
                        num_correction_clicks=int(prompt_cfg.num_correction_clicks),
                        requested_prompt_frame=prompt_frame,
                        box_jitter_pct=self._box_jitter_pct,
                        rng=self._prompt_rng,
                        point_sampling=self._point_sampling,
                    )
                    sample_prompts.append(prompt_rec)

                # Propagate forward; collect (frame, obj) → mask
                obj_ids = [o.obj_id for o in non_empty]
                preds = collect_video_predictions(
                    self.video_predictor, state, obj_ids,
                    square_size=(1024 if self._score_square_grid else None),
                )

            # Free the inference state before next sample (large)
            self.video_predictor.reset_state(state)

            # Write per-(frame, obj) artifacts + records
            sample_records: List[pio.PerSampleRecord] = []
            for obj in non_empty:
                for f_idx in range(n_frames):
                    gt_mask = obj.per_frame_masks.get(f_idx)
                    got = preds.get((f_idx, obj.obj_id))
                    pred_sq = None
                    if isinstance(got, tuple):
                        pred_mask, pred_sq = got
                    else:
                        pred_mask = got
                    if gt_mask is None and pred_mask is None:
                        continue
                    if pred_mask is None:
                        # Predictor didn't track this frame for this object;
                        # treat as all-zero so the analysis sees it.
                        pred_mask = np.zeros((h, w), dtype=bool)
                        if self._score_square_grid:
                            pred_sq = np.zeros((1024, 1024), dtype=bool)
                    gt_to_save = gt_mask if gt_mask is not None else np.zeros((h, w), dtype=bool)

                    iou = pio.compute_iou(pred_mask, gt_to_save)
                    dice = pio.compute_dice(pred_mask, gt_to_save)
                    iou_sq = dice_sq = None
                    if pred_sq is not None:
                        gt_sq = F.interpolate(
                            torch.as_tensor(gt_to_save.astype(np.float32))[None, None],
                            (1024, 1024), mode="nearest",
                        )[0, 0].numpy() > 0.5
                        iou_sq = pio.compute_iou(pred_sq, gt_sq)
                        dice_sq = pio.compute_dice(pred_sq, gt_sq)

                    pred_path = pio.pred_mask_path(sample_dir, f_idx, obj.obj_id)
                    gt_path = pio.gt_mask_path(sample_dir, f_idx, obj.obj_id)
                    pio.save_mask_png(pred_path, pred_mask)
                    pio.save_mask_png(gt_path, gt_to_save)

                    notes = "" if gt_mask is not None else "no_gt_this_frame"
                    sample_records.append(pio.PerSampleRecord(
                        dataset=data_cfg.name,
                        sample_id=sample.sample_id,
                        frame_idx=f_idx,
                        obj_id=obj.obj_id,
                        category_id=obj.category_id,
                        category_name=obj.category_name,
                        model=self.model_label,
                        prompt_protocol=self.prompt_protocol_str,
                        iou=iou, dice=dice,
                        pred_area_px=int(pred_mask.astype(bool).sum()),
                        gt_area_px=int(gt_to_save.astype(bool).sum()),
                        pred_mask_path=str(pred_path.relative_to(self.output_root)),
                        gt_mask_path=str(gt_path.relative_to(self.output_root)),
                        notes=notes,
                        iou_sq1024=iou_sq,
                        dice_sq1024=dice_sq,
                    ))

            pio.write_json(
                pio.prompts_json_path(sample_dir),
                {"prompts": [r.to_dict() for r in sample_prompts]},
            )
            pio.write_json(
                pio.meta_json_path(sample_dir),
                {
                    "sample_id": sample.sample_id,
                    "dataset": data_cfg.name,
                    "n_frames": int(n_frames),
                    "n_objects": len(non_empty),
                    "images_dir": str(sample.images_dir),
                    "image_size_hw": [int(h), int(w)],
                    "records": [r.to_csv_row() for r in sample_records],
                },
            )

            all_records.extend(sample_records)
            n_done += 1

            if (i + 1) % log_freq == 0 or (i + 1) == len(samples):
                elapsed = time.perf_counter() - t_start
                rate = (i + 1) / max(elapsed, 1e-6)
                logger.info(
                    f"[{i + 1:4d}/{len(samples)}] done={n_done} skipped_done={n_skipped_done} "
                    f"skipped_empty={n_skipped_empty} | {rate:.2f} videos/s"
                )

        pio.write_per_sample_csv(self.output_root, all_records)
        elapsed = time.perf_counter() - t_start
        logger.info(
            f"Stage 1 complete: {len(all_records)} records | done={n_done} "
            f"skipped_done={n_skipped_done} skipped_empty={n_skipped_empty} | "
            f"elapsed={elapsed:.1f}s | output_root={self.output_root}"
        )


def _clone_prompt(p: PromptRecord, mark_iter: Optional[int] = None) -> PromptRecord:
    """Shallow copy of a `PromptRecord`. Used to snapshot the prompt state at
    each iteration milestone in A3 mode (subsequent corrections mutate the
    original record's `points`/`labels`).

    If `mark_iter` is set, the iteration is appended to `type` for clarity
    in the saved prompts.json (e.g. ``point_iter3``).
    """
    new_type = p.type if mark_iter is None else f"{p.type}_iter{mark_iter}"
    return PromptRecord(
        frame_idx=p.frame_idx,
        obj_id=p.obj_id,
        type=new_type,
        points=[list(pt) for pt in p.points],
        labels=list(p.labels),
        box=list(p.box) if p.box is not None else None,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    """CLI entry. Same Hydra arg conventions as the other recipes (no torchrun)."""
    import sys

    config_dir = None
    config_name = "save_predictions"
    overrides: List[str] = []
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] in ("-c", "--config-dir") and i + 1 < len(args):
            config_dir = str(pathlib.Path(args[i + 1]).resolve())
            i += 2
        elif args[i] in ("-cn", "--config-name") and i + 1 < len(args):
            config_name = args[i + 1]
            i += 2
        elif "=" in args[i] or args[i].startswith("+") or args[i].startswith("~"):
            overrides.append(args[i])
            i += 1
        else:
            i += 1

    if config_dir is None:
        config_dir = str(pathlib.Path(__file__).parent.resolve() / "configs")

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=config_dir, version_base=None)
    hydra_cfg = compose(config_name=config_name, overrides=overrides)
    OmegaConf.resolve(hydra_cfg)

    cfg_dict = OmegaConf.to_container(hydra_cfg, resolve=True)
    cfg_node = ConfigNode(cfg_dict)

    recipe = SavePredictionsRecipe(cfg_node, hydra_cfg)
    recipe.setup()
    recipe.run()


if __name__ == "__main__":
    main()
