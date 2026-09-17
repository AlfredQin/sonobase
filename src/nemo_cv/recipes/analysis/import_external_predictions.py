"""Import an external (non-SAM2) model's masks as a Stage-1 prediction run.

Specialist baselines (nnU-Net, EchoNet, DeepLabV3+, the ACOUSLIC-AI challenge
solutions, ...) run in their own venvs and cannot call `save_predictions.py`.
They write a *neutral* layout instead; this module turns that layout into a
Stage-1 run directory that every Stage-2 analysis (A1/A2/A4, B1/B3, C2, T1.x,
T2.x, S1-3, few-shot aggregation) consumes unchanged via `--runs label=<dir>`.

Neutral layout (written by the external model):

    <neutral_dir>/manifest.json
        {
          "model_label": "nnunet_resenc_m",
          "dataset": "HC18",
          "kind": "image" | "video",
          "source": "<abs path of the weights / results folder>",
          "class_map": {"<saus category_id>": <int value in the label map>},
          "frame_convention": "saus",        # frame index = SaUS frame index
          "extra": {...}                     # free-form provenance
        }
    <neutral_dir>/<DS>/<sample_id>/pred_<frame:05d>_labelmap.png          # layout "labelmap"
        uint8 label map at the native SaUS image size (frame 00000 for images)
    <neutral_dir>/<DS>/<sample_id>/pred_<frame:05d>_class_<value>.png     # layout "per_class"
        one binary PNG per class value in class_map (may overlap; for models
        that predict nested / multi-label structures)
    manifest.layout selects one of the two (default "labelmap").

Rules:

* The prediction for a GT object of SaUS category ``c`` is the full semantic
  class-``c`` mask (``labelmap == class_map[c]``). The GT box is never used to
  select components — the specialist is unprompted.
* The row population is taken from a *reference* Stage-1 run (``--rows-from``,
  normally the archived ``sonobase_<DS>_box_0corr``) so that the paper-table
  cell (``100 * mean(iou)`` over every CSV row) is computed on the identical
  ``(sample_id, frame_idx, obj_id)`` set. ``gt_area_px`` is asserted equal
  row by row; a mismatch aborts before anything is written for that sample.
  Without ``--rows-from`` (datasets with no archived run) the rule is: one row
  per non-empty GT object (images) / per GT-bearing frame (videos).
* A missing label map for a required frame is an error, never "empty".
* All artefacts are written through `prediction_io` — this file adds a second
  Stage-1 *producer*, not a second layout.

Example:

    cd src && uv run python -m nemo_cv.recipes.analysis.import_external_predictions \
        --dataset HC18 \
        --neutral-dir ../external/nnunet_us/neutral/HC18 \
        --run-dir experiments/analysis/predictions/nnunet_resenc_m_HC18_none \
        --rows-from experiments/analysis/predictions/sonobase_HC18_box_0corr
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from omegaconf import OmegaConf
from PIL import Image

from nemo_cv.components.analysis import prediction_io as pio
from nemo_cv.recipes.analysis.save_predictions import (
    GtObject,
    VideoGtObject,
    iter_image_samples,
    iter_video_samples,
    load_gt_objects_sa1b,
    load_gt_objects_video,
)

logger = logging.getLogger("import_external_predictions")

PROMPT_PROTOCOL = "none"
RowKey = Tuple[str, int, int]  # (sample_id, frame_idx, obj_id)


# ---------------------------------------------------------------------------
# Neutral-layout helpers
# ---------------------------------------------------------------------------


@dataclass
class NeutralManifest:
    model_label: str
    dataset: str
    kind: str
    source: str
    class_map: Dict[int, int]
    extra: dict
    layout: str = "labelmap"

    @classmethod
    def load(cls, neutral_dir: pathlib.Path) -> "NeutralManifest":
        m = pio.read_json(neutral_dir / "manifest.json")
        for k in ("model_label", "dataset", "kind", "source", "class_map"):
            if k not in m:
                raise ValueError(f"{neutral_dir/'manifest.json'} lacks required key '{k}'")
        if m["kind"] not in ("image", "video"):
            raise ValueError(f"manifest.kind must be image|video, got {m['kind']!r}")
        if m.get("frame_convention", "saus") != "saus":
            raise ValueError("Only frame_convention='saus' is supported")
        class_map = {int(k): int(v) for k, v in m["class_map"].items()}
        if not class_map:
            raise ValueError("manifest.class_map is empty")
        layout = str(m.get("layout", "labelmap"))
        if layout not in ("labelmap", "per_class"):
            raise ValueError(f"manifest.layout must be labelmap|per_class, got {layout!r}")
        return cls(
            model_label=str(m["model_label"]),
            dataset=str(m["dataset"]),
            kind=str(m["kind"]),
            source=str(m["source"]),
            class_map=class_map,
            extra=dict(m.get("extra") or {}),
            layout=layout,
        )


def load_labelmap(path: pathlib.Path, expected_hw: Tuple[int, int]) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing label map {path}. A required frame with no prediction file is an "
            "error (the external model must write every frame it is scored on)."
        )
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        # Tolerate an RGB/RGBA PNG whose channels are identical (some writers do this).
        if not np.array_equal(arr[..., 0], arr[..., 1]):
            raise ValueError(f"{path}: multi-channel label map with differing channels")
        arr = arr[..., 0]
    if tuple(arr.shape) != tuple(expected_hw):
        raise ValueError(
            f"{path}: label map shape {arr.shape} != GT shape {tuple(expected_hw)}. "
            "The external model must write masks at native SaUS resolution."
        )
    return arr


def class_value(class_map: Dict[int, int], category_id: Optional[int]) -> Optional[int]:
    """Label-map value for a GT category; None if the category is unmapped."""
    if category_id is None:
        return next(iter(class_map.values())) if len(class_map) == 1 else None
    return class_map.get(int(category_id))


class FramePreds:
    """Lazy access to one frame's class masks under either neutral layout."""

    def __init__(self, neutral_dir: pathlib.Path, dataset: str, sample_id: str, frame_idx: int,
                 layout: str, expected_hw: Tuple[int, int]):
        self.dir = neutral_dir / dataset / sample_id
        self.frame_idx = frame_idx
        self.layout = layout
        self.hw = expected_hw
        self._labelmap: Optional[np.ndarray] = None

    def mask(self, value: int) -> np.ndarray:
        if self.layout == "labelmap":
            if self._labelmap is None:
                self._labelmap = load_labelmap(self.dir / f"pred_{self.frame_idx:05d}_labelmap.png", self.hw)
            return self._labelmap == value
        path = self.dir / f"pred_{self.frame_idx:05d}_class_{value}.png"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing class mask {path}. A required frame/class with no prediction file is an "
                "error (the external model must write every class it is scored on)."
            )
        arr = np.array(Image.open(path).convert("L"))
        if tuple(arr.shape) != tuple(self.hw):
            raise ValueError(f"{path}: mask shape {arr.shape} != GT shape {tuple(self.hw)}")
        return arr > 0


# ---------------------------------------------------------------------------
# Reference-run row population
# ---------------------------------------------------------------------------


@dataclass
class ReferenceRows:
    by_sample: Dict[str, Dict[RowKey, dict]]

    @classmethod
    def load(cls, run_dir: pathlib.Path, dataset: str) -> "ReferenceRows":
        rows = pio.read_per_sample_csv(run_dir)
        by_sample: Dict[str, Dict[RowKey, dict]] = {}
        for r in rows:
            if r["dataset"] != dataset:
                continue
            key = (r["sample_id"], int(r["frame_idx"]), int(r["obj_id"]))
            by_sample.setdefault(r["sample_id"], {})[key] = r
        if not by_sample:
            raise ValueError(f"Reference run {run_dir} has no rows for dataset {dataset}")
        return cls(by_sample=by_sample)


def _diff_rows(planned: Dict[RowKey, int], reference: Dict[RowKey, dict], sample_id: str) -> List[str]:
    """Compare planned (key -> gt_area_px) with the reference rows of one sample."""
    diffs: List[str] = []
    pk, rk = set(planned), set(reference)
    for k in sorted(pk - rk):
        diffs.append(f"{sample_id}: planned row {k} absent from reference")
    for k in sorted(rk - pk):
        diffs.append(f"{sample_id}: reference row {k} not produced")
    for k in sorted(pk & rk):
        ref_area = reference[k]["gt_area_px"]
        if ref_area is not None and int(ref_area) != int(planned[k]):
            diffs.append(f"{sample_id}: gt_area_px {planned[k]} != reference {ref_area} at {k}")
    return diffs


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------


@dataclass
class Plan:
    """Rows to emit for one sample: key -> (gt_mask or None, category_id, category_name)."""
    rows: Dict[RowKey, Tuple[Optional[np.ndarray], Optional[int], Optional[str]]]
    n_frames: int
    hw: Tuple[int, int]


def plan_image(sample_id: str, objects: List[GtObject], class_map: Dict[int, int],
               unmapped: str) -> Tuple[Plan, int]:
    rows = {}
    n_unmapped = 0
    hw = None
    for obj in objects:
        if obj.mask.sum() == 0:
            continue  # mirrors save_predictions: empty GT objects get no row
        hw = tuple(obj.mask.shape)
        if class_value(class_map, obj.category_id) is None:
            n_unmapped += 1
            if unmapped == "error":
                raise ValueError(
                    f"{sample_id}: GT object {obj.obj_id} has category_id={obj.category_id} "
                    f"not in class_map {class_map}; pass --unmapped-objects skip to drop it"
                )
            continue
        rows[(sample_id, 0, obj.obj_id)] = (obj.mask, obj.category_id, obj.category_name)
    if hw is None:
        hw = (0, 0)
    return Plan(rows=rows, n_frames=1, hw=hw), n_unmapped


def plan_video(sample_id: str, objects: List[VideoGtObject], n_frames: int, hw: Tuple[int, int],
               class_map: Dict[int, int], unmapped: str,
               reference: Optional[Dict[RowKey, dict]]) -> Tuple[Plan, int]:
    rows = {}
    n_unmapped = 0
    non_empty = [o for o in objects if o.per_frame_masks]
    for obj in non_empty:
        if class_value(class_map, obj.category_id) is None:
            n_unmapped += 1
            if unmapped == "error":
                raise ValueError(
                    f"{sample_id}: GT object {obj.obj_id} has category_id={obj.category_id} "
                    f"not in class_map {class_map}; pass --unmapped-objects skip to drop it"
                )
            continue
        if reference is not None:
            frames = sorted(f for (sid, f, oid) in reference if oid == obj.obj_id)
        else:
            frames = sorted(obj.per_frame_masks)  # GT-bearing frames only
        for f in frames:
            rows[(sample_id, f, obj.obj_id)] = (obj.per_frame_masks.get(f), obj.category_id, obj.category_name)
    return Plan(rows=rows, n_frames=n_frames, hw=hw), n_unmapped


def emit_sample(run_root: pathlib.Path, dataset: str, sample_id: str, plan: Plan,
                neutral_dir: pathlib.Path, manifest: "NeutralManifest", model_label: str,
                image_ref: str, kind: str) -> List[pio.PerSampleRecord]:
    sdir = pio.sample_dir(run_root, dataset, sample_id)
    records: List[pio.PerSampleRecord] = []
    frames: Dict[int, FramePreds] = {}
    for (sid, f_idx, obj_id), (gt_mask, cat_id, cat_name) in sorted(plan.rows.items(), key=lambda kv: (kv[0][2], kv[0][1])):
        if f_idx not in frames:
            frames[f_idx] = FramePreds(neutral_dir, dataset, sample_id, f_idx, manifest.layout, plan.hw)
        value = class_value(manifest.class_map, cat_id)
        assert value is not None  # planned rows are mapped by construction
        pred_mask = frames[f_idx].mask(value)
        gt_to_save = gt_mask if gt_mask is not None else np.zeros(plan.hw, dtype=bool)
        iou = pio.compute_iou(pred_mask, gt_to_save)
        dice = pio.compute_dice(pred_mask, gt_to_save)
        pred_path = pio.pred_mask_path(sdir, f_idx, obj_id)
        gt_path = pio.gt_mask_path(sdir, f_idx, obj_id)
        pio.save_mask_png(pred_path, pred_mask)
        pio.save_mask_png(gt_path, gt_to_save)
        records.append(pio.PerSampleRecord(
            dataset=dataset,
            sample_id=sample_id,
            frame_idx=f_idx,
            obj_id=obj_id,
            category_id=cat_id,
            category_name=cat_name,
            model=model_label,
            prompt_protocol=PROMPT_PROTOCOL,
            iou=iou, dice=dice,
            pred_area_px=int(pred_mask.astype(bool).sum()),
            gt_area_px=int(gt_to_save.astype(bool).sum()),
            pred_mask_path=str(pred_path.relative_to(run_root)),
            gt_mask_path=str(gt_path.relative_to(run_root)),
            notes="" if gt_mask is not None else "no_gt_this_frame",
        ))
    pio.write_json(pio.prompts_json_path(sdir), {"prompts": []})
    meta = {
        "sample_id": sample_id,
        "dataset": dataset,
        "n_frames": int(plan.n_frames),
        "n_objects": len({k[2] for k in plan.rows}),
        ("image_path" if kind == "image" else "images_dir"): image_ref,
        "image_size_hw": [int(plan.hw[0]), int(plan.hw[1])],
        "records": [r.to_csv_row() for r in records],
    }
    pio.write_json(pio.meta_json_path(sdir), meta)
    return records


def records_from_meta(run_root: pathlib.Path, dataset: str, sample_id: str) -> List[pio.PerSampleRecord]:
    meta = pio.read_json(pio.meta_json_path(pio.sample_dir(run_root, dataset, sample_id)))
    out = []
    for r in meta["records"]:
        r = dict(r)
        r.pop("iteration", None)
        out.append(pio.PerSampleRecord(**{k: v for k, v in r.items() if k in pio.PerSampleRecord.__dataclass_fields__}))
    return out


def _git_hash(cwd: pathlib.Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return ""


def run(args: argparse.Namespace) -> int:
    neutral_dir = pathlib.Path(args.neutral_dir).resolve()
    run_root = pathlib.Path(args.run_dir).resolve()
    manifest = NeutralManifest.load(neutral_dir)
    dataset = args.dataset
    if manifest.dataset != dataset:
        raise ValueError(f"--dataset {dataset} != manifest.dataset {manifest.dataset}")
    model_label = args.model_label or manifest.model_label

    saus_dir = pathlib.Path(args.saus_dir or os.path.join(os.environ["DATASET_DIR"], dataset))
    split_list = args.split_list or os.path.join(os.environ["ANNOTATION_DIR"], dataset, "test_list.txt")
    data_cfg = OmegaConf.create({
        "name": dataset, "kind": manifest.kind, "saus_dir": str(saus_dir),
        "split_list_txt": str(split_list), "image_ext": ".jpg", "gt_ext": ".json", "gt_suffix": "_manual",
    })

    reference = ReferenceRows.load(pathlib.Path(args.rows_from).resolve(), dataset) if args.rows_from else None

    signature = {
        "dataset": dataset,
        "model_label": model_label,
        "prompt_protocol": PROMPT_PROTOCOL,
        "ckpt_path": manifest.source,
    }
    run_root.mkdir(parents=True, exist_ok=True)
    pio.check_run_signature(run_root, dataset, signature)
    pio.write_manifest(
        run_root,
        run_name=run_root.name,
        model={"label": model_label, "ckpt_path": manifest.source, "external_manifest": str(neutral_dir / "manifest.json"),
               "class_map": {str(k): v for k, v in manifest.class_map.items()}, "layout": manifest.layout,
               "extra": manifest.extra},
        dataset={"name": dataset, "kind": manifest.kind, "split_list_txt": str(split_list), "saus_dir": str(saus_dir)},
        prompt_protocol={"type": PROMPT_PROTOCOL, "num_correction_clicks": 0, "box_jitter_pct": 0},
        extra={
            "importer": "nemo_cv.recipes.analysis.import_external_predictions",
            "importer_git_hash": _git_hash(pathlib.Path(__file__).parent),
            "rows_from": str(pathlib.Path(args.rows_from).resolve()) if args.rows_from else None,
            "unmapped_objects": args.unmapped_objects,
        },
    )

    samples = list(iter_image_samples(data_cfg) if manifest.kind == "image" else iter_video_samples(data_cfg))
    if args.limit:
        samples = samples[: args.limit]
    logger.info(f"{dataset}: {len(samples)} samples from {split_list}; kind={manifest.kind}; "
                f"class_map={manifest.class_map}; rows_from={'yes' if reference else 'no'}")

    all_records: List[pio.PerSampleRecord] = []
    n_done = n_resumed = n_skipped_empty = n_unmapped_total = 0
    seen_ids = set()
    for i, sample in enumerate(samples):
        sid = sample.sample_id
        seen_ids.add(sid)
        if not args.overwrite and pio.is_sample_complete(run_root, dataset, sid):
            all_records.extend(records_from_meta(run_root, dataset, sid))
            n_resumed += 1
            continue

        ref_rows = None
        if reference is not None:
            ref_rows = reference.by_sample.get(sid)

        if manifest.kind == "image":
            objects = load_gt_objects_sa1b(sample.gt_json_path)
            plan, n_unmapped = plan_image(sid, objects, manifest.class_map, args.unmapped_objects)
            image_ref = str(sample.image_path)
        else:
            objects, n_frames, hw = load_gt_objects_video(sample.gt_json_path)
            plan, n_unmapped = plan_video(sid, objects, n_frames, hw, manifest.class_map,
                                          args.unmapped_objects, ref_rows)
            image_ref = str(sample.images_dir)
        n_unmapped_total += n_unmapped

        if not plan.rows:
            n_skipped_empty += 1
            if ref_rows:
                raise RuntimeError(f"{sid}: no rows planned but reference has {len(ref_rows)} rows")
            continue

        if reference is not None:
            if ref_rows is None:
                raise RuntimeError(f"{sid}: sample is in the split list but absent from the reference run")
            planned_areas = {k: int(v[0].sum()) if v[0] is not None else 0 for k, v in plan.rows.items()}
            ref_cmp = ref_rows
            if args.unmapped_objects == "skip":
                # Only compare the categories this model can produce (e.g. EchoNet: LV endo only).
                mapped = set(manifest.class_map)
                ref_cmp = {k: r for k, r in ref_rows.items()
                           if r.get("category_id") is None or int(r["category_id"]) in mapped}
            diffs = _diff_rows(planned_areas, ref_cmp, sid)
            if diffs:
                for d in diffs[:10]:
                    logger.error(d)
                logger.error(f"{sid}: {len(diffs)} row mismatches vs reference; aborting (nothing written for this sample)")
                return 1

        recs = emit_sample(run_root, dataset, sid, plan, neutral_dir, manifest, model_label, image_ref, manifest.kind)
        all_records.extend(recs)
        n_done += 1
        if (i + 1) % 25 == 0 or (i + 1) == len(samples):
            logger.info(f"[{i + 1:4d}/{len(samples)}] done={n_done} resumed={n_resumed} "
                        f"skipped_empty={n_skipped_empty} rows={len(all_records)}")

    if reference is not None and not args.limit:
        extra = set(reference.by_sample) - seen_ids
        if extra:
            logger.error(f"Reference run has {len(extra)} samples not in the split list, e.g. {sorted(extra)[:5]}")
            return 1

    pio.write_per_sample_csv(run_root, all_records)
    logger.info(f"Import complete: {len(all_records)} records | done={n_done} resumed={n_resumed} "
                f"skipped_empty={n_skipped_empty} unmapped_objects={n_unmapped_total} | {run_root}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="SaUS dataset name, e.g. HC18")
    p.add_argument("--neutral-dir", required=True, help="Directory holding manifest.json + <DS>/<sid>/pred_*_labelmap.png")
    p.add_argument("--run-dir", required=True, help="Stage-1 run directory to create (e.g. .../nnunet_resenc_m_HC18_none)")
    p.add_argument("--rows-from", default=None, help="Reference Stage-1 run whose (sample, frame, obj) rows define the population")
    p.add_argument("--saus-dir", default=None, help="Default: $DATASET_DIR/<dataset>")
    p.add_argument("--split-list", default=None, help="Default: $ANNOTATION_DIR/<dataset>/test_list.txt")
    p.add_argument("--model-label", default=None, help="Default: manifest.model_label")
    p.add_argument("--unmapped-objects", choices=["error", "skip"], default="error",
                   help="What to do with GT objects whose category is not in class_map")
    p.add_argument("--limit", type=int, default=0, help="Process only the first N samples (smoke)")
    p.add_argument("--overwrite", action="store_true", help="Re-import samples that already have meta.json")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
