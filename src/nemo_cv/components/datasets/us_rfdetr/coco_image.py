"""SaUS-layout per-dataset detection dataset for US-RFDETR.

A single SaUS sub-dataset (BUS-BRA, DDTI, FUGC, KidneyUS, LUMINOUS,
MMOTU-3d, ACOUSLIC) loaded with **per-category labels** preserved from
``dataset_info.json``. This is the workhorse for 7 of the 9 datasets
in scope.

Two annotation formats are auto-detected per sample:

- **Image-based** (SA-1B JSON): one ``gt/<name>.json`` per image with
  ``annotations[i].category_id`` + ``annotations[i].segmentation`` (RLE).
  Datasets: BUS-BRA, DDTI, FUGC, KidneyUS, LUMINOUS, MMOTU-3d.
- **Video-based** (SA-V masklet JSON): one ``gt/<name>_manual.json``
  per video with ``masklet[ann_step][obj_id]`` RLE masks +
  ``masklet_category_id`` per object. Datasets: ACOUSLIC.

Categories and metadata are loaded from
``<data_root>/dataset_info.json``. Split files
(``train_list.txt``, ``test_list.txt``, optional ``val_list.txt``)
live under ``<annotation_dir>/`` (= ``${DET_ANNOTATION_DIR}/<dataset>/``).

Ported from ``USSam/projects/DenseUS/saus_coco_det.py`` with these
changes:

* Lightning `LightningDataModule` baseclass replaced by a plain
  configurable class — the US-RFDETR recipe builds DataLoaders directly
  in its `setup()`. Module exposes `train_dataset`, `val_datasets`,
  `test_datasets`, plus `num_classes`, `dataset_name`, `categories`
  attributes the recipe reads.
* `get_*_transforms` / `SanitizeBoundingBoxes` / `detection_collate_fn`
  moved into the shared :mod:`.transforms` module.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pycocotools.mask as mask_utils
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors

from nemo_cv.components.datasets.us_rfdetr.transforms import (
    get_train_transforms,
    get_val_transforms,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Dataset
# ---------------------------------------------------------------------------


class SaUSCocoDetDataset(Dataset):
    """Single SaUS sub-dataset with per-category labels and instance masks.

    Parameters
    ----------
    data_root : str
        Root of the SaUS sub-dataset (e.g. ``$DATASET_DIR/BUS-BRA``, where
        ``DATASET_DIR=$WORK/Dataset/SaUS`` is the canonical layout).
    file_list_txt : str
        Path to a split file (one sample name per line, with or without
        extension).
    transforms : callable, optional
        Torchvision-v2 transforms applied to ``(image, target)`` pairs.
    multiplier : int
        Repeat training samples N times (default 1) to upsample small
        datasets so each epoch sees enough examples to fill DDP shards.
    """

    SUPPORTED_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

    def __init__(
        self,
        data_root: str,
        file_list_txt: str,
        transforms: Optional[Callable] = None,
        multiplier: int = 1,
    ):
        self.data_root = Path(data_root)
        self.img_folder = self.data_root / "images"
        self.gt_folder = self.data_root / "gt"
        self.transforms = transforms

        # Categories from dataset_info.json (1-indexed in SA-1B convention)
        info_path = self.data_root / "dataset_info.json"
        if info_path.exists():
            with open(info_path) as f:
                info = json.load(f)
            self.categories = {
                cat["id"]: cat["name"] for cat in info.get("categories", [])
            }
        else:
            self.categories = {}
        self.num_classes = max(len(self.categories), 1)

        # Read sample names; auto-detect video folder vs flat image
        with open(file_list_txt, "r") as f:
            raw_names = [
                os.path.splitext(line.strip())[0]
                for line in f
                if line.strip()
            ]

        self.samples: List[Tuple[str, Optional[str]]] = []
        for name in raw_names:
            folder = self.img_folder / name
            if folder.is_dir():
                # Video: expand to one entry per frame
                frames = sorted(
                    f for f in folder.iterdir()
                    if f.suffix.lower() in self.SUPPORTED_IMG_EXTS
                )
                for frame_path in frames:
                    self.samples.append((name, frame_path.stem))
            else:
                self.samples.append((name, None))

        base_len = len(self.samples)
        if multiplier > 1:
            self.samples = self.samples * int(multiplier)

        logger.info(
            f"SaUSCocoDetDataset[{self.data_root.name}]: "
            f"{base_len} samples"
            + (f" x {multiplier} = {len(self.samples)}" if multiplier > 1 else "")
            + f", {self.num_classes} classes: {list(self.categories.values())}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        name, frame_stem = self.samples[idx]
        image = self._load_image(name, frame_stem)
        W, H = image.size

        boxes, labels, masks_list, areas = self._load_annotations(
            name, frame_stem, H, W
        )

        if len(boxes) == 0:
            target = {
                "boxes": tv_tensors.BoundingBoxes(
                    torch.zeros((0, 4), dtype=torch.float32),
                    format=tv_tensors.BoundingBoxFormat.XYXY,
                    canvas_size=(H, W),
                ),
                "labels": torch.zeros(0, dtype=torch.int64),
                "image_id": idx,
                "area": torch.zeros(0, dtype=torch.float32),
                "iscrowd": torch.zeros(0, dtype=torch.int64),
            }
            if masks_list is not None:
                target["masks"] = tv_tensors.Mask(
                    torch.zeros((0, H, W), dtype=torch.uint8)
                )
        else:
            target = {
                "boxes": tv_tensors.BoundingBoxes(
                    torch.as_tensor(boxes, dtype=torch.float32),
                    format=tv_tensors.BoundingBoxFormat.XYXY,
                    canvas_size=(H, W),
                ),
                "labels": torch.as_tensor(labels, dtype=torch.int64),
                "image_id": idx,
                "area": torch.as_tensor(areas, dtype=torch.float32),
                "iscrowd": torch.zeros(len(boxes), dtype=torch.int64),
            }
            if masks_list is not None and len(masks_list) == len(boxes):
                target["masks"] = tv_tensors.Mask(torch.stack(masks_list))

        image = tv_tensors.Image(image)
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target

    # ------------------------------------------------------------------
    #  Image loading
    # ------------------------------------------------------------------
    def _load_image(self, name: str, frame_stem: Optional[str]) -> Image.Image:
        if frame_stem is not None:
            folder = self.img_folder / name
            for ext in self.SUPPORTED_IMG_EXTS:
                p = folder / (frame_stem + ext)
                if p.exists():
                    return Image.open(p).convert("RGB")
        for ext in self.SUPPORTED_IMG_EXTS:
            p = self.img_folder / (name + ext)
            if p.exists():
                return Image.open(p).convert("RGB")
        raise FileNotFoundError(
            f"Image not found: '{name}' (frame={frame_stem}) in {self.img_folder}"
        )

    # ------------------------------------------------------------------
    #  Annotation loading
    # ------------------------------------------------------------------
    def _load_annotations(
        self, name: str, frame_stem: Optional[str], H: int, W: int
    ) -> Tuple[List, List, Optional[List], List]:
        """Returns (boxes_xyxy, labels, masks_or_None, areas)."""
        if frame_stem is not None:
            return self._load_sav_annotations(name, frame_stem, H, W)
        return self._load_sa1b_annotations(name, H, W)

    def _load_sa1b_annotations(
        self, name: str, H: int, W: int
    ) -> Tuple[List, List, Optional[List], List]:
        """Parse SA-1B JSON with category_id and RLE segmentation."""
        json_path = self.gt_folder / (name + ".json")
        if not json_path.exists():
            return [], [], None, []

        with open(json_path) as f:
            data = json.load(f)

        annotations = data.get("annotations", [])
        boxes, labels, masks_list, areas = [], [], [], []

        for ann in annotations:
            seg = ann.get("segmentation", None)
            cat_id = ann.get("category_id", 1)
            mask_tensor = None

            if seg is not None and isinstance(seg, dict) and "counts" in seg:
                if isinstance(seg["counts"], list):
                    rle = mask_utils.frPyObjects(seg, H, W)
                else:
                    rle = seg
                mask_np = mask_utils.decode(rle)
                mask_tensor = torch.as_tensor(mask_np, dtype=torch.uint8)

                ys, xs = np.where(mask_np > 0)
                if len(ys) == 0:
                    continue
                # +1: xs/ys.max() is the last foreground pixel INDEX; the box's
                # exclusive right/bottom edge is one past it. Matches the COCO
                # [x,y,w,h] branch below and the continuous boxes torchmetrics
                # expects.
                x1, y1 = float(xs.min()), float(ys.min())
                x2, y2 = float(xs.max() + 1), float(ys.max() + 1)
            elif "bbox" in ann:
                bx, by, bw, bh = ann["bbox"]
                x1, y1, x2, y2 = bx, by, bx + bw, by + bh
            else:
                continue

            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(W), x2), min(float(H), y2)
            if x2 - x1 < 1.0 or y2 - y1 < 1.0:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(cat_id)
            areas.append((x2 - x1) * (y2 - y1))
            if mask_tensor is not None:
                masks_list.append(mask_tensor)

        has_masks = len(masks_list) == len(boxes) and len(masks_list) > 0
        return boxes, labels, masks_list if has_masks else None, areas

    def _load_sav_annotations(
        self, name: str, frame_stem: str, H: int, W: int
    ) -> Tuple[List, List, Optional[List], List]:
        """Parse SA-V masklet JSON for a specific frame, with categories."""
        sav_json = self.gt_folder / (name + "_manual.json")
        if not sav_json.exists():
            return [], [], None, []

        with open(sav_json) as f:
            data = json.load(f)

        masklet_key = "masklet" if "masklet" in data else "masks"
        masklets = data[masklet_key]

        ann_every = int(data.get("ann_every", 1))
        if "fps" in data:
            fps = (
                int(data["fps"][0])
                if isinstance(data["fps"], list)
                else int(data["fps"])
            )
            if fps > 0 and 24 % fps == 0:
                ann_every = 24 // fps

        frame_idx = int(frame_stem)
        ann_idx = frame_idx // ann_every
        if ann_idx >= len(masklets):
            return [], [], None, []

        cat_ids = data.get("masklet_category_id", None)

        frame_masks = masklets[ann_idx]
        boxes, labels, masks_list, areas = [], [], [], []

        for obj_id, rle in enumerate(frame_masks):
            if rle is None:
                continue
            if not isinstance(rle, dict) or "counts" not in rle:
                continue

            mask_np = mask_utils.decode(rle)
            ys, xs = np.where(mask_np > 0)
            if len(ys) == 0:
                continue

            # +1: xs/ys.max() is the last foreground pixel INDEX; the box's
            # exclusive right/bottom edge is one past it. Matches the COCO
            # [x,y,w,h] convention and the continuous boxes torchmetrics expects.
            x1, y1 = float(xs.min()), float(ys.min())
            x2, y2 = float(xs.max() + 1), float(ys.max() + 1)
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(W), x2), min(float(H), y2)
            if x2 - x1 < 1.0 or y2 - y1 < 1.0:
                continue

            cat_id = (
                cat_ids[obj_id]
                if cat_ids and obj_id < len(cat_ids)
                else 1
            )

            boxes.append([x1, y1, x2, y2])
            labels.append(cat_id)
            areas.append((x2 - x1) * (y2 - y1))
            masks_list.append(torch.as_tensor(mask_np, dtype=torch.uint8))

        has_masks = len(masks_list) == len(boxes) and len(masks_list) > 0
        return boxes, labels, masks_list if has_masks else None, areas


# ---------------------------------------------------------------------------
#  DataModule
# ---------------------------------------------------------------------------


class SaUSCocoDetDataModule:
    """Plain data-module for a single SaUS sub-dataset.

    Not a Lightning ``LightningDataModule`` — the US-RFDETR recipe owns
    the train / val / test loop and constructs DataLoaders itself in
    ``setup()``. This class just bundles the (Dataset, transforms,
    metadata) triplets and exposes them as attributes.

    After ``setup()``:

    * ``train_dataset``: ``SaUSCocoDetDataset`` or ``None``
    * ``val_datasets``: ``list[SaUSCocoDetDataset]`` (one per val split)
    * ``test_datasets``: ``list[SaUSCocoDetDataset]`` (one per test split)
    * ``val_dataset_names``, ``test_dataset_names``: per-loader labels
    * ``num_classes``: from ``dataset_info.json`` (or ``1`` if missing)
    * ``dataset_name``: the leaf folder name (e.g. ``"BUS-BRA"``)

    Parameters
    ----------
    data_root : str
        Root of the SaUS sub-dataset.
    annotation_dir : str
        Directory containing the per-split text files
        (``train_list.txt``, optional ``val_list.txt``, ``test_list.txt``).
        Defaults to ``${DET_ANNOTATION_DIR}/<dataset>``.
    image_size : int
        Target square resolution.
    multiplier : int
        Repeat training samples N times.
    val_from_test : bool
        If no ``val_list.txt`` exists, fall back to using
        ``test_list.txt`` for validation. Default ``False`` to avoid
        accidental train↔test leakage during development.
    """

    def __init__(
        self,
        data_root: str,
        annotation_dir: str,
        image_size: int = 1024,
        multiplier: int = 1,
        val_from_test: bool = False,
    ):
        self.data_root = Path(data_root)
        self.annotation_dir = Path(annotation_dir)
        self.image_size = int(image_size)
        self.multiplier = int(multiplier)
        self.val_from_test = bool(val_from_test)

        self.train_dataset: Optional[Dataset] = None
        self.val_datasets: List[Dataset] = []
        self.test_datasets: List[Dataset] = []
        self.val_dataset_names: List[str] = []
        self.test_dataset_names: List[str] = []

        # Metadata from dataset_info.json (loaded eagerly so the recipe
        # can read num_classes before setup() to size model heads)
        info_path = self.data_root / "dataset_info.json"
        if info_path.exists():
            with open(info_path) as f:
                info = json.load(f)
            self.categories: Dict[int, str] = {
                cat["id"]: cat["name"] for cat in info.get("categories", [])
            }
        else:
            self.categories = {}
        self.num_classes = max(len(self.categories), 1)
        self.dataset_name = self.data_root.name

    def setup(self) -> None:
        """Build train / val / test ``Dataset`` instances."""
        train_t = get_train_transforms(self.image_size)
        val_t = get_val_transforms(self.image_size)

        train_list = self.annotation_dir / "train_list.txt"
        val_list = self.annotation_dir / "val_list.txt"
        test_list = self.annotation_dir / "test_list.txt"

        if train_list.exists():
            self.train_dataset = SaUSCocoDetDataset(
                data_root=str(self.data_root),
                file_list_txt=str(train_list),
                transforms=train_t,
                multiplier=self.multiplier,
            )

        if val_list.exists():
            self.val_datasets = [SaUSCocoDetDataset(
                data_root=str(self.data_root),
                file_list_txt=str(val_list),
                transforms=val_t,
            )]
            self.val_dataset_names = [f"{self.dataset_name}-val"]
        elif self.val_from_test and test_list.exists():
            self.val_datasets = [SaUSCocoDetDataset(
                data_root=str(self.data_root),
                file_list_txt=str(test_list),
                transforms=val_t,
            )]
            self.val_dataset_names = [f"{self.dataset_name}-test-as-val"]

        if test_list.exists():
            self.test_datasets = [SaUSCocoDetDataset(
                data_root=str(self.data_root),
                file_list_txt=str(test_list),
                transforms=val_t,
            )]
            self.test_dataset_names = [f"{self.dataset_name}-test"]

        logger.info(
            f"SaUSCocoDetDataModule[{self.dataset_name}] setup: "
            f"train={len(self.train_dataset) if self.train_dataset else 0}, "
            f"val={sum(len(d) for d in self.val_datasets)} "
            f"({len(self.val_datasets)} loaders), "
            f"test={sum(len(d) for d in self.test_datasets)} "
            f"({len(self.test_datasets)} loaders)"
        )
