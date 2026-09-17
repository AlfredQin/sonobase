"""Standard COCO-triplet detection dataset for US-RFDETR (Fetus).

The Fetus dataset ships in vanilla COCO format::

    coco_detection/
      images/
        train/<id>.png
        val/<id>.png
        test/<id>.png
      annotations/
        instances_train.json
        instances_val.json
        instances_test.json

9 categories: thalami, midbrain, palate, fourth ventricle, cisterna
magna, nuchal translucency, nasal tip, nasal skin, nasal bone.

Ported from ``USSam/projects/DenseUS/fetus_det.py`` with the same
structural changes as :mod:`.coco_image`: drop Lightning baseclass,
expose ``num_classes`` / ``train_dataset`` / ``val_datasets`` /
``test_datasets`` as attributes for the recipe to consume.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pycocotools.mask as mask_utils
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset
from torchvision import tv_tensors

from nemo_cv.components.datasets.us_rfdetr.transforms import (
    get_train_transforms,
    get_val_transforms,
)

logger = logging.getLogger(__name__)


class FetusDetDataset(Dataset):
    """Standard COCO detection dataset for fetal ultrasound."""

    def __init__(
        self,
        data_root: str,
        ann_file: str,
        transforms: Optional[Callable] = None,
    ):
        self.data_root = Path(data_root)
        self.transforms = transforms

        with open(ann_file, "r") as f:
            coco = json.load(f)

        self.categories = {cat["id"]: cat["name"] for cat in coco["categories"]}
        self.num_classes = len(self.categories)
        self.images = {img["id"]: img for img in coco["images"]}
        self.image_ids = sorted(self.images.keys())

        self.img_anns: Dict[int, List[Dict]] = {}
        for ann in coco["annotations"]:
            self.img_anns.setdefault(ann["image_id"], []).append(ann)

        n_with_ann = sum(1 for iid in self.image_ids if iid in self.img_anns)
        logger.info(
            f"FetusDetDataset: {len(self.image_ids)} images "
            f"({n_with_ann} with annotations, {self.num_classes} classes) "
            f"from {ann_file}"
        )

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]

        img_path = self.data_root / img_info["file_name"]
        if not img_path.exists():
            img_path = self.data_root / "images" / img_info["file_name"]
        image = Image.open(img_path).convert("RGB")
        W, H = image.size

        anns = self.img_anns.get(img_id, [])
        boxes, labels, masks_list, areas, iscrowd = [], [], [], [], []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            x1 = max(0.0, x)
            y1 = max(0.0, y)
            x2 = min(float(W), x + w)
            y2 = min(float(H), y + h)
            if x2 - x1 < 1.0 or y2 - y1 < 1.0:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(ann["category_id"])
            areas.append(ann.get("area", (x2 - x1) * (y2 - y1)))
            iscrowd.append(int(ann.get("iscrowd", 0)))

            seg = ann.get("segmentation", None)
            if seg and isinstance(seg, list) and len(seg) > 0:
                rles = mask_utils.frPyObjects(seg, H, W)
                rle = mask_utils.merge(rles)
                masks_list.append(
                    torch.as_tensor(mask_utils.decode(rle), dtype=torch.uint8)
                )
            elif seg and isinstance(seg, dict):
                masks_list.append(
                    torch.as_tensor(mask_utils.decode(seg), dtype=torch.uint8)
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
            if masks_list:
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
                "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
            }
            if len(masks_list) == len(boxes):
                target["masks"] = tv_tensors.Mask(torch.stack(masks_list))

        image = tv_tensors.Image(image)
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target


class FetusDetDataModule:
    """Plain data-module for the Fetus COCO-triplet dataset.

    See :class:`.coco_image.SaUSCocoDetDataModule` for the recipe-side
    contract this class fulfils.

    Parameters
    ----------
    data_root : str
        Root directory containing ``images/`` and ``annotations/``.
    image_size : int
        Target image resolution.
    multiplier : int
        Repeat training samples N times per epoch (for small datasets).
    """

    def __init__(
        self,
        data_root: str,
        image_size: int = 1024,
        multiplier: int = 1,
    ):
        self.data_root = Path(data_root)
        self.image_size = int(image_size)
        self.multiplier = max(int(multiplier), 1)

        self.train_dataset: Optional[Dataset] = None
        self.val_datasets: List[Dataset] = []
        self.test_datasets: List[Dataset] = []
        self.val_dataset_names: List[str] = []
        self.test_dataset_names: List[str] = []

        # Eagerly read the train annotation (if present) for num_classes
        ann_dir = self.data_root / "annotations"
        train_ann = ann_dir / "instances_train.json"
        if train_ann.exists():
            with open(train_ann) as f:
                coco = json.load(f)
            self.categories: Dict[int, str] = {
                cat["id"]: cat["name"] for cat in coco.get("categories", [])
            }
        else:
            self.categories = {}
        self.num_classes = max(len(self.categories), 1)
        self.dataset_name = "Fetus"

    def setup(self) -> None:
        ann_dir = self.data_root / "annotations"
        train_t = get_train_transforms(self.image_size)
        val_t = get_val_transforms(self.image_size)

        train_ann = ann_dir / "instances_train.json"
        val_ann = ann_dir / "instances_val.json"
        test_ann = ann_dir / "instances_test.json"

        if train_ann.exists():
            ds = FetusDetDataset(
                data_root=str(self.data_root),
                ann_file=str(train_ann),
                transforms=train_t,
            )
            if self.multiplier > 1:
                self.train_dataset = ConcatDataset([ds] * self.multiplier)
            else:
                self.train_dataset = ds

        if val_ann.exists():
            self.val_datasets = [FetusDetDataset(
                data_root=str(self.data_root),
                ann_file=str(val_ann),
                transforms=val_t,
            )]
            self.val_dataset_names = ["Fetus-val"]

        if test_ann.exists():
            self.test_datasets = [FetusDetDataset(
                data_root=str(self.data_root),
                ann_file=str(test_ann),
                transforms=val_t,
            )]
            self.test_dataset_names = ["Fetus-test"]

        logger.info(
            f"FetusDetDataModule setup: "
            f"train={len(self.train_dataset) if self.train_dataset else 0}, "
            f"val={sum(len(d) for d in self.val_datasets)}, "
            f"test={sum(len(d) for d in self.test_datasets)}"
        )
