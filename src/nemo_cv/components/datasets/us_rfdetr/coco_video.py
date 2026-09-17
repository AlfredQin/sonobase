"""COCO-video detection dataset for US-RFDETR (CVA-Net).

CVA-Net layout::

    CVA_Net/
      rawframes/
        benign/<video_id>/<frame>.png
        malignant/<video_id>/<frame>.png
      trainval.json     # COCO-video format (154 videos)
      test.json         # COCO-video format (32 videos)

Annotation format (COCO-video)::

    {
      "categories": [{"id": 1, "name": "benign"}, {"id": 2, "name": "malignant"}],
      "videos":     [{"id": 70, "name": "malignant/x...", ...}],
      "images":     [{"id": ..., "file_name": "...", "video_id": 70,
                       "frame_id": 0, ...}],
      "annotations":[{"id": ..., "image_id": ..., "bbox": [x,y,w,h],
                       "category_id": 2, ...}]
    }

The train/val split is done at **video level** (deterministic seeded
shuffle of `video_ids`) to avoid frame-level leakage between splits.

Three corrupted videos that the DenseUS reference excluded are also
excluded here: ``x66ef02e7f1b9a0ef``, ``x63c9ba1377f35bf6``,
``x5a1c46ec6377e946``. One further train video, ``x1282311c38f808f``, is
excluded because it is a content duplicate of a test video (train/test
leakage — see ``_CVA_NET_TEST_DUP_VIDEOS``).

Ported from ``USSam/projects/DenseUS/cva_net.py`` with the same
no-Lightning, recipe-side-DataLoader-construction conventions as the
sibling modules.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors

from nemo_cv.components.datasets.us_rfdetr.transforms import (
    get_train_transforms,
    get_val_transforms,
)

logger = logging.getLogger(__name__)


# Three CVA-Net videos that the reference DenseUS run excluded due to
# corrupted frames / annotation issues — preserved here for parity.
_CVA_NET_CORRUPT_VIDEOS = (
    "x66ef02e7f1b9a0ef",
    "x63c9ba1377f35bf6",
    "x5a1c46ec6377e946",
)

# Train-pool video that is a byte-for-byte content duplicate of a *test* video:
# benign/x1282311c38f808f == malignant/x9c1c965d274c457 (106 of 110 frames are
# pixel-identical, with conflicting class labels). Dropping it from the train
# pool prevents train->test leakage; the canonical test video is left untouched
# (this id appears only in trainval.json). Found by the detection train/test
# leakage audit — see docs/DATA_SPLITS.md.
_CVA_NET_TEST_DUP_VIDEOS = (
    "x1282311c38f808f",
)

# Both reasons feed the same frame-name filter in CVANetDataset.
_CVA_NET_EXCLUDE_VIDEOS = _CVA_NET_CORRUPT_VIDEOS + _CVA_NET_TEST_DUP_VIDEOS


class CVANetDataset(Dataset):
    """CVA-Net detection dataset (COCO-video format)."""

    def __init__(
        self,
        data_root: str,
        ann_file: str,
        video_ids: Optional[List[int]] = None,
        transforms: Optional[Callable] = None,
        include_empty: bool = False,
    ):
        self.data_root = Path(data_root)
        self.rawframes_dir = self.data_root / "rawframes"
        self.transforms = transforms

        with open(ann_file, "r") as f:
            coco = json.load(f)

        self.categories = {cat["id"]: cat["name"] for cat in coco["categories"]}
        self.num_classes = len(self.categories)

        if video_ids is not None:
            video_ids_set = set(video_ids)
            images = [img for img in coco["images"] if img["video_id"] in video_ids_set]
        else:
            images = coco["images"]

        # Drop the corrupted-video frames
        images = [
            img for img in images
            if not any(bad in img["file_name"] for bad in _CVA_NET_EXCLUDE_VIDEOS)
        ]

        self.images = {img["id"]: img for img in images}

        # Per-image annotation index
        valid_img_ids = set(self.images.keys())
        self.img_anns: Dict[int, List[Dict]] = {}
        for ann in coco["annotations"]:
            if ann["image_id"] in valid_img_ids:
                self.img_anns.setdefault(ann["image_id"], []).append(ann)

        # Drop frames without annotations — empty frames create only false
        # positives in mAP.
        if not include_empty:
            before = len(self.images)
            self.images = {
                iid: img for iid, img in self.images.items()
                if iid in self.img_anns
            }
            logger.info(
                f"CVANetDataset: filtered {before - len(self.images)} "
                f"empty frames (include_empty=False)"
            )

        self.image_ids = sorted(self.images.keys())
        n_with_ann = sum(1 for iid in self.image_ids if iid in self.img_anns)
        logger.info(
            f"CVANetDataset: {len(self.image_ids)} images "
            f"({n_with_ann} with annotations) from {ann_file}"
        )

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]

        img_path = self.rawframes_dir / img_info["file_name"]
        image = Image.open(img_path).convert("RGB")
        W, H = image.size

        anns = self.img_anns.get(img_id, [])
        boxes, labels, areas, iscrowd = [], [], [], []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            # Clamp (CVA-Net has some negative-coord boxes)
            x1 = max(0.0, x)
            y1 = max(0.0, y)
            x2 = min(float(W), x + w)
            y2 = min(float(H), y + h)
            if x2 - x1 < 1.0 or y2 - y1 < 1.0:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(ann["category_id"])
            areas.append((x2 - x1) * (y2 - y1))
            iscrowd.append(int(ann.get("iscrowd", False)))

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

        image = tv_tensors.Image(image)
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target


class CVANetDataModule:
    """Plain data-module for CVA-Net video detection.

    Splits ``trainval.json`` at the video level using a deterministic
    seeded shuffle so no two frames from the same video end up in
    different splits. ``test.json`` is used as-is.

    See :class:`.coco_image.SaUSCocoDetDataModule` for the recipe-side
    contract.

    Parameters
    ----------
    data_root : str
        Root directory containing ``rawframes/``, ``trainval.json``,
        ``test.json``.
    image_size : int
        Target image resolution.
    val_ratio : float
        Fraction of trainval videos used for validation (default 0.15).
    seed : int
        Deterministic-shuffle seed for the video-level split.
    """

    def __init__(
        self,
        data_root: str,
        image_size: int = 1024,
        val_ratio: float = 0.15,
        seed: int = 42,
    ):
        self.data_root = Path(data_root)
        self.image_size = int(image_size)
        self.val_ratio = float(val_ratio)
        self.seed = int(seed)

        self.train_dataset: Optional[Dataset] = None
        self.val_datasets: List[Dataset] = []
        self.test_datasets: List[Dataset] = []
        self.val_dataset_names: List[str] = []
        self.test_dataset_names: List[str] = []

        # Eager num_classes (read from trainval.json)
        trainval_file = self.data_root / "trainval.json"
        if trainval_file.exists():
            with open(trainval_file) as f:
                tv = json.load(f)
            self.categories: Dict[int, str] = {
                cat["id"]: cat["name"] for cat in tv.get("categories", [])
            }
        else:
            self.categories = {}
        self.num_classes = max(len(self.categories), 1)
        self.dataset_name = "CVA-Net"

    def setup(self) -> None:
        trainval_file = self.data_root / "trainval.json"
        test_file = self.data_root / "test.json"

        train_t = get_train_transforms(self.image_size)
        val_t = get_val_transforms(self.image_size)

        with open(trainval_file, "r") as f:
            trainval = json.load(f)

        all_videos = trainval["videos"]
        video_ids = [v["id"] for v in all_videos]
        rng = random.Random(self.seed)
        rng.shuffle(video_ids)

        n_val = max(1, int(len(video_ids) * self.val_ratio))
        val_video_ids = video_ids[:n_val]
        train_video_ids = video_ids[n_val:]

        logger.info(
            f"CVA-Net video-level split: "
            f"{len(train_video_ids)} train / {len(val_video_ids)} val videos "
            f"(from {len(video_ids)} total, val_ratio={self.val_ratio}, seed={self.seed})"
        )

        self.train_dataset = CVANetDataset(
            data_root=str(self.data_root),
            ann_file=str(trainval_file),
            video_ids=train_video_ids,
            transforms=train_t,
        )
        self.val_datasets = [CVANetDataset(
            data_root=str(self.data_root),
            ann_file=str(trainval_file),
            video_ids=val_video_ids,
            transforms=val_t,
        )]
        self.val_dataset_names = ["CVA-Net-val"]

        if test_file.exists():
            self.test_datasets = [CVANetDataset(
                data_root=str(self.data_root),
                ann_file=str(test_file),
                video_ids=None,
                transforms=val_t,
            )]
            self.test_dataset_names = ["CVA-Net-test"]

        logger.info(
            f"CVANetDataModule setup: "
            f"train={len(self.train_dataset)}, "
            f"val={sum(len(d) for d in self.val_datasets)}, "
            f"test={sum(len(d) for d in self.test_datasets)}"
        )
