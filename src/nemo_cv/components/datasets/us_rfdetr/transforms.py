"""Shared detection / instance-segmentation transforms for US-RFDETR.

The DenseUS source had three identical copies of these — one in each of
``data_module.py``, ``saus_coco_det.py``, ``cva_net.py``. We fold them
into a single module here to remove the drift risk; the dataset wrappers
import from this file.

All transforms operate on torchvision-v2 ``(image, target)`` pairs where
``image`` is a ``tv_tensors.Image`` and ``target`` is a dict with
``boxes`` (``tv_tensors.BoundingBoxes`` in XYXY format), ``labels``,
optional ``masks`` (``tv_tensors.Mask``), ``image_id``, ``area``,
``iscrowd``.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
from torchvision.transforms import v2 as T


__all__ = [
    "SanitizeBoundingBoxes",
    "detection_collate_fn",
    "get_train_transforms",
    "get_val_transforms",
]


class SanitizeBoundingBoxes(nn.Module):
    """Drop boxes whose width or height fell below ``min_size`` (in pixels)
    after a geometric augmentation. Also filters the parallel ``labels``,
    ``masks``, ``area``, ``iscrowd`` tensors when present so the per-box
    arrays stay aligned.

    Boxes with zero area would cause RF-DETR's GIoU loss to NaN; running
    this *immediately after* `RandomAffine`/`RandomResizedCrop`/etc. is
    the standard mitigation.
    """

    def __init__(self, min_size: float = 1.0):
        super().__init__()
        self.min_size = float(min_size)

    def forward(self, image, target):
        if "boxes" not in target or target["boxes"].shape[0] == 0:
            return image, target
        boxes = target["boxes"]
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        keep = (w >= self.min_size) & (h >= self.min_size)
        if keep.all():
            return image, target
        n = boxes.shape[0]
        target = {
            k: (v[keep] if isinstance(v, torch.Tensor) and v.shape[0] == n else v)
            for k, v in target.items()
        }
        return image, target


def detection_collate_fn(batch):
    """Collate into ``(List[Tensor], List[dict])``.

    Detection batches cannot be stacked because images are different
    sizes (post-Resize they're square 1024×1024 here, but in general
    detection collate is left as lists for compatibility with NestedTensor).
    """
    images, targets = zip(*batch)
    return list(images), list(targets)


def get_train_transforms(image_size: int = 1024):
    """Standard detection training augmentations.

    Mirror the DenseUS reference run: square resize → horizontal flip →
    mild affine (rotation/translate/scale) → degenerate-box sanitisation
    → photometric distort → grayscale → blur → tensor + ImageNet
    normalise.

    The augmentations are deliberately light — ultrasound has a strong
    speckle / orientation bias and aggressive geometric augmentation can
    invalidate clinically meaningful spatial relationships.
    """
    return T.Compose([
        T.Resize((image_size, image_size), antialias=True),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        SanitizeBoundingBoxes(min_size=2.0),
        T.RandomPhotometricDistort(p=0.5),
        T.RandomGrayscale(p=0.1),
        T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms(image_size: int = 1024):
    """Eval-time transforms: square resize + tensor + ImageNet normalise."""
    return T.Compose([
        T.Resize((image_size, image_size), antialias=True),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
