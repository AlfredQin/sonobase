"""Detection / instance-segmentation datasets for US-RFDETR.

Three flavours covering the 9 datasets in scope:

* :class:`SaUSCocoDetDataModule` (+ :class:`SaUSCocoDetDataset`) —
  SaUS-layout sub-datasets with category labels and SA-1B (image) or
  SA-V (video) RLE masks. Covers BUS-BRA, DDTI, FUGC, KidneyUS,
  LUMINOUS, MMOTU-3d, ACOUSLIC.
* :class:`FetusDetDataModule` (+ :class:`FetusDetDataset`) — standard
  COCO-triplet for the Fetus dataset.
* :class:`CVANetDataModule` (+ :class:`CVANetDataset`) — COCO-video
  format with deterministic video-level train/val split, for CVA-Net.

All three DataModules expose the same recipe-side contract (see the
class docstrings): ``setup()`` populates ``train_dataset``,
``val_datasets``, ``test_datasets``, ``val_dataset_names``,
``test_dataset_names``; ``num_classes``, ``dataset_name``,
``categories`` are available pre-setup.

The shared transforms (``get_train_transforms``, ``get_val_transforms``,
``SanitizeBoundingBoxes``, ``detection_collate_fn``) live in
:mod:`.transforms`.
"""

from nemo_cv.components.datasets.us_rfdetr.coco_image import (
    SaUSCocoDetDataModule,
    SaUSCocoDetDataset,
)
from nemo_cv.components.datasets.us_rfdetr.coco_standard import (
    FetusDetDataModule,
    FetusDetDataset,
)
from nemo_cv.components.datasets.us_rfdetr.coco_video import (
    CVANetDataModule,
    CVANetDataset,
)
from nemo_cv.components.datasets.us_rfdetr.transforms import (
    SanitizeBoundingBoxes,
    detection_collate_fn,
    get_train_transforms,
    get_val_transforms,
)

__all__ = [
    "SaUSCocoDetDataModule",
    "SaUSCocoDetDataset",
    "FetusDetDataModule",
    "FetusDetDataset",
    "CVANetDataModule",
    "CVANetDataset",
    "SanitizeBoundingBoxes",
    "detection_collate_fn",
    "get_train_transforms",
    "get_val_transforms",
]
