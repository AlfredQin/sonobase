"""FUGC (Fetal Ultrasound Grand Challenge) dataset preprocessing.

Dataset: Fetal ultrasound segmentation with two anatomical structures
Source: FUGC.zip → FUGC (Dataset)/dataset/{split}/

Data format:
- dataset/train/labeled_data/images/{id}.png: RGB ultrasound images (336x544)
- dataset/train/labeled_data/labels/{id}.png: Multi-class masks
- dataset/val/images/{id}.png, dataset/val/labels/{id}.png
- dataset/test/images/{id}.png, dataset/test/labels/{id}.png
- dataset/train/unlabeled_data/images/{id}.png: Unlabeled images (not used)

Mask pixel values:
    0 = background
    1 = structure_1 (fetal structure, larger area)
    2 = structure_2 (fetal structure, smaller area)

Stats: 440 labeled images (train: 50, val: 90, test: 300), all have both classes

Output: SAM2-compatible format (SA-1B style)
"""

import argparse
import logging
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ..utils.sam2_annotator import SAM2VideoAnnotator

DATASET_NAME = "FUGC"

CATEGORIES = [
    {"supercategory": "fetal", "id": 1, "name": "structure_1"},
    {"supercategory": "fetal", "id": 2, "name": "structure_2"},
]


def parse_args():
    parser = argparse.ArgumentParser(description=f"Convert {DATASET_NAME} to SAM2 format")
    parser.add_argument(
        "--path",
        type=str,
        default=f"/mnt/data/Dataset/UltraSound/Raw/{DATASET_NAME}.zip",
        help="Path to dataset (zip file or directory)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=f"/mnt/data/Dataset/SaUS/{DATASET_NAME}",
        help="Output directory",
    )
    parser.add_argument("--format", type=str, choices=["sa-v", "sa-1b"], default="sa-1b")
    parser.add_argument("--save-viz", action="store_true")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--zip", action="store_true")
    parser.add_argument("--delete", action="store_true")
    return parser.parse_args()


def collect_samples(data_dir: Path) -> List[Dict]:
    """Collect labeled image-mask pairs from all splits.

    FUGC.zip extracts to FUGC (Dataset)/dataset/{split}/.
    Only labeled images (with corresponding labels) are collected.
    """
    samples = []

    # Handle nested folder structure from zip extraction
    if (data_dir / "FUGC (Dataset)").exists():
        data_dir = data_dir / "FUGC (Dataset)"

    dataset_dir = data_dir / "dataset"
    if not dataset_dir.exists():
        # Maybe data_dir is already the dataset dir
        dataset_dir = data_dir

    # Splits with labeled data (train/labeled_data, val, test)
    split_paths = [
        ("train", dataset_dir / "train" / "labeled_data"),
        ("val", dataset_dir / "val"),
        ("test", dataset_dir / "test"),
    ]

    for split_name, split_dir in split_paths:
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"

        if not images_dir.exists() or not labels_dir.exists():
            continue

        for img_file in sorted(images_dir.glob("*.png")):
            img_id = img_file.stem
            label_file = labels_dir / f"{img_id}.png"

            if label_file.exists():
                samples.append({
                    "image_path": img_file,
                    "mask_path": label_file,
                    "image_id": img_id,
                    "split": split_name,
                })

    return samples


def main():
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    save_path = Path(args.save_dir)
    data_path = Path(args.path)
    use_temp_dir = data_path.suffix.lower() == ".zip"

    annotator = SAM2VideoAnnotator(
        out_dir=save_path,
        dataset_name=DATASET_NAME,
        categories=CATEGORIES,
        save_visualization=args.save_viz,
        video_environment="ultrasound",
        video_split="train",
    )

    logger.info(f"Output format: {args.format}")

    if use_temp_dir:
        temp_context = tempfile.TemporaryDirectory()
    else:
        from contextlib import nullcontext
        temp_context = nullcontext(str(data_path))

    with temp_context as temp_dir:
        if use_temp_dir:
            logger.info(f"Extracting {data_path} to {temp_dir}")
            shutil.unpack_archive(str(data_path), temp_dir)
            data_dir = Path(temp_dir)
        else:
            data_dir = data_path

        samples = collect_samples(data_dir)
        logger.info(f"Found {len(samples)} labeled samples")

        if args.max_images:
            samples = samples[:args.max_images]

        processed = 0
        for idx, sample in enumerate(samples):
            image = cv2.imread(str(sample["image_path"]))
            if image is None:
                continue
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            mask = cv2.imread(str(sample["mask_path"]), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue

            # Mask values are already 0/1/2 matching category IDs
            mask = mask.astype(np.uint8)

            if not np.any(mask > 0):
                continue

            image_name = f"{sample['split']}_{sample['image_id']}"

            annotator.add_2d(
                image_rgb,
                mask,
                video_name=image_name,
                format=args.format,
                metadata={
                    "image_id": sample["image_id"],
                    "split": sample["split"],
                    "dataset": DATASET_NAME,
                },
            )
            processed += 1

            if (idx + 1) % 50 == 0:
                logger.info(f"Processed {idx + 1}/{len(samples)}")

        logger.info(f"Total processed: {processed}")

    stats = annotator.finalize()
    logger.info(f"Dataset stats: {stats}")

    if args.zip:
        shutil.make_archive(str(save_path), "zip", save_path)
        if args.delete:
            shutil.rmtree(save_path)

    logger.info("Done!")


if __name__ == "__main__":
    main()
