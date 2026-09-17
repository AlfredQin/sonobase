"""S1 (Breast Ultrasound Dataset) preprocessing.

Dataset: Breast tumor ultrasound classification and segmentation
Source: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8205136/

Data format:
- Data/TrainingDataSet/BreastTumourImages/{id}.jpg: Training images
- Data/TrainingDataSet/Expanded-3-channel-Labels/{id}.png: Training masks
- Data/TestingDataSet/BreastTumourImages/{id}.jpg: Test images
- Data/TestingDataSet/Expanded-3-channel-Labels/{id}.png: Test masks

Mask values:
- 127 -> benign_tumour (category 1)
- 255 -> malignant_tumour (category 2)

Output: SAM2-compatible format (SA-1B style)
"""

import argparse
import logging
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ..utils.sam2_annotator import SAM2VideoAnnotator

DATASET_NAME = "S1"

CATEGORIES = [
    {"supercategory": "nodule", "id": 1, "name": "benign_tumour"},
    {"supercategory": "nodule", "id": 2, "name": "malignant_tumour"},
]


def parse_args():
    parser = argparse.ArgumentParser(description=f"Convert {DATASET_NAME} to SAM2 format")
    parser.add_argument(
        "--path",
        type=str,
        default=f"/mnt/data/Dataset/UltraSound/Raw/{DATASET_NAME}.zip",
        help="Path to dataset",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=f"/mnt/data/Dataset/SaUS/{DATASET_NAME}",
        help="Output directory",
    )
    parser.add_argument("--format", type=str, choices=["sa-v", "sa-1b"], default="sa-1b")
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "test", "all"],
        default="all",
        help="Which split to process",
    )
    parser.add_argument("--save-viz", action="store_true")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--zip", action="store_true")
    parser.add_argument("--delete", action="store_true")
    return parser.parse_args()


def rgb2gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def collect_samples(data_dir: Path, split: str = "all") -> List[Dict]:
    """Collect image-mask pairs."""
    samples = []

    # Handle nested folder structure
    if (data_dir / "S1").exists():
        data_dir = data_dir / "S1"
    if (data_dir / "Data").exists():
        data_dir = data_dir / "Data"

    split_mapping = {
        "train": "TrainingDataSet",
        "test": "TestingDataSet",
    }

    splits = ["train", "test"] if split == "all" else [split]

    for split_name in splits:
        split_folder = split_mapping.get(split_name)
        if not split_folder:
            continue

        split_dir = data_dir / split_folder
        if not split_dir.exists():
            continue

        images_dir = split_dir / "BreastTumourImages"
        # Prefer Expanded-3-channel-Labels over General-1-channel-Labels
        masks_dir = split_dir / "Expanded-3-channel-Labels"
        if not masks_dir.exists():
            masks_dir = split_dir / "General-1-channel-Labels"

        if not images_dir.exists() or not masks_dir.exists():
            continue

        for img_file in sorted(images_dir.glob("*.jpg")):
            img_id = img_file.stem
            mask_file = masks_dir / f"{img_id}.png"

            if mask_file.exists():
                samples.append({
                    "image_path": img_file,
                    "mask_path": mask_file,
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

    video_split = args.split if args.split != "all" else "train"

    annotator = SAM2VideoAnnotator(
        out_dir=save_path,
        dataset_name=DATASET_NAME,
        categories=CATEGORIES,
        save_visualization=args.save_viz,
        video_environment="ultrasound",
        video_split=video_split,
    )

    logger.info(f"Output format: {args.format}, split: {args.split}")

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

        samples = collect_samples(data_dir, args.split)
        logger.info(f"Found {len(samples)} samples")

        if args.max_images:
            samples = samples[:args.max_images]

        processed = 0
        for idx, sample in enumerate(samples):
            image = cv2.imread(str(sample["image_path"]))
            if image is None:
                continue
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]

            mask = rgb2gray(cv2.imread(str(sample["mask_path"]))).astype(np.uint8)
            
            # Convert mask values to category IDs
            # 127 -> benign (1), 255 -> malignant (2)
            mask_labeled = np.zeros((h, w), dtype=np.uint8)
            mask_labeled[mask == 127] = 1  # benign
            mask_labeled[mask == 255] = 2  # malignant

            if not np.any(mask_labeled > 0):
                continue

            image_name = f"{sample['split']}_{sample['image_id']}"
            annotator.add_2d(
                image_rgb,
                mask_labeled,
                video_name=image_name,
                format=args.format,
                metadata={
                    "image_id": sample["image_id"],
                    "split": sample["split"],
                    "dataset": DATASET_NAME,
                },
            )
            processed += 1

            if (idx + 1) % 100 == 0:
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
