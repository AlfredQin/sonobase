"""HC18 (Head Circumference) dataset preprocessing.

Dataset: Fetal head ultrasound segmentation from Grand Challenge
Source: https://hc18.grand-challenge.org/

Data format:
- training_set.zip contains:
  - {id}_HC.png: Ultrasound images
  - {id}_HC_Annotation.png: Border-only annotations (need fill)

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
from scipy import ndimage

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ..utils.sam2_annotator import SAM2VideoAnnotator

DATASET_NAME = "HC18"

CATEGORIES = [
    {"supercategory": "fetal", "id": 1, "name": "fetal_head"},
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
    parser.add_argument("--save-viz", action="store_true")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--zip", action="store_true")
    parser.add_argument("--delete", action="store_true")
    return parser.parse_args()


def rgb2gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def fill_mask_labeled(mask: np.ndarray) -> np.ndarray:
    """Fill holes in labeled mask (for border-only annotations)."""
    unique_labels = np.unique(mask)
    filled_mask = np.zeros_like(mask)

    for label in unique_labels:
        if label == 0:
            continue
        binary_mask = mask == label
        filled_binary = ndimage.binary_fill_holes(binary_mask)
        filled_mask[filled_binary] = label

    return filled_mask


def collect_samples(data_dir: Path) -> List[Dict]:
    """Collect image-mask pairs."""
    samples = []

    # Find PNG files
    png_files = list(data_dir.rglob("*.png"))

    # Build mapping by ID
    mapping = defaultdict(dict)
    for f in png_files:
        filename = f.stem
        parts = filename.split("_")[:2]
        idx = "_".join(parts)
        
        if "Annotation" in filename:
            mapping[idx]["mask"] = f
        else:
            mapping[idx]["img"] = f

    for idx, data in mapping.items():
        if "img" in data and "mask" in data:
            samples.append({
                "image_path": data["img"],
                "mask_path": data["mask"],
                "image_id": idx,
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
            
            # Extract nested training_set.zip
            training_zip = Path(temp_dir) / "training_set.zip"
            if training_zip.exists():
                logger.info("Extracting training_set.zip")
                shutil.unpack_archive(str(training_zip), Path(temp_dir) / "training_set")
            
            data_dir = Path(temp_dir)
        else:
            data_dir = data_path

        samples = collect_samples(data_dir)
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

            # Load mask (border-only, value 255)
            mask = rgb2gray(cv2.imread(str(sample["mask_path"]))).astype(np.uint8)
            
            # Fill the border to get solid mask
            mask_filled = fill_mask_labeled(mask)
            mask_binary = (mask_filled // 255).astype(np.uint8)

            if not np.any(mask_binary > 0):
                continue

            image_name = f"{sample['image_id']}"
            annotator.add_2d(
                image_rgb,
                mask_binary,
                video_name=image_name,
                format=args.format,
                metadata={
                    "image_id": sample["image_id"],
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
