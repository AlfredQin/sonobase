"""LUMINOUS (Lung Ultrasound) dataset preprocessing.

Dataset: Lung ultrasound B-mode images with B-line annotations
Source: https://data.mendeley.com/datasets/8hndxv4ffy/1

Data format:
- B-mode/{patient_id}_{frame_id}_Bmode.tif: Ultrasound images
- Masks/{patient_id}_{frame_id}_Mask.tif: Binary masks (single or multiple)
- Masks/{patient_id}_{frame_id}_Mask1.tif, Mask2.tif: Multiple masks per image

Category: b_line (lung ultrasound B-line artifacts)

Output: SAM2-compatible format (SA-V for videos, SA-1B for images)
"""

import argparse
import logging
import re
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

DATASET_NAME = "LUMINOUS"

CATEGORIES = [
    {"supercategory": "artifact", "id": 1, "name": "b_line"},
]


def parse_args():
    parser = argparse.ArgumentParser(description=f"Convert {DATASET_NAME} to SAM2 format")
    parser.add_argument(
        "--path",
        type=str,
        default="/mnt/data/Dataset/UltraSound/Raw/LUMINOUS.zip",
        help="Path to dataset",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=f"/mnt/data/Dataset/SaUS/{DATASET_NAME}",
        help="Output directory",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["sa-v", "sa-1b"],
        default="sa-1b",
        help="Output format",
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


def parse_image_id(filename: str) -> str:
    """Extract patient_id_frame_id from filename like '100_1_Bmode.tif'."""
    match = re.match(r"(\d+_\d+)_Bmode\.tif", filename)
    if match:
        return match.group(1)
    return None


def collect_samples(data_dir: Path) -> List[Dict]:
    """Collect image-mask pairs from the dataset."""
    samples = []

    # Handle nested folder structure
    if (data_dir / "LUMINOUS_Database").exists():
        data_dir = data_dir / "LUMINOUS_Database"
    elif (data_dir / "LUMINOUS").exists():
        data_dir = data_dir / "LUMINOUS"

    bmode_dir = data_dir / "B-mode"
    masks_dir = data_dir / "Masks"

    if not bmode_dir.exists() or not masks_dir.exists():
        return samples

    # Build mask mapping: image_id -> list of mask files
    mask_mapping = defaultdict(list)
    for mask_file in masks_dir.glob("*.tif"):
        # Pattern: {patient_id}_{frame_id}_Mask.tif or {patient_id}_{frame_id}_Mask1.tif
        name = mask_file.stem
        match = re.match(r"(\d+_\d+)_Mask(\d*)", name)
        if match:
            image_id = match.group(1)
            mask_mapping[image_id].append(mask_file)

    # Match images to masks
    for img_file in sorted(bmode_dir.glob("*.tif")):
        image_id = parse_image_id(img_file.name)
        if image_id is None:
            continue

        if image_id in mask_mapping:
            samples.append({
                "image_path": img_file,
                "mask_paths": sorted(mask_mapping[image_id]),
                "image_id": image_id,
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

            # Combine all masks for this image
            mask_combined = np.zeros((h, w), dtype=np.uint8)
            for mask_path in sample["mask_paths"]:
                mask = rgb2gray(cv2.imread(str(mask_path)))
                if mask is not None:
                    # Binarize and combine
                    mask_binary = (mask > 127).astype(np.uint8)
                    mask_combined = np.maximum(mask_combined, mask_binary)

            if not np.any(mask_combined > 0):
                continue

            image_name = f"{sample['image_id']}"
            annotator.add_2d(
                image_rgb,
                mask_combined,
                video_name=image_name,
                format=args.format,
                metadata={
                    "image_id": sample["image_id"],
                    "num_masks": len(sample["mask_paths"]),
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
