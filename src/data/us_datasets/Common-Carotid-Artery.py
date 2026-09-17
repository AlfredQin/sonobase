"""Common Carotid Artery Ultrasound Images dataset preprocessing.

Dataset: Carotid artery ultrasound segmentation
Source: https://data.mendeley.com/datasets/d4xt63mgjm/1

Structure:
- US images/{id}.png: Ultrasound images
- Expert mask images/{id}.png: Binary segmentation masks

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

DATASET_NAME = "Common-Carotid-Artery"

CATEGORIES = [
    {"supercategory": "vessel", "id": 1, "name": "carotid_artery"},
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


def collect_samples(data_dir: Path) -> List[Dict]:
    """Collect image-mask pairs."""
    samples = []

    # Handle nested folder structure
    if (data_dir / "Common Carotid Artery Ultrasound Images").exists():
        data_dir = data_dir / "Common Carotid Artery Ultrasound Images"

    # Find image and mask directories
    img_dir = None
    mask_dir = None

    for candidate in ["US images", "images", "US_images"]:
        if (data_dir / candidate).exists():
            img_dir = data_dir / candidate
            break

    for candidate in ["Expert mask images", "masks", "Expert_mask_images"]:
        if (data_dir / candidate).exists():
            mask_dir = data_dir / candidate
            break

    if img_dir is None or mask_dir is None:
        return samples

    # Build mapping by image ID
    img_files = {f.stem: f for f in img_dir.glob("*.png")}
    mask_files = {f.stem: f for f in mask_dir.glob("*.png")}

    for img_id, img_path in sorted(img_files.items()):
        if img_id in mask_files:
            samples.append({
                "image_path": img_path,
                "mask_path": mask_files[img_id],
                "image_id": img_id,
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

            mask = rgb2gray(cv2.imread(str(sample["mask_path"])))
            mask_binary = np.zeros_like(mask, dtype=np.uint8)
            mask_binary[mask > 127] = 1

            if not np.any(mask_binary > 0):
                continue

            # Clean up image_id for valid filename
            clean_id = sample["image_id"].replace(" ", "_")
            image_name = f"{clean_id}"

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
