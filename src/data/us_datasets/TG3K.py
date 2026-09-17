"""TG3K (Thyroid Gland 3K) dataset preprocessing.

Dataset: Thyroid gland ultrasound segmentation
Source: https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/TN3K.md

Data format:
- tg3k/thyroid-image/{id}.jpg: Ultrasound images
- tg3k/thyroid-mask/{id}.jpg: Binary masks

Category: thyroid_gland

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

DATASET_NAME = "TG3K"

CATEGORIES = [
    {"supercategory": "gland", "id": 1, "name": "thyroid_gland"},
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
    if (data_dir / "Thyroid Dataset").exists():
        data_dir = data_dir / "Thyroid Dataset"
    
    if (data_dir / "tg3k").exists():
        data_dir = data_dir / "tg3k"

    images_dir = data_dir / "thyroid-image"
    masks_dir = data_dir / "thyroid-mask"

    if not images_dir.exists() or not masks_dir.exists():
        return samples

    for img_file in sorted(images_dir.glob("*.jpg")):
        img_id = img_file.stem
        mask_file = masks_dir / f"{img_id}.jpg"

        if mask_file.exists():
            samples.append({
                "image_path": img_file,
                "mask_path": mask_file,
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
            h, w = image.shape[:2]

            mask = rgb2gray(cv2.imread(str(sample["mask_path"])))
            
            # Binarize mask (threshold due to JPG compression)
            mask_binary = np.zeros((h, w), dtype=np.uint8)
            mask_binary[mask >= 150] = 1

            # Remove small connected components (e.g. machine-type text in top-left)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                mask_binary, connectivity=8
            )
            if num_labels > 1:
                # Keep only components whose area >= 1% of image area
                min_area = int(h * w * 0.01)
                cleaned = np.zeros_like(mask_binary)
                for lbl in range(1, num_labels):  # skip background (0)
                    if stats[lbl, cv2.CC_STAT_AREA] >= min_area:
                        cleaned[labels == lbl] = 1
                mask_binary = cleaned

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

            if (idx + 1) % 500 == 0:
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
