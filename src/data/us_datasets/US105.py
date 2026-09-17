"""US105 (Ultrasound 105 Images Dataset) preprocessing.

Dataset: 2D ultrasound images with tumor segmentation masks
Source: https://www.researchgate.net/publication/329586355_100_2D_US_Images_and_Tumor_Segmentation_Masks

Data format:
- 105 US Images/{id}.png: Ultrasound images
- 105 US Masks/{id} G man.png: Binary tumor masks

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

DATASET_NAME = "US105"

CATEGORIES = [
    {"supercategory": "tumor", "id": 1, "name": "tumor"},
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
    """Collect image-mask pairs from the dataset."""
    samples = []

    # Handle nested folder structure
    images_dir = None
    masks_dir = None
    
    for candidate_base in [data_dir, data_dir / DATASET_NAME]:
        # Try different folder name patterns
        for img_folder in ["105 US Images", "105_US_Images", "Images"]:
            candidate_img = candidate_base / img_folder
            if candidate_img.exists():
                images_dir = candidate_img
                break
        
        for mask_folder in ["105 US Masks", "105_US_Masks", "Masks"]:
            candidate_mask = candidate_base / mask_folder
            if candidate_mask.exists():
                masks_dir = candidate_mask
                break
        
        if images_dir and masks_dir:
            break

    if images_dir is None or masks_dir is None:
        return samples

    # Build mapping from image ID to mask file
    # Mask files have pattern: "{id} G man.png"
    mask_mapping = {}
    for mask_file in masks_dir.glob("*.png"):
        # Extract numeric ID from mask filename
        name = mask_file.stem
        # Handle patterns like "001 G man" or just "001"
        parts = name.split()
        if parts:
            try:
                idx = str(int(parts[0]))
                mask_mapping[idx] = mask_file
            except ValueError:
                pass

    # Match images to masks
    for img_file in sorted(images_dir.glob("*.png")):
        try:
            # Extract numeric ID from image filename
            idx = str(int(img_file.stem))
            if idx in mask_mapping:
                samples.append({
                    "image_path": img_file,
                    "mask_path": mask_mapping[idx],
                    "image_id": idx,
                })
        except ValueError:
            continue

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
            
            # Extract nested zips (105USImages.zip, 105USMasks.zip)
            for nested_zip in data_dir.glob("*.zip"):
                logger.info(f"Extracting nested archive: {nested_zip.name}")
                shutil.unpack_archive(str(nested_zip), str(data_dir))
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

            mask = rgb2gray(cv2.imread(str(sample["mask_path"]))).astype(np.uint8)

            # Binarize mask (divide by 255 to get 0/1)
            mask_binary = np.zeros((h, w), dtype=np.uint8)
            mask_binary[mask > 127] = 1

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
