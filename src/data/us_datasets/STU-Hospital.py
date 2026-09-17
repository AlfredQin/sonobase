"""STU-Hospital (Breast Ultrasound Dataset) preprocessing.

Dataset: Breast tumor ultrasound segmentation from STU Hospital
Source: https://github.com/xbhlk/STU-Hospital

Data format:
- Hospital/image_{id}.png: Ultrasound images
- Hospital/mask_{id}.png: Binary segmentation masks

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

DATASET_NAME = "STU-Hospital"

CATEGORIES = [
    {"supercategory": "nodule", "id": 1, "name": "benign_tumour"},
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


def imresize_like(img: np.ndarray, target: np.ndarray, interpolation: str = "nearest") -> np.ndarray:
    """Resize image to match target shape."""
    interp_map = {
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
    }
    h, w = target.shape[:2]
    return cv2.resize(img, (w, h), interpolation=interp_map.get(interpolation, cv2.INTER_NEAREST))


def collect_samples(data_dir: Path) -> List[Dict]:
    """Collect image-mask pairs from the dataset."""
    samples = []

    # Handle nested folder structure
    hospital_dir = None
    for candidate in [
        data_dir / "Hospital",
        data_dir / "STU-Hospital" / "Hospital",
        data_dir / "STU-Hospital-master" / "Hospital",
    ]:
        if candidate.exists():
            hospital_dir = candidate
            break
    
    # Also check if files are directly in STU-Hospital folder
    if hospital_dir is None:
        for candidate in [data_dir / "STU-Hospital", data_dir / "STU-Hospital-master", data_dir]:
            if (candidate / "image_1.png").exists() or any(candidate.glob("image_*.png")):
                hospital_dir = candidate
                break

    if hospital_dir is None:
        return samples

    # Find all mask files and pair with images
    mapping = defaultdict(dict)
    for file in hospital_dir.glob("*.png"):
        filename = file.name
        if filename.startswith("mask_"):
            idx = filename.replace("mask_", "").replace(".png", "")
            mapping[idx]["mask"] = file
        elif filename.startswith("image_"):
            idx = filename.replace("image_", "").replace(".png", "")
            mapping[idx]["image"] = file
        elif filename.startswith("Test_Image_"):
            # Alternate naming pattern
            idx = filename.replace("Test_Image_", "").replace(".png", "")
            mapping[idx]["image"] = file

    for idx, paths in sorted(mapping.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0):
        if "image" in paths and "mask" in paths:
            samples.append({
                "image_path": paths["image"],
                "mask_path": paths["mask"],
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

            mask = rgb2gray(cv2.imread(str(sample["mask_path"]))).astype(np.uint8)

            # Resize mask if needed
            if mask.shape[:2] != (h, w):
                mask = imresize_like(mask, image, interpolation="nearest")

            # Binarize mask
            mask_binary = np.zeros((h, w), dtype=np.uint8)
            mask_binary[mask == 255] = 1

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

            if (idx + 1) % 20 == 0:
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
