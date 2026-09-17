"""STMUS_NDA (Segmentation of Transverse Musculoskeletal Ultrasound) preprocessing.

Dataset: Deep learning segmentation of transverse musculoskeletal US images
Source: https://data.mendeley.com/datasets/3jykz7wz8d/1

Data format:
- Polito-Radboud-DeepLearningUS/{BB,GM,TA}/{Healthy,Pathological}/Images/*.png
- Polito-Radboud-DeepLearningUS/{BB,GM,TA}/{Healthy,Pathological}/Masks/*.png

Muscle types:
- BB: Biceps Brachii
- GM: Gastrocnemius Medialis
- TA: Tibialis Anterior

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

DATASET_NAME = "STMUS-NDA"

CATEGORIES = [
    {"supercategory": "muscle", "id": 1, "name": "transverse_musculoskeletal"},
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
    base_dir = None
    for candidate in [
        data_dir / "Polito-Radboud-DeepLearningUS",
        data_dir / DATASET_NAME / "Polito-Radboud-DeepLearningUS",
    ]:
        if candidate.exists():
            base_dir = candidate
            break

    if base_dir is None:
        return samples

    # Muscle types
    muscle_types = ["BB", "GM", "TA"]
    conditions = ["Healthy", "Pathological"]

    for muscle in muscle_types:
        for condition in conditions:
            images_dir = base_dir / muscle / condition / "Images"
            masks_dir = base_dir / muscle / condition / "Masks"

            if not images_dir.exists() or not masks_dir.exists():
                continue

            for img_file in sorted(images_dir.glob("*.png")):
                img_id = img_file.stem
                mask_file = masks_dir / f"{img_id}.png"

                if mask_file.exists():
                    samples.append({
                        "image_path": img_file,
                        "mask_path": mask_file,
                        "image_id": img_id,
                        "muscle_type": muscle,
                        "condition": condition,
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

            # Binarize mask
            mask_binary = np.zeros((h, w), dtype=np.uint8)
            mask_binary[mask != 0] = 1

            if not np.any(mask_binary > 0):
                continue

            image_name = f"{sample['muscle_type']}_{sample['condition']}_{sample['image_id']}"
            annotator.add_2d(
                image_rgb,
                mask_binary,
                video_name=image_name,
                format=args.format,
                metadata={
                    "image_id": sample["image_id"],
                    "muscle_type": sample["muscle_type"],
                    "condition": sample["condition"],
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
