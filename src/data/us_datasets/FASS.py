"""FASS (Fetal Abdominal Structures Segmentation) dataset preprocessing.

Dataset: Fetal abdominal ultrasound segmentation
Source: https://data.mendeley.com/datasets/4gcpm9dsc3/1

Data format:
- ARRAY_FORMAT/{patient}_{img}.npy: Contains both image and mask structures
  - 'image': RGB image array
  - 'structures': dict with 'artery', 'liver', 'stomach', 'vein' binary masks

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

DATASET_NAME = "FASS"

CATEGORIES = [
    {"supercategory": "organ", "id": 1, "name": "artery"},
    {"supercategory": "organ", "id": 2, "name": "liver"},
    {"supercategory": "organ", "id": 3, "name": "stomach"},
    {"supercategory": "organ", "id": 4, "name": "vein"},
]

NAME_TO_ID = {"artery": 1, "liver": 2, "stomach": 3, "vein": 4}


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


def collect_samples(data_dir: Path) -> List[Dict]:
    """Collect NPY files containing image and mask data."""
    samples = []

    # Find ARRAY_FORMAT folder
    array_dir = None
    for candidate in [
        data_dir / "ARRAY_FORMAT",
        data_dir / "Fetal Abdominal Structures Segmentation Dataset Using Ultrasonic Images" / "ARRAY_FORMAT",
    ]:
        if candidate.exists():
            array_dir = candidate
            break

    if array_dir is None:
        # Search recursively
        for path in data_dir.rglob("ARRAY_FORMAT"):
            if path.is_dir():
                array_dir = path
                break

    if array_dir is None:
        return samples

    # Collect all NPY files
    for npy_file in sorted(array_dir.glob("*.npy")):
        samples.append({
            "npy_path": npy_file,
            "image_id": npy_file.stem,
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
            
            # Handle nested zip
            nested_zip = Path(temp_dir) / "Fetal Abdominal Structures Segmentation Dataset Using Ultrasonic Images" / "Fetal Abdominal Structures Segmentation Dataset Using Ultrasonic Images.zip"
            if nested_zip.exists():
                logger.info("Extracting nested archive")
                shutil.unpack_archive(str(nested_zip), temp_dir)
            
            data_dir = Path(temp_dir)
        else:
            data_dir = data_path

        samples = collect_samples(data_dir)
        logger.info(f"Found {len(samples)} samples")

        if args.max_images:
            samples = samples[:args.max_images]

        processed = 0
        for idx, sample in enumerate(samples):
            try:
                # Load NPY file containing image and structures
                data = np.load(str(sample["npy_path"]), allow_pickle=True).item()
                
                image = data.get("image")
                structures = data.get("structures", {})
                
                if image is None:
                    continue
                
                # Ensure image is uint8 RGB
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8) if image.max() <= 1 else image.astype(np.uint8)
                
                # Convert to RGB if grayscale
                if image.ndim == 2:
                    image = np.stack([image] * 3, axis=-1)
                
                h, w = image.shape[:2]
                
                # Create combined mask from structures
                combined_mask = np.zeros((h, w), dtype=np.uint8)
                for key, value in structures.items():
                    if key in NAME_TO_ID:
                        label = NAME_TO_ID[key]
                        combined_mask[value > 0] = label

                if not np.any(combined_mask > 0):
                    continue

                image_name = f"{sample['image_id']}"
                annotator.add_2d(
                    image,
                    combined_mask,
                    video_name=image_name,
                    format=args.format,
                    metadata={
                        "image_id": sample["image_id"],
                        "dataset": DATASET_NAME,
                    },
                )
                processed += 1

            except Exception as e:
                logger.warning(f"Error processing {sample['image_id']}: {e}")
                continue

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
