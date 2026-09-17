"""FALLMUD (Fetal Aponeurosis and Fascicle) dataset preprocessing.

Dataset: Muscle ultrasound segmentation
Source: https://kalisteo.cea.fr/index.php/fallmud/

Data format:
- images/{id}.png: Ultrasound images
- aponeurosis_masks/{id}.png: Aponeurosis masks (used)
- fascicle_masks/{id}.png: Fascicle masks (not used - contains thin lines)

Note: Aponeurosis masks typically contain two separate lines (upper and lower).
      These are split into separate instances using connected components.

Centers: NeilCronin, RyanCunningham

Output: SAM2-compatible format (SA-1B style)
"""

import argparse
import logging
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ..utils.sam2_annotator import SAM2VideoAnnotator

DATASET_NAME = "FALLMUD"

CATEGORIES = [
    {"supercategory": "muscle", "id": 1, "name": "aponeurosis"},
]

CENTERS = ["NeilCronin", "RyanCunningham"]


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


def imrotate(img: np.ndarray, angle: float, auto_bound: bool = False) -> np.ndarray:
    """Rotate image by angle degrees."""
    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    if auto_bound:
        cos = np.abs(matrix[0, 0])
        sin = np.abs(matrix[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        matrix[0, 2] += (new_w - w) / 2
        matrix[1, 2] += (new_h - h) / 2
        return cv2.warpAffine(img, matrix, (new_w, new_h))
    
    return cv2.warpAffine(img, matrix, (w, h))


def natural_sort_key(s: str) -> List:
    """Generate key for natural sorting of strings with numbers."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'([0-9]+)', s)]


def find_mask_path(masks_folder: Path, image_filename: str) -> Optional[Path]:
    """Find corresponding mask file for an image."""
    base_name = Path(image_filename).stem
    
    for f in masks_folder.glob(f"{base_name}.*"):
        if f.stem == base_name:
            return f
    
    return None


def collect_samples(data_dir: Path) -> List[Dict]:
    """Collect image-mask pairs from all centers (aponeurosis only)."""
    samples = []

    # Handle nested folder structure
    if (data_dir / "FALLMUD").exists():
        data_dir = data_dir / "FALLMUD"

    for center in CENTERS:
        center_dir = data_dir / center
        if not center_dir.exists():
            continue

        images_folder = center_dir / "images"
        aponeurosis_folder = center_dir / "aponeurosis_masks"

        if not images_folder.exists():
            continue

        for img_file in sorted(images_folder.iterdir(), key=lambda x: natural_sort_key(x.name)):
            if img_file.suffix.lower() not in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
                continue

            apo_mask = find_mask_path(aponeurosis_folder, img_file.name)

            if apo_mask:
                samples.append({
                    "image_path": img_file,
                    "aponeurosis_path": apo_mask,
                    "image_id": img_file.stem,
                    "center": center,
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

            # Load aponeurosis mask only
            apo_mask = rgb2gray(cv2.imread(str(sample["aponeurosis_path"])))

            # Handle rotated masks from RyanCunningham
            if apo_mask.shape[:2] != (h, w):
                apo_mask = imrotate(apo_mask, 90, auto_bound=True)

            # Resize if still mismatched
            if apo_mask.shape[:2] != (h, w):
                apo_mask = cv2.resize(apo_mask, (w, h), interpolation=cv2.INTER_NEAREST)

            # Binarize mask (threshold at 100 due to compression artifacts)
            apo_binary = (apo_mask >= 100).astype(np.uint8)

            if not np.any(apo_binary > 0):
                continue

            # Use connected components to split into separate objects
            # Each connected component becomes a separate instance with category_id=1
            num_labels, labels = cv2.connectedComponents(apo_binary)
            
            # Create instance mask where each component has a unique instance ID
            # but all share category_id=1 (aponeurosis)
            # We use the label directly as the mask value (label 0 is background)
            instance_mask = labels.astype(np.uint8)

            if num_labels <= 1:  # Only background
                continue

            image_name = f"{sample['center']}_{sample['image_id']}"
            annotator.add_2d(
                image_rgb,
                instance_mask,
                video_name=image_name,
                format=args.format,
                metadata={
                    "image_id": sample["image_id"],
                    "center": sample["center"],
                    "dataset": DATASET_NAME,
                    "num_aponeurosis": num_labels - 1,  # Exclude background
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
