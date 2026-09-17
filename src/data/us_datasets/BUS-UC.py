"""BUS_UC (Breast Ultrasound UC) dataset preprocessing.

Dataset: Breast ultrasound dataset from University of Chile
Source: https://data.mendeley.com/datasets/3ksd7w7jkx/1

Data format:
- {category}/images/{id}.png: Ultrasound image
- {category}/masks/{id}.png: Segmentation mask

Categories: Benign, Malignant

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

DATASET_NAME = "BUS-UC"

CATEGORIES = [
    {"supercategory": "tumor", "id": 1, "name": "breast_benign"},
    {"supercategory": "tumor", "id": 2, "name": "breast_malignant"},
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
    """Collect image-mask pairs from Benign and Malignant folders."""
    samples = []
    
    # Handle nested folder structure (BUS_UC/BUS_UC/All/...)
    for candidate in [
        data_dir / "BUS_UC" / "BUS_UC",
        data_dir / "BUS_UC",
        data_dir,
    ]:
        if (candidate / "All").exists() or (candidate / "Benign").exists():
            data_dir = candidate
            break
    
    # Check for "All" folder structure (combined dataset)
    if (data_dir / "All").exists():
        all_dir = data_dir / "All"
        images_dir = all_dir / "images"
        masks_dir = all_dir / "masks"
        
        if images_dir.exists():
            for img_file in sorted(images_dir.glob("*.png")):
                img_id = img_file.stem
                mask_file = masks_dir / f"{img_id}.png"
                
                if mask_file.exists():
                    samples.append({
                        "image_path": img_file,
                        "mask_path": mask_file,
                        "image_id": img_id,
                        "category": "unknown",
                        "label": 1,  # Single label for combined
                    })
            return samples
    
    for category, label in [("Benign", 1), ("Malignant", 2)]:
        category_dir = data_dir / category
        if not category_dir.exists():
            continue
        
        images_dir = category_dir / "images"
        masks_dir = category_dir / "masks"
        
        if not images_dir.exists():
            continue
        
        for img_file in sorted(images_dir.glob("*.png")):
            img_id = img_file.stem
            mask_file = masks_dir / f"{img_id}.png"
            
            if mask_file.exists():
                samples.append({
                    "image_path": img_file,
                    "mask_path": mask_file,
                    "image_id": img_id,
                    "category": category,
                    "label": label,
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
            mask_labeled = np.zeros_like(mask, dtype=np.uint8)
            mask_labeled[mask > 127] = sample["label"]
            
            # Resize mask if dimensions don't match
            if mask_labeled.shape[:2] != image_rgb.shape[:2]:
                mask_labeled = cv2.resize(
                    mask_labeled,
                    (image_rgb.shape[1], image_rgb.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            
            if not np.any(mask_labeled > 0):
                continue
            
            image_name = f"{sample['category']}_{sample['image_id']}"
            annotator.add_2d(
                image_rgb,
                mask_labeled,
                video_name=image_name,
                format=args.format,
                metadata={
                    "image_id": sample["image_id"],
                    "category": sample["category"],
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
