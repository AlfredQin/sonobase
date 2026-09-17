"""BUSI (Breast Ultrasound Images) dataset preprocessing.

Dataset: Breast ultrasound dataset with benign/malignant/normal categories
Source: https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/BUSI.md

Data format:
- {category}/{name}.png: Ultrasound image
- {category}/{name}_mask.png, {name}_mask_1.png, ...: Segmentation masks

Categories: benign, malignant (normal has no masks)

Output: SAM2-compatible format (SA-1B style)
"""

import argparse
import logging
import os
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

DATASET_NAME = "BUSI"

CATEGORIES = [
    {"supercategory": "nodule", "id": 1, "name": "breast_nodule"},
]


def parse_args():
    parser = argparse.ArgumentParser(description=f"Convert {DATASET_NAME} to SAM2 format")
    parser.add_argument(
        "--path",
        type=str,
        default="/mnt/data/Dataset/UltraSound/Raw/BUSI.zip",
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
    """Collect image-mask pairs, handling multiple masks per image."""
    samples = []
    
    # Handle nested folder structure
    if (data_dir / "Dataset_BUSI_with_GT").exists():
        data_dir = data_dir / "Dataset_BUSI_with_GT"
    
    # Build mapping of image -> list of masks
    img_to_masks = defaultdict(list)
    
    for category in ["benign", "malignant"]:
        category_dir = data_dir / category
        if not category_dir.exists():
            continue
        
        for filename in os.listdir(category_dir):
            if "_mask" in filename:
                # Construct image filename from mask filename
                # e.g., "benign (1)_mask.png" -> "benign (1).png"
                # e.g., "benign (1)_mask_1.png" -> "benign (1).png"
                base_name = filename.split("_mask")[0]
                img_filename = f"{base_name}.png"
                
                img_path = category_dir / img_filename
                mask_path = category_dir / filename
                
                if img_path.exists():
                    img_to_masks[(str(img_path), category)].append(str(mask_path))
    
    for (img_path, category), mask_paths in sorted(img_to_masks.items()):
        img_id = Path(img_path).stem
        samples.append({
            "image_path": Path(img_path),
            "mask_paths": [Path(p) for p in sorted(mask_paths)],
            "image_id": img_id,
            "category": category,
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
            
            # Combine all masks (multiple nodules get same label since single category)
            combined_mask = np.zeros((h, w), dtype=np.uint8)
            
            for mask_path in sample["mask_paths"]:
                mask = rgb2gray(cv2.imread(str(mask_path)))
                if mask is not None:
                    combined_mask[mask >= 150] = 1
            
            if not np.any(combined_mask > 0):
                continue
            
            # Clean up image_id for valid filename
            clean_id = sample["image_id"].replace(" ", "_").replace("(", "").replace(")", "")
            image_name = f"{sample['category']}_{clean_id}"
            
            annotator.add_2d(
                image_rgb,
                combined_mask,
                video_name=image_name,
                format=args.format,
                metadata={
                    "image_id": sample["image_id"],
                    "category": sample["category"],
                    "num_masks": len(sample["mask_paths"]),
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
