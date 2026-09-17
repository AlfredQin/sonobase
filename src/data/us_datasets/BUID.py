"""BUID (Breast Ultrasound Images Database) dataset preprocessing.

Dataset: Breast ultrasound classification and segmentation dataset
Source: https://qamebi.com/breast-ultrasound-images-database/

Data format:
- {category}/{id} {category} Image.bmp: Ultrasound image
- {category}/{id} {category} Mask.tif: Segmentation mask

Categories: Benign, Malignant

Output: SAM2-compatible format (SA-1B style)
"""

import argparse
import logging
import os
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

DATASET_NAME = "BUID"

CATEGORIES = [
    {"supercategory": "nodule", "id": 1, "name": "breast_nodule"},
]


def parse_args():
    parser = argparse.ArgumentParser(description=f"Convert {DATASET_NAME} to SAM2 format")
    parser.add_argument(
        "--path",
        type=str,
        default="/mnt/data/Dataset/UltraSound/Raw/BUID.zip",
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
    
    for category in ["Benign", "Malignant"]:
        category_dir = data_dir / category
        if not category_dir.exists():
            continue
        
        for filename in os.listdir(category_dir):
            if "Image.bmp" in filename:
                img_path = category_dir / filename
                mask_filename = filename.replace("Image.bmp", "Mask.tif")
                mask_path = category_dir / mask_filename
                
                if mask_path.exists():
                    sample_id = filename.split()[0]
                    samples.append({
                        "image_path": img_path,
                        "mask_path": mask_path,
                        "category": category,
                        "sample_id": sample_id,
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
            
            # Handle nested BUID folder
            if (data_dir / "BUID").exists():
                data_dir = data_dir / "BUID"
            
            # Extract nested zips if present
            for zf in ["Benign.zip", "Malignant.zip"]:
                zf_path = data_dir / zf
                if zf_path.exists():
                    logger.info(f"Extracting nested archive: {zf}")
                    shutil.unpack_archive(str(zf_path), str(data_dir))
        else:
            data_dir = data_path
            # Extract nested zips if present
            for zf in ["Benign.zip", "Malignant.zip"]:
                zf_path = data_dir / zf
                if zf_path.exists() and not (data_dir / zf.replace(".zip", "")).exists():
                    shutil.unpack_archive(str(zf_path), str(data_dir))
        
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
            mask_binary[mask >= 150] = 1
            
            if not np.any(mask_binary > 0):
                continue
            
            image_name = f"{sample['category']}_{sample['sample_id']}"
            annotator.add_2d(
                image_rgb,
                mask_binary,
                video_name=image_name,
                format=args.format,
                metadata={
                    "sample_id": sample["sample_id"],
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
