"""BrEaST-Lesions dataset preprocessing.

Dataset: Breast ultrasound lesion segmentation dataset
Source: https://www.nature.com/articles/s41597-024-02984-z

Data format:
- case{id}.png: Ultrasound image
- case{id}_tumor.png: Tumor mask (optional)
- case{id}_cyst.png: Cyst mask (optional)

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

DATASET_NAME = "BrEaST"

# Category definitions
CATEGORIES = [
    {"supercategory": "nodule", "id": 1, "name": "breast_tumor"},
    {"supercategory": "nodule", "id": 2, "name": "breast_cyst"},
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=f"Convert {DATASET_NAME} to SAM2 format"
    )
    parser.add_argument(
        "--path",
        type=str,
        default="/mnt/data/Dataset/UltraSound/Raw/BrEaST.zip",
        help="Path to dataset (zip or folder)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=f"/mnt/data/Dataset/SaUS/{DATASET_NAME}",
        help="Output directory",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["sa-v", "sa-1b"],
        default="sa-1b",
        help="Output format",
    )
    parser.add_argument("--save-viz", action="store_true", help="Save visualizations")
    parser.add_argument("--max-images", type=int, default=None, help="Max images to process")
    parser.add_argument("--zip", action="store_true", help="Zip output")
    parser.add_argument("--delete", action="store_true", help="Delete after zipping")
    return parser.parse_args()


def rgb2gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def collect_samples(data_dir: Path) -> List[Dict]:
    """Collect image-mask pairs."""
    samples = []
    
    # Find the data folder (may have nested folder after extraction)
    if (data_dir / "BrEaST-Lesions_USG-images_and_masks").exists():
        data_dir = data_dir / "BrEaST-Lesions_USG-images_and_masks"
    
    # Find all image files (without _tumor or _cyst suffix)
    for png_file in sorted(data_dir.glob("*.png")):
        filename = png_file.stem
        if "_tumor" in filename or "_cyst" in filename:
            continue
        
        sample = {
            "image_path": png_file,
            "image_id": filename,
            "tumor_mask": data_dir / f"{filename}_tumor.png",
            "cyst_mask": data_dir / f"{filename}_cyst.png",
        }
        samples.append(sample)
    
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
            
            # Create combined mask
            mask = np.zeros((h, w), dtype=np.uint8)
            instance_to_category = {}
            
            # Load tumor mask (instance 1 -> category 1)
            if sample["tumor_mask"].exists():
                tumor = rgb2gray(cv2.imread(str(sample["tumor_mask"])))
                tumor_binary = (tumor > 0).astype(np.uint8)
                mask[tumor_binary > 0] = 1
                instance_to_category[1] = 1
            
            # Load cyst mask (instance 2 -> category 2)
            if sample["cyst_mask"].exists():
                cyst = rgb2gray(cv2.imread(str(sample["cyst_mask"])))
                cyst_binary = (cyst > 0).astype(np.uint8)
                mask[cyst_binary > 0] = 2
                instance_to_category[2] = 2
            
            if not np.any(mask > 0):
                continue
            
            image_name = f"{sample['image_id']}"
            annotator.add_2d(
                image_rgb,
                mask,
                video_name=image_name,
                format=args.format,
                instance_to_category=instance_to_category,
                metadata={"image_id": sample["image_id"], "dataset": DATASET_NAME},
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
