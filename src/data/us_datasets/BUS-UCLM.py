"""BUS-UCLM dataset preprocessing.

Dataset: Breast ultrasound dataset from UCLM
Source: https://data.mendeley.com/datasets/7fvgj4jsp7/1

Data format:
- images/{id}.png: Ultrasound image
- masks/{id}.png: RGB color-coded mask
  - Green [0,255,0]: Benign lesion (label 1)
  - Red [255,0,0]: Malignant lesion (label 2)

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

DATASET_NAME = "BUS-UCLM"

CATEGORIES = [
    {"supercategory": "nodule", "id": 1, "name": "benign_breast_lesion"},
    {"supercategory": "nodule", "id": 2, "name": "malignant_breast_lesion"},
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


def collect_samples(data_dir: Path) -> List[Dict]:
    """Collect image-mask pairs."""
    samples = []
    
    # Handle nested folder structure
    # Pattern: "BUS-UCLM Breast ultrasound lesion segmentation dataset/BUS-UCLM/..."
    for subdir in data_dir.iterdir():
        if subdir.is_dir() and "BUS-UCLM" in subdir.name:
            if (subdir / "BUS-UCLM").exists():
                data_dir = subdir / "BUS-UCLM"
                break
            elif (subdir / "images").exists():
                data_dir = subdir
                break
    
    if (data_dir / "BUS-UCLM").exists():
        data_dir = data_dir / "BUS-UCLM"
    
    images_dir = data_dir / "images"
    masks_dir = data_dir / "masks"
    
    if not images_dir.exists():
        return samples
    
    for img_file in sorted(images_dir.glob("*.png")):
        img_id = img_file.stem
        mask_file = masks_dir / f"{img_id}.png"
        
        if mask_file.exists():
            samples.append({
                "image_path": img_file,
                "mask_path": mask_file,
                "image_id": img_id,
            })
    
    return samples


def parse_rgb_mask(mask_bgr: np.ndarray) -> np.ndarray:
    """Parse RGB color-coded mask to label map.
    
    Green [0,255,0] -> 1 (benign)
    Red [255,0,0] -> 2 (malignant)
    """
    # Convert BGR to RGB
    mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
    
    h, w = mask_rgb.shape[:2]
    label_map = np.zeros((h, w), dtype=np.uint8)
    
    # Green = benign (label 1)
    green_mask = (mask_rgb[:, :, 0] < 50) & (mask_rgb[:, :, 1] > 200) & (mask_rgb[:, :, 2] < 50)
    label_map[green_mask] = 1
    
    # Red = malignant (label 2)
    red_mask = (mask_rgb[:, :, 0] > 200) & (mask_rgb[:, :, 1] < 50) & (mask_rgb[:, :, 2] < 50)
    label_map[red_mask] = 2
    
    return label_map


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
            
            mask_bgr = cv2.imread(str(sample["mask_path"]))
            label_map = parse_rgb_mask(mask_bgr)
            
            if not np.any(label_map > 0):
                continue
            
            image_name = f"{sample['image_id']}"
            annotator.add_2d(
                image_rgb,
                label_map,
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
