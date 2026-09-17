"""BUSI_WHU dataset preprocessing.

Dataset: Breast ultrasound dataset from WHU
Source: BUSI_WHU Breast Cancer Ultrasound Image Dataset

Data format:
- train/images/*.png, train/masks/*.png
- valid/images/*.png, valid/masks/*.png  
- test/images/*.png, test/masks/*.png

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

DATASET_NAME = "BUSI-WHU"

CATEGORIES = [
    {"supercategory": "nodule", "id": 1, "name": "breast_nodule"},
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
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "valid", "test", "all"],
        default="all",
        help="Which split to process",
    )
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--zip", action="store_true")
    parser.add_argument("--delete", action="store_true")
    return parser.parse_args()


def rgb2gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def collect_samples(data_dir: Path, split: str) -> List[Dict]:
    """Collect image-mask pairs from specified splits."""
    samples = []
    
    # Handle nested folder structure
    for candidate in [
        data_dir / "BUSI_WHU Breast Cancer Ultrasound Image  Dataset",
        data_dir / "BUSI_WHU",
        data_dir,
    ]:
        if (candidate / "train").exists():
            data_dir = candidate
            break
    
    splits = ["train", "valid", "test"] if split == "all" else [split]
    
    for split_name in splits:
        split_dir = data_dir / split_name
        if not split_dir.exists():
            continue
        
        # Try different folder structures
        possible_img_dirs = [
            split_dir / "img" / "ori",  # BUSI_WHU structure
            split_dir / "images",
            split_dir / "img",
            split_dir,
        ]
        possible_gt_dirs = [
            split_dir / "gt" / "ori",  # BUSI_WHU structure
            split_dir / "masks",
            split_dir / "gt",
            split_dir,
        ]
        
        images_dir = None
        masks_dir = None
        
        for img_dir in possible_img_dirs:
            if img_dir.exists() and any(img_dir.glob("*.bmp")) or any(img_dir.glob("*.png")):
                images_dir = img_dir
                break
        
        for gt_dir in possible_gt_dirs:
            if gt_dir.exists():
                masks_dir = gt_dir
                break
        
        if images_dir is None:
            continue
        
        # Collect image files
        img_files = list(images_dir.glob("*.bmp")) + list(images_dir.glob("*.png"))
        
        for img_file in sorted(img_files):
            img_id = img_file.stem
            
            # Skip if it's a mask file itself
            if "_mask" in img_id or img_id.startswith("mask_"):
                continue
            
            # Try to find corresponding mask
            mask_file = None
            if masks_dir:
                for ext in [".bmp", ".png"]:
                    for pattern in [img_id, f"{img_id}_anno", f"{img_id}_mask", f"mask_{img_id}"]:
                        candidate = masks_dir / f"{pattern}{ext}"
                        if candidate.exists():
                            mask_file = candidate
                            break
                    if mask_file:
                        break
            
            if mask_file and mask_file.exists():
                samples.append({
                    "image_path": img_file,
                    "mask_path": mask_file,
                    "image_id": img_id,
                    "split": split_name,
                })
    
    return samples


def main():
    args = parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    
    save_path = Path(args.save_dir)
    data_path = Path(args.path)
    use_temp_dir = data_path.suffix.lower() == ".zip"
    
    video_split = args.split if args.split != "all" else "train"
    
    annotator = SAM2VideoAnnotator(
        out_dir=save_path,
        dataset_name=DATASET_NAME,
        categories=CATEGORIES,
        save_visualization=args.save_viz,
        video_environment="ultrasound",
        video_split=video_split,
    )
    
    logger.info(f"Output format: {args.format}, split: {args.split}")
    
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
            
            # Check for nested RAR file that needs extraction
            for rar_file in data_dir.rglob("*.rar"):
                logger.info(f"Found RAR archive: {rar_file.name}")
                import subprocess
                result = subprocess.run(
                    ["unrar", "x", "-o+", str(rar_file), str(data_dir)],
                    capture_output=True, text=True
                )
                if result.returncode != 0:
                    logger.warning(f"Failed to extract RAR: {result.stderr}")
        else:
            data_dir = data_path
        
        samples = collect_samples(data_dir, args.split)
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
            mask_binary[mask > 127] = 1
            
            if not np.any(mask_binary > 0):
                continue
            
            image_name = f"{sample['split']}_{sample['image_id']}"
            annotator.add_2d(
                image_rgb,
                mask_binary,
                video_name=image_name,
                format=args.format,
                metadata={
                    "image_id": sample["image_id"],
                    "split": sample["split"],
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
