"""BUS-BRA (Breast Ultrasound Brazil) dataset preprocessing.

Dataset: Breast ultrasound dataset with cross-validation splits
Source: https://github.com/wgomezf/BUS-BRA

Data format:
- Images/bus_{id}.png: Ultrasound image
- Masks/mask_{id}.png: Segmentation mask
- 5-fold-cv.csv: Cross-validation split info with pathology labels

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
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ..utils.sam2_annotator import SAM2VideoAnnotator

DATASET_NAME = "BUS-BRA"

CATEGORIES = [
    {"supercategory": "nodule", "id": 1, "name": "breast_nodule_benign"},
    {"supercategory": "nodule", "id": 2, "name": "breast_nodule_malignant"},
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
    parser.add_argument("--test-fold", type=int, default=1, help="Fold to use as test set")
    parser.add_argument("--split", type=str, choices=["train", "test", "all"], default="all")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--zip", action="store_true")
    parser.add_argument("--delete", action="store_true")
    return parser.parse_args()


def rgb2gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def collect_samples(data_dir: Path, csv_path: Path, test_fold: int, split: str) -> List[Dict]:
    """Collect samples with pathology info from CSV."""
    samples = []
    
    df = pd.read_csv(csv_path)
    images_dir = data_dir / "Images"
    masks_dir = data_dir / "Masks"
    
    for _, row in df.iterrows():
        case_id = row["ID"]
        pathology = row["Pathology"]
        kfold = row["kFold"]
        
        # Filter by split
        is_test = kfold == test_fold
        if split == "train" and is_test:
            continue
        if split == "test" and not is_test:
            continue
        
        # Find image file
        img_path = None
        for ext in [".png", ".jpg"]:
            candidate = images_dir / f"{case_id}{ext}"
            if candidate.exists():
                img_path = candidate
                break
        
        if img_path is None:
            continue
        
        # Find mask file
        idx = case_id.split("_")[-1]
        mask_path = masks_dir / f"mask_{idx}.png"
        
        if not mask_path.exists():
            continue
        
        label = 1 if pathology == "benign" else 2
        
        samples.append({
            "image_path": img_path,
            "mask_path": mask_path,
            "case_id": case_id,
            "pathology": pathology,
            "label": label,
            "kfold": kfold,
            "is_test": is_test,
        })
    
    return samples


def main():
    args = parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    
    save_path = Path(args.save_dir)
    data_path = Path(args.path)
    use_temp_dir = data_path.suffix.lower() == ".zip"
    
    video_split = "train" if args.split != "test" else "test"
    
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
            data_dir = Path(temp_dir) / "BUSBRA"
        else:
            data_dir = data_path if (data_path / "Images").exists() else data_path / "BUSBRA"
        
        csv_path = data_dir / "5-fold-cv.csv"
        samples = collect_samples(data_dir, csv_path, args.test_fold, args.split)
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
            
            if not np.any(mask_labeled > 0):
                continue
            
            image_name = f"{sample['case_id']}"
            annotator.add_2d(
                image_rgb,
                mask_labeled,
                video_name=image_name,
                format=args.format,
                metadata={
                    "case_id": sample["case_id"],
                    "pathology": sample["pathology"],
                    "kfold": sample["kfold"],
                    "is_test": sample["is_test"],
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
