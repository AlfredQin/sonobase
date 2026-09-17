"""MMOTU_3d (Multi-Modality Ovarian Tumor Ultrasound 3D) dataset preprocessing.

Dataset: Ovarian tumor 3D ultrasound segmentation dataset
Source: https://github.com/cv516Buaa/MMOTU_DS2Net

Data format:
- OTU_3d/images/{id}.JPG: Ultrasound images
- OTU_3d/annotations/{id}_binary.PNG or {id}_binary_binary.PNG: Binary masks
- OTU_3d/train.txt: Training set image list
- OTU_3d/val.txt: Validation set image list
- OTU_3d/val_cls.txt: Validation set with class labels (optional)

Output: SAM2-compatible format (SA-1B style)
"""

import argparse
import logging
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

import cv2
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ..utils.sam2_annotator import SAM2VideoAnnotator

DATASET_NAME = "MMOTU-3d"

CATEGORIES = [
    {"supercategory": "nodule", "id": 1, "name": "ovarian_lesion"},
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
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "all"],
        default="all",
        help="Which split to process",
    )
    parser.add_argument("--save-viz", action="store_true")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--zip", action="store_true")
    parser.add_argument("--delete", action="store_true")
    return parser.parse_args()


def rgb2gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def load_split_file(txt_path: Path) -> Set[str]:
    """Load split file containing image names."""
    images = set()
    if not txt_path.exists():
        return images
    
    with open(txt_path, "r") as f:
        for line in f:
            img_name = line.strip()
            if img_name:
                images.add(img_name)
    
    return images


def find_mask_file(annotations_dir: Path, img_id: str) -> Optional[Path]:
    """Find the corresponding mask file for an image ID."""
    # Try different naming patterns
    patterns = [
        f"{img_id}_binary.PNG",
        f"{img_id}_binary_binary.PNG",
        f"{img_id}_binary.png",
        f"{img_id}_binary_binary.png",
    ]
    
    for pattern in patterns:
        mask_path = annotations_dir / pattern
        if mask_path.exists():
            return mask_path
    
    return None


def collect_samples(data_dir: Path, split: str = "all") -> List[Dict]:
    """Collect image-mask pairs."""
    samples = []

    # Handle nested folder structure
    if (data_dir / "OTU_3d").exists():
        data_dir = data_dir / "OTU_3d"

    images_dir = data_dir / "images"
    annotations_dir = data_dir / "annotations"

    if not images_dir.exists():
        return samples

    # Load split info from txt files
    train_images = load_split_file(data_dir / "train.txt")
    val_images = load_split_file(data_dir / "val.txt")

    # Build split lookup
    split_info = {}
    for img_name in train_images:
        split_info[img_name] = "train"
    for img_name in val_images:
        split_info[img_name] = "val"

    # Collect image-mask pairs
    for img_file in sorted(images_dir.glob("*.JPG")) + sorted(images_dir.glob("*.jpg")):
        img_id = img_file.stem
        img_key = img_file.name  # e.g., "123.JPG"

        # Get split info
        sample_split = split_info.get(img_key, "unknown")

        # Filter by split
        if split != "all" and sample_split != split:
            continue

        # Find mask file
        mask_file = find_mask_file(annotations_dir, img_id)
        if mask_file is None:
            continue

        samples.append({
            "image_path": img_file,
            "mask_path": mask_file,
            "image_id": img_id,
            "split": sample_split,
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
            h, w = image.shape[:2]

            mask = rgb2gray(cv2.imread(str(sample["mask_path"]))).astype(np.uint8)
            
            # Binarize mask (single lesion class)
            mask_binary = np.zeros((h, w), dtype=np.uint8)
            mask_binary[mask > 0] = 1

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
