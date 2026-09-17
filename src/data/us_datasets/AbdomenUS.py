"""Convert AbdomenUS dataset to SAM2-compatible format.

Dataset: Abdominal Ultrasound Simulation and Segmentation
Source: https://www.kaggle.com/datasets/ignaciorlando/ussimandsegm

The dataset contains real ultrasound (RUS) and artificial ultrasound (AUS) images
with semantic segmentation masks for abdominal organs.

Categories:
    1: liver (RGB: 100, 0, 100)
    2: kidney (RGB: 255, 255, 0)
    3: pancreas (RGB: 0, 0, 255)
    4: vessels (RGB: 255, 0, 0)
    5: adrenals (RGB: 0, 255, 255)
    6: gallbladder (RGB: 0, 255, 0)
    7: bones (RGB: 255, 255, 255)
    8: spleen (RGB: 255, 0, 255)

Usage:
    python projects/US_Datasets/AbdomenUS.py --path /path/to/AbdomenUS.zip --save-dir /path/to/output

Output format: SA-1B (single images with multiple object annotations)
"""

import argparse
import logging
import shutil
import tempfile
from pathlib import Path
import sys
import cv2
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ..utils.sam2_annotator import SAM2VideoAnnotator
from ..utils.path import mkdir_or_exist

DATASET_NAME = "AbdomenUS"

# Category definitions
CATEGORIES = [
    {"supercategory": "organ", "id": 1, "name": "liver"},
    {"supercategory": "organ", "id": 2, "name": "kidney"},
    {"supercategory": "organ", "id": 3, "name": "pancreas"},
    {"supercategory": "vessel", "id": 4, "name": "vessels"},
    {"supercategory": "organ", "id": 5, "name": "adrenals"},
    {"supercategory": "organ", "id": 6, "name": "gallbladder"},
    {"supercategory": "bone", "id": 7, "name": "bones"},
    {"supercategory": "organ", "id": 8, "name": "spleen"},
]

# RGB color to category ID mapping
# Note: OpenCV reads as BGR, so we need to convert
COLOR_TO_CATEGORY = {
    (100, 0, 100): 1,    # liver - purple
    (255, 255, 0): 2,    # kidney - yellow
    (0, 0, 255): 3,      # pancreas - blue
    (255, 0, 0): 4,      # vessels - red
    (0, 255, 255): 5,    # adrenals - cyan
    (0, 255, 0): 6,      # gallbladder - green
    (255, 255, 255): 7,  # bones - white
    (255, 0, 255): 8,    # spleen - magenta
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=f"Convert {DATASET_NAME} to SAM2 format"
    )
    parser.add_argument(
        "--path",
        type=str,
        help="Path to dataset (zip file or directory)",
        default=f"/mnt/data/Dataset/UltraSound/Raw/{DATASET_NAME}.zip",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        help="Output directory for SAM2 format dataset",
        default=f"/mnt/data/Dataset/SaUS/{DATASET_NAME}",
    )
    parser.add_argument(
        "--save-viz",
        action="store_true",
        help="Save visualization images with overlaid masks",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["sa-1b", "sa-v"],
        default="sa-1b",
        help="Output format: sa-1b (image) or sa-v (video)",
    )
    parser.add_argument(
        "--include-aus",
        action="store_true",
        help="Include AUS (artificial ultrasound) images (lower quality)",
    )
    return parser.parse_args()


def rgb_mask_to_label_map(rgb_mask: np.ndarray) -> np.ndarray:
    """Convert RGB mask to label map using color mapping.
    
    Args:
        rgb_mask: RGB mask array (H, W, 3) in RGB format.
        
    Returns:
        Label map (H, W) with category IDs.
    """
    label_map = np.zeros(rgb_mask.shape[:2], dtype=np.uint8)
    
    for rgb, category_id in COLOR_TO_CATEGORY.items():
        # Create boolean mask where all RGB channels match
        matches = np.all(rgb_mask == rgb, axis=2)
        label_map[matches] = category_id
    
    return label_map


def process_dataset(
    data_dir: Path,
    annotator: SAM2VideoAnnotator,
    output_format: str,
    include_aus: bool = False,
) -> int:
    """Process all images in the dataset.
    
    Args:
        data_dir: Path to extracted dataset directory.
        annotator: SAM2VideoAnnotator instance.
        output_format: "sa-1b" or "sa-v".
        include_aus: Whether to include artificial ultrasound images.
        
    Returns:
        Number of images processed.
    """
    count = 0
    
    # Process RUS (Real Ultrasound) - we use test set since train has no GT
    rus_dirs = [
        ("RUS", "test"),
    ]
    
    # Optionally include AUS (Artificial Ultrasound)
    if include_aus:
        rus_dirs.extend([
            ("AUS", "train"),
            ("AUS", "test"),
        ])
    
    for dataset_type, split in rus_dirs:
        mask_dir = data_dir / "abdominal_US" / dataset_type / "annotations" / split
        image_dir = data_dir / "abdominal_US" / dataset_type / "images" / split
        
        if not mask_dir.exists():
            logging.warning(f"Mask directory not found: {mask_dir}")
            continue
        
        if not image_dir.exists():
            logging.warning(f"Image directory not found: {image_dir}")
            continue
        
        logging.info(f"Processing {dataset_type}/{split}...")
        
        for idx, mask_path in enumerate(sorted(mask_dir.glob("*.png"))):
            # Find corresponding image (could be .jpg or .png)
            image_path = image_dir / f"{mask_path.stem}.jpg"
            if not image_path.exists():
                image_path = image_dir / f"{mask_path.stem}.png"
            
            if not image_path.exists():
                logging.warning(f"Image not found for mask: {mask_path}")
                continue
            
            # Load image (cv2 loads as BGR)
            img_bgr = cv2.imread(str(image_path))
            if img_bgr is None:
                logging.warning(f"Failed to load image: {image_path}")
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            # Load mask (cv2 loads as BGR, we need RGB for color matching)
            mask_bgr = cv2.imread(str(mask_path))
            if mask_bgr is None:
                logging.warning(f"Failed to load mask: {mask_path}")
                continue
            mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
            
            # Convert RGB mask to label map
            label_map = rgb_mask_to_label_map(mask_rgb)
            
            # Skip if no annotations
            if not np.any(label_map > 0):
                logging.debug(f"No annotations in mask: {mask_path}")
                continue
            
            # Generate unique name
            image_name = f"{dataset_type}_{split}_{mask_path.stem}"
            
            # Add to annotator
            if output_format == "sa-1b":
                annotator.add_2d_sa1b(
                    img_rgb,
                    label_map,
                    image_name=image_name,
                    metadata={
                        "source_dataset": DATASET_NAME,
                        "dataset_type": dataset_type,
                        "split": split,
                        "original_filename": mask_path.stem,
                    },
                )
            else:  # sa-v
                annotator.add_2d(
                    img_rgb,
                    label_map,
                    video_name=image_name,
                    metadata={
                        "source_dataset": DATASET_NAME,
                        "dataset_type": dataset_type,
                        "split": split,
                        "original_filename": mask_path.stem,
                    },
                )
            
            count += 1
            
            if count % 5 == 0:
                logging.info(f"Processed {count} images...")
    
    return count


def main() -> None:
    args = parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    save_path = Path(args.save_dir)
    mkdir_or_exist(save_path)
    
    logging.info(f"Converting {DATASET_NAME} to SAM2 format ({args.format})")
    logging.info(f"Input: {args.path}")
    logging.info(f"Output: {save_path}")
    
    # Initialize annotator
    annotator = SAM2VideoAnnotator(
        out_dir=save_path,
        dataset_name=DATASET_NAME,
        categories=CATEGORIES,
        save_visualization=args.save_viz,
        video_environment="ultrasound",
        video_split="train",  # Will be overwritten per-image in metadata
    )
    
    # Process dataset
    if args.path.endswith(".zip"):
        # Extract to temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            logging.info(f"Extracting dataset to {temp_dir}...")
            shutil.unpack_archive(args.path, temp_dir)
            logging.info("Extraction complete")
            
            count = process_dataset(
                Path(temp_dir) / "abdominal_US",
                annotator,
                args.format,
                args.include_aus,
            )
    else:
        # Use directory directly
        count = process_dataset(
            Path(args.path),
            annotator,
            args.format,
            args.include_aus,
        )
    
    # Finalize
    stats = annotator.finalize()
    
    logging.info("=" * 60)
    logging.info(f"Conversion complete!")
    logging.info(f"  Images processed: {count}")
    logging.info(f"  Total annotations: {stats['total_annotations']}")
    logging.info(f"  Output directory: {save_path}")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
