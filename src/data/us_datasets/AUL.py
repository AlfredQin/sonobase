"""AUL (Annotated Ultrasound Liver) dataset preprocessing.

Dataset: Liver ultrasound classification dataset with segmentation annotations
Classes: Benign, Malignant, Normal

Data format:
- {category}/image/{id}.jpg: Ultrasound images (1024x768 RGB)
- {category}/segmentation/liver/{id}.json: Liver region polygon
- {category}/segmentation/mass/{id}.json: Tumor/mass polygon (not for Normal)
- {category}/segmentation/outline/{id}.json: Ultrasound FOV outline polygon

JSON format: [[x1, y1], [x2, y2], ...] polygon coordinates

Output: SAM2-compatible format (SA-V or SA-1B style)
"""

import argparse
import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ..utils.sam2_annotator import SAM2VideoAnnotator

DATASET_NAME = "AUL"

# Category definitions
# We treat segmentation types (liver, mass) as categories
# Note: category_id 1=liver, 2=mass (benign), 3=mass (malignant)
CATEGORIES = [
    {"supercategory": "organ", "id": 1, "name": "liver"},
    {"supercategory": "lesion", "id": 2, "name": "mass_benign"},
    {"supercategory": "lesion", "id": 3, "name": "mass_malignant"},
]

# Classification categories (Benign, Malignant, Normal)
CLASS_CATEGORIES = ["Benign", "Malignant", "Normal"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=f"Convert {DATASET_NAME} to SAM2 format"
    )
    parser.add_argument(
        "--path",
        type=str,
        help="Path to the dataset (zip file or extracted folder)",
        default="/mnt/data/Dataset/UltraSound/Raw/AUL.zip",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        help="Output directory for SAM2 format dataset",
        default=f"/mnt/data/Dataset/SaUS/{DATASET_NAME}",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["sa-v", "sa-1b"],
        default="sa-1b",
        help="Output format: sa-v (video style) or sa-1b (image style)",
    )
    parser.add_argument(
        "--save-viz",
        action="store_true",
        help="Save visualization images with overlaid masks",
    )
    parser.add_argument(
        "--include-liver",
        action="store_true",
        default=True,
        help="Include liver segmentation in output",
    )
    parser.add_argument(
        "--include-mass",
        action="store_true",
        default=True,
        help="Include mass/tumor segmentation in output",
    )
    parser.add_argument(
        "--mass-only",
        action="store_true",
        help="Only include images with mass segmentation (exclude Normal)",
    )
    parser.add_argument(
        "--apply-outline-mask",
        action="store_true",
        default=True,
        help="Apply outline mask to remove regions outside ultrasound FOV",
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=None,
        choices=["Benign", "Malignant", "Normal"],
        help="Specific categories to process (default: all)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of images to process (for debugging)",
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        help="Compress output dataset to zip file",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete uncompressed output after zipping",
    )
    return parser.parse_args()


def load_polygon_from_json(json_path: Path) -> Optional[np.ndarray]:
    """Load polygon coordinates from JSON file.
    
    Args:
        json_path: Path to JSON file containing [[x1,y1], [x2,y2], ...]
    
    Returns:
        Numpy array of shape (N, 2) or None if file doesn't exist
    """
    if not json_path.exists():
        return None
    
    with open(json_path, "r") as f:
        coords = json.load(f)
    
    if not coords:
        return None
    
    return np.array(coords, dtype=np.float32)


def polygon_to_mask(
    polygon: np.ndarray,
    height: int,
    width: int,
    fill_value: int = 1,
) -> np.ndarray:
    """Convert polygon to binary mask.
    
    Args:
        polygon: Polygon coordinates (N, 2) as [x, y] pairs
        height: Image height
        width: Image width
        fill_value: Value to fill inside polygon
    
    Returns:
        Binary mask of shape (height, width)
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    
    if polygon is None or len(polygon) < 3:
        return mask
    
    # Convert to integer coordinates for cv2
    pts = polygon.astype(np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], fill_value)
    
    return mask


def apply_outline_mask(
    image: np.ndarray,
    mask: np.ndarray,
    outline_polygon: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply outline mask to remove regions outside ultrasound FOV.
    
    Args:
        image: RGB image
        mask: Segmentation mask
        outline_polygon: Outline polygon coordinates
    
    Returns:
        Tuple of (masked_image, masked_mask)
    """
    if outline_polygon is None:
        return image, mask
    
    h, w = image.shape[:2]
    outline_mask = polygon_to_mask(outline_polygon, h, w, fill_value=1)
    
    # Apply to image (set outside to black)
    masked_image = image.copy()
    masked_image[outline_mask == 0] = 0
    
    # Apply to mask
    masked_mask = mask.copy()
    masked_mask[outline_mask == 0] = 0
    
    return masked_image, masked_mask


def collect_samples(data_dir: Path, categories: Optional[List[str]] = None) -> List[Dict]:
    """Collect all samples from the dataset.
    
    Args:
        data_dir: Root directory of the dataset
        categories: List of categories to include (None = all)
    
    Returns:
        List of sample dicts with paths and metadata
    """
    samples = []
    
    if categories is None:
        categories = CLASS_CATEGORIES
    
    for category in categories:
        cat_dir = data_dir / category
        if not cat_dir.exists():
            continue
        
        img_dir = cat_dir / "image"
        seg_dir = cat_dir / "segmentation"
        
        if not img_dir.exists():
            continue
        
        for img_file in sorted(img_dir.glob("*.jpg")):
            img_id = img_file.stem
            
            sample = {
                "image_path": img_file,
                "image_id": img_id,
                "category": category,
                "liver_path": seg_dir / "liver" / f"{img_id}.json",
                "mass_path": seg_dir / "mass" / f"{img_id}.json",
                "outline_path": seg_dir / "outline" / f"{img_id}.json",
            }
            samples.append(sample)
    
    return samples


def process_sample(
    sample: Dict,
    annotator: SAM2VideoAnnotator,
    include_liver: bool = True,
    include_mass: bool = True,
    apply_outline: bool = True,
    output_format: str = "sa-1b",
) -> bool:
    """Process a single sample and add to annotator.
    
    Args:
        sample: Sample dict with paths and metadata
        annotator: SAM2VideoAnnotator instance
        include_liver: Whether to include liver segmentation
        include_mass: Whether to include mass segmentation
        apply_outline: Whether to apply outline mask
        output_format: Output format
    
    Returns:
        True if sample was processed successfully
    """
    # Load image
    image = cv2.imread(str(sample["image_path"]))
    if image is None:
        return False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    h, w = image.shape[:2]
    
    # Load polygons
    liver_poly = load_polygon_from_json(sample["liver_path"]) if include_liver else None
    mass_poly = load_polygon_from_json(sample["mass_path"]) if include_mass else None
    outline_poly = load_polygon_from_json(sample["outline_path"]) if apply_outline else None
    
    # Create combined mask with different instance IDs
    # instance 1 = liver, instance 2 = mass
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Determine mass category ID based on classification
    if sample["category"] == "Benign":
        mass_category_id = 2  # mass_benign
    elif sample["category"] == "Malignant":
        mass_category_id = 3  # mass_malignant
    else:
        mass_category_id = None  # Normal - no mass
    
    # Instance to category mapping
    instance_to_category = {}
    
    # Add liver (instance 1 -> category 1)
    if liver_poly is not None:
        liver_mask = polygon_to_mask(liver_poly, h, w, fill_value=1)
        mask[liver_mask > 0] = 1
        instance_to_category[1] = 1  # liver
    
    # Add mass (instance 2 -> category 2 or 3 based on classification)
    if mass_poly is not None and mass_category_id is not None:
        mass_mask = polygon_to_mask(mass_poly, h, w, fill_value=2)
        mask[mass_mask > 0] = 2  # Mass overwrites liver where they overlap
        instance_to_category[2] = mass_category_id
    
    # Check if we have any annotations
    if not np.any(mask > 0):
        return False
    
    # Apply outline mask if requested
    if apply_outline and outline_poly is not None:
        image, mask = apply_outline_mask(image, mask, outline_poly)
    
    # Generate unique name
    image_name = f"{sample['category']}_{sample['image_id']}"
    
    # Add to annotator
    annotator.add_2d(
        image,
        mask,
        video_name=image_name,
        format=output_format,
        instance_to_category=instance_to_category,
        metadata={
            "image_id": sample["image_id"],
            "category": sample["category"],
            "dataset": DATASET_NAME,
            "has_liver": liver_poly is not None,
            "has_mass": mass_poly is not None,
        },
    )
    
    return True


def main():
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    
    save_path = Path(args.save_dir)
    data_path = Path(args.path)
    
    # Determine if we need to extract from zip
    use_temp_dir = data_path.suffix.lower() == ".zip"
    
    # Determine categories to process
    categories = args.categories
    if args.mass_only:
        # Exclude Normal if only processing mass
        if categories is None:
            categories = ["Benign", "Malignant"]
        else:
            categories = [c for c in categories if c != "Normal"]
    
    # Create annotator
    annotator = SAM2VideoAnnotator(
        out_dir=save_path,
        dataset_name=DATASET_NAME,
        categories=CATEGORIES,
        save_visualization=args.save_viz,
        video_environment="ultrasound",
        video_split="train",
    )
    
    logger.info(f"Output format: {args.format}")
    logger.info(f"Categories: {categories or 'all'}")
    logger.info(f"Include liver: {args.include_liver}")
    logger.info(f"Include mass: {args.include_mass}")
    logger.info(f"Apply outline mask: {args.apply_outline_mask}")
    
    # Context manager for temp directory
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
            
            # Handle nested zip files (Benign.zip, Malignant.zip, Normal.zip)
            for nested_zip in data_dir.glob("*.zip"):
                logger.info(f"Extracting nested archive: {nested_zip.name}")
                shutil.unpack_archive(str(nested_zip), data_dir)
            
            logger.info("Extraction complete")
        else:
            data_dir = data_path
        
        # Collect all samples
        samples = collect_samples(data_dir, categories)
        logger.info(f"Found {len(samples)} samples")
        
        if not samples:
            logger.error("No samples found!")
            return
        
        # Limit samples if requested
        if args.max_images:
            samples = samples[:args.max_images]
            logger.info(f"Processing only {len(samples)} samples")
        
        # Process samples
        processed = 0
        skipped = 0
        
        for idx, sample in enumerate(samples):
            success = process_sample(
                sample,
                annotator,
                include_liver=args.include_liver,
                include_mass=args.include_mass,
                apply_outline=args.apply_outline_mask,
                output_format=args.format,
            )
            
            if success:
                processed += 1
            else:
                skipped += 1
            
            if (idx + 1) % 50 == 0:
                logger.info(
                    f"Processed {idx + 1}/{len(samples)} samples "
                    f"({processed} success, {skipped} skipped)"
                )
        
        logger.info(f"Total: {processed} processed, {skipped} skipped")
    
    # Finalize dataset
    stats = annotator.finalize()
    logger.info(f"Dataset stats: {stats}")
    
    # Optional: zip output
    if args.zip:
        zip_path = save_path.with_suffix(".zip")
        logger.info(f"Compressing to {zip_path}")
        shutil.make_archive(str(save_path), "zip", save_path)
        
        if args.delete:
            logger.info(f"Deleting uncompressed folder: {save_path}")
            shutil.rmtree(save_path)
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
