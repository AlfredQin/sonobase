"""UBPD (Ultrasound Brachial Plexus Dataset) preprocessing.

Dataset: Ultrasound brachial plexus segmentation with multiple structures
Source: https://ubpd.worldwidetracing.com:9443/

Data format:
- JPEGImages/{id}.jpg: Ultrasound images
- json_train/{id}.json: LabelMe-style polygon annotations

Categories:
- jingmai -> vein (category 1)
- dongmai -> artery (category 2)
- jirouzuzhi -> muscle (category 3)
- shenjing -> nerve (category 4)
- zhifang -> fat (category 5)
- jizhu -> spine (category 6)

Output: SAM2-compatible format (SA-1B style)
"""

import argparse
import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from PIL import Image, ImageDraw

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ..utils.sam2_annotator import SAM2VideoAnnotator

DATASET_NAME = "UBPD"

CATEGORIES = [
    {"supercategory": "vessel", "id": 1, "name": "vein"},
    {"supercategory": "vessel", "id": 2, "name": "artery"},
    {"supercategory": "tissue", "id": 3, "name": "muscle"},
    {"supercategory": "nerve", "id": 4, "name": "nerve"},
    {"supercategory": "tissue", "id": 5, "name": "fat"},
    {"supercategory": "bone", "id": 6, "name": "spine"},
]

# Mapping from Chinese label names to category IDs
LABEL_TO_ID = {
    "jingmai": 1,  # vein
    "dongmai": 2,  # artery
    "jirouzuzhi": 3,  # muscle
    "shenjing": 4,  # nerve
    "zhifang": 5,  # fat
    "jizhu": 6,  # spine
}


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


def polygon_to_mask(polygon: List[List[float]], height: int, width: int, category_id: int) -> np.ndarray:
    """Convert polygon points to a filled mask."""
    mask = Image.new("L", (width, height), 0)
    points = [(p[0], p[1]) for p in polygon]
    ImageDraw.Draw(mask).polygon(points, outline=category_id, fill=category_id)
    return np.array(mask)


def collect_samples(data_dir: Path) -> List[Dict]:
    """Collect image-annotation pairs from the dataset."""
    samples = []

    # Handle nested folder structure
    images_dir = None
    json_dir = None
    
    for candidate_base in [data_dir, data_dir / DATASET_NAME]:
        candidate_img = candidate_base / "JPEGImages"
        candidate_json = candidate_base / "json_train"
        if candidate_img.exists() and candidate_json.exists():
            images_dir = candidate_img
            json_dir = candidate_json
            break

    if images_dir is None or json_dir is None:
        return samples

    for img_file in sorted(images_dir.glob("*.jpg")):
        img_id = img_file.stem
        json_file = json_dir / f"{img_id}.json"

        if json_file.exists():
            samples.append({
                "image_path": img_file,
                "json_path": json_file,
                "image_id": img_id,
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
        skipped = 0
        for idx, sample in enumerate(samples):
            try:
                image = cv2.imread(str(sample["image_path"]))
                if image is None:
                    continue
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                h, w = image.shape[:2]

                # Load JSON annotation
                with open(sample["json_path"], "r", errors="ignore") as f:
                    annotation = json.load(f)

                # Create mask from polygons
                mask_combined = np.zeros((h, w), dtype=np.uint8)
                
                for shape in annotation.get("shapes", []):
                    label = shape.get("label", "")
                    points = shape.get("points", [])
                    
                    if label not in LABEL_TO_ID:
                        continue
                    
                    category_id = LABEL_TO_ID[label]
                    mask = polygon_to_mask(points, h, w, category_id)
                    # Overwrite with new category (later annotations take precedence)
                    mask_combined[mask > 0] = mask[mask > 0]

                if not np.any(mask_combined > 0):
                    skipped += 1
                    continue

                image_name = f"{sample['image_id']}"
                annotator.add_2d(
                    image_rgb,
                    mask_combined,
                    video_name=image_name,
                    format=args.format,
                    metadata={
                        "image_id": sample["image_id"],
                        "dataset": DATASET_NAME,
                    },
                )
                processed += 1

                if (idx + 1) % 200 == 0:
                    logger.info(f"Processed {idx + 1}/{len(samples)}")

            except Exception as e:
                logger.warning(f"Error processing {sample['image_id']}: {e}")
                continue

        logger.info(f"Total processed: {processed}, skipped: {skipped}")

    stats = annotator.finalize()
    logger.info(f"Dataset stats: {stats}")

    if args.zip:
        shutil.make_archive(str(save_path), "zip", save_path)
        if args.delete:
            shutil.rmtree(save_path)

    logger.info("Done!")


if __name__ == "__main__":
    main()
