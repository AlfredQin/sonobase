"""GIST514-DB dataset preprocessing.

Dataset: Gastrointestinal stromal tumor (GIST) ultrasound classification/segmentation
Source: https://github.com/howardchina/query2

Data format:
- usd514_jpeg_roi/images/{id}.jpg: Ultrasound images
- usd514_jpeg_roi/annotations/train_anno_crop_split_0.json: Training annotations (COCO format)
- usd514_jpeg_roi/annotations/val_anno_crop_split_0.json: Validation annotations (COCO format)

Categories: lmym (1), gist (2)

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

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ..utils.sam2_annotator import SAM2VideoAnnotator

DATASET_NAME = "GIST514-DB"

CATEGORIES = [
    {"supercategory": "tumor", "id": 1, "name": "lmym"},
    {"supercategory": "tumor", "id": 2, "name": "gist"},
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
        choices=["train", "val", "all"],
        default="all",
        help="Which split to process",
    )
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--zip", action="store_true")
    parser.add_argument("--delete", action="store_true")
    return parser.parse_args()


def polygon_to_mask(segmentation: List, height: int, width: int) -> np.ndarray:
    """Convert polygon segmentation to binary mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    
    for polygon in segmentation:
        if len(polygon) < 6:  # Need at least 3 points
            continue
        
        # Reshape to (N, 2) array
        points = np.array(polygon).reshape(-1, 2).astype(np.int32)
        cv2.fillPoly(mask, [points], 1)
    
    return mask


def process_coco_annotations(
    coco_data: Dict,
    images_dir: Path,
    split_name: str,
) -> List[Dict]:
    """Process COCO format annotations into samples."""
    samples = []
    
    # Build image lookup
    image_lookup = {img["id"]: img for img in coco_data["images"]}
    
    # Group annotations by image
    image_to_anns = {}
    for ann in coco_data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in image_to_anns:
            image_to_anns[img_id] = []
        image_to_anns[img_id].append(ann)
    
    for img_id, image_info in image_lookup.items():
        filename = image_info["file_name"]
        img_path = images_dir / filename
        
        if not img_path.exists():
            continue
        
        annotations = image_to_anns.get(img_id, [])
        if not annotations:
            continue
        
        samples.append({
            "image_path": img_path,
            "image_id": img_path.stem,
            "height": image_info["height"],
            "width": image_info["width"],
            "annotations": annotations,
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
        else:
            data_dir = data_path

        # Find data folder
        if (data_dir / "usd514_jpeg_roi").exists():
            data_dir = data_dir / "usd514_jpeg_roi"

        images_dir = data_dir / "images"
        annotations_dir = data_dir / "annotations"

        all_samples = []

        # Process splits
        splits_to_process = []
        if args.split == "all":
            splits_to_process = [("train", "train_anno_crop_split_0.json"), ("val", "val_anno_crop_split_0.json")]
        elif args.split == "train":
            splits_to_process = [("train", "train_anno_crop_split_0.json")]
        else:
            splits_to_process = [("val", "val_anno_crop_split_0.json")]

        for split_name, json_name in splits_to_process:
            json_path = annotations_dir / json_name
            if not json_path.exists():
                logger.warning(f"Annotation file not found: {json_path}")
                continue

            with open(json_path, "r") as f:
                coco_data = json.load(f)

            samples = process_coco_annotations(coco_data, images_dir, split_name)
            logger.info(f"Found {len(samples)} samples for {split_name}")
            all_samples.extend(samples)

        logger.info(f"Total samples: {len(all_samples)}")

        if args.max_images:
            all_samples = all_samples[:args.max_images]

        processed = 0
        for idx, sample in enumerate(all_samples):
            image = cv2.imread(str(sample["image_path"]))
            if image is None:
                continue
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = sample["height"], sample["width"]

            # Create combined mask from all annotations
            combined_mask = np.zeros((h, w), dtype=np.uint8)
            
            for ann in sample["annotations"]:
                if "segmentation" not in ann:
                    continue
                
                category_id = ann["category_id"]
                mask = polygon_to_mask(ann["segmentation"], h, w)
                combined_mask[mask > 0] = category_id

            if not np.any(combined_mask > 0):
                continue

            image_name = f"{sample['split']}_{sample['image_id']}"
            annotator.add_2d(
                image_rgb,
                combined_mask,
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
                logger.info(f"Processed {idx + 1}/{len(all_samples)}")

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
