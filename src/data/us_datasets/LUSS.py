"""LUSS (Lung Ultrasound Segmentation) Phantom dataset preprocessing.

Dataset: Lung ultrasound phantom dataset with artefact segmentation
Source: https://archive.researchdata.leeds.ac.uk/1263/

Data format:
- images/{id}.png: Ultrasound images
- masks/{id}.png: Multi-class artefact masks

Categories: rib, pleural_line, A_line, B_line, B_line_confluence

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

DATASET_NAME = "LUSS"

CATEGORIES = [
    {"supercategory": "artefact", "id": 1, "name": "rib"},
    {"supercategory": "artefact", "id": 2, "name": "pleural_line"},
    {"supercategory": "artefact", "id": 3, "name": "A_line"},
    {"supercategory": "artefact", "id": 4, "name": "B_line"},
    {"supercategory": "artefact", "id": 5, "name": "B_line_confluence"},
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
        choices=["train", "test", "all"],
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


def semantic_to_instance_mask(mask: np.ndarray) -> tuple[np.ndarray, Dict[int, int]]:
    """Split each semantic class into connected-component instances.

    The raw LUSS masks encode semantic class IDs. For disconnected regions of
    the same class, we create separate instance IDs while preserving the
    category ID through the returned mapping.
    """
    instance_mask = np.zeros(mask.shape, dtype=np.uint16)
    instance_to_category = {}
    next_instance_id = 1

    for category_id in sorted(int(x) for x in np.unique(mask) if x != 0):
        binary_mask = (mask == category_id).astype(np.uint8)
        num_components, labeled = cv2.connectedComponents(binary_mask)

        for component_id in range(1, num_components):
            instance_mask[labeled == component_id] = next_instance_id
            instance_to_category[next_instance_id] = category_id
            next_instance_id += 1

    return instance_mask, instance_to_category


def collect_samples(data_dir: Path, split: str = "all") -> List[Dict]:
    """Collect image-mask pairs.
    
    Args:
        data_dir: Root data directory
        split: Which split to process ('train', 'test', 'all')
    """
    samples = []

    # Handle nested folder structure
    if (data_dir / "data").exists():
        data_dir = data_dir / "data"

    splits = ["train", "test"] if split == "all" else [split]

    for split_name in splits:
        split_dir = data_dir / split_name
        if not split_dir.exists():
            continue

        images_dir = split_dir / "images"
        masks_dir = split_dir / "masks"

        if not images_dir.exists() or not masks_dir.exists():
            continue

        for img_file in sorted(images_dir.glob("*.png")):
            img_id = img_file.stem
            mask_file = masks_dir / f"{img_id}.png"

            if mask_file.exists():
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

    annotator = SAM2VideoAnnotator(
        out_dir=save_path,
        dataset_name=DATASET_NAME,
        categories=CATEGORIES,
        save_visualization=args.save_viz,
        video_environment="ultrasound",
        video_split="train",
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

            # Load semantic multi-class mask and split disconnected regions into
            # per-instance IDs while preserving the original category labels.
            semantic_mask = rgb2gray(cv2.imread(str(sample["mask_path"]))).astype(np.uint8)

            if not np.any(semantic_mask > 0):
                continue

            instance_mask, instance_to_category = semantic_to_instance_mask(semantic_mask)

            split_name = sample.get("split", "unknown")
            image_name = f"{split_name}_{sample['image_id']}"
            annotator.add_2d(
                image_rgb,
                instance_mask,
                video_name=image_name,
                format=args.format,
                instance_to_category=instance_to_category,
                metadata={
                    "image_id": sample["image_id"],
                    "split": split_name,
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
