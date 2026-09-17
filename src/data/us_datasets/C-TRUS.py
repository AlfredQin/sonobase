"""C-TRUS (Prostate Transrectal Ultrasound) dataset preprocessing.

Dataset: Prostate ultrasound segmentation dataset
Structure:
- original/{id}.jpg: Ultrasound images
- labels/{id}.jpg: Binary segmentation masks
- c-trus.csv: Metadata with quality labels

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

DATASET_NAME = "C-TRUS"

CATEGORIES = [
    {"supercategory": "organ", "id": 1, "name": "prostate"},
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
    parser.add_argument(
        "--quality",
        type=str,
        nargs="+",
        default=None,
        choices=["high", "medium", "low"],
        help="Filter by quality labels",
    )
    parser.add_argument("--zip", action="store_true")
    parser.add_argument("--delete", action="store_true")
    return parser.parse_args()


def rgb2gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def collect_samples(data_dir: Path, quality_filter: List[str] = None) -> List[Dict]:
    """Collect image-mask pairs with optional quality filtering."""
    samples = []

    # Handle nested folder structure
    if (data_dir / "c-trus").exists():
        data_dir = data_dir / "c-trus"

    img_dir = data_dir / "original"
    mask_dir = data_dir / "labels"
    csv_path = data_dir / "c-trus.csv"

    if not img_dir.exists() or not mask_dir.exists():
        return samples

    # Load metadata if available
    metadata = {}
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            metadata[row["file"]] = {
                "quality": row.get("quality_name", "unknown"),
                "patient": row.get("patient", None),
                "fold": row.get("testitem_in_fold", None),
            }

    # Collect image-mask pairs
    for img_file in sorted(img_dir.glob("*.jpg")):
        img_id = img_file.stem
        mask_file = mask_dir / f"{img_id}.jpg"

        if not mask_file.exists():
            continue

        # Get metadata
        file_meta = metadata.get(img_file.name, {})
        quality = file_meta.get("quality", "unknown")

        # Filter by quality if specified
        if quality_filter and quality not in quality_filter:
            continue

        samples.append({
            "image_path": img_file,
            "mask_path": mask_file,
            "image_id": img_id,
            "quality": quality,
            "patient": file_meta.get("patient"),
            "fold": file_meta.get("fold"),
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
    if args.quality:
        logger.info(f"Quality filter: {args.quality}")

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

        samples = collect_samples(data_dir, args.quality)
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

            # Split disconnected regions into separate object instances
            num_labels, labels = cv2.connectedComponents(mask_binary, connectivity=8)
            if num_labels > 2:  # more than 1 foreground component
                mask_binary = labels.astype(np.uint8)  # 0=bg, 1..K=instances
            # else: single component, keep as-is (all foreground = 1)

            image_name = f"{sample['image_id']}"
            annotator.add_2d(
                image_rgb,
                mask_binary,
                video_name=image_name,
                format=args.format,
                metadata={
                    "image_id": sample["image_id"],
                    "quality": sample["quality"],
                    "patient": sample["patient"],
                    "fold": sample["fold"],
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
