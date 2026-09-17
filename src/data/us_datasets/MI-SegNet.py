"""MI-SegNet (Myocardial Infarction Segmentation Network) dataset preprocessing.

Dataset: Cardiac ultrasound for myocardial infarction segmentation
Source: https://github.com/KU-CVLAB/MI-SegNet

Data format:
- Training/img/imgXXXX.png: Training images
- Training/label/labelXXXX.png: Training masks
- ValS/img/imgXXXX.png: Validation images
- ValS/label/labelXXXX.png: Validation masks
- TS3/img/imgXXXX.png: Test images
- TS3/label/labelXXXX.png: Test masks

Category: myocardium (heart muscle)

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

DATASET_NAME = "MI-SegNet"

CATEGORIES = [
    {"supercategory": "organ", "id": 1, "name": "myocardium"},
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
    parser.add_argument(
        "--format",
        type=str,
        choices=["sa-v", "sa-1b"],
        default="sa-1b",
        help="Output format",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test", "all"],
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


def collect_samples(data_dir: Path, split: str = "all") -> List[Dict]:
    """Collect image-mask pairs from the dataset."""
    samples = []

    # Handle nested folder structure
    if (data_dir / "MI_SegNet_dataset").exists():
        data_dir = data_dir / "MI_SegNet_dataset"
    elif (data_dir / "MI-SegNet").exists():
        data_dir = data_dir / "MI-SegNet"

    # Split folder mapping
    split_folders = {
        "train": "Training",
        "val": "ValS",
        "test": "TS3",
    }

    splits = ["train", "val", "test"] if split == "all" else [split]

    for split_name in splits:
        split_folder = split_folders.get(split_name)
        if not split_folder:
            continue

        split_dir = data_dir / split_folder
        if not split_dir.exists():
            continue

        img_dir = split_dir / "img"
        label_dir = split_dir / "label"

        if not img_dir.exists() or not label_dir.exists():
            continue

        for img_file in sorted(img_dir.glob("*.png")):
            # Extract image ID (e.g., img0001 -> 0001)
            img_id = img_file.stem.replace("img", "")
            label_file = label_dir / f"label{img_id}.png"

            if label_file.exists():
                samples.append({
                    "image_path": img_file,
                    "mask_path": label_file,
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

            # Binarize mask
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

            if (idx + 1) % 500 == 0:
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
