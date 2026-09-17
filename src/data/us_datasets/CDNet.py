"""CDNet (Cervical Dilation Network) dataset preprocessing.

Dataset: Cervical dilation ultrasound segmentation
Source: CDNet.zip → Images/ and Masks/ at top level

Data format:
- Images/{name}.png: RGB ultrasound images (various sizes)
- Masks/{name}.png: Binary segmentation masks (0=background, 1=cervix)

Stats: 327 image-mask pairs, 137 patient sequences

Category: cervix (id=1)

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

DATASET_NAME = "CDNet"

CATEGORIES = [
    {"supercategory": "cervix", "id": 1, "name": "cervix"},
]


def parse_args():
    parser = argparse.ArgumentParser(description=f"Convert {DATASET_NAME} to SAM2 format")
    parser.add_argument(
        "--path",
        type=str,
        default=f"/mnt/data/Dataset/UltraSound/Raw/{DATASET_NAME}.zip",
        help="Path to dataset (zip file or directory)",
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


def collect_samples(data_dir: Path) -> List[Dict]:
    """Collect image-mask pairs from CDNet directory.

    CDNet.zip extracts Images/ and Masks/ directly (no wrapper folder).
    """
    samples = []

    # Handle possible nested folder structure
    if (data_dir / "CDNet").exists():
        data_dir = data_dir / "CDNet"

    images_dir = data_dir / "Images"
    masks_dir = data_dir / "Masks"

    if not images_dir.exists() or not masks_dir.exists():
        return samples

    for img_file in sorted(images_dir.glob("*.png")):
        img_id = img_file.stem
        mask_file = masks_dir / f"{img_id}.png"

        if mask_file.exists():
            samples.append({
                "image_path": img_file,
                "mask_path": mask_file,
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
        for idx, sample in enumerate(samples):
            image = cv2.imread(str(sample["image_path"]))
            if image is None:
                continue
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            mask = cv2.imread(str(sample["mask_path"]), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue

            # Mask values are already 0/1; ensure uint8
            mask_binary = (mask > 0).astype(np.uint8)

            if not np.any(mask_binary > 0):
                continue

            image_name = sample["image_id"]

            annotator.add_2d(
                image_rgb,
                mask_binary,
                video_name=image_name,
                format=args.format,
                metadata={
                    "image_id": sample["image_id"],
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
