"""JNU-IFM (Jinan University - Intrapartum Fetal Monitoring) dataset preprocessing.

Dataset: Ultrasound video dataset for fetal head and symphysis pubis segmentation
Source: https://zenodo.org/records/7851339

Structure (inside JNU-IFM.zip → us_data.zip → us_data/):
    {video_timestamp}/
        image/{video_timestamp}_{frame_id}.png      # Grayscale ultrasound frames (1295x1026)
        mask/{video_timestamp}_{frame_id}_mask.png   # Segmentation masks
        mask_enhance/                                 # Visualization (not used)
        frame_label.csv                              # Frame-level labels

Mask pixel values:
    0 = background
    7 = SP (symphysis pubis)
    8 = Head (fetal head)

Frame labels (in frame_label.csv):
    3 = None, 4 = OnlySP, 5 = OnlyHead, 6 = SP+Head

Stats: 78 videos, 6224 frames, 51 patients

Output: SAM2-compatible format (SA-1B for individual frames, SA-V for video)
"""

import argparse
import logging
import os
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

DATASET_NAME = "JNU-IFM"

# Mask pixel value 7 → SP (id 1), pixel value 8 → Head (id 2)
CATEGORIES = [
    {"supercategory": "fetal", "id": 1, "name": "symphysis_pubis"},
    {"supercategory": "fetal", "id": 2, "name": "fetal_head"},
]

# Mapping from raw mask pixel values to category IDs
LABEL_MAP = {7: 1, 8: 2}


def parse_args():
    parser = argparse.ArgumentParser(description=f"Convert {DATASET_NAME} to SAM2 format")
    parser.add_argument(
        "--path",
        type=str,
        default=f"/mnt/data/Dataset/UltraSound/Raw/{DATASET_NAME}.zip",
        help="Path to dataset zip file",
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
        default="sa-v",
        help="Output format (sa-1b: individual frames, sa-v: video sequences)",
    )
    parser.add_argument("--save-viz", action="store_true")
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--zip", action="store_true")
    parser.add_argument("--delete", action="store_true")
    return parser.parse_args()


def gray2rgb(gray: np.ndarray) -> np.ndarray:
    """Convert grayscale image to RGB."""
    if gray.ndim == 2:
        return np.stack([gray, gray, gray], axis=-1)
    elif gray.ndim == 3 and gray.shape[-1] == 1:
        return np.repeat(gray, 3, axis=-1)
    return gray


def remap_mask(mask: np.ndarray) -> np.ndarray:
    """Remap raw mask pixel values (7=SP, 8=Head) to category IDs (1, 2)."""
    remapped = np.zeros_like(mask)
    for raw_val, cat_id in LABEL_MAP.items():
        remapped[mask == raw_val] = cat_id
    return remapped


def collect_samples(data_dir: Path) -> List[Dict]:
    """Collect video folders with image-mask pairs.

    Returns a list of dicts, one per video, each containing
    the video name and sorted list of (image_path, mask_path, frame_id) tuples.
    """
    samples = []

    # Handle nested folder structure
    if (data_dir / "us_data").exists():
        data_dir = data_dir / "us_data"

    for video_dir in sorted(data_dir.iterdir()):
        if not video_dir.is_dir():
            continue

        image_dir = video_dir / "image"
        mask_dir = video_dir / "mask"

        if not image_dir.exists() or not mask_dir.exists():
            continue

        video_name = video_dir.name
        frames = []

        for img_file in sorted(image_dir.glob("*.png")):
            # Extract frame ID: {video_name}_{frame_id}.png
            stem = img_file.stem
            # Frame ID is after the last underscore
            parts = stem.rsplit("_", 1)
            if len(parts) != 2:
                continue
            frame_id = parts[1]

            mask_file = mask_dir / f"{stem}_mask.png"
            if mask_file.exists():
                frames.append({
                    "image_path": img_file,
                    "mask_path": mask_file,
                    "frame_id": int(frame_id),
                })

        if frames:
            # Sort by frame_id for proper video ordering
            frames.sort(key=lambda x: x["frame_id"])
            samples.append({
                "video_name": video_name,
                "frames": frames,
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

            # Handle nested us_data.zip inside the outer zip
            inner_zip = Path(temp_dir) / "us_data.zip"
            if inner_zip.exists():
                logger.info("Extracting nested us_data.zip")
                shutil.unpack_archive(str(inner_zip), temp_dir)

            data_dir = Path(temp_dir)
        else:
            data_dir = data_path

        samples = collect_samples(data_dir)
        logger.info(f"Found {len(samples)} videos")

        if args.max_videos:
            samples = samples[:args.max_videos]

        processed = 0
        total_frames = 0

        for idx, sample in enumerate(samples):
            video_name = sample["video_name"]
            frames_data = sample["frames"]

            if args.format == "sa-1b":
                # Export each annotated frame individually
                for frame_info in frames_data:
                    image = cv2.imread(str(frame_info["image_path"]), cv2.IMREAD_GRAYSCALE)
                    if image is None:
                        continue

                    mask_raw = cv2.imread(str(frame_info["mask_path"]), cv2.IMREAD_GRAYSCALE)
                    if mask_raw is None:
                        continue

                    mask = remap_mask(mask_raw)

                    # Skip frames with no annotations
                    if not np.any(mask > 0):
                        continue

                    image_rgb = gray2rgb(image)
                    frame_name = f"{video_name}_{frame_info['frame_id']:04d}"

                    annotator.add_2d(
                        image_rgb,
                        mask,
                        video_name=frame_name,
                        format="sa-1b",
                        metadata={
                            "video_name": video_name,
                            "frame_id": frame_info["frame_id"],
                            "dataset": DATASET_NAME,
                        },
                    )
                    total_frames += 1
            else:
                # Export as video sequence (only frames with annotations)
                frames = []
                frame_masks = []
                frame_ids = []

                for frame_info in frames_data:
                    image = cv2.imread(str(frame_info["image_path"]), cv2.IMREAD_GRAYSCALE)
                    if image is None:
                        continue

                    mask_raw = cv2.imread(str(frame_info["mask_path"]), cv2.IMREAD_GRAYSCALE)
                    if mask_raw is None:
                        continue

                    mask = remap_mask(mask_raw)

                    if not np.any(mask > 0):
                        continue

                    frames.append(gray2rgb(image))
                    frame_masks.append(mask)
                    frame_ids.append(frame_info["frame_id"])

                if frames:
                    frames_arr = np.stack(frames, axis=0)
                    masks_arr = np.stack(frame_masks, axis=0)

                    annotator.add_video(
                        frames_arr,
                        masks_arr,
                        video_name=video_name,
                        metadata={
                            "video_name": video_name,
                            "frame_ids": frame_ids,
                            "num_annotated": len(frames),
                            "dataset": DATASET_NAME,
                        },
                    )
                    total_frames += len(frames)

            processed += 1

            if (idx + 1) % 10 == 0:
                logger.info(f"Processed {idx + 1}/{len(samples)} videos ({total_frames} frames)")

        logger.info(f"Total: {processed} videos, {total_frames} annotated frames")

    stats = annotator.finalize()
    logger.info(f"Dataset stats: {stats}")

    if args.zip:
        shutil.make_archive(str(save_path), "zip", save_path)
        if args.delete:
            shutil.rmtree(save_path)

    logger.info("Done!")


if __name__ == "__main__":
    main()
