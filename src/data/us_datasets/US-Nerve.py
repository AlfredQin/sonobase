"""US_Nerve (Ultrasound Nerve Segmentation) dataset preprocessing.

Dataset: Kaggle ultrasound nerve segmentation competition
Source: https://www.kaggle.com/c/ultrasound-nerve-segmentation/data

Data format:
- train/{patient_id}_{frame_id}.tif: Ultrasound images
- train/{patient_id}_{frame_id}_mask.tif: Binary nerve masks

Category: cervical_nerves (brachial plexus)

Output: SAM2-compatible format (SA-V for videos, SA-1B for images)
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

DATASET_NAME = "US-Nerve"

CATEGORIES = [
    {"supercategory": "nerve", "id": 1, "name": "cervical_nerves"},
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
        default="sa-v",
        help="Output format (sa-v treats each patient as video)",
    )
    parser.add_argument("--min-mask-pixels", type=int, default=10, help="Minimum mask area")
    parser.add_argument("--save-viz", action="store_true")
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--zip", action="store_true")
    parser.add_argument("--delete", action="store_true")
    return parser.parse_args()


def rgb2gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def collect_samples(data_dir: Path) -> Dict[str, List[Dict]]:
    """Collect image-mask pairs grouped by patient ID."""
    patients = defaultdict(list)

    # Handle nested folder structure
    train_dir = None
    for candidate in [
        data_dir / "train",
        data_dir / "ultrasound-nerve-segmentation" / "train",
    ]:
        if candidate.exists():
            train_dir = candidate
            break

    if train_dir is None:
        return patients

    # Find all mask files
    mask_files = {}
    for mask_file in train_dir.glob("*_mask.tif"):
        # Pattern: {patient_id}_{frame_id}_mask.tif
        name = mask_file.stem.replace("_mask", "")
        mask_files[name] = mask_file

    # Match images to masks
    for img_file in sorted(train_dir.glob("*.tif")):
        if "_mask" in img_file.name:
            continue

        name = img_file.stem  # {patient_id}_{frame_id}
        if name not in mask_files:
            continue

        parts = name.split("_")
        if len(parts) < 2:
            continue

        patient_id = parts[0]
        frame_id = int(parts[1])

        patients[patient_id].append({
            "image_path": img_file,
            "mask_path": mask_files[name],
            "frame_id": frame_id,
        })

    # Sort frames within each patient
    for patient_id in patients:
        patients[patient_id] = sorted(patients[patient_id], key=lambda x: x["frame_id"])

    return patients


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
        min_mask_pixels=args.min_mask_pixels,
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

        patients = collect_samples(data_dir)
        logger.info(f"Found {len(patients)} patients")

        if args.max_patients:
            patient_ids = list(patients.keys())[:args.max_patients]
            patients = {k: patients[k] for k in patient_ids}

        processed = 0
        total_frames = 0
        frames_with_mask = 0

        for patient_id, frames in patients.items():
            video_name = f"{patient_id}"

            if args.format == "sa-v":
                # Process as video
                processed_frames = []
                processed_masks = []

                for frame_data in frames:
                    img = cv2.imread(str(frame_data["image_path"]))
                    if img is None:
                        continue
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    mask = rgb2gray(cv2.imread(str(frame_data["mask_path"]))).astype(np.uint8)

                    # Binarize mask
                    mask_binary = np.zeros_like(mask, dtype=np.uint8)
                    mask_binary[mask >= 150] = 1

                    # Include all frames for video, but track which have masks
                    processed_frames.append(img_rgb)
                    processed_masks.append(mask_binary)

                    if np.any(mask_binary > 0):
                        frames_with_mask += 1

                if processed_frames:
                    frames_arr = np.stack(processed_frames, axis=0)
                    masks_arr = np.stack(processed_masks, axis=0)

                    # Only add if at least one frame has a mask
                    if np.any(masks_arr > 0):
                        annotator.add_video(
                            frames_arr,
                            masks_arr,
                            video_name=video_name,
                            metadata={
                                "patient_id": patient_id,
                                "num_frames": len(processed_frames),
                                "dataset": DATASET_NAME,
                            },
                        )
                        total_frames += len(processed_frames)
                        processed += 1

            else:
                # Process as individual images (SA-1B)
                for frame_data in frames:
                    img = cv2.imread(str(frame_data["image_path"]))
                    if img is None:
                        continue
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    mask = rgb2gray(cv2.imread(str(frame_data["mask_path"]))).astype(np.uint8)

                    mask_binary = np.zeros_like(mask, dtype=np.uint8)
                    mask_binary[mask >= 150] = 1

                    # Skip frames without mask for SA-1B format
                    if not np.any(mask_binary > 0):
                        continue

                    frame_name = f"{video_name}_{frame_data['frame_id']:04d}"
                    annotator.add_2d(
                        img_rgb,
                        mask_binary,
                        video_name=frame_name,
                        format="sa-1b",
                        metadata={
                            "patient_id": patient_id,
                            "frame_id": frame_data["frame_id"],
                            "dataset": DATASET_NAME,
                        },
                    )
                    total_frames += 1
                    frames_with_mask += 1

                processed += 1

            if processed % 10 == 0:
                logger.info(f"Processed {processed}/{len(patients)} patients ({total_frames} frames)")

        logger.info(f"Total: {processed} patients, {total_frames} frames, {frames_with_mask} with masks")

    stats = annotator.finalize()
    logger.info(f"Dataset stats: {stats}")

    if args.zip:
        shutil.make_archive(str(save_path), "zip", save_path)
        if args.delete:
            shutil.rmtree(save_path)

    logger.info("Done!")


if __name__ == "__main__":
    main()
