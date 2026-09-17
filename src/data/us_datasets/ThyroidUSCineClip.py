"""ThyroidUSCineClip (Thyroid Ultrasound Cine Clip) dataset preprocessing.

Dataset: Thyroid ultrasound cine clips with segmentation masks
Source: https://stanfordaimi.azurewebsites.net/datasets/a72f2b02-7b53-4c5d-963c-d7253220bfd5

Data format:
- dataset.hdf5 containing:
  - image: ultrasound frames
  - mask: segmentation masks
  - annot_id: annotation IDs
  - frame_num: frame numbers

Category: thyroid

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

DATASET_NAME = "ThyroidUSCineClip"

CATEGORIES = [
    {"supercategory": "thyroid", "id": 1, "name": "thyroid"},
]


def parse_args():
    parser = argparse.ArgumentParser(description=f"Convert {DATASET_NAME} to SAM2 format")
    parser.add_argument(
        "--path",
        type=str,
        default=f"/mnt/data/Dataset/UltraSound/Raw/{DATASET_NAME}",
        help="Path to dataset folder containing dataset.hdf5",
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
        help="Output format (sa-v treats each clip as video)",
    )
    parser.add_argument("--save-viz", action="store_true")
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--zip", action="store_true")
    parser.add_argument("--delete", action="store_true")
    return parser.parse_args()


def fill_mask(mask: np.ndarray) -> np.ndarray:
    """Fill holes in binary mask."""
    mask_filled = mask.copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        cv2.drawContours(mask_filled, [contour], -1, 1, -1)
    return mask_filled


def main():
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    # Lazy import h5py
    try:
        import h5py
    except ImportError:
        logger.error("h5py is required. Install with: pip install h5py")
        return

    save_path = Path(args.save_dir)
    data_path = Path(args.path)

    annotator = SAM2VideoAnnotator(
        out_dir=save_path,
        dataset_name=DATASET_NAME,
        categories=CATEGORIES,
        save_visualization=args.save_viz,
        video_environment="ultrasound",
        video_split="train",
    )

    logger.info(f"Output format: {args.format}")

    hdf5_file = data_path / "dataset.hdf5"
    if not hdf5_file.exists():
        # Try alternative paths
        for candidate in [
            data_path / "thyroidultrasoundcineclip" / "dataset.hdf5",
            data_path,
        ]:
            if candidate.is_file() and str(candidate).endswith(".hdf5"):
                hdf5_file = candidate
                break
            elif (candidate / "dataset.hdf5").exists():
                hdf5_file = candidate / "dataset.hdf5"
                break

    if not hdf5_file.exists():
        logger.error(f"Cannot find dataset.hdf5 in {data_path}")
        return

    logger.info(f"Reading HDF5 file: {hdf5_file}")

    with h5py.File(str(hdf5_file), "r") as hdf:
        images = hdf["image"][:]
        masks = hdf["mask"][:]
        annot_ids = hdf["annot_id"][:]
        frame_nums = hdf["frame_num"][:]

        logger.info(f"Loaded {len(images)} frames from HDF5")
        logger.info(f"Image shape: {images.shape}, Mask shape: {masks.shape}")

        # Group frames by annotation ID (each annot_id is a video/clip)
        clips = defaultdict(list)
        for idx in range(len(images)):
            # Handle bytes or string annot_id with trailing underscore
            raw_annot_id = annot_ids[idx]
            if isinstance(raw_annot_id, bytes):
                raw_annot_id = raw_annot_id.decode("utf-8")
            # Remove trailing underscore and convert to int
            annot_id = int(str(raw_annot_id).rstrip("_"))
            
            raw_frame_num = frame_nums[idx]
            if isinstance(raw_frame_num, bytes):
                raw_frame_num = raw_frame_num.decode("utf-8")
            frame_num = int(str(raw_frame_num).rstrip("_"))
            clips[annot_id].append({
                "idx": idx,
                "frame_num": frame_num,
                "image": images[idx],
                "mask": masks[idx],
            })

        logger.info(f"Found {len(clips)} clips")

        if args.max_clips:
            clip_ids = list(clips.keys())[:args.max_clips]
            clips = {k: clips[k] for k in clip_ids}

        processed = 0
        total_frames = 0

        for clip_id, frames in clips.items():
            # Sort by frame number
            frames = sorted(frames, key=lambda x: x["frame_num"])

            video_name = f"{clip_id:04d}"

            if args.format == "sa-v":
                # Process as video
                processed_frames = []
                processed_masks = []

                for frame_data in frames:
                    img = frame_data["image"]
                    mask = frame_data["mask"]

                    # Handle grayscale images
                    if img.ndim == 2:
                        pass  # Keep as grayscale
                    elif img.ndim == 3 and img.shape[2] == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    # Binarize and fill mask
                    mask = (mask != 0).astype(np.uint8)
                    mask = fill_mask(mask)

                    if not np.any(mask > 0):
                        continue

                    processed_frames.append(img)
                    processed_masks.append(mask)

                if processed_frames:
                    frames_arr = np.stack(processed_frames, axis=0)
                    masks_arr = np.stack(processed_masks, axis=0)

                    annotator.add_video(
                        frames_arr,
                        masks_arr,
                        video_name=video_name,
                        metadata={
                            "clip_id": clip_id,
                            "num_frames": len(processed_frames),
                            "dataset": DATASET_NAME,
                        },
                    )
                    total_frames += len(processed_frames)
                    processed += 1

            else:
                # Process as individual images (SA-1B)
                for frame_idx, frame_data in enumerate(frames):
                    img = frame_data["image"]
                    mask = frame_data["mask"]

                    if img.ndim == 2:
                        pass
                    elif img.ndim == 3 and img.shape[2] == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    mask = (mask != 0).astype(np.uint8)
                    mask = fill_mask(mask)

                    if not np.any(mask > 0):
                        continue

                    frame_name = f"{video_name}_{frame_idx:04d}"
                    annotator.add_2d(
                        img,
                        mask,
                        video_name=frame_name,
                        format="sa-1b",
                        metadata={
                            "clip_id": clip_id,
                            "frame_idx": frame_idx,
                            "frame_num": frame_data["frame_num"],
                            "dataset": DATASET_NAME,
                        },
                    )
                    total_frames += 1

                processed += 1

            if processed % 100 == 0:
                logger.info(f"Processed {processed}/{len(clips)} clips ({total_frames} frames)")

        logger.info(f"Total: {processed} clips, {total_frames} frames")

    stats = annotator.finalize()
    logger.info(f"Dataset stats: {stats}")

    if args.zip:
        shutil.make_archive(str(save_path), "zip", save_path)
        if args.delete:
            shutil.rmtree(save_path)

    logger.info("Done!")


if __name__ == "__main__":
    main()
