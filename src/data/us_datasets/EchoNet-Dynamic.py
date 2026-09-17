"""EchoNet-Dynamic dataset preprocessing.

Dataset: Echocardiography video dataset with left ventricle segmentation
Source: https://github.com/echonet/dynamic

Structure:
- Videos/{filename}.avi: Echo video files
- VolumeTracings.csv: Polygon coordinates for LV segmentation

Annotations are provided as polygon contours in CSV that need to be
converted to binary masks.

Output: SAM2-compatible format (SA-V for video, SA-1B for frames)
"""

import argparse
import logging
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ..utils.sam2_annotator import SAM2VideoAnnotator

DATASET_NAME = "EchoNet-Dynamic"

CATEGORIES = [
    {"supercategory": "cardiac", "id": 1, "name": "left_ventricle"},
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
        help="Output format (sa-1b recommended - only annotated frames)",
    )
    parser.add_argument("--save-viz", action="store_true")
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--zip", action="store_true")
    parser.add_argument("--delete", action="store_true")
    return parser.parse_args()


class VideoReader:
    """Simple video reader using OpenCV."""

    def __init__(self, path: str):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self._len = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self._len

    def __getitem__(self, idx: int) -> Optional[np.ndarray]:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None

    def __del__(self):
        if hasattr(self, "cap"):
            self.cap.release()


def create_segmentation_masks(df: pd.DataFrame, data_path: Path) -> Dict:
    """Create binary masks from polygon annotations in CSV."""
    segmentation_masks = {}

    grouped = df.groupby("FileName")

    for video_name, group_df in grouped:
        video_path = data_path / "Videos" / video_name
        if not video_path.exists():
            continue

        try:
            vid = VideoReader(str(video_path))
            width, height = vid.width, vid.height
        except Exception:
            continue

        video_masks = {}
        frame_grouped = group_df.groupby("Frame")

        for frame_id, frame_df in frame_grouped:
            # Extract polygon points
            x1 = frame_df["X1"].tolist()[1:]  # Skip first point (reference)
            y1 = frame_df["Y1"].tolist()[1:]
            x2 = frame_df["X2"].tolist()[1:]
            y2 = frame_df["Y2"].tolist()[1:]

            if not x1:
                continue

            # Create closed polygon from two contours
            points_a = [(x1[i], y1[i]) for i in range(len(x1))]
            points_b = [(x2[i], y2[i]) for i in range(len(x1) - 1, -1, -1)]
            points = points_a + points_b

            if points[0] != points[-1]:
                points.append(points[0])

            # Create binary mask
            poly_points = np.array([points], dtype=np.int32)
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask, poly_points, 1)

            video_masks[int(frame_id)] = mask

        if video_masks:
            segmentation_masks[video_name] = video_masks

    return segmentation_masks


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
            # Handle nested folder
            if (data_dir / "EchoNet-Dynamic").exists():
                data_dir = data_dir / "EchoNet-Dynamic"
        else:
            data_dir = data_path

        # Load annotations
        csv_path = data_dir / "VolumeTracings.csv"
        if not csv_path.exists():
            logger.error(f"VolumeTracings.csv not found at {csv_path}")
            return

        logger.info("Loading annotations...")
        df = pd.read_csv(csv_path)
        segmentation_masks = create_segmentation_masks(df, data_dir)
        logger.info(f"Found {len(segmentation_masks)} videos with annotations")

        video_names = list(segmentation_masks.keys())
        if args.max_videos:
            video_names = video_names[:args.max_videos]

        processed = 0
        total_frames = 0

        for idx, video_name in enumerate(video_names):
            video_path = data_dir / "Videos" / video_name
            if not video_path.exists():
                continue

            try:
                vid = VideoReader(str(video_path))
            except Exception as e:
                logger.warning(f"Error reading {video_name}: {e}")
                continue

            masks = segmentation_masks[video_name]
            video_id = video_name.split(".")[0]

            if args.format == "sa-1b":
                # Export only annotated frames
                for frame_id, mask in masks.items():
                    img = vid[frame_id]
                    if img is None:
                        continue

                    image_name = f"{video_id}_{frame_id:04d}"
                    annotator.add_2d(
                        img,
                        mask,
                        video_name=image_name,
                        format="sa-1b",
                        metadata={
                            "video_name": video_name,
                            "video_id": video_id,
                            "frame_id": frame_id,
                            "dataset": DATASET_NAME,
                        },
                    )
                    total_frames += 1
            else:
                # Export as video (only frames with annotations)
                frames = []
                frame_masks = []
                frame_ids = sorted(masks.keys())

                for frame_id in frame_ids:
                    img = vid[frame_id]
                    if img is None:
                        continue
                    frames.append(img)
                    frame_masks.append(masks[frame_id])

                if frames:
                    frames_arr = np.stack(frames, axis=0)
                    masks_arr = np.stack(frame_masks, axis=0)

                    video_name_clean = f"{video_id}"
                    annotator.add_video(
                        frames_arr,
                        masks_arr,
                        video_name=video_name_clean,
                        metadata={
                            "video_name": video_name,
                            "video_id": video_id,
                            "frame_ids": frame_ids,
                            "dataset": DATASET_NAME,
                        },
                    )
                    total_frames += len(frames)

            processed += 1

            if (idx + 1) % 100 == 0:
                logger.info(f"Processed {idx + 1}/{len(video_names)} videos ({total_frames} frames)")

        logger.info(f"Total: {processed} videos, {total_frames} frames")

    stats = annotator.finalize()
    logger.info(f"Dataset stats: {stats}")

    if args.zip:
        shutil.make_archive(str(save_path), "zip", save_path)
        if args.delete:
            shutil.rmtree(save_path)

    logger.info("Done!")


if __name__ == "__main__":
    main()
