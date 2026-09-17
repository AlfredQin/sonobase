"""RVENet (Right Ventricle Echocardiography Network) dataset preprocessing.

Dataset: Large-scale echocardiography dataset for right ventricle analysis
Source: https://rvenet.github.io/

Data format:
- train/{video_id}.dcm: DICOM videos (echocardiography)
- Annotations-MedSAM2/{video_id}/{frame_id:04d}.png: Segmentation masks
- metadata.csv: Video metadata with train/validation split

Mask categories (values 1-5):
1. Right ventricle free wall
2. Right ventricle cavity
3. Interventricular septum
4. Tricuspid valve
5. Right atrium

Note: Masks are at half resolution compared to DICOM frames.

Output: SAM2-compatible format (SA-V for videos, SA-1B for individual frames)
"""

import argparse
import csv
import logging
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ..utils.sam2_annotator import SAM2VideoAnnotator

DATASET_NAME = "RVENet"

# Categories based on typical RV segmentation
CATEGORIES = [
    {"supercategory": "cardiac", "id": 1, "name": "rv_free_wall"},
    {"supercategory": "cardiac", "id": 2, "name": "rv_cavity"},
    {"supercategory": "cardiac", "id": 3, "name": "interventricular_septum"},
    {"supercategory": "cardiac", "id": 4, "name": "tricuspid_valve"},
    {"supercategory": "cardiac", "id": 5, "name": "right_atrium"},
]


def parse_args():
    parser = argparse.ArgumentParser(description=f"Convert {DATASET_NAME} to SAM2 format")
    parser.add_argument(
        "--path",
        type=str,
        default="/mnt/data/Dataset/UltraSound/RVENet",
        help="Path to extracted dataset (containing train/ and Annotations-MedSAM2/)",
    )
    parser.add_argument(
        "--raw-path",
        type=str,
        default="/mnt/data/Dataset/UltraSound/Raw/RVENet",
        help="Path to raw zip files (for metadata.csv)",
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
        help="Output format (sa-v treats each echo as video)",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "validation", "all"],
        default="all",
        help="Which split to process",
    )
    parser.add_argument("--save-viz", action="store_true")
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--zip", action="store_true")
    parser.add_argument("--delete", action="store_true")
    return parser.parse_args()


def load_metadata(metadata_path: Path) -> Dict[str, Dict]:
    """Load metadata CSV and return mapping from FileName to metadata."""
    metadata = {}
    with open(metadata_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metadata[row["FileName"]] = row
    return metadata


def read_dicom_video(dcm_path: Path) -> Optional[np.ndarray]:
    """Read DICOM file and return video frames."""
    try:
        import pydicom
        dcm = pydicom.dcmread(str(dcm_path))
        frames = dcm.pixel_array  # Shape: (num_frames, H, W, 3) or (num_frames, H, W)
        return frames
    except Exception as e:
        logging.warning(f"Error reading DICOM {dcm_path}: {e}")
        return None


def load_masks(mask_dir: Path, num_frames: int, target_size: Tuple[int, int]) -> Optional[np.ndarray]:
    """Load mask PNG files and resize to target size.
    
    Args:
        mask_dir: Directory containing mask PNG files
        num_frames: Expected number of frames
        target_size: (height, width) to resize masks to
    
    Returns:
        Mask array of shape (num_frames, H, W)
    """
    if not mask_dir.exists():
        return None
    
    masks = []
    for frame_idx in range(num_frames):
        mask_path = mask_dir / f"{frame_idx:04d}.png"
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                # Resize mask to match video frame size
                mask = cv2.resize(mask, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)
                masks.append(mask)
            else:
                masks.append(np.zeros(target_size, dtype=np.uint8))
        else:
            masks.append(np.zeros(target_size, dtype=np.uint8))
    
    return np.stack(masks, axis=0)


def collect_videos(
    data_dir: Path,
    annotations_dir: Path,
    metadata: Dict[str, Dict],
    split: str = "all",
) -> List[Dict]:
    """Collect video-annotation pairs."""
    videos = []
    
    # Find all annotated videos
    if not annotations_dir.exists():
        return videos
    
    for ann_folder in sorted(annotations_dir.iterdir()):
        if not ann_folder.is_dir():
            continue
        
        video_id = ann_folder.name
        
        # Check metadata for split
        if video_id in metadata:
            video_split = metadata[video_id].get("Split", "train")
            if split != "all" and video_split != split:
                continue
        else:
            # No metadata, assume train
            video_split = "train"
        
        # Find corresponding DICOM file
        dcm_path = data_dir / "train" / f"{video_id}.dcm"
        if not dcm_path.exists():
            dcm_path = data_dir / "validation" / f"{video_id}.dcm"
        if not dcm_path.exists():
            dcm_path = data_dir / f"{video_id}.dcm"
        
        if dcm_path.exists():
            videos.append({
                "video_id": video_id,
                "dcm_path": dcm_path,
                "mask_dir": ann_folder,
                "split": video_split,
                "metadata": metadata.get(video_id, {}),
            })
    
    return videos


def main():
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    # Check for pydicom
    try:
        import pydicom
    except ImportError:
        logger.error("pydicom is required. Install with: pip install pydicom")
        return

    save_path = Path(args.save_dir)
    data_path = Path(args.path)
    raw_path = Path(args.raw_path)

    # Load metadata
    metadata_path = raw_path / "metadata.csv"
    if not metadata_path.exists():
        metadata_path = data_path / "metadata.csv"
    
    if metadata_path.exists():
        metadata = load_metadata(metadata_path)
        logger.info(f"Loaded metadata for {len(metadata)} videos")
    else:
        metadata = {}
        logger.warning("No metadata.csv found, proceeding without metadata")

    # Find annotations directory
    annotations_dir = data_path / "Annotations-MedSAM2"
    if not annotations_dir.exists():
        logger.error(f"Annotations directory not found: {annotations_dir}")
        return

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

    videos = collect_videos(data_path, annotations_dir, metadata, args.split)
    logger.info(f"Found {len(videos)} videos with annotations")

    if args.max_videos:
        videos = videos[:args.max_videos]

    processed = 0
    total_frames = 0

    for idx, video in enumerate(videos):
        try:
            # Read DICOM video
            frames = read_dicom_video(video["dcm_path"])
            if frames is None:
                continue

            num_frames = frames.shape[0]
            h, w = frames.shape[1], frames.shape[2]

            # Load masks and resize to video frame size
            masks = load_masks(video["mask_dir"], num_frames, (h, w))
            if masks is None:
                continue

            # Convert frames to RGB if needed
            if frames.ndim == 4 and frames.shape[3] == 3:
                # Already RGB/BGR
                frames_rgb = frames[..., ::-1]  # BGR to RGB
            else:
                # Grayscale, convert to RGB
                frames_rgb = np.stack([frames] * 3, axis=-1)

            video_name = video["video_id"]

            if args.format == "sa-v":
                # Process as video
                # Filter frames with masks
                valid_indices = []
                for i in range(num_frames):
                    if np.any(masks[i] > 0):
                        valid_indices.append(i)

                if not valid_indices:
                    continue

                video_frames = frames_rgb[valid_indices].astype(np.uint8)
                video_masks = masks[valid_indices]

                annotator.add_video(
                    video_frames,
                    video_masks,
                    video_name=video_name,
                    metadata={
                        "video_id": video["video_id"],
                        "split": video["split"],
                        "patient_group": video["metadata"].get("PatientGroup", ""),
                        "view_type": video["metadata"].get("VideoViewType", ""),
                        "fps": video["metadata"].get("FPS", ""),
                        "dataset": DATASET_NAME,
                    },
                )
                total_frames += len(valid_indices)
                processed += 1

            else:
                # Process as individual frames (SA-1B)
                frame_count = 0
                for frame_idx in range(num_frames):
                    mask = masks[frame_idx]
                    if not np.any(mask > 0):
                        continue

                    frame = frames_rgb[frame_idx].astype(np.uint8)
                    frame_name = f"{video_name}_{frame_idx:04d}"

                    annotator.add_2d(
                        frame,
                        mask,
                        video_name=frame_name,
                        format="sa-1b",
                        metadata={
                            "video_id": video["video_id"],
                            "frame_idx": frame_idx,
                            "split": video["split"],
                            "patient_group": video["metadata"].get("PatientGroup", ""),
                            "view_type": video["metadata"].get("VideoViewType", ""),
                            "dataset": DATASET_NAME,
                        },
                    )
                    frame_count += 1
                    total_frames += 1

                if frame_count > 0:
                    processed += 1

            if (idx + 1) % 50 == 0:
                logger.info(f"Processed {idx + 1}/{len(videos)} videos ({total_frames} frames)")

        except Exception as e:
            logger.warning(f"Error processing {video['video_id']}: {e}")
            continue

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
