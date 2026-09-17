"""Brachial Plexus ultrasound dataset preprocessing.

Dataset: Regional ultrasound video dataset for brachial plexus nerve segmentation
Source: https://github.com/Regional-US/brachial_plexus

Data format:
- data/{vendor}/videos/{subject_id}.mp4: Ultrasound video (360x512, 30fps)
- data/{vendor}/ac_masks/{subject_id}/{subject_id}_{frame:03d}.jpg: Nerve plexus mask
- data/{vendor}/needle/needle_masks/{subject_id}/{subject_id}_{frame:03d}.jpg: Needle mask (optional)

Vendors: Butterfly, eSaote, Sonosite

Output: SAM2-compatible format (SA-V style for video, SA-1B for images)
"""

import argparse
import logging
import shutil
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

DATASET_NAME = "Brachial-Plexus"

# Category definitions
CATEGORIES = [
    {"supercategory": "nerve", "id": 1, "name": "nerve_plexus"},
    {"supercategory": "needle", "id": 2, "name": "needle"},
]

# Supported vendors
VENDORS = ["Butterfly", "eSaote", "Sonosite"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=f"Convert {DATASET_NAME} to SAM2 format"
    )
    parser.add_argument(
        "--path",
        type=str,
        help="Path to the dataset (zip file or extracted folder)",
        default=f"/mnt/data/Dataset/UltraSound/Raw/{DATASET_NAME}.zip",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        help="Output directory for SAM2 format dataset",
        default=f"/mnt/data/Dataset/SaUS/{DATASET_NAME}",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["sa-v", "sa-1b"],
        default="sa-v",
        help="Output format: sa-v (video style) or sa-1b (image style)",
    )
    parser.add_argument("--min-mask-pixels", type=int, default=10, help="Minimum mask area")
    parser.add_argument(
        "--save-viz",
        action="store_true",
        help="Save visualization images with overlaid masks",
    )
    parser.add_argument(
        "--vendors",
        type=str,
        nargs="+",
        default=None,
        choices=VENDORS,
        help="Specific vendors to process (default: all)",
    )
    parser.add_argument(
        "--include-needle",
        action="store_true",
        default=True,
        help="Include needle segmentation when available",
    )
    parser.add_argument(
        "--skip-empty-frames",
        action="store_true",
        default=False,
        help="Skip frames without any mask (only for sa-1b format)",
    )
    parser.add_argument(
        "--mask-threshold",
        type=int,
        default=100,
        help="Threshold for binarizing mask images (default: 100)",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Maximum number of videos to process (for debugging)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="FPS for video metadata (default: 30)",
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        help="Compress output dataset to zip file",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete uncompressed output after zipping",
    )
    return parser.parse_args()


def rgb2gray(img: np.ndarray) -> np.ndarray:
    """Convert RGB image to grayscale."""
    if img.ndim == 2:
        return img
    if img.shape[2] == 1:
        return img[:, :, 0]
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def load_mask_image(
    mask_path: Path,
    threshold: int = 100,
    label_value: int = 1,
) -> Optional[np.ndarray]:
    """Load and binarize a mask image.
    
    Args:
        mask_path: Path to mask image
        threshold: Threshold for binarization
        label_value: Value to assign to foreground pixels
    
    Returns:
        Binary mask array or None if file doesn't exist
    """
    if not mask_path.exists():
        return None
    
    mask_img = cv2.imread(str(mask_path))
    if mask_img is None:
        return None
    
    mask_gray = rgb2gray(mask_img)
    binary_mask = np.zeros_like(mask_gray, dtype=np.uint8)
    binary_mask[mask_gray >= threshold] = label_value
    
    return binary_mask


def collect_video_data(data_dir: Path, vendors: Optional[List[str]] = None) -> List[Dict]:
    """Collect all video data from the dataset.
    
    Args:
        data_dir: Root data directory
        vendors: List of vendors to include (None = all)
    
    Returns:
        List of video info dicts
    """
    if vendors is None:
        vendors = VENDORS
    
    videos = []
    
    for vendor in vendors:
        vendor_dir = data_dir / "data" / vendor
        if not vendor_dir.exists():
            continue
        
        videos_dir = vendor_dir / "videos"
        ac_masks_dir = vendor_dir / "ac_masks"
        needle_masks_dir = vendor_dir / "needle" / "needle_masks"
        
        if not videos_dir.exists():
            continue
        
        for video_file in sorted(videos_dir.glob("*.mp4")):
            subject_id = video_file.stem
            
            video_info = {
                "video_path": video_file,
                "subject_id": subject_id,
                "vendor": vendor,
                "ac_masks_dir": ac_masks_dir / subject_id,
                "needle_masks_dir": needle_masks_dir / subject_id,
                "has_needle_masks": (needle_masks_dir / subject_id).exists(),
            }
            videos.append(video_info)
    
    return videos


def process_video_as_video(
    video_info: Dict,
    annotator: SAM2VideoAnnotator,
    include_needle: bool = True,
    mask_threshold: int = 100,
) -> Tuple[int, int]:
    """Process a video and add as SA-V format.
    
    Args:
        video_info: Video info dict
        annotator: SAM2VideoAnnotator instance
        include_needle: Whether to include needle masks
        mask_threshold: Threshold for mask binarization
    
    Returns:
        Tuple of (frames_processed, annotations_count)
    """
    video_path = video_info["video_path"]
    subject_id = video_info["subject_id"]
    vendor = video_info["vendor"]
    ac_masks_dir = video_info["ac_masks_dir"]
    needle_masks_dir = video_info["needle_masks_dir"]
    has_needle = video_info["has_needle_masks"] and include_needle
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0, 0
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frames = []
    masks = []
    annotations_count = 0
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        
        # Load masks
        h, w = frame.shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Load AC mask (nerve plexus) - label 1
        ac_mask_path = ac_masks_dir / f"{subject_id}_{frame_idx:03d}.jpg"
        ac_mask = load_mask_image(ac_mask_path, mask_threshold, label_value=1)
        if ac_mask is not None:
            combined_mask[ac_mask > 0] = 1
            if np.any(ac_mask > 0):
                annotations_count += 1
        
        # Load needle mask - label 2
        if has_needle:
            needle_mask_path = needle_masks_dir / f"{subject_id}_{frame_idx:03d}.jpg"
            needle_mask = load_mask_image(needle_mask_path, mask_threshold, label_value=2)
            if needle_mask is not None:
                combined_mask[needle_mask > 0] = 2  # Needle overwrites nerve
                if np.any(needle_mask > 0):
                    annotations_count += 1
        
        masks.append(combined_mask)
        frame_idx += 1
    
    cap.release()
    
    if not frames:
        return 0, 0
    
    # Stack into arrays
    video_array = np.stack(frames, axis=0)
    mask_array = np.stack(masks, axis=0)
    
    # Generate video name
    video_name = f"{vendor}_{subject_id}"
    
    # Add as video
    annotator.add_video(
        video_array,
        mask_array,
        video_name=video_name,
        video_duration=frame_count / fps if fps > 0 else None,
        metadata={
            "subject_id": subject_id,
            "vendor": vendor,
            "fps": fps,
            "frame_count": frame_count,
            "has_needle_masks": has_needle,
            "dataset": DATASET_NAME,
        },
    )
    
    return len(frames), annotations_count


def process_video_as_images(
    video_info: Dict,
    annotator: SAM2VideoAnnotator,
    include_needle: bool = True,
    mask_threshold: int = 100,
    skip_empty: bool = False,
) -> Tuple[int, int]:
    """Process a video and add frames as SA-1B images.
    
    Args:
        video_info: Video info dict
        annotator: SAM2VideoAnnotator instance
        include_needle: Whether to include needle masks
        mask_threshold: Threshold for mask binarization
        skip_empty: Whether to skip frames without masks
    
    Returns:
        Tuple of (frames_processed, annotations_count)
    """
    video_path = video_info["video_path"]
    subject_id = video_info["subject_id"]
    vendor = video_info["vendor"]
    ac_masks_dir = video_info["ac_masks_dir"]
    needle_masks_dir = video_info["needle_masks_dir"]
    has_needle = video_info["has_needle_masks"] and include_needle
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0, 0
    
    frames_processed = 0
    annotations_count = 0
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Load masks
        h, w = frame.shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Load AC mask (nerve plexus) - label 1
        ac_mask_path = ac_masks_dir / f"{subject_id}_{frame_idx:03d}.jpg"
        ac_mask = load_mask_image(ac_mask_path, mask_threshold, label_value=1)
        if ac_mask is not None:
            combined_mask[ac_mask > 0] = 1
        
        # Load needle mask - label 2
        if has_needle:
            needle_mask_path = needle_masks_dir / f"{subject_id}_{frame_idx:03d}.jpg"
            needle_mask = load_mask_image(needle_mask_path, mask_threshold, label_value=2)
            if needle_mask is not None:
                combined_mask[needle_mask > 0] = 2
        
        # Skip empty frames if requested
        if skip_empty and not np.any(combined_mask > 0):
            frame_idx += 1
            continue
        
        # Generate image name
        image_name = f"{vendor}_{subject_id}_{frame_idx:04d}"
        
        # Add as image
        annotator.add_2d(
            frame_rgb,
            combined_mask,
            video_name=image_name,
            format="sa-1b",
            metadata={
                "subject_id": subject_id,
                "vendor": vendor,
                "frame_idx": frame_idx,
                "has_needle_mask": has_needle and (needle_masks_dir / f"{subject_id}_{frame_idx:03d}.jpg").exists(),
                "dataset": DATASET_NAME,
            },
        )
        
        frames_processed += 1
        if np.any(combined_mask > 0):
            annotations_count += np.unique(combined_mask[combined_mask > 0]).size
        
        frame_idx += 1
    
    cap.release()
    
    return frames_processed, annotations_count


def main():
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    
    save_path = Path(args.save_dir)
    data_path = Path(args.path)
    
    # Determine if we need to extract from zip
    use_temp_dir = data_path.suffix.lower() == ".zip"
    
    # Create annotator
    annotator = SAM2VideoAnnotator(
        out_dir=save_path,
        dataset_name=DATASET_NAME,
        categories=CATEGORIES,
        fps=args.fps,
        save_visualization=args.save_viz,
        video_environment="ultrasound",
        video_split="train",
        min_mask_pixels=args.min_mask_pixels,
    )
    
    logger.info(f"Output format: {args.format}")
    logger.info(f"Vendors: {args.vendors or 'all'}")
    logger.info(f"Include needle: {args.include_needle}")
    logger.info(f"Mask threshold: {args.mask_threshold}")
    
    # Context manager for temp directory
    if use_temp_dir:
        temp_context = tempfile.TemporaryDirectory()
    else:
        from contextlib import nullcontext
        temp_context = nullcontext(str(data_path))
    
    with temp_context as temp_dir:
        if use_temp_dir:
            logger.info(f"Extracting {data_path} to {temp_dir}")
            shutil.unpack_archive(str(data_path), temp_dir)
            logger.info("Extraction complete")
            data_dir = Path(temp_dir)
            
            # Handle nested folder structure (brachial_plexus/data/...)
            if (data_dir / "brachial_plexus").exists():
                data_dir = data_dir / "brachial_plexus"
        else:
            data_dir = data_path
        
        # Collect all videos
        videos = collect_video_data(data_dir, args.vendors)
        logger.info(f"Found {len(videos)} videos")
        
        if not videos:
            logger.error("No videos found!")
            return
        
        # Limit videos if requested
        if args.max_videos:
            videos = videos[:args.max_videos]
            logger.info(f"Processing only {len(videos)} videos")
        
        # Process videos
        total_frames = 0
        total_annotations = 0
        
        for idx, video_info in enumerate(videos):
            if args.format == "sa-v":
                frames, anns = process_video_as_video(
                    video_info,
                    annotator,
                    include_needle=args.include_needle,
                    mask_threshold=args.mask_threshold,
                )
            else:
                frames, anns = process_video_as_images(
                    video_info,
                    annotator,
                    include_needle=args.include_needle,
                    mask_threshold=args.mask_threshold,
                    skip_empty=args.skip_empty_frames,
                )
            
            total_frames += frames
            total_annotations += anns
            
            if (idx + 1) % 10 == 0:
                logger.info(
                    f"Processed {idx + 1}/{len(videos)} videos "
                    f"({total_frames} frames, {total_annotations} annotations)"
                )
        
        logger.info(f"Total: {total_frames} frames, {total_annotations} annotations")
    
    # Finalize dataset
    stats = annotator.finalize()
    logger.info(f"Dataset stats: {stats}")
    
    # Optional: zip output
    if args.zip:
        zip_path = save_path.with_suffix(".zip")
        logger.info(f"Compressing to {zip_path}")
        shutil.make_archive(str(save_path), "zip", save_path)
        
        if args.delete:
            logger.info(f"Deleting uncompressed folder: {save_path}")
            shutil.rmtree(save_path)
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
