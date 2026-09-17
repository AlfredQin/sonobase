"""MouseBrainTumor (GL261 Glioblastoma) dataset preprocessing.

Dataset: High-resolution ultrasound data for AI-based segmentation in mouse
         brain tumor (GL261 glioblastoma).
Source: https://doi.org/10.6084/m9.figshare.27237894
Paper: https://doi.org/10.1038/s41597-025-05619-z

The archive is a zip-of-zips: the outer zip contains 38 inner zips, each a
recording session from one mouse.  Each inner zip holds:
    rec__YYYYMMDD_HHMMSS/
        Images/   frame{N}_img.png  (or "Extracted Frames{N}img.png")
        Masks/    frame{N}_img_mask.png  (suffix may include annotator name)
        ReadMe.xlsx, *.mp4

Masks are soft (averaged from 5 annotators) and thresholded to binary here.
Each recording is treated as a video; connected-component labeling splits
multi-blob masks into separate instances.

Categories: tumor
Output: SAM2-compatible format (SA-V by default, SA-1B optional)
"""

import argparse
import io
import logging
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ..utils.sam2_annotator import SAM2VideoAnnotator

DATASET_NAME = "MouseBrainTumor"

CATEGORIES = [
    {"supercategory": "tumor", "id": 1, "name": "tumor"},
]

MASK_THRESHOLD = 0


def parse_args():
    parser = argparse.ArgumentParser(description=f"Convert {DATASET_NAME} to SAM2 format")
    parser.add_argument(
        "--path",
        type=str,
        default=f"/mnt/data/Dataset/UltraSound/Raw/{DATASET_NAME}.zip",
        help="Path to dataset zip",
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
        help="Output format: sa-v (video) or sa-1b (image)",
    )
    parser.add_argument("--save-viz", action="store_true")
    parser.add_argument("--max-recordings", type=int, default=None)
    parser.add_argument("--min-mask-pixels", type=int, default=10)
    parser.add_argument(
        "--skip-empty",
        action="store_true",
        default=True,
        help="Skip frames with empty masks",
    )
    parser.add_argument("--zip", action="store_true")
    parser.add_argument("--delete", action="store_true")
    return parser.parse_args()


def _mask_to_gray(mask: np.ndarray) -> np.ndarray:
    """Convert a potentially multi-channel mask to single-channel grayscale."""
    if mask.ndim == 3:
        return mask[:, :, 0]
    return mask


def _find_mask_for_image(
    image_name: str,
    mask_names: List[str],
) -> Optional[str]:
    """Find the corresponding mask path for an image.

    Image stems look like ``frame123_img`` or ``Extracted Frames123img``.
    Mask filenames append ``_mask.png`` or ``_{annotator}_mask.png``.
    """
    stem = image_name.rsplit(".png", 1)[0]
    for mname in mask_names:
        mbase = mname.rsplit("/", 1)[-1]
        if mbase.startswith(stem) and mbase.endswith("_mask.png"):
            return mname
    return None


def _natural_sort_key(name: str):
    """Sort key that orders embedded numbers numerically."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def collect_recordings(zip_path: Path) -> List[Dict]:
    """List the inner recording zips inside the outer archive."""
    recordings = []
    with zipfile.ZipFile(zip_path) as outer:
        for name in sorted(outer.namelist()):
            if name.endswith(".zip"):
                rec_id = name.replace(".zip", "")
                recordings.append({"name": name, "rec_id": rec_id})
    return recordings


def load_recording(
    outer_zip: zipfile.ZipFile,
    rec_entry: str,
    skip_empty: bool = True,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    """Load all image-mask pairs from one inner recording zip.

    Returns (frames, masks, rec_id) where frames is (N, H, W, 3) uint8 RGB
    and masks is (N, H, W) uint8 binary.  Returns (None, None, rec_id) when
    no valid frames are found.
    """
    inner_data = outer_zip.read(rec_entry)
    inner_zip = zipfile.ZipFile(io.BytesIO(inner_data))
    rec_id = rec_entry.replace(".zip", "")

    all_names = inner_zip.namelist()
    image_names = sorted(
        [n for n in all_names if "/Images/" in n and n.endswith(".png")],
        key=_natural_sort_key,
    )
    mask_names = [n for n in all_names if "/Masks/" in n and n.endswith(".png")]

    frames, masks = [], []
    for img_name in image_names:
        img_basename = img_name.rsplit("/", 1)[-1]
        mask_name = _find_mask_for_image(img_basename, mask_names)
        if mask_name is None:
            continue

        img_buf = np.frombuffer(inner_zip.read(img_name), dtype=np.uint8)
        img = cv2.imdecode(img_buf, cv2.IMREAD_COLOR)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask_buf = np.frombuffer(inner_zip.read(mask_name), dtype=np.uint8)
        mask = cv2.imdecode(mask_buf, cv2.IMREAD_UNCHANGED)
        if mask is None:
            continue
        mask_gray = _mask_to_gray(mask)
        binary = (mask_gray > MASK_THRESHOLD).astype(np.uint8)

        if skip_empty and not np.any(binary):
            continue

        frames.append(img_rgb)
        masks.append(binary)

    inner_zip.close()

    if not frames:
        return None, None, rec_id

    return np.stack(frames), np.stack(masks), rec_id


def process_recording_as_video(
    annotator: SAM2VideoAnnotator,
    frames: np.ndarray,
    masks: np.ndarray,
    rec_id: str,
) -> int:
    """Process one recording as a video.

    Each mouse has a single GL261 tumor injection, so all foreground pixels
    across frames belong to the same object (instance 1).  Frames are sparsely
    sampled from the recording video, making 3-D connected-component labeling
    unreliable (non-consecutive frames break spatial continuity).
    """
    instance_masks = (masks > 0).astype(np.uint16)

    if not np.any(instance_masks):
        return 0

    annotator.add_video(
        frames,
        instance_masks,
        video_name=rec_id,
        metadata={"recording_id": rec_id, "dataset": DATASET_NAME},
        instance_to_category={1: 1},
        default_category_id=1,
    )
    return frames.shape[0]


def process_recording_as_images(
    annotator: SAM2VideoAnnotator,
    frames: np.ndarray,
    masks: np.ndarray,
    rec_id: str,
    output_format: str = "sa-1b",
) -> int:
    """Process one recording frame-by-frame (SA-1B image format)."""
    count = 0
    for i in range(frames.shape[0]):
        binary = masks[i]
        if not np.any(binary):
            continue

        instance_mask = (binary > 0).astype(np.uint16)

        image_name = f"{rec_id}_{i:04d}"
        annotator.add_2d(
            frames[i],
            instance_mask,
            video_name=image_name,
            format=output_format,
            instance_to_category={1: 1},
            metadata={
                "recording_id": rec_id,
                "frame_idx": i,
                "dataset": DATASET_NAME,
            },
        )
        count += 1
    return count


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    save_path = Path(args.save_dir)
    data_path = Path(args.path)

    if not data_path.exists():
        logger.error(f"Dataset not found: {data_path}")
        return

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

    recordings = collect_recordings(data_path)
    logger.info(f"Found {len(recordings)} recordings in {data_path.name}")

    if args.max_recordings:
        recordings = recordings[: args.max_recordings]

    total_frames = 0

    with zipfile.ZipFile(data_path) as outer_zip:
        for idx, rec in enumerate(recordings):
            frames, masks, rec_id = load_recording(
                outer_zip, rec["name"], skip_empty=args.skip_empty
            )

            if frames is None:
                logger.info(
                    f"[{idx + 1}/{len(recordings)}] {rec_id}: "
                    f"skipped (no valid frames)"
                )
                continue

            if args.format == "sa-v":
                count = process_recording_as_video(
                    annotator, frames, masks, rec_id
                )
            else:
                count = process_recording_as_images(
                    annotator, frames, masks, rec_id, output_format=args.format
                )

            total_frames += count
            logger.info(
                f"[{idx + 1}/{len(recordings)}] {rec_id}: "
                f"{count} frames, total: {total_frames}"
            )

    logger.info(f"Total frames processed: {total_frames}")

    stats = annotator.finalize()
    logger.info(f"Dataset stats: {stats}")

    if args.zip:
        zip_path = save_path.with_suffix(".zip")
        logger.info(f"Compressing to {zip_path}")
        shutil.make_archive(str(save_path), "zip", save_path)
        if args.delete:
            shutil.rmtree(save_path)

    logger.info("Done!")


if __name__ == "__main__":
    main()
