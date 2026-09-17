"""ACOUSLIC (Automated measurement of fetal abdominal Circumference
using Ultrasound imaging from Fetal Screening) dataset preprocessing.

Dataset: Fetal abdomen segmentation in stacked 2D ultrasound sweeps
Source: https://acouslic-ai.grand-challenge.org/

Data format:
- images/stacked_fetal_ultrasound/{uuid}.mha: Stacked 2D ultrasound sweeps (num_frames, H, W)
- masks/stacked_fetal_abdomen/{uuid}.mha: Segmentation masks (num_frames, H, W)
- circumferences/fetal_abdominal_circumferences_per_sweep.csv: Metadata

Mask labels:
    0 = background
    1, 2 = fetal abdomen (from different sweeps within the stacked volume)

Only a small subset of frames per volume have annotations (~3-39 out of 840).
We extract only annotated frames.

Output: SAM2-compatible format (SA-1B style for individual frames, SA-V for video)
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ..utils.sam2_annotator import SAM2VideoAnnotator

DATASET_NAME = "ACOUSLIC"

CATEGORIES = [
    {"supercategory": "fetal", "id": 1, "name": "fetal_abdomen"},
]


def parse_args():
    parser = argparse.ArgumentParser(description=f"Convert {DATASET_NAME} to SAM2 format")
    parser.add_argument(
        "--path",
        type=str,
        default="/mnt/data/Dataset/UltraSound/ACOUSLIC",
        help="Path to dataset directory",
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
    parser.add_argument("--max-volumes", type=int, default=None)
    parser.add_argument("--zip", action="store_true")
    parser.add_argument("--delete", action="store_true")
    return parser.parse_args()


def read_mha(file_path: str) -> np.ndarray:
    """Read an MHA file and return the numpy array.

    Args:
        file_path: Path to .mha file.

    Returns:
        Numpy array with shape (num_frames, H, W).
    """
    try:
        import SimpleITK as sitk
    except ImportError:
        raise ImportError(
            "SimpleITK is required for MHA file processing. "
            "Install with: pip install SimpleITK"
        )

    sitk_img = sitk.ReadImage(file_path)
    return sitk.GetArrayFromImage(sitk_img)


def gray2rgb(gray: np.ndarray) -> np.ndarray:
    """Convert grayscale image to RGB."""
    if gray.ndim == 2:
        return np.stack([gray, gray, gray], axis=-1)
    elif gray.ndim == 3 and gray.shape[-1] == 1:
        return np.repeat(gray, 3, axis=-1)
    return gray


def collect_samples(data_dir: Path) -> List[Dict]:
    """Collect matched image-mask MHA pairs."""
    samples = []

    images_dir = data_dir / "images" / "stacked_fetal_ultrasound"
    masks_dir = data_dir / "masks" / "stacked_fetal_abdomen"

    if not images_dir.exists() or not masks_dir.exists():
        return samples

    for img_file in sorted(images_dir.glob("*.mha")):
        uuid = img_file.stem
        mask_file = masks_dir / f"{uuid}.mha"

        if mask_file.exists():
            samples.append({
                "image_path": img_file,
                "mask_path": mask_file,
                "uuid": uuid,
            })

    return samples


def main():
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

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
    logger.info(f"Input: {data_path}")
    logger.info(f"Output: {save_path}")

    samples = collect_samples(data_path)
    logger.info(f"Found {len(samples)} volumes")

    if args.max_volumes:
        samples = samples[:args.max_volumes]

    processed = 0
    total_frames = 0

    for idx, sample in enumerate(samples):
        try:
            img_vol = read_mha(str(sample["image_path"]))  # (N, H, W), uint8
            mask_vol = read_mha(str(sample["mask_path"]))   # (N, H, W), uint8
        except Exception as e:
            logger.warning(f"Error reading {sample['uuid']}: {e}")
            continue

        # Unify mask labels: both 1 and 2 represent fetal abdomen → map to 1
        mask_vol = (mask_vol > 0).astype(np.uint8)

        # Find annotated frames
        annotated_indices = [
            i for i in range(mask_vol.shape[0])
            if np.any(mask_vol[i] > 0)
        ]

        if not annotated_indices:
            continue

        uuid = sample["uuid"]

        if args.format == "sa-1b":
            # Export each annotated frame individually
            for frame_idx in annotated_indices:
                frame = img_vol[frame_idx]  # (H, W), uint8
                mask = mask_vol[frame_idx]  # (H, W), uint8

                frame_rgb = gray2rgb(frame)
                image_name = f"{uuid}_{frame_idx:04d}"

                annotator.add_2d(
                    frame_rgb,
                    mask,
                    video_name=image_name,
                    format="sa-1b",
                    metadata={
                        "uuid": uuid,
                        "frame_idx": frame_idx,
                        "total_frames": int(img_vol.shape[0]),
                        "dataset": DATASET_NAME,
                    },
                )
                total_frames += 1
        else:
            # Export annotated frames as a video sequence
            frames = []
            frame_masks = []

            for frame_idx in annotated_indices:
                frame_rgb = gray2rgb(img_vol[frame_idx])
                frames.append(frame_rgb)
                frame_masks.append(mask_vol[frame_idx])

            frames_arr = np.stack(frames, axis=0)
            masks_arr = np.stack(frame_masks, axis=0)

            annotator.add_video(
                frames_arr,
                masks_arr,
                video_name=uuid,
                metadata={
                    "uuid": uuid,
                    "frame_indices": annotated_indices,
                    "total_frames": int(img_vol.shape[0]),
                    "num_annotated": len(annotated_indices),
                    "dataset": DATASET_NAME,
                },
            )
            total_frames += len(frames)

        processed += 1

        if (idx + 1) % 10 == 0:
            logger.info(f"Processed {idx + 1}/{len(samples)} volumes ({total_frames} frames)")

    logger.info(f"Total: {processed} volumes, {total_frames} annotated frames")

    stats = annotator.finalize()
    logger.info(f"Dataset stats: {stats}")

    if args.zip:
        import shutil
        shutil.make_archive(str(save_path), "zip", save_path)
        if args.delete:
            shutil.rmtree(save_path)

    logger.info("Done!")


if __name__ == "__main__":
    main()
