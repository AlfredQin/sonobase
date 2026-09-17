"""SegThy (Thyroid Segmentation) dataset preprocessing.

Dataset: 3D freehand ultrasound thyroid segmentation
Source: https://www.cs.cit.tum.de/camp/publications/segthy-dataset/

Data format:
- ground_truth_data/US/{id}_US.nii: 3D ultrasound volumes
- ground_truth_data/US_thyroid_label/{id}.nii: Multi-class segmentation masks

Categories:
1. thyroid - thyroid gland
2. carotid - carotid artery
3. jugular - jugular vein

Output: SAM2-compatible format (SA-V for 3D volumes, SA-1B for slices)
"""

import argparse
import logging
import re
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ..utils.sam2_annotator import SAM2VideoAnnotator
from ..utils.image import imflip

DATASET_NAME = "SegThy"

CATEGORIES = [
    {"supercategory": "gland", "id": 1, "name": "thyroid"},
    {"supercategory": "vessel", "id": 2, "name": "carotid"},
    {"supercategory": "vessel", "id": 3, "name": "jugular"},
]


def parse_args():
    parser = argparse.ArgumentParser(description=f"Convert {DATASET_NAME} to SAM2 format")
    parser.add_argument(
        "--path",
        type=str,
        default="/mnt/data/Dataset/UltraSound/Raw/SegThy.zip",
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
        help="Output format (sa-v treats each volume as video)",
    )
    parser.add_argument("--min-mask-pixels", type=int, default=10, help="Minimum mask area")
    parser.add_argument(
        "--view",
        type=str,
        choices=["axial", "sagittal", "coronal"],
        default="axial",
        help="Slice view to extract",
    )
    parser.add_argument("--save-viz", action="store_true")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--zip", action="store_true")
    parser.add_argument("--delete", action="store_true")
    return parser.parse_args()


def extract_slices(nii_path: str, view: str = "axial") -> np.ndarray:
    """Extract 2D slices from a NIfTI file."""
    import SimpleITK as sitk

    img = sitk.ReadImage(nii_path)
    arr = sitk.GetArrayFromImage(img)

    if view == "axial":
        return arr  # (Z, Y, X)
    elif view == "sagittal":
        return np.transpose(arr, (2, 0, 1))  # (X, Z, Y)
    elif view == "coronal":
        return np.transpose(arr, (1, 0, 2))  # (Y, Z, X)
    return arr


def collect_cases(data_dir: Path) -> List[Dict]:
    """Collect volume-label pairs from the dataset."""
    cases = []

    # Handle nested folder structure
    for candidate in [
        data_dir / "SegThy" / "US_volunteer_dataset" / "ground_truth_data",
        data_dir / "US_volunteer_dataset" / "ground_truth_data",
        data_dir / "segthy-dataset" / "ground_truth_data",
        data_dir / "ground_truth_data",
    ]:
        if candidate.exists():
            data_dir = candidate
            break

    us_dir = data_dir / "US"
    label_dir = data_dir / "US_thyroid_label"

    if not us_dir.exists() or not label_dir.exists():
        return cases

    # Match US volumes to labels
    for us_file in sorted(us_dir.glob("*.nii")):
        # Pattern: {id}_US.nii -> {id}.nii
        case_id = us_file.stem.replace("_US", "")
        label_file = label_dir / f"{case_id}.nii"

        if label_file.exists():
            cases.append({
                "us_path": us_file,
                "label_path": label_file,
                "case_id": case_id,
            })

    return cases


def main():
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    # Lazy import SimpleITK
    try:
        import SimpleITK as sitk
    except ImportError:
        logger.error("SimpleITK is required. Install with: pip install SimpleITK")
        return

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

    logger.info(f"Output format: {args.format}, view: {args.view}")

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

        cases = collect_cases(data_dir)
        logger.info(f"Found {len(cases)} cases")

        if args.max_cases:
            cases = cases[:args.max_cases]

        processed = 0
        total_slices = 0

        for idx, case in enumerate(cases):
            try:
                # Load volumes
                imgs = extract_slices(str(case["us_path"]), args.view).astype(np.uint8)
                masks = extract_slices(str(case["label_path"]), args.view).astype(np.uint8)

                if imgs.shape != masks.shape:
                    logger.warning(f"Shape mismatch for {case['case_id']}")
                    continue

                video_name = f"{case['case_id']}"

                if args.format == "sa-v":
                    # Process as video
                    processed_frames = []
                    processed_masks = []

                    num_slices = imgs.shape[0]
                    for slice_idx in range(num_slices):
                        img_slice = imgs[slice_idx]
                        mask_slice = masks[slice_idx]

                        # Skip empty masks
                        if not np.any(mask_slice > 0):
                            continue

                        # Flip for better visualization
                        img_slice = imflip(img_slice, "diagonal")
                        mask_slice = imflip(mask_slice, "diagonal")

                        processed_frames.append(img_slice)
                        processed_masks.append(mask_slice)

                    if processed_frames:
                        frames_arr = np.stack(processed_frames, axis=0)
                        masks_arr = np.stack(processed_masks, axis=0)

                        annotator.add_video(
                            frames_arr,
                            masks_arr,
                            video_name=video_name,
                            metadata={
                                "case_id": case["case_id"],
                                "view": args.view,
                                "dataset": DATASET_NAME,
                            },
                        )
                        total_slices += len(processed_frames)

                else:
                    # Process as individual slices (SA-1B)
                    num_slices = imgs.shape[0]
                    for slice_idx in range(num_slices):
                        img_slice = imgs[slice_idx]
                        mask_slice = masks[slice_idx]

                        if not np.any(mask_slice > 0):
                            continue

                        img_slice = imflip(img_slice, "diagonal")
                        mask_slice = imflip(mask_slice, "diagonal")

                        slice_name = f"{video_name}_{slice_idx:04d}"
                        annotator.add_2d(
                            img_slice,
                            mask_slice,
                            video_name=slice_name,
                            format="sa-1b",
                            metadata={
                                "case_id": case["case_id"],
                                "slice_idx": slice_idx,
                                "view": args.view,
                                "dataset": DATASET_NAME,
                            },
                        )
                        total_slices += 1

                processed += 1

                if (idx + 1) % 10 == 0:
                    logger.info(f"Processed {idx + 1}/{len(cases)} cases ({total_slices} slices)")

            except Exception as e:
                logger.warning(f"Error processing {case['case_id']}: {e}")
                continue

        logger.info(f"Total: {processed} cases, {total_slices} slices")

    stats = annotator.finalize()
    logger.info(f"Dataset stats: {stats}")

    if args.zip:
        shutil.make_archive(str(save_path), "zip", save_path)
        if args.delete:
            shutil.rmtree(save_path)

    logger.info("Done!")


if __name__ == "__main__":
    main()
