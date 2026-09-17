"""MUP (Micro Ultrasound Prostate) Segmentation dataset preprocessing.

Dataset: Micro ultrasound prostate segmentation dataset
Source: https://zenodo.org/records/10475293

Data format:
- Micro_Ultrasound_Prostate_Segmentation_Dataset/
  - train/
    - micro_ultrasound_scans/microUS_train_{id}.nii.gz
    - expert_annotations/expert_annotation_train_{id}.nii.gz
    - non_expert_annotations/nonexpert_annotation_train_{id}.nii.gz
  - test/
    - micro_ultrasound_scans/microUS_test_{id}.nii.gz
    - expert_annotations/expert_annotation_test_{id}.nii.gz
    - medical_student_annotations/
    - master_student_annotations/
    - clinician_annotations/

Output: SAM2-compatible format (SA-V for 3D volumes, SA-1B for 2D slices)
"""

import argparse
import logging
import re
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
from ..utils.image import imflip

DATASET_NAME = "MUP"

CATEGORIES = [
    {"supercategory": "organ", "id": 1, "name": "prostate"},
]

# Annotation types available
ANNOTATION_TYPES = ["expert", "non_expert", "medical_student", "master_student", "clinician"]


def parse_args():
    parser = argparse.ArgumentParser(description=f"Convert {DATASET_NAME} to SAM2 format")
    parser.add_argument(
        "--path",
        type=str,
        default="/mnt/data/Dataset/UltraSound/Raw/MUP.zip",
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
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "test", "all"],
        default="all",
        help="Which split to process",
    )
    parser.add_argument(
        "--annotation-type",
        type=str,
        choices=ANNOTATION_TYPES,
        default="expert",
        help="Which annotation type to use",
    )
    parser.add_argument("--save-viz", action="store_true")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument(
        "--slice-axis",
        type=int,
        default=0,
        help="Axis to slice along for 3D volumes (0=axial)",
    )
    parser.add_argument(
        "--min-mask-ratio",
        type=float,
        default=0.001,
        help="Minimum ratio of non-zero pixels in mask to include slice",
    )
    parser.add_argument("--zip", action="store_true")
    parser.add_argument("--delete", action="store_true")
    return parser.parse_args()


def extract_case_id(filename: str) -> Optional[str]:
    """Extract case ID from filename like 'microUS_train_01.nii.gz' -> '01'."""
    match = re.search(r"_(\d+)\.nii", filename)
    if match:
        return match.group(1)
    return None


def create_background_mask(volume: np.ndarray) -> np.ndarray:
    """Create mask for background regions (where image is zero).
    
    A voxel is considered background if any slice through it in any
    dimension is entirely zero.
    """
    mask = np.zeros(volume.shape, dtype=bool)

    for axis in range(3):
        axis_sum = np.sum(volume, axis=axis)
        expanded = np.expand_dims(axis_sum, axis=axis)
        mask |= (expanded == 0)

    return mask


def collect_cases(
    data_dir: Path,
    split: str = "all",
    annotation_type: str = "expert",
) -> List[Dict]:
    """Collect image-mask pairs from the dataset."""
    cases = []

    # Handle nested folder structure
    dataset_dir = None
    for candidate in [
        data_dir / "Micro_Ultrasound_Prostate_Segmentation_Dataset",
        data_dir,
    ]:
        if (candidate / "train").exists() or (candidate / "test").exists():
            dataset_dir = candidate
            break

    if dataset_dir is None:
        return cases

    splits = ["train", "test"] if split == "all" else [split]

    for split_name in splits:
        split_dir = dataset_dir / split_name
        if not split_dir.exists():
            continue

        scans_dir = split_dir / "micro_ultrasound_scans"
        
        # Determine annotation folder based on type
        if annotation_type == "expert":
            ann_dir = split_dir / "expert_annotations"
            ann_prefix = "expert_annotation"
        elif annotation_type == "non_expert":
            ann_dir = split_dir / "non_expert_annotations"
            ann_prefix = "nonexpert_annotation"
        elif annotation_type == "medical_student":
            ann_dir = split_dir / "medical_student_annotations"
            ann_prefix = "medical_student_annotation"
        elif annotation_type == "master_student":
            ann_dir = split_dir / "master_student_annotations"
            ann_prefix = "master_student_annotation"
        elif annotation_type == "clinician":
            ann_dir = split_dir / "clinician_annotations"
            ann_prefix = "clinician_annotation"
        else:
            ann_dir = split_dir / "expert_annotations"
            ann_prefix = "expert_annotation"

        if not scans_dir.exists() or not ann_dir.exists():
            continue

        # Build case ID mapping
        for scan_file in sorted(scans_dir.glob("*.nii.gz")):
            case_id = extract_case_id(scan_file.name)
            if case_id is None:
                continue

            # Find corresponding annotation
            ann_file = ann_dir / f"{ann_prefix}_{split_name}_{case_id}.nii.gz"
            if not ann_file.exists():
                continue

            cases.append({
                "image_path": scan_file,
                "mask_path": ann_file,
                "case_id": case_id,
                "split": split_name,
                "annotation_type": annotation_type,
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

    video_split = args.split if args.split != "all" else "train"

    annotator = SAM2VideoAnnotator(
        out_dir=save_path,
        dataset_name=DATASET_NAME,
        categories=CATEGORIES,
        save_visualization=args.save_viz,
        video_environment="ultrasound",
        video_split=video_split,
    )

    logger.info(f"Output format: {args.format}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Annotation type: {args.annotation_type}")

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

        cases = collect_cases(data_dir, args.split, args.annotation_type)
        logger.info(f"Found {len(cases)} cases")

        if args.max_cases:
            cases = cases[:args.max_cases]

        processed = 0
        total_slices = 0

        for idx, case in enumerate(cases):
            try:
                # Load NIfTI volumes
                img_sitk = sitk.ReadImage(str(case["image_path"]))
                img_np = sitk.GetArrayFromImage(img_sitk).astype(np.uint8)

                mask_sitk = sitk.ReadImage(str(case["mask_path"]))
                mask_np = sitk.GetArrayFromImage(mask_sitk).astype(np.uint8)

                if img_np.shape != mask_np.shape:
                    logger.warning(f"Shape mismatch for case {case['case_id']}")
                    continue

                # Remove annotations outside ultrasound FOV
                bg_mask = create_background_mask(img_np)
                mask_np[bg_mask] = 0

                video_name = f"{case['split']}_{case['case_id']}"

                if args.format == "sa-v":
                    # Process as video (3D volume)
                    processed_frames = []
                    processed_masks = []

                    num_slices = img_np.shape[args.slice_axis]
                    for slice_idx in range(num_slices):
                        if args.slice_axis == 0:
                            img_slice = img_np[slice_idx]
                            mask_slice = mask_np[slice_idx]
                        elif args.slice_axis == 1:
                            img_slice = img_np[:, slice_idx]
                            mask_slice = mask_np[:, slice_idx]
                        else:
                            img_slice = img_np[:, :, slice_idx]
                            mask_slice = mask_np[:, :, slice_idx]

                        # Skip empty masks
                        mask_ratio = np.sum(mask_slice > 0) / mask_slice.size
                        if mask_ratio < args.min_mask_ratio:
                            continue

                        # Flip for consistency
                        img_slice = imflip(img_slice, "vertical")
                        mask_slice = imflip(mask_slice, "vertical")

                        # Binarize mask
                        mask_slice = (mask_slice > 0).astype(np.uint8)

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
                                "split": case["split"],
                                "annotation_type": case["annotation_type"],
                                "dataset": DATASET_NAME,
                            },
                        )
                        total_slices += len(processed_frames)

                else:
                    # Process as individual images (SA-1B)
                    num_slices = img_np.shape[args.slice_axis]
                    for slice_idx in range(num_slices):
                        if args.slice_axis == 0:
                            img_slice = img_np[slice_idx]
                            mask_slice = mask_np[slice_idx]
                        elif args.slice_axis == 1:
                            img_slice = img_np[:, slice_idx]
                            mask_slice = mask_np[:, slice_idx]
                        else:
                            img_slice = img_np[:, :, slice_idx]
                            mask_slice = mask_np[:, :, slice_idx]

                        # Skip empty masks
                        mask_ratio = np.sum(mask_slice > 0) / mask_slice.size
                        if mask_ratio < args.min_mask_ratio:
                            continue

                        # Flip for consistency
                        img_slice = imflip(img_slice, "vertical")
                        mask_slice = imflip(mask_slice, "vertical")

                        # Binarize mask
                        mask_slice = (mask_slice > 0).astype(np.uint8)

                        slice_name = f"{video_name}_{slice_idx:04d}"
                        annotator.add_2d(
                            img_slice,
                            mask_slice,
                            video_name=slice_name,
                            format="sa-1b",
                            metadata={
                                "case_id": case["case_id"],
                                "split": case["split"],
                                "slice_idx": slice_idx,
                                "annotation_type": case["annotation_type"],
                                "dataset": DATASET_NAME,
                            },
                        )
                        total_slices += 1

                processed += 1

                if (idx + 1) % 10 == 0:
                    logger.info(f"Processed {idx + 1}/{len(cases)} cases ({total_slices} slices)")

            except Exception as e:
                logger.warning(f"Error processing case {case['case_id']}: {e}")
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
