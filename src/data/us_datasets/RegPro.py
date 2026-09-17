"""RegPro (Registration Prostate) dataset preprocessing.

Dataset: Multi-modality prostate ultrasound registration dataset
Source: https://muregpro.github.io/data.html

Data format:
- train/us_images/{case_id}.nii.gz: Ultrasound images (3D)
- train/us_labels/{case_id}.nii.gz: Segmentation masks (4D: [slices, H, W, labels])
- val/us_images/{case_id}.nii.gz: Validation ultrasound images
- val/us_labels/{case_id}.nii.gz: Validation masks

Note: MR images are excluded, only US images are processed.

Output: SAM2-compatible format (SA-V for 3D volumes, SA-1B for 2D slices)
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
from ..utils.image import imflip

DATASET_NAME = "RegPro"

CATEGORIES = [
    {"supercategory": "organ", "id": 1, "name": "prostate"},
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
        help="Output format",
    )
    parser.add_argument("--min-mask-pixels", type=int, default=10, help="Minimum mask area")
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "all"],
        default="all",
        help="Which split to process",
    )
    parser.add_argument(
        "--view",
        type=str,
        choices=["axial", "sagittal", "coronal"],
        default="sagittal",
        help="Slice view to extract",
    )
    parser.add_argument("--save-viz", action="store_true")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--zip", action="store_true")
    parser.add_argument("--delete", action="store_true")
    return parser.parse_args()


def extract_slices(nii_path: str, view: str = "sagittal") -> np.ndarray:
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


def collect_cases(data_dir: Path, split: str = "all") -> List[Dict]:
    """Collect ultrasound image-mask pairs."""
    cases = []

    # Handle nested folder structure
    if (data_dir / "RegPro").exists():
        data_dir = data_dir / "RegPro"
    elif (data_dir / "regPro").exists():
        data_dir = data_dir / "regPro"

    splits = ["train", "val"] if split == "all" else [split]

    for split_name in splits:
        split_dir = data_dir / split_name
        if not split_dir.exists():
            continue

        us_images_dir = split_dir / "us_images"
        us_labels_dir = split_dir / "us_labels"

        if not us_images_dir.exists() or not us_labels_dir.exists():
            continue

        for img_file in sorted(us_images_dir.glob("*.nii.gz")):
            case_id = img_file.stem.replace(".nii", "")
            mask_file = us_labels_dir / f"{case_id}.nii.gz"

            if mask_file.exists():
                cases.append({
                    "image_path": img_file,
                    "mask_path": mask_file,
                    "case_id": case_id,
                    "split": split_name,
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
        min_mask_pixels=args.min_mask_pixels,
    )

    logger.info(f"Output format: {args.format}, split: {args.split}, view: {args.view}")

    if use_temp_dir:
        temp_context = tempfile.TemporaryDirectory()
    else:
        from contextlib import nullcontext
        temp_context = nullcontext(str(data_path))

    with temp_context as temp_dir:
        if use_temp_dir:
            logger.info(f"Extracting {data_path} to {temp_dir}")
            shutil.unpack_archive(str(data_path), temp_dir)
            
            # Extract nested train.zip and val.zip
            for nested in ["train.zip", "val.zip"]:
                nested_path = Path(temp_dir) / nested
                if nested_path.exists():
                    shutil.unpack_archive(str(nested_path), temp_dir)
            
            data_dir = Path(temp_dir)
        else:
            data_dir = data_path

        cases = collect_cases(data_dir, args.split)
        logger.info(f"Found {len(cases)} cases")

        if args.max_cases:
            cases = cases[:args.max_cases]

        processed = 0
        total_slices = 0

        for idx, case in enumerate(cases):
            try:
                # Load images
                imgs = extract_slices(str(case["image_path"]), args.view)
                
                # Load masks (4D: [slices, H, W, labels] or [labels, Z, Y, X])
                sitk_mask = sitk.ReadImage(str(case["mask_path"]))
                masks_raw = sitk.GetArrayFromImage(sitk_mask)
                
                # Handle 4D masks - transpose to match image slicing
                if masks_raw.ndim == 4:
                    # Assuming format is [labels, Z, Y, X], transpose based on view
                    if args.view == "sagittal":
                        masks_raw = np.transpose(masks_raw, (0, 3, 1, 2))  # [labels, X, Z, Y]
                    elif args.view == "coronal":
                        masks_raw = np.transpose(masks_raw, (0, 2, 1, 3))  # [labels, Y, Z, X]
                    # else axial: [labels, Z, Y, X] -> keep as is
                
                video_name = f"{case['split']}_{case['case_id']}"

                if args.format == "sa-v":
                    processed_frames = []
                    processed_masks = []

                    num_slices = imgs.shape[0]
                    for slice_idx in range(num_slices):
                        img_slice = imgs[slice_idx]
                        
                        # Combine all label channels into single mask
                        if masks_raw.ndim == 4:
                            mask_slice = np.zeros_like(img_slice, dtype=np.uint8)
                            for label_idx in range(masks_raw.shape[0]):
                                label_mask = masks_raw[label_idx, slice_idx]
                                if label_mask.shape != img_slice.shape:
                                    continue
                                mask_slice[label_mask != 0] = 1  # Single prostate class
                        else:
                            mask_slice = masks_raw[slice_idx].astype(np.uint8)
                            mask_slice[mask_slice != 0] = 1

                        if not np.any(mask_slice != 0):
                            continue

                        # Flip for consistency
                        img_slice = imflip(img_slice, "diagonal")
                        mask_slice = imflip(mask_slice, "diagonal")

                        processed_frames.append(img_slice.astype(np.uint8))
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
                                "view": args.view,
                                "dataset": DATASET_NAME,
                            },
                        )
                        total_slices += len(processed_frames)

                else:
                    # SA-1B format
                    num_slices = imgs.shape[0]
                    for slice_idx in range(num_slices):
                        img_slice = imgs[slice_idx]
                        
                        if masks_raw.ndim == 4:
                            mask_slice = np.zeros_like(img_slice, dtype=np.uint8)
                            for label_idx in range(masks_raw.shape[0]):
                                label_mask = masks_raw[label_idx, slice_idx]
                                if label_mask.shape != img_slice.shape:
                                    continue
                                mask_slice[label_mask != 0] = 1
                        else:
                            mask_slice = masks_raw[slice_idx].astype(np.uint8)
                            mask_slice[mask_slice != 0] = 1

                        if not np.any(mask_slice != 0):
                            continue

                        img_slice = imflip(img_slice, "diagonal")
                        mask_slice = imflip(mask_slice, "diagonal")

                        slice_name = f"{video_name}_{slice_idx:04d}"
                        annotator.add_2d(
                            img_slice.astype(np.uint8),
                            mask_slice,
                            video_name=slice_name,
                            format="sa-1b",
                            metadata={
                                "case_id": case["case_id"],
                                "split": case["split"],
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
