"""TDSC-ABUS (Tumor Detection and Segmentation Challenge - ABUS) dataset preprocessing.

Dataset: 3D Automated Breast Ultrasound (ABUS) tumor segmentation
Source: https://tdsc-abus2023.grand-challenge.org/

Data format:
- Data/DATA_XXX.nrrd: 3D ABUS volumes
- Mask/MASK_XXX.nrrd: Binary tumor segmentation masks

Category: breast_tumor

Output: SAM2-compatible format (SA-V for 3D volumes, SA-1B for slices)
"""

import argparse
import logging
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ..utils.sam2_annotator import SAM2VideoAnnotator

DATASET_NAME = "TDSC-ABUS"

CATEGORIES = [
    {"supercategory": "tumor", "id": 1, "name": "breast_tumor"},
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
        help="Output format (sa-v treats each volume as video)",
    )
    parser.add_argument(
        "--view",
        type=str,
        choices=["axial", "sagittal", "coronal"],
        default="axial",
        help="Slice view to extract",
    )
    parser.add_argument(
        "--min-mask-ratio",
        type=float,
        default=0.0001,
        help="Minimum ratio of non-zero pixels to include slice",
    )
    parser.add_argument("--save-viz", action="store_true")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--zip", action="store_true")
    parser.add_argument("--delete", action="store_true")
    return parser.parse_args()


def extract_slices(nrrd_path: str, view: str = "axial") -> np.ndarray:
    """Extract 2D slices from a NRRD file."""
    import SimpleITK as sitk

    img = sitk.ReadImage(nrrd_path)
    arr = sitk.GetArrayFromImage(img)

    if view == "axial":
        return arr  # (Z, Y, X)
    elif view == "sagittal":
        return np.transpose(arr, (2, 0, 1))  # (X, Z, Y)
    elif view == "coronal":
        return np.transpose(arr, (1, 0, 2))  # (Y, Z, X)
    return arr


def collect_cases(data_dir: Path) -> List[Dict]:
    """Collect volume-mask pairs from the dataset."""
    cases = []

    # Handle nested folder structure
    for candidate in [
        data_dir / "TDSC-ABUS",
        data_dir / "TDSC_ABUS",
        data_dir,
    ]:
        if (candidate / "Data").exists() and (candidate / "Mask").exists():
            data_dir = candidate
            break

    data_folder = data_dir / "Data"
    mask_folder = data_dir / "Mask"

    if not data_folder.exists() or not mask_folder.exists():
        return cases

    # Match data to masks
    for data_file in sorted(data_folder.glob("*.nrrd")):
        # Pattern: DATA_XXX.nrrd -> MASK_XXX.nrrd
        case_id = data_file.stem.replace("DATA_", "")
        mask_file = mask_folder / f"MASK_{case_id}.nrrd"

        if mask_file.exists():
            cases.append({
                "data_path": data_file,
                "mask_path": mask_file,
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
                imgs = extract_slices(str(case["data_path"]), args.view).astype(np.uint8)
                masks = extract_slices(str(case["mask_path"]), args.view).astype(np.uint8)

                if imgs.shape != masks.shape:
                    logger.warning(f"Shape mismatch for case {case['case_id']}")
                    continue

                # Binarize mask
                masks = (masks > 0).astype(np.uint8)

                video_name = f"{case['case_id']}"

                if args.format == "sa-v":
                    # Process as video
                    processed_frames = []
                    processed_masks = []

                    num_slices = imgs.shape[0]
                    for slice_idx in range(num_slices):
                        img_slice = imgs[slice_idx]
                        mask_slice = masks[slice_idx]

                        # Skip slices with insufficient mask coverage
                        mask_ratio = np.sum(mask_slice > 0) / mask_slice.size
                        if mask_ratio < args.min_mask_ratio:
                            continue

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

                        mask_ratio = np.sum(mask_slice > 0) / mask_slice.size
                        if mask_ratio < args.min_mask_ratio:
                            continue

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
