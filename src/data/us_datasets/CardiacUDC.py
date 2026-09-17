"""CardiacUDC dataset preprocessing.

Dataset: Cardiac ultrasound dataset from multiple sites
Source: https://www.kaggle.com/datasets/xiaoweixumedicalai/cardiacudc-dataset

Structure:
- cardiacUDC_dataset/Site_{site}/{patient}_image.nii.gz
- cardiacUDC_dataset/Site_{site}/{patient}_label.nii.gz

Labels: 1=left ventricle, 2=left atrium, 3=right atrium, 4=right ventricle, 
        5=epicardium, 6=left ventricle2

Note: Some masks only have border annotations - need fill_holes preprocessing.

Output: SAM2-compatible format (SA-V for 3D volumes, SA-1B for 2D slices)
"""

import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from scipy import ndimage

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ..utils.sam2_annotator import SAM2VideoAnnotator
from ..utils.image import imflip

DATASET_NAME = "CardiacUDC"

CATEGORIES = [
    {"supercategory": "cardiac", "id": 1, "name": "left_ventricle"},
    {"supercategory": "cardiac", "id": 2, "name": "left_atrium"},
    {"supercategory": "cardiac", "id": 3, "name": "right_atrium"},
    {"supercategory": "cardiac", "id": 4, "name": "right_ventricle"},
    {"supercategory": "cardiac", "id": 5, "name": "epicardium"},
    {"supercategory": "cardiac", "id": 6, "name": "left_ventricle2"},
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
        help="Output format",
    )
    parser.add_argument("--save-viz", action="store_true")
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument(
        "--view",
        type=str,
        default="axial",
        choices=["axial", "sagittal", "coronal"],
        help="Slice view to extract",
    )
    parser.add_argument(
        "--min-mask-ratio",
        type=float,
        default=0.025,
        help="Minimum ratio of non-zero pixels in mask",
    )
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
    else:
        return arr


def fill_mask_labeled(mask: np.ndarray) -> np.ndarray:
    """Fill holes in labeled mask (for border-only annotations)."""
    unique_labels = np.unique(mask)
    filled_mask = np.zeros_like(mask)

    for label in unique_labels:
        if label == 0:
            continue
        binary_mask = mask == label
        filled_binary_mask = ndimage.binary_fill_holes(binary_mask)
        filled_mask[filled_binary_mask] = label

    return filled_mask


def collect_patients(data_dir: Path) -> List[Dict]:
    """Collect patient data organized by site."""
    patients = []

    # Find the dataset folder
    dataset_dir = None
    for candidate in [
        data_dir / "cardiacUDC_dataset",
        data_dir,
    ]:
        if candidate.exists() and any(candidate.glob("Site_*")):
            dataset_dir = candidate
            break

    if dataset_dir is None:
        return patients

    # Collect by site
    for site_dir in sorted(dataset_dir.glob("Site_*")):
        if not site_dir.is_dir():
            continue

        site_name = site_dir.name

        # Group files by patient
        patient_files = defaultdict(dict)
        for nii_file in site_dir.glob("*.nii.gz"):
            filename = nii_file.name
            if "_image.nii.gz" in filename:
                patient_id = filename.replace("_image.nii.gz", "")
                patient_files[patient_id]["image"] = nii_file
            elif "_label.nii.gz" in filename:
                patient_id = filename.replace("_label.nii.gz", "")
                patient_files[patient_id]["label"] = nii_file

        for patient_id, files in patient_files.items():
            if "image" in files and "label" in files:
                patients.append({
                    "site": site_name,
                    "patient_id": patient_id,
                    "image_path": files["image"],
                    "label_path": files["label"],
                })

    return patients


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

            # Handle nested zip if present (legacy support)
            nested_zip = Path(temp_dir) / "cardiacUDC_dataset.zip"
            if nested_zip.exists():
                logger.info("Found nested zip, extracting...")
                shutil.unpack_archive(str(nested_zip), temp_dir)

            data_dir = Path(temp_dir)
        else:
            data_dir = data_path

        patients = collect_patients(data_dir)
        logger.info(f"Found {len(patients)} patients")

        if args.max_patients:
            patients = patients[:args.max_patients]

        processed = 0
        total_slices = 0

        for idx, patient in enumerate(patients):
            try:
                imgs = extract_slices(str(patient["image_path"]), args.view)
                masks = extract_slices(str(patient["label_path"]), args.view)

                if imgs.shape != masks.shape:
                    logger.warning(f"Shape mismatch for {patient['patient_id']}")
                    continue

                # Normalize images to uint8
                imgs = ((imgs - imgs.min()) / (imgs.max() - imgs.min() + 1e-8) * 255).astype(np.uint8)

                video_name = f"{patient['site']}_{patient['patient_id']}"

                if args.format == "sa-v":
                    # Process as video
                    processed_frames = []
                    processed_masks = []

                    for i in range(len(imgs)):
                        img = imgs[i]
                        mask = masks[i].astype(np.uint8)

                        # Skip empty masks
                        if not np.any(mask != 0):
                            continue

                        # Fill holes in border-only masks
                        mask = fill_mask_labeled(mask)

                        # Check minimum mask ratio
                        ratio = np.sum(mask != 0) / mask.size
                        if ratio < args.min_mask_ratio:
                            continue

                        # Flip diagonal (as in original code)
                        img = imflip(img, "diagonal")
                        mask = imflip(mask, "diagonal")

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
                                "site": patient["site"],
                                "patient_id": patient["patient_id"],
                                "dataset": DATASET_NAME,
                            },
                        )
                        total_slices += len(processed_frames)
                else:
                    # Process as individual images
                    for i in range(len(imgs)):
                        img = imgs[i]
                        mask = masks[i].astype(np.uint8)

                        if not np.any(mask != 0):
                            continue

                        mask = fill_mask_labeled(mask)
                        ratio = np.sum(mask != 0) / mask.size
                        if ratio < args.min_mask_ratio:
                            continue

                        img = imflip(img, "diagonal")
                        mask = imflip(mask, "diagonal")

                        slice_name = f"{video_name}_{i:04d}"
                        annotator.add_2d(
                            img,
                            mask,
                            video_name=slice_name,
                            format="sa-1b",
                            metadata={
                                "site": patient["site"],
                                "patient_id": patient["patient_id"],
                                "slice_idx": i,
                                "dataset": DATASET_NAME,
                            },
                        )
                        total_slices += 1

                processed += 1

                if (idx + 1) % 10 == 0:
                    logger.info(f"Processed {idx + 1}/{len(patients)} patients ({total_slices} slices)")

            except Exception as e:
                logger.warning(f"Error processing {patient['patient_id']}: {e}")
                continue

        logger.info(f"Total: {processed} patients, {total_slices} slices")

    stats = annotator.finalize()
    logger.info(f"Dataset stats: {stats}")

    if args.zip:
        shutil.make_archive(str(save_path), "zip", save_path)
        if args.delete:
            shutil.rmtree(save_path)

    logger.info("Done!")


if __name__ == "__main__":
    main()
