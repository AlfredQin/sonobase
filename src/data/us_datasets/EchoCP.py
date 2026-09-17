"""EchoCP (Echocardiography Chamber Parsing) dataset preprocessing.

Dataset: Cardiac ultrasound 3D volumes for chamber segmentation
Source: https://www.kaggle.com/datasets/xiaoweixumedicalai/echocp

Structure:
- EchoCP_dataset/{id}_{r/v}_image.nii.gz: Image volumes
- EchoCP_dataset/{id}_{r/v}_label.nii.gz: Label volumes

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

DATASET_NAME = "EchoCP"

CATEGORIES = [
    {"supercategory": "chamber", "id": 1, "name": "cardiac_chamber"},
]


def parse_args():
    parser = argparse.ArgumentParser(description=f"Convert {DATASET_NAME} to SAM2 format")
    parser.add_argument(
        "--path",
        type=str,
        default=f"/mnt/data/Dataset/UltraSound/Raw/{DATASET_NAME}.zip",
        help="Path to dataset (zip file or extracted folder)",
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
    parser.add_argument("--save-viz", action="store_true")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument(
        "--view",
        type=str,
        default="axial",
        choices=["axial", "sagittal", "coronal"],
        help="Slice view to extract",
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
        return arr
    elif view == "sagittal":
        return np.transpose(arr, (2, 0, 1))
    elif view == "coronal":
        return np.transpose(arr, (1, 0, 2))
    return arr


def collect_cases(data_dir: Path) -> List[Dict]:
    """Collect image-label pairs."""
    cases = []

    # Handle nested folder structure
    if (data_dir / "EchoCP_dataset").exists():
        data_dir = data_dir / "EchoCP_dataset"
    elif (data_dir / "EchoCP").exists():
        data_dir = data_dir / "EchoCP" / "EchoCP_dataset"

    # Group files by case ID
    case_files = defaultdict(dict)
    for nii_file in data_dir.glob("*.nii.gz"):
        filename = nii_file.stem.replace(".nii", "")
        parts = filename.rsplit("_", 1)
        if len(parts) == 2:
            case_id = parts[0]
            file_type = parts[1]  # image or label
            case_files[case_id][file_type] = nii_file

    for case_id, files in sorted(case_files.items()):
        if "image" in files and "label" in files:
            cases.append({
                "case_id": case_id,
                "image_path": files["image"],
                "label_path": files["label"],
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
            
            # Handle split archive format (EchoCP contains .z01 and .change2zip files)
            z01_files = list(data_dir.glob("*.z01"))
            change2zip_files = list(data_dir.glob("*.change2zip"))
            
            if z01_files and change2zip_files:
                import subprocess
                # Rename .change2zip to .zip
                change2zip = change2zip_files[0]
                main_zip = change2zip.with_suffix(".zip")
                shutil.move(str(change2zip), str(main_zip))
                
                # Combine split archive parts
                logger.info("Combining split archive parts...")
                combined_zip = data_dir / "combined.zip"
                result = subprocess.run(
                    ["zip", "-s", "0", str(main_zip), "--out", str(combined_zip)],
                    capture_output=True, text=True, cwd=str(data_dir)
                )
                if result.returncode == 0:
                    # Extract combined archive
                    logger.info("Extracting combined archive...")
                    shutil.unpack_archive(str(combined_zip), str(data_dir))
                else:
                    logger.warning(f"Failed to combine split archive: {result.stderr}")
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
                imgs = extract_slices(str(case["image_path"]), args.view).astype(np.uint8)
                masks = extract_slices(str(case["label_path"]), args.view).astype(np.uint8)

                if imgs.shape != masks.shape:
                    logger.warning(f"Shape mismatch for {case['case_id']}")
                    continue

                video_name = f"{case['case_id']}"

                if args.format == "sa-v":
                    # Process as video
                    processed_frames = []
                    processed_masks = []

                    for i in range(len(imgs)):
                        img = imgs[i]
                        mask = masks[i]

                        if not np.any(mask != 0):
                            continue

                        # Apply diagonal flip and binarize mask
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
                                "case_id": case["case_id"],
                                "dataset": DATASET_NAME,
                            },
                        )
                        total_slices += len(processed_frames)
                else:
                    # Process as individual images
                    for i in range(len(imgs)):
                        img = imgs[i]
                        mask = masks[i]

                        if not np.any(mask != 0):
                            continue

                        img = imflip(img, "diagonal")
                        mask = imflip(mask, "diagonal")
                        mask[mask > 0] = 1

                        slice_name = f"{video_name}_{i:04d}"
                        annotator.add_2d(
                            img,
                            mask,
                            video_name=slice_name,
                            format="sa-1b",
                            metadata={
                                "case_id": case["case_id"],
                                "slice_idx": i,
                                "dataset": DATASET_NAME,
                            },
                        )
                        total_slices += 1

                processed += 1

                if (idx + 1) % 50 == 0:
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
