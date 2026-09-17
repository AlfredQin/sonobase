"""CardiacNet dataset preprocessing.

Dataset: Cardiac ultrasound 3D volumes for ASD/PAH diagnosis
Structure:
- CardiacNet-{ASD/PAH}/{split}/{case_id}_image.nii/ (directory with NIfTI)
- CardiacNet-{ASD/PAH}/{split}/{case_id}_label.nii (label file)

Labels: 1=left ventricle, 2=right ventricle, 3=left atrium, 4=right atrium

Output: SAM2-compatible format (SA-V for 3D volumes, SA-1B for 2D slices)
"""

import argparse
import logging
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import cv2
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ..utils.sam2_annotator import SAM2VideoAnnotator

DATASET_NAME = "CardiacNet"

CATEGORIES = [
    {"supercategory": "cardiac", "id": 1, "name": "left_ventricle"},
    {"supercategory": "cardiac", "id": 2, "name": "right_ventricle"},
    {"supercategory": "cardiac", "id": 3, "name": "left_atrium"},
    {"supercategory": "cardiac", "id": 4, "name": "right_atrium"},
]


def parse_args():
    parser = argparse.ArgumentParser(description=f"Convert {DATASET_NAME} to SAM2 format")
    parser.add_argument(
        "--path",
        type=str,
        default="/mnt/data/Dataset/UltraSound/Raw/CardiacNet.zip",
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
        help="Output format (sa-1b only keeps labeled slices, sa-v treats each volume as video)",
    )
    parser.add_argument("--min-mask-pixels", type=int, default=10, help="Minimum mask area")
    parser.add_argument("--save-viz", action="store_true")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument(
        "--slice-axis",
        type=int,
        default=0,
        help="Axis to slice along (0=axial, 1=coronal, 2=sagittal)",
    )
    parser.add_argument("--zip", action="store_true")
    parser.add_argument("--delete", action="store_true")
    return parser.parse_args()


@dataclass
class CardiacNetCase:
    subdataset: str  # CardiacNet-ASD / CardiacNet-PAH
    split: str  # ASD/Non-ASD/PAH/Non-PAH
    case_id: str
    image_file: Path
    label_file: Path


def find_single_nii_file(folder: Path) -> Optional[Path]:
    """Return the single .nii/.nii.gz inside a directory."""
    if not folder.exists() or not folder.is_dir():
        return None
    nii = sorted(list(folder.glob("*.nii")) + list(folder.glob("*.nii.gz")))
    if not nii:
        nii = sorted(list(folder.rglob("*.nii")) + list(folder.rglob("*.nii.gz")))
    if not nii:
        return None
    # Pick largest if multiple
    nii.sort(key=lambda p: p.stat().st_size, reverse=True)
    return nii[0]


def iter_cases(root: Path) -> Iterator[CardiacNetCase]:
    """Yield all cases with both image and label."""
    for img_dir in sorted(root.rglob("*_image.nii")):
        if not img_dir.is_dir():
            continue

        label_file = Path(str(img_dir).replace("_image.nii", "_label.nii"))
        if not label_file.exists() or not label_file.is_file():
            continue

        image_file = find_single_nii_file(img_dir)
        if image_file is None:
            continue

        try:
            split = img_dir.parent.name
            subdataset = img_dir.parent.parent.name
            case_id = img_dir.name.replace("_image.nii", "")
        except Exception:
            split = "unknown"
            subdataset = "unknown"
            case_id = img_dir.stem

        yield CardiacNetCase(
            subdataset=subdataset,
            split=split,
            case_id=case_id,
            image_file=image_file,
            label_file=label_file,
        )


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    """Normalize volume to 0-255 uint8."""
    vol_min = volume.min()
    vol_max = volume.max()
    if vol_max > vol_min:
        volume = (volume - vol_min) / (vol_max - vol_min) * 255
    return volume.astype(np.uint8)


def main():
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    # Lazy import SimpleITK
    try:
        import SimpleITK as sitk
    except ImportError:
        logger.error("SimpleITK is required for CardiacNet. Install with: pip install SimpleITK")
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

    logger.info(f"Output format: {args.format}")

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

        cases = list(iter_cases(data_dir))
        logger.info(f"Found {len(cases)} cases")

        if args.max_cases:
            cases = cases[:args.max_cases]

        processed = 0
        total_slices = 0

        for idx, case in enumerate(cases):
            try:
                img_np = sitk.GetArrayFromImage(sitk.ReadImage(str(case.image_file)))
                lbl_np = sitk.GetArrayFromImage(sitk.ReadImage(str(case.label_file)))

                if img_np.shape != lbl_np.shape:
                    logger.warning(f"Shape mismatch for {case.case_id}: {img_np.shape} vs {lbl_np.shape}")
                    continue

                # Normalize image
                img_np = normalize_volume(img_np)

                video_name = f"{case.subdataset}_{case.split}_{case.case_id}"

                if args.format == "sa-v":
                    # Treat as video (3D volume)
                    annotator.add_3d(
                        img_np,
                        lbl_np.astype(np.uint8),
                        slice_axis=args.slice_axis,
                        video_name=video_name,
                        metadata={
                            "subdataset": case.subdataset,
                            "split": case.split,
                            "case_id": case.case_id,
                            "dataset": DATASET_NAME,
                        },
                    )
                    total_slices += img_np.shape[args.slice_axis]
                else:
                    # Treat each slice as separate image
                    num_slices = img_np.shape[args.slice_axis]
                    for slice_idx in range(num_slices):
                        if args.slice_axis == 0:
                            img_slice = img_np[slice_idx]
                            lbl_slice = lbl_np[slice_idx]
                        elif args.slice_axis == 1:
                            img_slice = img_np[:, slice_idx]
                            lbl_slice = lbl_np[:, slice_idx]
                        else:
                            img_slice = img_np[:, :, slice_idx]
                            lbl_slice = lbl_np[:, :, slice_idx]

                        # Skip slices with no labels (all zeros) or very few label pixels
                        if np.sum(lbl_slice > 0) < 10:
                            continue

                        slice_name = f"{video_name}_{slice_idx:04d}"
                        annotator.add_2d(
                            img_slice,
                            lbl_slice.astype(np.uint8),
                            video_name=slice_name,
                            format="sa-1b",
                            metadata={
                                "subdataset": case.subdataset,
                                "split": case.split,
                                "case_id": case.case_id,
                                "slice_idx": slice_idx,
                                "dataset": DATASET_NAME,
                            },
                        )
                        total_slices += 1

                processed += 1

                if (idx + 1) % 10 == 0:
                    logger.info(f"Processed {idx + 1}/{len(cases)} cases ({total_slices} slices)")

            except Exception as e:
                logger.warning(f"Error processing {case.case_id}: {e}")
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
