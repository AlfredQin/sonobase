"""Convert CAMUS dataset to SAM2-compatible format.

Dataset: Cardiac Acquisitions for Multi-structure Ultrasound Segmentation (CAMUS)
Source: https://www.creatis.insa-lyon.fr/Challenge/camus/

CAMUS is a cardiac ultrasound dataset containing 2D echocardiography sequences.
Each patient has two views (2-chamber and 4-chamber), stored as NIfTI volumes
where each slice is a frame in the cardiac cycle.

Dataset structure:
    database_nifti/
        patient0001/
            patient0001_2CH_half_sequence.nii     # 2-chamber view video
            patient0001_2CH_half_sequence_gt.nii  # 2-chamber ground truth
            patient0001_4CH_half_sequence.nii     # 4-chamber view video
            patient0001_4CH_half_sequence_gt.nii  # 4-chamber ground truth
            patient0001_2CH_ED.nii, _ES.nii       # End-diastole, end-systole frames
            ...
        patient0002/
        ...

NIfTI format:
    - Shape: (num_frames, height, width), e.g. (18, 389, 549)
    - Image values: float32, 0-255
    - GT values: 0=background, 1=endocardium, 2=epicardium, 3=atrium_wall

Categories:
    1: endocardium (left ventricle cavity)
    2: epicardium (left ventricle myocardium)
    3: atrium_wall (left atrium)

Output format: SA-V (video format) - each patient view becomes a video sequence.

Usage:
    python projects/US_Datasets/CAMUS.py --path /path/to/CAMUS --save-dir /path/to/output
"""

import argparse
import logging
import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ..utils.sam2_annotator import SAM2VideoAnnotator
from ..utils.path import mkdir_or_exist

DATASET_NAME = "CAMUS"

# Category definitions
CATEGORIES = [
    {"supercategory": "cardiac", "id": 1, "name": "endocardium"},
    {"supercategory": "cardiac", "id": 2, "name": "epicardium"},
    {"supercategory": "cardiac", "id": 3, "name": "atrium_wall"},
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=f"Convert {DATASET_NAME} to SAM2 format"
    )
    parser.add_argument(
        "--path",
        type=str,
        help="Path to CAMUS dataset directory (database_nifti parent)",
        default="/mnt/data/Dataset/UltraSound/CAMUS",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        help="Output directory for SAM2 format dataset",
        default=f"/mnt/data/Dataset/SaUS/{DATASET_NAME}",
    )
    parser.add_argument(
        "--save-viz",
        action="store_true",
        help="Save visualization images with overlaid masks",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="Frames per second for the output video metadata",
    )
    parser.add_argument(
        "--max-patients",
        type=int,
        default=None,
        help="Maximum number of patients to process (for debugging)",
    )
    return parser.parse_args()


def extract_slices(file_path: str) -> np.ndarray:
    """Extract slices from a NIfTI file.
    
    Args:
        file_path: Path to the .nii or .nii.gz file.
        
    Returns:
        3D numpy array of slices (num_slices, height, width).
    """
    try:
        import SimpleITK as sitk
    except ImportError:
        raise ImportError(
            "SimpleITK is required for NIfTI file processing. "
            "Install with: pip install SimpleITK"
        )
    
    sitk_img = sitk.ReadImage(file_path)
    data = sitk.GetArrayFromImage(sitk_img)
    return data


def gray2rgb(gray: np.ndarray) -> np.ndarray:
    """Convert grayscale image to RGB."""
    if gray.ndim == 2:
        return np.stack([gray, gray, gray], axis=-1)
    elif gray.ndim == 3 and gray.shape[-1] == 1:
        return np.repeat(gray, 3, axis=-1)
    return gray


def process_patient(
    patient_dir: Path,
    patient_id: str,
    annotator: SAM2VideoAnnotator,
) -> int:
    """Process a single patient's data.
    
    Args:
        patient_dir: Path to patient directory.
        patient_id: Patient identifier (e.g., "patient0001").
        annotator: SAM2VideoAnnotator instance.
        
    Returns:
        Number of videos processed (0, 1, or 2).
    """
    count = 0
    
    # Process each view (2CH and 4CH)
    for view_name in ["2CH", "4CH"]:
        img_path = patient_dir / f"{patient_id}_{view_name}_half_sequence.nii"
        mask_path = patient_dir / f"{patient_id}_{view_name}_half_sequence_gt.nii"
        
        if not img_path.exists():
            logging.warning(f"Missing image file: {img_path}")
            continue
        if not mask_path.exists():
            logging.warning(f"Missing mask file: {mask_path}")
            continue
        
        try:
            # Load image and mask volumes
            images = extract_slices(str(img_path))  # float32, 0-255
            masks = extract_slices(str(mask_path))  # float32, 0-3
            
            # Convert to appropriate types
            images = images.astype(np.uint8)
            masks = masks.astype(np.uint8)
            
            # Clean up masks - remove annotations outside ultrasound FOV
            # (where image is black/zero)
            invalid_mask = images == 0
            masks[invalid_mask] = 0
            
            # Convert grayscale images to RGB
            images_rgb = np.stack([gray2rgb(img) for img in images], axis=0)
            
            # Skip if no valid annotations
            if not np.any(masks > 0):
                logging.debug(f"No annotations in {patient_id}/{view_name}")
                continue
            
            # Generate video name
            video_name = f"{patient_id}_{view_name}"
            
            # Add as video sequence
            annotator.add_video(
                images_rgb,
                masks,
                video_name=video_name,
                metadata={
                    "source_dataset": DATASET_NAME,
                    "patient_id": patient_id,
                    "view": view_name,
                    "num_frames": len(images),
                },
            )
            
            count += 1
            
        except Exception as e:
            logging.error(f"Error processing {patient_id}/{view_name}: {e}")
            continue
    
    return count


def main() -> None:
    args = parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    save_path = Path(args.save_dir)
    mkdir_or_exist(save_path)
    
    data_path = Path(args.path)
    use_temp_dir = data_path.suffix.lower() == ".zip"
    
    logging.info(f"Converting {DATASET_NAME} to SAM2 format (SA-V)")
    logging.info(f"Input: {data_path}")
    logging.info(f"Output: {save_path}")
    
    # Context manager for temp directory
    if use_temp_dir:
        import tempfile
        import shutil
        temp_context = tempfile.TemporaryDirectory()
    else:
        from contextlib import nullcontext
        temp_context = nullcontext(str(data_path))
    
    with temp_context as temp_dir:
        if use_temp_dir:
            logging.info(f"Extracting {data_path} to {temp_dir}")
            import shutil
            shutil.unpack_archive(str(data_path), temp_dir)
            base_dir = Path(temp_dir)
        else:
            base_dir = data_path
        
        # Locate database_nifti directory
        if (base_dir / "database_nifti").exists():
            data_dir = base_dir / "database_nifti"
        elif (base_dir / "CAMUS" / "database_nifti").exists():
            data_dir = base_dir / "CAMUS" / "database_nifti"
        else:
            data_dir = base_dir
        
        # Initialize annotator
        annotator = SAM2VideoAnnotator(
            out_dir=save_path,
            dataset_name=DATASET_NAME,
            categories=CATEGORIES,
            fps=args.fps,
            save_visualization=args.save_viz,
            video_environment="ultrasound",
            video_split="train",
        )
        
        # Get all patient directories
        patient_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("patient")])
        
        if not patient_dirs:
            logging.error(f"No patient directories found in {data_dir}")
            return
        
        logging.info(f"Found {len(patient_dirs)} patients")
        
        if args.max_patients:
            patient_dirs = patient_dirs[:args.max_patients]
            logging.info(f"Processing first {args.max_patients} patients only")
        
        patient_count = 0
        video_count = 0
        
        for idx, patient_dir in enumerate(patient_dirs):
            patient_id = patient_dir.name
            videos = process_patient(patient_dir, patient_id, annotator)
            
            if videos > 0:
                patient_count += 1
                video_count += videos
            
            if (idx + 1) % 50 == 0:
                logging.info(f"Processed {idx + 1}/{len(patient_dirs)} patients, {video_count} videos...")
        
        logging.info(f"Total: {patient_count} patients, {video_count} videos")
    
    # Finalize (outside with block so temp dir can be cleaned up)
    stats = annotator.finalize()
    
    logging.info("=" * 60)
    logging.info(f"Conversion complete!")
    logging.info(f"  Patients processed: {patient_count}")
    logging.info(f"  Videos created: {video_count}")
    logging.info(f"  Total frames: {stats['total_frames']}")
    logging.info(f"  Total annotations: {stats['total_annotations']}")
    logging.info(f"  Output directory: {save_path}")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
