"""ASUS (Automated Spine Ultrasound Segmentation) dataset preprocessing.

Dataset: Spine ultrasound segmentation dataset (volumetric tracked ultrasound)
Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC7654705/

The dataset contains tracked ultrasound scans of the spine for scoliosis
visualization and measurement. Each patient's data is a 3D volume where
consecutive slices represent spatial positions along the spine axis.

The spine masks contain multiple disconnected vertebrae per slice, each
treated as a separate object instance. 3D connected component labeling
is used to assign consistent instance IDs to vertebrae spanning multiple
adjacent slices.

Data format:
- q{idx}_ultrasound.npy: Ultrasound slices, shape (N, 128, 128, 1), float16 in [0, 1]
- q{idx}_segmentation.npy: Binary spine masks, shape (N, 128, 128, 1), uint8 with values [0, 1]

Output: SAM2-compatible format (SA-V by default, each patient volume as one video)
"""

import argparse
import logging
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import ndimage

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ..utils.sam2_annotator import SAM2VideoAnnotator
from ..utils.image import imflip

DATASET_NAME = "ASUS"

# Category definition
# Each vertebra in the spine is treated as a separate instance of the same category
CATEGORIES = [
    {"supercategory": "spine", "id": 1, "name": "spine"},
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=f"Convert {DATASET_NAME} to SAM2 format"
    )
    parser.add_argument(
        "--path",
        type=str,
        help="Path to the dataset (zip file or extracted folder)",
        default="/mnt/data/Dataset/UltraSound/Raw/ASUS.zip",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        help="Output directory for SAM2 format dataset",
        default=f"/mnt/data/Dataset/SaUS/{DATASET_NAME}",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["sa-v", "sa-1b"],
        default="sa-v",
        help="Output format: sa-v (video style) or sa-1b (image style)",
    )
    parser.add_argument("--min-mask-pixels", type=int, default=10, help="Minimum mask area")
    parser.add_argument(
        "--save-viz",
        action="store_true",
        help="Save visualization images with overlaid masks",
    )
    parser.add_argument(
        "--flip-diagonal",
        action="store_true",
        default=True,
        help="Apply diagonal flip transformation (original code behavior)",
    )
    parser.add_argument(
        "--max-patients",
        type=int,
        default=None,
        help="Maximum number of patients to process (for debugging)",
    )
    parser.add_argument(
        "--skip-empty",
        action="store_true",
        default=True,
        help="Skip frames with empty masks",
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        help="Compress output dataset to zip file",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete uncompressed output after zipping",
    )
    return parser.parse_args()


def collect_npy_files(data_dir: Path) -> dict:
    """Collect and organize .npy files by patient ID.
    
    Returns:
        Dict mapping patient_id -> {"img": path, "mask": path}
    """
    mapping = defaultdict(lambda: defaultdict(str))
    
    for npy_file in data_dir.rglob("*.npy"):
        filename = npy_file.name.lower()
        # Extract patient ID: q000_ultrasound.npy -> q000
        patient_id = filename.split("_")[0]
        
        if "segmentation" in filename:
            mapping[patient_id]["mask"] = str(npy_file)
        elif "ultrasound" in filename:
            mapping[patient_id]["img"] = str(npy_file)
    
    return dict(mapping)


def _prepare_volume(
    images: np.ndarray,
    masks: np.ndarray,
    flip_diagonal: bool = True,
    skip_empty: bool = True,
) -> tuple:
    """Preprocess a patient volume: flip, filter empty slices, 3D instance labeling.
    
    Uses 3D connected component labeling so vertebrae spanning multiple
    adjacent slices receive consistent instance IDs across the volume.
    
    Args:
        images: Image array (N, H, W, 1) or (N, H, W, C)
        masks: Mask array (N, H, W, 1) or (N, H, W)
        flip_diagonal: Whether to apply diagonal flip
        skip_empty: Whether to discard slices with empty masks
    
    Returns:
        (frames, instance_masks) — both as (M, H, W, ...) numpy arrays
        with M <= N after empty-slice filtering.  frames are uint8 RGB,
        instance_masks have per-voxel integer instance IDs (0 = background).
        Returns (None, None) if no valid slices remain.
    """
    n_slices = images.shape[0]
    
    # --- 1. Squeeze and flip all masks first (needed for 3D labeling) ---
    masks_2d = np.empty((n_slices,) + masks.shape[1:3], dtype=masks.dtype)
    imgs_proc = []
    
    for i in range(n_slices):
        mask = masks[i]
        img = images[i]
        
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)
        
        if flip_diagonal:
            img = imflip(img, direction="diagonal")
            mask = imflip(mask, direction="diagonal")
        
        # Convert image to uint8 RGB
        if img.ndim == 3 and img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        if img.dtype != np.uint8:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        
        masks_2d[i] = mask
        imgs_proc.append(img)
    
    imgs_proc = np.stack(imgs_proc, axis=0)
    
    # --- 2. 3D connected component labeling over the full volume ---
    # This gives each vertebra a consistent ID across adjacent slices.
    labeled_vol, _ = ndimage.label(masks_2d > 0)
    
    # --- 3. Filter out empty slices ---
    if skip_empty:
        keep = np.array([np.any(labeled_vol[i] != 0) for i in range(n_slices)])
        if not np.any(keep):
            return None, None
        imgs_proc = imgs_proc[keep]
        labeled_vol = labeled_vol[keep]
    
    # Ensure IDs fit in uint16 (max 65535 instances per volume, typically ~400)
    return imgs_proc, labeled_vol.astype(np.uint16)


def process_patient_volume(
    annotator: SAM2VideoAnnotator,
    patient_id: str,
    images: np.ndarray,
    masks: np.ndarray,
    flip_diagonal: bool = True,
    skip_empty: bool = True,
) -> int:
    """Process a patient's volumetric scan as a video (sa-v format).
    
    Each patient volume is treated as a single video.  3D connected
    component labeling assigns consistent instance IDs to vertebrae
    across adjacent slices so SAM2 can track them.
    
    Args:
        annotator: SAM2VideoAnnotator instance
        patient_id: Patient identifier
        images: Image array (N, H, W, 1)
        masks: Mask array (N, H, W, 1)
        flip_diagonal: Whether to apply diagonal flip
        skip_empty: Whether to skip slices with empty masks
    
    Returns:
        Number of slices processed
    """
    frames, instance_masks = _prepare_volume(
        images, masks, flip_diagonal=flip_diagonal, skip_empty=skip_empty
    )
    
    if frames is None:
        return 0
    
    # Map every instance ID to category 1 ("spine")
    all_ids = set(int(x) for x in np.unique(instance_masks) if x != 0)
    inst_to_cat = {inst_id: 1 for inst_id in all_ids}
    
    annotator.add_video(
        frames,
        instance_masks,
        video_name=patient_id,
        metadata={"patient_id": patient_id, "dataset": DATASET_NAME},
        instance_to_category=inst_to_cat,
        default_category_id=1,
    )
    
    return frames.shape[0]


def _split_mask_into_instances_2d(mask: np.ndarray) -> np.ndarray:
    """Split a single 2D binary mask into instance-labeled mask.
    
    Used only for the sa-1b (image) export path.
    
    Args:
        mask: Binary mask (H, W) with values {0, 1}.
    
    Returns:
        Instance-labeled mask (H, W), background = 0.
    """
    if not np.any(mask):
        return mask.astype(np.uint8)
    labeled, _ = ndimage.label(mask > 0)
    return labeled.astype(np.uint8)


def process_patient_as_images(
    annotator: SAM2VideoAnnotator,
    patient_id: str,
    images: np.ndarray,
    masks: np.ndarray,
    flip_diagonal: bool = True,
    skip_empty: bool = True,
    output_format: str = "sa-1b",
) -> int:
    """Process a single patient's data as individual images.
    
    Args:
        annotator: SAM2VideoAnnotator instance
        patient_id: Patient identifier  
        images: Image array (N, H, W, 1) or (N, H, W, C)
        masks: Mask array (N, H, W, 1)
        flip_diagonal: Whether to apply diagonal flip
        skip_empty: Whether to skip frames with empty masks
        output_format: Output format ("sa-v" or "sa-1b")
    
    Returns:
        Number of images processed
    """
    count = 0
    
    for i in range(images.shape[0]):
        img = images[i]
        mask = masks[i]
        
        # Squeeze mask if needed
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)
        
        # Skip empty masks if requested
        if skip_empty and not np.any(mask != 0):
            continue
        
        # Apply diagonal flip if requested
        if flip_diagonal:
            img = imflip(img, direction="diagonal")
            mask = imflip(mask, direction="diagonal")
        
        # Split binary mask into separate instances (one per connected component)
        # Each vertebra in the spine becomes its own object instance
        instance_mask = _split_mask_into_instances_2d(mask)
        
        # Convert image to uint8 RGB
        if img.ndim == 3 and img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        if img.dtype != np.uint8:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        
        # Generate unique image name
        image_name = f"{patient_id}_{i:04d}"
        
        # Map all instance IDs to category 1 ("spine") since they are all spine parts
        object_ids = [int(x) for x in np.unique(instance_mask) if x != 0]
        instance_to_category = {obj_id: 1 for obj_id in object_ids}
        
        # Add image with instance-level mask
        annotator.add_2d(
            img,
            instance_mask,
            video_name=image_name,
            format=output_format,
            instance_to_category=instance_to_category,
            metadata={"patient_id": patient_id, "frame_idx": i, "dataset": DATASET_NAME},
        )
        count += 1
    
    return count


def main():
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    
    save_path = Path(args.save_dir)
    data_path = Path(args.path)
    
    # Determine if we need to extract from zip
    use_temp_dir = data_path.suffix.lower() == ".zip"
    
    # Create annotator
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
    logger.info(f"Save visualization: {args.save_viz}")

    # Context manager for temp directory
    if use_temp_dir:
        temp_context = tempfile.TemporaryDirectory()
    else:
        # Use nullcontext if data is already extracted
        from contextlib import nullcontext
        temp_context = nullcontext(str(data_path))
    
    with temp_context as temp_dir:
        if use_temp_dir:
            logger.info(f"Extracting {data_path} to {temp_dir}")
            shutil.unpack_archive(str(data_path), temp_dir)
            logger.info("Extraction complete")
            data_dir = Path(temp_dir)
        else:
            data_dir = data_path
        
        # Collect all .npy files
        file_mapping = collect_npy_files(data_dir)
        logger.info(f"Found {len(file_mapping)} patients")
        
        if not file_mapping:
            logger.error("No data files found!")
            return
        
        # Sort by patient ID for reproducibility
        patient_ids = sorted(file_mapping.keys())
        
        # Limit patients if requested
        if args.max_patients:
            patient_ids = patient_ids[:args.max_patients]
            logger.info(f"Processing only {len(patient_ids)} patients")
        
        total_images = 0
        
        for idx, patient_id in enumerate(patient_ids):
            data = file_mapping[patient_id]
            
            if not data["img"] or not data["mask"]:
                logger.warning(f"Missing data for patient {patient_id}, skipping")
                continue
            
            # Load data
            images = np.load(data["img"])
            masks = np.load(data["mask"])
            
            if images.shape[0] != masks.shape[0]:
                logger.warning(
                    f"Shape mismatch for patient {patient_id}: "
                    f"images {images.shape} vs masks {masks.shape}"
                )
                continue
            
            # Process based on output format
            if args.format == "sa-v":
                count = process_patient_volume(
                    annotator,
                    patient_id,
                    images,
                    masks,
                    flip_diagonal=args.flip_diagonal,
                    skip_empty=args.skip_empty,
                )
            else:
                count = process_patient_as_images(
                    annotator,
                    patient_id,
                    images,
                    masks,
                    flip_diagonal=args.flip_diagonal,
                    skip_empty=args.skip_empty,
                    output_format=args.format,
                )
            
            total_images += count
            
            if (idx + 1) % 1 == 0:
                logger.info(
                    f"Processed patient {patient_id} ({idx + 1}/{len(patient_ids)}): "
                    f"{count} frames, total: {total_images}"
                )
        
        logger.info(f"Total images processed: {total_images}")
    
    # Finalize dataset
    stats = annotator.finalize()
    logger.info(f"Dataset stats: {stats}")
    
    # Optional: zip output
    if args.zip:
        zip_path = save_path.with_suffix(".zip")
        logger.info(f"Compressing to {zip_path}")
        shutil.make_archive(str(save_path), "zip", save_path)
        
        if args.delete:
            logger.info(f"Deleting uncompressed folder: {save_path}")
            shutil.rmtree(save_path)
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
