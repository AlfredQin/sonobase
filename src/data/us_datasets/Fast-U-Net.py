"""Fast-U-Net (AC/HC Measurement) dataset preprocessing.

Dataset: Fetal abdominal/head circumference ultrasound segmentation
Source: https://github.com/vahidashkani/Fast-U-Net

Data format (RAR archives):
- Dataset/AC/AC_image{1,2}.rar: Abdominal circumference images
- Dataset/AC/AC_mask.rar: AC masks
- Dataset/HC/HC_image{1,2,3}.rar: Head circumference images
- Dataset/HC/HC_mask.rar: HC masks

Note: Requires 'unrar' command-line tool

Output: SAM2-compatible format (SA-1B style)
"""

import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from scipy import ndimage
from skimage.morphology import erosion, dilation, square

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ..utils.sam2_annotator import SAM2VideoAnnotator

DATASET_NAME = "Fast-U-Net"

CATEGORIES = [
    {"supercategory": "measurement", "id": 1, "name": "AC_HC"},
]


def parse_args():
    parser = argparse.ArgumentParser(description=f"Convert {DATASET_NAME} to SAM2 format")
    parser.add_argument(
        "--path",
        type=str,
        default=f"/mnt/data/Dataset/UltraSound/Raw/{DATASET_NAME}.zip",
        help="Path to dataset (zip file or folder containing Dataset/AC and Dataset/HC)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=f"/mnt/data/Dataset/SaUS/{DATASET_NAME}",
        help="Output directory",
    )
    parser.add_argument("--format", type=str, choices=["sa-v", "sa-1b"], default="sa-1b")
    parser.add_argument("--save-viz", action="store_true")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--zip", action="store_true")
    parser.add_argument("--delete", action="store_true")
    return parser.parse_args()


def rgb2gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def unpack_rar(rar_path: Path, dest_path: Path) -> bool:
    """Extract RAR archive using unrar command."""
    try:
        subprocess.run(
            ["unrar", "x", "-o+", str(rar_path), str(dest_path)],
            check=True,
            capture_output=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logging.warning(f"Failed to extract {rar_path}: {e}")
        return False


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component in binary mask."""
    labeled, num_features = ndimage.label(mask)
    if num_features == 0:
        return mask
    
    component_sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
    largest_component = np.argmax(component_sizes) + 1
    
    result = np.zeros_like(mask)
    result[labeled == largest_component] = 1
    return result


def process_ac_data(data_dir: Path, temp_dir: Path, logger) -> List[Dict]:
    """Extract and collect AC samples."""
    samples = []
    
    ac_dir = data_dir / "Dataset" / "AC"
    if not ac_dir.exists():
        logger.warning(f"AC directory not found: {ac_dir}")
        return samples
    
    # Extract RAR files
    for rar_name in ["AC_image1.rar", "AC_image2.rar", "AC_mask.rar"]:
        rar_path = ac_dir / rar_name
        if rar_path.exists():
            logger.info(f"Extracting {rar_name}")
            unpack_rar(rar_path, temp_dir)
    
    # Collect samples
    mapping = defaultdict(dict)
    for png_file in temp_dir.rglob("*.png"):
        idx = png_file.stem
        modality = "mask" if "mask" in str(png_file).lower() else "img"
        mapping[idx][modality] = png_file
    
    for idx, data in mapping.items():
        if "img" in data and "mask" in data:
            samples.append({
                "image_path": data["img"],
                "mask_path": data["mask"],
                "image_id": f"AC_{idx}",
                "category": "AC",
            })
    
    return samples


def process_hc_data(data_dir: Path, temp_dir: Path, logger) -> List[Dict]:
    """Extract and collect HC samples."""
    samples = []
    
    hc_dir = data_dir / "Dataset" / "HC"
    if not hc_dir.exists():
        logger.warning(f"HC directory not found: {hc_dir}")
        return samples
    
    # Extract RAR files
    for rar_name in ["HC_image1.rar", "HC_image2.rar", "HC_image3.rar", "HC_mask.rar"]:
        rar_path = hc_dir / rar_name
        if rar_path.exists():
            logger.info(f"Extracting {rar_name}")
            unpack_rar(rar_path, temp_dir)
    
    # Collect samples
    mapping = defaultdict(dict)
    for png_file in temp_dir.rglob("*.png"):
        filename = png_file.stem
        # HC files have format like "HC_001_image" or "HC_001_mask"
        parts = filename.split("_")[:2]
        idx = "_".join(parts)
        modality = "mask" if "mask" in str(png_file).lower() else "img"
        mapping[idx][modality] = png_file
    
    for idx, data in mapping.items():
        if "img" in data and "mask" in data:
            samples.append({
                "image_path": data["img"],
                "mask_path": data["mask"],
                "image_id": idx,
                "category": "HC",
            })
    
    return samples


def main():
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

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

        # Find the Dataset folder (might be nested like Fast-U-Net/Dataset/)
        dataset_dir = None
        for candidate in [
            data_dir / "Dataset",
            data_dir / "Fast-U-Net" / "Dataset",
            data_dir / "Fast_U_Net" / "Dataset",
        ]:
            if candidate.exists():
                dataset_dir = candidate
                break
        
        # Also search recursively if not found
        if dataset_dir is None:
            for path in data_dir.rglob("Dataset"):
                if path.is_dir() and (path / "AC").exists():
                    dataset_dir = path
                    break
        
        if dataset_dir is None:
            logger.error(f"Dataset folder not found in {data_dir}")
            logger.info(f"Contents of {data_dir}: {list(data_dir.iterdir())}")
            return
        
        logger.info(f"Found Dataset folder: {dataset_dir}")

        # Use another temp dir for extracted RAR files
        with tempfile.TemporaryDirectory() as rar_temp:
            rar_temp_path = Path(rar_temp)
            
            # Extract all RAR files
            for subdir in ["AC", "HC"]:
                src_dir = dataset_dir / subdir
                if src_dir.exists():
                    rar_files = list(src_dir.glob("*.rar"))
                    logger.info(f"Found {len(rar_files)} RAR files in {subdir}")
                    for rar_file in rar_files:
                        logger.info(f"Extracting {rar_file.name}")
                        unpack_rar(rar_file, rar_temp_path)
                else:
                    logger.warning(f"Subdir not found: {src_dir}")
            
            # Collect all samples
            all_png_files = list(rar_temp_path.rglob("*.png"))
            logger.info(f"Found {len(all_png_files)} PNG files after RAR extraction")
            
            if len(all_png_files) == 0:
                logger.warning(f"No PNG files found in {rar_temp_path}")
                logger.info(f"Contents of rar_temp: {list(rar_temp_path.iterdir())}")
            
            mapping = defaultdict(lambda: defaultdict(dict))
            for png_file in all_png_files:
                # Determine category (AC or HC)
                category = "AC" if "AC" in str(png_file) or "ac" in str(png_file) else "HC"
                idx = png_file.stem
                modality = "mask" if "mask" in str(png_file).lower() else "img"
                mapping[category][idx][modality] = png_file
            
            all_samples = []
            for category, cat_mapping in mapping.items():
                for idx, data in cat_mapping.items():
                    if "img" in data and "mask" in data:
                        all_samples.append({
                            "image_path": data["img"],
                            "mask_path": data["mask"],
                            "image_id": f"{category}_{idx}",
                            "category": category,
                        })
            
            logger.info(f"Total matched samples (with both img and mask): {len(all_samples)}")
            
            if args.max_images:
                all_samples = all_samples[:args.max_images]
            
            processed = 0
            for idx, sample in enumerate(all_samples):
                image = cv2.imread(str(sample["image_path"]))
                if image is None:
                    continue
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                h, w = image.shape[:2]
                
                mask = rgb2gray(cv2.imread(str(sample["mask_path"])))
                mask_binary = (mask // 255).astype(np.uint8)
                
                if not np.any(mask_binary > 0):
                    continue
                
                # Clean up mask: erosion -> keep largest -> dilation
                mask_binary = erosion(mask_binary, square(12))
                mask_binary = keep_largest_component(mask_binary)
                mask_binary = dilation(mask_binary, square(12))
                
                if not np.any(mask_binary > 0):
                    continue
                
                image_name = f"{sample['image_id']}"
                annotator.add_2d(
                    image_rgb,
                    mask_binary,
                    video_name=image_name,
                    format=args.format,
                    metadata={
                        "image_id": sample["image_id"],
                        "category": sample["category"],
                        "dataset": DATASET_NAME,
                    },
                )
                processed += 1
                
                if (idx + 1) % 100 == 0:
                    logger.info(f"Processed {idx + 1}/{len(all_samples)}")
            
            logger.info(f"Total processed: {processed}")

    stats = annotator.finalize()
    logger.info(f"Dataset stats: {stats}")

    if args.zip:
        shutil.make_archive(str(save_path), "zip", save_path)
        if args.delete:
            shutil.rmtree(save_path)

    logger.info("Done!")


if __name__ == "__main__":
    main()
