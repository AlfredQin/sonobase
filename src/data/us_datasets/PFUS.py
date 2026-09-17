"""PFUS (Pelvic Floor Ultrasound) dataset preprocessing.

Dataset: Transperineal pelvic floor ultrasound with multi-structure annotations
Source: https://github.com/cirmuw/PFUS

Data format:
- data/PXXX/frame_XXX.png: Ultrasound frames
- data/PXXX/frame_XXX.json: Polygon annotations with multiple structures

Categories:
- Pubis (1), Urethra (2), Bladder (3), Uterus (4), Vagina (5),
- Anus (6), Rectum (7), Levator ani muscle (8)

Output: SAM2-compatible format (SA-V for videos, SA-1B for images)
"""

import argparse
import json
import logging
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from PIL import Image, ImageDraw

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ..utils.sam2_annotator import SAM2VideoAnnotator

DATASET_NAME = "PFUS"

CATEGORIES = [
    {"supercategory": "bone", "id": 1, "name": "pubis"},
    {"supercategory": "organ", "id": 2, "name": "urethra"},
    {"supercategory": "organ", "id": 3, "name": "bladder"},
    {"supercategory": "organ", "id": 4, "name": "uterus"},
    {"supercategory": "organ", "id": 5, "name": "vagina"},
    {"supercategory": "organ", "id": 6, "name": "anus"},
    {"supercategory": "organ", "id": 7, "name": "rectum"},
    {"supercategory": "muscle", "id": 8, "name": "levator_ani_muscle"},
]

# Label name to category ID mapping
LABEL_TO_ID = {
    "Pubis": 1,
    "pubis": 1,
    "Urethra": 2,
    "urethra": 2,
    "Bladder": 3,
    "bladder": 3,
    "Uterus": 4,
    "uterus": 4,
    "Vagina": 5,
    "vagina": 5,
    "Anus": 6,
    "anus": 6,
    "Rectum": 7,
    "rectum": 7,
    "Levator ani muscle": 8,
    "levator_ani_muscle": 8,
    "levator ani muscle": 8,
    "Levator_ani_muscle": 8,
}


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
        help="Output format (sa-v treats each patient as video)",
    )
    parser.add_argument("--save-viz", action="store_true")
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--zip", action="store_true")
    parser.add_argument("--delete", action="store_true")
    return parser.parse_args()


def polygon_to_mask(polygon: List[List[float]], height: int, width: int, category_id: int) -> np.ndarray:
    """Convert polygon points to a filled mask."""
    mask = Image.new("L", (width, height), 0)
    points = [(p[0], p[1]) for p in polygon]
    if len(points) >= 3:
        ImageDraw.Draw(mask).polygon(points, outline=category_id, fill=category_id)
    return np.array(mask)


def collect_patients(data_dir: Path) -> Dict[str, List[Dict]]:
    """Collect frame data grouped by patient ID."""
    patients = defaultdict(list)

    # Handle nested folder structure
    if (data_dir / "PFUS1").exists():
        data_dir = data_dir / "PFUS1"
    elif (data_dir / "pfus1").exists():
        data_dir = data_dir / "pfus1"

    if (data_dir / "data").exists():
        data_dir = data_dir / "data"

    # Find all patient folders
    for patient_dir in sorted(data_dir.iterdir()):
        if not patient_dir.is_dir():
            continue
        if not patient_dir.name.startswith("P"):
            continue

        patient_id = patient_dir.name

        # Find all frames for this patient
        for img_file in sorted(patient_dir.glob("*.png")):
            frame_id = img_file.stem  # e.g., frame_000
            json_file = patient_dir / f"{frame_id}.json"

            if json_file.exists():
                patients[patient_id].append({
                    "image_path": img_file,
                    "json_path": json_file,
                    "frame_id": frame_id,
                })

    return patients


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

        patients = collect_patients(data_dir)
        logger.info(f"Found {len(patients)} patients")

        if args.max_patients:
            patient_ids = list(patients.keys())[:args.max_patients]
            patients = {k: patients[k] for k in patient_ids}

        processed = 0
        total_frames = 0

        for patient_id, frames in patients.items():
            video_name = f"{patient_id}"

            if args.format == "sa-v":
                # Process as video
                processed_frames = []
                processed_masks = []

                for frame_data in frames:
                    image = cv2.imread(str(frame_data["image_path"]))
                    if image is None:
                        continue
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    h, w = image.shape[:2]

                    # Load JSON annotation
                    try:
                        with open(frame_data["json_path"], "r") as f:
                            annotations = json.load(f)
                    except Exception as e:
                        logger.warning(f"Error reading {frame_data['json_path']}: {e}")
                        continue

                    # Create mask from polygons
                    mask_combined = np.zeros((h, w), dtype=np.uint8)

                    for ann in annotations:
                        label = ann.get("label", "")
                        polygon = ann.get("pol", [])

                        if label not in LABEL_TO_ID:
                            continue

                        category_id = LABEL_TO_ID[label]
                        mask = polygon_to_mask(polygon, h, w, category_id)
                        # Later categories take precedence
                        mask_combined[mask > 0] = mask[mask > 0]

                    processed_frames.append(image_rgb)
                    processed_masks.append(mask_combined)

                if processed_frames:
                    # Ensure all frames have the same shape (use first frame as reference)
                    ref_h, ref_w = processed_frames[0].shape[:2]
                    resized_frames = []
                    resized_masks = []
                    for frame, mask in zip(processed_frames, processed_masks):
                        if frame.shape[:2] != (ref_h, ref_w):
                            frame = cv2.resize(frame, (ref_w, ref_h), interpolation=cv2.INTER_LINEAR)
                            mask = cv2.resize(mask, (ref_w, ref_h), interpolation=cv2.INTER_NEAREST)
                        resized_frames.append(frame)
                        resized_masks.append(mask)
                    
                    frames_arr = np.stack(resized_frames, axis=0)
                    masks_arr = np.stack(resized_masks, axis=0)

                    # Only add if at least one frame has a mask
                    if np.any(masks_arr > 0):
                        annotator.add_video(
                            frames_arr,
                            masks_arr,
                            video_name=video_name,
                            metadata={
                                "patient_id": patient_id,
                                "num_frames": len(processed_frames),
                                "dataset": DATASET_NAME,
                            },
                        )
                        total_frames += len(processed_frames)
                        processed += 1

            else:
                # Process as individual images (SA-1B)
                for frame_data in frames:
                    image = cv2.imread(str(frame_data["image_path"]))
                    if image is None:
                        continue
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    h, w = image.shape[:2]

                    try:
                        with open(frame_data["json_path"], "r") as f:
                            annotations = json.load(f)
                    except Exception as e:
                        continue

                    mask_combined = np.zeros((h, w), dtype=np.uint8)

                    for ann in annotations:
                        label = ann.get("label", "")
                        polygon = ann.get("pol", [])

                        if label not in LABEL_TO_ID:
                            continue

                        category_id = LABEL_TO_ID[label]
                        mask = polygon_to_mask(polygon, h, w, category_id)
                        mask_combined[mask > 0] = mask[mask > 0]

                    if not np.any(mask_combined > 0):
                        continue

                    frame_name = f"{patient_id}_{frame_data['frame_id']}"
                    annotator.add_2d(
                        image_rgb,
                        mask_combined,
                        video_name=frame_name,
                        format="sa-1b",
                        metadata={
                            "patient_id": patient_id,
                            "frame_id": frame_data["frame_id"],
                            "dataset": DATASET_NAME,
                        },
                    )
                    total_frames += 1

                processed += 1

            if processed % 20 == 0:
                logger.info(f"Processed {processed}/{len(patients)} patients ({total_frames} frames)")

        logger.info(f"Total: {processed} patients, {total_frames} frames")

    stats = annotator.finalize()
    logger.info(f"Dataset stats: {stats}")

    if args.zip:
        shutil.make_archive(str(save_path), "zip", save_path)
        if args.delete:
            shutil.rmtree(save_path)

    logger.info("Done!")


if __name__ == "__main__":
    main()
