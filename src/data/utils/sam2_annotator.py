"""Ultrasound dataset exporter to SAM2 video format.

This module provides a clean interface for converting 2D images, 3D volumes,
and videos into SAM2-compatible datasets for training and evaluation.

SAM2 Video Format (SA-V compatible):
- Images folder: {video_name}/{frame_id:05d}.jpg
- GT folder: {video_name}_manual.json containing RLE-encoded masks per frame

JSON structure (SA-V style):
{
    # Video metadata
    "video_id": "video_name",
    "video_duration": 10.0,
    "video_frame_count": 240,
    "video_height": 480,
    "video_width": 640,
    "video_resolution": 307200,
    "video_environment": "ultrasound",
    "video_split": "train",
    
    # Masklet data (core annotation)
    "masklet": [  # list of frames
        [rle_mask_obj1, rle_mask_obj2, ...],  # frame 0
        [rle_mask_obj1, rle_mask_obj2, ...],  # frame 1
        ...
    ],
    
    # Masklet metadata
    "masklet_id": [0, 1, 2],
    "masklet_size_rel": [0.05, 0.1, 0.02],
    "masklet_size_abs": [15360, 30720, 6144],
    "masklet_size_bucket": ["medium", "large", "small"],
    "masklet_visibility_changes": [0, 2, 1],
    "masklet_first_appeared_frame": [0, 0, 10],
    "masklet_frame_count": [240, 238, 230],
    "masklet_type": ["manual", "manual", "manual"],
    "masklet_num": 3,
    
    # Additional metadata
    "fps": 24,
    "ann_every": 1,
}

Key behavior:
- 2D images: treated as single-frame videos
- 3D volumes: each slice becomes a frame (treated as video)
- Videos: each frame is exported with consistent object tracking

Masks are expected as label maps (0=background, 1..K=object ids). Each unique
non-zero label becomes a separate tracked object across frames.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


def _try_import_cv2():
    try:
        import cv2
        return cv2
    except ImportError:
        return None


def _try_import_pil():
    try:
        from PIL import Image
        return Image
    except ImportError:
        return None


def _encode_rle(binary_mask: np.ndarray) -> Dict[str, Any]:
    """Encode binary mask to COCO RLE format."""
    from pycocotools import mask as mask_utils

    if binary_mask.dtype != np.uint8:
        binary_mask = binary_mask.astype(np.uint8)

    rle = mask_utils.encode(np.asfortranarray(binary_mask))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def _decode_rle(rle: Dict[str, Any]) -> np.ndarray:
    """Decode COCO RLE to binary mask."""
    from pycocotools import mask as mask_utils

    rle_copy = dict(rle)
    if isinstance(rle_copy.get("counts"), str):
        rle_copy["counts"] = rle_copy["counts"].encode("utf-8")

    mask = mask_utils.decode(rle_copy)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask.astype(np.uint8)


def _rle_area(rle: Dict[str, Any]) -> int:
    """Get area from RLE mask."""
    from pycocotools import mask as mask_utils
    
    rle_copy = dict(rle)
    if isinstance(rle_copy.get("counts"), str):
        rle_copy["counts"] = rle_copy["counts"].encode("utf-8")
    return int(mask_utils.area(rle_copy))


def _ensure_uint8_image(img: np.ndarray) -> np.ndarray:
    """Convert image to uint8 HxWx3 (RGB).

    Accepts:
    - HxW (grayscale)
    - HxWx1
    - HxWx3
    - float images in [0,1] or [0,255]
    """
    if img.ndim == 2:
        img = img[..., None]
    if img.ndim != 3:
        raise ValueError(f"Expected 2D/3D image array, got shape {img.shape}")

    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    elif img.shape[2] != 3:
        raise ValueError(f"Expected channel dim 1 or 3, got shape {img.shape}")

    if img.dtype == np.uint8:
        return img

    img_f = img.astype(np.float32)
    if np.nanmax(img_f) <= 1.5:
        img_f = img_f * 255.0

    img_f = np.clip(img_f, 0.0, 255.0)
    return img_f.astype(np.uint8)


def _save_image(path: Path, image_uint8_rgb: np.ndarray, format: str = "jpg") -> None:
    """Save image to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)

    PIL_Image = _try_import_pil()
    if PIL_Image is not None:
        quality = 95 if format.lower() in ("jpg", "jpeg") else None
        kwargs = {"quality": quality} if quality else {}
        PIL_Image.fromarray(image_uint8_rgb).save(path, **kwargs)
        return

    cv2 = _try_import_cv2()
    if cv2 is not None:
        bgr = image_uint8_rgb[..., ::-1]
        cv2.imwrite(str(path), bgr)
        return

    raise RuntimeError("Neither PIL nor cv2 available for image saving")


def _overlay_mask(
    image_uint8_rgb: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int],
    alpha: float = 0.5,
) -> np.ndarray:
    """Overlay a binary mask onto an RGB image."""
    cv2 = _try_import_cv2()
    if cv2 is None:
        # Simple numpy fallback
        result = image_uint8_rgb.copy()
        mask_bool = mask > 0
        for i, c in enumerate(color):
            result[..., i] = np.where(
                mask_bool,
                (1 - alpha) * result[..., i] + alpha * c,
                result[..., i],
            )
        return result.astype(np.uint8)

    bgr = image_uint8_rgb[..., ::-1].copy()
    overlay = bgr.copy()
    color_bgr = (color[2], color[1], color[0])
    overlay[mask > 0] = color_bgr
    blended = cv2.addWeighted(overlay, alpha, bgr, 1.0 - alpha, 0)
    return blended[..., ::-1]


def _compute_size_bucket(rel_size: float) -> str:
    """Compute size bucket from relative size (SA-V style)."""
    if rel_size < 0.01:
        return "small"
    elif rel_size < 0.1:
        return "medium"
    else:
        return "large"


def _compute_bbox_from_mask(binary_mask: np.ndarray) -> List[float]:
    """Compute bounding box [x, y, width, height] from binary mask."""
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return [0.0, 0.0, 0.0, 0.0]
    
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    return [float(x_min), float(y_min), float(x_max - x_min + 1), float(y_max - y_min + 1)]


@dataclass
class SAM2Category:
    """Category metadata for SAM2 dataset."""
    id: int
    name: str
    supercategory: str = "object"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "supercategory": self.supercategory,
        }


@dataclass
class MaskletStats:
    """Statistics for a single masklet (object) across frames."""
    obj_id: int
    obj_idx: int  # Index in the masklet list (0-based)
    total_area: float = 0.0
    frame_count: int = 0
    first_appeared_frame: int = -1
    visibility_changes: int = 0
    was_visible: bool = False
    
    # Per-category info (if categories provided)
    category_id: Optional[int] = None
    category_name: Optional[str] = None


@dataclass
class VideoSample:
    """Internal representation of a video sample being built."""
    video_name: str
    frames: List[np.ndarray] = field(default_factory=list)
    masks: List[np.ndarray] = field(default_factory=list)
    object_ids: Optional[List[int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class SAM2VideoAnnotator:
    """Export 2D / 3D / video samples to SAM2 video segmentation format (SA-V compatible).

    The SAM2 format organizes data as:
    - {out_dir}/images/{video_name}/{frame_id:05d}.jpg
    - {out_dir}/gt/{video_name}_manual.json

    The JSON format follows SA-V conventions with full metadata fields.

    Usage:
        annotator = SAM2VideoAnnotator("/path/to/output", dataset_name="my_dataset")

        # Add single 2D image (becomes 1-frame video)
        annotator.add_2d(image, mask, video_name="sample_001")

        # Add 3D volume (each slice becomes a frame)
        annotator.add_3d(volume, mask_volume, video_name="patient_001_vol")

        # Add video directly
        annotator.add_video(frames, masks, video_name="video_001")

        # Finalize and write all data
        annotator.finalize()
    """

    def __init__(
        self,
        out_dir: Union[str, Path],
        *,
        dataset_name: str = "ultrasound",
        categories: Optional[Sequence[Union[SAM2Category, Dict[str, Any]]]] = None,
        fps: int = 24,
        ann_every: int = 1,
        image_format: str = "jpg",
        save_visualization: bool = False,
        viz_alpha: float = 0.5,
        min_mask_pixels: int = 10,
        skip_empty_annotations: bool = True,
        skip_empty_frames: bool = True,
        video_environment: str = "ultrasound",
        video_split: str = "train",
        frame_id_step: int = 1,
    ):
        """Initialize the SAM2 video annotator.

        Args:
            out_dir: Root output directory for the dataset.
            dataset_name: Name of the dataset (used for logging).
            categories: Optional list of category definitions.
            fps: Frames per second metadata in the JSON.
            ann_every: Annotation frequency (1 = every frame annotated).
            image_format: Output image format ("jpg" or "png").
            save_visualization: Whether to save visualization images with overlaid masks.
            viz_alpha: Alpha blending for visualization overlays.
            min_mask_pixels: Minimum number of mask pixels to consider valid.
            skip_empty_annotations: If True, skip writing image/JSON pairs when all masks
                have fewer pixels than min_mask_pixels (SA-1B only). Default True.
            skip_empty_frames: If True, filter out frames with masks below min_mask_pixels
                threshold when writing videos/volumes (SA-V format). Default True.
            video_environment: Environment label (e.g., "ultrasound", "Indoor", "Outdoor").
            video_split: Dataset split label (e.g., "train", "val", "test").
            frame_id_step: Step between frame IDs (e.g., 4 for every 4th frame like SA-V).
        """
        self.out_dir = Path(out_dir)
        self.dataset_name = dataset_name
        self.fps = fps
        self.ann_every = ann_every
        self.image_format = image_format.lower()
        self.save_visualization = save_visualization
        self.viz_alpha = viz_alpha
        self.min_mask_pixels = min_mask_pixels
        self.skip_empty_annotations = skip_empty_annotations
        self.skip_empty_frames = skip_empty_frames
        self.video_environment = video_environment
        self.video_split = video_split
        self.frame_id_step = frame_id_step

        # Directory structure
        self.images_dir = self.out_dir / "images"
        self.gt_dir = self.out_dir / "gt"
        self.viz_dir = self.out_dir / "visualization"

        # Categories
        self._categories: List[Dict[str, Any]] = []
        self._category_id_to_name: Dict[int, str] = {}
        if categories:
            self._set_categories(categories)

        # Track all video names to avoid duplicates
        self._video_names: set = set()

        # Statistics
        self._stats = {
            "videos": 0,
            "total_frames": 0,
            "total_annotations": 0,
            "skipped_empty": 0,
        }

        # Ensure directories exist
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.gt_dir.mkdir(parents=True, exist_ok=True)
        if self.save_visualization:
            self.viz_dir.mkdir(parents=True, exist_ok=True)

    def _set_categories(
        self, categories: Sequence[Union[SAM2Category, Dict[str, Any]]]
    ) -> None:
        """Set category definitions."""
        for cat in categories:
            if isinstance(cat, SAM2Category):
                cat_dict = cat.to_dict()
            elif isinstance(cat, dict):
                cat_dict = dict(cat)
            else:
                raise TypeError(f"Unsupported category type: {type(cat)}")
            
            self._categories.append(cat_dict)
            self._category_id_to_name[cat_dict["id"]] = cat_dict.get("name", f"class_{cat_dict['id']}")

    def _generate_video_name(self, prefix: str = "video") -> str:
        """Generate a unique video name."""
        idx = len(self._video_names)
        while True:
            name = f"{prefix}_{idx:06d}"
            if name not in self._video_names:
                return name
            idx += 1

    def _get_object_ids_from_mask(self, mask: np.ndarray) -> List[int]:
        """Extract unique non-zero object IDs from a mask."""
        unique_ids = np.unique(mask)
        return sorted([int(x) for x in unique_ids if x != 0])

    def _get_consistent_object_ids(self, masks: List[np.ndarray]) -> List[int]:
        """Get all unique object IDs across all frames."""
        all_ids: set = set()
        for mask in masks:
            all_ids.update(self._get_object_ids_from_mask(mask))
        return sorted(all_ids)

    def _encode_frame_masks(
        self,
        mask: np.ndarray,
        object_ids: List[int],
    ) -> Tuple[List[Optional[Dict[str, Any]]], List[int]]:
        """Encode masks for all objects in a single frame.

        Args:
            mask: Label map (H, W) with object IDs.
            object_ids: List of all object IDs to track.

        Returns:
            Tuple of (list of RLE dicts or None, list of areas per object).
        """
        rle_masks: List[Optional[Dict[str, Any]]] = []
        areas: List[int] = []

        for obj_id in object_ids:
            binary_mask = (mask == obj_id).astype(np.uint8)
            pixel_count = np.count_nonzero(binary_mask)
            
            if pixel_count >= self.min_mask_pixels:
                rle = _encode_rle(binary_mask)
                rle_masks.append(rle)
                areas.append(pixel_count)
            else:
                rle_masks.append(None)
                areas.append(0)

        return rle_masks, areas

    def _compute_masklet_metadata(
        self,
        frame_annotations: List[List[Optional[Dict[str, Any]]]],
        object_ids: List[int],
        frame_areas: List[List[int]],
        video_resolution: int,
    ) -> Dict[str, Any]:
        """Compute SA-V style masklet metadata.
        
        Args:
            frame_annotations: Per-frame list of RLE masks per object.
            object_ids: List of object IDs.
            frame_areas: Per-frame list of areas per object.
            video_resolution: Total pixels in a frame (H * W).
            
        Returns:
            Dict with masklet metadata fields.
        """
        num_objects = len(object_ids)
        num_frames = len(frame_annotations)
        
        # Initialize stats per object
        masklet_id = list(range(num_objects))
        masklet_size_rel: List[float] = []
        masklet_size_abs: List[float] = []
        masklet_size_bucket: List[str] = []
        masklet_visibility_changes: List[int] = []
        masklet_first_appeared_frame: List[float] = []
        masklet_frame_count: List[int] = []
        masklet_type: List[str] = []
        
        for obj_idx in range(num_objects):
            # Collect areas across frames where object is visible
            obj_areas = []
            visible_frames = 0
            first_frame = -1
            visibility_changes = 0
            was_visible = False
            
            for frame_idx in range(num_frames):
                area = frame_areas[frame_idx][obj_idx]
                is_visible = frame_annotations[frame_idx][obj_idx] is not None
                
                if is_visible:
                    obj_areas.append(area)
                    visible_frames += 1
                    if first_frame < 0:
                        first_frame = frame_idx
                
                # Track visibility changes
                if is_visible != was_visible:
                    if frame_idx > 0:  # Don't count initial appearance
                        visibility_changes += 1
                was_visible = is_visible
            
            # Compute average size
            if obj_areas:
                avg_area = sum(obj_areas) / len(obj_areas)
                rel_size = avg_area / video_resolution
            else:
                avg_area = 0.0
                rel_size = 0.0
            
            masklet_size_rel.append(round(rel_size, 10))
            masklet_size_abs.append(round(avg_area, 10))
            masklet_size_bucket.append(_compute_size_bucket(rel_size))
            masklet_visibility_changes.append(visibility_changes)
            masklet_first_appeared_frame.append(float(first_frame * self.frame_id_step) if first_frame >= 0 else 0.0)
            masklet_frame_count.append(num_frames)  # Total frames in video
            masklet_type.append("manual")
        
        return {
            "masklet_id": masklet_id,
            "masklet_size_rel": masklet_size_rel,
            "masklet_size_abs": masklet_size_abs,
            "masklet_size_bucket": masklet_size_bucket,
            "masklet_visibility_changes": masklet_visibility_changes,
            "masklet_first_appeared_frame": masklet_first_appeared_frame,
            "masklet_frame_count": masklet_frame_count,
            "masklet_type": masklet_type,
            "masklet_stability_score": [None] * num_objects,  # Not computed for manual annotations
            "masklet_num": num_objects,
        }

    def _write_video(
        self,
        video_name: str,
        frames: List[np.ndarray],
        masks: List[np.ndarray],
        metadata: Optional[Dict[str, Any]] = None,
        video_duration: Optional[float] = None,
        instance_to_category: Optional[Dict[int, int]] = None,
        default_category_id: Optional[int] = None,
    ) -> None:
        """Write a single video sample to disk.

        Args:
            video_name: Unique name for the video.
            frames: List of image arrays (H, W) or (H, W, C).
            masks: List of label map arrays (H, W).
            metadata: Optional metadata to include in JSON.
            video_duration: Optional video duration in seconds.
            instance_to_category: Optional mapping from instance ID to category ID.
                When provided, masklet_category_id uses the mapped category ID
                instead of the raw instance/object ID.
            default_category_id: Fallback category ID when instance_to_category
                is provided but a specific instance ID is not in the mapping.
        """
        if video_name in self._video_names:
            raise ValueError(f"Video name '{video_name}' already exists")
        self._video_names.add(video_name)

        if len(frames) != len(masks):
            raise ValueError(
                f"Frames and masks count mismatch: {len(frames)} vs {len(masks)}"
            )

        if len(frames) == 0:
            logger.warning(f"Skipping empty video: {video_name}")
            self._video_names.discard(video_name)
            return

        # Filter out frames with masks below threshold if skip_empty_frames is enabled
        if self.skip_empty_frames:
            filtered_frames = []
            filtered_masks = []
            skipped_count = 0
            for frame, mask in zip(frames, masks):
                # Check if any object in this frame has enough pixels
                has_valid_mask = False
                for obj_id in np.unique(mask):
                    if obj_id == 0:
                        continue
                    if np.count_nonzero(mask == obj_id) >= self.min_mask_pixels:
                        has_valid_mask = True
                        break
                if has_valid_mask:
                    filtered_frames.append(frame)
                    filtered_masks.append(mask)
                else:
                    skipped_count += 1
            
            if skipped_count > 0:
                logger.debug(f"Video '{video_name}': filtered {skipped_count} frames below threshold")
            
            frames = filtered_frames
            masks = filtered_masks
            
            if len(frames) == 0:
                logger.debug(f"Skipping video '{video_name}': no frames with valid masks")
                self._video_names.discard(video_name)
                self._stats["skipped_empty"] += 1
                return

        # Get consistent object IDs across all frames
        object_ids = self._get_consistent_object_ids(masks)
        if not object_ids:
            logger.warning(f"No objects found in video: {video_name}")
            self._video_names.discard(video_name)
            return

        # Create video directory
        video_images_dir = self.images_dir / video_name
        video_images_dir.mkdir(parents=True, exist_ok=True)

        if self.save_visualization:
            video_viz_dir = self.viz_dir / video_name
            video_viz_dir.mkdir(parents=True, exist_ok=True)

        # Process frames
        frame_annotations: List[List[Optional[Dict[str, Any]]]] = []
        frame_areas: List[List[int]] = []
        rng = np.random.default_rng(42)
        colors = {
            obj_id: tuple(int(x) for x in rng.integers(50, 256, size=3))
            for obj_id in object_ids
        }

        # Get video dimensions from first frame
        first_frame_uint8 = _ensure_uint8_image(frames[0])
        video_height, video_width = first_frame_uint8.shape[:2]
        video_resolution = video_height * video_width

        for frame_idx, (frame, mask) in enumerate(zip(frames, masks)):
            # Compute frame ID based on step
            frame_id = frame_idx * self.frame_id_step
            
            # Save image
            frame_uint8 = _ensure_uint8_image(frame)
            image_path = video_images_dir / f"{frame_id:05d}.{self.image_format}"
            _save_image(image_path, frame_uint8, format=self.image_format)

            # Encode masks
            rle_masks, areas = self._encode_frame_masks(mask, object_ids)
            frame_annotations.append(rle_masks)
            frame_areas.append(areas)

            # Save visualization
            if self.save_visualization:
                viz_image = frame_uint8.copy()
                for obj_idx, obj_id in enumerate(object_ids):
                    if rle_masks[obj_idx] is not None:
                        binary_mask = (mask == obj_id).astype(np.uint8)
                        viz_image = _overlay_mask(
                            viz_image, binary_mask, colors[obj_id], self.viz_alpha
                        )
                viz_path = video_viz_dir / f"{frame_id:05d}.{self.image_format}"
                _save_image(viz_path, viz_image, format=self.image_format)

        # Compute video duration
        if video_duration is None:
            # Estimate from fps and frame count
            video_duration = len(frames) / self.fps

        # Compute masklet metadata
        masklet_meta = self._compute_masklet_metadata(
            frame_annotations, object_ids, frame_areas, video_resolution
        )

        # Build SA-V style JSON annotation
        json_data: Dict[str, Any] = {
            # Video metadata
            "video_id": video_name,
            "video_duration": round(video_duration, 3),
            "video_frame_count": float(len(frames)),
            "video_height": float(video_height),
            "video_width": float(video_width),
            "video_resolution": float(video_resolution),
            "video_environment": self.video_environment,
            "video_split": self.video_split,
            
            # Core masklet data
            "masklet": frame_annotations,
            
            # Masklet metadata
            **masklet_meta,
            
            # Annotation parameters
            "fps": self.fps,
            "ann_every": self.ann_every,
            "frame_id_step": self.frame_id_step,
            
            # Object ID mapping (for reference)
            "object_ids": object_ids,
        }
        
        # Add categories if defined
        if self._categories:
            json_data["categories"] = self._categories
            
            # Add category info per masklet
            masklet_category_id = []
            masklet_category_name = []
            for obj_id in object_ids:
                if instance_to_category is not None:
                    # Explicit instance -> category mapping (e.g. all vertebrae -> spine)
                    cat_id = instance_to_category.get(obj_id, default_category_id or obj_id)
                elif obj_id in self._category_id_to_name:
                    # Mask value matches a defined category (semantic segmentation)
                    cat_id = obj_id
                else:
                    # Fallback
                    cat_id = default_category_id if default_category_id is not None else obj_id
                cat_name = self._category_id_to_name.get(cat_id, f"class_{cat_id}")
                masklet_category_id.append(cat_id)
                masklet_category_name.append(cat_name)
            json_data["masklet_category_id"] = masklet_category_id
            json_data["masklet_category_name"] = masklet_category_name
        
        # Add custom metadata
        if metadata:
            json_data["metadata"] = metadata

        json_path = self.gt_dir / f"{video_name}_manual.json"
        with open(json_path, "w") as f:
            json.dump(json_data, f)

        # Update stats
        self._stats["videos"] += 1
        self._stats["total_frames"] += len(frames)
        self._stats["total_annotations"] += sum(
            1 for frame_anns in frame_annotations
            for ann in frame_anns if ann is not None
        )

        logger.debug(
            f"Wrote video '{video_name}': {len(frames)} frames, "
            f"{len(object_ids)} objects"
        )

    def add_2d(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        *,
        video_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        format: str = "sa-v",
        instance_to_category: Optional[Dict[int, int]] = None,
        default_category_id: int = 1,
    ) -> str:
        """Add a single 2D image as a one-frame video (SA-V) or SA-1B format.

        Args:
            image: Image array (H, W) or (H, W, C).
            mask: Label map (H, W) with instance IDs (0=background, 1..K=instances).
            video_name: Optional name for the video/image. Auto-generated if not provided.
            metadata: Optional metadata dict.
            format: Output format - "sa-v" (video style) or "sa-1b" (image style).
            instance_to_category: Optional dict mapping instance ID -> category ID (SA-1B only).
            default_category_id: Default category ID when mapping not provided (SA-1B only).

        Returns:
            The video/image name used.
        """
        if format.lower() == "sa-1b":
            return self.add_2d_sa1b(
                image, mask, 
                image_name=video_name, 
                metadata=metadata,
                instance_to_category=instance_to_category,
                default_category_id=default_category_id,
            )
        
        if video_name is None:
            video_name = self._generate_video_name("img")

        meta = dict(metadata or {})
        meta["source_type"] = "image"

        self._write_video(video_name, [image], [mask], meta)
        return video_name

    def add_2d_sa1b(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        *,
        image_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        stability_scores: Optional[List[float]] = None,
        predicted_ious: Optional[List[float]] = None,
        point_coords: Optional[List[List[List[float]]]] = None,
        instance_to_category: Optional[Dict[int, int]] = None,
        default_category_id: int = 1,
    ) -> str:
        """Add a single 2D image in SA-1B format.

        SA-1B format stores each image with its own JSON file containing
        multiple object annotations, each with bounding box, area, segmentation,
        and optional SAM-specific fields (stability_score, predicted_iou, point_coords).

        Output structure:
        - {out_dir}/images/{image_name}.jpg
        - {out_dir}/gt/{image_name}.json

        Args:
            image: Image array (H, W) or (H, W, C).
            mask: Label map (H, W) with instance IDs (0=background, 1..K=instances).
                  Note: mask values are INSTANCE IDs, not category IDs.
            image_name: Optional name for the image. Auto-generated if not provided.
            metadata: Optional metadata dict.
            stability_scores: Optional list of stability scores per object.
            predicted_ious: Optional list of predicted IoU per object.
            point_coords: Optional list of point coordinates per object.
            instance_to_category: Optional dict mapping instance ID -> category ID.
                  If not provided, uses mask values as category IDs (semantic segmentation mode).
                  Example: {1: 1, 2: 1, 3: 2} means instances 1,2 are category 1, instance 3 is category 2.
            default_category_id: Default category ID when instance_to_category is not provided
                  and mask value doesn't match any defined category. Default is 1.

        Returns:
            The image name used.
        
        Note on mask interpretation:
            - If instance_to_category is provided: mask values are instance IDs, 
              mapped to categories via the dict.
            - If instance_to_category is None and categories are defined: mask values 
              are treated as category IDs (semantic segmentation).
            - If no categories defined: mask values are just object indices.
        """
        if image_name is None:
            image_name = self._generate_video_name("sa")
        
        if image_name in self._video_names:
            raise ValueError(f"Image name '{image_name}' already exists")
        self._video_names.add(image_name)

        # Process image
        image_uint8 = _ensure_uint8_image(image)
        height, width = image_uint8.shape[:2]
        
        # Get object IDs from mask
        object_ids = self._get_object_ids_from_mask(mask)
        if not object_ids:
            if self.skip_empty_annotations:
                self._video_names.discard(image_name)
                self._stats["skipped_empty"] += 1
                logger.debug(f"Skipped empty image '{image_name}': no objects found")
                return image_name
            else:
                logger.warning(f"No objects found in image: {image_name}")

        # Build annotations list (SA-1B style)
        annotations = []
        rng = np.random.default_rng(hash(image_name) % (2**32))
        
        for obj_idx, obj_id in enumerate(object_ids):
            binary_mask = (mask == obj_id).astype(np.uint8)
            pixel_count = np.count_nonzero(binary_mask)
            
            if pixel_count < self.min_mask_pixels:
                continue
            
            # Encode segmentation
            rle = _encode_rle(binary_mask)
            
            # Compute bbox
            bbox = _compute_bbox_from_mask(binary_mask)
            
            # Build annotation
            ann: Dict[str, Any] = {
                "id": int(rng.integers(0, 2**31)),
                "bbox": bbox,
                "area": int(pixel_count),
                "segmentation": rle,
            }
            
            # Add optional SAM fields
            if stability_scores is not None and obj_idx < len(stability_scores):
                ann["stability_score"] = stability_scores[obj_idx]
            else:
                ann["stability_score"] = 1.0  # Default for manual annotations
            
            if predicted_ious is not None and obj_idx < len(predicted_ious):
                ann["predicted_iou"] = predicted_ious[obj_idx]
            else:
                ann["predicted_iou"] = 1.0  # Default for manual annotations
            
            if point_coords is not None and obj_idx < len(point_coords):
                ann["point_coords"] = point_coords[obj_idx]
            else:
                # Generate center point as default
                center_y, center_x = np.where(binary_mask)
                if len(center_x) > 0:
                    cx = float(np.mean(center_x))
                    cy = float(np.mean(center_y))
                    ann["point_coords"] = [[cx, cy]]
            
            # Add crop_box (same as bbox for simplicity)
            ann["crop_box"] = bbox
            
            # Determine category ID for this instance
            if instance_to_category is not None:
                # Explicit instance -> category mapping provided
                cat_id = instance_to_category.get(obj_id, default_category_id)
            elif obj_id in self._category_id_to_name:
                # Mask value matches a defined category (semantic segmentation mode)
                cat_id = obj_id
            else:
                # No mapping, use default
                cat_id = default_category_id
            
            # Add category info
            ann["category_id"] = cat_id
            if cat_id in self._category_id_to_name:
                ann["category_name"] = self._category_id_to_name[cat_id]
            
            annotations.append(ann)

        # Skip writing if no valid annotations and skip_empty_annotations is enabled
        if self.skip_empty_annotations and not annotations:
            self._video_names.discard(image_name)
            self._stats["skipped_empty"] += 1
            logger.debug(f"Skipped empty image '{image_name}': no annotations above min_mask_pixels threshold")
            return image_name

        # Save image (only after we know we'll write the JSON)
        image_path = self.images_dir / f"{image_name}.{self.image_format}"
        _save_image(image_path, image_uint8, format=self.image_format)

        # Build SA-1B style JSON
        json_data: Dict[str, Any] = {
            "image": {
                "image_id": len(self._video_names) - 1,
                "width": width,
                "height": height,
                "file_name": f"{image_name}.{self.image_format}",
            },
            "annotations": annotations,
        }
        
        # Add custom metadata
        if metadata:
            json_data["metadata"] = metadata

        # Save JSON (flat structure for SA-1B)
        json_path = self.gt_dir / f"{image_name}.json"
        with open(json_path, "w") as f:
            json.dump(json_data, f)

        # Save visualization
        if self.save_visualization:
            viz_image = image_uint8.copy()
            colors = {
                obj_id: tuple(int(x) for x in rng.integers(50, 256, size=3))
                for obj_id in object_ids
            }
            for obj_id in object_ids:
                binary_mask = (mask == obj_id).astype(np.uint8)
                if np.count_nonzero(binary_mask) >= self.min_mask_pixels:
                    viz_image = _overlay_mask(viz_image, binary_mask, colors[obj_id], self.viz_alpha)
            viz_path = self.viz_dir / f"{image_name}.{self.image_format}"
            _save_image(viz_path, viz_image, format=self.image_format)

        # Update stats
        self._stats["videos"] += 1
        self._stats["total_frames"] += 1
        self._stats["total_annotations"] += len(annotations)

        logger.debug(f"Wrote SA-1B image '{image_name}': {len(annotations)} annotations")
        return image_name

    def add_3d(
        self,
        volume: np.ndarray,
        mask: np.ndarray,
        *,
        slice_axis: int = 0,
        video_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add a 3D volume as a video (each slice becomes a frame).

        Args:
            volume: Volume array (D, H, W) or (D, H, W, C).
            mask: Label volume (D, H, W).
            slice_axis: Axis to slice along (default 0).
            video_name: Optional name for the video.
            metadata: Optional metadata dict.

        Returns:
            The video name used.
        """
        if video_name is None:
            video_name = self._generate_video_name("vol")

        vol = np.moveaxis(volume, slice_axis, 0)
        m = np.moveaxis(mask, slice_axis, 0)

        if vol.shape[0] != m.shape[0]:
            raise ValueError(
                f"Volume/mask slice count mismatch: {vol.shape[0]} vs {m.shape[0]}"
            )

        frames = [vol[i] for i in range(vol.shape[0])]
        masks = [m[i] for i in range(m.shape[0])]

        meta = dict(metadata or {})
        meta.update({
            "source_type": "volume",
            "slice_axis": slice_axis,
            "num_slices": len(frames),
        })

        self._write_video(video_name, frames, masks, meta)
        return video_name

    def add_video(
        self,
        video: np.ndarray,
        mask: np.ndarray,
        *,
        frame_axis: int = 0,
        video_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        video_duration: Optional[float] = None,
        instance_to_category: Optional[Dict[int, int]] = None,
        default_category_id: Optional[int] = None,
    ) -> str:
        """Add a video sequence.

        Args:
            video: Video array (T, H, W) or (T, H, W, C).
            mask: Mask array (T, H, W).  Each unique non-zero value is
                treated as a distinct tracked object (instance).
            frame_axis: Axis for frames (default 0).
            video_name: Optional name for the video.
            metadata: Optional metadata dict.
            video_duration: Optional video duration in seconds.
            instance_to_category: Optional mapping {instance_id: category_id}.
                Use this when mask instance IDs differ from category IDs
                (e.g. all vertebrae instances map to a single "spine" category).
            default_category_id: Fallback category ID for instances not
                found in *instance_to_category*.

        Returns:
            The video name used.
        """
        if video_name is None:
            video_name = self._generate_video_name("vid")

        vid = np.moveaxis(video, frame_axis, 0)
        m = np.moveaxis(mask, frame_axis, 0)

        if vid.shape[0] != m.shape[0]:
            raise ValueError(
                f"Video/mask frame count mismatch: {vid.shape[0]} vs {m.shape[0]}"
            )

        frames = [vid[i] for i in range(vid.shape[0])]
        masks = [m[i] for i in range(m.shape[0])]

        meta = dict(metadata or {})
        meta.update({
            "source_type": "video",
            "num_frames": len(frames),
        })

        self._write_video(
            video_name, frames, masks, meta, video_duration,
            instance_to_category=instance_to_category,
            default_category_id=default_category_id,
        )
        return video_name

    def add_frames(
        self,
        frames: List[np.ndarray],
        masks: List[np.ndarray],
        *,
        video_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        video_duration: Optional[float] = None,
    ) -> str:
        """Add a sequence of frames directly (most flexible API).

        Args:
            frames: List of image arrays.
            masks: List of corresponding mask arrays.
            video_name: Optional name for the video.
            metadata: Optional metadata dict.
            video_duration: Optional video duration in seconds.

        Returns:
            The video name used.
        """
        if video_name is None:
            video_name = self._generate_video_name("seq")

        self._write_video(video_name, frames, masks, metadata, video_duration)
        return video_name

    def get_stats(self) -> Dict[str, int]:
        """Get dataset statistics."""
        return dict(self._stats)

    def get_video_names(self) -> List[str]:
        """Get list of all video names in the dataset."""
        return sorted(self._video_names)

    def write_metadata(self, filename: str = "dataset_info.json") -> Path:
        """Write dataset metadata file.

        Args:
            filename: Name of the metadata file.

        Returns:
            Path to the written file.
        """
        meta_path = self.out_dir / filename
        metadata = {
            "dataset_name": self.dataset_name,
            "num_videos": self._stats["videos"],
            "total_frames": self._stats["total_frames"],
            "total_annotations": self._stats["total_annotations"],
            "categories": self._categories,
            "fps": self.fps,
            "ann_every": self.ann_every,
            "frame_id_step": self.frame_id_step,
            "video_environment": self.video_environment,
            "video_split": self.video_split,
            "video_names": sorted(self._video_names),
        }
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        return meta_path

    def finalize(self) -> Dict[str, Any]:
        """Finalize the dataset and write metadata.

        Returns:
            Dataset statistics.
        """
        self.write_metadata()
        skipped_msg = ""
        if self._stats["skipped_empty"] > 0:
            skipped_msg = f", {self._stats['skipped_empty']} skipped (empty)"
        logger.info(
            f"Dataset '{self.dataset_name}' complete: "
            f"{self._stats['videos']} videos, "
            f"{self._stats['total_frames']} frames, "
            f"{self._stats['total_annotations']} annotations{skipped_msg}"
        )
        return self.get_stats()


def convert_coco_to_sam2(
    coco_json_path: Union[str, Path],
    images_dir: Union[str, Path],
    out_dir: Union[str, Path],
    *,
    dataset_name: str = "converted",
    group_by_meta_key: Optional[str] = "group_id",
) -> SAM2VideoAnnotator:
    """Convert a COCO-style dataset to SAM2 video format.

    This is useful for converting existing USCocoAnnotator outputs to SAM2 format.

    Args:
        coco_json_path: Path to COCO JSON file.
        images_dir: Directory containing the images.
        out_dir: Output directory for SAM2 format.
        dataset_name: Name for the converted dataset.
        group_by_meta_key: Metadata key to group frames into videos.
            If None, each image becomes a separate single-frame video.

    Returns:
        The SAM2VideoAnnotator instance used for conversion.
    """
    from collections import defaultdict

    coco_json_path = Path(coco_json_path)
    images_dir = Path(images_dir)

    with open(coco_json_path) as f:
        coco = json.load(f)

    annotator = SAM2VideoAnnotator(
        out_dir,
        dataset_name=dataset_name,
        categories=coco.get("categories", []),
    )

    # Build image ID -> annotations mapping
    img_id_to_anns: Dict[int, List[Dict]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        img_id_to_anns[int(ann["image_id"])].append(ann)

    # Group images by video/group
    if group_by_meta_key:
        groups: Dict[Any, List[Dict]] = defaultdict(list)
        for img in coco.get("images", []):
            meta = img.get("meta", {})
            group_key = meta.get(group_by_meta_key, img["id"])
            groups[group_key].append(img)
        # Sort each group by index
        for group_key in groups:
            groups[group_key].sort(
                key=lambda x: x.get("meta", {}).get("index", x["id"])
            )
    else:
        # Each image is its own group
        groups = {img["id"]: [img] for img in coco.get("images", [])}

    # Process each group
    for group_key, images in groups.items():
        frames = []
        masks = []

        for img_info in images:
            # Load image
            img_path = images_dir / img_info["file_name"]
            if not img_path.exists():
                # Try without "images/" prefix
                img_path = images_dir / Path(img_info["file_name"]).name

            PIL_Image = _try_import_pil()
            cv2 = _try_import_cv2()

            if PIL_Image is not None:
                frame = np.array(PIL_Image.open(img_path).convert("RGB"))
            elif cv2 is not None:
                frame = cv2.imread(str(img_path))[..., ::-1]
            else:
                raise RuntimeError("Neither PIL nor cv2 available")

            frames.append(frame)

            # Reconstruct mask from annotations
            h, w = img_info["height"], img_info["width"]
            mask = np.zeros((h, w), dtype=np.uint8)

            for ann in img_id_to_anns.get(int(img_info["id"]), []):
                seg = ann.get("segmentation")
                if isinstance(seg, dict):  # RLE
                    binary = _decode_rle(seg)
                    cat_id = int(ann.get("category_id", 1))
                    mask[binary > 0] = cat_id

            masks.append(mask)

        # Determine video name
        meta = images[0].get("meta", {})
        video_name = meta.get("group_name") or f"video_{group_key}"

        annotator.add_frames(frames, masks, video_name=video_name, metadata=meta)

    annotator.finalize()
    return annotator
