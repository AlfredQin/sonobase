"""
Shared utilities for dataset split generation.

This module provides the building blocks used by all split functions:
  - Sample-name extraction (from dataset_info.json or folder scan)
  - Random splitting with per-call seeding (order-independent determinism)
  - Split-file I/O
  - CSV helpers

These utilities are intentionally stateless — no config, no registry.
"""

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


# ──────────────────────────────────────────────────────────────────────
# Data types
# ──────────────────────────────────────────────────────────────────────

@dataclass
class SplitResult:
    """Outcome of a single dataset's split operation."""

    dataset_name: str
    category: str       # "pretrain", "benchmark", "external"
    status: str         # "ok" | "dry_run" | "exists" | "not_found" | "no_samples" | "skipped"
    n_train: int = 0
    n_val: int = 0
    n_test: int = 0
    n_total: int = 0
    is_video: bool = False
    message: str = ""


# ──────────────────────────────────────────────────────────────────────
# CSV helpers
# ──────────────────────────────────────────────────────────────────────

def read_dataset_list(csv_path: Path) -> List[str]:
    """Read dataset names from a CSV file (one name per line, no header)."""
    names: List[str] = []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row[0].strip():
                names.append(row[0].strip())
    return names


# ──────────────────────────────────────────────────────────────────────
# Sample-name extraction
# ──────────────────────────────────────────────────────────────────────

def get_sample_names(dataset_path: Path) -> List[str]:
    """
    Get all sample/item names for a dataset.

    Resolution order:
        1. dataset_info.json  →  "video_names" key  (authoritative)
        2. Scan images/ folder  (fallback)

    Returns:
        Sorted list of sample names.
    """
    # 1. Try dataset_info.json
    # info_path = dataset_path / "dataset_info.json"
    # if info_path.exists():
    #     try:
    #         with open(info_path, "r") as f:
    #             info = json.load(f)
    #         if "video_names" in info and info["video_names"]:
    #             return sorted(info["video_names"])
    #     except (json.JSONDecodeError, KeyError):
    #         pass

    # 2. Fallback: scan images/ folder
    images_dir = dataset_path / "images"
    if not images_dir.exists():
        return []
    return _scan_images_folder(images_dir)


def is_video_based(dataset_path: Path) -> bool:
    """
    Determine if a dataset is video-based.

    Video-based datasets have subdirectories under images/, where each
    subdirectory represents a video/volume.  Image-based datasets have
    image files directly under images/.
    """
    images_dir = dataset_path / "images"
    if not images_dir.exists():
        return False
    items = list(images_dir.iterdir())
    return bool(items) and any(item.is_dir() for item in items[:10])


def _scan_images_folder(images_dir: Path) -> List[str]:
    """Scan images/ folder to get sample names (handles both image and video)."""
    items = list(images_dir.iterdir())
    if not items:
        return []

    has_dirs = any(item.is_dir() for item in items[:10])

    if has_dirs:
        # Video-based: directory names are sample names
        return sorted(d.name for d in images_dir.iterdir() if d.is_dir())
    else:
        # Image-based: file stems are sample names
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        return sorted(
            f.stem
            for f in images_dir.iterdir()
            if f.is_file() and f.suffix.lower() in image_exts
        )


# ──────────────────────────────────────────────────────────────────────
# Splitting logic
# ──────────────────────────────────────────────────────────────────────

def split_samples(
    items: List[str],
    ratios: Dict[str, float],
    seed: int = 42,
) -> Dict[str, List[str]]:
    """
    Split items into subsets according to the given ratios.

    Uses a **per-call** Random instance so the result is deterministic
    regardless of global random state or processing order.

    Args:
        items:  List of sample names.
        ratios: e.g. ``{"train": 0.95, "val": 0.05}`` or
                ``{"train": 0.7, "val": 0.1, "test": 0.2}``.
        seed:   Random seed.

    Returns:
        Dict mapping split name → sorted list of sample names.
    """
    items = items.copy()
    rng = random.Random(seed)
    rng.shuffle(items)

    n = len(items)
    splits: Dict[str, List[str]] = {}
    start = 0

    split_names = list(ratios.keys())
    for i, name in enumerate(split_names):
        if i == len(split_names) - 1:
            # Last split gets all remaining items
            splits[name] = items[start:]
        else:
            end = start + max(1, int(n * ratios[name]))  # at least 1 sample
            splits[name] = items[start:end]
            start = end

    # Sort each split for deterministic file output
    for name in splits:
        splits[name] = sorted(splits[name])

    return splits


# ──────────────────────────────────────────────────────────────────────
# File I/O
# ──────────────────────────────────────────────────────────────────────

def write_split_file(filepath: Path, items: List[str]):
    """Write sample names to a text file, one per line."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        for item in items:
            f.write(f"{item}\n")
