#!/usr/bin/env python3
"""Detect duplicate / overlapping images across SaUS sub-datasets.

Ported from the old ``USSam/projects/US_Datasets/utils/check_overlap.py`` into
the sonobase project (2026-05-25). Logic is unchanged; only the defaults are
env-aware and it is package-runnable.

Computes a hash for every image across all sub-datasets and reports pairs that
share an identical hash, both **cross-dataset** (same image in two datasets)
and **intra-dataset** (same image under two names in one dataset). The intra
report is what ``dedup_splits.py`` consumes.

Handles both SaUS layouts:
- image-based (SA-1B): ``images/<name>.jpg``
- video-based (SA-V):  ``images/<video>/<frame>.jpg``

Run from ``src/``:

    uv run python -m data.utils.check_overlap \\
        --path $DATASET_DIR --hash-algo md5 \\
        --output $DATASET_DIR/overlap_report.json

Hash algorithms:
- ``md5``      exact pixel-content match (what the production overlap_report used)
- ``file_md5`` exact file-bytes match (fastest)
- ``phash``    perceptual (DCT) hash, robust to re-encoding
"""

import argparse
import hashlib
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ── hashing helpers ──────────────────────────────────────────────────────────

def phash(image: np.ndarray, hash_size: int = 32, highfreq_factor: int = 4) -> str:
    """DCT-based perceptual hash. Defaults yield a 64-bit (16-hex) hash."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    resized = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(resized))
    low_size = hash_size // highfreq_factor
    dct_low = dct[:low_size, :low_size]
    median = np.median(dct_low)
    bits = (dct_low > median).flatten()
    hash_int = 0
    for bit in bits:
        hash_int = (hash_int << 1) | int(bit)
    return f"{hash_int:016x}"


def md5_hash(image: np.ndarray) -> str:
    """MD5 of raw pixel data (exact content match)."""
    return hashlib.md5(image.tobytes()).hexdigest()


def file_md5(path: Path) -> str:
    """MD5 of the file bytes (exact file match)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


HASH_FUNCTIONS = {
    "phash": phash,
    "md5": md5_hash,
    "file_md5": file_md5,
}


# ── dataset enumeration ──────────────────────────────────────────────────────

def is_video_dataset(dataset_path: Path) -> bool:
    """Video-based datasets have subdirectories inside ``images/``."""
    images_dir = dataset_path / "images"
    if not images_dir.exists():
        return False
    for item in images_dir.iterdir():
        return item.is_dir()
    return False


def iter_images(dataset_path: Path):
    """Yield ``(relative_name, absolute_path)`` for every image in a sub-dataset.

    For video datasets the relative name is ``<video>/<frame>`` so that
    downstream dedup can tell intra-video from cross-video duplicates.
    """
    images_dir = dataset_path / "images"
    if not images_dir.exists():
        return

    if is_video_dataset(dataset_path):
        for video_dir in sorted(images_dir.iterdir()):
            if not video_dir.is_dir():
                continue
            for img_file in sorted(video_dir.iterdir()):
                if img_file.suffix.lower() in IMAGE_EXTENSIONS:
                    yield f"{video_dir.name}/{img_file.name}", img_file
    else:
        for img_file in sorted(images_dir.iterdir()):
            if img_file.is_file() and img_file.suffix.lower() in IMAGE_EXTENSIONS:
                yield img_file.name, img_file


# ── core logic ───────────────────────────────────────────────────────────────

def _hash_one(task: Tuple[str, str, str, str]) -> Optional[Tuple[str, str, str]]:
    """Hash one image. ``task = (dataset, rel_name, path_str, hash_algo)``.

    Top-level (picklable) so it can run inside a multiprocessing pool.
    Returns ``(hash, dataset, rel_name)`` or ``None`` on read failure.
    """
    ds_name, rel_name, path_str, hash_algo = task
    hash_fn = HASH_FUNCTIONS[hash_algo]
    try:
        if hash_algo in ("phash", "md5"):
            img = cv2.imread(path_str, cv2.IMREAD_COLOR)
            if img is None:
                return None
            h = hash_fn(img)
        else:
            h = hash_fn(Path(path_str))
    except Exception:
        return None
    return (h, ds_name, rel_name)


def compute_hashes(
    saus_root: Path,
    datasets: Optional[List[str]],
    hash_algo: str = "md5",
    num_workers: int = 1,
) -> Dict[str, List[Tuple[str, str]]]:
    """Hash all images. Returns ``{hash: [(dataset, rel_name), ...]}``.

    Sources per hash are sorted before returning so the downstream
    "keep-first" dedup choice is deterministic regardless of ``num_workers``.
    """
    hash_to_sources: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    if datasets is None:
        datasets = sorted(
            d.name for d in saus_root.iterdir()
            if d.is_dir() and (d / "images").exists()
        )

    # Enumerate first (cheap), then hash (expensive) — optionally in parallel.
    tasks: List[Tuple[str, str, str, str]] = []
    for ds_name in datasets:
        ds_path = saus_root / ds_name
        if not ds_path.is_dir():
            logger.warning(f"Dataset directory not found: {ds_path}")
            continue
        ds_tasks = [(ds_name, rel, str(p), hash_algo) for rel, p in iter_images(ds_path)]
        tasks.extend(ds_tasks)
        logger.info(f"  {ds_name}: {len(ds_tasks)} images queued")

    logger.info(f"Hashing {len(tasks)} images with {num_workers} worker(s)...")
    if num_workers > 1 and tasks:
        import multiprocessing as mp
        with mp.Pool(num_workers) as pool:
            for res in pool.imap_unordered(_hash_one, tasks, chunksize=256):
                if res is not None:
                    h, ds, rel = res
                    hash_to_sources[h].append((ds, rel))
    else:
        for t in tasks:
            res = _hash_one(t)
            if res is not None:
                h, ds, rel = res
                hash_to_sources[h].append((ds, rel))

    for h in hash_to_sources:
        hash_to_sources[h].sort()

    logger.info(f"Total images hashed: {sum(len(v) for v in hash_to_sources.values())}")
    return hash_to_sources


def find_overlaps(
    hash_to_sources: Dict[str, List[Tuple[str, str]]],
) -> Dict[Tuple[str, str], List[Tuple[str, str, str]]]:
    """Return ``{(dataset_a, dataset_b): [(hash, name_a, name_b), ...]}``.

    ``dataset_a == dataset_b`` entries are intra-dataset duplicates.
    """
    pair_overlaps: Dict[Tuple[str, str], List[Tuple[str, str, str]]] = defaultdict(list)

    for h, sources in hash_to_sources.items():
        if len(sources) < 2:
            continue
        ds_to_names: Dict[str, List[str]] = defaultdict(list)
        for ds, name in sources:
            ds_to_names[ds].append(name)

        ds_list = sorted(ds_to_names.keys())
        for i in range(len(ds_list)):
            for j in range(i + 1, len(ds_list)):
                da, db = ds_list[i], ds_list[j]
                for na in ds_to_names[da]:
                    for nb in ds_to_names[db]:
                        pair_overlaps[(da, db)].append((h, na, nb))

        for ds, names in ds_to_names.items():
            if len(names) > 1:
                for i in range(len(names)):
                    for j in range(i + 1, len(names)):
                        pair_overlaps[(ds, ds)].append((h, names[i], names[j]))

    return dict(sorted(pair_overlaps.items()))


def print_report(pair_overlaps: Dict[Tuple[str, str], List[Tuple[str, str, str]]]) -> None:
    if not pair_overlaps:
        print("\n✅ No overlaps found between any sub-datasets!")
        return

    cross_pairs = {k: v for k, v in pair_overlaps.items() if k[0] != k[1]}
    intra_pairs = {k: v for k, v in pair_overlaps.items() if k[0] == k[1]}

    print("\n" + "=" * 70)
    print("OVERLAP REPORT")
    print("=" * 70)

    if cross_pairs:
        print(f"\n🔴 Cross-dataset overlaps in {len(cross_pairs)} pair(s):\n")
        print(f"  {'Dataset A':<25} {'Dataset B':<25} {'# Overlaps':>10}")
        print(f"  {'-'*25} {'-'*25} {'-'*10}")
        for (da, db), items in sorted(cross_pairs.items(), key=lambda x: -len(x[1])):
            print(f"  {da:<25} {db:<25} {len(items):>10}")
    else:
        print("\n✅ No cross-dataset overlaps found!")

    if intra_pairs:
        print(f"\n⚠️  Intra-dataset duplicates in {len(intra_pairs)} dataset(s):\n")
        for (ds, _), items in sorted(intra_pairs.items(), key=lambda x: -len(x[1])):
            print(f"  {ds}: {len(items)} duplicate pair(s)")


def save_report(
    pair_overlaps: Dict[Tuple[str, str], List[Tuple[str, str, str]]],
    output_path: Path,
) -> None:
    report = {
        "summary": {
            "total_overlap_pairs": sum(len(v) for v in pair_overlaps.values()),
            "cross_dataset_pairs": sum(len(v) for k, v in pair_overlaps.items() if k[0] != k[1]),
            "intra_dataset_duplicates": sum(len(v) for k, v in pair_overlaps.items() if k[0] == k[1]),
        },
        "overlaps": [],
    }
    for (da, db), items in sorted(pair_overlaps.items(), key=lambda x: -len(x[1])):
        report["overlaps"].append({
            "dataset_a": da,
            "dataset_b": db,
            "type": "intra" if da == db else "cross",
            "count": len(items),
            "matches": [{"hash": h, "name_a": na, "name_b": nb} for h, na, nb in items],
        })
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report saved to {output_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Check for image overlap between SaUS sub-datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--path", type=str, default=os.environ.get("DATASET_DIR"),
        help="Path to SaUS dataset root (default: $DATASET_DIR)",
    )
    parser.add_argument(
        "--datasets", type=str, nargs="*", default=None,
        help="Specific sub-datasets to check (default: all)",
    )
    parser.add_argument(
        "--hash-algo", type=str, default="md5", choices=list(HASH_FUNCTIONS.keys()),
        help="md5 (exact pixel), file_md5 (exact bytes), phash (perceptual). Default: md5",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Save JSON report to this path (default: print only)",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Parallel hashing processes (default: 1; set to $SLURM_CPUS_PER_TASK on cluster)",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if not args.path:
        logger.error("No --path given and $DATASET_DIR is unset.")
        return
    saus_root = Path(args.path)
    if not saus_root.exists():
        logger.error(f"SaUS root not found: {saus_root}")
        return

    logger.info(f"SaUS root: {saus_root}")
    logger.info(f"Hash algorithm: {args.hash_algo}")

    hash_to_sources = compute_hashes(
        saus_root, args.datasets, hash_algo=args.hash_algo, num_workers=args.workers
    )
    logger.info(f"Unique hashes: {len(hash_to_sources)}")

    pair_overlaps = find_overlaps(hash_to_sources)
    print_report(pair_overlaps)

    if args.output:
        save_report(pair_overlaps, Path(args.output))


if __name__ == "__main__":
    main()
