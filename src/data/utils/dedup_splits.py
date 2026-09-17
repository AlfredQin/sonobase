#!/usr/bin/env python3
"""Deduplicate SaUS datasets from a ``check_overlap.py`` intra-dataset report.

Ported from the old ``USSam/projects/US_Datasets/utils/dedup_splits.py`` into
the sonobase project (2026-05-25). Logic is unchanged; defaults are env-aware,
it is package-runnable, and list writing reuses ``dataset_split.write_split_file``.

Handles four kinds of **intra-dataset** duplicates (cross-dataset overlaps in
the report are reported but not auto-removed — they need a manual keep decision):

1. Image-based — remove the duplicate image + annotation + split-list entry.
2. Whole-video duplicate — remove the video folder + annotation + split entry.
3. Intra-video frame duplicate — delete only the duplicate frame files (video
   stays in the split); the GT JSON should be regenerated afterwards.
4. Mixed / partial cross-video — delete the shared frames from the later video.

Operates on a **copy** of the data (``--data-dir``, e.g. SaUSDedup) so the
original ``SaUS`` is preserved, and writes deduplicated split lists to
``--output-dir`` (default ``{splits_dir}_dedup``).

Run from ``src/``:

    uv run python -m data.utils.dedup_splits \\
        --overlap-report $DATASET_DIR/overlap_report.json \\
        --splits-dir $ANNOTATION_DIR \\
        --data-dir $WORK/Dataset/Ultrasound/SaUSDedup \\
        --output-dir ${ANNOTATION_DIR}_dedup

    # Dry run (nothing modified):
    uv run python -m data.utils.dedup_splits --dry-run ...
"""

import argparse
import json
import logging
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .dataset_split import write_split_file

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ── Classification ───────────────────────────────────────────────────────────

def classify_overlaps(matches: List[dict]) -> dict:
    """Split overlap matches into cross-video and intra-video groups."""
    is_video = any("/" in m["name_a"] for m in matches)

    if not is_video:
        return {"is_video": False, "cross_video": [], "intra_video": [], "cross_video_pairs": {}}

    cross, intra = [], []
    pairs: Dict[Tuple[str, str], int] = defaultdict(int)
    for m in matches:
        va = m["name_a"].split("/")[0]
        vb = m["name_b"].split("/")[0]
        if va != vb:
            cross.append(m)
            pairs[tuple(sorted([va, vb]))] += 1
        else:
            intra.append(m)

    return {
        "is_video": True,
        "cross_video": cross,
        "intra_video": intra,
        "cross_video_pairs": dict(pairs),
    }


def is_whole_video_dup(va: str, vb: str, shared: int, data_dir: Path, ds: str) -> bool:
    """True if two videos overlap in >= 90% of both their frames."""
    dir_a = data_dir / ds / "images" / va
    dir_b = data_dir / ds / "images" / vb
    n_a = len([f for f in dir_a.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]) if dir_a.is_dir() else 0
    n_b = len([f for f in dir_b.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]) if dir_b.is_dir() else 0
    if n_a == 0 or n_b == 0:
        return False
    return (shared / n_a >= 0.9) and (shared / n_b >= 0.9)


# ── Actions ──────────────────────────────────────────────────────────────────

class DeduplicationPlan:
    """Collects all deduplication actions before executing them."""

    def __init__(self):
        self.split_removals: Dict[str, Set[str]] = defaultdict(set)
        self.file_deletions: List[Tuple[Path, str]] = []
        self.gt_frame_removals: List[Tuple[Path, Set[str], str]] = []

    def remove_from_splits(self, dataset: str, entry: str):
        self.split_removals[dataset].add(entry)

    def delete_image(self, img_path: Path, gt_path: Path, desc: str):
        self.file_deletions.append((img_path, f"image: {desc}"))
        if gt_path.exists():
            self.file_deletions.append((gt_path, f"annotation: {desc}"))

    def delete_video(self, video_img_dir: Path, gt_path: Path, desc: str):
        self.file_deletions.append((video_img_dir, f"video dir: {desc}"))
        if gt_path.exists():
            self.file_deletions.append((gt_path, f"annotation: {desc}"))

    def delete_frame_from_video(self, frame_path: Path, video_name: str, frame_name: str, desc: str):
        self.file_deletions.append((frame_path, f"frame: {desc}"))

    def summary(self) -> str:
        total_split = sum(len(v) for v in self.split_removals.values())
        return "\n".join([
            f"Split removals: {total_split} entries across {len(self.split_removals)} datasets",
            f"File deletions: {len(self.file_deletions)} files/dirs",
            f"GT frame removals: {len(self.gt_frame_removals)} JSON updates",
        ])


# ── Plan building ────────────────────────────────────────────────────────────

def build_plan(overlaps: List[dict], data_dir: Path) -> DeduplicationPlan:
    plan = DeduplicationPlan()

    for entry in overlaps:
        ds = entry["dataset_a"]
        matches = entry["matches"]
        info = classify_overlaps(matches)

        if not info["is_video"]:
            _plan_image_dedup(plan, ds, matches, data_dir)
        else:
            cross_pairs = info["cross_video_pairs"]
            if cross_pairs:
                for (va, vb), shared in cross_pairs.items():
                    if is_whole_video_dup(va, vb, shared, data_dir, ds):
                        _plan_whole_video_dedup(plan, ds, vb, data_dir)
                    else:
                        _plan_partial_cross_video_dedup(plan, ds, va, vb, info["cross_video"], data_dir)
            if info["intra_video"]:
                _plan_intra_video_dedup(plan, ds, info["intra_video"], data_dir)

    return plan


def _plan_image_dedup(plan: DeduplicationPlan, ds: str, matches: List[dict], data_dir: Path):
    """Case 1: image dataset — keep first per hash, remove the rest."""
    hash_to_names: Dict[str, Set[str]] = defaultdict(set)
    for m in matches:
        hash_to_names[m["hash"]].add(Path(m["name_a"]).stem)
        hash_to_names[m["hash"]].add(Path(m["name_b"]).stem)

    for h, names in hash_to_names.items():
        for name in sorted(names)[1:]:
            plan.remove_from_splits(ds, name)
            for ext in [".jpg", ".png", ".bmp", ".tif"]:
                img_path = data_dir / ds / "images" / f"{name}{ext}"
                if img_path.exists():
                    gt_path = data_dir / ds / "gt" / f"{name}.json"
                    plan.delete_image(img_path, gt_path, f"[{ds}] {name}")
                    break

    logger.info(f"  [{ds}] IMAGE: {len(plan.split_removals.get(ds, set()))} images to remove")


def _plan_whole_video_dedup(plan: DeduplicationPlan, ds: str, video_to_remove: str, data_dir: Path):
    """Case 2: whole-video duplicate — drop video from splits + delete folder."""
    plan.remove_from_splits(ds, video_to_remove)
    video_dir = data_dir / ds / "images" / video_to_remove
    gt_path = data_dir / ds / "gt" / f"{video_to_remove}_manual.json"
    plan.delete_video(video_dir, gt_path, f"[{ds}] {video_to_remove}")
    logger.info(f"  [{ds}] WHOLE-VIDEO: remove '{video_to_remove}'")


def _plan_intra_video_dedup(plan: DeduplicationPlan, ds: str, intra_matches: List[dict], data_dir: Path):
    """Case 3: duplicate frames within a video — delete the later frame files."""
    video_hash_to_frames: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    for m in intra_matches:
        video = m["name_a"].split("/")[0]
        video_hash_to_frames[(video, m["hash"])].add(m["name_a"].split("/")[1])
        video_hash_to_frames[(video, m["hash"])].add(m["name_b"].split("/")[1])

    frames_to_delete: Dict[str, Set[str]] = defaultdict(set)
    for (video, _h), frames in video_hash_to_frames.items():
        for frame in sorted(frames)[1:]:
            frames_to_delete[video].add(frame)

    total = 0
    for video, frames in sorted(frames_to_delete.items()):
        for frame in sorted(frames):
            frame_path = data_dir / ds / "images" / video / frame
            plan.delete_frame_from_video(frame_path, video, frame, f"[{ds}] {video}/{frame}")
            total += 1

    if total > 0:
        logger.info(f"  [{ds}] INTRA-VIDEO: {total} duplicate frames to delete")


def _plan_partial_cross_video_dedup(plan: DeduplicationPlan, ds: str, va: str, vb: str,
                                    cross_matches: List[dict], data_dir: Path):
    """Case 4 (cross part): delete the shared frames from the later video (vb)."""
    frames_to_delete = set()
    for m in cross_matches:
        ma_video = m["name_a"].split("/")[0]
        mb_video = m["name_b"].split("/")[0]
        if tuple(sorted([ma_video, mb_video])) != (va, vb):
            continue
        frames_to_delete.add(m["name_b"].split("/")[1] if mb_video == vb else m["name_a"].split("/")[1])

    for frame in sorted(frames_to_delete):
        frame_path = data_dir / ds / "images" / vb / frame
        plan.delete_frame_from_video(frame_path, vb, frame, f"[{ds}] {vb}/{frame} (dup of {va})")

    if frames_to_delete:
        logger.info(f"  [{ds}] PARTIAL-CROSS: {len(frames_to_delete)} frames deleted from '{vb}' (shared with '{va}')")


# ── Execution ────────────────────────────────────────────────────────────────

def execute_plan(plan: DeduplicationPlan, splits_dir: Path, output_dir: Path, dry_run: bool = False) -> Dict[str, dict]:
    """Write deduplicated split files and delete duplicate data files."""
    stats = {}
    for ds_dir in sorted(splits_dir.iterdir()):
        if not ds_dir.is_dir():
            continue
        ds_name = ds_dir.name
        out_ds_dir = output_dir / ds_name
        removals = plan.split_removals.get(ds_name, set())
        ds_stats = {"removed": {}, "total_removed": 0}

        for list_file in sorted(ds_dir.glob("*_list.txt")):
            split_name = list_file.stem
            with open(list_file) as f:
                entries = [line.strip() for line in f if line.strip()]
            filtered = [e for e in entries if e not in removals] if removals else entries
            removed_count = len(entries) - len(filtered)
            ds_stats["removed"][split_name] = removed_count
            ds_stats["total_removed"] += removed_count

            if not dry_run:
                write_split_file(out_ds_dir / list_file.name, filtered)

            if removed_count > 0:
                logger.info(f"  [{ds_name}/{list_file.name}] {len(entries)} → {len(filtered)} (removed {removed_count})")

        stats[ds_name] = ds_stats

    deleted_count = 0
    for path, _desc in plan.file_deletions:
        if path.exists():
            if dry_run:
                logger.info(f"  [DRY] Would delete: {path}")
            else:
                shutil.rmtree(path) if path.is_dir() else path.unlink()
                logger.debug(f"  Deleted: {path}")
            deleted_count += 1
        else:
            logger.debug(f"  Already missing: {path}")

    logger.info(f"Data files/dirs deleted: {deleted_count}")
    return stats


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Deduplicate SaUS datasets from a check_overlap intra report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _ds = os.environ.get("DATASET_DIR")
    parser.add_argument(
        "--overlap-report", type=str,
        default=(f"{_ds}/overlap_report.json" if _ds else None),
        help="Path to overlap_report.json (default: $DATASET_DIR/overlap_report.json)",
    )
    parser.add_argument(
        "--splits-dir", type=str, default=os.environ.get("ANNOTATION_DIR"),
        help="Original split directory to filter (default: $ANNOTATION_DIR)",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="COPY of the SaUS data to modify in place (images/videos deleted here). "
             "Copy $DATASET_DIR here first to preserve the original.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output dir for deduplicated splits (default: {splits_dir}_dedup)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print actions only; modify nothing")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if not args.overlap_report or not args.splits_dir:
        logger.error("Need --overlap-report and --splits-dir (or $DATASET_DIR / $ANNOTATION_DIR).")
        return
    overlap_path = Path(args.overlap_report)
    splits_dir = Path(args.splits_dir)
    output_dir = Path(args.output_dir) if args.output_dir else Path(str(splits_dir) + "_dedup")

    for p, name in [(overlap_path, "Overlap report"), (splits_dir, "Splits dir")]:
        if not p.exists():
            logger.error(f"{name} not found: {p}")
            return
    if not args.data_dir:
        logger.error("Need --data-dir (a copy of SaUS to modify; preserves the original).")
        return
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}\nCopy $DATASET_DIR to {data_dir} first.")
        return

    with open(overlap_path) as f:
        report = json.load(f)

    logger.info(f"Overlap report: {report['summary']}")
    logger.info(f"Splits dir:  {splits_dir}")
    logger.info(f"Data dir:    {data_dir}")
    logger.info(f"Output dir:  {output_dir}")

    intra_overlaps = [o for o in report["overlaps"] if o["type"] == "intra"]
    logger.info(f"Intra-dataset overlap entries: {len(intra_overlaps)}")

    logger.info("\nBuilding deduplication plan...")
    plan = build_plan(intra_overlaps, data_dir)
    logger.info(f"\n{plan.summary()}")

    if args.dry_run:
        logger.info("\n[DRY RUN] Showing what would be done...\n")
    logger.info(f"\nProcessing split files → {output_dir}")
    stats = execute_plan(plan, splits_dir, output_dir, dry_run=args.dry_run)

    logger.info("\n" + "=" * 60)
    logger.info("DEDUPLICATION SUMMARY")
    logger.info("=" * 60)
    total_removed = 0
    for ds, ds_stats in sorted(stats.items()):
        if ds_stats["total_removed"] > 0:
            parts = ", ".join(f"{k}: -{v}" for k, v in ds_stats["removed"].items() if v > 0)
            logger.info(f"  {ds}: {parts}")
            total_removed += ds_stats["total_removed"]

    logger.info(f"\nSplit entries removed: {total_removed}")
    logger.info(f"Data files/dirs deleted: {len(plan.file_deletions)}")
    logger.info(f"Output splits: {output_dir}")
    if args.dry_run:
        logger.info("[DRY RUN] No files were actually modified.")
    logger.info("Done!")


if __name__ == "__main__":
    main()
