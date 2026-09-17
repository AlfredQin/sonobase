#!/usr/bin/env python3
"""Regenerate train/val/test split files for SaUS datasets — **dedup-first**.

Ported + modernized from the old `USSam/.../utils/generate_splits.py`
(2026-05-25). Two changes from the original:

1. **Category membership is derived from the project configs** (single source
   of truth), NOT from stale `dataset_split/*.csv`:
     - benchmark = datasets in BOTH `38_pt_8_bm.yaml` and `8_bm_7_ext.yaml`
       (trained *and* evaluated)
     - external  = datasets in `8_bm_7_ext.yaml` only (eval-only)
     - pretrain  = datasets in `38_pt_8_bm.yaml` only
2. **Dedup-first, on intact data.** Run on the intact `SaUS` root and pass
   `--overlap-report` to dedup the pool *before* splitting:
     - image datasets → drop the non-kept members of each exact-dup group;
     - video datasets → drop redundant near-duplicate *videos* (videos sharing
       byte-identical frames are clustered; one representative is kept).
   So no exact-duplicate image and no duplicate video can straddle splits.
   Intra-video frame dups are left intact — within-split redundancy, not
   leakage. We do NOT read from `SaUSDedup`: its per-frame deletions left some
   videos as 1-frame husks with stale GT (broken for training).

Strategies:
  pretrain   train/val      = 0.95 / 0.05
  benchmark  train/val/test = 0.70 / 0.10 / 0.20
  external   train/test     = 0.10 / 0.90   (train = few-shot pool)
Custom per-dataset splitters: CAMUS (patient-grouped), RegPro (official prefix).
Seed = 42.

Run from `src/`:
  uv run python -m data.utils.generate_splits \\
    --saus-root $WORK/Dataset/Ultrasound/SaUS \\
    --overlap-report $WORK/Dataset/Ultrasound/overlap_report_YYYYMMDD.json \\
    --output-dir $WORK/Dataset/Ultrasound/Annotations/38_pt_8_bm_7_ext_v2 \\
    [--dry-run]
"""

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

from .dataset_split import (
    SplitResult,
    get_sample_names,
    is_video_based,
    split_samples,
    write_split_file,
)

DEFAULT_SEED = 42
PRETRAIN_RATIOS = {"train": 0.95, "val": 0.05}
BENCHMARK_RATIOS = {"train": 0.70, "val": 0.10, "test": 0.20}
EXTERNAL_RATIOS = {"train": 0.10, "test": 0.90}

_REPO_SRC = Path(__file__).resolve().parents[2]  # .../src
DEFAULT_PRETRAIN_CFG = _REPO_SRC / "configs/pretrain/data/38_pt_8_bm.yaml"
DEFAULT_EVAL_CFG = _REPO_SRC / "configs/pretrain/data/8_bm_7_ext.yaml"

SplitFunc = Callable[..., SplitResult]


# ── category membership (from configs) ────────────────────────────────────────

def _datasets_in_config(path: Path) -> Set[str]:
    """Dataset names referenced as `${scratch.dataset_dir}/<NAME>/images`."""
    txt = Path(path).read_text()
    return set(re.findall(r"\$\{scratch\.dataset_dir\}/([^/]+)/images", txt))


def derive_categories(pretrain_cfg: Path, eval_cfg: Path) -> Dict[str, List[str]]:
    train_set = _datasets_in_config(pretrain_cfg)   # 38 pt + 8 bm
    eval_set = _datasets_in_config(eval_cfg)         # 8 bm + 7 ext
    benchmark = sorted(train_set & eval_set)
    external = sorted(eval_set - train_set)
    pretrain = sorted(train_set - set(benchmark))
    return {"pretrain": pretrain, "benchmark": benchmark, "external": external}


# ── optional pool dedup (image-level) ─────────────────────────────────────────

def _image_drop_set(ds: str, overlap_report: Optional[dict]) -> Set[str]:
    """Image datasets: non-kept members of each intra exact-dup group (stems)."""
    if not overlap_report:
        return set()
    drop: Set[str] = set()
    for o in overlap_report.get("overlaps", []):
        if o["type"] != "intra" or o["dataset_a"] != ds:
            continue
        if any("/" in m["name_a"] for m in o["matches"]):  # video handled separately
            continue
        groups: Dict[str, Set[str]] = defaultdict(set)
        for m in o["matches"]:
            groups[m["hash"]].add(Path(m["name_a"]).stem)
            groups[m["hash"]].add(Path(m["name_b"]).stem)
        for _h, mem in groups.items():
            drop.update(sorted(mem)[1:])  # keep first, drop rest
    return drop


def _video_drop_set(ds: str, overlap_report: Optional[dict]) -> Set[str]:
    """Video datasets: redundant near-duplicate videos (whole video names to drop).

    Two videos that share any byte-identical frame would leak that frame if
    split apart. We cluster videos by shared-frame adjacency (connected
    components) and keep one representative per cluster (alphabetically first),
    dropping the rest from the pool. Intra-video frame dups are left alone —
    they're within-split redundancy, not leakage.
    """
    if not overlap_report:
        return set()
    drop: Set[str] = set()
    for o in overlap_report.get("overlaps", []):
        if o["type"] != "intra" or o["dataset_a"] != ds:
            continue
        if not any("/" in m["name_a"] for m in o["matches"]):
            continue
        edges: Dict[str, Set[str]] = defaultdict(set)
        for m in o["matches"]:
            va, vb = m["name_a"].split("/")[0], m["name_b"].split("/")[0]
            if va != vb:
                edges[va].add(vb)
                edges[vb].add(va)
        seen: Set[str] = set()
        for v in sorted(edges):
            if v in seen:
                continue
            comp, stack = [], [v]
            while stack:
                x = stack.pop()
                if x in seen:
                    continue
                seen.add(x)
                comp.append(x)
                stack.extend(y for y in edges[x] if y not in seen)
            rep = min(comp)
            drop.update(c for c in comp if c != rep)
    return drop


def _pool(dataset_path: Path, ds: str, overlap_report: Optional[dict]) -> List[str]:
    samples = get_sample_names(dataset_path)
    drop = _image_drop_set(ds, overlap_report) | _video_drop_set(ds, overlap_report)
    return sorted(s for s in samples if s not in drop) if drop else samples


# ── splitters ─────────────────────────────────────────────────────────────────

def make_default_splitter(category: str, ratios: Dict[str, float], seed: int) -> SplitFunc:
    def _split(ds: str, dataset_path: Path, out_path: Path,
               overlap_report=None, dry_run=False) -> SplitResult:
        if not dataset_path.exists():
            return SplitResult(ds, category, "not_found", message=str(dataset_path))
        samples = _pool(dataset_path, ds, overlap_report)
        if not samples:
            return SplitResult(ds, category, "no_samples")
        splits = split_samples(samples, ratios, seed=seed)
        if not dry_run:
            for name, items in splits.items():
                write_split_file(out_path / f"{name}_list.txt", items)
        return SplitResult(ds, category, "dry_run" if dry_run else "ok",
                           n_train=len(splits.get("train", [])),
                           n_val=len(splits.get("val", [])),
                           n_test=len(splits.get("test", [])),
                           n_total=len(samples), is_video=is_video_based(dataset_path))
    return _split


def split_CAMUS(ds: str, dataset_path: Path, out_path: Path,
                overlap_report=None, dry_run=False) -> SplitResult:
    """Patient-grouped 0.7/0.1/0.2 so a patient's 2CH+4CH stay in one split."""
    images_dir = dataset_path / "images"
    if not images_dir.exists():
        return SplitResult(ds, "benchmark", "no_samples")
    folders = sorted(d.name for d in images_dir.iterdir() if d.is_dir())
    pids = sorted({f.split("_")[0] for f in folders})
    rng = random.Random(DEFAULT_SEED)
    rng.shuffle(pids)
    n = len(pids)
    n_tr, n_val = int(n * 0.7), int(n * 0.1)
    sets = {"train": set(pids[:n_tr]), "val": set(pids[n_tr:n_tr + n_val]), "test": set(pids[n_tr + n_val:])}
    out = {k: sorted(f for f in folders if f.split("_")[0] in v) for k, v in sets.items()}
    if not dry_run:
        for name, items in out.items():
            write_split_file(out_path / f"{name}_list.txt", items)
    return SplitResult(ds, "benchmark", "dry_run" if dry_run else "ok",
                       n_train=len(out["train"]), n_val=len(out["val"]),
                       n_test=len(out["test"]), n_total=len(folders), is_video=True)


def split_RegPro(ds: str, dataset_path: Path, out_path: Path,
                 overlap_report=None, dry_run=False) -> SplitResult:
    """Keep official val (`val_*`) as test; split official train (`train_*`) 0.9/0.1."""
    samples = _pool(dataset_path, ds, overlap_report)
    if not samples:
        return SplitResult(ds, "benchmark", "no_samples")
    official_val = sorted(s for s in samples if s.startswith("val_"))
    official_train = sorted(s for s in samples if s.startswith("train_"))
    sp = split_samples(official_train, {"train": 0.9, "val": 0.1}, seed=DEFAULT_SEED)
    out = {"train": sp["train"], "val": sp["val"], "test": official_val}
    if not dry_run:
        for name, items in out.items():
            write_split_file(out_path / f"{name}_list.txt", items)
    return SplitResult(ds, "benchmark", "dry_run" if dry_run else "ok",
                       n_train=len(out["train"]), n_val=len(out["val"]),
                       n_test=len(out["test"]), n_total=len(samples),
                       is_video=is_video_based(dataset_path))


SPLIT_OVERRIDES: Dict[str, SplitFunc] = {"CAMUS": split_CAMUS, "RegPro": split_RegPro}


# ── orchestration ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Regenerate dedup-first train/val/test splits")
    p.add_argument("--saus-root", required=True, help="Deduplicated data root (e.g. SaUSDedup)")
    p.add_argument("--output-dir", required=True, help="Annotation dir to write <ds>/<split>_list.txt")
    p.add_argument("--pretrain-config", default=str(DEFAULT_PRETRAIN_CFG))
    p.add_argument("--eval-config", default=str(DEFAULT_EVAL_CFG))
    p.add_argument("--overlap-report", default=None, help="Optional: drop image dups from the pool")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    saus_root = Path(args.saus_root)
    out_root = Path(args.output_dir)
    cats = derive_categories(Path(args.pretrain_config), Path(args.eval_config))
    report = json.load(open(args.overlap_report)) if args.overlap_report else None

    defaults = {
        "pretrain": make_default_splitter("pretrain", PRETRAIN_RATIOS, args.seed),
        "benchmark": make_default_splitter("benchmark", BENCHMARK_RATIOS, args.seed),
        "external": make_default_splitter("external", EXTERNAL_RATIOS, args.seed),
    }

    print(f"saus-root : {saus_root}")
    print(f"output    : {out_root}")
    print(f"categories: pretrain={len(cats['pretrain'])} benchmark={len(cats['benchmark'])} external={len(cats['external'])}")
    pool_src = f"overlap-report ({args.overlap_report})" if report else "data-root only (assumed deduplicated)"
    print(f"dedup pool: {pool_src}")
    print(f"dry-run   : {args.dry_run}\n")

    for category, names in cats.items():
        print(f"[{category.upper()}] {len(names)} datasets")
        for ds in names:
            out_path = out_root / ds
            if not args.dry_run:
                out_path.mkdir(parents=True, exist_ok=True)
            fn = SPLIT_OVERRIDES.get(ds, defaults[category])
            r = fn(ds, saus_root / ds, out_path, overlap_report=report, dry_run=args.dry_run)
            ct = f"train={r.n_train} val={r.n_val} test={r.n_test} total={r.n_total}"
            print(f"  [{r.status:7}] {ds:<22} {'video' if r.is_video else 'image':5} {ct}")
    print("\nDone.")


if __name__ == "__main__":
    main()
