"""Deterministic few-shot subset selection.

The same subset must be used for all 3 models at a given (dataset, N, seed),
so that SonoBase, MedSAM2, and SAM2 see identical training images.

We satisfy that with a single canonical helper:

    select_subset(train_ids, N, seed) -> List[str]   # always returns the
        same N elements for the same (train_ids, N, seed) triple.

For each (dataset, N, seed) we materialize TWO files:

  1. A `<DATASET>_N<N>_seed<seed>.txt` (one sample id per line) consumed
     directly by `SaUSRawDataset.file_list_txt` / `JSONRawDataset.file_list_txt`.
     This is what the existing dataset machinery reads.

  2. A sibling `<DATASET>_N<N>_seed<seed>.json` for reproducibility. Contains the resolved sample ids plus
     enough metadata to re-derive the subset from scratch.

Both files end up under `experiments/few_shot/few_shot_splits/`. The two
formats are interchangeable — `load_split_json` returns the same list as
parsing the txt.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pathlib
import random
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SplitSpec:
    """Records exactly what subset was used and how it was derived."""

    dataset: str
    N: int
    seed: int
    sample_ids: List[str]          # the actual N selected ids
    train_pool_size: int           # the size of the pool we sampled from
    train_pool_source: str         # absolute path of the source train_list.txt
    training_unit: str             # "video" or "frame"
    train_pool_hash: str = ""      # sha-256 of the source pool (file order)

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "N": int(self.N),
            "seed": int(self.seed),
            "training_unit": self.training_unit,
            "train_pool_size": int(self.train_pool_size),
            "train_pool_source": self.train_pool_source,
            "train_pool_hash": self.train_pool_hash,
            "sample_ids": list(self.sample_ids),
        }


# ---------------------------------------------------------------------------
# Core selection
# ---------------------------------------------------------------------------


def select_subset(train_ids: List[str], N: int, seed: int) -> List[str]:
    """Select N training examples deterministically.

    Args:
        train_ids: full pool of available training sample ids (one string per id).
        N: number of examples to pick.
        seed: integer RNG seed.

    Returns:
        Sorted list of N selected ids — sorted so the on-disk file is
        deterministic regardless of `random.sample`'s internal order.

    Raises:
        ValueError: if N > len(train_ids).
    """
    if N <= 0:
        raise ValueError(f"select_subset: N must be > 0, got {N}.")
    if N > len(train_ids):
        raise ValueError(
            f"select_subset: N={N} exceeds train pool size ({len(train_ids)}). "
            "Drop N or use a different dataset."
        )
    rng = random.Random(seed)
    # `rng.sample` consumes ids in deterministic order given the seed, so the
    # returned set is identical across all 3 models when called with the
    # same (train_ids, N, seed). Sorting the output makes the on-disk file
    # byte-identical too.
    return sorted(rng.sample(list(train_ids), N))


# ---------------------------------------------------------------------------
# Loading the source train_list
# ---------------------------------------------------------------------------


def load_train_pool(train_list_txt: str) -> List[str]:
    """Read a SaUS-style train_list.txt (one sample id per line).

    Strips empty lines / whitespace; preserves order of the source file.
    """
    path = pathlib.Path(train_list_txt).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Train list not found: {path}")
    with path.open() as f:
        return [line.strip() for line in f if line.strip()]


def compute_pool_hash(pool: List[str]) -> str:
    """SHA-256 of the train pool in file order.

    `select_subset` depends on both the contents AND the order of the pool
    (`random.sample` indexes the sequence), so the hash is taken over the
    order-preserving list — it must catch a reordered train_list.txt, not
    just a changed-content one.
    """
    return hashlib.sha256("\n".join(pool).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Materializing the on-disk subset (txt + json)
# ---------------------------------------------------------------------------


def _safe_basename(dataset: str, N: int, seed: int) -> str:
    return f"{dataset}_N{int(N)}_seed{int(seed)}"


def split_paths(splits_dir: pathlib.Path, dataset: str, N: int, seed: int):
    """Return (txt_path, json_path) for a (dataset, N, seed) triple."""
    base = splits_dir / _safe_basename(dataset, N, seed)
    return base.with_suffix(".txt"), base.with_suffix(".json")


def write_split(
    splits_dir: pathlib.Path,
    spec: SplitSpec,
) -> pathlib.Path:
    """Materialize both the txt (consumed by datasets) and json (for review).

    Returns the path of the .txt file (the one the dataset machinery uses).
    """
    splits_dir.mkdir(parents=True, exist_ok=True)
    txt_path, json_path = split_paths(splits_dir, spec.dataset, spec.N, spec.seed)

    txt_path.write_text("\n".join(spec.sample_ids) + "\n")

    # Atomic JSON write so a kill mid-flush doesn't leave a half file.
    import os as _os
    tmp = json_path.with_suffix(json_path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(spec.to_dict(), f, indent=2)
    _os.replace(tmp, json_path)

    logger.info(
        f"Wrote split: {spec.dataset} N={spec.N} seed={spec.seed} "
        f"unit={spec.training_unit} → {txt_path.name} ({len(spec.sample_ids)} ids)"
    )
    return txt_path


def load_split_json(json_path: pathlib.Path) -> SplitSpec:
    with json_path.open() as f:
        d = json.load(f)
    return SplitSpec(
        dataset=d["dataset"],
        N=int(d["N"]),
        seed=int(d["seed"]),
        sample_ids=list(d["sample_ids"]),
        train_pool_size=int(d["train_pool_size"]),
        train_pool_source=d["train_pool_source"],
        training_unit=d.get("training_unit", "video"),
        train_pool_hash=d.get("train_pool_hash", ""),
    )


# ---------------------------------------------------------------------------
# Top-level helper used by the recipe
# ---------------------------------------------------------------------------


def materialize_subset(
    splits_dir: pathlib.Path,
    dataset: str,
    N: int,
    seed: int,
    train_list_txt: str,
    training_unit: str = "video",
    overwrite: bool = False,
) -> pathlib.Path:
    """Idempotently produce `<dataset>_N<N>_seed<seed>.{txt,json}` and return the txt path.

    A cached split is reused only when the full request (dataset, N, seed,
    training_unit) AND the source train pool (resolved path + a
    content/order hash) all match; otherwise it is regenerated. This is
    critical for cross-model fairness: all 3 models at the same (dataset,
    N, seed) must consume the SAME on-disk subset — and that subset must
    reflect the current train pool, not a stale one.
    """
    txt_path, json_path = split_paths(splits_dir, dataset, N, seed)

    # Load the source pool up front: its content + order fully determine the
    # subset, so the pool hash gates whether a cached split is still valid.
    pool = load_train_pool(train_list_txt)
    pool_hash = compute_pool_hash(pool)
    train_pool_source = str(pathlib.Path(train_list_txt).resolve())

    if not overwrite and json_path.is_file() and txt_path.is_file():
        existing = load_split_json(json_path)
        reusable = (
            existing.dataset == dataset
            and existing.N == N
            and existing.seed == seed
            and existing.training_unit == training_unit
            and existing.train_pool_source == train_pool_source
            and existing.train_pool_hash == pool_hash
        )
        if reusable:
            logger.info(
                f"Reusing existing split: {json_path.name} (n={len(existing.sample_ids)})"
            )
            return txt_path
        # Any difference in the request OR the source pool means the cached
        # subset is no longer valid — regenerate rather than silently reuse.
        logger.warning(
            f"Existing split {json_path.name} does not match the current "
            f"request — regenerating.\n"
            f"  existing: N={existing.N} seed={existing.seed} "
            f"ds={existing.dataset} unit={existing.training_unit} "
            f"pool_size={existing.train_pool_size} "
            f"pool_hash={existing.train_pool_hash[:12] or '(none)'} "
            f"source={existing.train_pool_source}\n"
            f"  current : N={N} seed={seed} ds={dataset} unit={training_unit} "
            f"pool_size={len(pool)} pool_hash={pool_hash[:12]} "
            f"source={train_pool_source}"
        )

    sample_ids = select_subset(pool, N, seed)
    spec = SplitSpec(
        dataset=dataset,
        N=N,
        seed=seed,
        sample_ids=sample_ids,
        train_pool_size=len(pool),
        train_pool_source=train_pool_source,
        training_unit=training_unit,
        train_pool_hash=pool_hash,
    )
    return write_split(splits_dir, spec)
