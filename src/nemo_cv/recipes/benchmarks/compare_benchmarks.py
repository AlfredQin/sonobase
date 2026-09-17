"""Aggregate per-model `test_metrics.json` files into a single comparison CSV.

Each `test_sam2.py` run writes a `results/test_metrics.json` file that has
the structure:

    {
      "checkpoint": "/abs/path/...",
      "n_datasets": 2,
      "datasets": {
        "BUSI":  {"n_samples": 8,  "miou": 0.42, "dice": 0.55, "test_loss": 0.39},
        "CAMUS": {"n_samples": 1942, "miou": 0.86, "dice": 0.92, "test_loss": 4.5}
      },
      "aggregate_macro": {"miou": 0.64, "dice": 0.74},
      "aggregate_micro": {"miou": 0.86, "dice": 0.92}
    }

This script reads N such files (one per model under comparison), and emits a
single wide CSV with one row per (dataset, metric) and one column per model:

    dataset,metric,sam2_no_ft,medsam2,sonobase
    BUSI,miou,0.4200,0.5100,0.7900
    BUSI,dice,0.5500,0.6300,0.8500
    BUSI,test_loss,0.3900,0.3100,0.1500
    CAMUS,miou,0.8600,0.7800,0.8800
    ...
    AGGREGATE_MACRO,miou,0.6400,0.6450,0.8350
    AGGREGATE_MICRO,miou,0.8584,0.7723,0.8800

Usage (CLI):

    uv run python -m nemo_cv.recipes.benchmarks.compare_benchmarks \\
        --runs \\
          sam2_no_ft=experiments/benchmarks/busi_camus/sam2/p0/results/test_metrics.json \\
          medsam2=experiments/benchmarks/busi_camus/medsam2/p0/results/test_metrics.json \\
          sonobase=experiments/benchmarks/busi_camus/hiera_b_conv_s_conv_t/p0/results/test_metrics.json \\
        --output experiments/benchmarks/busi_camus/comparison.csv

Each `--runs` entry is `<model_label>=<path>`. The label becomes the column
header. Missing values render as the empty string.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import pathlib
import sys
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Bookkeeping fields we don't want to compare across models row-by-row.
_BOOKKEEPING = {"n_samples", "n_iters_per_rank", "elapsed_sec"}


def _parse_run_arg(arg: str) -> Tuple[str, str]:
    """Parse `<label>=<path>` into a `(label, path)` tuple."""
    if "=" not in arg:
        raise ValueError(
            f"--runs entry must be `label=path`, got: {arg!r}. "
            "Example: sam2_no_ft=./experiments/.../test_metrics.json"
        )
    label, path = arg.split("=", 1)
    label, path = label.strip(), path.strip()
    if not label:
        raise ValueError(f"Empty label in --runs entry: {arg!r}")
    if not path:
        raise ValueError(f"Empty path in --runs entry: {arg!r}")
    return label, path


def _load_run(path: str) -> Dict:
    """Load a `test_metrics.json` file, with helpful error messages."""
    p = pathlib.Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"test_metrics.json not found: {p}")
    with p.open() as f:
        return json.load(f)


def _collect_metric_names(runs: Dict[str, Dict]) -> List[str]:
    """Discover the union of metric names across all runs.

    Order: walk runs in argument order; within each run, walk datasets in
    file order; within each dataset, walk keys in file order. First-seen wins.
    """
    names: List[str] = []
    seen = set()
    for run in runs.values():
        for ds_record in run.get("datasets", {}).values():
            for k in ds_record.keys():
                if k in _BOOKKEEPING or k in seen:
                    continue
                names.append(k)
                seen.add(k)
    return names


def _collect_dataset_names(runs: Dict[str, Dict]) -> List[str]:
    """Discover the union of dataset names across all runs (preserving order)."""
    names: List[str] = []
    seen = set()
    for run in runs.values():
        for ds in run.get("datasets", {}).keys():
            if ds in seen:
                continue
            names.append(ds)
            seen.add(ds)
    return names


def _format(v: Optional[float]) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def build_comparison_rows(runs: Dict[str, Dict]) -> List[List[str]]:
    """Build the comparison-table rows.

    Returns a list of rows where the first row is the header. Each row is a
    list of strings.
    """
    labels = list(runs.keys())
    metric_names = _collect_metric_names(runs)
    dataset_names = _collect_dataset_names(runs)

    rows: List[List[str]] = []
    rows.append(["dataset", "metric"] + labels)

    # Per-dataset rows
    for ds in dataset_names:
        for m in metric_names:
            row = [ds, m]
            for label, run in runs.items():
                ds_record = run.get("datasets", {}).get(ds, {})
                row.append(_format(ds_record.get(m)))
            rows.append(row)

    # Aggregate rows: macro and micro, for each metric
    for agg_key in ("aggregate_macro", "aggregate_micro"):
        agg_label = agg_key.upper()
        for m in metric_names:
            row = [agg_label, m]
            any_value = False
            for label, run in runs.items():
                v = run.get(agg_key, {}).get(m)
                if v is not None:
                    any_value = True
                row.append(_format(v))
            # Only emit aggregate rows if at least one model has the value
            # (e.g. test_loss may be absent if loss wasn't computed).
            if any_value:
                rows.append(row)

    return rows


def write_csv(rows: List[List[str]], output_path: str) -> None:
    out = pathlib.Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        for row in rows:
            w.writerow(row)
    logger.info(f"Wrote {out}")


def print_table(rows: List[List[str]]) -> None:
    """Pretty-print the comparison rows to stdout (for terminal-friendly view)."""
    if not rows:
        return
    n_cols = len(rows[0])
    col_w = [max(len(r[i]) for r in rows) for i in range(n_cols)]
    sep = "-+-".join("-" * w for w in col_w)
    for i, row in enumerate(rows):
        line = " | ".join(cell.ljust(col_w[j]) for j, cell in enumerate(row))
        print(line)
        if i == 0:
            print(sep)


def main(argv: Optional[List[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(
        description=(
            "Aggregate per-model test_metrics.json files into a comparison CSV."
        )
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help=(
            "Space-separated list of `label=path` entries. The label becomes "
            "the column header for that model in the output CSV; the path "
            "points at a `test_metrics.json` produced by `test_sam2.py`."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination CSV path. Parent directories will be created if needed.",
    )
    parser.add_argument(
        "--print",
        dest="print_table",
        action="store_true",
        help="Also pretty-print the comparison to stdout.",
    )
    args = parser.parse_args(argv)

    pairs = [_parse_run_arg(s) for s in args.runs]
    labels = [p[0] for p in pairs]
    if len(set(labels)) != len(labels):
        raise ValueError(f"Duplicate model label in --runs: {labels}")

    runs: Dict[str, Dict] = {}
    for label, path in pairs:
        logger.info(f"Loading {label}: {path}")
        runs[label] = _load_run(path)

    rows = build_comparison_rows(runs)
    write_csv(rows, args.output)
    if args.print_table:
        print()
        print_table(rows)


if __name__ == "__main__":
    main()
