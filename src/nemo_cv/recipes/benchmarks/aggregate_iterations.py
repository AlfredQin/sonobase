"""B1 helper — fold N single-iteration ``test_metrics.json`` files into one
long-format ``test_metrics_iterations.csv``.

The default ("B1") iterative-correction sweep launches the standard
``test_sam2.py`` recipe ``len(iterations)`` times — one process per
correction-click count — and each run drops a ``test_metrics.json`` /
``test_metrics.csv`` pair under its own
``${checkpoint.save_dir}/results/`` directory. To produce the long-format
CSV that downstream consumers (e.g. the A3 click-efficiency analysis,
the ``compare_b1_vs_b2.sh`` helper) expect, this script consumes those N
JSON files and emits a single CSV with the same schema as the one
written by the B2 ``IterativeEvalRecipe``:

    dataset,iteration,n_samples,<metric_1>,<metric_2>,...,test_loss
    ...
    AGGREGATE_MACRO,<iter>,<total_n>,...
    AGGREGATE_MICRO,<iter>,<total_n>,...

Sharing the schema between the two pipelines means downstream code does
not need to know which workflow produced the table.

CLI
---
::

    python -m nemo_cv.recipes.benchmarks.aggregate_iterations \\
        --inputs <iter>:<path> [<iter>:<path> ...] \\
        --output <csv_path> [--json-output <json_path>]

Example::

    python -m nemo_cv.recipes.benchmarks.aggregate_iterations \\
        --inputs \\
            0:./experiments/.../iter0/results/test_metrics.json \\
            1:./experiments/.../iter1/results/test_metrics.json \\
            3:./experiments/.../iter3/results/test_metrics.json \\
            5:./experiments/.../iter5/results/test_metrics.json \\
            7:./experiments/.../iter7/results/test_metrics.json \\
        --output ./experiments/.../merged/test_metrics_iterations.csv \\
        --json-output ./experiments/.../merged/test_metrics_iterations.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import sys
from typing import Dict, List, Tuple

# Bookkeeping fields we strip out of the per-dataset metric dicts when
# building the long-format CSV column list; everything else is treated
# as a numeric metric to carry through.
_BOOKKEEPING = {"n_samples", "n_iters_per_rank", "elapsed_sec"}


def _parse_inputs(raw: List[str]) -> List[Tuple[int, pathlib.Path]]:
    """Parse ``<iter>:<path>`` argument pairs."""
    out: List[Tuple[int, pathlib.Path]] = []
    for s in raw:
        if ":" not in s:
            raise ValueError(
                f"--inputs entry must be `<iter>:<path>`, got {s!r}"
            )
        iter_str, path_str = s.split(":", 1)
        iter_val = int(iter_str)
        path = pathlib.Path(path_str).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"input JSON not found: {path}")
        out.append((iter_val, path))
    out.sort(key=lambda x: x[0])
    return out


def _round_or_blank(v) -> str:
    if v is None:
        return ""
    try:
        if isinstance(v, float) and math.isnan(v):
            return ""
        return f"{float(v):.6f}"
    except (TypeError, ValueError):
        return str(v)


def aggregate(inputs: List[Tuple[int, pathlib.Path]]) -> Dict:
    """Load each per-iteration JSON, normalise into a single nested dict.

    Returns a dict with the same structure as
    ``IterativeEvalRecipe._write_iteration_results`` writes to JSON, so the
    two pipelines yield interchangeable artifacts.
    """
    if not inputs:
        raise ValueError("--inputs must contain at least one <iter>:<path> entry")

    iterations = [it for it, _ in inputs]
    if len(set(iterations)) != len(iterations):
        raise ValueError(f"duplicate iteration values in --inputs: {iterations}")

    records: Dict[Tuple[str, int], Dict[str, float]] = {}
    metric_names_seen: List[str] = []
    has_loss = False
    checkpoint_path: str = ""
    ds_order: List[str] = []
    seen_ds = set()

    for it, path in inputs:
        with path.open() as f:
            blob = json.load(f)
        if "datasets" not in blob:
            raise ValueError(
                f"{path}: missing top-level `datasets` key — is this a "
                "test_sam2.py output?"
            )
        if not checkpoint_path:
            checkpoint_path = str(blob.get("checkpoint", ""))
        for ds_name, ds_rec in blob["datasets"].items():
            if ds_name not in seen_ds:
                ds_order.append(ds_name)
                seen_ds.add(ds_name)
            rec = {k: v for k, v in ds_rec.items() if k not in _BOOKKEEPING or k == "n_samples"}
            for k in rec:
                if k in {"n_samples", "test_loss"}:
                    continue
                if k not in metric_names_seen:
                    metric_names_seen.append(k)
            if "test_loss" in rec:
                has_loss = True
            records[(ds_name, it)] = rec

    macro: Dict[int, Dict[str, float]] = {it: {} for it in iterations}
    micro: Dict[int, Dict[str, float]] = {it: {} for it in iterations}
    total_per_iter: Dict[int, int] = {}
    for it in iterations:
        valid = [
            (records[(ds, it)], records[(ds, it)]["n_samples"])
            for ds in ds_order
            if (ds, it) in records
        ]
        total_per_iter[it] = sum(n for _, n in valid)
        for name in metric_names_seen:
            vals = [(r[name], n) for r, n in valid if r.get(name) is not None]
            if not vals:
                continue
            macro[it][name] = sum(v for v, _ in vals) / len(vals)
            denom = sum(n for _, n in vals)
            micro[it][name] = sum(v * n for v, n in vals) / max(denom, 1)
        if has_loss:
            losses = [(r["test_loss"], n) for r, n in valid if "test_loss" in r]
            if losses:
                macro[it]["test_loss"] = sum(v for v, _ in losses) / len(losses)
                denom = sum(n for _, n in losses)
                micro[it]["test_loss"] = sum(v * n for v, n in losses) / max(denom, 1)

    return {
        "checkpoint": checkpoint_path,
        "iterations": iterations,
        "n_datasets": len(ds_order),
        "ds_order": ds_order,
        "metric_names": metric_names_seen,
        "has_loss": has_loss,
        "records": records,
        "aggregate_macro": macro,
        "aggregate_micro": micro,
        "total_per_iter": total_per_iter,
    }


def write_csv(out_path: pathlib.Path, agg: Dict) -> None:
    metric_names = agg["metric_names"]
    has_loss = agg["has_loss"]
    loss_col = ["test_loss"] if has_loss else []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "iteration", "n_samples"] + metric_names + loss_col)
        for ds in agg["ds_order"]:
            for it in agg["iterations"]:
                key = (ds, it)
                if key not in agg["records"]:
                    continue
                r = agg["records"][key]
                row = [ds, it, r["n_samples"]] + [_round_or_blank(r.get(m)) for m in metric_names]
                if has_loss:
                    row.append(_round_or_blank(r.get("test_loss")))
                w.writerow(row)
        w.writerow([])
        for it in agg["iterations"]:
            row = (
                ["AGGREGATE_MACRO", it, agg["total_per_iter"][it]]
                + [_round_or_blank(agg["aggregate_macro"][it].get(m)) for m in metric_names]
            )
            if has_loss:
                row.append(_round_or_blank(agg["aggregate_macro"][it].get("test_loss")))
            w.writerow(row)
        for it in agg["iterations"]:
            row = (
                ["AGGREGATE_MICRO", it, agg["total_per_iter"][it]]
                + [_round_or_blank(agg["aggregate_micro"][it].get(m)) for m in metric_names]
            )
            if has_loss:
                row.append(_round_or_blank(agg["aggregate_micro"][it].get("test_loss")))
            w.writerow(row)


def write_json(out_path: pathlib.Path, agg: Dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "checkpoint": agg["checkpoint"],
        "iterations": agg["iterations"],
        "n_datasets": agg["n_datasets"],
        "datasets": {
            ds: {
                str(it): agg["records"].get((ds, it), {})
                for it in agg["iterations"]
            }
            for ds in agg["ds_order"]
        },
        "aggregate_macro": {
            str(it): agg["aggregate_macro"][it] for it in agg["iterations"]
        },
        "aggregate_micro": {
            str(it): agg["aggregate_micro"][it] for it in agg["iterations"]
        },
    }
    with out_path.open("w") as f:
        json.dump(blob, f, indent=2)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--inputs",
        required=True,
        nargs="+",
        help=(
            "List of `<iter>:<path>` pairs. Each `<path>` is a "
            "`test_metrics.json` produced by a single-iteration test run."
        ),
    )
    p.add_argument(
        "--output",
        required=True,
        type=pathlib.Path,
        help="Output CSV path (long-format).",
    )
    p.add_argument(
        "--json-output",
        type=pathlib.Path,
        default=None,
        help=(
            "Optional output JSON path mirroring the structure written by "
            "`IterativeEvalRecipe._write_iteration_results`. When omitted, "
            "only the CSV is written."
        ),
    )
    return p


def main(argv: List[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    inputs = _parse_inputs(args.inputs)
    agg = aggregate(inputs)
    write_csv(args.output, agg)
    print(f"Wrote {args.output}", file=sys.stderr)
    if args.json_output is not None:
        write_json(args.json_output, agg)
        print(f"Wrote {args.json_output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
