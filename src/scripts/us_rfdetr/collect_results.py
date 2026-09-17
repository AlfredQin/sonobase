#!/usr/bin/env python3
"""Collect US-RFDETR per-cell results into one tidy CSV + a markdown report.

Every cell writes ``<arm>/<dataset>/<backbone>/results/test_metrics.json``; this
walks those, joins them across arms, and regenerates the tables. Numbers in the
paper come from here rather than from a hand-maintained CSV: a transcribed table
silently drifts from the runs it claims to report.

Usage
-----
    python scripts/us_rfdetr/collect_results.py \\
        --arm bbox_3e-4=/path/to/sonobase/src/experiments/us_rfdetr \\
        --arm segm=./experiments/us_rfdetr_seg \\
        --arm bbox_1e-4=./experiments/us_rfdetr_lr1e-4 \\
        --out ./experiments/us_rfdetr_tables

Arm roots are passed explicitly rather than discovered, so the arms may live in
different directories.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

# The three backbones every cell is run with. Anything else in an arm directory
# (e.g. `sonobase_hotlr_collapse_14311`, archived LR probes kept for the record)
# is deliberately skipped so archived variants never leak into a headline table.
BACKBONES = ["sam2_no_ft", "medsam2", "sonobase"]

# Headline ordering. cva_net is bbox-only (its COCO-video annotations carry no
# `segmentation` field) so it is absent from the segm arm by construction.
DATASET_ORDER = [
    "cva_net", "fetus", "acouslic", "bus_bra",
    "ddti", "fugc", "kidneyus", "luminous", "mmotu_3d",
]

# Excluded from headline means: all three backbones score 0.0 bbox mAP on ~118
# train samples — RF-DETR never converges. Reported, but footnoted, not averaged.
DEGENERATE = {"mmotu_3d"}

# Cells whose LR deviates from their arm's nominal value, because the
# divergence rule fired. Without this, the LR-ablation delta silently compares
# FUGC-at-1e-4 against FUGC-at-1e-4 and reads as a real effect when it is only
# run-to-run noise.
ARM_LR_EXCEPTIONS = {"bbox_3e-4": {"fugc": "1e-4 (divergence fallback)"}}

METRICS = [
    "bbox_mAP", "bbox_mAP_50", "bbox_mAP_75",
    "segm_mAP", "segm_mAP_50", "segm_mAP_75",
]


def read_cell(cell_dir: Path) -> Optional[Dict[str, float]]:
    """Read one cell's test metrics, or None if it never produced any."""
    f = cell_dir / "results" / "test_metrics.json"
    if not f.is_file():
        return None
    with open(f) as fh:
        blob = json.load(fh)

    # `datasets` is keyed by loader name (e.g. "DDTI-test"). Every cell trains
    # on exactly one dataset, so there is a single entry; guard anyway.
    per_ds = blob.get("datasets", {})
    if len(per_ds) != 1:
        raise ValueError(f"{f}: expected exactly 1 test loader, got {sorted(per_ds)}")
    (record,) = per_ds.values()

    # torchmetrics reports -1 for a metric it cannot define — no detections
    # survived the score threshold, or the loader carried no GT of that type.
    # Drop those instead of letting a sentinel average in as a score of -1.
    out = {m: record[m] for m in METRICS if m in record and record[m] >= 0.0}
    # best_epoch / best_val_* come from the val-side selection, and are what the
    # divergence-triggered LR rule is allowed to look at.
    for k in ("best_epoch", "best_val_bbox_mAP"):
        if k in blob:
            out[k] = blob[k]
    return out


def diverged(cell_dir: Path, threshold: float = 0.5) -> Optional[bool]:
    """True if this cell's training collapsed, judged on VALIDATION only.

    The criterion is "validation bbox_mAP stayed exactly 0 for more than
    `threshold` of the epochs" — a failed optimisation run, not a merely weak
    one. It reads only the val curve: never the test metrics, and never which
    backbone produced it, so it applies identically to every method.

    Returns None when there is no val curve to judge.
    """
    f = cell_dir / "validation.jsonl"
    if not f.is_file():
        return None
    vals = []
    with open(f) as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            hit = [v for k, v in rec.items() if k.endswith("/bbox_mAP")]
            if hit:
                vals.append(hit[0])
    if not vals:
        return None
    return sum(1 for v in vals if v == 0) > threshold * len(vals)


def collect(arms: Dict[str, Path]) -> List[dict]:
    rows: List[dict] = []
    for arm, root in arms.items():
        if not root.is_dir():
            raise SystemExit(f"arm '{arm}': no such directory: {root}")
        for ds_dir in sorted(root.iterdir()):
            if not ds_dir.is_dir():
                continue
            for backbone in BACKBONES:
                cell = ds_dir / backbone
                metrics = read_cell(cell)
                if metrics is None:
                    continue
                rows.append({"arm": arm, "dataset": ds_dir.name,
                             "backbone": backbone, "_dir": str(cell),
                             "_diverged": diverged(cell), **metrics})
    return rows


def apply_divergence_rule(rows: List[dict], name: str,
                          default_arm: str, fallback_arm: str) -> List[dict]:
    """Synthesise the protocol-B arm: default LR everywhere, fallback on collapse.

    Single shared LR is the protocol; the fallback is only a failure handler,
    triggered by the val curve and blind to both method and test score.

    Note the published `default_arm` already carries the FUGC fallback baked in
    (that row was re-run at 1e-4 for all three backbones before this rule
    existed). The rule is idempotent there: those cells no longer look diverged,
    so they keep their existing value — which is exactly the intent.
    """
    fb = {(r["dataset"], r["backbone"]): r for r in rows if r["arm"] == fallback_arm}
    out, fired = [], []
    for r in rows:
        if r["arm"] != default_arm:
            continue
        key = (r["dataset"], r["backbone"])
        if r.get("_diverged") and key in fb:
            new = dict(fb[key]); new["arm"] = name
            new["_rule"] = "fallback (3e-4 diverged)"
            fired.append(f"{key[0]}/{key[1]}")
            out.append(new)
        else:
            new = dict(r); new["arm"] = name
            out.append(new)
    if fired:
        print(f"divergence rule fired on {len(fired)} cell(s): {', '.join(sorted(fired))}")
    return out


def _order(datasets) -> List[str]:
    """Known datasets in headline order, then any unknown ones alphabetically."""
    known = [d for d in DATASET_ORDER if d in datasets]
    return known + sorted(datasets - set(known))


def pivot(rows: List[dict], arm: str, metric: str) -> List[str]:
    """One markdown table: datasets down, backbones across, for a single metric."""
    sel = {(r["dataset"], r["backbone"]): r[metric]
           for r in rows if r["arm"] == arm and metric in r}
    if not sel:
        return [f"_no {metric} recorded for arm `{arm}`_", ""]

    datasets = _order({d for d, _ in sel})
    lines = [f"| dataset | {' | '.join(BACKBONES)} |",
             f"|---|{'---|' * len(BACKBONES)}"]

    for ds in datasets:
        cells = []
        vals = {b: sel.get((ds, b)) for b in BACKBONES}
        present = [v for v in vals.values() if v is not None]
        best = max(present) if present else None
        for b in BACKBONES:
            v = vals[b]
            if v is None:
                cells.append("n/a")
            elif v == best and len(present) > 1:
                cells.append(f"**{v:.4f}**")
            else:
                cells.append(f"{v:.4f}")
        label = f"~~{ds}~~" if ds in DEGENERATE else ds
        lines.append(f"| {label} | {' | '.join(cells)} |")

    # Mean is taken only over rows where EVERY backbone produced a value.
    # Averaging each column over whatever subset it happens to have would make
    # the columns incomparable — which is exactly what happens on segm_mAP,
    # where a diverged baseline scores no masks at all and drops out.
    complete = [d for d in datasets
                if d not in DEGENERATE and all((d, b) in sel for b in BACKBONES)]
    means = []
    for b in BACKBONES:
        vs = [sel[(d, b)] for d in complete]
        means.append(f"**{sum(vs) / len(vs):.4f}**" if vs else "—")
    lines.append(f"| **mean ({len(complete)} complete ds)** | {' | '.join(means)} |")
    lines.append("")

    dropped = [d for d in datasets if d not in DEGENERATE and d not in complete]
    if dropped:
        lines.append(f"_Not in the mean — no {metric} on every backbone: "
                     f"{', '.join(dropped)}._")
        lines.append("")
    return lines


def delta_table(rows: List[dict], arm_a: str, arm_b: str, metric: str) -> List[str]:
    """arm_b minus arm_a, per cell — the single-variable comparison."""
    a = {(r["dataset"], r["backbone"]): r[metric]
         for r in rows if r["arm"] == arm_a and metric in r}
    b = {(r["dataset"], r["backbone"]): r[metric]
         for r in rows if r["arm"] == arm_b and metric in r}
    shared = sorted(set(a) & set(b))
    if not shared:
        return [f"_no overlap between `{arm_a}` and `{arm_b}`_", ""]

    exceptions = ARM_LR_EXCEPTIONS.get(arm_a, {})
    lines = [f"| dataset | {' | '.join(BACKBONES)} |",
             f"|---|{'---|' * len(BACKBONES)}"]
    noted = []
    for ds in _order({d for d, _ in shared}):
        cells = []
        for bb in BACKBONES:
            k = (ds, bb)
            cells.append(f"{b[k] - a[k]:+.4f}" if k in a and k in b else "n/a")
        cells = [c.replace("+", "**+") + "**" if c.startswith("+") else c for c in cells]
        label = ds
        if ds in exceptions:
            label = f"{ds} ⚠️"
            noted.append(f"`{ds}` — the `{arm_a}` row already ran at "
                         f"{exceptions[ds]}, so this delta is run-to-run "
                         f"variance, not an LR effect")
        lines.append(f"| {label} | {' | '.join(cells)} |")
    lines.append("")
    for n in noted:
        lines.append(f"> ⚠️ {n}")
    if noted:
        lines.append("")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=PATH",
                    help="arm name and its experiments root; repeatable")
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument("--rule", metavar="NAME=DEFAULT,FALLBACK", default=None,
                    help="synthesise an arm applying the divergence rule: use "
                         "DEFAULT everywhere, falling back to FALLBACK on any "
                         "cell whose DEFAULT run collapsed on validation")
    args = ap.parse_args()

    arms: Dict[str, Path] = {}
    for spec in args.arm:
        if "=" not in spec:
            raise SystemExit(f"--arm expects NAME=PATH, got {spec!r}")
        name, _, path = spec.partition("=")
        arms[name] = Path(path).expanduser()

    rows = collect(arms)
    if not rows:
        raise SystemExit("no test_metrics.json found under any arm")

    arm_order = list(arms)
    if args.rule:
        name, _, spec = args.rule.partition("=")
        default_arm, _, fallback_arm = spec.partition(",")
        for a in (default_arm, fallback_arm):
            if a not in arms:
                raise SystemExit(f"--rule references unknown arm {a!r}")
        rows += apply_divergence_rule(rows, name, default_arm, fallback_arm)
        # Headline first: the protocol arm leads the report.
        arm_order = [name] + arm_order

    args.out.mkdir(parents=True, exist_ok=True)

    # ---- tidy long CSV: one row per cell, every metric a column ----
    csv_path = args.out / "e3_results.csv"
    fields = ["arm", "dataset", "backbone"] + METRICS + ["best_epoch", "best_val_bbox_mAP"]
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["arm"], r["dataset"], r["backbone"])):
            w.writerow(r)

    # ---- markdown report ----
    md: List[str] = ["# E3 US-RFDETR — regenerated results", "",
                     f"Generated from {len(rows)} cells across {len(arms)} arms by "
                     "`scripts/us_rfdetr/collect_results.py`. Bold = best backbone "
                     "in the row. Struck-through rows are degenerate and excluded "
                     "from the mean.", ""]
    for arm in arm_order:
        md += [f"## arm: `{arm}`", ""]
        for metric in ("bbox_mAP", "segm_mAP"):
            if any(r["arm"] == arm and metric in r for r in rows):
                md += [f"### {metric}", ""] + pivot(rows, arm, metric)

    if "bbox_3e-4" in arms and "bbox_1e-4" in arms:
        md += ["## LR ablation — bbox_mAP delta (1e-4 minus 3e-4)", "",
               "Single-variable: same checkpoint, same v2 splits, same 4-GPU "
               "layout. Positive means 1e-4 is better.", ""]
        md += delta_table(rows, "bbox_3e-4", "bbox_1e-4", "bbox_mAP")

    if "bbox_3e-4" in arms and "segm" in arms:
        md += ["## Multitask cost — bbox_mAP delta (seg arm minus detection-only)", "",
               "Positive means adding the mask head *helped* detection.", ""]
        md += delta_table(rows, "bbox_3e-4", "segm", "bbox_mAP")

    md_path = args.out / "e3_results.md"
    md_path.write_text("\n".join(md))

    print(f"wrote {csv_path} ({len(rows)} cells)")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
