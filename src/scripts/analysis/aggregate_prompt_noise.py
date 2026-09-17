#!/usr/bin/env python3
"""Aggregate the 3-seed randomised-prompt study.

Two arms with different jobs:

  point  center (deterministic, every published run) -> uniform, 3 seeds.
         This is a PROTOCOL CHANGE. The quantity of interest is the
         *differential* gain: if all three models gain equally, the prompt is
         merely more realistic and the paper's story is untouched; only a
         differential is a result.

  box    jitter 0.10, 3 seeds, against the same protocol at one seed. This is
         NOT a protocol change -- it is seed replication, and its job is to put
         an SD on numbers previously reported from a single draw.

Writes a tidy CSV and prints the tables. Aggregation matches the paper: macro
over samples within a dataset, then unweighted macro over datasets.

Usage:
  python scripts/analysis/aggregate_prompt_noise.py [--out results.csv]
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import statistics as st
import sys
from typing import Dict, List, Optional, Tuple

SRC = pathlib.Path(__file__).resolve().parents[2]
ARCHIVE = SRC / "experiments" / "analysis" / "predictions"
NOISE = SRC / "experiments" / "analysis" / "predictions_prompt_noise"

BENCH = ["BUSI", "Brachial-Plexus", "C-TRUS", "CAMUS", "HC18", "PFUS", "RegPro", "TG3K"]
EXT = ["ACOUSLIC", "BUS-BRA", "DDTI", "FUGC", "KidneyUS", "LUMINOUS", "MMOTU-3d"]
DATASETS = BENCH + EXT
MODELS = ["sonobase", "medsam2", "sam2_no_ft"]
SEEDS = [42, 2026, 1337]
SHORT = {"sonobase": "SB", "medsam2": "MS2", "sam2_no_ft": "S2"}


def cell(run: pathlib.Path) -> Optional[Tuple[float, float, int]]:
    """(mIoU, Dice, n) macro over samples, or None if the run is absent."""
    p = run / "per_sample_metrics.csv"
    if not p.exists():
        return None
    iou: List[float] = []
    dice: List[float] = []
    with p.open() as f:
        for r in csv.DictReader(f):
            try:
                iou.append(float(r["iou"]))
                dice.append(float(r["dice"]))
            except (ValueError, KeyError):
                continue
    if not iou:
        return None
    return 100 * st.mean(iou), 100 * st.mean(dice), len(iou)


def seeded(model: str, ds: str, prompt: str) -> List[Tuple[float, float, int]]:
    suffix = "jit10" if prompt == "box" else "unif"
    out = []
    for s in SEEDS:
        c = cell(NOISE / f"{model}_{ds}_{prompt}_0corr_{suffix}_s{s}")
        if c is not None:
            out.append(c)
    return out


def baseline(model: str, ds: str, prompt: str) -> Optional[Tuple[float, float, int]]:
    """The comparison arm: center point / single-seed jit10 box."""
    name = f"{model}_{ds}_{prompt}_0corr" + ("_jit10" if prompt == "box" else "")
    return cell(ARCHIVE / name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="write the tidy CSV here")
    a = ap.parse_args()

    rows = []
    missing = []
    for prompt in ("point", "box"):
        for ds in DATASETS:
            for m in MODELS:
                cs = seeded(m, ds, prompt)
                if len(cs) != len(SEEDS):
                    missing.append(f"{m}/{ds}/{prompt}: {len(cs)}/{len(SEEDS)} seeds")
                    continue
                base = baseline(m, ds, prompt)
                ious = [c[0] for c in cs]
                dices = [c[1] for c in cs]
                ns = {c[2] for c in cs}
                rows.append({
                    "prompt": prompt, "dataset": ds, "model": m,
                    "n": cs[0][2],
                    "n_consistent": len(ns) == 1,
                    "miou_mean": st.mean(ious), "miou_sd": st.stdev(ious),
                    "dice_mean": st.mean(dices), "dice_sd": st.stdev(dices),
                    "miou_base": base[0] if base else None,
                    "dice_base": base[1] if base else None,
                })
    if missing:
        print(f"MISSING {len(missing)} cells:")
        for x in missing[:12]:
            print(f"  {x}")
        print()
    bad_n = [r for r in rows if not r["n_consistent"]]
    if bad_n:
        print(f"WARNING: {len(bad_n)} cells have differing item counts across seeds "
              "-- the seeds are not evaluating the same items:")
        for r in bad_n[:8]:
            print(f"  {r['model']}/{r['dataset']}/{r['prompt']}")
        print()

    idx = {(r["prompt"], r["dataset"], r["model"]): r for r in rows}

    def macro(prompt: str, model: str, group: List[str], key: str) -> Optional[float]:
        vals = [idx[(prompt, d, model)][key] for d in group
                if (prompt, d, model) in idx and idx[(prompt, d, model)][key] is not None]
        return st.mean(vals) if len(vals) == len(group) else None

    for prompt in ("point", "box"):
        arm = ("center -> uniform (PROTOCOL CHANGE)" if prompt == "point"
               else "jitter 0.10, 1 seed -> 3 seeds (seed replication)")
        print("=" * 96)
        print(f"{prompt.upper()}  —  {arm}")
        print("=" * 96)
        print(f"{'dataset':<17}" + "".join(f"{SHORT[m]+' mean±sd (Δ vs base)':>26}" for m in MODELS))
        for ds in DATASETS:
            line = f"{ds:<17}"
            for m in MODELS:
                r = idx.get((prompt, ds, m))
                if r is None:
                    line += f"{'--':>26}"
                    continue
                d = ("" if r["miou_base"] is None
                     else f" ({r['miou_mean'] - r['miou_base']:+5.2f})")
                line += f"{r['miou_mean']:12.2f}±{r['miou_sd']:4.2f}{d:>9}"
            print(line)
        for name, group in (("Bench-8", BENCH), ("Ext-7", EXT), ("All-15", DATASETS)):
            line = f"{name+' macro':<17}"
            for m in MODELS:
                mu = macro(prompt, m, group, "miou_mean")
                bs = macro(prompt, m, group, "miou_base")
                sd = macro(prompt, m, group, "miou_sd")
                if mu is None:
                    line += f"{'--':>26}"
                    continue
                d = "" if bs is None else f" ({mu - bs:+5.2f})"
                line += f"{mu:12.2f}±{sd:4.2f}{d:>9}"
            print(line)
        print()

        # Leader check — the claim "SonoBase leads every dataset" lives or dies here.
        lost = []
        for ds in DATASETS:
            vals = {m: idx[(prompt, ds, m)]["miou_mean"] for m in MODELS
                    if (prompt, ds, m) in idx}
            if len(vals) < 3:
                continue
            best = max(vals, key=vals.get)
            if best != "sonobase":
                lost.append((ds, vals["sonobase"], best, vals[best]))
        print(f"  cells where SonoBase is NOT the leader ({prompt}):")
        print("    none" if not lost else "")
        for ds, sb, b, bv in lost:
            print(f"    {ds:<17} SonoBase {sb:5.2f}  <  {SHORT[b]} {bv:5.2f}  ({sb-bv:+.2f})")
        print()

    # Seed spread — is 3 enough?
    print("=" * 96)
    print("SEED SPREAD  (per-cell SD across the 3 seeds, mIoU)")
    print("=" * 96)
    for prompt in ("point", "box"):
        sds = [r["miou_sd"] for r in rows if r["prompt"] == prompt]
        if not sds:
            continue
        worst = sorted((r for r in rows if r["prompt"] == prompt),
                       key=lambda r: -r["miou_sd"])[:4]
        print(f"  {prompt:<6} median {st.median(sds):.2f}   mean {st.mean(sds):.2f}   "
              f"max {max(sds):.2f}")
        for r in worst:
            print(f"           {r['model']}/{r['dataset']}: SD {r['miou_sd']:.2f} "
                  f"(mean {r['miou_mean']:.2f})")
    print()

    if a.out:
        p = pathlib.Path(a.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {len(rows)} rows -> {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
