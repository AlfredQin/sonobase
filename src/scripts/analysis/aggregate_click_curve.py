#!/usr/bin/env python3
"""Aggregate the click-efficiency curve under the Tables 1-2 prompt protocol.

Consumes the Stage-1 per-sample records from two roots, which together make one
curve under one prompt distribution:

    predictions_prompt_noise/   correction level 0   (already existed)
    predictions_click_curve/    correction levels 1, 3, 5, 7

Both are jittered-box / uniform-point, three seeds — the protocol behind
Tables 1-2. An exact ground-truth prompt evaluated once starts 1-2 points
higher, which is why the curve must be rebuilt rather than reused.

Per dataset the metric is the mean of the `iou` column, exactly as the Tables 1-2
macros were reproduced. Datasets are then macro-averaged within each tier, and
the three seeds averaged last, so the reported SD is seed-to-seed on the macro.

Clicks-to-target is reported as the first measured level that reaches the target,
with the interpolated crossing shown alongside; the measured levels are the only
honest answer, since 2 and 4 clicks were never run.

Usage (from src/):
    python scripts/analysis/aggregate_click_curve.py [--target 80.0] [--out DIR]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics

ROOT_0 = "./experiments/analysis/predictions_prompt_noise"
ROOT_N = "./experiments/analysis/predictions_click_curve"

BENCH = ["BUSI", "Brachial-Plexus", "C-TRUS", "CAMUS", "HC18", "PFUS", "RegPro", "TG3K"]
EXT = ["ACOUSLIC", "BUS-BRA", "DDTI", "FUGC", "KidneyUS", "LUMINOUS", "MMOTU-3d"]
TIERS = {"benchmark": BENCH, "external": EXT}
MODELS = ["sonobase", "medsam2", "sam2_no_ft"]
PROMPTS = ["box", "point"]
SEEDS = [42, 2026, 1337]
CORRS = [0, 1, 3, 5, 7]

# Spec section 7.3: a correction click costs 2 s of clinician time, a box prompt
# 4 s, an initial point click 2 s.
SEC_PER_CLICK, SEC_BOX, SEC_POINT = 2.0, 4.0, 2.0


def run_dir(model: str, ds: str, prompt: str, corr: int, seed: int) -> str:
    suffix = "jit10" if prompt == "box" else "unif"
    root = ROOT_0 if corr == 0 else ROOT_N
    return f"{root}/{model}_{ds}_{prompt}_{corr}corr_{suffix}_s{seed}"


def dataset_miou(model: str, ds: str, prompt: str, corr: int, seed: int):
    f = f"{run_dir(model, ds, prompt, corr, seed)}/per_sample_metrics.csv"
    if not os.path.exists(f):
        return None
    vals = [float(r["iou"]) for r in csv.DictReader(open(f)) if r.get("iou")]
    return 100.0 * sum(vals) / len(vals) if vals else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--target", type=float, default=80.0, help="mIoU%% target")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    curves, missing = {}, []
    for model in MODELS:
        for prompt in PROMPTS:
            for tier, dsets in TIERS.items():
                for corr in CORRS:
                    per_seed = []
                    for seed in SEEDS:
                        vals = [dataset_miou(model, d, prompt, corr, seed) for d in dsets]
                        if any(v is None for v in vals):
                            missing += [(model, d, prompt, corr, seed)
                                        for d, v in zip(dsets, vals) if v is None]
                            continue
                        per_seed.append(sum(vals) / len(vals))
                    if per_seed:
                        curves[(model, prompt, tier, corr)] = (
                            statistics.mean(per_seed),
                            statistics.stdev(per_seed) if len(per_seed) > 1 else 0.0,
                            len(per_seed),
                        )

    if missing:
        print(f"[warn] {len(missing)} missing cells, e.g. {missing[:3]}")

    def cross(model, prompt, tier):
        """First measured level reaching target, plus the interpolated crossing."""
        pts = [(c, curves[(model, prompt, tier, c)][0])
               for c in CORRS if (model, prompt, tier, c) in curves]
        if not pts:
            return None, None
        for i, (c, v) in enumerate(pts):
            if v >= args.target:
                if i == 0:
                    return c, 0.0
                c0, v0 = pts[i - 1]
                interp = c0 + (args.target - v0) * (c - c0) / (v - v0)
                return c, interp
        return None, None

    lines = []
    def emit(s=""):
        print(s); lines.append(s)

    emit("Click-efficiency curve — Tables 1-2 prompt protocol "
         "(jittered box / uniform point, 3 seeds)")
    emit("mIoU %, macro over datasets, mean +/- seed-to-seed SD\n")
    for tier in ("benchmark", "external"):
        for prompt in PROMPTS:
            emit(f"--- {tier} ({len(TIERS[tier])} datasets), {prompt} prompt ---")
            emit(f"{'model':12s} " + " ".join(f"{c:>14d}" for c in CORRS))
            for model in MODELS:
                row = []
                for c in CORRS:
                    e = curves.get((model, prompt, tier, c))
                    row.append(f"{e[0]:8.2f}+/-{e[1]:4.2f}" if e else f"{'--':>14s}")
                emit(f"{model:12s} " + " ".join(row))
            emit("")

    emit(f"Clicks to reach {args.target:.0f}% mIoU  (initial prompt "
         f"{SEC_BOX:.0f}s box / {SEC_POINT:.0f}s point, {SEC_PER_CLICK:.0f}s per click)")
    emit(f"{'model':12s} {'prompt':6s} {'tier':10s} {'start':>7s} {'clicks':>7s} "
         f"{'interp':>7s} {'seconds':>8s}")
    summary = {}
    for model in MODELS:
        for prompt in PROMPTS:
            for tier in ("benchmark", "external"):
                start = curves.get((model, prompt, tier, 0))
                clicks, interp = cross(model, prompt, tier)
                base = SEC_BOX if prompt == "box" else SEC_POINT
                secs = base + SEC_PER_CLICK * clicks if clicks is not None else None
                summary[f"{model}|{prompt}|{tier}"] = dict(
                    start=start[0] if start else None, clicks=clicks,
                    interpolated=interp, seconds=secs)
                emit(f"{model:12s} {prompt:6s} {tier:10s} "
                     f"{start[0] if start else float('nan'):7.2f} "
                     f"{('never' if clicks is None else clicks):>7} "
                     f"{(f'{interp:.2f}' if interp is not None else '--'):>7} "
                     f"{(f'{secs:.0f}' if secs is not None else '--'):>8}")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        payload = {"target": args.target,
                   "protocol": "jittered box (0.10) / uniform point, seeds 42/2026/1337",
                   "curves": {f"{m}|{p}|{t}|{c}": dict(mean=v[0], sd=v[1], n_seeds=v[2])
                              for (m, p, t, c), v in curves.items()},
                   "clicks_to_target": summary}
        with open(f"{args.out}/click_curve_v2.json", "w") as f:
            json.dump(payload, f, indent=2)
        with open(f"{args.out}/click_curve_v2.txt", "w") as f:
            f.write("\n".join(lines) + "\n")
        emit(f"\nwritten: {args.out}/click_curve_v2.{{json,txt}}")


if __name__ == "__main__":
    main()
