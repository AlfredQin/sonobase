#!/usr/bin/env python3
"""Emit Supplementary Tables S2/S3 from the prediction archive.

Two protocols now coexist in the paper, deliberately, and this script is the
single place either table body is produced so the .tex can never drift from the
records:

  deterministic  exact GT box / RITM centre click, native geometry, no seed.
                 What all 26 downstream analyses (A1-A4, B1-B3, C2, S1-S3,
                 T1-T3) consume, so their clinical endpoints are not
                 seed-dependent.

  noised         box jitter 0.10 / uniform point, native geometry, mean +- SD
                 over 3 seeds. SAM2's own training-time samplers
                 (`sample_box_points(noise=0.1, noise_bound=20)` and
                 `get_next_point(method="uniform")`) -- neither value tuned by
                 us. What Tables 1/2/S2/S3 report.

Bolding marks the actual row leader, not SonoBase unconditionally: a table that
bolds by name cannot tell you when the claim it supports has stopped being true.

Two layouts, same numbers:

  supp  S2/S3 -- dataset x prompt rows, mIoU and Dice per model.
  main  Tables 1/2 -- model rows, dataset columns, mIoU only. These carry no
        +- because ten columns of "78.7$\\pm$0.3" force \\resizebox down to an
        unreadable size; the caption points at S2/S3 for the spread and states
        its bound, which is what a reader needs from a headline table.

Usage:
  python scripts/analysis/make_supp_tables.py --protocol noised --style supp --outdir <dir>
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
SEED_CSV = (SRC / "experiments" / "2026-08-04_eval-protocol-reconciliation"
            / "results" / "prompt_noise_3seed.csv")

BENCH = ["BUSI", "Brachial-Plexus", "C-TRUS", "CAMUS", "HC18", "PFUS", "RegPro", "TG3K"]
EXT = ["ACOUSLIC", "BUS-BRA", "DDTI", "FUGC", "KidneyUS", "LUMINOUS", "MMOTU-3d"]
MODELS = ["sonobase", "medsam2", "sam2_no_ft"]

HEADER = r"""\begin{tabular}{|c|c|cc|cc|cc|}
\hline
\textbf{Dataset} & \textbf{Prompt} & \multicolumn{2}{c|}{\textbf{SonoBase}} &
\multicolumn{2}{c|}{\textbf{MedSAM2}} & \multicolumn{2}{c|}{\textbf{SAM2 (no ft)}} \\
& & \textbf{mIoU} & \textbf{Dice} & \textbf{mIoU} & \textbf{Dice} & \textbf{mIoU} & \textbf{Dice} \\
\hline"""


def deterministic_cell(model: str, ds: str, prompt: str) -> Optional[Tuple[float, float]]:
    p = ARCHIVE / f"{model}_{ds}_{prompt}_0corr" / "per_sample_metrics.csv"
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
    return (100 * st.mean(iou), 100 * st.mean(dice)) if iou else None


def load_noised() -> Dict[Tuple[str, str, str], Tuple[float, float, float, float]]:
    out = {}
    with SEED_CSV.open() as f:
        for r in csv.DictReader(f):
            out[(r["prompt"], r["dataset"], r["model"])] = (
                float(r["miou_mean"]), float(r["miou_sd"]),
                float(r["dice_mean"]), float(r["dice_sd"]),
            )
    return out


def fmt(v: float, sd: Optional[float], lead: bool) -> str:
    s = f"{v:.1f}" if sd is None else rf"{v:.1f}$\pm${sd:.1f}"
    return rf"\textbf{{{s}}}" if lead else s


def build(group: List[str], protocol: str, noised) -> Tuple[str, List[str]]:
    lines = [HEADER]
    lost = []
    acc = {(p, m): [] for p in ("point", "box") for m in MODELS}
    for ds in group:
        for i, prompt in enumerate(("point", "box")):
            vals = {}
            for m in MODELS:
                if protocol == "noised":
                    c = noised.get((prompt, ds, m))
                    vals[m] = None if c is None else (c[0], c[1], c[2], c[3])
                else:
                    c = deterministic_cell(m, ds, prompt)
                    vals[m] = None if c is None else (c[0], None, c[1], None)
            if any(v is None for v in vals.values()):
                lost.append(f"MISSING {ds}/{prompt}")
                continue
            best = max(MODELS, key=lambda m: vals[m][0])
            if best != "sonobase":
                lost.append(f"{ds}/{prompt}: SonoBase {vals['sonobase'][0]:.1f} "
                            f"< {best} {vals[best][0]:.1f}")
            head = rf"{ds} & {prompt.capitalize()}" if i == 0 else rf"& {prompt.capitalize()}"
            cells = []
            for m in MODELS:
                mu, msd, d, dsd = vals[m]
                acc[(prompt, m)].append((mu, msd or 0.0, d, dsd or 0.0))
                cells.append(fmt(mu, msd, m == best))
                cells.append(fmt(d, dsd, m == best))
            lines.append(head + " & " + " & ".join(cells) + r" \\")
    lines.append(r"\hline")
    for i, prompt in enumerate(("point", "box")):
        head = (rf"\textbf{{Average ({len(group)})}} & " if i == 0 else r"& ") + prompt.capitalize()
        means = {m: tuple(st.mean(x) for x in zip(*acc[(prompt, m)])) for m in MODELS}
        best = max(MODELS, key=lambda m: means[m][0])
        cells = []
        for m in MODELS:
            mu, msd, d, dsd = means[m]
            sd_m = msd if protocol == "noised" else None
            sd_d = dsd if protocol == "noised" else None
            cells.append(fmt(mu, sd_m, m == best))
            cells.append(fmt(d, sd_d, m == best))
        lines.append(head + " & " + " & ".join(cells) + r" \\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n", lost


LABEL = {"sonobase": r"\textbf{SonoBase}",
         "medsam2": "MedSAM2 (Hiera-T)     ",
         "sam2_no_ft": "SAM2 (Hiera-B+, no ft)"}
COLS = {"BUSI": "BUSI", "Brachial-Plexus": "B-Plexus", "C-TRUS": "C-TRUS",
        "CAMUS": "CAMUS", "HC18": "HC18", "PFUS": "PFUS", "RegPro": "RegPro",
        "TG3K": "TG3K", "ACOUSLIC": "ACOUSLIC", "BUS-BRA": "BUS-BRA",
        "DDTI": "DDTI", "FUGC": "FUGC", "KidneyUS": "KidneyUS",
        "LUMINOUS": "LUMINOUS", "MMOTU-3d": "MMOTU-3d"}


def build_main(group, protocol, noised):
    """Tables 1/2 layout: model rows, dataset columns, mIoU only."""
    def val(prompt, ds, m):
        if protocol == "noised":
            c = noised.get((prompt, ds, m))
            return None if c is None else c[0]
        c = deterministic_cell(m, ds, prompt)
        return None if c is None else c[0]

    ncol = len(group) + 2
    head = (r"\textbf{Model (Prompt)} & "
            + " & ".join(rf"\textbf{{{COLS[d]}}}" for d in group)
            + r" & \textbf{Avg} \\")
    lines = [r"\begin{tabular}{l" + "c" * len(group) + r"|c}", r"\toprule", head, r"\midrule"]
    lost = []
    for k, prompt in enumerate(("point", "box")):
        if k:
            lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{{ncol}}}{{l}}{{\textit{{{prompt.capitalize()} prompt}}}} \\")
        vals = {m: [val(prompt, d, m) for d in group] for m in MODELS}
        assert not any(v is None for vs in vals.values() for v in vs), f"missing cell in {prompt}"
        avg = {m: st.mean(vals[m]) for m in MODELS}
        for d_i, d in enumerate(group):
            if max(MODELS, key=lambda m: vals[m][d_i]) != "sonobase":
                lost.append(f"{d}/{prompt}")
        for m in ("sam2_no_ft", "medsam2", "sonobase"):
            b = m == "sonobase"
            cells = [fmt(v, None, b) for v in vals[m]] + [fmt(avg[m], None, b)]
            lines.append(f"{LABEL[m]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n", lost


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=("deterministic", "noised"), required=True)
    ap.add_argument("--style", choices=("supp", "main"), default="supp")
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()

    noised = load_noised() if a.protocol == "noised" else {}
    outdir = pathlib.Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    suffix = "native" if a.protocol == "deterministic" else "noised"

    names = (("tabS2", BENCH), ("tabS3", EXT)) if a.style == "supp" else \
            (("tab1", BENCH), ("tab2", EXT))
    for name, group in names:
        body, lost = (build if a.style == "supp" else build_main)(group, a.protocol, noised)
        p = outdir / f"{name}_{suffix}.tex"
        p.write_text(body)
        print(f"wrote {p}")
        if lost:
            print("  !! SonoBase is not the row leader in:")
            for x in lost:
                print(f"     {x}")
        else:
            print("  SonoBase leads every row.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
