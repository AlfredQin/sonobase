#!/usr/bin/env python
"""Fig. 5 replacement — click-efficiency under the Tables 1-2 prompt protocol.

Four panels: {benchmark, external} x {box, point}. One line per model, the 80 %
usability threshold marked, seed-to-seed SD as a band.

Palette is the manuscript's own, reused unchanged so this sits beside the other
figures: SonoBase #009E73, MedSAM2 #D55E00, SAM2 #BBBBBB. That triple was
validated for CVD separation before its first use (worst normal-vision dE 21.6,
worst CVD dE 15.5) and is not re-derived here.

Usage:
    python scripts/analysis/make_click_curve_figure.py \
        --results ../results_click_curve/click_curve_v2.json --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_SB, C_M2, C_S2 = "#009E73", "#D55E00", "#BBBBBB"
MODELS = [("sonobase", "SonoBase", C_SB),
          ("medsam2", "MedSAM2", C_M2),
          ("sam2_no_ft", "SAM2", C_S2)]
INK, INK_MUTED = "#333333", "#6E6E6E"
GRID = dict(color="#E3E3E3", lw=0.6)
CORRS = [0, 1, 3, 5, 7]

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 7, "axes.labelsize": 7.5,
    "axes.titlesize": 8, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.8, "axes.linewidth": 0.7,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "pdf.fonttype": 42,
})


def tidy(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#8A8A8A"); ax.spines["bottom"].set_color("#8A8A8A")
    ax.set_axisbelow(True); ax.yaxis.grid(True, **GRID); ax.xaxis.grid(False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    d = json.load(open(args.results))
    cur, target = d["curves"], d["target"]
    get = lambda m, p, t, c: cur.get(f"{m}|{p}|{t}|{c}")

    fig = plt.figure(figsize=(7.2, 4.4))
    gs = fig.add_gridspec(2, 2, hspace=0.52, wspace=0.24,
                          left=0.085, right=0.985, top=0.90, bottom=0.115)

    panels = [("benchmark", "box", "a", "Benchmark (8 datasets), box"),
              ("external", "box", "b", "External (7 datasets), box"),
              ("benchmark", "point", "c", "Benchmark (8 datasets), point"),
              ("external", "point", "d", "External (7 datasets), point")]

    for k, (tier, prompt, letter, title) in enumerate(panels):
        ax = fig.add_subplot(gs[k // 2, k % 2])
        ax.axhline(target, color="#8A8A8A", lw=0.8, ls="--", zorder=1)
        ax.text(7.05, target, f" {target:.0f}%", va="center", ha="left",
                fontsize=6.0, color=INK_MUTED)
        for mkey, mlabel, colour in MODELS:
            xs, ys, los, his = [], [], [], []
            for c in CORRS:
                e = get(mkey, prompt, tier, c)
                if not e:
                    continue
                xs.append(c); ys.append(e["mean"])
                los.append(e["mean"] - e["sd"]); his.append(e["mean"] + e["sd"])
            if not xs:
                continue
            ax.fill_between(xs, los, his, color=colour, alpha=0.18, lw=0, zorder=2)
            ax.plot(xs, ys, "-o", color=colour, lw=1.6, ms=3.6, mec="white",
                    mew=0.7, zorder=3, label=mlabel)
        ax.set_xticks(CORRS)
        ax.set_xlim(-0.35, 7.35); ax.set_ylim(20, 95)
        ax.set_xlabel("Correction clicks")
        if k % 2 == 0:
            ax.set_ylabel("mIoU (%)")
        ax.set_title(title, pad=5, loc="left")
        tidy(ax)
        ax.text(-0.155 if k % 2 == 0 else -0.115, 1.10, letter, transform=ax.transAxes,
                fontsize=9.5, fontweight="bold", va="top", ha="left")
        if k == 0:
            ax.legend(loc="lower right", frameon=False, handlelength=1.5,
                      handletextpad=0.5, borderpad=0.2, labelspacing=0.3)

    fig.text(0.085, 0.012,
             "Jittered box (10%) / uniformly-sampled point, mean of three seeds; band is "
             "seed-to-seed SD. Levels 2, 4 and 6 were not run.",
             fontsize=5.8, color=INK_MUTED, ha="left")

    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}/click_curve_v2.{ext}", dpi=400,
                    bbox_inches="tight", facecolor="white")
    print(f"written: {args.out}/click_curve_v2.{{pdf,png}}")


if __name__ == "__main__":
    main()
