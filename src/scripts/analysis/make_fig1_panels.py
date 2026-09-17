#!/usr/bin/env python
"""Generate the Figure 1 panels: a (SonoCorpus overview), c (evaluation framework),
d (representative qualitative results).

Panel b is composed in LaTeX from two existing artwork files (Overall.png and
Image_Pyramid_Encoder.png), so nothing is generated for it here.

Corpus totals are recomputed from the deduplicated split lists + dataset_info.json
rather than hard-coded, so the banner cannot drift from the text again.

Colours are Okabe-Ito throughout (colourblind-safe by construction); the model
colours match the rest of the manuscript figures.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import make_paper_figures as M   # style, palette, prediction loaders

OUT = M.OUT
ANN = os.environ.get("ANNOTATION_DIR", "")   # split lists, one directory per dataset
RAW = os.environ.get("DATASET_DIR", "")      # converted datasets (dataset_info.json per dataset)

# Okabe-Ito
OI = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#666666"]

BENCH = set(M.ID8)
EXT   = set(M.OOD7)

# Descriptive metadata (clinical application / country / vendor) is not carried in
# dataset_info.json; these counts are the ones reported in the manuscript and in the
# previous overview artwork, retained verbatim so the figure does not invent data.
CLINICAL = [("Breast", 11), ("Cardiac", 7), ("Thyroid", 6), ("Fetal", 5),
            ("Nerve/Plexus", 3), ("MSK", 3), ("Cervix", 2), ("Lung", 2),
            ("Prostate", 2), ("Liver", 2)]
COUNTRY  = [("China", 19), ("USA", 5), ("Germany", 4), ("Spain", 3), ("Canada", 3),
            ("UK", 3), ("Netherlands", 2), ("Poland", 2)]
VENDOR   = [("GE", 15), ("Siemens", 7), ("Philips", 6), ("Mindray", 4),
            ("Toshiba", 3), ("Esaote", 3), ("Hitachi", 3), ("Canon", 1)]
FORMATS  = [("2D images", 36), ("Video", 12), ("3D volumes", 5)]


# ---------------------------------------------------------------- corpus stats
def corpus():
    """Deduplicated frame / mask totals and per-tier composition, from the data."""
    if not (ANN and RAW):
        raise SystemExit("set ANNOTATION_DIR (split lists) and DATASET_DIR (converted datasets)")
    tier = {"Pretrain": [0, 0], "Benchmark": [0, 0], "External": [0, 0]}   # frames, ndatasets
    frames = masks = 0
    for ds in sorted(os.listdir(ANN)):
        p = os.path.join(ANN, ds)
        if not os.path.isdir(p):
            continue
        n = sum(sum(1 for l in open(os.path.join(p, f)) if l.strip())
                for f in ("train_list.txt", "val_list.txt", "test_list.txt")
                if os.path.exists(os.path.join(p, f)))
        j = json.load(open(f"{RAW}/{ds}/dataset_info.json"))
        nv = j["num_videos"] or 1
        f_ = j["total_frames"] * n / nv
        masks += j["total_annotations"] * n / nv
        frames += f_
        t = "Benchmark" if ds in BENCH else ("External" if ds in EXT else "Pretrain")
        tier[t][0] += f_; tier[t][1] += 1
    return frames, masks, tier


def _hbar(ax, items, color, title, xlabel):
    labs = [k for k, _ in items][::-1]
    vals = [v for _, v in items][::-1]
    y = np.arange(len(labs))
    ax.barh(y, vals, color=color, height=0.68)
    ax.set_yticks(y); ax.set_yticklabels(labs)
    for yi, v in zip(y, vals):
        ax.text(v + max(vals) * 0.02, yi, str(v), va="center", fontsize=5.6, color="#444444")
    ax.set_xlim(0, max(vals) * 1.16)
    ax.set_title(title, fontsize=7, pad=3)
    ax.set_xlabel(xlabel, fontsize=6)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#8A8A8A"); ax.spines["bottom"].set_color("#8A8A8A")
    ax.set_axisbelow(True); ax.xaxis.grid(True, color="#E8E8E8", lw=0.6); ax.yaxis.grid(False)
    ax.tick_params(labelsize=6)


def _donut(ax, items, colors, title):
    """Donut with a legend rather than inline labels -- small wedges collide badly
    when the percentage is drawn on the ring."""
    vals = [v for _, v in items]; labs = [k for k, _ in items]
    tot = sum(vals)
    w, _ = ax.pie(vals, colors=colors[:len(vals)], startangle=90, radius=1.0,
                  wedgeprops=dict(width=0.44, edgecolor="white", linewidth=1.1))
    ax.set_title(title, fontsize=6.8, pad=2)
    ax.legend(w, [f"{l}  {100*v/tot:.1f}%" for l, v in zip(labs, vals)],
              loc="upper center", bbox_to_anchor=(0.5, -0.02), frameon=False,
              fontsize=5.5, handlelength=0.85, handleheight=0.85,
              borderpad=0.1, labelspacing=0.28, handletextpad=0.45)


# ---------------------------------------------------------------- PANEL a
def fig1a():
    frames, masks, tier = corpus()
    fig = plt.figure(figsize=(7.3, 2.18))
    gs = fig.add_gridspec(2, 4, left=0.112, right=0.985, top=0.672, bottom=0.165,
                          wspace=0.70, hspace=1.15,
                          width_ratios=[1.12, 0.72, 0.80, 1.12])

    banner = (f"{frames/1000:.0f}k images / frames / slices     "
              f"{masks/1e6:.2f}M segmentation masks     "
              f"53 public datasets     24 clinical applications")
    fig.text(0.5, 0.925, "SonoCorpus", ha="center", va="center",
             fontsize=10.5, fontweight="bold", color="#111111")
    fig.text(0.5, 0.838, banner, ha="center", va="center", fontsize=6.9, color="#333333")
    fig.patches.append(FancyBboxPatch(
        (0.055, 0.795), 0.89, 0.175, transform=fig.transFigure,
        boxstyle="round,pad=0.012,rounding_size=0.014",
        fc="#F2F7FB", ec="#9DC3DE", lw=0.8, zorder=-1))

    _hbar(fig.add_subplot(gs[:, 0]), CLINICAL, OI[0],
          "Clinical coverage (top 10)", "Number of datasets")
    _donut(fig.add_subplot(gs[0, 1]), FORMATS, [OI[2], OI[1], OI[5]], "Data formats\n(datasets)")
    _donut(fig.add_subplot(gs[1, 1]),
           [("Pretrain", tier["Pretrain"][0]), ("Benchmark", tier["Benchmark"][0]),
            ("External", tier["External"][0])],
           [OI[4], OI[5], OI[3]], "Evaluation tiers\n(by frames)")
    _hbar(fig.add_subplot(gs[:, 2]), COUNTRY, OI[2],
          "Geographic diversity (top 8)", "Number of datasets")
    _hbar(fig.add_subplot(gs[:, 3]), VENDOR, OI[3],
          "Scanner vendors (top 8)", "Number of datasets")

    fig.text(0.006, 0.985, "a", fontsize=10, fontweight="bold", va="top", ha="left")
    fig.savefig(f"{OUT}/fig1a_sonocorpus_overview.pdf")
    fig.savefig(f"{OUT}/fig1a_sonocorpus_overview.png", dpi=240)
    plt.close(fig)
    print(f"  fig1a_sonocorpus_overview.pdf  ({frames:,.0f} frames / {masks:,.0f} masks)")


# ---------------------------------------------------------------- PANEL c
def fig1c():
    frames, masks, tier = corpus()
    fig = plt.figure(figsize=(7.1, 1.58))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    tiers = [
        ("Pretrain", f"{tier['Pretrain'][1]} datasets", f"{tier['Pretrain'][0]/1000:.0f}k frames",
         "training data", OI[4]),
        ("Benchmark", f"{tier['Benchmark'][1]} datasets", f"{tier['Benchmark'][0]/1000:.0f}k frames",
         "held out; cross-modality", OI[5]),
        ("External", f"{tier['External'][1]} datasets", f"{tier['External'][0]/1000:.1f}k frames",
         "fully external; domain shift", OI[3]),
    ]
    BX, BW, BH = 2.5, 25.5, 20.0
    ys = [70, 43, 16]
    for (name, nds, nfr, sub, col), y in zip(tiers, ys):
        ax.add_patch(FancyBboxPatch((BX, y), BW, BH, boxstyle="round,pad=0.4,rounding_size=1.6",
                                    fc=col, ec="none", alpha=0.16, zorder=1))
        ax.add_patch(FancyBboxPatch((BX, y), BW, BH, boxstyle="round,pad=0.4,rounding_size=1.6",
                                    fc="none", ec=col, lw=1.15, zorder=2))
        ax.text(BX + 1.6, y + BH - 5.0, name, fontsize=8.2, fontweight="bold", color="#111111")
        ax.text(BX + 1.6, y + BH - 10.2, f"{nds} · {nfr}", fontsize=6.4, color="#333333")
        ax.text(BX + 1.6, y + BH - 15.0, sub, fontsize=6.0, color="#666666", style="italic")

    analyses = [
        ("Segmentation accuracy", "Tables 1-2"),
        ("Clinical measurement", "Fig. 3"),
        ("Catastrophic-failure rescue", "Fig. 4"),
        ("Workflow efficiency", "Fig. 5"),
        ("Few-shot adaptation", "Table 5"),
    ]
    AX_, AW, AH = 63.0, 34.5, 12.6
    ay = np.linspace(78, 4, len(analyses))
    for (lab, ref), y in zip(analyses, ay):
        ax.add_patch(FancyBboxPatch((AX_, y), AW, AH, boxstyle="round,pad=0.35,rounding_size=1.3",
                                    fc="#F6F6F6", ec="#B9B9B9", lw=0.85, zorder=2))
        ax.text(AX_ + 1.7, y + AH / 2 + 1.5, lab, fontsize=6.8, va="center", color="#111111")
        ax.text(AX_ + 1.7, y + AH / 2 - 3.6, ref, fontsize=5.6, va="center", color="#777777")

    # Pretrain is training data, not an evaluation tier: it trains the model and
    # stops there. Only Benchmark and External feed the downstream analyses.
    ax.add_patch(FancyBboxPatch((BX + BW + 4.0, ys[0] + 4.0), 20.0, 11.5,
                                boxstyle="round,pad=0.35,rounding_size=1.3",
                                fc="#FFFFFF", ec="#888888", lw=0.9, ls="--", zorder=2))
    ax.text(BX + BW + 14.0, ys[0] + 9.8, "SonoBase\npretraining", fontsize=6.3,
            ha="center", va="center", color="#333333")
    ax.add_patch(FancyArrowPatch((BX + BW + 0.6, ys[0] + BH / 2),
                                 (BX + BW + 3.4, ys[0] + BH / 2 - 0.5),
                                 arrowstyle="-|>", mutation_scale=7, lw=1.0,
                                 color=tiers[0][4], zorder=1))

    hub = (BX + BW + 8.0, 33)
    for (_, _, _, _, col), y in zip(tiers[1:], ys[1:]):
        ax.add_patch(FancyArrowPatch((BX + BW + 0.6, y + BH / 2), hub,
                                     arrowstyle="-", lw=1.05, color=col, alpha=0.85, zorder=1))
    for y in ay:
        ax.add_patch(FancyArrowPatch(hub, (AX_ - 0.6, y + AH / 2),
                                     arrowstyle="-|>", mutation_scale=7,
                                     lw=0.85, color="#999999", zorder=1))
    ax.plot(*hub, marker="o", ms=4.2, color="#666666", zorder=3)
    ax.text(hub[0] + 1.0, hub[1] - 10.0, "evaluated on", fontsize=5.8, ha="center", color="#666666")

    fig.text(0.006, 0.975, "c", fontsize=10, fontweight="bold", va="top", ha="left")
    fig.savefig(f"{OUT}/fig1c_evaluation_framework.pdf")
    fig.savefig(f"{OUT}/fig1c_evaluation_framework.png", dpi=240)
    plt.close(fig)
    print("  fig1c_evaluation_framework.pdf")


# ---------------------------------------------------------------- PANEL d
def _pick_typical(d):
    """Median-IoU SonoBase case for one dataset -- a representative result, not a
    hand-picked best. Requires all three models to have scored the same image."""
    sb, m2, s2 = (M._per_image(m, d) for m in ("sonobase", "medsam2", "sam2_no_ft"))
    if not all([sb, m2, s2]):
        return None
    keys = sorted(set(sb) & set(m2) & set(s2))
    if not keys:
        return None
    keys.sort(key=lambda k: sb[k])
    return keys[len(keys) // 2]


def fig1d(rows, nblocks=2):
    """Qualitative grid. Anatomies are split across `nblocks` side-by-side blocks of
    five columns each, so six anatomies occupy three rows rather than six -- stacked
    six-deep the float is taller than the page."""
    cases = []
    for d, anat in rows:
        k = _pick_typical(d)
        c = M._load_case(d, k) if k else None
        if c is None:
            print(f"    ! {d}: no renderable case, skipped"); continue
        cases.append((d, anat, c))
    if not cases:
        return

    nrow = int(np.ceil(len(cases) / nblocks))
    figw = 7.1
    LEFT, GUT, TOP, BOT = 0.058, 0.040, 0.043, 0.012          # figure-width fractions
    cellw = (1 - LEFT * nblocks - GUT * (nblocks - 1)) / (5 * nblocks) * figw
    rowh = []
    for r in range(nrow):
        # guard before indexing: in a comprehension the `if` is applied after the
        # iterable is built, so the index has to be filtered in its own pass.
        idxs = [k_ * nrow + r for k_ in range(nblocks) if k_ * nrow + r < len(cases)]
        ars = [cases[i][2]["img"].shape[0] / cases[i][2]["img"].shape[1] for i in idxs]
        rowh.append(cellw * float(np.clip(np.mean(ars) if ars else 0.8, 0.60, 1.00)))
    figh = (TOP + BOT) * figw + sum(rowh)
    fig = plt.figure(figsize=(figw, figh))

    heads = ["Input", "GT", "SonoBase", "MedSAM2", "SAM2"]
    order = [("sonobase", M.C_SB), ("medsam2", M.C_M2), ("sam2_no_ft", M.C_S2)]
    for b in range(nblocks):
        sub = cases[b * nrow:(b + 1) * nrow]
        if not sub:
            continue
        l = LEFT + b * (5 * cellw / figw + GUT)
        gs = fig.add_gridspec(nrow, 5, height_ratios=rowh,
                              left=l, right=l + 5 * cellw / figw,
                              top=1 - TOP * figw / figh, bottom=BOT * figw / figh,
                              wspace=0.045, hspace=0.09)
        _block(fig, gs, sub, heads, order)

    fig.text(0.002, 0.995, "d", fontsize=10, fontweight="bold", va="top", ha="left")
    fig.savefig(f"{OUT}/fig1d_qualitative.pdf")
    fig.savefig(f"{OUT}/fig1d_qualitative.png", dpi=240)
    plt.close(fig)
    print(f"  fig1d_qualitative.pdf  ({len(cases)} anatomies, {nrow} rows x {nblocks} blocks)")


def _block(fig, gs, cases, heads, order):
    """Draw one 5-column block: Input | GT | SonoBase | MedSAM2 | SAM2."""
    for i, (d, anat, c) in enumerate(cases):
        cols = [(None, None), (c["gt"], M.C_GT)] + [(c["preds"][m], col) for m, col in order]
        for j, (mask, col) in enumerate(cols):
            ax = fig.add_subplot(gs[i, j])
            M._draw(ax, c["img"], mask, col, c["pt"] if j == 0 else None)
            if i == 0:
                ax.set_title(heads[j], fontsize=5.8, pad=1.8)
            if j >= 2:
                m = order[j - 2][0]
                ax.text(0.045, 0.045, f"{100 * c['ious'][m]:.0f}", transform=ax.transAxes,
                        fontsize=5.0, color=M.C_TXT.get(col, col), fontweight="bold",
                        va="bottom", ha="left",
                        bbox=dict(fc="white", ec="none", alpha=0.80, pad=0.6))
            if j == 0:
                # Rotated: the blocks are too narrow for a horizontal row label. The
                # font is sized so the longest label still clears its row -- rows are
                # ~0.36-0.60in tall, and at 5.8pt a ten-character label overruns the
                # short ones and collides with its neighbour.
                ax.text(-0.09, 0.5, anat, transform=ax.transAxes, fontsize=5.0,
                        va="center", ha="center", rotation=90, fontweight="bold")


# Thyroid is back: DDTI and TG3K per-image predictions are archived now (568 and 719
# cases), so the row no longer has to be dropped. TG3K is the gland rather than the
# nodule, which matches the other rows -- they are anatomies, not lesions.
ROWS = [("CAMUS", "Cardiac"), ("HC18", "Fetal\nhead"), ("ACOUSLIC", "Fetal\nabd."),
        ("BUSI", "Breast"), ("KidneyUS", "Kidney"), ("RegPro", "Prostate"),
        ("TG3K", "Thyroid")]

if __name__ == "__main__":
    import sys
    only = sys.argv[1] if len(sys.argv) > 1 else None
    print("writing Figure 1 panels to", OUT)
    if only in (None, "a"): fig1a()
    if only in (None, "c"): fig1c()
    if only in (None, "d"): fig1d(ROWS)
    print("done")
