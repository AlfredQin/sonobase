#!/usr/bin/env python
"""Figure 1, portrait: SonoCorpus + SonoBase on a single full-height page.

Replaces the four-piece LaTeX composition (fig1a + Overall.png +
Image_Pyramid_Encoder.png + fig1c + fig1d) with one self-contained figure, so the
panel hierarchy is fixed here rather than by float placement.

Two deliberate departures from the previous artwork:

  * No logos. Manufacturer, repository and university marks and the country flags
    are gone -- they are decoration that implies endorsement, and the vendor and
    country facts they gestured at are now charts in panel a, sourced per dataset.
  * The panel-a counts are computed from the dataset inventory and the
    country/scanner table rather than hard-coded.

Usage (from src/scripts/analysis):
    SONOBASE_DATA_ROOT=<repo> FIGOUT=<dir> \
        FIG1_INVENTORY_CSV=... FIG1_COUNTRY_SCANNER_CSV=... FIG1_DATASETS_XLSX=... \
        python make_fig1_portrait.py
"""
from __future__ import annotations

import csv
import os
import re
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import make_paper_figures as M
import make_fig1_panels as P

OUT = M.OUT
# Panel a is built from three per-dataset tables that are not part of this
# repository; point the environment variables at your own copies:
#   FIG1_INVENTORY_CSV        dataset, tier, modality, n_units, frames, masks, labels,
#                             n_train, n_val, n_test, info_* columns
#   FIG1_COUNTRY_SCANNER_CSV  dataset, country, scanner
#   FIG1_DATASETS_XLSX        workbook with a `datasets` sheet holding an `Organs` column
# The per-dataset country / scanner / application values are published with the
# SonoCorpus manifest record on Zenodo.
DATA = M.DATA
INV = os.environ.get("FIG1_INVENTORY_CSV", "")
CS = os.environ.get("FIG1_COUNTRY_SCANNER_CSV", "")
XLSX = os.environ.get("FIG1_DATASETS_XLSX", "")

# The five datasets that keep slice structure after conversion. ASUS and CardiacNet
# are volumetric at source but their converters flatten to 2D, and MMOTU-3d reads
# JPGs despite the name, so neither belongs here.
VOLUMETRIC = {"EchoCP", "MUP", "RegPro", "SegThy", "TDSC-ABUS"}

OI = P.OI
INK, MUTED = "#333333", "#6E6E6E"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 6.4, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.0, "ytick.major.size": 2.0, "pdf.fonttype": 42,
})


def _head(fig, y, letter, title):
    fig.text(0.010, y, letter, fontsize=10, fontweight="bold", va="bottom", ha="left")
    fig.text(0.032, y, title, fontsize=8.2, fontweight="bold", va="bottom",
             ha="left", color=INK)


# ------------------------------------------------------------------ statistics
def _norm_country(p):
    return {"usa": "United States", "uk": "United Kingdom",
            "porland": "Poland"}.get(p.strip().lower(), p.strip())


# The inventory's organ strings are verbose and one is misspelled; the charts need
# short labels that still read unambiguously.
SHORT = {"Breast Lesions": "Breast", "Thyroid nodules": "Thyroid",
         "Fetal Abdomen": "Fetal abd.", "Fetal Head": "Fetal head",
         "Cervic": "Cervix", "Carotid Artery": "Carotid",
         "Patent Foramen Ovale": "PFO", "United States": "USA",
         "United Kingdom": "UK", "Sierra Leone": "S. Leone"}


def short(k):
    return SHORT.get(k, k)


VENDOR_PATTERNS = [
    ("GE", r"\bge\b|general electric|logiq|voluson|vivid|vingmed|invenia"),
    ("Philips", r"philips|epiq|ie33|hdi |sonos|cx50|affiniti"),
    ("Siemens", r"siemens|acuson"),
    ("Hitachi", r"hitachi|aloka|arietta|noblus|preirus|hi-vision"),
    ("Toshiba", r"toshiba|nemio|aplio|xario"),
    ("Mindray", r"mindray|resona|umt-|dc-8"),
    ("Esaote", r"esaote|mylab"),
    ("SonoSite", r"sonosite|fujifilm|m-turbo|m turbo"),
    ("Telemed", r"telemed|micrus"),
    ("SuperSonic", r"supersonic|aixplorer"),
    ("Samsung", r"samsung|medison|rs85"),
    ("Canon", r"canon"), ("Olympus", r"olympus"), ("Butterfly", r"butterfly"),
]


PRED = f"{DATA}/src/experiments/analysis/predictions"


def heldout_frames(inv):
    """Held-out test-partition frames of the Benchmark tier, counted rather than asserted:
    unique (video, frame) pairs with ground truth in the evaluated box-prompt prediction
    records (the frames behind Tables 1-2). Falls back to a proportional estimate
    frames * n_test / n_units for any dataset without a prediction record, and reports both."""
    total, est_total, detail = 0, 0, []
    for r in inv:
        if r["tier"] != "Benchmark":
            continue
        est = int(r["frames"]) * int(r["n_test"]) / max(int(r["n_units"]), 1)
        f = f"{PRED}/sonobase_{r['dataset']}_box_0corr/per_sample_metrics.csv"
        if os.path.exists(f):
            seen = set()
            for row in csv.DictReader(open(f)):
                if (row.get("notes") or "") == "no_gt_this_frame":
                    continue
                seen.add((row["sample_id"], row["frame_idx"]))
            n = len(seen); src = "evaluated"
        else:
            n = round(est); src = "estimated"
        total += n; est_total += est; detail.append((r["dataset"], n, round(est), src))
    return total, est_total, detail


def stats():
    """Every number in panel a, derived rather than asserted."""
    if not (INV and CS and XLSX):
        raise SystemExit("set FIG1_INVENTORY_CSV, FIG1_COUNTRY_SCANNER_CSV and FIG1_DATASETS_XLSX")
    inv = list(csv.DictReader(open(INV)))
    frames = sum(int(r["frames"]) for r in inv)
    masks = sum(int(r["masks"]) for r in inv)

    fmt = Counter()
    for r in inv:
        k = ("3D volumes" if r["dataset"] in VOLUMETRIC
             else ("Videos" if r["modality"] == "Video" else "2D images"))
        fmt[k] += int(r["frames"])
    formats = [(k, fmt[k]) for k in ("2D images", "Videos", "3D volumes")]

    tierf = Counter()
    for r in inv:
        tierf[r["tier"]] += int(r["frames"])
    tiers = [(k, tierf[k]) for k in ("Pretrain", "Benchmark", "External")]

    cs = list(csv.DictReader(open(CS)))
    cc, vc = Counter(), Counter()
    for r in cs:
        c = (r["country"] or "").strip()
        if c and c != "UNKNOWN":
            for p in re.split(r"[;,]", c):
                if p.strip():
                    cc[_norm_country(p)] += 1
        s = (r["scanner"] or "").lower()
        if s and s != "unknown":
            for name, pat in VENDOR_PATTERNS:
                if re.search(pat, s):
                    vc[name] += 1

    # Organ groups, not the fine-grained label vocabulary: the inventory's `labels`
    # column has 87 distinct strings, which is a label count, not the "clinical
    # applications" figure the manuscript reports.
    import openpyxl
    ws = openpyxl.load_workbook(XLSX, data_only=True)["datasets"]
    xr = list(ws.iter_rows(min_row=2, values_only=True))
    oi = {h: i for i, h in enumerate(xr[0]) if h}["Organs"]
    org = Counter()
    for r in xr[1:]:
        if not r[0] or str(r[0]).startswith("Total"):
            continue
        for p in str(r[oi] or "").split(","):
            if p.strip():
                org[p.strip()] += 1

    return dict(frames=frames, masks=masks, formats=formats, tiers=tiers,
                countries=cc, vendors=vc, organs=org, n_datasets=len(inv))


# ------------------------------------------------------------------ panel a
def panel_a(fig, S, strip_gs, chart_gs, ytitle):
    _head(fig, ytitle, "a", "SonoCorpus: a large-scale, diverse ultrasound dataset")

    ax = fig.add_subplot(strip_gs); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.002, 0.04), 0.996, 0.92,
                                boxstyle="round,pad=0.004,rounding_size=0.02",
                                transform=ax.transAxes, fc="#F7F7F7",
                                ec="#DDDDDD", lw=0.7, zorder=0))
    cells = [(f"{S['frames']/1000:.0f}k", "images / frames / slices"),
             (f"{S['masks']/1e6:.2f}M", "segmentation masks"),
             (f"{S['n_datasets']}", "public datasets"),
             (f"{len(S['organs'])}", "clinical applications")]
    for i, (big, small) in enumerate(cells):
        x = 0.125 + i * 0.25
        ax.text(x, 0.66, big, fontsize=11.5, fontweight="bold", ha="center",
                va="center", color=INK, transform=ax.transAxes)
        ax.text(x, 0.24, small, fontsize=6.2, ha="center", va="center",
                color=MUTED, transform=ax.transAxes)

    charts = chart_gs.subgridspec(2, 2, hspace=0.95, wspace=0.58)

    ax = fig.add_subplot(charts[0, 0])
    P._hbar(ax, [(short(k), v) for k, v in S["organs"].most_common(10)][::-1], OI[0],
            "Clinical coverage (top 10)", "Number of datasets")

    ax = fig.add_subplot(charts[0, 1])
    P._donut(ax, S["formats"], [OI[2], OI[1], OI[5]], "Data formats (by frames)")

    ax = fig.add_subplot(charts[1, 0])
    P._hbar(ax, [(short(k), v) for k, v in S["countries"].most_common(8)][::-1], OI[2],
            f"Geographic diversity (top 8 of {len(S['countries'])})",
            "Number of datasets")

    ax = fig.add_subplot(charts[1, 1])
    P._hbar(ax, S["vendors"].most_common(8)[::-1], OI[3],
            "Scanner vendors (top 8)", "Number of datasets")


# ------------------------------------------------------------------ panel b
EXAMPLES = [("ACOUSLIC", "Fetal abd."), ("BUSI", "Breast"), ("KidneyUS", "Kidney"),
            ("TG3K", "Thyroid"), ("CAMUS", "Cardiac"), ("HC18", "Fetal head"),
            ("RegPro", "Prostate")]


def panel_b(fig, gs, ytitle):
    _head(fig, ytitle, "b", "Data composition and model architecture")

    lr = gs.subgridspec(1, 2, width_ratios=[0.92, 1.0], wspace=0.10)

    # -- left: dataset examples, ground truth only (this is the corpus, not a result)
    ex = lr[0].subgridspec(2, 4, hspace=0.30, wspace=0.06)
    drawn = 0
    for i, (ds, label) in enumerate(EXAMPLES):
        k = P._pick_typical(ds)
        c = M._load_case(ds, k) if k else None
        if c is None:
            print(f"    ! {ds}: no renderable case, skipped")
            continue
        ax = fig.add_subplot(ex[drawn // 4, drawn % 4])
        M._draw(ax, c["img"], c["gt"], M.C_GT, None)
        ax.set_title(label, fontsize=5.6, pad=1.4, color=INK)
        drawn += 1
    for j in range(drawn, 8):
        fig.add_subplot(ex[j // 4, j % 4]).axis("off")

    # -- right: architecture
    ax = fig.add_subplot(lr[1]); ax.axis("off")
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)

    def box(x, y, w, h, text, fc, fs=5.6):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.6,rounding_size=2.2",
                                    fc=fc, ec="#9A9A9A", lw=0.7, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=INK, zorder=3)

    def arrow(p, q, style="-|>"):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=6,
                                     lw=0.8, color="#666666", zorder=1,
                                     shrinkA=1.5, shrinkB=1.5))

    box(3, 62, 26, 15, "Image\npyramid encoder", "#F6D9DC")
    box(3, 20, 22, 13, "Prompt\nencoder", "#D9EDD9")
    box(38, 60, 22, 19, "Memory\nattention", "#D8E4F5")
    box(38, 18, 22, 17, "Mask\ndecoder", "#FBE2C8")
    box(71, 62, 25, 15, "Memory\nbank", "#CFE8F5")
    box(71, 30, 25, 15, "Memory\ndecoder", "#FBEFC8")

    ax.text(3, 14.5, "Prompt", fontsize=5.6, ha="left", va="top", color=MUTED)
    arrow((29, 69.5), (38, 69.5))
    arrow((25, 26.5), (38, 26.5))
    arrow((49, 60), (49, 35))
    arrow((60, 26.5), (71, 37.5))
    arrow((83.5, 45), (83.5, 62))
    arrow((71, 69.5), (60, 69.5))
    arrow((60, 26.5), (99, 26.5))
    ax.text(99, 22.0, "Prediction", fontsize=5.6, ha="right", va="center", color=INK)
    ax.text(50, 92, "SonoBase", fontsize=6.8, fontweight="bold", ha="center",
            va="center", color=INK)
    ax.text(50, 85, "memory-conditioned promptable segmentation",
            fontsize=5.4, ha="center", va="center", color=MUTED)


# ------------------------------------------------------------------ panel c
def panel_c(fig, gs, S, ytitle):
    _head(fig, ytitle, "c", "Evaluation framework")

    ax = fig.add_subplot(gs); ax.axis("off")
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)

    inv = list(csv.DictReader(open(INV)))
    tier_n = Counter(r["tier"] for r in inv)
    tier_f = Counter()
    for r in inv:
        tier_f[r["tier"]] += int(r["frames"])
    # Benchmark: only the test partition is withheld, so show the held-out frames, not the
    # whole-dataset count (its train splits are inside the 46-dataset training corpus).
    ho_total, ho_est, ho_detail = heldout_frames(inv)
    print(f"  benchmark held-out frames: {ho_total:,} evaluated (proportional estimate {ho_est:,.0f})")
    for d, n, e, src in ho_detail:
        print(f"    {d:16s} {n:6d} {src:9s} (est {e:6d})")

    # "Held out" alone was wrong for the Benchmark tier: its *train* splits are part
    # of the training corpus and only its test partitions are withheld, so the corpus
    # is 38 + 8 = 46 datasets. Say that on the figure rather than leave it to Methods.
    tiers = [("Pretrain", "Train + val splits used", "#F3DCEB", OI[4]),
             ("Benchmark", "Train splits used; test held out", "#DCEBF7", OI[0]),
             ("External", "Fully withheld from training", "#FBE6D5", OI[3])]
    ys = [66, 37, 8]
    for (name, sub, fc, col), y in zip(tiers, ys):
        ax.add_patch(FancyBboxPatch((1, y), 30, 24,
                                    boxstyle="round,pad=0.6,rounding_size=2.2",
                                    fc=fc, ec=col, lw=0.9, zorder=2))
        ax.text(3.5, y + 17, name, fontsize=6.6, fontweight="bold", color=INK, zorder=3)
        frames_line = (f"{ho_total/1000:.0f}k test frames" if name == "Benchmark"
                       else f"{tier_f[name]/1000:.0f}k frames")
        ax.text(3.5, y + 9.5, f"{tier_n[name]} datasets\n{frames_line}",
                fontsize=5.5, color=INK, zorder=3, va="center")
        ax.text(30, y + 4, sub, fontsize=5.0, color=MUTED, ha="right", zorder=3)

    hub = (46, 45)
    for y, (_, _, _, col) in zip(ys, tiers):
        ax.add_patch(FancyArrowPatch((31.5, y + 12), hub, arrowstyle="-", lw=0.9,
                                     color=col, alpha=0.85, zorder=1))
    ax.text(16, -6.5, "training corpus = 38 + 8 = 46 datasets", fontsize=5.2,
            ha="center", va="center", color=MUTED, style="italic")
    ax.plot(*hub, marker="o", ms=3.8, color="#555555", zorder=3)
    ax.text(hub[0], hub[1] - 6.5, "SonoBase", fontsize=6.0, fontweight="bold",
            ha="center", color=INK)
    ax.text(hub[0], hub[1] - 12.0, "evaluated on", fontsize=5.2, ha="center", color=MUTED)

    outs = ["Segmentation accuracy  (Tables 1-2)", "Clinical measurement  (Fig. 3)",
            "Catastrophic-failure rescue  (Fig. 4)", "Workflow efficiency  (Fig. 5)",
            "Few-shot adaptation  (Table 5)"]
    oy = np.linspace(83, 5, len(outs))
    for t, y in zip(outs, oy):
        ax.add_patch(FancyBboxPatch((60, y - 1), 39, 11,
                                    boxstyle="round,pad=0.5,rounding_size=2.0",
                                    fc="#FFFFFF", ec="#B9B9B9", lw=0.7, zorder=2))
        ax.text(79.5, y + 4.5, t, fontsize=5.4, ha="center", va="center",
                color=INK, zorder=3)
        ax.add_patch(FancyArrowPatch(hub, (59.5, y + 4.5), arrowstyle="-|>",
                                     mutation_scale=6, lw=0.7, color="#AAAAAA", zorder=1))


# ------------------------------------------------------------------ panel d
# Six anatomies in a clean 3x2. The fetal-abdomen row was only ever a stand-in for
# thyroid, whose per-image predictions are archived now, so it is not needed and a
# seventh row would leave an empty cell -- expensive on a portrait page.
ROWS_D = [("CAMUS", "Cardiac"), ("HC18", "Fetal\nhead"), ("BUSI", "Breast"),
          ("KidneyUS", "Kidney"), ("RegPro", "Prostate"), ("TG3K", "Thyroid")]


def panel_d(fig, gs, ytitle):
    _head(fig, ytitle, "d", "Qualitative results across organs and modalities")

    cases = []
    for ds, anat in ROWS_D:
        k = P._pick_typical(ds)
        c = M._load_case(ds, k) if k else None
        if c is None:
            print(f"    ! {ds}: no renderable case, skipped")
            continue
        cases.append((ds, anat, c))

    nrow = int(np.ceil(len(cases) / 2))
    heads = ["Input", "GT", "SonoBase", "MedSAM2", "SAM2"]
    order = [("sonobase", M.C_SB), ("medsam2", M.C_M2), ("sam2_no_ft", M.C_S2)]
    rowh = []
    for r in range(nrow):
        idxs = [b * nrow + r for b in range(2) if b * nrow + r < len(cases)]
        ars = [cases[i][2]["img"].shape[0] / cases[i][2]["img"].shape[1] for i in idxs]
        rowh.append(float(np.clip(np.mean(ars), 0.55, 1.00)))
    blocks = gs.subgridspec(1, 2, wspace=0.11)
    for b in range(2):
        sub = cases[b * nrow:(b + 1) * nrow]
        if not sub:
            continue
        inner = blocks[b].subgridspec(nrow, 5, wspace=0.045, hspace=0.16,
                                      height_ratios=rowh)
        P._block(fig, inner, sub, heads, order)


# ------------------------------------------------------------------ main
def main():
    S = stats()
    print(f"  corpus: {S['frames']:,} frames / {S['masks']:,} masks / "
          f"{S['n_datasets']} datasets / {len(S['organs'])} applications")
    print(f"  formats: {S['formats']}")
    print(f"  countries: {len(S['countries'])} distinct, top {S['countries'].most_common(3)}")
    print(f"  vendors: top {S['vendors'].most_common(3)}")

    fig = plt.figure(figsize=(7.1, 9.2))
    R = 0.988
    # Panel a's bar labels need a wide left gutter. Giving panel a its own gridspec
    # keeps that gutter out of b/c/d, which would otherwise sit needlessly indented.
    strip = fig.add_gridspec(1, 1, left=0.030, right=R, top=0.968, bottom=0.928)[0]
    charts = fig.add_gridspec(1, 1, left=0.112, right=0.972, top=0.898, bottom=0.706)[0]
    panel_a(fig, S, strip, charts, 0.974)

    gb = fig.add_gridspec(1, 1, left=0.030, right=R, top=0.638, bottom=0.500)[0]
    panel_b(fig, gb, 0.650)

    gc = fig.add_gridspec(1, 1, left=0.030, right=R, top=0.462, bottom=0.306)[0]
    panel_c(fig, gc, S, 0.472)

    gd = fig.add_gridspec(1, 1, left=0.030, right=R, top=0.268, bottom=0.010)[0]
    panel_d(fig, gd, 0.278)

    os.makedirs(OUT, exist_ok=True)
    fig.savefig(f"{OUT}/fig1_portrait.pdf")
    fig.savefig(f"{OUT}/fig1_portrait.png", dpi=240)
    plt.close(fig)
    print(f"  fig1_portrait.pdf  ({len(ROWS_D)} anatomies)")


if __name__ == "__main__":
    main()
