#!/usr/bin/env python
"""Regenerate the manuscript figures from the v2 result records.

Palette validated for CVD separation (worst normal dE 21.6, worst CVD dE 15.5)
before use: SonoBase green / MedSAM2 vermillion / SAM2 light gray.
"""
import json, csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Records are read from `SONOBASE_DATA_ROOT` (default: this repository), in the
# layout the analysis, few-shot and detection scripts write under src/experiments/.
# Figures go to `FIGOUT` (default: <repo>/figures).
_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
DATA = os.environ.get("SONOBASE_DATA_ROOT", REPO)
E1   = f"{DATA}/src/experiments/2026-05-28_e1-production-pretrain-benchmark/results"
AN   = f"{DATA}/src/experiments/analysis"
FS   = f"{DATA}/src/experiments/few_shot/few_shot_results"
RF   = f"{DATA}/src/experiments/2026-05-30_e3-us-rfdetr-detection/tables/e3_results.csv"
OUT  = os.environ.get("FIGOUT", f"{REPO}/figures")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- style
C_SB, C_M2, C_S2 = "#009E73", "#D55E00", "#BBBBBB"
MODELS = [("SonoBase", "SonoBase", C_SB), ("MedSAM2", "MedSAM2", C_M2), ("SAM2-no-ft", "SAM2", C_S2)]
GRID = dict(color="#E3E3E3", lw=0.6)
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 7, "axes.labelsize": 7.5,
    "axes.titlesize": 8, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.8, "axes.linewidth": 0.7,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "pdf.fonttype": 42,
})

def tidy(ax, ygrid=True):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#8A8A8A"); ax.spines["bottom"].set_color("#8A8A8A")
    if ygrid:
        ax.set_axisbelow(True); ax.yaxis.grid(True, **GRID); ax.xaxis.grid(False)

def panel(ax, letter, dx=-0.16, dy=1.06):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=9.5,
            fontweight="bold", va="top", ha="left")

# ---------------------------------------------------------------- data
def load_cmp(fn):
    d = {}
    for r in csv.DictReader(open(f"{E1}/{fn}")):
        d.setdefault(r["dataset"], {})[r["metric"]] = {
            k: float(r[k]) for k in ("SonoBase", "MedSAM2", "SAM2-no-ft")}
    return d


def load_noised():
    """Fig 2 plots Tables 1--2, so it must read what those tables report: the
    3-seed randomised-prompt record, not the single-prompt comparison csvs.
    Falls back to those with a loud notice rather than silently plotting the
    old protocol beside the new tables: a figure must not disagree with the
    table it illustrates.

    Returns (BOX, PT) in load_cmp's shape, values as fractions.
    """
    src = (f"{DATA}/src/experiments/2026-08-04_eval-protocol-reconciliation"
           f"/results/prompt_noise_3seed.csv")
    if not os.path.exists(src):
        print(f"  !! 3-seed record not found ({src}); Fig 2 falls back to the E1 "
              f"protocol and will NOT match Tables 1-2")
        return load_cmp("comparison_box.csv"), load_cmp("comparison_point.csv")
    NAME = {"sonobase": "SonoBase", "medsam2": "MedSAM2", "sam2_no_ft": "SAM2-no-ft"}
    out = {"box": {}, "point": {}}
    for r in csv.DictReader(open(src)):
        cell = out[r["prompt"]].setdefault(r["dataset"], {"miou": {}, "dice": {}})
        cell["miou"][NAME[r["model"]]] = float(r["miou_mean"]) / 100.0
        cell["dice"][NAME[r["model"]]] = float(r["dice_mean"]) / 100.0
    print(f"  Fig 2 source: {os.path.basename(src)} (3-seed randomised prompts)")
    return out["box"], out["point"]


BOX, PT = load_noised()
ID8 = ["BUSI", "Brachial-Plexus", "C-TRUS", "CAMUS", "HC18", "PFUS", "RegPro", "TG3K"]
OOD7 = ["ACOUSLIC", "BUS-BRA", "DDTI", "FUGC", "KidneyUS", "LUMINOUS", "MMOTU-3d"]
SHORT = {"Brachial-Plexus": "B-Plexus", "ThyroidUSCineClip": "ThyroidCine", "MMOTU-3d": "MMOTU-3d"}
def sn(d): return SHORT.get(d, d)
def macro(src, ds, m): return 100 * np.mean([src[d]["miou"][m] for d in ds])

# ================================================================ FIG 2
def fig_generalization():
    fig = plt.figure(figsize=(7.2, 5.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.3], hspace=0.62, wspace=0.30,
                          left=0.085, right=0.985, top=0.93, bottom=0.145)

    # a,b : macro bars, point vs box
    for k, (ax, ds, title) in enumerate([
            (fig.add_subplot(gs[0, 0]), ID8, "Benchmark (8 held-out datasets)"),
            (fig.add_subplot(gs[0, 1]), OOD7, "External (7 datasets, domain shift)")]):
        w, x = 0.26, np.arange(2)
        for i, (key, lab, col) in enumerate(MODELS):
            vals = [macro(PT, ds, key), macro(BOX, ds, key)]
            b = ax.bar(x + (i - 1) * w, vals, w * 0.92, color=col,
                       edgecolor="white", linewidth=0.8, label=lab, zorder=3)
            for r, v in zip(b, vals):
                ax.text(r.get_x() + r.get_width() / 2, v + 1.2, f"{v:.1f}",
                        ha="center", va="bottom", fontsize=5.9, color="#333333")
        ax.set_xticks(x); ax.set_xticklabels(["Point prompt", "Box prompt"])
        ax.set_ylabel("mIoU (%)"); ax.set_ylim(0, 112); ax.set_yticks([0,20,40,60,80,100])
        ax.set_title(title, pad=6)
        tidy(ax); panel(ax, "ab"[k])
        if k == 0:
            ax.legend(frameon=False, loc="upper center", ncol=3, handlelength=1.0,
                      handletextpad=0.4, columnspacing=1.0, borderpad=0.15,
                      bbox_to_anchor=(0.5, 1.02))
    # headline cross-comparison as a recessive reference line, no arrow
    axa = fig.axes[0]
    ref = macro(BOX, ID8, "SAM2-no-ft")
    axa.axhline(ref, color="#8A8A8A", lw=0.7, ls=":", zorder=1)
    axa.text(-0.42, ref + 1.8, "SAM2 box", fontsize=5.6, color="#666666", ha="left")

    # c : per-dataset heatmap (SonoBase advantage over best baseline)
    axc = fig.add_subplot(gs[1, 0])
    ds_all = ID8 + OOD7
    mat = np.array([[100 * PT[d]["miou"][m] for d in ds_all] for m, _, _ in
                    [(k, l, c) for k, l, c in MODELS]])
    im = axc.imshow(mat, aspect="auto", cmap="Greens", vmin=0, vmax=100)
    axc.set_yticks(range(3)); axc.set_yticklabels([l for _, l, _ in MODELS])
    axc.set_xticks(range(15)); axc.set_xticklabels([sn(d) for d in ds_all], rotation=90)
    for i in range(3):
        for j in range(15):
            axc.text(j, i, f"{mat[i,j]:.0f}", ha="center", va="center", fontsize=5.1,
                     color="white" if mat[i, j] > 55 else "#333333")
    axc.set_title("Per-dataset mIoU, point prompt (%)", pad=26)
    axc.axvline(7.5, color="#444444", lw=0.9)
    axc.text(3.5, -0.80, "in-distribution", ha="center", fontsize=6, color="#555555")
    axc.text(11.0, -0.80, "external", ha="center", fontsize=6, color="#555555")
    for s in axc.spines.values(): s.set_visible(False)
    axc.tick_params(length=0); panel(axc, "c", dx=-0.13, dy=1.22)

    # d : SonoBase - MedSAM2 gap per dataset
    axd = fig.add_subplot(gs[1, 1])
    gaps = sorted([(sn(d), 100 * (PT[d]["miou"]["SonoBase"] - PT[d]["miou"]["MedSAM2"]))
                   for d in ds_all], key=lambda t: t[1])
    y = np.arange(len(gaps))
    axd.barh(y, [g for _, g in gaps], 0.68, color=C_SB, edgecolor="white",
             linewidth=0.6, zorder=3)
    axd.set_yticks(y); axd.set_yticklabels([n for n, _ in gaps], fontsize=6)
    axd.axvline(0, color="#8A8A8A", lw=0.7)
    axd.set_xlabel("SonoBase − MedSAM2 (mIoU pp, point)")
    for i, (_, g) in enumerate(gaps):
        axd.text(g + 1.2, i, f"+{g:.0f}", va="center", fontsize=5.6, color="#333333")
    axd.set_xlim(0, max(g for _, g in gaps) * 1.16)
    axd.set_title("SonoBase leads on all 15 datasets", pad=6)
    tidy(axd, ygrid=False); axd.set_axisbelow(True)
    axd.xaxis.grid(True, **GRID); axd.yaxis.grid(False)
    panel(axd, "d", dx=-0.24, dy=1.08)

    fig.savefig(f"{OUT}/fig2_generalization.pdf"); fig.savefig(f"{OUT}/fig2_generalization.png", dpi=200); plt.close(fig)
    print("  fig2_generalization.pdf")

# ================================================================ FIG 3
def fig_clinical():
    ef = list(csv.DictReader(open(f"{AN}/a1_camus_ef/CAMUS_box_0corr/per_patient.csv")))
    def col(rows, c):
        return np.array([float(r[c]) if r[c] not in ("", None) else np.nan for r in rows])
    gt = col(ef, "gt_ef"); pr = col(ef, "pred_ef_biplane_ase__sonobase")
    q = np.array([r["image_quality"] for r in ef])
    ok = ~np.isnan(gt) & ~np.isnan(pr); gt, pr, q = gt[ok], pr[ok], q[ok]

    fig = plt.figure(figsize=(7.2, 4.6))
    gs = fig.add_gridspec(2, 3, hspace=0.62, wspace=0.36,
                          left=0.075, right=0.985, top=0.92, bottom=0.11)

    # a HC scatter
    ax = fig.add_subplot(gs[0, 0])
    hc = list(csv.DictReader(open(f"{AN}/a2_hc18_hc/HC18_box_0corr/per_sample.csv")))
    g = np.array([float(r["gt_hc_mm"]) for r in hc])
    p = np.array([float(r["pred_hc_mm__sonobase"]) for r in hc])
    ax.scatter(g, p, s=8, color=C_SB, alpha=0.7, linewidths=0, zorder=3)
    lo, hi = min(g.min(), p.min()) * 0.97, max(g.max(), p.max()) * 1.03
    ax.plot([lo, hi], [lo, hi], color="#8A8A8A", lw=0.7, ls="--")
    mae = np.mean(np.abs(p - g))
    ax.text(0.04, 0.94, f"MAE {mae:.2f} mm\nr = {np.corrcoef(g,p)[0,1]:.4f}",
            transform=ax.transAxes, fontsize=6.2, va="top")
    ax.set_xlabel("Ground-truth HC (mm)"); ax.set_ylabel("SonoBase HC (mm)")
    ax.set_title("Head circumference", pad=5); tidy(ax); panel(ax, "a")

    # b GA error distribution
    ax = fig.add_subplot(gs[0, 1])
    ga = list(csv.DictReader(open(f"{AN}/t2_1_hc_ga/HC18_box_0corr/per_sample.csv")))
    errs = {k: np.array([float(r[f"abs_err_days__{v}"]) for r in ga])
            for k, v in [("SonoBase", "sonobase"), ("MedSAM2", "medsam2"), ("SAM2", "sam2_no_ft")]}
    bins = np.linspace(0, 15, 31)
    for (lab, col) in [("SonoBase", C_SB), ("MedSAM2", C_M2), ("SAM2", C_S2)]:
        ax.hist(np.clip(errs[lab], 0, 15), bins=bins, histtype="step", lw=1.2,
                color=col, label=lab, zorder=3)
    ax.axvline(3, color="#555555", lw=0.7, ls=":")
    ax.text(3.25, ax.get_ylim()[1] * 0.92, "3-day\ntolerance", fontsize=5.7, color="#555555")
    ax.set_xlabel("Gestational-age error (days, clipped at 15)")
    ax.set_ylabel("Images"); ax.set_title("Gestational age", pad=5)
    ax.legend(frameon=False, loc="upper right", handlelength=1.0, handletextpad=0.5,
              borderpad=0.2, labelspacing=0.25)
    tidy(ax); panel(ax, "b")

    # c Bland-Altman
    ax = fig.add_subplot(gs[0, 2])
    mean, diff = (gt + pr) / 2, pr - gt
    bias, sd = diff.mean(), diff.std(ddof=1)
    ax.scatter(mean, diff, s=9, color=C_SB, alpha=0.75, linewidths=0, zorder=3)
    ax.axhline(bias, color="#333333", lw=0.9)
    ax.axhline(bias + 1.96 * sd, color="#8A8A8A", lw=0.7, ls="--")
    ax.axhline(bias - 1.96 * sd, color="#8A8A8A", lw=0.7, ls="--")
    ax.text(0.98, 0.94, f"bias {bias:+.2f}%", transform=ax.transAxes, ha="right",
            fontsize=6, color="#333333")
    ax.text(0.98, 0.87, f"LoA [{bias-1.96*sd:.1f}, {bias+1.96*sd:.1f}]",
            transform=ax.transAxes, ha="right", fontsize=6, color="#555555")
    ax.set_xlabel("Mean of GT and predicted EF (%)"); ax.set_ylabel("Predicted − GT EF (%)")
    ax.set_title("Ejection fraction agreement", pad=5); tidy(ax); panel(ax, "c")

    # d EF scatter by quality
    ax = fig.add_subplot(gs[1, 0])
    shades = {"Good": "#00past", "Medium": "", "Poor": ""}
    for lbl, mk, al in [("Good", "o", 0.9), ("Medium", "s", 0.75), ("Poor", "^", 0.9)]:
        m = q == lbl
        ax.scatter(gt[m], pr[m], s=11, marker=mk, alpha=al, linewidths=0,
                   color=C_SB, label=f"{lbl} (n={m.sum()})", zorder=3)
    lo, hi = 5, 75
    ax.plot([lo, hi], [lo, hi], color="#8A8A8A", lw=0.7, ls="--", zorder=2)
    r = np.corrcoef(gt, pr)[0, 1]
    ax.text(0.04, 0.94, f"r = {r:.3f}", transform=ax.transAxes, fontsize=6.4, va="top")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Ground-truth EF (%)"); ax.set_ylabel("SonoBase EF (%)")
    ax.set_title("EF by image quality", pad=5)
    ax.legend(frameon=False, loc="lower right", handlelength=0.8, handletextpad=0.4,
              borderpad=0.15, labelspacing=0.25)
    tidy(ax); panel(ax, "d")

    # e reclassification
    ax = fig.add_subplot(gs[1, 1])
    rep = json.load(open(f"{AN}/a1_camus_ef/CAMUS_box_0corr/analysis_report.json"))["models"]
    x, w = np.arange(2), 0.26
    for i, (key, lab, colr) in enumerate(MODELS):
        k = {"SonoBase": "sonobase", "MedSAM2": "medsam2", "SAM2-no-ft": "sam2_no_ft"}[key]
        vals = [rep[k]["reclass_at_40_pct"], rep[k]["reclass_at_35_pct"]]
        b = ax.bar(x + (i - 1) * w, vals, w * 0.92, color=colr, edgecolor="white",
                   linewidth=0.8, label=lab, zorder=3)
        for rr, v in zip(b, vals):
            ax.text(rr.get_x() + rr.get_width() / 2, v + 0.8, f"{v:.0f}", ha="center",
                    fontsize=5.9, color="#333333")
    ax.set_xticks(x); ax.set_xticklabels(["HFrEF\n(EF ≤ 40%)", "ICD\n(EF ≤ 35%)"])
    ax.set_ylabel("Patients reclassified (%)"); ax.set_ylim(0, 56)
    ax.legend(frameon=False, loc="upper left", ncol=1, handlelength=1.0,
              handletextpad=0.4, borderpad=0.15, labelspacing=0.22)
    ax.set_title("Threshold reclassification", pad=5); tidy(ax); panel(ax, "e")

    # f AC under shift
    ax = fig.add_subplot(gs[1, 2])
    accs = {}
    for pr_name, f in [("point", "ACOUSLIC_point_0corr"), ("box", "ACOUSLIC_box_0corr")]:
        rep = json.load(open(f"{AN}/a4_acouslic_ac/{f}/analysis_report.json"))["models"]
        accs[pr_name] = [rep[k]["mae_mm"] for k in ("sonobase", "medsam2", "sam2_no_ft")]
    x, w = np.arange(2), 0.26
    order = [("SonoBase", C_SB, 0), ("MedSAM2", C_M2, 1), ("SAM2", C_S2, 2)]
    for i, (lab, colr, idx) in enumerate(order):
        vals = [accs["point"][idx], accs["box"][idx]]
        b = ax.bar(x + (i - 1) * w, vals, w * 0.92, color=colr, edgecolor="white",
                   linewidth=0.8, label=lab, zorder=3)
        for rr, v in zip(b, vals):
            if v > 100:
                ax.text(rr.get_x() + rr.get_width() / 2, 101, f"{v:.0f} \u2191",
                        ha="center", fontsize=5.7, color="#8A2A2A")
            else:
                ax.text(rr.get_x() + rr.get_width() / 2, v + 2.5, f"{v:.0f}",
                        ha="center", fontsize=5.9, color="#333333")
    ax.set_xticks(x); ax.set_xticklabels(["Point prompt", "Box prompt"])
    ax.set_ylim(0, 118); ax.set_yticks([0,25,50,75,100]); ax.set_ylabel("AC error (mm)")
    ax.set_title("Abdominal circumference (POCUS)", pad=5); tidy(ax); panel(ax, "f")

    fig.savefig(f"{OUT}/fig3_clinical.pdf"); fig.savefig(f"{OUT}/fig3_clinical.png", dpi=200); plt.close(fig)
    print("  fig3_clinical.pdf")

# ================================================================ FIG 5
def fig_workflow():
    rep = json.load(open(f"{AN}/a3_click_efficiency/analysis_report.json"))
    fac = rep["facet_summaries"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35))
    fig.subplots_adjust(left=0.072, right=0.985, top=0.84, bottom=0.20, wspace=0.34)

    for k, (tier, title) in enumerate([("benchmark", "Benchmark datasets"),
                                       ("external", "External datasets")]):
        ax = axes[k]
        for key, lab, colr in MODELS:
            m = {"SonoBase": "sonobase", "MedSAM2": "medsam2", "SAM2-no-ft": "sam2_no_ft"}[key]
            for prompt, ls, mk in [("point", "-", "o"), ("box", "--", "s")]:
                f = fac.get(f"{m}__{prompt}__{tier}")
                if not f: continue
                it = sorted(int(i) for i in f["miou_macro_pct"])
                ys = [f["miou_macro_pct"][str(i)] for i in it]
                ax.plot(it, ys, ls, color=colr, lw=1.3, marker=mk, ms=2.8,
                        markeredgewidth=0, zorder=3)
        ax.axhline(80, color="#555555", lw=0.8, ls=":")
        ax.text(7.15, 71.5, "80% threshold", fontsize=5.7, color="#555555", ha="right")
        ax.set_xlabel("Corrective clicks"); ax.set_ylim(0, 108)
        ax.set_yticks([0,20,40,60,80,100]); ax.set_xlim(-0.25, 7.25)
        if k == 0: ax.set_ylabel("mIoU (%)")
        ax.set_title(title, pad=5); tidy(ax); panel(ax, "ab"[k], dx=-0.20)

    handles = [Line2D([], [], color=c, lw=1.4, label=l) for _, l, c in MODELS] + \
              [Line2D([], [], color="#555555", lw=1.2, ls="-", marker="o", ms=2.8, label="point"),
               Line2D([], [], color="#555555", lw=1.2, ls="--", marker="s", ms=2.8, label="box")]
    axes[0].legend(handles=handles, frameon=False, loc="lower right", ncol=1,
                   handlelength=1.3, handletextpad=0.5, borderpad=0.2, labelspacing=0.22)

    # c time to threshold, grouped by tier
    ax = axes[2]
    x, w = np.arange(2), 0.26
    for i, (key, lab, colr) in enumerate(MODELS):
        m = {"SonoBase": "sonobase", "MedSAM2": "medsam2", "SAM2-no-ft": "sam2_no_ft"}[key]
        vals = []
        for tier in ("benchmark", "external"):
            f = fac.get(f"{m}__point__{tier}")
            vals.append(f.get("seconds_to_80") if f and f.get("seconds_to_80") else np.nan)
        ax.bar(x + (i - 1) * w, [0 if np.isnan(v) else v for v in vals], w * 0.92,
               color=colr, edgecolor="white", linewidth=0.8, label=lab, zorder=3)
        for xi, v in zip(x + (i - 1) * w, vals):
            if np.isnan(v):
                ax.text(xi, 0.6, "never", ha="center", va="bottom", fontsize=5.2,
                        color="#8A2A2A", rotation=90)
            else:
                ax.text(xi, v + 0.5, f"{v:.0f}s", ha="center", fontsize=5.9, color="#333333")
    ax.set_xticks(x); ax.set_xticklabels(["Benchmark", "External"])
    ax.set_ylabel("Seconds to 80% mIoU"); ax.set_ylim(0, 21)
    ax.set_title("Interaction cost (point init)", pad=5); tidy(ax); panel(ax, "c", dx=-0.22)

    fig.savefig(f"{OUT}/fig5_workflow.pdf"); fig.savefig(f"{OUT}/fig5_workflow.png", dpi=200); plt.close(fig)
    print("  fig5_workflow.pdf")

# ================================================================ FIG 6
def fig_breadth():
    fig = plt.figure(figsize=(7.2, 4.5))
    gs = fig.add_gridspec(2, 3, hspace=0.62, wspace=0.34,
                          left=0.075, right=0.985, top=0.92, bottom=0.11)

    # a downstream transfer: detection + instance segmentation (E3 curated table)
    ax = fig.add_subplot(gs[0, 0])
    rows = list(csv.DictReader(open(RF)))
    ORDER = ["cva_net","fetus","acouslic","bus_bra","ddti","fugc","kidneyus","luminous"]
    NAME = {"cva_net":"CVA-Net","fetus":"Fetus","acouslic":"ACOUSLIC","bus_bra":"BUS-BRA",
            "ddti":"DDTI","fugc":"FUGC","kidneyus":"KidneyUS","luminous":"LUMINOUS"}
    def cell(arm, ds, b, m):
        for r in rows:
            if r["arm"] == arm and r["dataset"] == ds and r["backbone"] == b:
                return float(r[m]) * 100 if r[m] else np.nan
        return np.nan
    # `bbox_3e-4`, not `headline_B`. headline_B is 3e-4 everywhere except one cell —
    # ddti/sam2_no_ft, which it takes from the 1e-4 rerun (0.175 against 0.019). That
    # rescues a single baseline on a single dataset while leaving MedSAM2 on the shared
    # schedule, which is exactly the asymmetry the table's protocol note disclaims.
    # bbox_3e-4 is internally uniform and agrees cell-for-cell with the authoritative
    # mAP_table_v2.csv. Macro: SAM2 36.5, MedSAM2 34.3, SonoBase 54.5.
    det = {b: [cell("bbox_3e-4", d, b, "bbox_mAP") for d in ORDER] for _, _, b in
           [(0, 0, "sam2_no_ft"), (0, 0, "medsam2"), (0, 0, "sonobase")]}
    seg = {b: [cell("segm", d, b, "segm_mAP") for d in ORDER] for b in
           ["sam2_no_ft", "medsam2", "sonobase"]}
    x, w = np.arange(2), 0.26
    for i, (key, lab, colr) in enumerate(MODELS):
        b = {"SonoBase": "sonobase", "MedSAM2": "medsam2", "SAM2-no-ft": "sam2_no_ft"}[key]
        dm = np.nanmean([v for d, v in zip(ORDER, det[b]) if not np.isnan(v)])
        sm = np.nanmean([v for d, v in zip(ORDER, seg[b]) if d != "fugc" and not np.isnan(v)])
        vals = [dm, sm]
        bars = ax.bar(x + (i - 1) * w, vals, w * 0.92, color=colr, edgecolor="white",
                      linewidth=0.8, label=lab, zorder=3)
        for rr, v in zip(bars, vals):
            ax.text(rr.get_x() + rr.get_width() / 2, v + 1.4, f"{v:.1f}", ha="center",
                    fontsize=5.9, color="#333333")
    ax.set_xticks(x); ax.set_xticklabels(["Detection\n(8 ds)", "Instance seg.\n(6 ds)"])
    ax.set_ylabel("mAP"); ax.set_ylim(0, 88)
    ax.set_title("Downstream transfer (frozen encoder)", pad=5, fontsize=7.4)
    ax.legend(frameon=False, loc="upper left", ncol=1, handlelength=0.9,
              handletextpad=0.35, borderpad=0.12, labelspacing=0.2, fontsize=6)
    tidy(ax); panel(ax, "a")

    # b cross-species, recomputed from v2 predictions
    ax = fig.add_subplot(gs[0, 1])
    P = f"{AN}/predictions"
    def mb_iou(model, prompt):
        f = f"{P}/{model}_MouseBrainTumor_{prompt}_0corr/per_sample_metrics.csv"
        v = [float(r["iou"]) for r in csv.DictReader(open(f)) if r["iou"]]
        return 100 * float(np.mean(v)), len(v)
    x, w = np.arange(2), 0.26
    for i, (key, lab, colr) in enumerate(MODELS):
        m = {"SonoBase": "sonobase", "MedSAM2": "medsam2", "SAM2-no-ft": "sam2_no_ft"}[key]
        vals = [mb_iou(m, "point")[0], mb_iou(m, "box")[0]]
        b = ax.bar(x + (i - 1) * w, vals, w * 0.92, color=colr, edgecolor="white",
                   linewidth=0.8, zorder=3)
        for rr, v in zip(b, vals):
            ax.text(rr.get_x() + rr.get_width() / 2, v + 1.3, f"{v:.0f}", ha="center",
                    fontsize=5.9, color="#333333")
    n = mb_iou("sonobase", "point")[1]
    ax.set_xticks(x); ax.set_xticklabels(["Point prompt", "Box prompt"])
    ax.set_ylabel("mIoU (%)"); ax.set_ylim(0, 80)
    ax.set_title(f"Cross-species (mouse brain, n={n:,} frames)", pad=5, fontsize=7.2)
    tidy(ax); panel(ax, "b")

    # c subgroup spread: prompt-dependence is the point
    ax = fig.add_subplot(gs[0, 2])
    def cells(p, arm):
        c = json.load(open(f"{AN}/{p}/analysis_report.json"))["cells"]
        return {g: 100 * c[g][arm]["mean_iou"] for g in c if arm in c[g]}
    groups = [("KidneyUS\n(5 scanners)", "s1_kidneyus_manufacturer/KidneyUS_0corr"),
              ("CAMUS\n(3 qualities)", "s2_camus_quality/CAMUS_0corr"),
              ("BUSI\n(2 pathologies)", "s3_busi_pathology/BUSI_0corr")]
    x, w = np.arange(len(groups)), 0.3
    for i, (arm, lab, hatch) in enumerate([("sonobase__point", "point", ""),
                                           ("sonobase__box", "box", "///")]):
        sp = [max(cells(p, arm).values()) - min(cells(p, arm).values()) for _, p in groups]
        b = ax.bar(x + (i - 0.5) * w, sp, w * 0.9, color=C_SB, alpha=1 if i else 0.55,
                   edgecolor="white", linewidth=0.8, hatch=hatch, label=lab, zorder=3)
        for rr, v in zip(b, sp):
            ax.text(rr.get_x() + rr.get_width() / 2, v + 0.9, f"{v:.1f}", ha="center",
                    fontsize=5.8, color="#333333")
    ax.set_xticks(x); ax.set_xticklabels([n for n, _ in groups], fontsize=5.8)
    ax.set_ylabel("Within-group spread (pp)"); ax.set_ylim(0, 50)
    ax.set_title("Spread is prompt-dependent", pad=5)
    ax.legend(frameon=False, loc="upper right", handlelength=1.0, handletextpad=0.45,
              borderpad=0.2)
    tidy(ax); panel(ax, "c")

    # d-f few-shot curves
    rows = list(csv.DictReader(open(f"{FS}/combined_summary.csv")))
    Ns = [1, 2, 5, 10, 20, 30]
    for k, ds in enumerate(["ACOUSLIC", "DDTI", "FUGC"]):
        ax = fig.add_subplot(gs[1, k])
        for key, lab, colr in MODELS:
            m = {"SonoBase": "sonobase", "MedSAM2": "medsam2", "SAM2-no-ft": "sam2_no_ft"}[key]
            mu, sd = [], []
            for n in Ns:
                r = [x for x in rows if x["dataset"] == ds and x["prompt"] == "box"
                     and x["model"] == m and x["N"] == str(n) and x["mIoU_mean"]]
                mu.append(100 * float(r[0]["mIoU_mean"]) if r else np.nan)
                sd.append(100 * float(r[0]["mIoU_std"]) if r else np.nan)
            mu, sd = np.array(mu), np.array(sd)
            ax.plot(Ns, mu, "-", color=colr, lw=1.3, marker="o", ms=2.8,
                    markeredgewidth=0, label=lab, zorder=3)
            ax.fill_between(Ns, mu - sd, mu + sd, color=colr, alpha=0.16, linewidth=0, zorder=2)
        ax.set_xscale("log"); ax.set_xticks(Ns)
        ax.set_xticklabels([str(n) for n in Ns]); ax.minorticks_off()
        ax.set_xlabel("Labelled examples (N)")
        if k == 0:
            ax.set_ylabel("mIoU (%), box prompt")
            ax.axvline(5, color="#555555", lw=0.7, ls=":")
            ax.text(5.4, 91.5, "primary\nendpoint", fontsize=5.4, color="#555555", va="top")
            ax.legend(frameon=False, loc="lower right", handlelength=1.1,
                      handletextpad=0.45, borderpad=0.2, labelspacing=0.22)
        ax.set_ylim(40, 95); ax.set_title(ds, pad=5)
        tidy(ax); panel(ax, "def"[k])

    fig.savefig(f"{OUT}/fig6_breadth.pdf"); fig.savefig(f"{OUT}/fig6_breadth.png", dpi=200); plt.close(fig)
    print("  fig6_breadth.pdf")


# ================================================================ FIG 4
# Panel c / Supplementary S3 are rendered from the archived v2 predictions
# rather than reused artwork, so every displayed IoU traces to the same
# per_sample_metrics.csv files that panels a and b are computed from.
PRED  = f"{AN}/predictions"
C_GT  = "#0072B2"                     # Okabe-Ito blue; kept distinct from the model colours
FAILM = [("medsam2", "MedSAM2", C_M2), ("sam2_no_ft", "SAM2", C_S2),
         ("sonobase", "SonoBase", C_SB)]
# SAM2's light grey reads as the mask overlay but not as text on a white chip
C_TXT = {C_S2: "#6E6E6E"}
FSHORT = {"MouseBrainTumor": "MouseBrain", "PorcineSpinalCord": "PorcineSC"}

def _per_image(model, d):
    """key -> IoU for one model/dataset under point prompting, 0 corrections."""
    f = f"{PRED}/{model}_{d}_point_0corr/per_sample_metrics.csv"
    if not os.path.exists(f): return None
    return {(r["sample_id"], r["frame_idx"], r["obj_id"]): float(r["iou"])
            for r in csv.DictReader(open(f)) if r["iou"]}

def _failure_data():
    """Per-dataset rates plus the pooled (baseline, SonoBase) IoU pairs.

    A catastrophic failure is IoU < 10 for *at least one* baseline while
    SonoBase exceeds 50. `base` is therefore the weaker of the two baselines,
    which makes the panel-b upper-left quadrant exactly the panel-a rate.
    """
    import glob
    ds = sorted({os.path.basename(x).replace("sonobase_", "").replace("_point_0corr", "")
                 for x in glob.glob(f"{PRED}/sonobase_*_point_0corr")})
    rows, base, sono, tot, either, both, only_m2, only_s2 = [], [], [], 0, 0, 0, 0, 0
    for d in ds:
        sb, m2, s2 = (_per_image(m, d) for m in ("sonobase", "medsam2", "sam2_no_ft"))
        if not all([sb, m2, s2]): continue
        keys = sorted(set(sb) & set(m2) & set(s2))
        cat = sum(1 for k in keys if min(m2[k], s2[k]) < 0.10 and sb[k] > 0.50)
        rows.append((d, len(keys), cat, 100 * cat / len(keys)))
        base += [100 * min(m2[k], s2[k]) for k in keys]
        sono += [100 * sb[k] for k in keys]
        tot += len(keys); either += cat
        both    += sum(1 for k in keys if max(m2[k], s2[k]) < 0.10 and sb[k] > 0.50)
        only_m2 += sum(1 for k in keys if m2[k] < 0.10 and sb[k] > 0.50)
        only_s2 += sum(1 for k in keys if s2[k] < 0.10 and sb[k] > 0.50)
    rows.sort(key=lambda r: r[3])
    return (rows, np.array(base), np.array(sono), tot,
            dict(either=100 * either / tot, both=100 * both / tot,
                 m2=100 * only_m2 / tot, s2=100 * only_s2 / tot))

def fig_failure():
    """a: per-dataset resolution rate.  b: joint per-image IoU density."""
    rows, base, sono, tot, pooled = _failure_data()

    fig = plt.figure(figsize=(5.15, 2.45))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.06, 1], wspace=0.52,
                          left=0.155, right=0.90, top=0.86, bottom=0.185)

    # ---- a : per-dataset rate
    ax = fig.add_subplot(gs[0, 0])
    y = np.arange(len(rows))
    ax.barh(y, [r[3] for r in rows], 0.66, color=C_SB, edgecolor="white",
            linewidth=0.6, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([FSHORT.get(r[0], sn(r[0])) for r in rows], fontsize=6.2)
    for i, r in enumerate(rows):
        ax.text(r[3] + 1.5, i, f"{r[3]:.1f}", va="center", fontsize=5.6, color="#333333")
    ax.axvline(pooled["either"], color="#555555", lw=0.8, ls="--", zorder=4)
    ax.text(pooled["either"] + 1.5, -0.95, f"pooled {pooled['either']:.1f}%", fontsize=5.6,
            color="#555555", ha="left", va="center")
    ax.set_xlim(0, max(r[3] for r in rows) * 1.20); ax.set_ylim(-1.4, len(rows) - 0.4)
    ax.set_xlabel("Baseline fails, SonoBase succeeds (% of images)", fontsize=6.8)
    ax.set_title("Resolution rate by dataset", pad=4, fontsize=7.5)
    tidy(ax, ygrid=False); ax.set_axisbelow(True)
    ax.xaxis.grid(True, **GRID); ax.yaxis.grid(False)
    panel(ax, "a", dx=-0.42, dy=1.13)

    # ---- b : joint density of per-image IoU
    ax = fig.add_subplot(gs[0, 1])
    ax.add_patch(plt.Rectangle((0, 50), 10, 50, facecolor=C_SB, alpha=0.10,
                               edgecolor="none", zorder=1))
    hb = ax.hexbin(base, sono, gridsize=34, bins="log", cmap="YlGnBu",
                   mincnt=1, linewidths=0, extent=(0, 100, 0, 100), zorder=2)
    ax.plot([0, 100], [0, 100], color="#8A8A8A", lw=0.6, ls=":", zorder=3)
    ax.axvline(10, color=C_SB, lw=0.7, zorder=4)
    ax.axhline(50, color=C_SB, lw=0.7, zorder=4)
    ax.text(12.5, 98, f"{pooled['either']:.1f}%", fontsize=6.4, color="#1A1A1A",
            ha="left", va="top", fontweight="bold",
            bbox=dict(fc="white", ec="none", alpha=0.80, pad=1.0))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100]); ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Weaker baseline IoU", fontsize=6.8)
    ax.set_ylabel("SonoBase IoU", fontsize=6.8)
    ax.set_title(f"Per-image IoU (n = {tot:,})", pad=4, fontsize=7.5)
    tidy(ax, ygrid=False)
    cb = fig.colorbar(hb, ax=ax, fraction=0.055, pad=0.035)
    cb.set_label("images", fontsize=6.0); cb.ax.tick_params(labelsize=5.4, width=0.6, size=2)
    cb.outline.set_linewidth(0.5)
    panel(ax, "b", dx=-0.30, dy=1.13)

    fig.savefig(f"{OUT}/fig4_failure.pdf"); fig.savefig(f"{OUT}/fig4_failure.png", dpi=220)
    plt.close(fig)
    print(f"  fig4_failure.pdf  (either {pooled['either']:.1f}% | both {pooled['both']:.1f}% | "
          f"MedSAM2 {pooled['m2']:.1f}% | SAM2 {pooled['s2']:.1f}% of {tot:,} images)")

# ------------------------------------------------ FIG 4c / SUPP S3 : examples
def _pick_case(d):
    """Median-severity catastrophic case for one dataset.

    Among qualifying images we take the one whose SonoBase IoU is the median,
    so the displayed example is representative of the resolved cases rather
    than the most flattering one available.
    """
    sb, m2, s2 = (_per_image(m, d) for m in ("sonobase", "medsam2", "sam2_no_ft"))
    if not all([sb, m2, s2]): return None
    q = [k for k in sorted(set(sb) & set(m2) & set(s2))
         if min(m2[k], s2[k]) < 0.10 and sb[k] > 0.50]
    if not q: return None
    q.sort(key=lambda k: sb[k])
    return q[len(q) // 2]

def _load_case(d, key):
    """Source frame, GT mask and each model's predicted mask for one case."""
    from PIL import Image
    sid, fidx, oid = key[0], int(key[1]), key[2]
    root = f"{PRED}/sonobase_{d}_point_0corr/{d}/{sid}"
    meta = json.load(open(f"{root}/meta.json"))
    if "image_path" in meta:
        src = meta["image_path"]
    else:
        src = f"{meta['images_dir']}/{fidx:05d}.jpg"
    if not os.path.exists(src): return None
    img = np.asarray(Image.open(src).convert("L"))
    stem = f"{fidx:05d}_obj_{oid}"
    gt = np.asarray(Image.open(f"{root}/gt_{stem}.png").convert("L")) > 127
    preds = {}
    for m, _, _ in FAILM:
        p = f"{PRED}/{m}_{d}_point_0corr/{d}/{sid}/pred_{stem}.png"
        if not os.path.exists(p): return None
        preds[m] = np.asarray(Image.open(p).convert("L")) > 127
    pt = None
    pf = f"{root}/prompts.json"
    if os.path.exists(pf):
        for pr in json.load(open(pf))["prompts"]:
            if pr["frame_idx"] == fidx and str(pr["obj_id"]) == str(oid) and pr.get("points"):
                pt = pr["points"][0]
    ious = {m: _per_image(m, d)[key] for m, _, _ in FAILM}
    return dict(img=img, gt=gt, preds=preds, pt=pt, ious=ious, sid=sid, fidx=fidx)

def _draw(ax, img, mask=None, color=None, pt=None):
    ax.imshow(img, cmap="gray", interpolation="nearest", aspect="equal")
    if mask is not None and mask.any():
        rgba = np.zeros((*mask.shape, 4))
        rgba[mask] = matplotlib.colors.to_rgba(color, 0.45)
        ax.imshow(rgba, interpolation="nearest", aspect="equal")
        ax.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=0.55)
    if pt is not None:
        ax.plot(pt[0], pt[1], marker="+", color="#FFD200", ms=3.6, mew=0.9, zorder=5)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_color("#CCCCCC"); s.set_linewidth(0.5)

def fig_failure_examples(datasets, fname, letter=None):
    """Grid of catastrophic-failure cases: one dataset per row, 5 columns."""
    cases = []
    for d in datasets:
        k = _pick_case(d)
        c = _load_case(d, k) if k else None
        if c is None:
            print(f"    ! {d}: no renderable case, skipped"); continue
        cases.append((d, c))
    if not cases: return

    CW, LEFT, TOP, BOT = 0.855, 0.74, 0.30, 0.06
    heights = [min(1.15, max(0.55, CW * c["img"].shape[0] / c["img"].shape[1]))
               for _, c in cases]
    figw, figh = 5.15, TOP + sum(heights) + BOT
    fig = plt.figure(figsize=(figw, figh))
    gs = fig.add_gridspec(len(cases), 5, height_ratios=heights,
                          left=LEFT / figw, right=0.995, top=1 - TOP / figh,
                          bottom=BOT / figh, wspace=0.05, hspace=0.07)
    heads = ["Input", "Ground truth"] + [lab for _, lab, _ in FAILM]
    for i, (d, c) in enumerate(cases):
        cols = [(None, None), (c["gt"], C_GT)] + \
               [(c["preds"][m], col) for m, _, col in FAILM]
        for j, (mask, col) in enumerate(cols):
            ax = fig.add_subplot(gs[i, j])
            _draw(ax, c["img"], mask, col, c["pt"] if j == 0 else None)
            if i == 0:
                ax.set_title(heads[j], fontsize=6.4, pad=2.5)
            if j >= 2:
                m = FAILM[j - 2][0]
                ax.text(0.035, 0.035, f"IoU {100 * c['ious'][m]:.1f}", transform=ax.transAxes,
                        fontsize=5.4, color=C_TXT.get(col, col), fontweight="bold",
                        va="bottom", ha="left",
                        bbox=dict(fc="white", ec="none", alpha=0.78, pad=0.9))
            if j == 0:
                ax.text(-0.06, 0.58, FSHORT.get(d, sn(d)), transform=ax.transAxes,
                        fontsize=6.3, va="center", ha="right", fontweight="bold")
                ax.text(-0.06, 0.40, c["sid"][:15], transform=ax.transAxes,
                        fontsize=4.6, color="#777777", va="center", ha="right")
        if i == 0 and letter:
            fig.text(0.008, 1 - 0.20 / figh, letter, fontsize=9.5, fontweight="bold",
                     va="top", ha="left")
    fig.savefig(f"{OUT}/{fname}.pdf"); fig.savefig(f"{OUT}/{fname}.png", dpi=220)
    plt.close(fig)
    print(f"  {fname}.pdf  ({len(cases)} rows)")


# ============================================== SUPPLEMENTARY S1 / S2 / S4
def fig_supp_convergence():
    """Per-dataset iterative-refinement curves (S1 benchmark, S2 external) and
    the 4-panel model-variant convergence grid (S4), all from the v2 a3 report."""
    rep = json.load(open(f"{AN}/a3_click_efficiency/analysis_report.json"))
    fac, tier = rep["facet_summaries"], rep["tier_assignment"]
    MK = {"sonobase": "SonoBase", "medsam2": "MedSAM2", "sam2_no_ft": "SAM2-no-ft"}

    def curve(m, prompt, t):
        f = fac.get(f"{m}__{prompt}__{t}")
        if not f: return None, None
        it = sorted(int(i) for i in f["miou_macro_pct"])
        return it, [f["miou_macro_pct"][str(i)] for i in it]

    # ---- S1 / S2: one panel per tier, both prompts, all models
    for tag, t, title in [("S1", "benchmark", "Benchmark datasets"),
                          ("S2", "external", "External datasets")]:
        ds = sorted(d for d, v in tier.items() if v == t)
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
        fig.subplots_adjust(left=0.075, right=0.985, top=0.82, bottom=0.20, wspace=0.22)
        for k, prompt in enumerate(["point", "box"]):
            ax = axes[k]
            for key, lab, colr in MODELS:
                m = {v: kk for kk, v in MK.items()}[key]
                it, ys = curve(m, prompt, t)
                if it is None: continue
                ax.plot(it, ys, "-", color=colr, lw=1.4, marker="o", ms=3,
                        markeredgewidth=0, label=lab, zorder=3)
            ax.axhline(80, color="#555555", lw=0.8, ls=":")
            ax.set_xlabel("Corrective clicks"); ax.set_xlim(-0.25, 7.25)
            ax.set_ylim(0, 105); ax.set_yticks([0, 20, 40, 60, 80, 100])
            if k == 0:
                ax.set_ylabel("mIoU (%)")
                ax.legend(frameon=False, loc="lower right", handlelength=1.1,
                          handletextpad=0.45, borderpad=0.2, labelspacing=0.22)
            ax.set_title(f"{prompt.capitalize()} prompt", pad=5)
            tidy(ax); panel(ax, "ab"[k], dx=-0.14)
        fig.suptitle(f"{title} — macro over {len(ds)} datasets, oracle refinement",
                     fontsize=8, y=0.99)
        fig.savefig(f"{OUT}/{tag.lower()}_iter_{t}.pdf")
        fig.savefig(f"{OUT}/{tag.lower()}_iter_{t}.png", dpi=200); plt.close(fig)
        print(f"  {tag.lower()}_iter_{t}.pdf")

    # ---- S4: 2x2 grid, tier x prompt, all three models
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.90, bottom=0.10,
                        wspace=0.22, hspace=0.42)
    letters = "abcd"
    for i, t in enumerate(["benchmark", "external"]):
        for j, prompt in enumerate(["point", "box"]):
            ax = axes[i][j]
            for key, lab, colr in MODELS:
                m = {v: kk for kk, v in MK.items()}[key]
                it, ys = curve(m, prompt, t)
                if it is None: continue
                ax.plot(it, ys, "-", color=colr, lw=1.4, marker="o", ms=3,
                        markeredgewidth=0, label=lab, zorder=3)
            ax.axhline(80, color="#555555", lw=0.8, ls=":")
            ax.text(7.15, 71.5, "80% threshold", fontsize=5.7, color="#555555", ha="right")
            ax.set_xlim(-0.25, 7.25); ax.set_ylim(0, 105)
            ax.set_yticks([0, 20, 40, 60, 80, 100])
            if i == 1: ax.set_xlabel("Corrective clicks")
            if j == 0: ax.set_ylabel("mIoU (%)")
            ax.set_title(f"{t.capitalize()}, {prompt} prompt", pad=5)
            tidy(ax); panel(ax, letters[i * 2 + j], dx=-0.16)
            if i == 0 and j == 0:
                ax.legend(frameon=False, loc="lower right", handlelength=1.1,
                          handletextpad=0.45, borderpad=0.2, labelspacing=0.22)
    fig.savefig(f"{OUT}/s4_click_convergence.pdf")
    fig.savefig(f"{OUT}/s4_click_convergence.png", dpi=200); plt.close(fig)
    print("  s4_click_convergence.pdf")

# Main text shows the three datasets the section names; S3 shows all eight
# datasets for which per-image scores were retained.
FAIL_MAIN = ["RegPro", "HC18", "MouseBrainTumor"]
FAIL_ALL  = ["RegPro", "HC18", "MouseBrainTumor", "CAMUS", "PorcineSpinalCord",
             "ACOUSLIC", "KidneyUS", "BUSI"]

if __name__ == "__main__":
    import sys
    only = sys.argv[1] if len(sys.argv) > 1 else None
    print("writing figures to", OUT)
    if only in (None, "failure"):
        fig_failure()
        fig_failure_examples(FAIL_MAIN, "fig4_qualitative", letter="c")
        fig_failure_examples(FAIL_ALL, "s3_qualitative")
    if only is None:
        fig_generalization(); fig_clinical(); fig_workflow(); fig_breadth()
        fig_supp_convergence()
    print("done")
