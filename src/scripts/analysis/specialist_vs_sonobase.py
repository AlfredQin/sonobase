#!/usr/bin/env python
"""Aggregate the specialist baselines next to SonoBase / MedSAM2 / SAM2.

Segmentation cells use exactly the paper-table definition (as in
`aggregate_prompt_noise.cell`: 100 * mean(iou) and 100 * mean(dice) over EVERY row
of per_sample_metrics.csv, then macro over datasets).
Baseline rows come from the noised 3-seed record (`prompt_noise_3seed.csv`, mean ± sd) — the Tables 1-2 protocol —
and, for reference, from the deterministic `*_box_0corr` / `*_point_0corr` runs. Specialist rows are single runs
(`<label>_<DS>_none`, unprompted), scored on the identical row population (importer `--rows-from`).

Outputs (into --out-dir):
  specialist_vs_sonobase.csv        one row per (dataset, model, protocol): miou, miou_sd, dice, dice_sd, n_rows
  specialist_by_category.csv        per-category cells (CAMUS / PFUS / Brachial-Plexus) incl. a no_gt_this_frame-filtered variant
  specialist_clinical.csv           scalar fields of every model in the A1 / A2 / A4 analysis_report.json files given
  table1_specialist_rows.tex        the extra "Specialist (unprompted)" block for Table 1 (8 Benchmark columns + Avg)
  specialist_paired_tests.csv       per dataset: specialist vs SonoBase box/point (exact) on the identical rows — paired Wilcoxon on
                                    IoU at row level and at sample level (per-video means; rows of one video are not independent),
                                    bootstrap 95 % CI of the mean IoU difference (seed 42), BH-FDR over the whole family

    uv run python scripts/analysis/specialist_vs_sonobase.py --pred-root experiments/analysis/predictions \
        --noised-csv <path>/prompt_noise_3seed.csv \
        --specialists nnunet_resenc_m dlv3p_r50 echonet_lv_zeroshot echonet_lv_ft \
        --clinical a1=experiments/analysis/a1_camus_ef/CAMUS_specialists/analysis_report.json ... --out-dir <exp>/results
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import statistics as st
import sys
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

BENCHMARK = ["BUSI", "Brachial-Plexus", "C-TRUS", "CAMUS", "HC18", "PFUS", "RegPro", "TG3K"]
EXTERNAL = ["ACOUSLIC", "BUS-BRA", "DDTI", "FUGC", "KidneyUS", "LUMINOUS", "MMOTU-3d"]
TEX_NAMES = {"nnunet_resenc_m": "nnU-Net ResEnc-M (2d, per dataset)", "dlv3p_r50": "DeepLabV3+ (R50)",
             "echonet_lv_zeroshot": "EchoNet-Dynamic LV (zero-shot; endocardium rows only)", "echonet_lv_ft": "EchoNet-Dynamic LV (CAMUS fine-tuned; endocardium rows only)",
             "acouslic_a3_cv": "ACOUSLIC-AI A3 (5-fold CV)", "acouslic_a2_cv": "ACOUSLIC-AI A2 (5-fold CV)", "acouslic_a1_cv": "ACOUSLIC-AI A1 (5-fold CV)"}


def rows_of(run: pathlib.Path) -> List[dict]:
    p = run / "per_sample_metrics.csv"
    return list(csv.DictReader(p.open())) if p.exists() else []


def cell(rows: List[dict]) -> Optional[Tuple[float, float, int]]:
    """Identical to aggregate_prompt_noise.cell: 100*mean(iou), 100*mean(dice), n over all rows."""
    iou, dice = [], []
    for r in rows:
        try:
            iou.append(float(r["iou"])); dice.append(float(r["dice"]))
        except (ValueError, KeyError):
            continue
    return (100 * st.mean(iou), 100 * st.mean(dice), len(iou)) if iou else None


def load_noised(path: pathlib.Path):
    out = {}
    for r in csv.DictReader(path.open()):
        out[(r["prompt"], r["dataset"], r["model"])] = (float(r["miou_mean"]), float(r["miou_sd"]), float(r["dice_mean"]), float(r["dice_sd"]), int(r["n"]))
    return out


def scalars(d: dict) -> dict:
    return {k: v for k, v in d.items() if isinstance(v, (int, float, str)) and not isinstance(v, bool)}


def _paired_rows(spec_rows: List[dict], ref_rows: List[dict]) -> Tuple[List[float], List[float], List[str]]:
    """Align specialist and reference rows on (sample_id, frame_idx, obj_id); returns (x, y, sample_ids)."""
    key = lambda r: (r["sample_id"], r["frame_idx"], r["obj_id"])
    ref = {key(r): float(r["iou"]) for r in ref_rows}
    x, y, sid = [], [], []
    for r in spec_rows:
        k = key(r)
        if k in ref:
            x.append(float(r["iou"])); y.append(ref[k]); sid.append(r["sample_id"])
    return x, y, sid


def _paired_test(x: List[float], y: List[float], n_boot: int = 2000, seed: int = 42) -> dict:
    """Paired Wilcoxon (scipy, zero-diff rows dropped by the 'wilcox' rule) + bootstrap CI of mean(x - y)."""
    from scipy.stats import wilcoxon
    d = [a - b for a, b in zip(x, y)]
    n = len(d); mean_d = st.mean(d) if d else float("nan")
    nz = [v for v in d if v != 0.0]
    if len(nz) >= 10:
        stat, pval = wilcoxon(nz, zero_method="wilcox", alternative="two-sided")
        pval = float(pval)
    else:
        pval = float("nan")
    rng = random.Random(seed)
    boots = []
    if n:
        for _ in range(n_boot):
            boots.append(st.mean(d[rng.randrange(n)] for _ in range(n)))
        boots.sort(); lo, hi = boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]
    else:
        lo = hi = float("nan")
    return {"n": n, "mean_diff": 100 * mean_d, "ci95_lo": 100 * lo, "ci95_hi": 100 * hi, "p_wilcoxon": pval, "n_nonzero_diff": len(nz)}


def _bh(pvals: List[float], alpha: float = 0.05) -> List[float]:
    """Benjamini-Hochberg q-values (NaN p stay NaN)."""
    idx = [i for i, p in enumerate(pvals) if p == p]
    m = len(idx); q = [float("nan")] * len(pvals)
    order = sorted(idx, key=lambda i: pvals[i]); prev = 1.0
    for rank, i in reversed(list(enumerate(order, start=1))):
        val = min(prev, pvals[i] * m / rank); q[i] = val; prev = val
    return q


def paired_tests(pred: pathlib.Path, datasets: List[str], specialists: List[str]) -> List[dict]:
    out = []
    for ds in datasets:
        for label in specialists:
            spec = rows_of(pred / f"{label}_{ds}_none")
            if not spec:
                continue
            for prompt in ("box", "point"):
                ref = rows_of(pred / f"sonobase_{ds}_{prompt}_0corr")
                x, y, sid = _paired_rows(spec, ref)
                if not x:
                    continue
                res = _paired_test(x, y); res.update({"dataset": ds, "specialist": label, "sonobase_protocol": f"{prompt}_exact", "level": "row"}); out.append(res)
                per = defaultdict(lambda: [[], []])
                for a, b, s_ in zip(x, y, sid):
                    per[s_][0].append(a); per[s_][1].append(b)
                if len(per) < len(x):  # video / volume datasets: rows of one sample are not independent
                    xs = [st.mean(v[0]) for v in per.values()]; ys = [st.mean(v[1]) for v in per.values()]
                    res = _paired_test(xs, ys); res.update({"dataset": ds, "specialist": label, "sonobase_protocol": f"{prompt}_exact", "level": "sample"}); out.append(res)
    q = _bh([r["p_wilcoxon"] for r in out])
    for r, qq in zip(out, q):
        r["q_bh"] = qq; r["significant_q05"] = ("yes" if qq == qq and qq < 0.05 else ("no" if qq == qq else ""))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-root", required=True)
    ap.add_argument("--noised-csv", required=True)
    ap.add_argument("--specialists", nargs="+", required=True, help="labels of <label>_<DS>_none run dirs")
    ap.add_argument("--datasets", nargs="+", default=BENCHMARK + EXTERNAL)
    ap.add_argument("--clinical", nargs="*", default=[], help="analysis=path/analysis_report.json (a1|a2|a4)")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    pred = pathlib.Path(args.pred_root); out = pathlib.Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    noised = load_noised(pathlib.Path(args.noised_csv))

    main_rows, cat_rows = [], []
    for ds in args.datasets:
        tier = "benchmark" if ds in BENCHMARK else "external"
        for prompt in ("point", "box"):
            for m in ("sonobase", "medsam2", "sam2_no_ft"):
                v = noised.get((prompt, ds, m))
                if v:
                    main_rows.append({"dataset": ds, "tier": tier, "model": m, "protocol": f"{prompt}_noised_3seed", "miou": v[0], "miou_sd": v[1], "dice": v[2], "dice_sd": v[3], "n_rows": v[4]})
                c = cell(rows_of(pred / f"{m}_{ds}_{prompt}_0corr"))
                if c:
                    main_rows.append({"dataset": ds, "tier": tier, "model": m, "protocol": f"{prompt}_exact", "miou": c[0], "miou_sd": "", "dice": c[1], "dice_sd": "", "n_rows": c[2]})
        for label in args.specialists:
            rows = rows_of(pred / f"{label}_{ds}_none")
            c = cell(rows)
            if not c:
                continue
            main_rows.append({"dataset": ds, "tier": tier, "model": label, "protocol": "none", "miou": c[0], "miou_sd": "", "dice": c[1], "dice_sd": "", "n_rows": c[2]})
            bycat = defaultdict(list)
            for r in rows:
                bycat[(r.get("category_id") or "", r.get("category_name") or "")].append(r)
            if len(bycat) > 1 or any(r.get("notes") == "no_gt_this_frame" for r in rows):
                ref = rows_of(pred / f"sonobase_{ds}_box_0corr")
                refcat = defaultdict(list)
                for r in ref:
                    refcat[(r.get("category_id") or "", r.get("category_name") or "")].append(r)
                for key, rr in sorted(bycat.items()):
                    for variant, filt in (("all_rows", lambda r: True), ("gt_frames_only", lambda r: r.get("notes") != "no_gt_this_frame")):
                        c1 = cell([r for r in rr if filt(r)]); c2 = cell([r for r in refcat.get(key, []) if filt(r)])
                        cat_rows.append({"dataset": ds, "category_id": key[0], "category_name": key[1], "variant": variant, "model": label,
                                         "miou": c1[0] if c1 else "", "dice": c1[1] if c1 else "", "n_rows": c1[2] if c1 else 0,
                                         "sonobase_box_exact_miou": c2[0] if c2 else "", "sonobase_box_exact_dice": c2[1] if c2 else ""})

    with (out / "specialist_vs_sonobase.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "tier", "model", "protocol", "miou", "miou_sd", "dice", "dice_sd", "n_rows"]); w.writeheader(); w.writerows(main_rows)
    with (out / "specialist_by_category.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "category_id", "category_name", "variant", "model", "miou", "dice", "n_rows", "sonobase_box_exact_miou", "sonobase_box_exact_dice"]); w.writeheader(); w.writerows(cat_rows)

    clin = []
    for spec in args.clinical:
        name, path = spec.split("=", 1)
        rep = json.load(open(path))
        for model, d in rep.get("models", {}).items():
            row = {"analysis": name, "report": path, "model": model}; row.update(scalars(d)); clin.append(row)
    if clin:
        fields = ["analysis", "report", "model"] + sorted({k for r in clin for k in r} - {"analysis", "report", "model"})
        with (out / "specialist_clinical.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(clin)

    tests = paired_tests(pred, args.datasets, args.specialists)
    with (out / "specialist_paired_tests.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "specialist", "sonobase_protocol", "level", "n", "mean_diff", "ci95_lo", "ci95_hi", "p_wilcoxon", "n_nonzero_diff", "q_bh", "significant_q05"]); w.writeheader(); w.writerows(tests)

    # Table 1 block (mIoU %, Benchmark columns); "--" where a specialist has no run
    lines = ["\\midrule", "\\multicolumn{10}{l}{\\textit{Specialist, unprompted (trained on the same train split; single run)}} \\\\"]
    for label in args.specialists:
        vals = {r["dataset"]: r["miou"] for r in main_rows if r["model"] == label and r["protocol"] == "none"}
        if not any(ds in vals for ds in BENCHMARK):
            continue
        cells = [f"{vals[ds]:.1f}" if ds in vals else "--" for ds in BENCHMARK]
        have = [vals[ds] for ds in BENCHMARK if ds in vals]
        avg = f"{st.mean(have):.1f}" + ("" if len(have) == len(BENCHMARK) else f"$^{{({len(have)}/8)}}$")
        lines.append(f"{TEX_NAMES.get(label, label)} & " + " & ".join(cells) + f" & {avg} \\\\")
    (out / "table1_specialist_rows.tex").write_text("\n".join(lines) + "\n")
    print(f"{len(main_rows)} rows, {len(cat_rows)} category rows, {len(clin)} clinical rows, {len(tests)} paired tests -> {out}")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
