#!/usr/bin/env python
"""Recompute bootstrap 95% CIs for the clinical MAE endpoints (Table S21).

Computes them from the per-sample records with the protocol stated in the table
caption: 10,000 bootstrap resamples, seed 42, percentile CI.

Prints a report and the LaTeX rows. Writes nothing.
"""
import argparse, csv, datetime, json, os, shutil, statistics as st
import numpy as np

# Analysis records: <SONOBASE_DATA_ROOT or this repo>/src/experiments/analysis.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
AN = os.environ.get("ANALYSIS_DIR",
                    os.path.join(os.environ.get("SONOBASE_DATA_ROOT", _REPO),
                                 "src", "experiments", "analysis"))

# analysis -> (directory, dataset stem, error-column prefix, unit)
ENDPOINTS = {
    "A2 HC18 (HC)":     ("a2_hc18_hc",      "HC18",     "abs_err_mm__", "mm"),
    "A4 ACOUSLIC (AC)": ("a4_acouslic_ac",  "ACOUSLIC", "abs_err_mm__", "mm"),
    # A1 and C2 are included so the recomputation can be validated against the
    # entries the archive *does* get right.
    "A1 CAMUS (EF)":    ("a1_camus_ef",     "CAMUS",    "abs_err_pct__", "%"),
    "C2 RegPro (vol)":  ("c2_regpro_volume", "RegPro",  "abs_err_ml__", "mL"),
}
MODELS = [("sonobase", "SonoBase"), ("medsam2", "MedSAM2"), ("sam2_no_ft", "SAM2")]

# the archive's bootstrap_ci block also carries a ground-truth control row per
# endpoint where one exists; keep it so the regenerated block is a superset-free
# replacement rather than a lossy one.
GT_COL = {"a1_camus_ef": "GT (Simpson's biplane)", "a2_hc18_hc": "GT (ellipse fit)"}
ARCHIVE = {                       # analysis code -> (dir, stem, prefix, metric, unit)
    "A1": ("a1_camus_ef", "CAMUS", "abs_err_pct__", "EF", "%"),
    "A2": ("a2_hc18_hc", "HC18", "abs_err_mm__", "HC", "mm"),
    "A4": ("a4_acouslic_ac", "ACOUSLIC", "abs_err_mm__", "AC", "mm"),
    "C2": ("c2_regpro_volume", "RegPro", "abs_err_ml__", "Volume", "mL"),
}


def regenerate(arm, n_boot, seed, dry_run=True):
    """Rebuild t1_1_significance/<arm>/analysis_report.json -> bootstrap_ci.

    The archived block is correct for A1 and C2 but records different quantities
    entirely for A2 and A4 (HC18 SonoBase 17.710 where the A2 report says 2.49;
    ACOUSLIC SonoBase 256.779 where A4 says 56.19). Regenerating the whole block
    with one documented seed makes it reproducible end to end rather than leaving
    a mixture of provenances.
    """
    path = f"{AN}/t1_1_significance/{arm}/analysis_report.json"
    if not os.path.exists(path):
        print(f"  ! {path} not found"); return
    rep = json.load(open(path))
    old = rep.get("bootstrap_ci", [])

    block = []
    for code, (d, stem, prefix, metric, unit) in ARCHIVE.items():
        # keep the archive's own model naming (raw keys, GT row verbatim) so the
        # regenerated block is a drop-in replacement and diffs cleanly
        models = []
        if d in GT_COL:
            models.append((GT_COL[d], GT_COL[d]))
        models += [(m, m) for m, _ in MODELS]
        for col, label in models:
            e = errors(d, stem, arm, prefix, col)
            if not e:
                continue
            mae, lo, hi = boot(e, n_boot, seed)
            block.append({"analysis": code, "metric": metric, "unit": unit,
                          "model": label, "n": len(e), "mae": mae,
                          "ci_lo": lo, "ci_hi": hi})

    print(f"  {arm}: {len(old)} archived entries -> {len(block)} regenerated")
    changed = 0
    for new in block:
        prev = next((o for o in old if o.get("analysis") == new["analysis"]
                     and o.get("model") == new["model"]), None)
        if prev is None:
            print(f"    + {new['analysis']} {new['model'][:22]:22} (new)"); changed += 1
        elif abs(prev.get("mae", 0) - new["mae"]) > 0.01:
            print(f"    ~ {new['analysis']} {new['model'][:22]:22} "
                  f"mae {prev['mae']:.3f} -> {new['mae']:.3f}   CORRECTED"); changed += 1
    if dry_run:
        print(f"    (dry run -- {changed} entries would change; pass --write to apply)")
        return

    bak = path.replace(".json", f".json.bak-{datetime.date.today().isoformat()}")
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
    rep["bootstrap_ci"] = block
    rep["bootstrap_ci_provenance"] = {
        "regenerated": datetime.date.today().isoformat(),
        "reason": ("archived A2/A4 entries recorded different quantities than the A2/A4 "
                   "analysis reports (e.g. HC18 sonobase 17.710 vs 2.49); A1/C2 were correct"),
        "method": f"percentile bootstrap of the mean, {n_boot} resamples, numpy default_rng(seed={seed})",
        "source": "the per_sample.csv / per_patient.csv of each endpoint",
        "script": "src/scripts/analysis/rerun_bootstrap_ci.py",
        "original_backed_up_to": os.path.basename(bak),
    }
    json.dump(rep, open(path, "w"), indent=1)
    print(f"    written; original -> {os.path.basename(bak)}")


def errors(directory, stem, arm, prefix, model):
    for name in ("per_sample.csv", "per_patient.csv"):
        p = f"{AN}/{directory}/{stem}_{arm}_0corr/{name}"
        if not os.path.exists(p):
            continue
        col = f"{prefix}{model}"
        out = []
        for r in csv.DictReader(open(p)):
            v = r.get(col, "")
            if v not in ("", None):
                try:
                    out.append(float(v))
                except ValueError:
                    pass
        if out:
            return out
    return []


def boot(vals, n_boot=10000, seed=42):
    """Percentile bootstrap CI of the mean. seed 42, matching the S21 caption."""
    a = np.asarray(vals, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    means = a[idx].mean(axis=1)
    return float(a.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--regenerate", action="store_true",
                    help="rebuild the t1_1_significance bootstrap_ci block")
    ap.add_argument("--write", action="store_true", help="with --regenerate, apply the change")
    a = ap.parse_args()

    if a.regenerate:
        print(f"regenerating bootstrap_ci  ({a.n_boot:,} resamples, seed {a.seed})\n")
        for arm in ("point", "box"):
            regenerate(arm, a.n_boot, a.seed, dry_run=not a.write)
        return

    print(f"percentile bootstrap, {a.n_boot:,} resamples, seed {a.seed}\n")
    print(f"{'endpoint':18} {'model':10} {'arm':6} {'n':>5} {'MAE':>10} {'95% CI':>22}")
    rows = {}
    for label, (d, stem, prefix, unit) in ENDPOINTS.items():
        for m, disp in MODELS:
            for arm in ("point", "box"):
                e = errors(d, stem, arm, prefix, m)
                if not e:
                    print(f"{label:18} {disp:10} {arm:6} {'--':>5}  (no per-sample column)")
                    continue
                mae, lo, hi = boot(e, a.n_boot, a.seed)
                rows[(label, disp, arm)] = (len(e), mae, lo, hi)
                print(f"{label:18} {disp:10} {arm:6} {len(e):>5} {mae:>10.2f} "
                      f"{'[' + format(lo, '.2f') + ', ' + format(hi, '.2f') + ']':>22}")
        print()

    print("=" * 78)
    print("LaTeX rows for the A2 / A4 blocks of Table S21")
    print("=" * 78)
    for label, block in (("A2 HC18 (HC)", "HC18 HC"), ("A4 ACOUSLIC (AC)", "ACOUSLIC AC")):
        first = True
        for m, disp in MODELS:
            for arm, an in (("point", "Point"), ("box", "Box")):
                k = (label, disp, arm)
                if k not in rows:
                    continue
                n, mae, lo, hi = rows[k]
                lead = f"\\multirow{{6}}{{*}}{{{block}}} " if first else ""
                first = False
                print(f"{lead}& {disp} ({an}) & {mae:.2f} & [{lo:.2f}, {hi:.2f}] \\\\")
        print("\\hline")


if __name__ == "__main__":
    main()
