#!/usr/bin/env python
"""Compute the EF gray-zone analysis (Table S8) from the per-patient records.

Reads a1_camus_ef/CAMUS_{point,box}_0corr/per_patient.csv and uses the EF range
30-50 stated in the manuscript and the ASE disc-pairing EF (`pred_ef_biplane_ase__`),
the paper's primary convention. The view-averaged `pred_ef_biplane__` is printed
alongside for comparison.

Writes nothing. Prints a report and the LaTeX body for review.
"""
import argparse, csv, math, os, statistics as st

# Analysis records: <SONOBASE_DATA_ROOT or this repo>/src/experiments/analysis.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
AN = os.environ.get("ANALYSIS_DIR",
                    os.path.join(os.environ.get("SONOBASE_DATA_ROOT", _REPO),
                                 "src", "experiments", "analysis"))
MODELS = [("sonobase", "SonoBase"), ("medsam2", "MedSAM2"), ("sam2_no_ft", "SAM2")]


def load(arm):
    p = f"{AN}/a1_camus_ef/CAMUS_{arm}_0corr/per_patient.csv"
    return list(csv.DictReader(open(p)))


def cohort(rows, lo, hi, model, field):
    """(gt, pred) pairs whose GROUND-TRUTH EF lies in [lo, hi] and whose prediction exists."""
    g, p = [], []
    for r in rows:
        try:
            gt = float(r["gt_ef"]); pr = float(r[f"{field}__{model}"])
        except (KeyError, ValueError):
            continue
        if not (math.isfinite(gt) and math.isfinite(pr)):
            continue
        if lo <= gt <= hi:
            g.append(gt); p.append(pr)
    return g, p


def summarize(g, p, thresholds=(40.0, 35.0)):
    n = len(g)
    if n == 0:
        return None
    err = [abs(a - b) for a, b in zip(p, g)]
    diff = [a - b for a, b in zip(p, g)]
    out = {"n": n, "mae": st.mean(err),
           "std": st.stdev(err) if n > 1 else 0.0,
           "bias": st.mean(diff)}
    mg, mp = st.mean(g), st.mean(p)
    num = sum((a - mg) * (b - mp) for a, b in zip(g, p))
    den = math.sqrt(sum((a - mg) ** 2 for a in g) * sum((b - mp) ** 2 for b in p))
    out["r"] = num / den if den else float("nan")
    for t in thresholds:                       # reclassification across an HFrEF cut-off
        gl = [a <= t for a in g]; pl = [b <= t for b in p]
        out[f"reclass_{int(t)}"] = 100.0 * sum(x != y for x, y in zip(gl, pl)) / n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", type=float, default=30.0)
    ap.add_argument("--hi", type=float, default=50.0)
    args = ap.parse_args()

    for field, tag in (("pred_ef_biplane_ase", "ASE disc-pairing (paper primary)"),
                       ("pred_ef_biplane", "view-averaged (archived module default)")):
        print(f"\n{'='*76}\n{tag}   gray zone GT EF in [{args.lo:.0f}, {args.hi:.0f}]\n{'='*76}")
        print(f"{'model':10} {'arm':6} {'n':>4} {'MAE':>8} {'std':>8} {'r':>7} "
              f"{'bias':>8} {'recl@40':>8} {'recl@35':>8}")
        for m, disp in MODELS:
            for arm in ("point", "box"):
                s = summarize(*cohort(load(arm), args.lo, args.hi, m, field))
                if s is None:
                    print(f"{disp:10} {arm:6} {'--':>4}"); continue
                print(f"{disp:10} {arm:6} {s['n']:>4} {s['mae']:>8.2f} {s['std']:>8.2f} "
                      f"{s['r']:>7.3f} {s['bias']:>+8.2f} {s['reclass_40']:>7.1f}% "
                      f"{s['reclass_35']:>7.1f}%")

    print(f"\n{'='*76}\nLaTeX body for Table S8 (ASE primary, {args.lo:.0f}-{args.hi:.0f})\n{'='*76}")
    for m, disp in MODELS:
        cells = []
        for arm in ("point", "box"):
            s = summarize(*cohort(load(arm), args.lo, args.hi, m, "pred_ef_biplane_ase"))
            cells.append(s)
        if not all(cells):
            continue
        lead = f"\\multirow{{2}}{{*}}{{{disp}}} "
        for i, (arm, s) in enumerate(zip(("Point", "Box"), cells)):
            pre = lead if i == 0 else ""
            print(f"{pre}& {arm} & {s['mae']:.2f} & {s['reclass_40']:.1f}\\% & "
                  f"{s['reclass_35']:.1f}\\% \\\\")
        print("\\hline")
    n = summarize(*cohort(load("box"), args.lo, args.hi, "sonobase", "pred_ef_biplane_ase"))["n"]
    print(f"\ncohort size n = {n} (patients with GT EF in range and a scored prediction)")


if __name__ == "__main__":
    main()
