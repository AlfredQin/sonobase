"""Analysis T1.3 — held-out linear recalibration of a clinical measurement.

Stage-2 analysis. The protocol of Supplementary Table S9b (EF), applied to any
per-sample record with a ground-truth column and one prediction column per
model:

  * Linear map GT = a + b * pred, fitted by ordinary least squares on the
    training fold only and applied to the held-out fold.
  * Split-half: `--n-splits` random 50/50 splits (NumPy Generator seeded with
    `--seed`, the same split sequence for every model so splits are paired);
    reported as the mean MAE over splits and the 2.5th–97.5th percentile of
    the split MAEs — a spread across splits, not a confidence interval.
  * Leave-one-out (LOO): fit on n-1, evaluate the held-out sample; MAE over all.
  * Sensitivity: bias-only correction (b fixed at 1, a = mean training residual).
  * Paired two-sided Wilcoxon signed-rank on the LOO-calibrated per-sample
    absolute errors, target model vs each baseline.
  * Leakage probe: the in-sample refit MAE must not exceed the LOO MAE.

Held-out calibration uses test labels for the fit (as S9b does); it is a
secondary analysis and must be labelled as such wherever it is reported.

`--group-col` (e.g. `seed` for few-shot records) runs the whole procedure
within each group and additionally reports the mean ± SD of the LOO MAE
across groups and a Wilcoxon on the per-sample LOO errors averaged over
groups (the paired unit used for the few-shot AC endpoint).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import pathlib
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from nemo_cv.components.analysis.stats.tests import paired_wilcoxon

logger = logging.getLogger(__name__)


def ols(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Closed-form OLS for y ~ a + b*x; returns (a, b)."""
    xm, ym = x.mean(), y.mean()
    b = float(((x - xm) * (y - ym)).sum() / ((x - xm) ** 2).sum())
    return float(ym - b * xm), b


def _fit(p: np.ndarray, g: np.ndarray, bias_only: bool) -> Tuple[float, float]:
    return ((g - p).mean(), 1.0) if bias_only else ols(p, g)


def split_half(p: np.ndarray, g: np.ndarray, n_splits: int, seed: int, bias_only: bool = False) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(g); maes = np.empty(n_splits)
    for k in range(n_splits):
        idx = rng.permutation(n); tr, te = idx[: n // 2], idx[n // 2:]
        a, b = _fit(p[tr], g[tr], bias_only)
        maes[k] = np.mean(np.abs(a + b * p[te] - g[te]))
    return maes


def loo(p: np.ndarray, g: np.ndarray, bias_only: bool = False) -> np.ndarray:
    n = len(g); err = np.empty(n); mask = np.ones(n, dtype=bool)
    for i in range(n):
        mask[i] = False
        a, b = _fit(p[mask], g[mask], bias_only)
        err[i] = abs(a + b * p[i] - g[i]); mask[i] = True
    return err


def calibrate_one(p: np.ndarray, g: np.ndarray, n_splits: int, seed: int) -> Dict:
    raw = p - g
    sh = split_half(p, g, n_splits, seed)
    e_loo = loo(p, g); e_bias = loo(p, g, bias_only=True)
    a, b = ols(p, g); in_sample = float(np.mean(np.abs(a + b * p - g)))
    if in_sample > e_loo.mean() + 1e-9:
        raise RuntimeError("leakage probe failed: in-sample MAE exceeds LOO MAE")
    return dict(n=int(len(g)), raw_mae=float(np.abs(raw).mean()), bias=float(raw.mean()),
                split_half_mae=float(sh.mean()), split_half_p2_5=float(np.percentile(sh, 2.5)),
                split_half_p97_5=float(np.percentile(sh, 97.5)), loo_mae=float(e_loo.mean()),
                bias_only_loo_mae=float(e_bias.mean()), in_sample_mae=in_sample,
                fit_a=a, fit_b=b, _loo_err=e_loo)


def run(rows: List[Dict[str, str]], gt_col: str, pred_cols: Dict[str, str], id_col: str, target: str,
        n_splits: int, seed: int, group_col: Optional[str]) -> Tuple[Dict, List[Dict]]:
    groups = sorted({r[group_col] for r in rows}, key=lambda v: (len(v), v)) if group_col else [None]
    out: Dict = {"protocol": dict(n_splits=n_splits, seed=seed, fit="GT = a + b*pred (OLS, training fold only)",
                                  split_half="mean MAE over splits; [2.5, 97.5] percentile of split MAEs",
                                  loo="fit on n-1, evaluate held-out; MAE over all",
                                  test="paired two-sided Wilcoxon on LOO-calibrated |error|"),
                 "groups": {}}
    per_sample_rows: List[Dict] = []
    loo_by_model_group: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(lambda: defaultdict(dict))  # model -> group -> id -> err
    for grp in groups:
        sub = [r for r in rows if (group_col is None or r[group_col] == grp)]
        sub = [r for r in sub if r[gt_col] not in ("", None) and all(r[c] not in ("", None) for c in pred_cols.values())]
        ids = [r[id_col] for r in sub]; g = np.array([float(r[gt_col]) for r in sub])
        res = {}
        for model, col in pred_cols.items():
            p = np.array([float(r[col]) for r in sub])
            res[model] = calibrate_one(p, g, n_splits, seed)
            for i, sid in enumerate(ids):
                loo_by_model_group[model][str(grp)][sid] = float(res[model]["_loo_err"][i])
        tests = {}
        for base in pred_cols:
            if base == target: continue
            W, pv = paired_wilcoxon(res[target]["_loo_err"], res[base]["_loo_err"])
            tests[f"{target}_vs_{base}"] = dict(p=float(pv), W=float(W),
                                             lower_mae=target if res[target]["loo_mae"] < res[base]["loo_mae"] else base)
        for i, sid in enumerate(ids):
            row = {id_col: sid, gt_col: f"{g[i]:.4f}"}
            if group_col: row[group_col] = grp
            for model in pred_cols: row[f"loo_cal_abs_err__{model}"] = f"{res[model]['_loo_err'][i]:.4f}"
            per_sample_rows.append(row)
        out["groups"][str(grp)] = {"models": {m: {k: v for k, v in r.items() if not k.startswith("_")} for m, r in res.items()}, "tests": tests}
    if group_col and len(groups) > 1:
        agg = {}
        for model in pred_cols:
            vals = np.array([out["groups"][str(grp)]["models"][model]["loo_mae"] for grp in groups])
            raws = np.array([out["groups"][str(grp)]["models"][model]["raw_mae"] for grp in groups])
            shs = np.array([out["groups"][str(grp)]["models"][model]["split_half_mae"] for grp in groups])
            agg[model] = dict(raw_mae_mean=float(raws.mean()), raw_mae_sd=float(raws.std(ddof=1)),
                              split_half_mae_mean=float(shs.mean()), split_half_mae_sd=float(shs.std(ddof=1)),
                              loo_mae_mean=float(vals.mean()), loo_mae_sd=float(vals.std(ddof=1)))
        # paired test on per-sample LOO errors averaged over groups
        def seed_avg(model):
            acc = defaultdict(list)
            for grp in groups:
                for sid, e in loo_by_model_group[model][str(grp)].items(): acc[sid].append(e)
            return {sid: float(np.mean(v)) for sid, v in acc.items()}
        t = seed_avg(target); tests = {}
        for base in pred_cols:
            if base == target: continue
            b = seed_avg(base); common = sorted(set(t) & set(b))
            W, pv = paired_wilcoxon(np.array([t[s] for s in common]), np.array([b[s] for s in common]))
            tests[f"{target}_vs_{base}"] = dict(p=float(pv), W=float(W), n=len(common),
                                             lower_mae=target if agg[target]["loo_mae_mean"] < agg[base]["loo_mae_mean"] else base)
        out["across_groups"] = {"models": agg, "tests_on_group_averaged_loo_errors": tests}
    return out, per_sample_rows


def fewshot_rows(raw_csv: str, N: int, prompt: str, models: List[str]) -> List[Dict[str, str]]:
    """Per-video, per-seed prediction table from `few_shot/aggregate.py`'s <DATASET>_raw.csv
    (per-frame rows; `pred_ac_mm` is already the per-video mean). Columns produced:
    seed, sample_id, gt_ac_mm, pred_ac_mm__<model>."""
    table: Dict[Tuple[str, str], Dict[str, str]] = {}
    for r in csv.DictReader(open(raw_csv)):
        if int(r["N"]) != N or r["prompt"] != prompt or r["model"] not in models: continue
        if r["pred_ac_mm"] in ("", None) or r["gt_ac_mm"] in ("", None): continue
        key = (r["seed"], r["sample_id"])
        row = table.setdefault(key, {"seed": r["seed"], "sample_id": r["sample_id"], "gt_ac_mm": r["gt_ac_mm"]})
        row[f"pred_ac_mm__{r['model']}"] = r["pred_ac_mm"]
    rows = [row for row in table.values() if all(f"pred_ac_mm__{m}" in row for m in models)]
    logger.info(f"few-shot N={N} {prompt}: {len(rows)} (seed, video) rows from {raw_csv}")
    return rows


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    p = argparse.ArgumentParser(description="Held-out linear recalibration (S9b protocol) — T1.3.")
    p.add_argument("--input", default=None, help="Per-sample CSV (A1/A2/A4/C2 style).")
    p.add_argument("--fewshot-raw", default=None, help="Alternative input: few_shot <DATASET>_raw.csv; builds a per-video, per-seed table (implies --group-col seed).")
    p.add_argument("--fewshot-N", type=int, default=5)
    p.add_argument("--fewshot-prompt", default="box")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--id-col", default="sample_id")
    p.add_argument("--gt-col", required=True)
    p.add_argument("--pred-template", required=True, help="Prediction column template with {model}, e.g. pred_ac_mm__{model}")
    p.add_argument("--models", nargs="+", default=["sonobase", "medsam2", "sam2_no_ft"])
    p.add_argument("--target-model", default="sonobase")
    p.add_argument("--group-col", default=None)
    p.add_argument("--n-splits", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if (args.input is None) == (args.fewshot_raw is None):
        p.error("give exactly one of --input or --fewshot-raw")
    if args.fewshot_raw:
        rows = fewshot_rows(args.fewshot_raw, args.fewshot_N, args.fewshot_prompt, args.models)
        args.group_col = args.group_col or "seed"
    else:
        rows = list(csv.DictReader(open(args.input)))
    pred_cols = {m: args.pred_template.format(model=m) for m in args.models}
    out, per_sample = run(rows, args.gt_col, pred_cols, args.id_col, args.target_model, args.n_splits, args.seed, args.group_col)
    out["inputs"] = dict(input=str(args.input or args.fewshot_raw), fewshot=(None if not args.fewshot_raw else dict(N=args.fewshot_N, prompt=args.fewshot_prompt)),
                         gt_col=args.gt_col, pred_cols=pred_cols, group_col=args.group_col)
    od = pathlib.Path(args.output_dir).expanduser().resolve(); od.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(od / "recalibration_report.json", "w"), indent=1)
    with open(od / "loo_calibrated_errors.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_sample[0].keys())); w.writeheader(); w.writerows(per_sample)
    for grp, g in out["groups"].items():
        logger.info(f"group={grp}")
        for m, r in g["models"].items():
            logger.info(f"  {m:11s} raw {r['raw_mae']:7.2f} bias {r['bias']:+7.2f} | split-half {r['split_half_mae']:6.2f} [{r['split_half_p2_5']:.2f}, {r['split_half_p97_5']:.2f}] | LOO {r['loo_mae']:6.2f} | bias-only LOO {r['bias_only_loo_mae']:6.2f}")
        for k, t in g["tests"].items(): logger.info(f"  {k}: p = {t['p']:.3e} (lower: {t['lower_mae']})")
    if "across_groups" in out:
        for m, r in out["across_groups"]["models"].items():
            logger.info(f"  ACROSS GROUPS {m:11s} raw {r['raw_mae_mean']:.2f} ± {r['raw_mae_sd']:.2f} | LOO {r['loo_mae_mean']:.2f} ± {r['loo_mae_sd']:.2f}")
        for k, t in out["across_groups"]["tests_on_group_averaged_loo_errors"].items(): logger.info(f"  {k}: p = {t['p']:.3e} (lower: {t['lower_mae']})")
    logger.info(f"Wrote {od / 'recalibration_report.json'}")


if __name__ == "__main__":
    main()
