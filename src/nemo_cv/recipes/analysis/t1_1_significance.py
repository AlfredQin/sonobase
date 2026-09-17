"""Analysis T1.1 — Significance tests with BH-FDR correction.

Stage-2 analysis. Reads multiple per-sample CSVs from the
clinical-measurement analyses (A1, A2, A4, C2) and runs:

  1. Paired Wilcoxon signed-rank tests for SonoBase vs each baseline
     (SAM2 no-ft, MedSAM2) on the per-sample absolute error.
  2. Cohen's κ at the 40 % and 35 % EF thresholds (CAMUS / A1 only).
  3. Bootstrap 95 % CIs on every reported MAE (n_bootstrap=10 000, seed=42).

All p-values are pooled into a SINGLE family and corrected with
Benjamini-Hochberg at q=0.05. The per-row table reports
both raw p and BH-corrected q with a yes/no significance flag.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import pathlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from nemo_cv.components.analysis.plots.style import display_name
from nemo_cv.components.analysis import record_io
from nemo_cv.components.analysis.plots.summary_table import summary_table_figure
from nemo_cv.components.analysis.stats.tests import (
    bh_fdr_correct, bootstrap_ci_mean, cohens_kappa, paired_wilcoxon,
)

logger = logging.getLogger(__name__)


HFREF_THRESHOLD = 40.0
ICD_THRESHOLD = 35.0


# -- Per-analysis CSV shapes (read once into a uniform "long" table) -------


@dataclass
class _Spec:
    """One per-analysis input. Wide CSVs from A1/A2/A4/C2."""
    analysis_id: str             # "A1" / "A2" / "A4" / "C2"
    metric_name: str             # "EF" / "HC" / "AC" / "Volume"
    unit: str                    # "%" / "mm" / "mL"
    csv_path: pathlib.Path
    pred_col_prefix: str         # "pred_ef_biplane__" / "pred_hc_mm__" / "pred_ac_mm__" / "pred_vol_ml__"
    abs_err_col_prefix: str      # "abs_err_pct__" / "abs_err_mm__" / etc.
    gt_col: str                  # "gt_ef" / "gt_hc_mm" / "gt_ac_mm" / "gt_vol_ml"
    sample_col: str              # "patient_id" or "sample_id"
    ef_threshold_kappa: bool = False   # only A1 has the EF reclass kappa


# Mapping from analysis_id to its CSV schema. Easy to extend.
_ANALYSIS_SPECS: Dict[str, Dict] = {
    "A1": dict(metric_name="EF", unit="%",
               pred_col_prefix="pred_ef_biplane__",
               abs_err_col_prefix="abs_err_pct__",
               gt_col="gt_ef", sample_col="patient_id",
               ef_threshold_kappa=True),
    "A2": dict(metric_name="HC", unit="mm",
               pred_col_prefix="pred_hc_mm__",
               abs_err_col_prefix="abs_err_mm__",
               gt_col="gt_hc_mm", sample_col="sample_id",
               ef_threshold_kappa=False),
    "A4": dict(metric_name="AC", unit="mm",
               pred_col_prefix="pred_ac_mm__",
               abs_err_col_prefix="abs_err_mm__",
               gt_col="gt_ac_mm", sample_col="sample_id",
               ef_threshold_kappa=False),
    "C2": dict(metric_name="Volume", unit="mL",
               pred_col_prefix="pred_vol_ml__",
               abs_err_col_prefix="abs_err_ml__",
               gt_col="gt_vol_ml", sample_col="sample_id",
               ef_threshold_kappa=False),
}


def _read_per_sample(csv_path: pathlib.Path) -> List[Dict[str, str]]:
    with csv_path.open() as f:
        return list(csv.DictReader(f))


def _models_in_csv(rows: List[Dict[str, str]], pred_prefix: str) -> List[str]:
    if not rows:
        return []
    return [k.split("__", 1)[1] for k in rows[0].keys() if k.startswith(pred_prefix)]


def _per_sample_abs_err(rows: List[Dict[str, str]], model: str,
                        abs_err_prefix: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    sample_col = "sample_id" if "sample_id" in rows[0] else "patient_id"
    for r in rows:
        sid = r[sample_col]
        v = r.get(f"{abs_err_prefix}{model}", "")
        if v == "":
            continue
        try:
            out[sid] = float(v)
        except ValueError:
            continue
    return out


def _per_sample_signed_pred(rows: List[Dict[str, str]], model: str,
                            pred_prefix: str, gt_col: str
                            ) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Return (gt, pred) dicts indexed by sample_id."""
    gt: Dict[str, float] = {}
    pred: Dict[str, float] = {}
    sample_col = "sample_id" if "sample_id" in rows[0] else "patient_id"
    for r in rows:
        sid = r[sample_col]
        try:
            g = float(r[gt_col]); p = float(r[f"{pred_prefix}{model}"])
        except (KeyError, ValueError):
            continue
        if not math.isfinite(p):
            continue
        gt[sid] = g; pred[sid] = p
    return gt, pred


def _aligned(a: Dict[str, float], b: Dict[str, float]
             ) -> Tuple[np.ndarray, np.ndarray]:
    keys = sorted(set(a.keys()) & set(b.keys()))
    return (np.array([a[k] for k in keys], dtype=float),
            np.array([b[k] for k in keys], dtype=float))


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    p = argparse.ArgumentParser(description="Significance tests with BH-FDR (T1.1).")
    p.add_argument("--inputs", nargs="+", required=True,
                   help="`<analysis_id>=<per_sample.csv>` per analysis (A1/A2/A4/C2).")
    p.add_argument("--subgroup-reports", nargs="*", default=[],
                   help="Optional `<analysis_id>=<analysis_report.json>` per subgroup analysis "
                        "(S1/S2/S3). Their pre-computed per-(subgroup × baseline × prompt) "
                        "Wilcoxon p-values are folded into the SAME global BH-FDR family. "
                        "If absent, the subgroup analyses' own raw p-values stand uncorrected.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--baseline-models", nargs="+",
                   default=["sam2_no_ft", "medsam2"],
                   help="Models to compare SonoBase against.")
    p.add_argument("--target-model", default="sonobase",
                   help="The model on the LEFT side of every paired test "
                        "(typically 'sonobase').")
    p.add_argument("--alpha", type=float, default=0.05)
    args = p.parse_args()

    out_dir = pathlib.Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs: Dict[str, _Spec] = {}
    for s in args.inputs:
        if "=" not in s:
            raise ValueError(f"--inputs entry must be `<id>=<csv>`, got: {s!r}")
        aid, path = s.split("=", 1)
        aid = aid.strip().upper()
        if aid not in _ANALYSIS_SPECS:
            raise ValueError(f"Unknown analysis id {aid!r}; expected one of {list(_ANALYSIS_SPECS)}")
        meta = _ANALYSIS_SPECS[aid]
        inputs[aid] = _Spec(
            analysis_id=aid,
            metric_name=meta["metric_name"], unit=meta["unit"],
            csv_path=pathlib.Path(path.strip()),
            pred_col_prefix=meta["pred_col_prefix"],
            abs_err_col_prefix=meta["abs_err_col_prefix"],
            gt_col=meta["gt_col"], sample_col=meta["sample_col"],
            ef_threshold_kappa=meta["ef_threshold_kappa"],
        )

    test_rows: List[Dict] = []          # collected for BH-FDR; one per paired test
    bootstrap_rows: List[Dict] = []     # one per (analysis, model) MAE bootstrap
    kappa_rows: List[Dict] = []         # only A1 produces these

    # ---- Fold pre-computed subgroup Wilcoxon p-values into the same family ----
    # The S1/S2/S3 recipes already write paired Wilcoxon raw p-values inside
    # their analysis_report.json (per (subgroup × baseline × prompt)). We
    # ingest those here so the global BH-FDR correction sees ONE family
    # spanning every secondary test in the project.
    for s in args.subgroup_reports:
        if "=" not in s:
            raise ValueError(f"--subgroup-reports entry must be `<id>=<json>`, got: {s!r}")
        aid, path = s.split("=", 1)
        aid = aid.strip().upper()
        json_path = pathlib.Path(path.strip())
        if not json_path.is_file():
            logger.warning(f"Subgroup report not found: {json_path}; skipping {aid}.")
            continue
        with json_path.open() as f:
            report = json.load(f)
        wcx = report.get("wilcoxon", {})
        for sg_name, by_key in wcx.items():
            for combo, payload in by_key.items():
                # combo looks like "<baseline>__<prompt>"
                if "__" not in combo:
                    continue
                baseline, prompt = combo.split("__", 1)
                raw_p = payload.get("raw_p")
                stat = payload.get("statistic")
                n = int(payload.get("n_paired", 0))
                if raw_p is None or not math.isfinite(raw_p) or n < 2:
                    continue
                test_rows.append({
                    "analysis": aid,
                    "metric": f"{report.get('subgroup_axis', 'subgroup')}={sg_name} ({prompt}) IoU",
                    "comparison": f"{args.target_model} vs {baseline}",
                    "n": n,
                    "raw_p": float(raw_p),
                    "wilcoxon_W": float(stat) if (stat is not None and math.isfinite(stat)) else float("nan"),
                })
        logger.info(f"Folded {len(wcx)} subgroup Wilcoxon entries from {aid} ({json_path.name})")

    for aid, spec in inputs.items():
        rows = _read_per_sample(spec.csv_path)
        models = _models_in_csv(rows, spec.pred_col_prefix)
        logger.info(f"{aid}: {len(rows)} rows, models = {models}")
        if not rows:
            continue

        # Bootstrap CIs on per-model MAE
        for m in models:
            err_dict = _per_sample_abs_err(rows, m, spec.abs_err_col_prefix)
            err_arr = np.fromiter(err_dict.values(), dtype=float)
            if err_arr.size == 0:
                continue
            mean, lo, hi = bootstrap_ci_mean(err_arr, n_bootstrap=10_000, seed=42)
            bootstrap_rows.append({
                "analysis": aid, "metric": spec.metric_name, "unit": spec.unit,
                "model": m, "n": int(err_arr.size),
                "mae": mean, "ci_lo": lo, "ci_hi": hi,
            })

        # Wilcoxon for target_model vs each baseline (paired by sample_id)
        target_err = _per_sample_abs_err(rows, args.target_model, spec.abs_err_col_prefix)
        for baseline in args.baseline_models:
            base_err = _per_sample_abs_err(rows, baseline, spec.abs_err_col_prefix)
            x, y = _aligned(target_err, base_err)
            if x.size < 2:
                logger.warning(f"{aid}: not enough paired samples for {args.target_model} vs {baseline} "
                               f"(n={x.size}); skipping Wilcoxon.")
                continue
            stat, pval = paired_wilcoxon(x, y)
            test_rows.append({
                "analysis": aid, "metric": f"{spec.metric_name} abs error ({spec.unit})",
                "comparison": f"{args.target_model} vs {baseline}",
                "n": int(x.size),
                "raw_p": pval,
                "wilcoxon_W": stat,
            })

        # Cohen's κ — only A1 (EF thresholds)
        if spec.ef_threshold_kappa:
            for thresh, name in ((HFREF_THRESHOLD, "EF≤40"),
                                 (ICD_THRESHOLD, "EF≤35")):
                for m in models:
                    gt_dict, pred_dict = _per_sample_signed_pred(
                        rows, m, spec.pred_col_prefix, spec.gt_col)
                    g, p = _aligned(gt_dict, pred_dict)
                    if g.size < 2:
                        continue
                    gt_cls = (g <= thresh).astype(int)
                    pr_cls = (p <= thresh).astype(int)
                    k = cohens_kappa(gt_cls, pr_cls)
                    kappa_rows.append({
                        "analysis": aid, "model": m, "threshold_name": name,
                        "threshold_value": thresh,
                        "n": int(g.size), "kappa": k,
                    })

    # Apply BH-FDR to ALL Wilcoxon p-values as a single family
    raw_ps = [r["raw_p"] for r in test_rows]
    rejected, qs = bh_fdr_correct(raw_ps, alpha=args.alpha)
    for r, ok, q in zip(test_rows, rejected, qs):
        r["bh_q"] = q
        r["sig"] = bool(ok)

    # Render the headline table
    headers = ["Analysis", "Metric", "Comparison", "n",
               "Raw p", f"BH q (q<{args.alpha})", "Sig"]
    table_rows = [headers]
    for r in test_rows:
        table_rows.append([
            r["analysis"], r["metric"], r["comparison"], f"{r['n']:d}",
            f"{r['raw_p']:.2e}", f"{r['bh_q']:.2e}",
            "yes" if r["sig"] else "no",
        ])
    summary_table_figure(table_rows, out_dir / "wilcoxon_table.pdf",
                         title=f"Paired Wilcoxon (BH-FDR @ q={args.alpha}) — T1.1")

    # Bootstrap CIs table
    bs_headers = ["Analysis", "Metric", "Model", "n", "MAE", "95% CI"]
    bs_rows = [bs_headers]
    for b in bootstrap_rows:
        bs_rows.append([
            b["analysis"], f"{b['metric']} ({b['unit']})",
            display_name(b["model"]),
            f"{b['n']:d}",
            f"{b['mae']:.2f}",
            f"[{b['ci_lo']:.2f}, {b['ci_hi']:.2f}]",
        ])
    summary_table_figure(bs_rows, out_dir / "bootstrap_ci_table.pdf",
                         title="MAE 95% CIs (bootstrap, n=10000) — T1.1")

    # Kappa table (A1 only)
    if kappa_rows:
        k_headers = ["Analysis", "Model", "Threshold", "n", "κ"]
        k_table = [k_headers]
        for k in kappa_rows:
            k_table.append([
                k["analysis"], display_name(k["model"]), k["threshold_name"],
                f"{k['n']:d}", f"{k['kappa']:.3f}",
            ])
        summary_table_figure(k_table, out_dir / "kappa_table.pdf",
                             title="Cohen's κ at clinical EF thresholds — T1.1")

    # Long-format CSV for downstream re-correction / paper supplement table
    csv_path = out_dir / "wilcoxon_results.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["analysis", "metric", "comparison", "n",
                           "raw_p", "bh_q", "sig", "wilcoxon_W"],
        )
        w.writeheader()
        for r in test_rows:
            w.writerow(r)
    logger.info(f"Wrote {csv_path}")

    # Every MAE in `bootstrap_ci` is owned by the analysis it names, not by
    # T1.1 — so point a checker at the exact CSV and column each one came from.
    # Without this the block is unfalsifiable from inside this directory, which
    # is how its A2/A4 entries once held values from a different quantity
    # entirely (HC18 17.710 against the 2.49 that A2 reports) with nothing to
    # flag it.
    externals = [
        record_io.external(
            ["bootstrap_ci"],
            select={"analysis": aid, "model": "$model"},
            stat="mae",
            file=str(spec.csv_path.resolve()),
            column=spec.abs_err_col_prefix + "{model}",
        )
        for aid, spec in inputs.items()
    ]

    with (out_dir / "analysis_report.json").open("w") as f:
        json.dump({
            "analysis": "T1.1",
            "title": "Significance tests with BH-FDR",
            "spec_section": "T1.1",
            "alpha": args.alpha,
            "target_model": args.target_model,
            "baseline_models": args.baseline_models,
            "wilcoxon_tests": test_rows,
            "bootstrap_ci": bootstrap_rows,
            "kappa": kappa_rows,
            "verification": record_io.declare(None, [], externals=externals),
        }, f, indent=2)

    logger.info(f"T1.1 analysis complete. Outputs at: {out_dir}")


if __name__ == "__main__":
    main()
