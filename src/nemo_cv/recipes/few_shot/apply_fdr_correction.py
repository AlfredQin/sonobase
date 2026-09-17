"""Standalone Benjamini-Hochberg FDR correction across the few-shot p-values.

Protocol:

  * Pool ALL secondary-test raw p-values across the 3 dataset summary CSVs
    into a SINGLE family.
  * Apply BH at q=0.05.
  * Write the q-values back into each CSV's `fdr_q_*` columns; populate
    `sig_*` based on q < 0.05.
  * The PRE-SPECIFIED PRIMARY ENDPOINT is excluded from the correction
    family and tagged `sig_vs_medsam2 = "PRIMARY"` with the raw p-value
    preserved unchanged.

Primary endpoint: ACOUSLIC × box × N=5 × SonoBase vs MedSAM2 × AC MAE.

Usage:
    python -m nemo_cv.recipes.few_shot.apply_fdr_correction \\
      --results-dir ./experiments/few_shot/few_shot_results
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import pathlib
from typing import Dict, List, Tuple

from nemo_cv.components.analysis.stats.tests import bh_fdr_correct

logger = logging.getLogger(__name__)


DATASETS = ("ACOUSLIC", "DDTI", "FUGC")
BASELINE_COLUMNS = ("vs_medsam2", "vs_sam2_no_ft")

PRIMARY_ENDPOINT = {
    "dataset": "ACOUSLIC", "prompt": "box", "N": "5",
    "model": "sonobase", "comparison": "vs_medsam2",
    # The ACOUSLIC primary endpoint is AC-MAE: aggregate.py runs the paired
    # Wilcoxon for ACOUSLIC on per-video AC error (|pred_AC - GT_AC|) — the
    # same numbers shown in the AC_MAE_mean column — so the p-value at this
    # row IS the AC-MAE primary endpoint test. (The DDTI / FUGC primary
    # endpoint is per-image IoU.)
}


def _load_csv(path: pathlib.Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
        headers = list(rows[0].keys()) if rows else []
    return headers, rows


def _is_primary(row: Dict[str, str], baseline: str) -> bool:
    return (
        row.get("dataset") == PRIMARY_ENDPOINT["dataset"]
        and row.get("prompt") == PRIMARY_ENDPOINT["prompt"]
        and str(row.get("N")) == PRIMARY_ENDPOINT["N"]
        and row.get("model") == PRIMARY_ENDPOINT["model"]
        and baseline == PRIMARY_ENDPOINT["comparison"]
    )


def _collect(results_dir: pathlib.Path) -> Tuple[Dict[pathlib.Path, Tuple[List[str], List[Dict]]], List[Tuple[pathlib.Path, int, str, float]]]:
    """Return ({csv_path: (headers, rows)}, [(csv_path, row_idx, baseline, raw_p), ...]).

    The p-value list contains only SECONDARY tests (primary excluded).
    """
    csv_state: Dict[pathlib.Path, Tuple[List[str], List[Dict]]] = {}
    secondaries: List[Tuple[pathlib.Path, int, str, float]] = []

    for ds in DATASETS:
        path = results_dir / f"{ds}_summary.csv"
        if not path.is_file():
            logger.warning(f"Missing summary CSV: {path} — skipping.")
            continue
        headers, rows = _load_csv(path)
        csv_state[path] = (headers, rows)

        for idx, row in enumerate(rows):
            for baseline in BASELINE_COLUMNS:
                col = f"raw_p_{baseline}"
                raw = row.get(col, "")
                if not raw:
                    continue
                try:
                    pval = float(raw)
                except ValueError:
                    continue
                if not math.isfinite(pval):
                    continue
                if _is_primary(row, baseline):
                    # Don't add to FDR family. Mark as PRIMARY at write time.
                    continue
                secondaries.append((path, idx, baseline, pval))

    return csv_state, secondaries


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    p = argparse.ArgumentParser(description="BH-FDR correction across few-shot summaries.")
    p.add_argument("--results-dir", required=True,
                   help="Directory containing <DATASET>_summary.csv files.")
    p.add_argument("--alpha", type=float, default=0.05)
    args = p.parse_args()

    results_dir = pathlib.Path(args.results_dir).expanduser().resolve()
    csv_state, secondaries = _collect(results_dir)
    if not csv_state:
        logger.error(f"No summary CSVs found under {results_dir}.")
        return

    # ---- Apply BH-FDR to the SINGLE family of secondary p-values ----
    raw_ps = [t[3] for t in secondaries]
    rejected, qvals = bh_fdr_correct(raw_ps, alpha=args.alpha)
    n_sig = sum(rejected)
    logger.info(
        f"BH-FDR: {len(raw_ps)} secondary tests, {n_sig} significant at q<{args.alpha}"
    )

    # ---- Write the q-values back into each row, in place ----
    for (path, idx, baseline, _), q, ok in zip(secondaries, qvals, rejected):
        headers, rows = csv_state[path]
        rows[idx][f"fdr_q_{baseline}"] = f"{q:.6e}"
        rows[idx][f"sig_{baseline}"] = "yes" if ok else "no"

    # ---- Tag the primary endpoint cell ----
    n_primary_marked = 0
    for path, (headers, rows) in csv_state.items():
        for idx, row in enumerate(rows):
            for baseline in BASELINE_COLUMNS:
                if _is_primary(row, baseline):
                    raw = row.get(f"raw_p_{baseline}", "")
                    if raw:
                        row[f"fdr_q_{baseline}"] = ""
                        row[f"sig_{baseline}"] = "PRIMARY"
                        n_primary_marked += 1
    logger.info(f"Marked {n_primary_marked} primary-endpoint cell(s).")

    # ---- Re-write each summary CSV ----
    for path, (headers, rows) in csv_state.items():
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=headers)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        logger.info(f"Updated {path}")

    # ---- Build combined CSV ----
    combined = results_dir / "combined_summary.csv"
    with combined.open("w", newline="") as f:
        master_headers = None
        w = None
        for path, (headers, rows) in csv_state.items():
            if master_headers is None:
                master_headers = headers
                w = csv.DictWriter(f, fieldnames=master_headers)
                w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in master_headers})
    logger.info(f"Wrote {combined}")


if __name__ == "__main__":
    main()
