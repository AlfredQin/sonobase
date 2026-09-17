"""Analysis A1 — CAMUS ejection fraction (Simpson's biplane).

Stage-2 analysis. The most complex of the clinical-measurement analyses.
The workflow is:

  1. For each patient (= 2 video samples: ``patient####_4CH`` + ``patient####_2CH``)
     read CAMUS metadata for ED-frame, ES-frame, GT EF, ImageQuality.
  2. For each (patient, view, phase ∈ {ED, ES}):
        a. Load the predicted LV-endocardium mask (object_id 0) at that frame.
        b. Compute Simpson's monoplane volume (px-volume units).
        c. CAMUS provides isotropic 0.5 mm/px after preprocessing; use that as
           the default pixel spacing. Override via ``--pixel-spacing-mm``.
  3. Biplane EF: ASE disc-pairing across the two views is the primary
     formula (Lang 2015); view-averaged volumes, EDV = (EDV_4CH + EDV_2CH) / 2
     with ESV likewise and EF = (EDV − ESV) / EDV × 100, are the comparator.
  4. Stratify by ImageQuality (Good / Medium / Poor) and report
     reclassification rates at the 40 % (HFrEF) and 35 % (ICD) thresholds.

The **first analysis run should be GT-only** (use the GT
masks themselves with this pipeline) so we can verify Simpson's-from-GT
correlates well with the clinical EF in the metadata (target r > 0.90).
Pass ``--include-gt-fit`` to add it as a synthetic model in the report.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import pathlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from nemo_cv.components.analysis import prediction_io as pio
from nemo_cv.components.analysis import record_io
from nemo_cv.components.analysis.measurements.simpsons_biplane import (
    biplane_ef, biplane_ef_ase, lv_volume_biplane_ase_mL,
    lv_volume_simpsons, SimpsonsResult,
)
from nemo_cv.components.analysis.metadata_loaders.camus import load_camus_metadata
from nemo_cv.components.analysis.metadata_loaders.default import ClinicalMeta
from nemo_cv.components.analysis.plots.bland_altman import bland_altman_panel
from nemo_cv.components.analysis.plots.scatter import scatter_with_identity, scatter_comparison
from nemo_cv.components.analysis.plots.style import (
    apply_nm_defaults, color_for, display_name,
)
from nemo_cv.components.analysis.plots.summary_table import summary_table_figure
from nemo_cv.components.analysis.stats.tests import (
    bland_altman_stats, cohens_kappa, pearson_r,
)

logger = logging.getLogger(__name__)


# CAMUS preprocessing standard: 0.5 mm/px isotropic after the SaUS conversion.
DEFAULT_PIXEL_SPACING_MM = 0.5

# Clinical thresholds
HFREF_THRESHOLD = 40.0      # EF ≤ 40 % = HFrEF
ICD_THRESHOLD = 35.0        # EF ≤ 35 % = ICD candidacy

# Object id = 0 in the SaUS CAMUS conversion is endocardium (verified from
# dataset_info.json: categories are [endocardium, epicardium, atrium_wall]).
LV_ENDO_OBJ_ID = 0

QUALITY_BINS = ("Good", "Medium", "Poor")


@dataclass
class _PatientResult:
    patient_id: str            # e.g. "patient0004"
    quality: Optional[str]     # worst of the 2 views
    gt_ef: Optional[float]
    pred_ef_4ch: Optional[float]
    pred_ef_2ch: Optional[float]
    pred_ef_biplane: Optional[float]        # view-averaged volumes (comparator)
    pred_ef_biplane_ase: Optional[float] = None  # ASE disc-pairing (primary)


@dataclass
class _ModelResult:
    model_label: str
    patients: List[_PatientResult]


def _load_pred_mask(run_dir: pathlib.Path, sample_id: str, frame_idx: int,
                    obj_id: int) -> Optional[np.ndarray]:
    """Load `pred_<frame:05d>_obj_<obj>.png` from the per-sample dir.

    Returns None if the file is missing — the patient is then dropped from
    the analysis with a warning.
    """
    p = (run_dir / sample_id /
         f"pred_{frame_idx:05d}_obj_{obj_id}.png")
    if not p.is_file():
        # Stage 1 may have written the GT/pred under <dataset>/<sid>/...
        p = (run_dir / "CAMUS" / sample_id /
             f"pred_{frame_idx:05d}_obj_{obj_id}.png")
        if not p.is_file():
            return None
    return np.array(Image.open(p).convert("L")) > 0


def _load_gt_mask_from_predrun(run_dir: pathlib.Path, sample_id: str, frame_idx: int,
                               obj_id: int) -> Optional[np.ndarray]:
    p = (run_dir / "CAMUS" / sample_id /
         f"gt_{frame_idx:05d}_obj_{obj_id}.png")
    if not p.is_file():
        return None
    return np.array(Image.open(p).convert("L")) > 0


def _compute_view_ef(
    run_dir: pathlib.Path,
    patient_id: str,
    view: str,
    meta_view: ClinicalMeta,
    pixel_spacing_mm: float,
    *,
    use_gt_masks: bool = False,
) -> Tuple[Optional[float], Optional[float], Optional[float],
           Optional[SimpsonsResult], Optional[SimpsonsResult]]:
    """Compute (EDV_mL, ESV_mL, EF_view_%, ed_simp, es_simp) for a single view.

    The two SimpsonsResult objects are returned so the caller can compute the
    ASE biplane disc-paired volume, which needs per-disc diameters from
    both views. Returns all-None if any required mask is missing or any
    Simpson's call fails (the patient analysis short-circuits to skip).
    """
    sample_id = f"{patient_id}_{view}"
    if meta_view.ed_frame_idx is None or meta_view.es_frame_idx is None:
        return None, None, None, None, None

    loader = _load_gt_mask_from_predrun if use_gt_masks else _load_pred_mask
    ed_mask = loader(run_dir, sample_id, meta_view.ed_frame_idx, LV_ENDO_OBJ_ID)
    es_mask = loader(run_dir, sample_id, meta_view.es_frame_idx, LV_ENDO_OBJ_ID)
    if ed_mask is None or es_mask is None:
        return None, None, None, None, None

    ed_simp = lv_volume_simpsons(ed_mask)
    es_simp = lv_volume_simpsons(es_mask)
    if ed_simp is None or es_simp is None:
        return None, None, None, None, None

    edv_ml = ed_simp.volume_mL(pixel_spacing_mm)
    esv_ml = es_simp.volume_mL(pixel_spacing_mm)
    if edv_ml <= 0:
        return edv_ml, esv_ml, None, ed_simp, es_simp
    ef = (edv_ml - esv_ml) / edv_ml * 100.0
    return edv_ml, esv_ml, ef, ed_simp, es_simp


def _worst_quality(qs: List[Optional[str]]) -> Optional[str]:
    """Worst (lowest) quality across a patient's views.

    Order: Good > Medium > Poor; missing → unknown.
    """
    rank = {"Good": 0, "Medium": 1, "Poor": 2}
    valid = [q for q in qs if q in rank]
    if not valid:
        return None
    return max(valid, key=rank.__getitem__)


def _patient_ids(metadata: Dict[str, ClinicalMeta]) -> List[str]:
    """Return the unique patient IDs (without _<view>) seen in metadata."""
    pids = set()
    for sid in metadata.keys():
        if "_" not in sid:
            continue
        pids.add(sid.rsplit("_", 1)[0])
    return sorted(pids)


def _load_run(
    run_dir: pathlib.Path,
    model_label: str,
    metadata: Dict[str, ClinicalMeta],
    pixel_spacing_mm: float,
    *,
    use_gt_masks: bool = False,
) -> _ModelResult:
    patients: List[_PatientResult] = []
    for pid in _patient_ids(metadata):
        meta_4ch = metadata.get(f"{pid}_4CH")
        meta_2ch = metadata.get(f"{pid}_2CH")
        if meta_4ch is None or meta_2ch is None:
            continue
        gt_ef = (meta_4ch.gt_clinical_value
                 if meta_4ch.gt_clinical_value is not None
                 else meta_2ch.gt_clinical_value)

        edv_4, esv_4, ef_4, ed_simp_4, es_simp_4 = _compute_view_ef(
            run_dir, pid, "4CH", meta_4ch, pixel_spacing_mm,
            use_gt_masks=use_gt_masks,
        )
        edv_2, esv_2, ef_2, ed_simp_2, es_simp_2 = _compute_view_ef(
            run_dir, pid, "2CH", meta_2ch, pixel_spacing_mm,
            use_gt_masks=use_gt_masks,
        )

        if edv_4 is None or edv_2 is None or esv_4 is None or esv_2 is None:
            ef_bp = None
            ef_bp_ase = None
        else:
            # Comparator: view-averaged monoplane volumes.
            ef_bp = biplane_ef(edv_4, esv_4, edv_2, esv_2)
            # ASE disc-pairing (primary): pair per-disc diameters across views.
            ef_bp_ase = None
            if None not in (ed_simp_4, es_simp_4, ed_simp_2, es_simp_2):
                edv_ase = lv_volume_biplane_ase_mL(ed_simp_4, ed_simp_2, pixel_spacing_mm)
                esv_ase = lv_volume_biplane_ase_mL(es_simp_4, es_simp_2, pixel_spacing_mm)
                if edv_ase is not None and esv_ase is not None:
                    ef_bp_ase = biplane_ef_ase(edv_ase, esv_ase)

        patients.append(_PatientResult(
            patient_id=pid,
            quality=_worst_quality([meta_4ch.image_quality, meta_2ch.image_quality]),
            gt_ef=float(gt_ef) if gt_ef is not None else None,
            pred_ef_4ch=ef_4, pred_ef_2ch=ef_2,
            pred_ef_biplane=ef_bp,
            pred_ef_biplane_ase=ef_bp_ase,
        ))
    return _ModelResult(model_label, patients)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _arrays_for(model: _ModelResult, quality: Optional[str] = None,
                *, ase: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Paired (GT, pred) EF arrays. ase=True selects the ASE disc-paired EF
    (primary); ase=False the view-averaged EF (comparator)."""
    g, p = [], []
    for pat in model.patients:
        pred = pat.pred_ef_biplane_ase if ase else pat.pred_ef_biplane
        if pat.gt_ef is None or pred is None:
            continue
        if not math.isfinite(pred):
            continue
        if quality is not None and pat.quality != quality:
            continue
        g.append(pat.gt_ef)
        p.append(pred)
    return np.asarray(g, dtype=float), np.asarray(p, dtype=float)


def _summarize(model: _ModelResult, quality: Optional[str] = None,
               *, ase: bool = False) -> Dict[str, float]:
    g, p = _arrays_for(model, quality, ase=ase)
    if g.size == 0:
        return {"n": 0}
    err = np.abs(p - g)
    ba = bland_altman_stats(g, p)
    gt_hfref = (g <= HFREF_THRESHOLD).astype(int)
    pr_hfref = (p <= HFREF_THRESHOLD).astype(int)
    gt_icd = (g <= ICD_THRESHOLD).astype(int)
    pr_icd = (p <= ICD_THRESHOLD).astype(int)
    return {
        "n": int(g.size),
        "mae_pct": float(err.mean()),
        "std_pct": float(err.std(ddof=1)) if g.size > 1 else 0.0,
        "pearson_r": pearson_r(g, p),
        "bias_pct": ba.bias,
        "loa_lower_pct": ba.loa_lower,
        "loa_upper_pct": ba.loa_upper,
        "reclass_at_40_pct": float((gt_hfref != pr_hfref).mean() * 100),
        "reclass_at_35_pct": float((gt_icd != pr_icd).mean() * 100),
        "kappa_at_40": cohens_kappa(gt_hfref, pr_hfref) if g.size >= 2 else float("nan"),
        "kappa_at_35": cohens_kappa(gt_icd, pr_icd) if g.size >= 2 else float("nan"),
    }


def _write_per_patient_csv(out_dir: pathlib.Path,
                           results: List[_ModelResult]) -> None:
    pids = sorted({pat.patient_id for r in results for pat in r.patients})
    headers = ["patient_id", "image_quality", "gt_ef"]
    for r in results:
        headers += [
            f"pred_ef_4ch__{r.model_label}",
            f"pred_ef_2ch__{r.model_label}",
            f"pred_ef_biplane__{r.model_label}",          # view-averaged (comparator)
            f"pred_ef_biplane_ase__{r.model_label}",      # ASE disc-pairing (primary)
            f"abs_err_pct__{r.model_label}",              # err of the ASE EF vs GT
        ]
    by_pid: Dict[str, Dict[str, _PatientResult]] = defaultdict(dict)
    quality_lookup: Dict[str, Optional[str]] = {}
    gt_lookup: Dict[str, Optional[float]] = {}
    for r in results:
        for pat in r.patients:
            by_pid[pat.patient_id][r.model_label] = pat
            quality_lookup.setdefault(pat.patient_id, pat.quality)
            if pat.gt_ef is not None:
                gt_lookup.setdefault(pat.patient_id, pat.gt_ef)

    path = out_dir / "per_patient.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for pid in pids:
            gt = gt_lookup.get(pid)
            row = [pid, quality_lookup.get(pid) or "", "" if gt is None else f"{gt:.4f}"]
            for r in results:
                pat = by_pid[pid].get(r.model_label)
                if pat is None:
                    row += ["", "", "", "", ""]
                    continue
                row += [
                    "" if pat.pred_ef_4ch is None else f"{pat.pred_ef_4ch:.4f}",
                    "" if pat.pred_ef_2ch is None else f"{pat.pred_ef_2ch:.4f}",
                    "" if pat.pred_ef_biplane is None else f"{pat.pred_ef_biplane:.4f}",
                    "" if pat.pred_ef_biplane_ase is None else f"{pat.pred_ef_biplane_ase:.4f}",
                    ("" if (pat.pred_ef_biplane_ase is None or gt is None
                            or not math.isfinite(pat.pred_ef_biplane_ase))
                     else f"{abs(pat.pred_ef_biplane_ase - gt):.4f}"),
                ]
            w.writerow(row)
    logger.info(f"Wrote {path}")


def _render_table(out_dir: pathlib.Path, all_summaries: Dict[str, Dict],
                  quality_summaries: Dict[str, Dict[str, Dict]]) -> None:
    headers = ["Model", "n", "MAE (%)", "r", "Bias (%)", "LoA (%)",
               "Recl@40 (%)", "κ@40", "Recl@35 (%)", "κ@35"]
    rows = [headers]
    for label, s in all_summaries.items():
        if s["n"] == 0:
            rows.append([display_name(label), "0", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a"])
            continue
        loa = f"[{s['loa_lower_pct']:.1f}, {s['loa_upper_pct']:.1f}]"
        rows.append([
            display_name(label), f"{s['n']:d}",
            f"{s['mae_pct']:.2f} ± {s['std_pct']:.2f}",
            f"{s['pearson_r']:.3f}",
            f"{s['bias_pct']:+.2f}", loa,
            f"{s['reclass_at_40_pct']:.1f}",
            f"{s['kappa_at_40']:.3f}",
            f"{s['reclass_at_35_pct']:.1f}",
            f"{s['kappa_at_35']:.3f}",
        ])
    summary_table_figure(rows, out_dir / "summary_table.pdf",
                         title="CAMUS Ejection Fraction (overall) — A1 summary")

    for q in QUALITY_BINS:
        rows = [headers]
        for label, qmap in quality_summaries.items():
            s = qmap.get(q, {"n": 0})
            if s["n"] == 0:
                rows.append([display_name(label), "0"] + ["n/a"] * 8)
                continue
            loa = f"[{s['loa_lower_pct']:.1f}, {s['loa_upper_pct']:.1f}]"
            rows.append([
                display_name(label), f"{s['n']:d}",
                f"{s['mae_pct']:.2f} ± {s['std_pct']:.2f}",
                f"{s['pearson_r']:.3f}",
                f"{s['bias_pct']:+.2f}", loa,
                f"{s['reclass_at_40_pct']:.1f}",
                f"{s['kappa_at_40']:.3f}",
                f"{s['reclass_at_35_pct']:.1f}",
                f"{s['kappa_at_35']:.3f}",
            ])
        summary_table_figure(rows, out_dir / f"summary_table_{q.lower()}.pdf",
                             title=f"CAMUS EF — quality={q} subset")


def _render_plots(out_dir: pathlib.Path, results: List[_ModelResult],
                  *, ase: bool = True) -> None:
    # ase=True (default) renders the ASE disc-paired EF that the report treats as
    # PRIMARY, so the correlation / Bland-Altman / scatter
    # figures match `models` in analysis_report.json rather than the view-averaged
    # comparator (`models_viewavg`).
    apply_nm_defaults()
    panels = []
    for r in results:
        g, p = _arrays_for(r, ase=ase)
        if g.size == 0:
            continue
        color = color_for(r.model_label); label = display_name(r.model_label)

        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        scatter_with_identity(ax, g, p,
                              title=f"{label} — EF predicted vs GT",
                              xlabel="Ground-truth EF (%)",
                              ylabel="Predicted EF (%)",
                              point_color=color)
        out = out_dir / f"correlation_{r.model_label}.pdf"
        fig.tight_layout(); fig.savefig(out); fig.savefig(out.with_suffix(".png"))
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5.0, 4.0))
        bland_altman_panel(ax, g, p,
                           title=f"{label} — EF Bland-Altman",
                           ylabel="EF difference (pred − GT, %)",
                           point_color=color)
        out = out_dir / f"bland_altman_{r.model_label}.pdf"
        fig.tight_layout(); fig.savefig(out); fig.savefig(out.with_suffix(".png"))
        plt.close(fig)

        # Per-quality Bland-Altman panels
        for q in QUALITY_BINS:
            gq, pq = _arrays_for(r, quality=q, ase=ase)
            if gq.size == 0:
                continue
            fig, ax = plt.subplots(figsize=(5.0, 4.0))
            bland_altman_panel(ax, gq, pq,
                               title=f"{label} — EF Bland-Altman ({q})",
                               ylabel="EF diff (pred − GT, %)",
                               point_color=color)
            out = out_dir / f"bland_altman_{r.model_label}_{q.lower()}.pdf"
            fig.tight_layout(); fig.savefig(out); fig.savefig(out.with_suffix(".png"))
            plt.close(fig)

        panels.append((label, g, p, color))

    if panels:
        scatter_comparison(panels, xlabel="Ground-truth EF (%)", ylabel="Predicted EF (%)",
                           output_path=out_dir / "scatter_comparison.pdf",
                           n_cols=min(3, len(panels)))


def _render_validation_plot(out_dir: pathlib.Path, gt_only_result: _ModelResult,
                            *, ase: bool = True) -> None:
    """Compare Simpson's-from-GT against the clinical EF in CAMUS metadata.

    ase=True (default) validates the ASE disc-paired EF (primary) against the metadata clinical EF, matching the report's primary formula.
    """
    apply_nm_defaults()
    gt_clinical, simpson_from_gt = [], []
    for pat in gt_only_result.patients:
        pred = pat.pred_ef_biplane_ase if ase else pat.pred_ef_biplane
        if pat.gt_ef is not None and pred is not None and math.isfinite(pred):
            gt_clinical.append(pat.gt_ef)
            simpson_from_gt.append(pred)
    g = np.asarray(gt_clinical); p = np.asarray(simpson_from_gt)
    if g.size == 0:
        logger.warning("No data for validation plot")
        return
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    scatter_with_identity(ax, g, p,
                          title="Simpson's-from-GT vs clinical EF (validation)",
                          xlabel="Clinical EF (%)",
                          ylabel="Simpson's biplane EF from GT masks (%)",
                          point_color="#2ca02c")
    out = out_dir / "validation_simpsons_vs_clinical.pdf"
    fig.tight_layout(); fig.savefig(out); fig.savefig(out.with_suffix(".png"))
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    p = argparse.ArgumentParser(description="CAMUS ejection fraction (A1).")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--output-dir", required=True)
    _camus_default = os.environ.get("CAMUS_ZIP")
    p.add_argument("--camus-zip", default=_camus_default, required=(_camus_default is None),
                   help="Path to CAMUS raw zip (default: $CAMUS_ZIP env var; required if unset).")
    p.add_argument("--pixel-spacing-mm", type=float, default=DEFAULT_PIXEL_SPACING_MM)
    p.add_argument("--include-gt-fit", action="store_true",
                   help="Add the 'GT (Simpson's biplane)' model that runs the same "
                        "pipeline on the GT masks themselves — validates the long-axis "
                        "detector + Simpson's discs.")
    args = p.parse_args()

    out_dir = pathlib.Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for s in args.runs:
        if "=" not in s:
            raise ValueError(f"--runs entry must be `label=path`, got: {s!r}")
        label, path = s.split("=", 1)
        runs.append((label.strip(), pathlib.Path(path.strip())))

    metadata = load_camus_metadata(args.camus_zip)
    logger.info(f"CAMUS metadata: {len(metadata)} patient-views")

    results: List[_ModelResult] = []
    if args.include_gt_fit and runs:
        results.append(_load_run(runs[0][1], "GT (Simpson's biplane)",
                                 metadata, args.pixel_spacing_mm, use_gt_masks=True))
    for label, rd in runs:
        results.append(_load_run(rd, label, metadata, args.pixel_spacing_mm))

    # Report BOTH biplane formulas. ASE disc-pairing is the PRIMARY (clinical
    # standard, Lang 2015); the view-averaged formula is carried as a
    # sensitivity-analysis comparator.
    overall: Dict[str, Dict] = {}                 # PRIMARY = ASE
    by_quality: Dict[str, Dict[str, Dict]] = {}
    overall_viewavg: Dict[str, Dict] = {}         # comparator = view-averaged
    for r in results:
        overall[r.model_label] = _summarize(r, ase=True)
        by_quality[r.model_label] = {q: _summarize(r, quality=q, ase=True) for q in QUALITY_BINS}
        overall_viewavg[r.model_label] = _summarize(r, ase=False)
        s = overall[r.model_label]; sv = overall_viewavg[r.model_label]
        if s["n"]:
            logger.info(
                f"[{r.model_label}] ASE: n={s['n']} MAE={s['mae_pct']:.2f}±{s['std_pct']:.2f}% "
                f"r={s['pearson_r']:.3f} bias={s['bias_pct']:+.2f}% "
                f"reclass@40={s['reclass_at_40_pct']:.1f}% κ@40={s['kappa_at_40']:.3f}"
            )
        if sv.get("n"):
            logger.info(
                f"[{r.model_label}] view-avg (comparator): n={sv['n']} "
                f"MAE={sv['mae_pct']:.2f}±{sv['std_pct']:.2f}% r={sv['pearson_r']:.3f} "
                f"bias={sv['bias_pct']:+.2f}%"
            )

    with (out_dir / "analysis_report.json").open("w") as f:
        json.dump({
            "analysis": "A1",
            "title": "CAMUS Ejection Fraction (Simpson's biplane)",
            "spec_section": "A1",
            "biplane_formula": {
                "primary": "ASE disc-pairing (Lang RM et al., JASE 2015) — `models`/`by_quality`",
                "comparator": "view-averaged volumes — `models_viewavg` (sensitivity check)",
            },
            "inputs": {label: str(rd) for label, rd in runs},
            "pixel_spacing_mm": args.pixel_spacing_mm,
            "thresholds": {"hfref": HFREF_THRESHOLD, "icd": ICD_THRESHOLD},
            "models": overall,
            "by_quality": by_quality,
            "models_viewavg": overall_viewavg,
            # `models_viewavg` is deliberately absent from the checks: the CSV
            # carries only the ASE absolute error, so the view-averaged MAEs
            # cannot be recomputed from this dump. A checker reports them as
            # uncovered rather than silently treating them as verified.
            "verification": record_io.declare("per_patient.csv", [
                record_io.check(["models", "$model", "mae_pct"],
                                "abs_err_pct__{model}"),
                record_io.check(["models", "$model", "std_pct"],
                                "abs_err_pct__{model}", agg="std"),
                record_io.check(["models", "$model", "n"],
                                "abs_err_pct__{model}", agg="count"),
                record_io.check(["by_quality", "$model", "$image_quality", "mae_pct"],
                                "abs_err_pct__{model}"),
                record_io.check(["by_quality", "$model", "$image_quality", "n"],
                                "abs_err_pct__{model}", agg="count"),
            ]),
        }, f, indent=2)

    _write_per_patient_csv(out_dir, results)
    _render_table(out_dir, overall, by_quality)
    _render_plots(out_dir, results, ase=True)          # figures = ASE primary
    if args.include_gt_fit and results:
        _render_validation_plot(out_dir, results[0], ase=True)

    logger.info(f"A1 analysis complete. Outputs at: {out_dir}")


if __name__ == "__main__":
    main()
