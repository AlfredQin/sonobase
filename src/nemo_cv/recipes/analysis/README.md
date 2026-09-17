# Analysis recipes

Code-level documentation for the **complete analysis suite**. The research
context (which clinical analyses, why each matters, statistical methodology)
is in the Methods and Supplementary Information of the SonoBase paper. This
README documents the
**code architecture**: what each module does, the shared dataflow between
Stage 1 and Stage 2, and how to add a new analysis.

---

## Two-stage architecture

```
                   GPU                                   CPU
┌────────────────────────────────┐    ┌────────────────────────────────┐
│ Stage 1: save_predictions.py   │    │ Stage 2: <analysis_id>.py      │
│ • One model × one dataset ×    │    │ • Reads Stage-1 outputs        │
│   one prompt protocol per run  │ →  │   (per_sample_metrics.csv +    │
│ • Single-GPU, no DDP           │    │   per-sample masks +           │
│ • Image AND video data paths   │    │   meta.json)                   │
│ • Iteration snapshots (A3)     │    │ • Computes clinical metrics,   │
│ • Saves per-sample masks +     │    │   stats, plots                 │
│   meta.json + per_sample.csv   │    │ • No model, no GPU             │
│ • Resumable                    │    │                                │
└────────────────────────────────┘    └────────────────────────────────┘
                  │                                     ▲
                  └─────────── on-disk artifacts ───────┘
```

---

## Module map

```
src/nemo_cv/
├── recipes/analysis/
│   ├── save_predictions.py          Stage 1 recipe
│   ├── compare_models.py            cross-analysis aggregator (master CSV)
│   ├── _failure_common.py           shared helpers for B2 + T3.2
│   ├── a1_camus_ef.py               Simpson's biplane EF on CAMUS
│   ├── a2_hc18_hc.py                HC18 head circumference (PILOT)
│   ├── a3_click_efficiency.py       convergence curves + clicks-to-80
│   ├── a4_acouslic_ac.py            ACOUSLIC abdominal circumference
│   ├── b1_camus_per_structure.py    per-structure mIoU/Dice
│   ├── b2_failure_catalog.py        catastrophic failure cases (SonoBase saves)
│   ├── b3_porcine_spinal.py         porcine spinal cord per-class
│   ├── c2_regpro_volume.py          RegPro prostate volume
│   ├── t1_1_significance.py         Wilcoxon + κ + bootstrap + BH-FDR
│   ├── t1_2_bland_altman_figures.py composite + main BA panels
│   ├── t1_3_recalibration.py     held-out linear recalibration (S9b protocol: split-half + LOO + bias-only, paired Wilcoxon on LOO errors); A1/A2/A4/C2 records or few-shot raw
│   ├── t2_1_hc_ga.py                HC → GA (Hadlock 1984)
│   ├── t2_2_ef_gray_zone.py         35–45 % gray-zone reclassification
│   ├── t2_3_fgr_screening.py        FGR sens/spec/PPV/NPV (conditional)
│   ├── t3_1_temporal_consistency.py CAMUS inter-frame IoU + breaks
│   └── t3_2_sonobase_failures.py    where SonoBase still fails
└── components/analysis/
    ├── prompts.py                   point/box derivation + correction clicks
    ├── prediction_io.py             on-disk schema; mask PNG I/O; manifest; per_sample_metrics.csv
    ├── measurements/
    │   ├── ellipse_fit.py           ellipse fit + Ramanujan perimeter (A2, A4)
    │   ├── simpsons_biplane.py      Long-axis detect + disc summation + biplane EF (A1)
    │   ├── volumetric.py            voxel sum × voxel-spacing → mL (C2)
    │   ├── ga_dating.py             Hadlock 1984 (T2.1)
    │   └── intergrowth_21st.py      AC 10th-percentile reference (T2.3)
    ├── stats/
    │   └── tests.py                 paired Wilcoxon, kappa, bootstrap CI, BH-FDR, Bland-Altman stats
    ├── plots/
    │   ├── style.py                 NM matplotlib defaults + per-model colour map
    │   ├── bland_altman.py          single-panel + composite BA
    │   ├── scatter.py               predicted-vs-GT scatter with identity + regression
    │   ├── summary_table.py         publication-grade table figure
    │   ├── convergence_curves.py    A3 convergence curves (single + grid)
    │   └── failure_catalog.py       side-by-side qualitative pages (B2, T3.2)
    └── metadata_loaders/
        ├── default.py               ClinicalMeta dataclass + no-op fallback
        ├── hc18.py                  per-image pixel spacing + GT HC from HC.zip
        ├── camus.py                 ED/ES/quality/EF from CAMUS.zip Info_*.cfg files
        ├── acouslic.py              per-video AC + (optional) GA from per-sweep CSV
        ├── regpro.py                per-case voxel spacing from NIfTI headers
        └── porcine.py               class names from dataset_info.json (no per-image meta)
```

The `components/analysis/` tree is **shared infrastructure** — every new
Stage-2 analysis should reuse it rather than duplicate logic.

---

## Stage 1 — `save_predictions.py`

Dispatches on `data.kind`:

  * `image` → `SAM2ImagePredictor.set_image()` + `predict()` per sample.
  * `video` → `SAM2VideoPredictor.init_state()` + `add_new_points_or_box()`
    on the first frame with GT + `propagate_in_video()`.

The model is built as `SAM2Train` (the released training-time wrapper);
for video runs the recipe class-swaps it to `SAM2VideoPredictor` after
load (both inherit from `SAM2Base` and share parameters — pure method-table
swap, no parameter copy).

**A3 click-efficiency mode** (image datasets only today): set
`prompt_protocol.iterations: [0, 1, 3, 5, 7]` and the recipe runs the
longest correction sequence once per object, snapshotting the predicted
mask + saving a CSV row at each milestone. Per-iter masks are named
`pred_<frame>_obj_<obj>_iter<N>.png` so they coexist in the same sample dir.

**Resumability**: a sample is "complete" iff its `meta.json` exists.
Re-runs skip done samples (re-reading their records back into the CSV).

---

## Stage 2 — analysis recipes

Every Stage-2 recipe follows the same pattern:

```python
def main():
    args = parser.parse_args()                # --runs <label>=<path> --output-dir <dir>
    metadata = load_<dataset>_metadata(...)   # per-dataset metadata loader
    results = [_load_run(rd, lbl, metadata) for lbl, rd in runs]
    summaries = {r.label: _summarize(r) for r in results}
    _write_per_sample_csv(out_dir, results)
    _write_report(out_dir, summaries)         # analysis_report.json (consistent shape!)
    _render_table(out_dir, summaries)         # summary_table.{pdf,png}
    _render_plots(out_dir, results)           # scatter / BA / convergence / etc.
```

`analysis_report.json` shape (consistent across analyses for the
cross-aggregator):

```json
{
  "analysis": "A2",
  "title": "HC18 head circumference measurement error",
  "spec_section": "A2",
  "inputs": {"sam2_no_ft": "...", "medsam2": "...", "sonobase": "..."},
  "models": {
    "sam2_no_ft": {"n": 201, "mae_mm": 83.3, ...},
    "medsam2":    {"n": 201, ...},
    "sonobase":   {"n": 201, ...}
  }
}
```

---

## What each analysis answers

| Recipe | Question | Headline number(s) |
|---|---|---|
| A1 | How accurately does each model predict EF from echo videos? | MAE ± SD; r; reclass @40/35; κ; per-quality breakdowns |
| A2 | How accurately does each model measure HC from fetal scans? | MAE ± SD; % within 3/5 mm; r; bias; LoA |
| A3 | How quickly does each model converge to 80 % mIoU with click corrections? | Clicks-to-80 → seconds-to-80 → time savings |
| A4 | How accurately does each model measure AC from sweep videos? | MAE ± SD; r; bias; LoA |
| B1 | Where in the cardiac structures does each model do well/badly? | per-structure (LV-endo / LV-epi / atrium) IoU+Dice |
| B2 | Where does SonoBase rescue cases that baselines fail catastrophically? | per-dataset rescue count; worst-K qualitative pages |
| B3 | How well does each model handle non-fetal/non-cardiac (porcine) structures? | per-class IoU+Dice |
| C2 | How accurately does each model estimate prostate volume from 3D US? | MAE ± SD (mL); r; bias; LoA |
| T1.1 | After multiple-comparisons correction, which model differences are real? | Raw p, BH-q, sig-yes/no per comparison; bootstrap CIs; κ |
| T1.2 | What do the publication BA figures look like? | 4 × 3 supplementary composite + 1 × 2 main panel |
| T2.1 | Translated to clinically meaningful units, how many days off is the GA estimate? | GA MAE (days), ≤3/7/>14 day percentages |
| T2.2 | In the borderline 35–45 % EF range where decisions are hardest, how does the model do? | n_gray, MAE, reclass @40, κ |
| T2.3 | Could the model be used for FGR screening? | Sens / spec / PPV / NPV / κ (CONDITIONAL on GA metadata) |
| T3.1 | Does the model lose track of structures across video frames? | mean inter-frame IoU; total breaks; sequences w/ 0 breaks |
| T3.2 | Where does SonoBase still fail? | worst-K per-dataset qualitative pages for manual categorisation |
| S1 | KidneyUS robustness across 5 scanner manufacturers | per-(manufacturer × model × prompt) mIoU/Dice + Wilcoxon |
| S2 | CAMUS robustness across 3 quality bins | per-(quality × model × prompt) mIoU/Dice + Wilcoxon |
| S3 | BUSI per-pathology segmentation accuracy | per-(pathology × model × prompt) mIoU/Dice + Wilcoxon |
| compare_models | All headline numbers in a single CSV | analysis × model → primary endpoint |

---

## Design notes & gotchas

  * **`SAM2Train` ↔ `SAM2VideoPredictor` class swap.** Both extend
    `SAM2Base` and register identical parameters. Stage 1 builds
    `SAM2Train` (because that's the released YAML target), then swaps
    `model.__class__` to `SAM2VideoPredictor` for video runs. Avoids the
    brittle alternative of trying to whitelist SAM2Train-only YAML kwargs.
  * **`unexpected=['no_mem_pos_enc']` on load.** Expected. We removed this
    parameter from `SAM2Base` for the `directly_add_no_mem_embed=True`
    branch (see `src/scripts/pretrain/README.md` Lessons Learned § 3) but
    Meta's released SAM2 / MedSAM2 checkpoints still include it.
  * **CAMUS GT masklet axis order** is `[N_frames][N_objects]` (frame-major),
    not `[N_objects][N_frames]`. Caught during the video Stage-1 smoke
    test — the recipe transposes correctly so `obj_id` is stable across
    frames (object 0 = endocardium, 1 = epicardium, 2 = atrium_wall).
  * **`prompts.json` content.** Initially saved empty placeholder records;
    fixed to capture the actual prompts (point coordinates, labels, box)
    used during inference.
  * **Atomic resume sentinel.** `meta.json` is written *last* in every
    per-sample loop (after all masks + prompts.json), so a kill mid-run
    leaves a partial dir without `meta.json` — re-run reprocesses it.
  * **CPU-only Stage 2.** Every Stage-2 recipe should run with no GPU
    visible. Verified: T2.1 runs in ~10 s on the existing A2 outputs.
  * **`statsmodels` + `scikit-learn` dependency.** Required by `stats/tests.py`
    (BH-FDR via `statsmodels.stats.multitest.multipletests`; Cohen's κ via
    `sklearn.metrics.cohen_kappa_score`). Install via `uv pip install
    statsmodels scikit-learn`.

---

## Adding a new Stage-2 analysis

  1. Write `src/nemo_cv/recipes/analysis/<analysis_id>.py` modelled on
     `a2_hc18_hc.py` (clinical-measurement) or `t2_1_hc_ga.py` (post-hoc
     wrapper) or `t1_1_significance.py` (multi-input statistics).
  2. Reuse the shared utilities under `components/analysis/` — extend
     them (rather than duplicate) when you need new measurements / stats /
     plots.
  3. Add a runner under `src/scripts/analysis/run_analyses/<id>.sh` that
     sources `_paths.sh` and invokes the recipe.
  4. Add an extractor branch in
     `nemo_cv.recipes.analysis.compare_models._EXTRACTORS` so the new
     analysis shows up in the cross-analysis CSV.

For a new dataset: add a Hydra `data/<NAME>.yaml`, a metadata loader
under `metadata_loaders/<name>.py`, and (if it's a new modality)
extend Stage 1's `_run_video()` / `_run_image()` if needed.
