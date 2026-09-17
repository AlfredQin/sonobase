# Analysis runner scripts

Shell wrappers around the full analysis pipeline. The pipeline is a
**two-stage split**:

  * **Stage 1 — predict**: a single-GPU pass over a held-out test split
    that saves per-sample mask PNGs + metadata to disk. Owned by
    `nemo_cv.recipes.analysis.save_predictions.SavePredictionsRecipe`.
  * **Stage 2 — analyse**: CPU-only passes over the Stage-1 outputs that
    compute clinical metrics, run statistical tests, and render
    publication figures. One module per analysis under
    `nemo_cv.recipes.analysis.<analysis_id>.py`.

```
src/scripts/analysis/
├── save_predictions/        Stage 1 — one shell script per (model × dataset × prompt) run, plus a generic helper
└── run_analyses/            Stage 2 — one shell script per (analysis × dataset × prompt)
```

---

## Quickstart — full pipeline (point prompt)

From `src/`:

```bash
# Stage 1 — produce predictions for every (model × dataset × prompt) needed
bash scripts/analysis/save_predictions/save_all_HC18.sh            # 3 models × {point, box}
bash scripts/analysis/save_predictions/save_all_CAMUS.sh           # 3 models × {point, box}
bash scripts/analysis/save_predictions/save_all_ACOUSLIC.sh
bash scripts/analysis/save_predictions/save_all_RegPro.sh
bash scripts/analysis/save_predictions/save_all_PorcineSpinalCord.sh
bash scripts/analysis/save_predictions/save_a3_iterations_HC18_point.sh   # for A3 click efficiency
bash scripts/analysis/save_predictions/save_a3_iterations_HC18_box.sh

# Stage 2 — run every analysis (skips ones whose Stage-1 inputs are missing)
bash scripts/analysis/run_analyses/run_all_analyses_point.sh
```

Each Stage-2 runner script writes its outputs to
`src/experiments/analysis/<analysis_id>/<scope>/`. The
`compare_models.sh` aggregator scans those and produces
`src/experiments/analysis/_cross_analysis/cross_analysis_summary.csv`
covering every analysis × model.

---

## Stage 1 — `save_predictions/`

| Script | What it does |
|---|---|
| `_save_one.sh <model> <dataset> <prompt>` | Generic single-(model × dataset × prompt) Stage-1 invocation. Used by every wrapper below. Resolves the right `.pt` ckpt per model and lazily converts the SonoBase DCP. |
| `save_all_HC18.sh` | 3 models × HC18 × {point, box}. |
| `save_all_CAMUS.sh` | 3 models × CAMUS × {point, box}. |
| `save_all_ACOUSLIC.sh` | 3 models × ACOUSLIC × {point, box}. |
| `save_all_RegPro.sh` | 3 models × RegPro × {point, box}. |
| `save_all_PorcineSpinalCord.sh` | 3 models × Porcine × {point, box}. |
| `save_a3_iterations_HC18_{point,box}.sh` | 3 models × HC18 × prompt × iterations [0,1,3,5,7] — required by A3. |

### Sonobase: checkpoint resolution (`SONOBASE_DCP` required)

All three models resolve their checkpoint through the shared
`scripts/_resolve_ckpt.sh`. For `sonobase` you **must** name a pretrain
checkpoint — there is no default, so a run can never silently fall back
to a smoke / stale checkpoint:

```bash
# the best-validation epoch (LOWEST_VAL), a DCP dir, or any symlink:
export SONOBASE_DCP=./experiments/sonobase/pretrain/<run>/hiera_b_conv_s_conv_t/LOWEST_VAL
# ...or, for any model, an already-converted .pt:
export CKPT_PATH=/path/to/checkpoint.pt
```

A `sonobase` DCP directory is converted to a sibling `.pt` via
`convert_sonobase_dcp_to_pt` once and cached, keyed on the resolved epoch
dir (`epoch_N_step_M.pt`) — a new pretrain run reconverts automatically.

> **Heads-up**: the smoke-test SonoBase checkpoint (the `busi_camus` run)
> was trained on breast + cardiac only — CAMUS is in-distribution, HC18 /
> ACOUSLIC / RegPro / Porcine are OOD and will score poorly. Point
> `SONOBASE_DCP` at the production `38_pt_8_bm` pretrain for paper numbers.

### Outputs (per Stage-1 run)

```
src/experiments/analysis/predictions/<run_name>/
├── manifest.json                                run-level metadata
├── per_sample_metrics.csv                       canonical join table — every Stage-2 analysis reads this
└── <DATASET>/<sample_id>/
    ├── meta.json                                per-sample metadata + records
    ├── prompts.json                             actual prompts issued
    ├── pred_<frame>_obj_<obj>.png               predicted mask (255 = fg)
    ├── gt_<frame>_obj_<obj>.png                 GT mask (mirrors pred)
    └── overlay_<frame>_obj_<obj>.png            (if `save_overlays=true` — default for image, off for video)
```

A3 click-efficiency runs additionally suffix `_iter<N>` on the predicted
mask filenames so multiple iterations coexist in the same sample dir.
The `iteration` column in `per_sample_metrics.csv` is null for non-A3 runs.

### Resume

A sample is "done" iff its `meta.json` exists. Re-running Stage 1 on the
same `output_dir` skips done samples and re-emits the per-sample CSV from
the existing meta files (verified: `done=0 skipped_done=N elapsed=0s`).

---

## Stage 2 — `run_analyses/`

Every Stage-2 script sources `_paths.sh` for its defaults; override any
of those exported variables in your shell to point at a non-standard
prediction layout.

| Script | Analysis | Stage-1 inputs needed |
|---|---|---|
| `a1_camus_ef_point.sh` / `a1_camus_ef_box.sh` | A1 | 3 × CAMUS × point/box |
| `a2_hc18_hc_point.sh` / `a2_hc18_hc_box.sh` | A2 | 3 × HC18 × point/box |
| `a3_click_efficiency.sh` | A3 | 3 × HC18 × {point, box} × iterations [0,1,3,5,7] |
| `a4_acouslic_ac_point.sh` / `a4_acouslic_ac_box.sh` | A4 | 3 × ACOUSLIC × point/box |
| `b1_camus_per_structure.sh` | B1 | re-uses A1's CAMUS Stage-1 |
| `b2_failure_catalog.sh` | B2 | 3 × HC18 × point (extend to other datasets in the script) |
| `b3_porcine_spinal.sh` | B3 | 3 × PorcineSpinalCord × point |
| `c2_regpro_volume.sh` | C2 | 3 × RegPro × point |
| `t1_1_significance.sh` | T1.1 | A1 + A2 + A4 + C2 per-sample CSVs (whichever exist) |
| `t1_2_bland_altman.sh` | T1.2 | A1 + A2 + A4 + C2 per-sample CSVs (whichever exist) |
| `t2_1_hc_ga_point.sh` | T2.1 | re-uses A2's HC18 Stage-1 |
| `t2_2_ef_gray_zone.sh` | T2.2 | A1's `per_patient.csv` |
| `t2_3_fgr_screening.sh` | T2.3 | 3 × ACOUSLIC × point + ACOUSLIC GA metadata (CONDITIONAL — auto-skips with a stub if GA absent) |
| `t3_1_temporal_consistency.sh` | T3.1 | re-uses A1's CAMUS Stage-1 |
| `t3_2_sonobase_failures.sh` | T3.2 | 3 × HC18 × point (extend in the script) |
| `s1_kidneyus_manufacturer.sh` | S1 | 6 × KidneyUS (3 models × 2 prompts) |
| `s2_camus_quality.sh`         | S2 | 6 × CAMUS (3 models × 2 prompts) — re-uses A1's CAMUS Stage-1 if present |
| `s3_busi_pathology.sh`        | S3 | 6 × BUSI (3 models × 2 prompts) |
| `compare_models.sh` | — | scans `experiments/analysis/` for any `analysis_report.json` |
| `run_all_analyses_point.sh` | — | wrapper that runs every analysis above in dependency order |

### What each analysis writes

```
src/experiments/analysis/<id>/<scope>/
├── analysis_report.json            machine-readable per-model metrics (consistent shape across analyses)
├── per_sample.csv | per_patient.csv | per_sequence.csv
├── summary_table.{pdf,png}
└── <id>-specific plots             (scatter / bland_altman / convergence / failure pages)
```

### Reference numbers (HC18 pilot, current smoke run)

| Analysis | Model | n | Primary metric |
|---|---|---|---|
| **A2** (HC) | GT (ellipse fit) | 201 | MAE = 1.37 ± 0.58 mm — validates ellipse-fit + Ramanujan against the analytical floor |
| **A2** (HC) | SonoBase (smoke) | 201 | MAE = 16.5 ± 17.6 mm |
| **T2.1** (GA) | SonoBase (smoke) | 201 | GA MAE = 10.55 ± 13.60 days, ≤7 days = 54.2 % |
| **T1.1** (Wilcoxon) | sonobase vs medsam2 (A2) | 201 | raw p = 8.6e-13, BH q = 8.6e-13, **sig** |

The GT-fit baseline reproducing the ~1.4 mm theoretical floor is the
strongest single validation we have that the measurement / stats /
plotting pipeline is correctly wired.

---

## Adding a new analysis

For each new (analysis, dataset, prompt protocol):

  1. Write `src/nemo_cv/recipes/analysis/<analysis_id>.py` modelled on
     `a2_hc18_hc.py` or `t2_1_hc_ga.py`.
  2. Add a runner under `run_analyses/<analysis_id>.sh` that sources
     `_paths.sh` and invokes the recipe.
  3. If a new dataset is involved, add the Hydra `data/<NAME>.yaml` and a
     metadata loader under `nemo_cv.components.analysis.metadata_loaders.`.
  4. Add an extractor branch in
     `nemo_cv.recipes.analysis.compare_models._EXTRACTORS` so the new
     analysis shows up in the cross-analysis CSV.
