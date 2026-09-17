# Few-shot adaptation runner scripts

Shell wrappers around the few-shot pipeline. The pipeline is **three stages**:

  * **Stage 0 — fine-tune** (single-GPU). Decoder-only training of one
    model on a deterministic seeded subset of N training samples.
    Owned by `nemo_cv.recipes.few_shot.finetune.FewShotFinetuneRecipe`.
  * **Stage 1 — predict** (single-GPU). Reuses the existing analysis
    `save_predictions.py` recipe with the fine-tuned `.pt` as input
    checkpoint. No new code — just CLI overrides via `_eval_one.sh`.
  * **Stage 2 — aggregate** (CPU). Reads per-(model × N × seed × prompt)
    Stage-1 outputs, computes mean ± std across seeds, runs paired
    Wilcoxon vs each baseline, applies BH-FDR globally across all 3
    datasets, and renders the publication figures.

```
src/scripts/few_shot/
├── _finetune_one.sh             generic helper: <dataset> <model> <N> <seed>
├── _eval_one.sh                 generic helper: <dataset> <model> <N> <seed> <prompt>
├── run_dataset.sh               full sweep for one dataset (54 fine-tunes + 108 evals)
├── run_n1_n5_acouslic.sh        early checkpoint: ACOUSLIC N=1 + N=5 (18 + 36)
└── aggregate_all.sh             Stage-2 aggregator + FDR + figures
```

---

## Quickstart — full pipeline

From `src/`:

```bash
# Day 1: ship N=1 + N=5 ACOUSLIC numbers as soon as they exist
bash scripts/few_shot/run_n1_n5_acouslic.sh

# Day 1 (rest), Day 2, Day 3: full sweeps
bash scripts/few_shot/run_dataset.sh ACOUSLIC
bash scripts/few_shot/run_dataset.sh DDTI
bash scripts/few_shot/run_dataset.sh FUGC

# Stage 2 — aggregation, FDR, figures
bash scripts/few_shot/aggregate_all.sh
```

Each `run_dataset.sh` invocation runs **54 fine-tunes + 108 evals** for one
dataset. Total full-sweep runtime is bounded by 162 fine-tunes (~few
minutes each on a single GPU) plus 324 evaluations (~few seconds to a few
minutes per eval depending on dataset size). Roughly **6–10 hours total**
on one GPU.

Outputs land under `src/experiments/few_shot/`:

```
src/experiments/few_shot/
├── few_shot_splits/
│   ├── ACOUSLIC_N1_seed42.txt        consumed by JSONRawDataset.file_list_txt
│   ├── ACOUSLIC_N1_seed42.json       reproducibility record (pool size, source path, etc.)
│   └── ...
├── checkpoints/
│   └── <DATASET>_<MODEL>_N<N>_seed<S>/
│       ├── final.pth                 fine-tuned model state_dict (Meta-SAM2 .pt schema)
│       └── train_log.jsonl           per-step loss + LR + elapsed time
├── predictions/
│   └── <DATASET>_<MODEL>_N<N>_seed<S>_<prompt>_0corr/
│       ├── manifest.json
│       ├── per_sample_metrics.csv    canonical analysis-tree schema
│       └── <DATASET>/<sample_id>/    per-sample mask PNGs + meta.json
└── few_shot_results/
    ├── <DATASET>_raw.csv             per-(model × N × seed × prompt × image)
    ├── <DATASET>_summary.csv         mean ± std + paired Wilcoxon raw_p + (post-FDR) q + sig
    ├── combined_summary.csv          all 3 datasets in one CSV
    ├── fig_fewshot_segmentation.{pdf,png}
    └── fig_fewshot_acouslic_clinical.{pdf,png}
```

---

## Stage 0 — `_finetune_one.sh`

```bash
bash _finetune_one.sh <DATASET> <MODEL> <N> <SEED> [extra hydra overrides]
```

Resolves the per-model base checkpoint via the shared
`scripts/_resolve_ckpt.sh`:
  * `sam2_no_ft` → `$CHECKPOINT_DIR/SAM2/sam2.1_hiera_base_plus.pt`
  * `medsam2`    → `$CHECKPOINT_DIR/MedSAM2/MedSAM2_latest.pt`
  * `sonobase`   → **requires** `SONOBASE_DCP` (the best-validation epoch
                   `LOWEST_VAL`, a DCP dir, or any symlink) or `CKPT_PATH`
                   (a converted `.pt`). No default — a run can never
                   silently fall back to a smoke / stale checkpoint. A DCP
                   dir is converted to `.pt` lazily (idempotent):

```bash
export SONOBASE_DCP=./experiments/sonobase/pretrain/<run>/hiera_b_conv_s_conv_t/LOWEST_VAL
```

Key behaviour:
  * **Same subset across all 3 models** at a given (DATASET, N, SEED).
    Materialized to `few_shot_splits/<DATASET>_N<N>_seed<S>.txt`. The
    file is model-AGNOSTIC by design.
  * **Resume**: if `final.pth` already exists, skips the run. Re-running
    the same command costs ~1 second.
  * **Decoder-only**: every `image_encoder.*` parameter has
    `requires_grad=False`; only `mask_decoder` + `prompt_encoder` update.
  * **Reuses existing pretrain stack**: `SAM2Train.forward()` +
    `MultiStepMultiMasksAndIous` + the same data transforms.

### Smoke-test reference numbers

`bash _finetune_one.sh ACOUSLIC sonobase 1 42` (verified end-to-end):
  * Fine-tune: 50 epochs × 1 step = 83 s; final.pth = 739 MB; train log
    in `train_log.jsonl`.
  * Eval: `bash _eval_one.sh ACOUSLIC sonobase 1 42 point` → 270 videos,
    5906 annotated frames, ~14 min, mean per-frame mIoU = **0.492**.

---

## Stage 1 — `_eval_one.sh`

```bash
bash _eval_one.sh <DATASET> <MODEL> <N> <SEED> <PROMPT>
```

Calls `nemo_cv.recipes.analysis.save_predictions` with:
  * `data=<DATASET>`              — analysis-tree data config
                                    (uses canonical project test split)
  * `experiment=<MODEL>`          — model overlay
                                    (sonobase / medsam2 / sam2_no_ft)
  * `ckpt_path=<final.pth>`       — the fine-tuned checkpoint
  * `prompt_protocol.type=<...>`  — point or box

The Stage-1 outputs land at
`experiments/few_shot/predictions/<DATASET>_<MODEL>_N<N>_seed<S>_<prompt>_0corr/`
which is exactly the directory layout `aggregate.py` expects.

> **Test split sources:** ACOUSLIC / DDTI / FUGC analysis data configs
> point at `${scratch.annotation_dir}/<DATASET>/test_list.txt` (the
> canonical project test list under `SaUS_Annotation/38_pt_8_bm_7_ext/`),
> NOT the raw `SaUS/<DATASET>/test_list.txt`. Sample counts: ACOUSLIC=270,
> DDTI=566, FUGC=396.

---

## Stage 2 — `aggregate_all.sh`

Three CPU-only steps invoked in order:

  1. `aggregate.py --dataset <DATASET>` — for each of 3 datasets, walk
     every Stage-1 prediction dir, build the per-(image × model × N × seed
     × prompt) raw CSV, then the mean ± std summary with paired Wilcoxon
     raw p-values for SonoBase vs each baseline.
  2. `apply_fdr_correction.py --results-dir ...` — pool every secondary
     raw p-value across the 3 dataset summaries into a SINGLE family,
     apply BH-FDR at q=0.05, write `fdr_q_*` and `sig_*` columns back into
     each CSV. The pre-specified primary endpoint (ACOUSLIC × box × N=5 ×
     SonoBase vs MedSAM2) is excluded and tagged `sig_vs_medsam2=PRIMARY`
     with the raw p preserved. Also produces `combined_summary.csv`.
  3. `build_figures.py` — renders the publication figures from
     `combined_summary.csv`:
       * `fig_fewshot_segmentation.{pdf,png}` — 3 cols × 2 rows of mIoU
         vs N curves (log-scale x, ±1 std bands, model colours from
         `plots/style.py`).
       * `fig_fewshot_acouslic_clinical.{pdf,png}` — 1 × 2 of AC MAE
         (mm) vs N with the GT-fit floor as a horizontal dashed line
         (default 7.39 mm; pass `--gt-fit-floor-mm` to override).

---

## Adding a new dataset / model

For a new dataset (e.g. BUSI-WHU):

  1. Add `src/configs/few_shot/data/<NAME>.yaml` — copy `DDTI.yaml` for
     image datasets or `ACOUSLIC.yaml` for video datasets.
  2. Add `src/configs/analysis/data/<NAME>.yaml` mirroring the same shape
     so `_eval_one.sh` can find the test split.
  3. Update `_finetune_one.sh` / `_eval_one.sh` only if a new model
     introduces a new image encoder; otherwise no script change needed.
  4. Update `aggregate.py`'s `DATASETS` tuple and re-run `aggregate_all.sh`.
