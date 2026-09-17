# Few-shot adaptation recipes

Code-level documentation for the few-shot suite. The research context
(motivation, dataset choice, statistical methodology) is in the Methods and
Supplementary Information of the SonoBase paper.

For the runner scripts and on-disk layout see `src/scripts/few_shot/README.md`.

---

## Module map

```
src/nemo_cv/
├── recipes/few_shot/
│   ├── finetune.py                     Stage 0 — single-GPU decoder-only fine-tune
│   ├── aggregate.py                    Stage 2.1 — per-dataset summary CSVs
│   ├── apply_fdr_correction.py         Stage 2.2 — BH-FDR across datasets
│   ├── build_figures.py                Stage 2.3 — publication figures
│   └── README.md
└── components/few_shot/
    └── splits.py                       deterministic per-(dataset, N, seed) subset
                                        selection + on-disk materialization
```

That's the entire codebase: **~1 800 LOC** of new code. Everything else
is reused from the existing pretrain / analysis stack:
  * `SAM2Train.forward()` for the forward pass.
  * `MultiStepMultiMasksAndIous` for the loss (the same one pretrain uses).
  * `SaUSRawDataset` / `JSONRawDataset` + `collate_fn` for data loading,
    with a custom `file_list_txt` per (dataset, N, seed).
  * `RandomHorizontalFlip` + `RandomAffine` (matched-pair on image+mask) —
    the few-shot augmentations are just a milder configuration of
    the existing pretrain transforms (degrees=15, scale=[0.9, 1.1]).
  * `SaUS_Annotation/38_pt_8_bm_7_ext/<DATASET>/train_list.txt` as the
    training pool for `select_subset`.
  * `nemo_cv.recipes.analysis.save_predictions` for evaluation —
    invoked by `scripts/few_shot/_eval_one.sh` with the fine-tuned
    `.pt` as input.

---

## `splits.py` — deterministic subset selection

**The same subset must be used for all 3 models at a given (dataset, N,
seed).** We satisfy this by:

  * Materializing one subset file per (dataset, N, seed) — model-agnostic
    name `<DATASET>_N<N>_seed<S>.txt`.
  * Recipe sets `data.file_list_txt` to that same path for all 3 models;
    `SaUSRawDataset` / `JSONRawDataset` reads it as the training pool.
  * Sibling `.json` records pool size, source path, and the resolved
    sample ids — drop into the supplementary materials for reviewer
    reproducibility.

`select_subset(train_ids, N, seed)` is a 5-line wrapper around
`random.Random(seed).sample(...)`, sorted to make the on-disk file
byte-deterministic.

---

## `finetune.py` — Stage 0 recipe

Single-GPU. The recipe class is `FewShotFinetuneRecipe`.

### Setup

  1. Init a 1-rank gloo process group (so the existing pretrain
     `DistributedSampler` doesn't crash on a single GPU).
  2. Materialize the seeded subset → on-disk `.txt` consumed by the
     dataset machinery.
  3. Build the dataset via `hydra.utils.instantiate(data.train_dataset)`
     (the existing `TorchTrainMixedDataset` pipeline).
  4. Build the model via the existing `build_model()`. Load the base
     checkpoint with `strict=False` (consistent with how the analysis
     `save_predictions.py` loads).
  5. **Freeze the encoder**: every `image_encoder.*` parameter gets
     `requires_grad=False`. The encoder's BatchNorm / dropout still
     runs (we need the features) but no gradients accumulate.
  6. AdamW(lr=1e-4, wd=0.01) over the trainable params + cosine LR to 0
     over `epochs × iters_per_epoch` steps.
  7. Determine epoch count from N via `epoch_schedule` (50/30/20 for
     N≤5/N∈{10,20}/N≥30).

### Training loop

Mirrors pretrain's `_forward_backward_step` minus DDP / step_scheduler /
checkpointer:

```python
with autocast():
    outputs = self.model(batch)         # SAM2Train.forward()
    loss    = self.loss_fn[batch.dict_key](outputs, batch.masks)
    if isinstance(loss, dict):
        loss = loss[CORE_LOSS_KEY]
loss.backward()
clip_grad_norm_(trainable, 1.0)
optimizer.step()
lr_scheduler.step()
```

### Checkpoint format

Final state_dict saved at `final.pth` in Meta-SAM2's
`{"model": <state_dict>, "_provenance": {...}}` shape. This is exactly
what the existing `save_predictions.py` recipe expects, so eval is a
single CLI invocation away — no conversion script.

### Resume

If `final.pth` already exists, the recipe short-circuits and exits. The
seed + same subset txt make re-runs unnecessary (full bit-determinism is
not promised because cudnn_benchmark=true; for that, set
`cuda.cudnn_deterministic=true` and accept the speed cost).

---

## `aggregate.py` — Stage 2.1

For one dataset, walks every (model × N × seed × prompt) Stage-1
prediction dir under `experiments/few_shot/predictions/`, then writes:

  * `<DATASET>_raw.csv` — long-format, one row per (model, N, seed,
    prompt, image, frame, obj). Includes per-image IoU, Dice, and
    (ACOUSLIC only) per-video predicted AC mm + abs error mm.
  * `<DATASET>_summary.csv` — wide-format, one row per (model, N, prompt)
    with mean ± std across the 3 seeds. Adds two paired-Wilcoxon raw
    p-value columns (`raw_p_vs_medsam2`, `raw_p_vs_sam2_no_ft`) for
    SonoBase rows; the four FDR columns (`fdr_q_*`, `sig_*`) are left
    empty until step 2.2.

Paired test definition: per-image IoU averaged across seeds → one
scalar per test image per model → paired Wilcoxon on the joined
`(SonoBase, baseline)` per-image vector.

---

## `apply_fdr_correction.py` — Stage 2.2

Standalone. Reads the 3 dataset summaries, **pools every secondary
raw p-value into a SINGLE family**, runs BH-FDR at q=0.05, writes the
q-values and sig flags back into each CSV. The pre-specified primary
endpoint is excluded from the family and tagged `sig=PRIMARY` with raw
p preserved.

Also writes `combined_summary.csv` (all 3 datasets concatenated) for the
figure-builder and the paper supplement.

---

## `build_figures.py` — Stage 2.3

Reads `combined_summary.csv`. Renders:

  1. `fig_fewshot_segmentation.{pdf,png}` — 3 cols (datasets) × 2 rows
     (point / box prompts) of mIoU vs N. Log-scale x with ticks at
     0/1/2/5/10/20/30 (the N grid), ±1 std bands across the 3
     seeds, model colours / markers / linestyles from
     `nemo_cv.components.analysis.plots.style`.
  2. `fig_fewshot_acouslic_clinical.{pdf,png}` — 1 × 2 of AC MAE (mm)
     vs N. Lower is better. Includes the GT-fit floor as a horizontal
     dashed line (default 7.39 mm).

PDF + PNG (300 DPI), Arial font ≥ 8 pt, all per the journal figure
requirements via `plots/style.apply_nm_defaults()`.

---

## Design notes & gotchas

  * **Single-process distributed group is mandatory**. The existing
    `TorchTrainMixedDataset.get_loader` always wraps with
    `DistributedSampler`, which calls `dist.get_world_size()` even on
    single-GPU. The recipe inits a 1-rank gloo group at setup. Same
    trick `convert_sonobase_dcp_to_pt.py` uses.
  * **`SAM2Train` is the right wrapper**, not `SAM2VideoPredictor` or
    `SAM2Base`. We need the training-time prompt-sampling logic during
    fine-tuning (matches how the base sonobase weights were trained).
  * **`unexpected=['no_mem_pos_enc']` on load is expected**. Same as
    everywhere else — Meta's released checkpoints carry this parameter
    that we conditionally drop in `SAM2Base`.
  * **Checkpoint size is large** (~700–800 MB per fine-tuned model
    because we save the full state_dict, not just the trainable diff).
    162 runs × ~750 MB ≈ 120 GB total. If disk becomes a problem,
    swap in a sparse-diff save format and add a sidecar that records
    the base ckpt path; the eval recipe then merges at load time.
  * **N=0 rows**: `aggregate.py` accepts `--zero-shot-csv` to populate
    the N=0 row from a previously-computed zero-shot result. Without
    that flag, N=0 cells are empty and the figures will simply omit
    the N=0 anchor point.
  * **ACOUSLIC AC values use the MHA-header 0.28 mm/px** and the Zenodo
    v1.1 circumference file (the v1.0 file is 2x too large and is rejected
    by the loader). `--pixel-spacing-mm` is for sensitivity checks only —
    never use it to fit predictions to a ground-truth file.

---

## Adding a new analysis component

Same conventions as `nemo_cv.recipes.analysis`:
  * Pure CPU helpers under `nemo_cv.components/`.
  * Single-GPU recipes under `nemo_cv.recipes/few_shot/`.
  * Hydra config under `src/configs/few_shot/`.
  * Shell wrappers under `src/scripts/few_shot/`.
