# Benchmark Hydra config tree (`src/configs/test_sam2/`)

This directory holds the YAML configs that drive the **SAM2 / MedSAM2 /
Sonobase benchmark comparison**. The configs use [Hydra](https://hydra.cc/docs/intro/)
to compose a single hierarchical config from many small files at runtime,
and the layout deliberately mirrors `src/configs/pretrain/` so anyone
familiar with the pretrain tree can navigate this one immediately.

If you want to **run** the benchmark, see
[`src/scripts/benchmarks/test_sam2/README.md`](../../scripts/benchmarks/test_sam2/README.md).
If you want to **understand the recipe code**, see
[`src/nemo_cv/recipes/benchmarks/README.md`](../../nemo_cv/recipes/benchmarks/README.md).

This document is for anyone who wants to **add a model variant, change a
hyperparameter, or add a test dataset**.

---

## Directory layout

```
src/configs/test_sam2/
├── test.yaml                                ROOT config; selected by `-cn test` (single-iter)
├── iterative_eval.yaml                      ROOT config; selected by `-cn iterative_eval` (B2 standalone)
│
├── scratch/
│   └── default.yaml                         Project-wide variables (paths, batch sizes, transforms, prompt knobs)
│
├── model/
│   ├── default.yaml                         SAM2 hyperparameters (NOT architecture — knobs like num_maskmem, prompt probs)
│   ├── sam2.yaml                            Test-time SAM2Train _target_ + composes sub-components
│   ├── image_encoder/                       Image-encoder variants
│   │   ├── hiera_b+.yaml                        Standard SAM2-Base+ encoder (used by sam2-no-ft)
│   │   ├── hiera_t.yaml                         Hiera-Tiny (used by MedSAM2)
│   │   ├── hiera_s.yaml                         Hiera-Small
│   │   ├── hiera_l.yaml                         Hiera-Large
│   │   ├── hiera_b_conv_s_conv_t.yaml           TriBranchTrunk (used by sonobase)
│   │   └── hiera_l_conv_b_conv_s.yaml           TriBranchTrunk (Hiera-L variant — for future sonobase)
│   ├── memory_attention/
│   │   └── default.yaml
│   └── memory_encoder/
│       └── default.yaml
│
├── data/
│   ├── busi_camus.yaml                      Smoke-test data (BUSI + CAMUS test splits)
│   └── 8_bm_7_ext.yaml                      Full benchmark (8 benchmark + 7 external = 15 datasets)
│
├── loss/
│   └── sam2.yaml                            Optional during test; provides per-dataset test_loss
│
├── metric/
│   └── sam2_vos.yaml                        Sam2VOSMeanIoU + Sam2VOSDiceScore
│
└── experiment/                              Per-run overlays (model variant + dataset choice)
    ├── test_sam2_no_ft_on_busi_camus.yaml             ← smoke
    ├── test_sam2_no_ft_on_8_bm_7_ext.yaml             ← full benchmark
    ├── test_medsam2_on_busi_camus.yaml                ← smoke
    ├── test_medsam2_on_8_bm_7_ext.yaml                ← full benchmark
    ├── test_hiera_b_conv_s_conv_t_on_busi_camus.yaml  ← smoke
    └── test_hiera_b_conv_s_conv_t_on_8_bm_7_ext.yaml  ← full benchmark
```

---

## How Hydra composes the config

`test.yaml` is the root, with a `defaults:` list that pulls together the
sub-configs:

```yaml
defaults:
  - scratch: default              # → cfg.scratch.* from scratch/default.yaml
  - model@model: sam2             # → cfg.model.*  from model/sam2.yaml (recursively)
  - loss@loss_fn: sam2            # → cfg.loss_fn.* from loss/sam2.yaml
  - metric@_global_: sam2_vos     # → merged AT TOP LEVEL (sets cfg.val_metrics.*)
  - data: 8_bm_7_ext              # → cfg.data.*   (default; overridden by experiments)
  - experiment: null              # ← intentionally null; overlay is opt-in
  - _self_                        # this file's own keys win on conflict
```

The Hydra packaging idioms (`<group>@<package>`, `_self_` ordering, the
`@_global_` directive, the `# @package _global_` header inside experiment
files) work exactly as in the pretrain tree. See the
[pretrain config README](../pretrain/README.md#how-hydra-composes-the-config)
for a complete walkthrough. The benchmark tree is structurally identical
modulo:

- `model/sam2.yaml` instead of `model/sam2_train.yaml` (test-time variant
  with two extra knobs from scratch — `num_correction_pt_per_frame_val` and
  `prob_to_use_box_input_for_eval`)
- No `optim/` group (not needed for inference)
- No `step_scheduler` block in `test.yaml` (no epoch loop)
- `recipe: nemo_cv.recipes.benchmarks.test_sam2.TestSam2BenchmarkRecipe`
- A top-level `ckpt_path: ???` field that **must** be overridden at the CLI

### `iterative_eval.yaml` — alternative root for B2 sweeps

A second root config, `iterative_eval.yaml`, sits next to `test.yaml`
and is selected by `-cn iterative_eval`. It composes the same defaults
list but adds two things:

```yaml
recipe: nemo_cv.recipes.benchmarks.iterative_eval.IterativeEvalRecipe
iterations: [0, 1, 3, 5, 7]
model:
  forward_backbone_per_frame_for_eval: false
```

The recipe runs the image encoder once per batch and loops the prompt
preparation + tracking head over the full `iterations` list, which is
why `forward_backbone_per_frame_for_eval` must be flipped to `false`
(otherwise the encoder would still be re-run per frame inside
`forward_tracking`, defeating the caching). All other groups
(`scratch`, `model.image_encoder`, `data`, `loss`, `metric`,
`experiment`) compose identically to `test.yaml`, so any existing
experiment overlay in `experiment/` works with both root configs.

The B2 path is currently a verification tool — production iterative
sweeps still loop `test.yaml` per iteration count via the helpers under
`src/scripts/benchmarks/test_sam2/iterative/` and merge the per-iter
JSONs with `aggregate_iterations.py`. Once B1 ≈ B2 has been validated
empirically on the BUSI + CAMUS smoke set, this hierarchy can be
revisited.

### Composition flow for one benchmark run

```
$ bash scripts/benchmarks/test_sam2/test_sam2_no_ft_on_busi_camus.sh

  → uv run torchrun ... -m nemo_cv.recipes.benchmarks.test_sam2
        -c ./configs/test_sam2 -cn test
        experiment=test_sam2_no_ft_on_busi_camus
        ckpt_path=${CHECKPOINT_DIR}/SAM2/sam2.1_hiera_base_plus.pt

Hydra reads test.yaml's defaults list and composes:

  1. scratch/default.yaml                    → cfg.scratch.*
  2. model/sam2.yaml                         → cfg.model.*
       └─ which itself has defaults:
            model/default.yaml               (SAM2 hyperparameters, merged into model.*)
            model/image_encoder/hiera_b_conv_s_conv_t.yaml  → cfg.model.image_encoder.*
                                                              (will be overridden by experiment)
            model/memory_attention/default.yaml             → cfg.model.memory_attention.*
            model/memory_encoder/default.yaml               → cfg.model.memory_encoder.*
  3. loss/sam2.yaml                          → cfg.loss_fn.*
  4. metric/sam2_vos.yaml                    → cfg.val_metrics.* (top-level via @_global_)
  5. data/8_bm_7_ext.yaml                    → cfg.data.*
  6. experiment/test_sam2_no_ft_on_busi_camus.yaml
        → applies the overlay:
            - override /model/image_encoder@model.image_encoder: hiera_b+
              (swap in the standard SAM2 encoder)
            - override /data: busi_camus
              (swap in the smoke-test dataset)
            - scratch.experiment_name: benchmarks/busi_camus/sam2/p0
  7. _self_                                  → test.yaml's own values (recipe, mode, ckpt_path)
  8. CLI overrides                           → ckpt_path=${CHECKPOINT_DIR}/SAM2/...
```

The fully-resolved config is what the recipe receives via `hydra_cfg`.

---

## What each group does

### `scratch/default.yaml` — project-wide variables

The benchmark tree's scratch is mostly identical to pretrain's. The few
test-specific entries are at the bottom of the file:

```yaml
# -----------------------------Model-----------------------------
num_correction_pt_per_frame_val: 0  # 0 ~ 7, 0 = no point corrections during eval
prob_to_use_box_input_for_eval: 0   # 0 = no box prompts during eval, 1 = always
```

These two knobs control the **prompting protocol** during evaluation:

- `num_correction_pt_per_frame_val=0` and `prob_to_use_box_input_for_eval=0`
  → no iterative correction for the prediction mask per frame. **Current default.**
- `num_correction_pt_per_frame_val=N` (1 ≤ N ≤ 7) → N point corrections per
  frame, sampled from ground-truth mask. SAM2's typical evaluation protocol.
- `prob_to_use_box_input_for_eval=1` → bounding-box prompt derived from the
  GT mask. Most common SAM2 evaluation protocol in the literature.

For a fair "SAM2-style" comparison, set one of these consistently across all
three models when running the full benchmark.

The full list of variables is in `scratch/default.yaml` — the relevant ones
to change between runs are:

| Variable | Meaning | Default |
|---|---|---|
| `seed` | Master seed | `42` |
| `resolution` | Input image side (square) | `1024` |
| `dataset_dir` | Root of test datasets on disk | `${DATASET_DIR}` (required — e.g. `$WORK/Dataset/SaUS`) |
| `annotation_dir` | Root of test split lists | `${ANNOTATION_DIR}` (required — e.g. `$WORK/Dataset/SaUS_Annotation/38_pt_8_bm_7_ext_v2`) |
| `test_image_batch_size` / `test_video_batch_size` | Per-rank batch sizes | `1 / 1` |
| `val_image_batch_size` / `val_video_batch_size` | Same as test (smoke-test data uses these) | `8 / 1` |
| `num_correction_pt_per_frame_val` | Point-prompt protocol | `0` |
| `prob_to_use_box_input_for_eval` | Box-prompt protocol | `0` |

### `model/sam2.yaml` — model architecture & hyperparameters

```yaml
defaults:
  - default                                       # ← merges model/default.yaml AT THIS GROUP LEVEL
  - image_encoder@image_encoder: hiera_b_conv_s_conv_t
  - memory_attention@memory_attention: default
  - memory_encoder@memory_encoder: default

_target_: nemo_cv.components.models.sam2.sam2_train.SAM2Train

num_correction_pt_per_frame_val: ${scratch.num_correction_pt_per_frame_val}
prob_to_use_box_input_for_eval: ${scratch.prob_to_use_box_input_for_eval}
```

The two interpolation lines at the bottom override the corresponding values
in `model/default.yaml` so that experiments can flip prompting protocol via
the scratch namespace alone.

The default `image_encoder@image_encoder: hiera_b_conv_s_conv_t` is
overridden by every experiment overlay to match the model variant under
test.

### `model/image_encoder/`

| File | Architecture | Used by |
|---|---|---|
| `hiera_b+.yaml` | Hiera-B+ (embed=112, num_heads=2) | sam2-no-ft |
| `hiera_t.yaml` | Hiera-Tiny (embed=96, num_heads=1, stages=[1,2,7,2]) | medsam2 |
| `hiera_s.yaml` | Hiera-Small | (none currently — available for future ablations) |
| `hiera_l.yaml` | Hiera-Large | (none currently) |
| `hiera_b_conv_s_conv_t.yaml` | **TriBranchTrunk**: Hiera-B + ConvNeXt-S + ConvNeXt-T fused via cross-branch deformable attention | sonobase |
| `hiera_l_conv_b_conv_s.yaml` | TriBranchTrunk with Hiera-L + ConvNeXt-B + ConvNeXt-S | (future, scaled-up sonobase) |

To benchmark a new SAM2-architecture model, add an `image_encoder/<your>.yaml`
following the same shape (a `_target_: ImageEncoder` block with `trunk` and
`neck` sub-blocks), then create an experiment overlay that selects it.

### `data/busi_camus.yaml` and `data/8_bm_7_ext.yaml`

Both files declare a top-level **`test:`** block with one entry per
evaluation dataset. Each entry is a `TorchTrainMixedDataset` wrapping
`VOSDataset(SaUSRawDataset(...) | JSONRawDataset(...))` with the eval-time
sampler (`EvalSampler`) and val-time transforms (resize + normalize, no
augmentation).

`busi_camus.yaml` is BUSI + CAMUS test splits (smoke-test, ~164 samples
total). `8_bm_7_ext.yaml` is the real benchmark (BUSI, Brachial-Plexus,
C-TRUS, CAMUS, HC18, PFUS, RegPro, TG3K + ACOUSLIC, BUS-BRA, DDTI, FUGC,
KidneyUS, LUMINOUS, MMOTU-3d — 15 datasets total).

The recipe is tolerant of either `data.test` or `data.val` as the top-level
key (some pretrain-tree files use `val`); within this benchmark tree both
use `test` for clarity.

### `loss/sam2.yaml` — optional test loss

The same `MultiStepMultiMasksAndIous` from the pretrain tree. Producing a
test-loss number lets you compare loss values directly across models when
the metrics are close. It's optional — set `loss_fn: null` in the experiment
overlay to skip.

### `metric/sam2_vos.yaml` — evaluation metrics

`Sam2VOSMeanIoU` + `Sam2VOSDiceScore` (binary; foreground only;
`include_background: false`). Loaded under `@_global_`, so the resolved
config has `cfg.val_metrics.miou` and `cfg.val_metrics.dice` at the top
level (the recipe reads them from there).

### `experiment/` overlays

Each overlay file is small. The pattern is:

```yaml
# @package _global_
defaults:
  - override /model/image_encoder@model.image_encoder: <encoder_variant>
  - override /data: <dataset_combo>

scratch:
  experiment_name: benchmarks/<dataset>/<model>/p0   # determines output dir
```

Result: a single overlay file pins the **encoder + dataset** combination
that defines a benchmark cell, leaving everything else (loss, metrics,
prompts, batch sizes) to the project-wide defaults.

The `p0` suffix in `experiment_name` is intentional: lets you run multiple
"protocols" (e.g. `p0` = no iterative correction, `p7` = 7 correction
points per frame, `p_box` = box prompt) without clobbering each other's
results.

---

## CLI override syntax

Same as pretrain. See
[pretrain config README — CLI cheat sheet](../pretrain/README.md#cli-override-syntax-cheat-sheet)
for the full reference. The most common overrides for benchmarking:

```bash
# Switch to box-prompt protocol
+scratch.prob_to_use_box_input_for_eval=1

# Switch to 1 point prompt
+scratch.num_correction_pt_per_frame_val=1

# Run on the full 15-dataset eval suite
data=8_bm_7_ext

# Different output subdir for a different protocol
scratch.experiment_name=benchmarks/8_bm_7_ext/sam2/p_box

# Pin a specific sonobase checkpoint
ckpt_path=./experiments/.../epoch_3_step_581.pt
```

---

## How to add a new model variant

Say you want to benchmark a 4th model — let's call it `myunet`.

1. **Add the encoder yaml** (or reuse an existing one):

   ```yaml
   # src/configs/test_sam2/model/image_encoder/myunet.yaml
   _target_: nemo_cv.components.models.<your>.<encoder_class>
   ...
   ```

2. **Create the experiment overlay**:

   ```yaml
   # src/configs/test_sam2/experiment/test_myunet_on_busi_camus.yaml
   # @package _global_
   defaults:
     - override /model/image_encoder@model.image_encoder: myunet
     - override /data: busi_camus
   scratch:
     experiment_name: benchmarks/busi_camus/myunet/p0
   ```

3. **Add a runner script** mirroring the existing ones:

   ```bash
   # src/scripts/benchmarks/test_sam2/test_myunet_on_busi_camus.sh
   uv run torchrun --nproc_per_node=4 -m nemo_cv.recipes.benchmarks.test_sam2 \
     -c ./configs/test_sam2 -cn test \
     experiment=test_myunet_on_busi_camus \
     ckpt_path=<path_to_pt>
   ```

4. **If the new model isn't in `.pt` form**, add a converter alongside
   `convert_sonobase_dcp_to_pt.py`. The pattern is straightforward — see
   that file for a working example.

5. **Update `run_all_busi_camus.sh`** to include your model in the
   `compare_benchmarks.py` invocation.

---

## How to add a new test dataset

1. Place data on disk in the SaUS layout (`<NAME>/{images,gt,test_list.txt}`).
2. Open `data/8_bm_7_ext.yaml` and add a new top-level entry inside `test:`,
   following the existing pattern (image vs video, batch size, sampler). The
   `RepeatFactorWrapper(ConcatDataset(VOSDataset(...)))` triple-wrap is the
   convention.
3. Re-run any benchmark — the new dataset will get its own per-dataset metric
   row in `test_metrics.{json,csv}` and in the comparison table.

---

## Known issues / gotchas

These are documented in more detail in
[scripts README — Lessons learned](../../scripts/benchmarks/test_sam2/README.md#lessons-learned).
Brief recap for config editors:

1. **Default `data: 8_bm_7_ext`** in `test.yaml` — fine for the real
   benchmark but every smoke-test experiment overrides it. Don't run the
   recipe without an `experiment=...` override unless you actually want the
   full 15-dataset eval.
2. **`num_correction_pt_per_frame_val: 0`** by default — eval still uses
   an initial point prompt (from `prob_to_use_pt_input_for_eval: 1.0` in
   `model/default.yaml`) but no per-frame iterative correction. Fix the
   protocol — correction points, box prompts — before comparing numbers.
3. **`resolution: 1024`** — Meta's SAM2 was trained at 1024.
4. **Sonobase requires conversion** — DCP→.pt step before benchmark; see
   the [scripts README](../../scripts/benchmarks/test_sam2/README.md#test_sonobase_on_busi_camussh).

---

## See also

- [`src/scripts/benchmarks/test_sam2/README.md`](../../scripts/benchmarks/test_sam2/README.md) —
  How to run; CLI overrides; output interpretation.
- [`src/nemo_cv/recipes/benchmarks/README.md`](../../nemo_cv/recipes/benchmarks/README.md) —
  How `hydra.utils.instantiate` resolves the configs into Python objects;
  the recipe's loop structure.
- [`src/configs/pretrain/README.md`](../pretrain/README.md) —
  The pretrain config tree this benchmark tree mirrors. Use it as the
  reference when in doubt about Hydra composition idioms.
