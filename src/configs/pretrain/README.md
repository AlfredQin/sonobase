# Pretrain Hydra config tree (`src/configs/pretrain/`)

This directory holds all the YAML configs that drive sonobase pretraining
and evaluation. The configs use [Hydra](https://hydra.cc/docs/intro/) to
compose a single hierarchical config from many small files at runtime.

If you want to **run** something, see
[`src/scripts/pretrain/README.md`](../../scripts/pretrain/README.md).
If you want to **understand the recipe code**, see
[`src/nemo_cv/recipes/sonobase/README.md`](../../nemo_cv/recipes/sonobase/README.md).

This document is for anyone who wants to **change a hyperparameter, add a
dataset, or create a new experiment**.

---

## Directory layout

```
src/configs/pretrain/
├── train.yaml                          ROOT config for training
├── test.yaml                           ROOT config for evaluation
│
├── scratch/
│   └── default.yaml                    Project-wide variables (seed, paths, batch sizes, transforms)
│
├── model/
│   ├── default.yaml                    SAM2 hyperparameters (NOT model architecture)
│   ├── sam2_train.yaml                 The SAM2Train _target_ + composes sub-components
│   ├── image_encoder/                  Image-encoder variants (TriBranch / Hiera-only / etc.)
│   │   ├── hiera_b_conv_s_conv_t.yaml      ← currently used
│   │   ├── hiera_b+.yaml
│   │   ├── hiera_l.yaml
│   │   ├── hiera_l_conv_b_conv_s.yaml
│   │   ├── hiera_s.yaml
│   │   └── hiera_t.yaml
│   ├── memory_attention/
│   │   └── default.yaml
│   └── memory_encoder/
│       └── default.yaml
│
├── data/
│   ├── busi.yaml                       Single-dataset (BUSI image only) — debugging
│   ├── busi_camus.yaml                 BUSI image + CAMUS video (train + val) ← current
│   ├── busi_camus_test.yaml            BUSI + CAMUS, val split only (used by test recipe)
│   ├── 38_pt_8_bm.yaml                 Full-scale: 38 pretrain + 8 benchmark train datasets
│   ├── 8_bm_7_ext.yaml                 15-dataset evaluation set (8 benchmark + 7 external)
│   └── transforms/
│       └── sam2_train.yaml             (legacy; transforms now live inside scratch/default.yaml)
│
├── loss/
│   └── sam2.yaml                       MultiStepMultiMasksAndIous, keyed by dataset name
│
├── optim/
│   └── sam2.yaml                       AdamW + layer decay + cosine schedule
│
├── metric/
│   └── sam2_vos.yaml                   Sam2VOSMeanIoU + Sam2VOSDiceScore
│
└── experiment/                         Per-experiment overlays (data + run name + epochs)
    ├── pretrain_hiera_b_conv_s_conv_t_on_busi_camus.yaml    ← canonical pretrain
    ├── pretrain_hiera_b_conv_s_conv_t_on_38_pt_8_bm.yaml
    ├── test_hiera_b_conv_s_conv_t_on_busi_camus.yaml        ← canonical test
    └── test_hiera_b_conv_s_conv_t_on_8_bm_7_ext.yaml
```

---

## How Hydra composes the config

Both `train.yaml` and `test.yaml` are root configs with a `defaults:` list
that tells Hydra which sub-configs to pull in and where to slot them. Here
is `train.yaml`'s defaults block (annotated):

```yaml
defaults:
  - scratch: default              # → cfg.scratch.* from scratch/default.yaml
  - model@model: sam2_train       # → cfg.model.*  from model/sam2_train.yaml (recursively)
  - loss@loss_fn: sam2            # → cfg.loss_fn.* from loss/sam2.yaml
  - optim@optim: sam2             # → cfg.optim.*  from optim/sam2.yaml
  - metric@_global_: sam2_vos     # → merged AT TOP LEVEL (sets cfg.val_metrics.*)
  - data: 38_pt_8_bm_5_ext        # → cfg.data.*   (NOTE: stale — see "Known issues")
  - experiment: null              # ← intentionally null; overlay is opt-in
  - _self_                        # this file's own keys win on conflict
```

Three Hydra concepts to understand:

1. **`<group>: <name>`** picks `<group>/<name>.yaml` and merges it into
   `cfg.<group>`.
2. **`<group>@<package>: <name>`** picks `<group>/<name>.yaml` but merges it
   under `cfg.<package>` instead of `cfg.<group>`. So
   `loss@loss_fn: sam2` puts `loss/sam2.yaml`'s contents under `cfg.loss_fn`,
   not `cfg.loss`. This is how the same file can be re-used at different
   slots in different recipes (the recipe code reads `hcfg.loss_fn`).
3. **`@_global_`** merges contents at the top level (no nesting). So
   `metric@_global_: sam2_vos` makes the `val_metrics:` block defined inside
   `metric/sam2_vos.yaml` appear as `cfg.val_metrics`.

The order matters: later entries override earlier ones, **and `_self_` is the
file you're reading** — placing it last means this file's direct entries
win against any defaults. Some other Hydra apps put `_self_` first; we put
it last.

### Composition flow for the canonical pretrain run

```
$ bash scripts/pretrain/pretrain_sonobase_on_sonocorpus.sh

  → uv run torchrun ... -m nemo_cv.recipes.sonobase.pretrain
        -c ./configs/pretrain
        experiment=pretrain_hiera_b_conv_s_conv_t_on_busi_camus
        scratch.experiment_name=...

Hydra reads train.yaml's defaults list and composes:

  1. scratch/default.yaml              → cfg.scratch.*
  2. model/sam2_train.yaml             → cfg.model.*
       └─ which itself has defaults:
            model/default.yaml         (SAM2 hyperparameters, merged into model.*)
            model/image_encoder/hiera_b_conv_s_conv_t.yaml  → cfg.model.image_encoder.*
            model/memory_attention/default.yaml             → cfg.model.memory_attention.*
            model/memory_encoder/default.yaml               → cfg.model.memory_encoder.*
  3. loss/sam2.yaml                    → cfg.loss_fn.*
  4. optim/sam2.yaml                   → cfg.optim.*
  5. metric/sam2_vos.yaml              → cfg.val_metrics.*  (top-level via @_global_)
  6. data/<whatever>.yaml              → cfg.data.*
  7. experiment/pretrain_hiera_b_conv_s_conv_t_on_busi_camus.yaml
        → applies the overlay (overrides /data: busi_camus and sets scratch.experiment_name)
  8. _self_                            → train.yaml's own values (recipe, mode, max_epochs, ...)
  9. CLI overrides                     → scratch.experiment_name=..., step_scheduler.global_batch_size=4
```

The final composed config is what the recipe receives via
`hydra_cfg`, and it's also snapshotted into every checkpoint as `config.yaml`.

---

## What each group does

### `scratch/` — project-wide variables

`scratch/default.yaml` is a **bag of variables** referenced by interpolation
(`${scratch.<key>}`) from everything else. Treat it as the "what to change
between runs" file. Key entries:

| Variable | Meaning | Defaults |
|---|---|---|
| `seed` | Master seed (DDP-ranked downstream) | `42` |
| `resolution` | Input image side (square crop) | `1024` |
| `num_epochs` | Number of training epochs | `40` |
| `dataset_dir` | Root of training datasets on disk | `${DATASET_DIR}` (required — e.g. `$WORK/Dataset/SaUS`) |
| `annotation_dir` | Root of val/test split lists | `${ANNOTATION_DIR}` (required — e.g. `$WORK/Dataset/SaUS_Annotation/38_pt_8_bm_7_ext_v2`) |
| `ckpt_path` | SAM2 init checkpoint (Meta release) | `${CHECKPOINT_DIR}/SAM2/sam2.1_hiera_base_plus.pt` (`CHECKPOINT_DIR` required — e.g. `$WORK/Checkpoints`) |
| `experiment_name` | Run identifier; becomes `./experiments/<name>/` | `sonobase/pretrain/default` |
| `train_image_batch_size` / `train_video_batch_size` | Per-rank batch for image / video datasets | `8 / 1` |
| `val_image_batch_size` / `val_video_batch_size` | Per-rank batch for val | `8 / 1` |
| `test_image_batch_size` / `test_video_batch_size` | Per-rank batch for test | `1 / 1` |
| `num_frames` | Frames per video clip during training | `8` |
| `max_num_objects` | Max objects to track per clip | `20` |
| `num_train_workers` | DataLoader `num_workers` | `8` |
| `train_video_sampler` / `train_image_sampler` | `_target_`-instantiated VOSSampler subclass | `RandomUniformSampler` |
| `val_*_sampler` | Eval sampler (returns all frames) | `EvalSampler` |
| `train_transforms` / `val_transforms` | Composed augmentation pipelines | see file |

If you need to override one value for a single run, use a CLI override
(`scratch.num_epochs=2`). If you need to change something across all runs,
edit this file. **Do not** sprinkle CLI overrides for things that should be
project defaults.

### `model/` — model architecture & SAM2 hyperparameters

`model/sam2_train.yaml` is the entry point. It looks like:

```yaml
defaults:
  - default                                       # ← merges model/default.yaml AT THIS GROUP LEVEL
  - image_encoder@image_encoder: hiera_b_conv_s_conv_t
  - memory_attention@memory_attention: default
  - memory_encoder@memory_encoder: default

_target_: nemo_cv.components.models.sam2.sam2_train.SAM2Train
```

So the final `cfg.model` has:

- `cfg.model._target_` = `SAM2Train` (instantiated by `hydra.utils.instantiate`)
- `cfg.model.image_encoder.*`, `cfg.model.memory_attention.*`,
  `cfg.model.memory_encoder.*` — sub-component configs (each has its own
  `_target_` for nested instantiation)
- `cfg.model.num_maskmem`, `cfg.model.image_size`,
  `cfg.model.directly_add_no_mem_embed`, etc. — SAM2 scalar
  hyperparameters from `model/default.yaml`

To change the image encoder variant, change the defaults entry to one of:

| File | What it is |
|---|---|
| `image_encoder/hiera_b_conv_s_conv_t.yaml` | TriBranchTrunk: Hiera-B + ConvNeXt-S + ConvNeXt-T (current default) |
| `image_encoder/hiera_b+.yaml` | Plain Hiera-B+ (SAM2's default) |
| `image_encoder/hiera_l.yaml` | Hiera-L |
| `image_encoder/hiera_l_conv_b_conv_s.yaml` | TriBranch with Hiera-L + ConvNeXt-B + ConvNeXt-S |
| `image_encoder/hiera_s.yaml` | Hiera-S (small) |
| `image_encoder/hiera_t.yaml` | Hiera-T (tiny) |

The TriBranchTrunk-style files configure three parallel image encoders
fused via cross-branch deformable attention; see the
[recipe README](../../nemo_cv/recipes/sonobase/README.md) for the
architecture explanation.

### `data/` — datasets

Each YAML in `data/` defines a `train:` and/or `val:` block. The `train` is
a single `TorchTrainMixedDataset`; `val` is a dict `{dataset_name:
TorchTrainMixedDataset}` so each val dataset gets its own loader and
per-dataset metrics.

| File | Purpose |
|---|---|
| `busi.yaml` | BUSI only (image dataset). Useful for fast iteration |
| `busi_camus.yaml` | BUSI image + CAMUS video, train + val splits |
| `busi_camus_test.yaml` | BUSI + CAMUS test split — for the test recipe smoke run |
| `38_pt_8_bm.yaml` | Full pretrain: 38 pretrain datasets + 8 benchmark, train + val |
| `8_bm_7_ext.yaml` | Eval suite: 8 benchmark + 7 external test datasets |

Each dataset entry inside a YAML has the structure:

```yaml
_target_: nemo_cv.components.datasets.sam2.utils.RepeatFactorWrapper
dataset:
  _target_: nemo_cv.components.datasets.sam2.utils.ConcatDataset
  datasets:
    - _target_: nemo_cv.components.datasets.sam2.vos_dataset.VOSDataset
      training: true
      transforms: ${scratch.train_transforms}
      multiplier: 1                                # ← repeat factor; see "Known issues" in scripts README
      video_dataset:
        _target_: nemo_cv.components.datasets.sam2.vos_raw_dataset.SaUSRawDataset   # or JSONRawDataset
        img_folder: ${scratch.dataset_dir}/<NAME>/images
        gt_folder:  ${scratch.dataset_dir}/<NAME>/gt
        file_list_txt: ${scratch.dataset_dir}/<NAME>/train_list.txt
      sampler: ${scratch.train_image_sampler}      # or train_video_sampler
```

Key knobs:

- `multiplier`: per-sample repeat factor. `1.0` → each sample exactly once
  per epoch (deterministic). Fractional (e.g. `0.02`) → stochastic
  inclusion. Greater than 1 → fixed N copies + stochastic remainder. **In
  the current `busi_camus.yaml` this is `0.02`, which is debug-scale.**
- `video_dataset._target_`: `SaUSRawDataset` for image datasets,
  `JSONRawDataset` for video datasets (loads frame-level annotations from a
  per-video JSON file).
- `sampler`: `RandomUniformSampler(num_frames, max_num_objects)` for training,
  `EvalSampler()` for val/test.

### `loss/` — loss functions

`loss/sam2.yaml` defines a single key `all` mapping to
`MultiStepMultiMasksAndIous`. The recipe reads `cfg.loss_fn` and builds an
`nn.ModuleDict` keyed by the YAML's top-level keys. At training time, each
batch carries a `dict_key` (set in the data config's collate_fn —
`dict_key: all` everywhere right now) and the recipe dispatches to
`self.loss_fn[batch.dict_key]`. This is how SAM2's training pipeline supports
multiple losses for multiple data sources — currently we only use one ("all").

The loss has four sub-components (mask BCE, dice, IoU prediction, class
prediction) weighted via `weight_dict`.

### `optim/` — optimizer + LR schedule

`optim/sam2.yaml` configures the SAM2 optimizer wrapper (see
`nemo_cv.components.optim.optimizer`). Pieces:

- `amp.enabled: true` + `amp_dtype: bfloat16` → wraps the forward pass in
  `torch.autocast(dtype=bfloat16)`. (GradScaler is disabled with bf16, as
  per AutoModel convention.)
- `optimizer._target_: torch.optim.AdamW`, `_partial_: true` → instantiates a
  partial that takes `params` later.
- `gradient_clip` → max_norm=0.1 clipper applied per step.
- `param_group_modifiers` → currently a single `layer_decay_param_modifier`
  with `layer_decay_value: 0.9` applied to `image_encoder.trunk` (with a
  pos_embed override). This applies the standard "lower learning rate for
  earlier layers" trick during fine-tuning.
- `options.lr` and `options.weight_decay` → per-param-group cosine and
  constant schedulers (separate cosine for the image encoder and for
  everything else). Schedulers tick on a `where` parameter (fractional
  training progress) computed by the recipe.

To change the LR range: edit `start_value` / `end_value` under
`options.lr.[*].scheduler`. To disable layer decay: empty the
`param_group_modifiers` list.

### `metric/` — validation metrics

`metric/sam2_vos.yaml` defines two metrics:

```yaml
val_metrics:
  miou:
    _target_: nemo_cv.components.metrics.segmentation.Sam2VOSMeanIoU
    num_classes: 2
    include_background: false
    input_format: index
  dice:
    _target_: nemo_cv.components.metrics.segmentation.Sam2VOSDiceScore
    ...
```

Note this file is loaded with `@_global_`, so `val_metrics:` lands at the top
level of the composed config (`cfg.val_metrics`), not under `cfg.metric`.

The recipe creates one set of metric instances per val dataset (per-dataset
metrics under `BUSI/miou`, `CAMUS/miou`, etc.) plus an aggregate set across
all datasets.

### `experiment/` — per-run overlays

Each experiment YAML is a small overlay that changes a few global config
values for a specific run. Example
(`experiment/pretrain_hiera_b_conv_s_conv_t_on_busi_camus.yaml`):

```yaml
# @package _global_                ← important: makes overrides apply at top level
defaults:
  - override /data: busi_camus     ← swap the data group's choice

scratch:
  experiment_name: sonobase/pretrain/busi_camus/hiera_b_conv_s_conv_t
  num_epochs: 10

checkpoint:
  save_dir: ./checkpoints/sonobase/pretrain/busi_camus/hiera_b_conv_s_conv_t
```

Two important Hydra idioms used here:

- **`# @package _global_`** at the top of the file is a directive that says
  "these keys merge at the top of the composed config", not nested under
  `cfg.experiment.*`. Without it, an experiment file's contents would land
  under `cfg.experiment` and not affect anything.
- **`override /data: busi_camus`** in the defaults list (note the leading `/`)
  is "absolute" — it overrides the choice in the *root* `train.yaml`'s
  defaults list, not relative to the experiment file's group.

The current experiment files set `scratch.experiment_name` (which controls
the output directory) and the data choice. Anything else can be changed at
the CLI when invoking the script.

---

## CLI override syntax (cheat sheet)

All of these are accepted on the command line after the `-c` argument:

| Syntax | Effect |
|---|---|
| `key=value` | Set existing key (errors if missing) |
| `+key=value` | Add a new key (errors if already present) |
| `++key=value` | Set or add (no error either way) |
| `~key` | Remove a key |
| `key.subkey=value` | Override nested |
| `key.subkey=[a,b,c]` | List value |
| `key.subkey="quoted string"` | Strings with spaces |
| `experiment=name` | Select an overlay from `experiment/` |
| `~experiment` | Disable any default experiment overlay |
| `data=name` | Override the data group choice |
| `+data.train.num_workers=0` | Add a key inside an existing nested config |

Hydra evaluates them left-to-right, after the defaults list and `_self_`.

---

## How to add a new experiment

1. **Pick a unique name.** Convention is
   `<recipe>_<encoder>_on_<dataset>.yaml`, e.g.
   `pretrain_hiera_l_on_busi_camus.yaml` or
   `test_hiera_b_conv_s_conv_t_on_8_bm_7_ext.yaml`.
2. **Copy an existing experiment** as a starting point:
   ```bash
   cp src/configs/pretrain/experiment/pretrain_hiera_b_conv_s_conv_t_on_busi_camus.yaml \
      src/configs/pretrain/experiment/pretrain_<encoder>_on_<dataset>.yaml
   ```
3. **Edit it.** Typical changes:
   - `defaults: - override /data: <your_data_yaml>`
   - Optionally `defaults: - override /model/image_encoder@model.image_encoder: <variant>`
     to change the image encoder while keeping everything else
   - `scratch.experiment_name: <unique_path>`
4. **Make a runner script** in `src/scripts/pretrain/` that points at it:
   ```bash
   uv run torchrun --nproc_per_node=4 -m nemo_cv.recipes.sonobase.pretrain \
     -c ./configs/pretrain \
     experiment=<your_experiment_name>
   ```
5. **Smoke-test** with a small `scratch.num_epochs=1` first.

## How to add a new dataset

1. Place your data on disk in the same layout as the existing datasets (under `${DATASET_DIR}`, e.g. `$WORK/Dataset/SaUS`):
   ```
   ${DATASET_DIR}/<MyDataset>/
   ├── images/
   ├── gt/
   ├── train_list.txt
   └── val_list.txt   (and test_list.txt if relevant)
   ```
2. Decide if it's an **image** dataset (each "video" is a single frame) or a
   **video** dataset (per-video sequences with annotations).
3. Open `data/busi_camus.yaml` (or whichever closest existing config), copy
   one of the `RepeatFactorWrapper(...)` blocks, and:
   - Update `img_folder`, `gt_folder`, `file_list_txt`.
   - Set `_target_` of `video_dataset` to `SaUSRawDataset` (image) or
     `JSONRawDataset` (video).
   - Adjust `sampler` (image vs video sampler from `scratch`).
4. Either edit an existing data YAML (if your new dataset is a sibling) or
   create a new `data/<my_data_combo>.yaml`. Keep `train:` and `val:` blocks
   in sync (same datasets, different splits).
5. Reference your new data file from an experiment overlay (`override /data:
   <my_data_combo>`).

---

## Known issues / gotchas

These are documented in more detail in the
[scripts README](../../scripts/pretrain/README.md#lessons-learned), but
mentioning here for config-editors:

1. **`train.yaml` defaults `data: 38_pt_8_bm_5_ext`** — that file does not
   exist. All current scripts pass `experiment=...` which overrides this, so
   it doesn't bite in practice, but a bare `uv run torchrun ... pretrain
   -c ./configs/pretrain` will fail with `MissingConfigException`. Either
   change the default to a real file, or always pass `experiment=...`.

2. **`pretrain_hiera_b_conv_s_conv_t_on_busi_camus.yaml` overrides
   `checkpoint.save_dir`** with a hardcoded `./checkpoints/...` path. This
   conflicts with the global `checkpoint.save_dir: ./experiments/${scratch.experiment_name}`
   in `train.yaml`. The hardcoded path wins (Hydra's `@_global_` overlay
   beats the root). The shell script then sets `scratch.experiment_name=...`
   which has no effect on save_dir. **Workaround**: when running, pass
   `+checkpoint.save_dir=./experiments/${scratch.experiment_name}` to undo
   the hardcoded override. Better: delete that block from the experiment
   YAML.

3. **`pretrain_hiera_b_conv_s_conv_t_on_38_pt_8_bm.yaml` references
   `38_pt_8_bm_5_ext`** (the same non-existent file).

4. **`multiplier: 0.02` in `busi_camus.yaml`** — see scripts README.
   Stochastic per-epoch length; debug-only; set to `1` for real runs.

5. **`data/transforms/sam2_train.yaml`** is unused. Transforms are now
   defined inline inside `scratch/default.yaml` (`train_transforms` /
   `val_transforms`) and referenced by interpolation. Safe to ignore /
   delete.

---

## See also

- [`src/scripts/pretrain/README.md`](../../scripts/pretrain/README.md) —
  How to run, command-line overrides, common operations.
- [`src/nemo_cv/recipes/sonobase/README.md`](../../nemo_cv/recipes/sonobase/README.md) —
  How `hydra.utils.instantiate` resolves the configs into actual Python
  objects, and the recipe's training loop.
- [Hydra documentation](https://hydra.cc/docs/intro/) — official reference
  for the override syntax and defaults list semantics.
