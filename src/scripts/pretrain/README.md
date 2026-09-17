# Pretrain & Test scripts (`src/scripts/pretrain/`)

This directory holds the **shell entry points** for training and evaluating the
sonobase model. Everything here is a thin wrapper around `torchrun -m
nemo_cv.recipes.sonobase.<recipe>` with a Hydra config selection. Read this
first if you only want to run something.

For *what the configs do*, see [`src/configs/pretrain/README.md`](../../configs/pretrain/README.md).
For *how the training code actually works*, see
[`src/nemo_cv/recipes/sonobase/README.md`](../../nemo_cv/recipes/sonobase/README.md).

---

## Contents

```
src/scripts/pretrain/
├── pretrain_sonobase_on_sonocorpus.sh      Pretrain the model (BUSI + CAMUS)
├── test_sonobase_on_busi_camus.sh      Evaluate on BUSI/CAMUS test splits
├── test_sonobase_on_8_bm_7_ext.sh      Evaluate on 8 benchmark + 7 external sets
└── slurm/                              Reserved for cluster job scripts (empty)
```

---

## TL;DR — run pretraining

```bash
cd src                                          # IMPORTANT: cd into src/ first
bash scripts/pretrain/pretrain_sonobase_on_sonocorpus.sh
```

Output goes to `src/experiments/<scratch.experiment_name>/`. Default for the
existing script is
`src/experiments/sonobase/pretrain/busi_camus/hiera_b_conv_s_conv_t/`.

To watch progress while it runs:

```bash
# from src/
tail -f experiments/sonobase/pretrain/busi_camus/hiera_b_conv_s_conv_t/training.jsonl
tail -f experiments/sonobase/pretrain/busi_camus/hiera_b_conv_s_conv_t/validation.jsonl
```

A successful run produces, per epoch:

```
.../<run_name>/
├── training.jsonl                Per-step training metrics (loss, lr, mem, ...)
├── validation.jsonl              Per-validation metrics (val_loss, mIoU, dice, ...)
├── epoch_0_step_X/               DCP checkpoint at end of epoch 0
│   ├── config.yaml               Frozen Hydra config used for this run
│   ├── losses.json               Train/val loss summary at this checkpoint
│   ├── model/                    DCP-sharded model weights
│   ├── optim/                    DCP-sharded optimizer state
│   ├── rng/                      RNG state (per DP rank)
│   ├── scaler.pt                 AMP GradScaler state
│   └── step_scheduler.pt         Step counter + epoch counter
├── epoch_1_step_Y/               (next epoch)
│   └── ...
├── LATEST -> epoch_N_step_Z      Symlink to the most recent checkpoint
└── LOWEST_VAL -> epoch_K_step_W  Symlink to the best (lowest val_loss) checkpoint
```

---

## Prerequisites

Before running anything in this directory:

| Need | Default location | How to verify |
|---|---|---|
| Python env (uv) | `.venv/` at repo root | `uv run python -c 'import torch; print(torch.__version__)'` |
| Datasets | `${DATASET_DIR}/{BUSI,CAMUS,...}/{images,gt,...}` (`DATASET_DIR` required — e.g. `$WORK/Dataset/SaUS`) | `ls ${DATASET_DIR}/BUSI/` |
| SAM2 init checkpoint | `${CHECKPOINT_DIR}/SAM2/sam2.1_hiera_base_plus.pt` (`CHECKPOINT_DIR` required — e.g. `$WORK/Checkpoints`) | `ls -la ${CHECKPOINT_DIR}/SAM2/sam2.1_hiera_base_plus.pt` |
| GPUs | 4× CUDA devices | `nvidia-smi --query-gpu=index,memory.used --format=csv` |
| `LD_LIBRARY_PATH=""` | (set by scripts) | The scripts already do `export LD_LIBRARY_PATH=""` to avoid CUDA library version conflicts |

If you change any of these defaults (different dataset path, different
checkpoint), set them in `src/configs/pretrain/scratch/default.yaml` (the
`scratch.dataset_dir`, `scratch.ckpt_path`, etc. variables) rather than in the
shell scripts. See the [config README](../../configs/pretrain/README.md) for
the full variable list.

---

## Anatomy of a script

Take `pretrain_sonobase_on_sonocorpus.sh` line-by-line:

```bash
#!/bin/bash

export CUDA_VISIBLE_DEVICES=1,2,3,4   # which physical GPUs to use; 4 of them
export HYDRA_FULL_ERROR=1             # show full Hydra tracebacks on config errors
export PYTHONPATH=$PYTHONPATH:$(pwd)/../   # add src/ so `nemo_cv` resolves (run from src/)
export LD_LIBRARY_PATH=""             # avoid system CUDA libs conflicting with .venv's

uv run torchrun --nproc_per_node=4 \
  -m nemo_cv.recipes.sonobase.pretrain \
  -c ./configs/pretrain \
  experiment=pretrain_hiera_b_conv_s_conv_t_on_busi_camus \
  step_scheduler.global_batch_size=4 \
  scratch.experiment_name=sonobase/pretrain/busi_camus/hiera_b_conv_s_conv_t
```

What each piece does:

- `uv run` — runs the command inside the project's uv-managed `.venv`.
- `torchrun --nproc_per_node=4` — launches 4 worker processes (one per GPU)
  with PyTorch's elastic launcher; sets up the process group needed by DDP.
- `-m nemo_cv.recipes.sonobase.pretrain` — entry point is `main()` in
  `src/nemo_cv/recipes/sonobase/pretrain.py`.
- `-c ./configs/pretrain` — Hydra config root (relative to the working
  directory, which is `src/`).
- `experiment=...` — selects an experiment overlay from
  `configs/pretrain/experiment/`. The overlay sets `_global_` package overrides
  (data choice, run name, etc.).
- `step_scheduler.global_batch_size=4` / `scratch.experiment_name=...` — direct
  CLI overrides; same syntax as inside any Hydra app (see [config
  README](../../configs/pretrain/README.md) for syntax reference).

The test scripts (`test_sonobase_on_*.sh`) are structurally identical but
target the test recipe (`-m nemo_cv.recipes.sonobase.test`) with `-cn test`
to select the test top-level config.

---

## Common operations

### Change the experiment name (avoid clobbering a previous run)

The default `experiment_name` is set in the experiment YAML
(e.g. `sonobase/pretrain/busi_camus/hiera_b_conv_s_conv_t`). The recipe
writes to `./experiments/${scratch.experiment_name}/` and **auto-resumes
from the latest checkpoint** if the dir already contains one
(see "Resuming" below).

To run a fresh fork alongside an existing one, override on the CLI:

```bash
bash scripts/pretrain/pretrain_sonobase_on_sonocorpus.sh \
  scratch.experiment_name=sonobase/pretrain/busi_camus/hiera_b_conv_s_conv_t__$(date +%Y%m%d_%H%M)
```

### Run on a different dataset

Either change the experiment override:

```
experiment=pretrain_hiera_b_conv_s_conv_t_on_38_pt_8_bm   # uses /data: 38_pt_8_bm
```

…or override `data` directly:

```
data=busi    # any file in configs/pretrain/data/<NAME>.yaml
```

### Run on different GPUs / different process count

Edit `CUDA_VISIBLE_DEVICES` and `--nproc_per_node` together — they must match
in count.

### Run for fewer epochs (smoke test)

```
scratch.num_epochs=2
```

### Override any nested config value

Standard Hydra syntax:

```
optim.optimizer.eps=1e-6                       # change AdamW eps
data.train.num_workers=0                       # disable dataloader workers
distributed.find_unused_parameters=false       # tighten DDP
```

### Add a new key (Hydra rejects this by default unless prefixed with `+`)

```
+wandb.project=sonobase   # add a wandb section that doesn't exist in train.yaml
```

---

## Resuming an interrupted run

The recipe uses `BaseRecipe.load_checkpoint(restore_from=None)` in `setup()`,
which **auto-detects the latest checkpoint** in `checkpoint.save_dir` (i.e.,
`./experiments/${scratch.experiment_name}/`) and resumes it. So:

- **To resume**: just re-run the same script with the same `experiment_name`.
  The recipe finds `LATEST -> epoch_N_step_M/` and loads model + optimizer +
  scheduler + RNG + dataloader state.
- **To start fresh under a new name**: change `scratch.experiment_name` to a
  new path (the simplest way is to append a date-stamped suffix, e.g.
  `experiment_name=...__$(date +%Y%m%d_%H%M)`).
- **To resume from a specific (non-latest) checkpoint**: pass
  `+checkpoint.restore_from=./experiments/.../epoch_3_step_X` on the CLI.
- **To start fresh under the same name as an existing run**: delete the
  experiment directory first (`rm -rf experiments/<name>/`).

What gets restored on resume:

| State | Mechanism |
|---|---|
| Model weights | DCP load from `epoch_N_step_M/model/` |
| Optimizer state (AdamW moments + step counters) | DCP load from `.../optim/` |
| AMP GradScaler | `torch.load` from `scaler.pt` |
| StepScheduler (step + epoch counters) | `torch.load` from `step_scheduler.pt` |
| Per-rank RNG state | DCP load from `rng/` |
| DataLoader state | DCP load (DistributedSampler epoch + iter index) |

The training run resumes at the next epoch boundary.

---

## Cluster (Slurm) usage

The `slurm/` subdirectory is empty as of this writing — to be filled in when
the project moves from local 4×GPU runs to multi-node cluster runs. Conventions
for that environment (read-only NGC container + project venv overlay) are
documented in `src/setup/` and the cluster setup scripts there.

---

## Output: how to read it

### `training.jsonl`

One line per training step (after the buffering fix, see "Lessons learned"
below). Schema:

```json
{
  "step": 42,
  "epoch": 1,
  "timestamp": "2026-04-25T...",
  "loss": 1.234,                            // total weighted loss
  "lr": 5e-05,                              // current LR (first param group)
  "mem": 24.1,                              // peak GPU memory (GiB)
  "where": 0.213,                           // fractional progress through training
  "batch_time": 1.27,                       // wall-clock seconds for this step
  "all_loss_mask": 0.123,                   // sub-loss components (per dataset key)
  "all_loss_dice": 0.456,
  "all_loss_iou": 0.234,
  "all_loss_class": 0.012
}
```

`where = (epoch + data_iter / iters_per_epoch) / max_epochs` — used by the LR
scheduler to interpolate between `start_value` and `end_value`. `where=1.0`
means training is over.

### `validation.jsonl`

One line per validation event (after `is_val_step` fires, currently every
epoch boundary by default):

```json
{
  "step": 41,
  "epoch": 2,
  "val_loss": 7.81,
  "lr": 4.57e-05,
  "mem": 8.4,
  "miou": 0.633,                            // aggregate (averaged across val datasets)
  "dice": 0.758,
  "BUSI/miou": 0.589,                       // per-dataset breakdown
  "BUSI/dice": 0.721,
  "CAMUS/miou": 0.634,
  "CAMUS/dice": 0.759
}
```

### Checkpoints

Each `epoch_N_step_M/` is a self-contained snapshot:

- `config.yaml` — the **fully-resolved** Hydra config that produced this run.
  Open it to see exactly what was trained (model, optimizer, data, all
  interpolated values resolved).
- `losses.json` — `{"train_loss": ..., "val_loss": ..., "miou": ...}` summary.
  Useful for quick triage without parsing the full JSONL files.
- `model/`, `optim/`, `rng/` — DCP-sharded directories. Don't open by hand;
  load via the recipe's checkpointer.
- `scaler.pt`, `step_scheduler.pt` — small `torch.save` blobs.

The `LATEST` and `LOWEST_VAL` symlinks point to the most recent and best
(by val_loss) checkpoints respectively. Use `LOWEST_VAL` when you want to
evaluate the best-so-far model.

---

## Lessons learned (read before debugging)

These are gotchas the team has hit while developing this code. Adding here so
the next person doesn't lose a day rediscovering them.

### 1. Validation metrics are recorded once per epoch by default

`step_scheduler.val_every_steps` is `null` in `train.yaml`, so validation only
runs when `is_ckpt_step` fires (i.e., end of epoch + `save_checkpoint_every_epoch=true`).
If you want mid-epoch validation, set `step_scheduler.val_every_steps=N` on the CLI.

### 2. The `MetricLogger` used to buffer 100 records before flushing

That meant `validation.jsonl` looked **empty** during early training because
the buffer (1 record per epoch) never reached 100 before the first save.
Fixed in `pretrain.py` `setup()` by setting
`self.metric_logger_valid.buffer_size = 1` (and `flush=True`). Same for
`metric_logger_train`. If you ever swap loggers, preserve this.

### 3. `multiplier: 0.02` in `data/busi_camus.yaml` is a debug-scale setting

Each VOSDataset has `multiplier: 0.02`, which means the `RepeatFactorWrapper`
includes each sample with **2% probability per epoch**. Practical effect:
~12 BUSI + ~14 CAMUS samples per epoch (binomial-distributed), roughly 5
batches per rank, ~50 SGD steps over 10 epochs — useful for verifying the
pipeline runs end-to-end, **not for real training**. For real training, set
`multiplier: 1` on each dataset entry.

### 4. Per-epoch dataloader length varies (and that breaks `is_ckpt_step`)

Because `RepeatFactorWrapper.set_epoch(epoch)` re-seeds the random sample
selection, `len(dataloader)` returns different values for different epochs
(when `multiplier < 1`). The `StepScheduler` computes `epoch_len` once at
init from `len(dataloader)` for epoch 0 and freezes it. Result: end-of-epoch
checkpoint detection (`step % epoch_len == epoch_len - 1`) **misfires** when
actual epoch length differs from the frozen value, leading to missing or
mis-named checkpoints (e.g., `epoch_2_step_41` instead of `epoch_1_step_41`).
Workarounds: (a) use `multiplier: 1` so length is constant; (b) trigger
checkpoints from the recipe's outer-loop boundary (`data_iter == iters_per_epoch - 1`)
instead of `is_ckpt_step` — this fix is documented in the recipe README and
not yet applied.

### 5. SAM2's `no_mem_pos_enc` was a "dead" parameter — fixed

When `directly_add_no_mem_embed=true` (your config), `no_mem_pos_enc` was
declared as an `nn.Parameter` but never used in the forward pass. PyTorch
optimizers create per-parameter state lazily on first non-None gradient, so
this parameter never got optimizer state. DCP saved nothing for it, but
expected to load it on resume → `RuntimeError: Missing key in checkpoint
state_dict: optim.state.no_mem_pos_enc.step`. **Fixed** by conditionally
registering the parameter in `SAM2Base.__init__` (mirrors the existing
pattern for `no_obj_ptr` and `no_obj_embed_spatial`).

### 6. `train.yaml` references a non-existent default data config

`train.yaml` has `data: 38_pt_8_bm_5_ext` in its defaults list, but no such
file exists in `configs/pretrain/data/`. This **only matters if you forget
the `experiment=...` override**, which sets `data:` via override directive.
All current scripts pass an `experiment=...` so they work, but a bare
`uv run torchrun ... -m nemo_cv.recipes.sonobase.pretrain -c ./configs/pretrain`
with no overrides will fail with `MissingConfigException`. To fix, change the
default to a real file (e.g. `busi_camus`) — left as-is for now to avoid
accidentally re-triggering anyone's habits.

### 7. `PYTHONPATH` in the scripts assumes you `cd` into `src/`

The scripts do `export PYTHONPATH=$PYTHONPATH:$(pwd)/../`, which expands to
`<repo_root>` when you run them from `src/`. If you run from elsewhere
(e.g. the repo root), `$(pwd)/../` becomes the parent of the repo and
`nemo_cv` won't import. **Always `cd src` before invoking these scripts.**

### 8. `find_unused_parameters: true` in the train config

Required because `forward_backbone_per_frame_for_eval=true` and a few other
flags route some parameters conditionally. Without it, the first DDP backward
pass would error. There's no easy way to remove this without changing the
SAM2 forward path; live with the ~1% overhead it costs.

---

## See also

- [`src/configs/pretrain/README.md`](../../configs/pretrain/README.md) —
  Hydra config tree, what each group does, how to add a new experiment.
- [`src/nemo_cv/recipes/sonobase/README.md`](../../nemo_cv/recipes/sonobase/README.md) —
  Architecture overview, recipe walkthrough, training loop internals,
  checkpoint format details.
