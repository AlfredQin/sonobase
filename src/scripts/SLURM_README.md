# Slurm scripts — cluster submission guide

Slurm wrappers for every long-running phase of the project. Each phase
keeps its `.sbatch` files next to the regular shell wrappers under
`src/scripts/<phase>/slurm/`, so the slurm scripts can `source` /
delegate to the same shell helpers (no logic duplication).

| Phase                                  | Slurm scripts                                                          | Typical wall time per job          |
|---|---|---|
| **Pretrain** (Phase 2)                 | [`scripts/pretrain/slurm/`](pretrain/slurm/)                           | 24 h × N nodes — needs resume      |
| **SonoBase test** (Phase 3)            | [`scripts/pretrain/slurm/test_sonobase_on_8_bm_7_ext.sbatch`](pretrain/slurm/test_sonobase_on_8_bm_7_ext.sbatch) | ~30-60 min on 4 GPUs               |
| **Benchmark iterative sweep** (Phase 4)| [`scripts/benchmarks/test_sam2/iterative/slurm/`](benchmarks/test_sam2/iterative/slurm/) | ~2-3 h per (model, prompt) on 4 GPUs |
| **US-RFDETR** (Phase 6)                | [`scripts/us_rfdetr/slurm/`](us_rfdetr/slurm/)                         | ~30 min per cell on 4 GPUs         |
| **Few-shot adaptation** (F1)           | [`scripts/few_shot/slurm/`](few_shot/slurm/)                           | ~6-8 h per dataset on 1 GPU        |
| **Analysis Stage 1** (predictions)     | [`scripts/analysis/slurm/_save_predictions_one.sbatch`](analysis/slurm/_save_predictions_one.sbatch) | ~30-60 min per dataset on 1 GPU    |
| **Analysis Stage 2** (CPU)             | [`scripts/analysis/slurm/run_analyses.sbatch`](analysis/slurm/run_analyses.sbatch) | ~30 min CPU-only                   |

---

## Cluster baseline assumed by every script

All `.sbatch` files use the same `#SBATCH` block defaults, derived from
the existing cluster (the old `USSam/projects/USSam3/scripts/slurm_train.sh`
documented `--partition=main` with multi-GPU nodes):

```
#SBATCH --partition=main
#SBATCH --nodes=1                 (bumped per script when DDP needs >1 node)
#SBATCH --ntasks-per-node=1       ← ONE srun task per node; torchrun spawns the per-GPU procs
#SBATCH --cpus-per-task=64-128    ← 16 CPUs per GPU; sized per --gpus-per-node
#SBATCH --gpus-per-node=4-8       ← 4 for short jobs, 8 for long pretrain
#SBATCH --mem=512G                ← NOT 0; leaves headroom on shared nodes
#SBATCH --time=…                  ← phase-specific wall limit
```

The `mem=512G` decision is deliberate (vs the old `mem=0` which meant
"all the node's memory") — leaves a margin for other tenants on the
shared cluster nodes.

The `ntasks-per-node=1` decision is also deliberate (vs the old
`ntasks-per-node=8`) — the old setup was Lightning's "one srun task per
GPU" pattern; we use `torchrun`, which manages per-GPU processes
internally from a single launcher process per node. This is the modern
PyTorch-recommended setup ([PyTorch SLURM docs](https://pytorch.org/docs/stable/elastic/run.html#deployment)).

All `.sbatch` files write per-job logs to a **relative** path
`slurm_logs/<jobname>_<JobID>[_<TaskID>].{out,err}` — resolved against
the directory where `sbatch` was invoked. The project convention is to
**invoke `sbatch` from inside the experiment dir** so logs consolidate
with the experiment they describe:

```bash
mkdir -p experiments/sonobase/pretrain/38_pt_8_bm/hiera_b_conv_s_conv_t/slurm_logs
cd      experiments/sonobase/pretrain/38_pt_8_bm/hiera_b_conv_s_conv_t
sbatch --export=ALL,EXPERIMENT=pretrain_hiera_b_conv_s_conv_t_on_38_pt_8_bm \
  <repo_root>/src/scripts/pretrain/slurm/_pretrain_one.sbatch
```

SLURM opens `--output` before the script runs, so the `slurm_logs/`
subdirectory must exist in your submit cwd *before* you call `sbatch`.

---

## Submission cheat sheet

### Phase 2 — Pretrain

```bash
# All 6 encoder variants on 38_pt_8_bm (array job, runs in parallel
# subject to GPU availability):
sbatch scripts/pretrain/slurm/pretrain_all_38_pt_8_bm.sbatch

# Just the headline TriBranchTrunk variant:
sbatch --export=ALL,EXPERIMENT=pretrain_hiera_b_conv_s_conv_t_on_38_pt_8_bm \
  scripts/pretrain/slurm/_pretrain_one.sbatch

# Multi-node (16 GPUs across 2 nodes):
sbatch --nodes=2 --export=ALL,EXPERIMENT=pretrain_hiera_b_conv_s_conv_t_on_38_pt_8_bm \
  scripts/pretrain/slurm/_pretrain_one.sbatch
```

If a 24h slot is too short (the headline TriBranchTrunk run typically
needs ~3 days on 8×A100), simply re-submit when it runs out — the
recipe's `BaseRecipe`-style DCP checkpointer auto-resumes from the
latest epoch in `experiments/<scratch.experiment_name>/`.

### Phase 3 — SonoBase test on `8_bm_7_ext`

```bash
sbatch --export=ALL,RESTORE_FROM=./experiments/sonobase/pretrain/38_pt_8_bm/hiera_b_conv_s_conv_t/LATEST \
  scripts/pretrain/slurm/test_sonobase_on_8_bm_7_ext.sbatch
```

### Phase 4 — Benchmark iterative-correction sweep

```bash
# All 6 (model × prompt) cells on 8_bm_7_ext (~2-3 h each, 6 array tasks)
sbatch scripts/benchmarks/test_sam2/iterative/slurm/b1_sweep_all_8_bm_7_ext.sbatch

# One specific cell:
sbatch --export=ALL,MODEL=sonobase,DATASET=8_bm_7_ext,PROMPT=box \
  scripts/benchmarks/test_sam2/iterative/slurm/_b1_sweep_one.sbatch
```

### Phase 5 — Analysis

```bash
# Stage 1 — predictions (1 GPU per dataset, 7 array tasks, ~1 hour each)
sbatch scripts/analysis/slurm/save_predictions_all.sbatch

# Stage 2 — CPU analyses (point + box, 2 array tasks, ~30 min each)
sbatch --dependency=afterok:<stage1_jobid> \
  scripts/analysis/slurm/run_analyses.sbatch

# Few-shot (3 datasets, ~6-8 h each on 1 GPU)
sbatch scripts/few_shot/slurm/run_all_datasets.sbatch

# Few-shot aggregator (CPU, follows the sweep)
sbatch --dependency=afterok:<sweep_jobid> \
  scripts/few_shot/slurm/aggregate_all.sbatch
```

### Phase 6 — US-RFDETR

```bash
# All 27 cells (~30 min each on 4 GPUs, 27 array tasks)
sbatch scripts/us_rfdetr/slurm/us_rfdetr_all.sbatch

# One specific cell:
sbatch --export=ALL,DATASET=acouslic,BACKBONE=sonobase \
  scripts/us_rfdetr/slurm/_us_rfdetr_one.sbatch

# Only the SonoBase backbone across all 9 datasets:
sbatch --array=2,5,8,11,14,17,20,23,26 \
  scripts/us_rfdetr/slurm/us_rfdetr_all.sbatch
```

---

## Monitoring + management

```bash
# Live job status
squeue -u $USER

# Per-job details
scontrol show job <jobid>
sacct -j <jobid> --format=JobID,JobName%30,State,Elapsed,MaxRSS,MaxVMSize

# Tail the live log (run from the same dir you invoked sbatch from)
tail -f slurm_logs/<jobname>_<jobid>.out

# Cancel
scancel <jobid>           # whole job (or all array tasks)
scancel <jobid>_<task>    # one array task
```

---

## Asset paths (env vars)

Every slurm script honours the same env-var overrides the regular shell
wrappers do, with cluster-defaulted paths:

| Var                  | Default                                                                  |
|---|---|
| `CHECKPOINT_DIR`     | *(required)* — e.g. `$WORK/Checkpoints` (SAM2/MedSAM2/DINOv3 release weights) |
| `DATASET_DIR`        | *(required)* — e.g. `$WORK/Dataset/SaUS` (preprocessed images + masks)        |
| `ANNOTATION_DIR`     | *(required)* — e.g. `$WORK/Dataset/SaUS_Annotation/38_pt_8_bm_7_ext_v2` (segmentation) |
| `DET_ANNOTATION_DIR` | *(required for Phase 6)* — e.g. `$WORK/Dataset/Ultrasound/Dense_Pred_Annotation` (detection) |
| `SONOBASE_DCP`       | `./experiments/sonobase/pretrain/busi_camus/hiera_b_conv_s_conv_t/LOWEST_VAL` (US-RFDETR sonobase init) |
| `HC18_ZIP`, `CAMUS_ZIP`, `REGPRO_ZIP`, `ACOUSLIC_CSV`, `PORCINE_SAUS` | (analysis-specific raw archives) |

Override at submission via `--export=ALL,KEY=VAL,...` or set them in
`~/.bashrc` once for permanent override.

---

## Notes on resume + idempotency

* **Pretrain** uses a DCP checkpointer; resubmitting the same
  `EXPERIMENT` continues from the latest written epoch. No `--restart`
  flag needed.
* **US-RFDETR train** does NOT mid-run resume by design (per-cell runs
  are short — ~30 min). A killed cell restarts from scratch on
  resubmission. Best-checkpoint write is atomic (`os.replace`), so a
  killed mid-write won't corrupt `best.pt`.
* **Benchmark iterative B1 sweep** is idempotent at the per-iter
  granularity: each iter writes its own `test_metrics.json`; the
  aggregator merges them at the end. Re-running a partially-done sweep
  re-does already-completed iters (no skip-if-exists logic in the
  current wrapper — fine because the runs are short).
* **Analysis Stage 1 (`save_predictions`)** has skip-if-exists logic
  built into the recipe (`is_sample_complete` + `meta.json` sentinel),
  so resubmissions are cheap if the first attempt mostly succeeded.
* **Analysis Stage 2** is fully deterministic CPU work; resubmission
  re-does everything but is cheap (~30 min).
