# Analysis — slurm scripts

Two-stage architecture (matches the rest of the analysis pipeline):

| Script | Stage | Resources | Purpose |
|---|---|---|---|
| [`_save_predictions_one.sbatch`](_save_predictions_one.sbatch) | Stage 1 | 1 GPU | Per-dataset Stage-1 prediction save; takes `DATASET` env var. Wraps `save_all_<DATASET>.sh`. |
| [`save_predictions_all.sbatch`](save_predictions_all.sbatch) | Stage 1 | 1 GPU × 7 array tasks | Array job for all 7 datasets in scope (HC18, CAMUS, ACOUSLIC, RegPro, PorcineSpinalCord, BUSI, KidneyUS). |
| [`run_analyses.sbatch`](run_analyses.sbatch) | Stage 2 | CPU only | Array job for the 2 prompt protocols (point + box). Wraps `run_all_analyses_<prompt>.sh`. Run after Stage 1 with `--dependency=afterok:<stage1_jobid>`. |

See the [top-level slurm README](../../SLURM_README.md) for the cluster
defaults and the submit / monitor cheat sheet. Stage 1 takes ~30-60 min
per dataset on 1 GPU; Stage 2 is ~30 min CPU-only.

Stage 1 has skip-if-exists logic (`is_sample_complete` + `meta.json`
sentinel) so resubmissions of partially-completed jobs are cheap.
