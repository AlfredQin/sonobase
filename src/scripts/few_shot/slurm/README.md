# Few-shot adaptation (F1) — slurm scripts

| Script | Purpose |
|---|---|
| [`_run_dataset.sbatch`](_run_dataset.sbatch) | Per-dataset full sweep (3 models × 6 N values × 3 seeds = 54 fine-tunes + 108 evals). Takes `DATASET` env var. |
| [`run_all_datasets.sbatch`](run_all_datasets.sbatch) | Array job for all 3 datasets in scope (ACOUSLIC, DDTI, FUGC). |
| [`aggregate_all.sbatch`](aggregate_all.sbatch) | CPU-only aggregator that produces the combined CSV + figures. Run after the per-dataset sweeps complete (use `--dependency=afterok:<sweep_jobid>`). |

See the [top-level slurm README](../../SLURM_README.md) for the cluster
defaults and the submit / monitor cheat sheet. Per-dataset sweeps take
~6-8 hours on 1 GPU; the aggregator is ~30 min CPU-only.
