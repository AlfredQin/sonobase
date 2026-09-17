# Benchmark iterative-correction sweep — slurm scripts

B1 (production-default) iterative-correction sweep on slurm. Each
(model × prompt) cell loops `test_sam2.py` over `iterations=[0,1,3,5,7]`
and aggregates the per-iter JSONs into a long-format CSV.

| Script | Purpose |
|---|---|
| [`_b1_sweep_one.sbatch`](_b1_sweep_one.sbatch) | Generic single-(model × dataset × prompt) cell launcher. Takes `MODEL`, `DATASET`, `PROMPT` env vars. |
| [`b1_sweep_all_8_bm_7_ext.sbatch`](b1_sweep_all_8_bm_7_ext.sbatch) | Array job for all 6 cells (3 models × 2 prompts) on the production `8_bm_7_ext` suite. |

See the [top-level slurm README](../../../../SLURM_README.md) for the
cluster defaults and the submit / monitor cheat sheet. Each cell takes
~2-3 hours on 4 GPUs (15 datasets × 5 iterations).
