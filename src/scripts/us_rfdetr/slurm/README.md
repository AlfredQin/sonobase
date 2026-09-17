# US-RFDETR — slurm scripts

| Script | Purpose |
|---|---|
| [`_us_rfdetr_one.sbatch`](_us_rfdetr_one.sbatch) | Generic single-(dataset × backbone) cell launcher. Takes `DATASET` + `BACKBONE` env vars. Handles lazy DCP→PT conversion for sonobase via `_resolve_ckpt.sh`. |
| [`us_rfdetr_all.sbatch`](us_rfdetr_all.sbatch) | Array job for all 27 cells (9 datasets × 3 backbones). Array index decomposes deterministically: `dataset_idx = id // 3`, `backbone_idx = id % 3`. |

See the [top-level slurm README](../../SLURM_README.md) for the cluster
defaults and the submit / monitor cheat sheet. Each cell takes ~30 min
on 4 GPUs.
