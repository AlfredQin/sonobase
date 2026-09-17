# Pretrain — slurm scripts

| Script | Purpose |
|---|---|
| [`_pretrain_one.sbatch`](_pretrain_one.sbatch) | Generic single-experiment launcher; takes `EXPERIMENT` env var + optional `EXTRA_OVERRIDES` / `GLOBAL_BATCH_SIZE`. DDP-aware; bump `--nodes=N` for multi-node runs. |
| [`pretrain_all_38_pt_8_bm.sbatch`](pretrain_all_38_pt_8_bm.sbatch) | Array job for all 6 production encoder variants on `38_pt_8_bm`. Each task sources `_pretrain_one.sbatch` with the right `EXPERIMENT` and per-encoder `scratch.ckpt_path` override. |
| [`test_sonobase_on_8_bm_7_ext.sbatch`](test_sonobase_on_8_bm_7_ext.sbatch) | Per-dataset SonoBase test on the 15-dataset held-out split (Phase 3). Takes `RESTORE_FROM` env var. |

See the [top-level slurm README](../../SLURM_README.md) for the cluster
defaults (`ntasks-per-node=1`, `mem=512G`, `--partition=main`) and the
submit / resume / monitor cheat sheet.

Resubmission resumes from the latest DCP checkpoint automatically — no
`--restart` flag.
