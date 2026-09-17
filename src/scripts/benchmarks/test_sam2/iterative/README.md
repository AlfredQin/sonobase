# Iterative-correction benchmark sweeps (B1 default + B2 standalone)

This folder contains the scripts that produce per-iteration metrics for
the SAM2 / MedSAM2 / Sonobase benchmark — one curve per
`num_correction_pt_per_frame_val ∈ {0, 1, 3, 5, 7}`. The output of these
sweeps is the input to the A3 click-efficiency analysis and to any
downstream click-cost / human-correction figures.

There are two pipelines available:

| | **B1** (production default) | **B2** (standalone verification) |
|---|---|---|
| Recipe | `nemo_cv.recipes.benchmarks.test_sam2` | `nemo_cv.recipes.benchmarks.iterative_eval` |
| Config | `configs/test_sam2/test.yaml` | `configs/test_sam2/iterative_eval.yaml` |
| Encoder pass | once **per iteration** (`N` total) | once **per batch**, reused across iters |
| Process launches per (model, prompt) | `len(iterations)` | 1 |
| Wall-time scaling | linear in `N` | ~constant in `N` |
| Output schema | per-iter `test_metrics.json` × `N`, then merged via `aggregate_iterations.py` | single `test_metrics_iterations.{json,csv}` |
| RNG path | independent fresh seed each iter | single sequence advances across iters |

Both pipelines emit the **same long-format CSV schema**, so downstream
consumers (A3 analysis, summary tables) do not have to know which path
produced the table:

```
dataset,iteration,n_samples,miou,dice,test_loss
BUSI_pathology_benign,0,64,0.7831,0.8612,0.3007
BUSI_pathology_benign,1,64,0.8504,0.9112,0.1844
...
AGGREGATE_MACRO,0,...
AGGREGATE_MICRO,0,...
```

## Why B1 is the default

* **Same code path as a single-iteration test run.** Each B1 process is a
  vanilla `test_sam2.py` invocation with one extra Hydra override
  (`scratch.num_correction_pt_per_frame_val=N`). There is no risk of
  state leakage between iterations or of a subtle bug in the iteration
  loop polluting cross-iteration metrics.
* **RNG fidelity.** `test_sam2.py` opens each evaluation under
  `ScopedRNG(seed=1, ranked=True)`, so per-iteration prompt sampling
  starts from the same seed every run. With B2 the same RNG sequence
  advances across iterations within a single process, so per-sample
  prompts at iter ≥ 1 differ between B1 and B2.
* **Wall time isn't a bottleneck on the smoke set.** A single
  `test_sam2.py` BUSI+CAMUS run takes ~minutes; the 5-iter B1 sweep is
  still well under an hour per (model, prompt).

## Why B2 exists

The image encoder dominates inference cost. A B2 sweep should be
roughly `len(iterations)`× faster than B1 with no change to model
weights or math. Once we have validated that B1 ≈ B2 on the smoke
benchmark we can promote B2 to the default for production sweeps. Until
then, B2 is exercised via the verification helper below and the results
are treated as "informational only".

## Files

```
_common.sh                          shared env + helpers (model→ckpt, prompt→prob_box, …)
_run_b1_one.sh    <m> <ds> <p> <i>  invoke test_sam2.py once with iter=i
_run_b1_sweep.sh  <m> <ds> <p>      loop _run_b1_one for default iters, then aggregate
_run_b2.sh        <m> <ds> <p>      single iterative_eval.py run for the same combo
run_b1_iterations_busi_camus.sh     B1 sweep: 3 models × 2 prompts on BUSI+CAMUS
run_b1_iterations_8_bm_7_ext.sh     B1 sweep: 3 models × 2 prompts on production set
run_b2_iterations_busi_camus.sh     B2 sweep: same coverage as the B1 smoke wrapper
compare_b1_vs_b2.sh                 run both smoke sweeps then diff per-iter CSVs
```

`<m>` = `sam2_no_ft | medsam2 | sonobase`,
`<ds>` = `busi_camus | 8_bm_7_ext`,
`<p>` = `point | box`,
`<i>` = integer correction-click count (default sweep is `0 1 3 5 7`).

## Quick start

```bash
cd src/   # all scripts assume CWD = src/

# Default production iterative-correction run on the smoke set
bash scripts/benchmarks/test_sam2/iterative/run_b1_iterations_busi_camus.sh

# Same on the full 8_bm_7_ext suite (significantly longer wall time)
bash scripts/benchmarks/test_sam2/iterative/run_b1_iterations_8_bm_7_ext.sh

# Validate that the encoder-cached B2 path agrees with B1 on the smoke set
bash scripts/benchmarks/test_sam2/iterative/compare_b1_vs_b2.sh
```

After the sweep finishes, the merged CSV per `(model, prompt)` lives at:

```
./experiments/benchmarks/<dataset>/<model_dir>/b1_<prompt>/test_metrics_iterations.csv
./experiments/benchmarks/<dataset>/<model_dir>/b2_<prompt>/results/test_metrics_iterations.csv
```

where `<model_dir>` is `sam2`, `medsam2`, or `hiera_b_conv_s_conv_t` for
the three models respectively.

## Output layout

```
experiments/benchmarks/<dataset>/<model_dir>/
├── b1_point/
│   ├── iter0/results/test_metrics.{json,csv}        # B1 per-iter run dirs
│   ├── iter1/results/test_metrics.{json,csv}
│   ├── iter3/...
│   ├── iter5/...
│   ├── iter7/...
│   └── test_metrics_iterations.{json,csv}           # B1 merged artifact
├── b1_box/   (same layout)
├── b2_point/
│   └── results/test_metrics_iterations.{json,csv}   # B2 single-pass artifact
└── b2_box/   (same layout)
```

## Customising the iteration set

The default `[0, 1, 3, 5, 7]` ladder is hard-coded as
`DEFAULT_ITERATIONS=(0 1 3 5 7)` in `_common.sh`. To override on a
per-call basis pass extra positional arguments after the prompt:

```bash
bash _run_b1_sweep.sh sam2_no_ft busi_camus point 0 2 5 10
bash _run_b2.sh       sam2_no_ft busi_camus point 0 2 5 10
```

`_run_b2.sh` translates the bash array into Hydra's list literal
(`iterations=[0,2,5,10]`) before invoking the recipe.

## RNG / numerical caveats (verification only)

When comparing B1 and B2 outputs, expect per-sample mask differences
due to the RNG-path divergence noted in the table at the top. Aggregate
metrics over a few hundred samples should agree to within a small
fraction of a point of mIoU / Dice; `compare_b1_vs_b2.sh` flags any
(`dataset`, `iteration`, `metric`) triple that violates both an
absolute and a relative tolerance (defaults `0.02` and `5%`).
Tolerances can be tightened via the `ABS_TOL` / `REL_TOL` env vars.
