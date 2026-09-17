# SAM2 / MedSAM2 / Sonobase benchmark scripts

This directory holds the **shell entry points** for the benchmark comparison
between three SAM2-architecture models on the same evaluation datasets:

| Model | Image encoder | Pretraining | Where the checkpoint lives |
|---|---|---|---|
| **sam2-no-ft** | Hiera-B+ | Meta's SA-V (general video object segmentation) — no medical fine-tune | `${CHECKPOINT_DIR}/SAM2/sam2.1_hiera_base_plus.pt` |
| **medsam2** | Hiera-Tiny | SAM2-Tiny + medical fine-tune by Bowang Lab | `${CHECKPOINT_DIR}/MedSAM2/MedSAM2_latest.pt` |
| **sonobase** | TriBranchTrunk (Hiera-B + ConvNeXt-S + ConvNeXt-T) | SAM2 init + sonobase pretraining on ultrasound | `experiments/sonobase/pretrain/<run>/<epoch_N_step_M>/` (DCP format) |

Read this first if you only want to run the benchmark.

For *what the configs do*, see [`src/configs/test_sam2/README.md`](../../../configs/test_sam2/README.md).
For *how the recipe code works*, see
[`src/nemo_cv/recipes/benchmarks/README.md`](../../../nemo_cv/recipes/benchmarks/README.md).

---

## Contents

```
src/scripts/benchmarks/test_sam2/
├── test_sam2_no_ft_on_busi_camus.sh        Smoke-test SAM2 (no-ft) on BUSI + CAMUS
├── test_medsam2_on_busi_camus.sh           Smoke-test MedSAM2 on BUSI + CAMUS
├── test_sonobase_on_busi_camus.sh          Convert sonobase DCP -> .pt, then test
├── run_all_busi_camus.sh                   Run all three sequentially + aggregate
└── iterative/                              Iterative-correction sweeps (B1 default + B2 verification)
    ├── README.md                           B1 vs B2 walkthrough, output schema, RNG caveats
    ├── _common.sh                          shared env + model/prompt/checkpoint helpers
    ├── _run_b1_one.sh                      single test_sam2.py call with iter=N (B1 inner loop)
    ├── _run_b1_sweep.sh                    B1 sweep for one (model, dataset, prompt) combo
    ├── _run_b2.sh                          B2 single-pass standalone for the same combo
    ├── run_b1_iterations_busi_camus.sh     B1 production sweep on the smoke set
    ├── run_b1_iterations_8_bm_7_ext.sh     B1 production sweep on the full 15-dataset suite
    ├── run_b2_iterations_busi_camus.sh     B2 verification sweep on the smoke set
    └── compare_b1_vs_b2.sh                 run both + per-(dataset, iteration, metric) diff
```

### Iterative corrections sweep (B1 default)

The single-iteration scripts above evaluate models at one fixed
correction-click count (defaulting to 0 — no per-frame iterative
correction). To produce the per-iteration metrics consumed by the A3
click-efficiency analysis, use the wrappers under `iterative/`:

```bash
cd src/

# Smoke set (3 models × 2 prompts × 5 iters on BUSI + CAMUS)
bash scripts/benchmarks/test_sam2/iterative/run_b1_iterations_busi_camus.sh

# Full suite (3 models × 2 prompts × 5 iters on the 8 benchmark + 7 external datasets)
bash scripts/benchmarks/test_sam2/iterative/run_b1_iterations_8_bm_7_ext.sh
```

Each sweep produces, per `(model, prompt)`, a merged long-format
`test_metrics_iterations.csv` at
`./experiments/benchmarks/<dataset>/<model_dir>/b1_<prompt>/`. See
[`iterative/README.md`](iterative/README.md) for the full B1 vs B2
contrast and validation workflow.

The "smoke-test" wording is deliberate: as of this writing, sonobase has only
been pretrained on the small `busi_camus` dataset (~750 videos) using
`multiplier: 0.02` debug-scale subsampling. These scripts validate that the
benchmarking *pipeline* is correct; they don't produce publication-quality
numbers. The real benchmark (15 datasets in `8_bm_7_ext`) will run after
sonobase is trained on the full `38_pt_8_bm` dataset.

---

## TL;DR — run the comparison

```bash
cd src                                                         # IMPORTANT
bash scripts/benchmarks/test_sam2/run_all_busi_camus.sh
```

This runs all three models in sequence and writes a comparison CSV to
`experiments/benchmarks/busi_camus/comparison.csv`. Total wall-clock on
4×A6000 is ~3 minutes for the smoke test (~25 minutes for `8_bm_7_ext` once
the experiment yamls for that exist for all three models).

---

## What each individual script does

### `test_sam2_no_ft_on_busi_camus.sh`

Uses the canonical SAM2 release. Output goes to
`experiments/benchmarks/busi_camus/sam2/p0/`.

```bash
uv run torchrun --nproc_per_node=4 -m nemo_cv.recipes.benchmarks.test_sam2 \
  -c ./configs/test_sam2 -cn test \
  experiment=test_sam2_no_ft_on_busi_camus \
  ckpt_path=${CHECKPOINT_DIR}/SAM2/sam2.1_hiera_base_plus.pt
```

The experiment overlay (`test_sam2_no_ft_on_busi_camus.yaml`) selects the
`hiera_b+` image encoder so the model architecture matches the released
weights.

### `test_medsam2_on_busi_camus.sh`

Same recipe, different image encoder + checkpoint. Output:
`experiments/benchmarks/busi_camus/medsam2/p0/`.

```bash
uv run torchrun --nproc_per_node=4 -m nemo_cv.recipes.benchmarks.test_sam2 \
  -c ./configs/test_sam2 -cn test \
  experiment=test_medsam2_on_busi_camus \
  ckpt_path=${CHECKPOINT_DIR}/MedSAM2/MedSAM2_latest.pt
```

The experiment overlay selects the `hiera_t` image encoder. (MedSAM2 is a
fine-tune of SAM2-Tiny — confirmed by checkpoint inspection: 471 keys total,
identical to `sam2.1_hiera_tiny.pt`.)

### `test_sonobase_on_busi_camus.sh`

Two steps:

1. **Resolve** the sonobase checkpoint — the script delegates to the shared
   `scripts/_resolve_ckpt.sh`, which lazily converts the DCP-format pretrain
   checkpoint to a single `.pt` (cached). It defaults to the busi_camus
   pretrain's best-validation epoch (`LOWEST_VAL`).
2. **Run** the test recipe pointing at that `.pt`.

```bash
PT_FILE="$(SONOBASE_DCP="./experiments/sonobase/pretrain/busi_camus/hiera_b_conv_s_conv_t/LOWEST_VAL" \
  bash scripts/_resolve_ckpt.sh sonobase)"

uv run torchrun --nproc_per_node=4 -m nemo_cv.recipes.benchmarks.test_sam2 \
  -c ./configs/test_sam2 -cn test \
  experiment=test_hiera_b_conv_s_conv_t_on_busi_camus \
  ckpt_path="$PT_FILE"
```

**Why a separate conversion step?** Sonobase pretraining writes DCP-sharded
directories (one `.distcp` shard per DDP rank under `<ckpt>/model/`). SAM2 and
MedSAM2 are released as plain `.pt` files. Rather than have the test recipe
support both formats, we convert sonobase to `.pt` once, then all three
models load through identical code. Output is reusable for any future test
runs from the same source checkpoint. See
[`src/nemo_cv/recipes/benchmarks/README.md`](../../../nemo_cv/recipes/benchmarks/README.md#converter)
for the converter's design.

After a new pretrain run the symlink advances to a new epoch dir, and the
resolver reconverts automatically — the `.pt` cache is keyed on the resolved
epoch (`epoch_N_step_M.pt`), so no manual cleanup is needed.

### `run_all_busi_camus.sh`

Runs the three above in order, then invokes the comparison aggregator
(`compare_benchmarks.py`) to produce a single CSV summarizing all three
models. Output: `experiments/benchmarks/busi_camus/comparison.csv` plus a
human-readable table printed to stdout.

---

## Output: how to read it

Each model run produces (under
`experiments/benchmarks/busi_camus/<model>/p0/`):

| File | Purpose |
|---|---|
| `results/test_metrics.json` | Full per-dataset record (mIoU, Dice, optional test_loss, sample counts) plus macro/micro aggregates over datasets, plus the absolute checkpoint path |
| `results/test_metrics.csv` | Same data as a flat CSV (one row per dataset + macro/micro aggregate rows) |
| `test.jsonl` | One JSON line per dataset, parity with training/validation jsonl |

Then `compare_benchmarks.py` aggregates all three `test_metrics.json` into a
single table:

```
dataset         | metric    | sam2_no_ft | medsam2 | sonobase
----------------+-----------+------------+---------+---------
BUSI            | miou      | 0.4237     | 0.7874  | 0.7961
BUSI            | dice      | 0.5236     | 0.8695  | 0.8806
BUSI            | test_loss | 1.0597     | 0.3020  | 0.2960
CAMUS           | miou      | 0.1636     | 0.8872  | 0.8044
…
AGGREGATE_MACRO | miou      | 0.2936     | 0.8373  | 0.8002
AGGREGATE_MICRO | miou      | 0.2651     | 0.8483  | 0.8011
```

`AGGREGATE_MACRO` is the unweighted mean across datasets;
`AGGREGATE_MICRO` weights by sample count.

---

## Common operations

### Re-test against a different sonobase checkpoint

By default the sonobase script resolves
`experiments/sonobase/pretrain/busi_camus/hiera_b_conv_s_conv_t/LOWEST_VAL`
(the best-validation epoch). To use a different checkpoint, export
`SONOBASE_DCP` (a pretrain run dir / DCP dir / symlink) or `CKPT_PATH` (an
already-converted `.pt`) before running — no need to edit the script:

```bash
cd src
SONOBASE_DCP=./experiments/.../epoch_N_step_M \
  bash scripts/benchmarks/test_sam2/test_sonobase_on_busi_camus.sh
```

### Run on the full benchmark (`8_bm_7_ext` — 15 datasets)

Three experiment yamls already exist for `8_bm_7_ext`
(`experiment/test_<model>_on_8_bm_7_ext.yaml`). To run those, just swap the
experiment name in the script:

```bash
# inside the .sh file
experiment=test_sam2_no_ft_on_8_bm_7_ext   # (or _medsam2_, or _hiera_b_conv_s_conv_t_)
```

But keep in mind: sonobase needs a checkpoint trained on `38_pt_8_bm`
(currently doesn't exist) before the comparison is meaningful.

### Use point or box prompts

Two scratch knobs in `src/configs/test_sam2/scratch/default.yaml` control
prompt input during evaluation:

- `num_correction_pt_per_frame_val`: 0 = no per-frame iterative correction
  (the model still receives the initial point prompt on the first frame
  because `prob_to_use_pt_input_for_eval: 1.0` in `model/default.yaml`).
  1–7 = N additional correction points per frame, sampled from the GT mask.
- `prob_to_use_box_input_for_eval`: 0 = no box prompts (current default;
  the model uses the point prompt only). 1 = always provide a
  bounding-box prompt derived from the GT mask instead of points.

Override at the CLI: `+scratch.num_correction_pt_per_frame_val=3`. SAM2 was
designed to consume prompts; in our default eval setup it receives a single
initial point prompt with no iterative refinement. For a fair comparison
protocol, decide on one prompting setting and apply it to all three models.

To produce the *whole curve* (`num_correction_pt_per_frame_val ∈
{0, 1, 3, 5, 7}`) for a given prompt type in one go, use the wrappers
under [`iterative/`](iterative/) instead of editing the scratch knob by
hand. Those wrappers vary the iteration count via `_run_b1_sweep.sh` and
emit a single long-format CSV.

### Converted-checkpoint caching

`scripts/_resolve_ckpt.sh` caches the converted `.pt` next to the source,
keyed on the **resolved epoch dir** (`epoch_N_step_M.pt`). A new pretrain run
advances the `LOWEST_VAL` / `LATEST` symlink to a new epoch, so the next
resolve produces a fresh `.pt` automatically — no manual cleanup needed.

---

## Lessons learned (read before debugging)

These are gotchas the team has hit while developing this code.

### 1. SAM2 release `.pt` files contain a `no_mem_pos_enc` parameter that our model no longer registers

Loading SAM2's `sam2.1_hiera_base_plus.pt` with `strict=False` reports
*1 unexpected key: `no_mem_pos_enc`*. This is **expected and harmless** — see
the [pretrain "Lessons Learned"](../../pretrain/README.md#lessons-learned)
for the full story. The recipe uses `strict=False` so the parameter is just
silently dropped. The same applies to MedSAM2 and to converted sonobase
checkpoints.

### 2. The converter must produce a `weights_only=True`-compatible `.pt`

PyTorch 2.10's `torch.load(weights_only=True)` rejects pickled
`TorchVersion` objects (a str subclass). The converter explicitly casts the
torch version string to `str` before saving. If you ever extend the
provenance dict with new fields, keep them all to plain `str`/`int`/`float`
or run `torch.load(weights_only=False)` (less safe).

### 3. The default eval protocol uses an initial point prompt but no iterative correction

`num_correction_pt_per_frame_val: 0` and `prob_to_use_box_input_for_eval: 0`
in `scratch/default.yaml`, combined with `prob_to_use_pt_input_for_eval: 1.0`
in `model/default.yaml`, mean: a single point prompt is sampled from the GT
mask on the first conditioning frame, no box prompts, no per-frame iterative
correction. The smoke-test numbers *are* directly comparable across models
because all three see the same input — but this is NOT the canonical
"iterative SAM2 evaluation" benchmark you'd see in a paper, which typically
uses 5–8 correction points per frame (for point-prompted) or a single box
prompt per object (for box-prompted).

To switch protocols at the CLI:

- `+scratch.num_correction_pt_per_frame_val=N` — N correction points per frame
- `+scratch.prob_to_use_box_input_for_eval=1` — box-prompted eval
- `+model.prob_to_use_pt_input_for_eval=0` — disable the initial point prompt
  too (rare; produces pure auto-segmentation, which SAM2 is not strictly
  designed for and which will tank scores)

Fix the protocol before running on the full `8_bm_7_ext` set so numbers
are comparable to published baselines.

### 4. Image size is 1024 — matches Meta's SAM2 release and SonoBase pretraining

The scratch config sets `resolution: 1024`. This matches both Meta's
published SAM2 benchmarks and SonoBase's pretraining resolution, so the
SAM2-no-ft / MedSAM2 / SonoBase three-way comparison is apples-to-apples
on positional embeddings. (Earlier iterations used 512 to save memory.)

### 5. Different seeds per rank can re-shuffle small test sets in surprising ways

The recipe wraps each test loop in `ScopedRNG(seed=1, ranked=True)`. With
small test sets (BUSI has 64 samples, distributed across 4 ranks → 16/rank,
batched as 8 → 2 batches per rank), per-rank ordering can affect timing but
should not affect metrics (torchmetrics is order-invariant). If you see
metric drift across re-runs, suspect (a) non-deterministic CUDA kernels or
(b) cudnn benchmark mode rather than the seed.

---

## See also

- [`src/configs/test_sam2/README.md`](../../../configs/test_sam2/README.md) —
  Config tree, what each YAML does, how to add another model variant.
- [`src/nemo_cv/recipes/benchmarks/README.md`](../../../nemo_cv/recipes/benchmarks/README.md) —
  Recipe code walkthrough, converter design, comparison-aggregator structure.
- [`src/scripts/pretrain/README.md`](../../pretrain/README.md) —
  How sonobase pretraining works (the source of the DCP checkpoints
  consumed here).
