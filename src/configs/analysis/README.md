# Hydra config tree — analysis

Hydra config root for the **analysis suite**. Stage-1 (the prediction-saving
recipe) lives here under `save_predictions.yaml`. Stage-2 analyses are
plain CPU-only Python scripts driven by `argparse` (no Hydra config —
they accept `--runs <label>=<path>` style entries from the runner shell
scripts; this keeps post-hoc analyses trivially composable across
different Stage-1 outputs).

For the code architecture and how to add an analysis, see
`nemo_cv/recipes/analysis/README.md`.

---

## Tree

```
src/configs/analysis/
├── save_predictions.yaml               top-level Stage-1 config
├── README.md                           this file
├── scratch/
│   └── default.yaml                    project-wide variables (paths, seeds, AMP knobs, raw-zip locations)
├── data/
│   ├── HC18.yaml                       SaUS image dataset
│   ├── BUSI.yaml                       SaUS image dataset (canonical 131-sample test split; subgroup S3)
│   ├── KidneyUS.yaml                   SaUS image dataset (canonical 419-sample test split; subgroup S1)
│   ├── PorcineSpinalCord.yaml          SaUS image dataset
│   ├── CAMUS.yaml                      SaUS video dataset (3 cardiac structures)
│   ├── ACOUSLIC.yaml                   SaUS video dataset (fetal abdomen sweep videos)
│   └── RegPro.yaml                     SaUS 3D dataset (per-slice 2.5D)
├── model/                              copy of test_sam2/model/ (kept self-contained)
│   ├── default.yaml
│   ├── sam2.yaml
│   ├── image_encoder/
│   │   ├── hiera_b+.yaml               for sam2-no-ft
│   │   ├── hiera_t.yaml                for medsam2
│   │   └── hiera_b_conv_s_conv_t.yaml  for sonobase (TriBranchTrunk)
│   ├── memory_attention/default.yaml
│   └── memory_encoder/default.yaml
└── experiment/                         model-only @_global_ overlays (encoder + model_label)
    ├── sam2_no_ft.yaml
    ├── medsam2.yaml
    └── sonobase.yaml
```

The `model/` subtree is **structurally identical** to
`src/configs/test_sam2/model/`. It's copied here (rather than symlinked)
so the analysis tree is self-contained and `cd src && uv run python -m
nemo_cv.recipes.analysis.save_predictions -c ./configs/analysis -cn …`
works without any extra path setup.

---

## Composition flow

Hydra defaults compose top-down. Every Stage-1 run goes through the
`_save_one.sh <model> <dataset> <prompt>` helper — e.g. `_save_one.sh sonobase HC18 point`:

```
save_predictions.yaml   (top-level)
├── scratch:    default              project-wide vars
├── model:      sam2                 SAM2Train + 3 sub-components
├── data:       HC18                 (via `data=HC18` CLI override)
└── experiment: sonobase             (via `experiment=sonobase`; model-only @_global_
                                       overlay — sets the image_encoder + model_label)
```

`_save_one.sh` additionally passes `prompt_protocol.type`, `run_name`, and
`output_dir` as plain CLI overrides. The three experiment overlays
(`sam2_no_ft`, `medsam2`, `sonobase`) differ only in the image-encoder
sub-tree — they carry no dataset- or prompt-specific state.

---

## Required CLI overrides

`save_predictions.yaml` deliberately leaves three keys as `???` (Hydra
"missing required value"). Every Stage-1 invocation must set them:

| Key             | What it is |
|---|---|
| `ckpt_path`     | Absolute path to the `.pt` checkpoint to evaluate. For sonobase, the *converted* file (DCP → PT). |
| `output_dir`    | Where the per-sample artifacts go. One directory per (model × dataset × prompt) run. |
| `run_name`      | Free-form identifier embedded into `manifest.json`. |
| `model_label`   | Short label saved into `per_sample_metrics.csv` (used as the legend / column key in Stage-2). |

The runner shell scripts under `src/scripts/analysis/save_predictions/`
fill these in.

---

## Other knobs you might override

| Key                                       | Default       | Why you'd change it |
|---|---|---|
| `prompt_protocol.type`                    | `point`       | switch to `box` (uses GT bounding box as initial prompt) |
| `prompt_protocol.num_correction_clicks`   | `0`           | `1..7` — single-shot prediction with that many corrections accumulated |
| `prompt_protocol.iterations`              | (unset)       | list of integers e.g. `[0,1,3,5,7]` — A3 mode: snapshot the prediction at each iter (saves N+1 masks per object, emits N+1 CSV rows with `iteration` populated) |
| `prompt_protocol.prompt_frame`            | `0`           | (video runs only) which frame to issue the initial prompt on; falls back to first frame with a non-empty GT for that object |
| `save_overlays`                           | `true` (image), `false` (video) | toggle overlay PNG generation |
| `max_samples`                             | `null`        | small int for fast smoke tests |
| `optim.amp.amp_dtype`                     | `bfloat16`    | `float16` on Volta-class GPUs that lack BF16 |
| `cuda.cudnn_deterministic`                | `true`        | `false` for ~5–10 % speedup at the cost of bit-reproducibility |
| `logging.log_freq`                        | `20` (image), `5` (video) | log progress every N samples |

---

## Adding new datasets / experiments

For a new image dataset (e.g. BrEaST, BUS-BRA): drop a
`data/<NAME>.yaml` with the 6-key shape from `HC18.yaml` and use the
`_save_one.sh` helper to launch.

For a new video dataset (e.g. EchoNet-Dynamic): same shape but with
`kind: video` and the GT JSON suffix (`gt_suffix`, `gt_ext`) matching the
SaUS conversion's filename convention.

For a new model variant: add an image_encoder yaml under
`model/image_encoder/` and either an experiment overlay or extend
`_save_one.sh`'s `case` block.

---

## Compose-check (after editing any YAML in this tree)

Useful sanity check:

```bash
cd src && uv run python -c "
import pathlib
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

cd = str(pathlib.Path('./configs/analysis').resolve())
GlobalHydra.instance().clear()
initialize_config_dir(config_dir=cd, version_base=None)
cfg = compose(config_name='save_predictions', overrides=[
    'data=CAMUS', '+model/image_encoder@model.image_encoder=hiera_b+',
    'ckpt_path=/tmp/x.pt', 'output_dir=/tmp/out', 'run_name=test',
    'model_label=test', '~experiment',
])
OmegaConf.resolve(cfg)
print(cfg.data.kind, OmegaConf.select(cfg, 'model.image_encoder.trunk._target_'))
"
```

This catches YAML mistakes (missing keys, broken interpolations) before
you ever burn GPU time.
