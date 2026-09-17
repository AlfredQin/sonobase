# Hydra config tree — few-shot

```
src/configs/few_shot/
├── finetune.yaml                      top-level Hydra config
├── README.md
├── scratch/default.yaml               project-wide variables
├── data/
│   ├── ACOUSLIC.yaml                  video dataset (default training_unit=video)
│   ├── DDTI.yaml                      image dataset
│   └── FUGC.yaml                      image dataset
├── loss/
│   └── sam2.yaml                      MultiStepMultiMasksAndIous (same as pretrain)
├── model/                             copy of test_sam2/model/ + sam2.yaml
│   ├── default.yaml
│   ├── sam2.yaml
│   ├── image_encoder/
│   │   ├── hiera_b+.yaml              for sam2-no-ft
│   │   ├── hiera_t.yaml               for medsam2
│   │   └── hiera_b_conv_s_conv_t.yaml for sonobase
│   ├── memory_attention/default.yaml
│   └── memory_encoder/default.yaml
└── experiment/
    ├── sonobase.yaml                  model-only overlay
    ├── medsam2.yaml
    └── sam2_no_ft.yaml
```

The `model/` and `loss/` subtrees are direct copies of their pretrain
counterparts. The few-shot recipe explicitly relies on the **identical
loss + forward path** as pretrain, so we want zero divergence.

---

## Composition flow

For `_finetune_one.sh ACOUSLIC sonobase 5 42`:

```
finetune.yaml   (top-level)
├── scratch:    default                project-wide vars (paths, transforms, seeds)
├── model:      sam2                   SAM2Train + 3 sub-components
│       └── image_encoder@model.image_encoder: hiera_b_conv_s_conv_t
│           (overridden by experiment overlay)
├── loss@loss_fn: sam2                 MultiStepMultiMasksAndIous keyed on "all"
├── data:       ACOUSLIC               TorchTrainMixedDataset wrapping JSONRawDataset
└── experiment: sonobase               @_global_ overlay; sets model_label + image_encoder
```

CLI overrides set: `ckpt_path`, `scratch.dataset_name`, `scratch.N`,
`scratch.fewshot_seed`. From those, `scratch.experiment_name` and
`scratch.subset_basename` are auto-resolved via interpolation.

---

## Required CLI overrides

`finetune.yaml` deliberately leaves these as `???`:

| Key                          | Set by                                     |
|---|---|
| `ckpt_path`                  | `_finetune_one.sh` (per-model resolution)  |
| `scratch.dataset_name`       | `_finetune_one.sh` (= the `data=<...>` choice) |
| `scratch.model_label`        | `experiment=<model>` overlay               |
| `scratch.N`                  | `_finetune_one.sh`                         |
| `scratch.fewshot_seed`       | `_finetune_one.sh`                         |

---

## ACOUSLIC training_unit

`scratch.training_unit: video` (default) — each training example is one
ACOUSLIC video, fed through SAM2Train as an 8-frame clip per training
step (the same `RandomUniformSampler` pretrain uses).

`scratch.training_unit: frame` (reserved) — each training example is one
annotated frame. The recipe currently raises `NotImplementedError` for
this code path; implementation is a follow-up that needs an
annotated-frame pre-extraction pass to produce the SaUS image-format
layout.

---

## Compose-check (after editing any YAML)

```bash
cd src && uv run python -c "
import pathlib
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

cd = str(pathlib.Path('./configs/few_shot').resolve())
GlobalHydra.instance().clear()
initialize_config_dir(config_dir=cd, version_base=None)
cfg = compose(config_name='finetune', overrides=[
    'data=ACOUSLIC', 'experiment=sonobase',
    'ckpt_path=/tmp/x.pt',
    'scratch.dataset_name=ACOUSLIC', 'scratch.N=5', 'scratch.fewshot_seed=42',
])
OmegaConf.resolve(cfg)
print(cfg.scratch.experiment_name, cfg.scratch.subset_basename, cfg.data.kind)
"
```

Expect output:
`ACOUSLIC_sonobase_N5_seed42 ACOUSLIC_N5_seed42 video`
