# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import re

import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from .build_sam import _load_checkpoint, _get_sam2_configs_dir, _compose_hydra_config


_SAM2_TARGET_PREFIX = "sam2."
_LOCAL_TARGET_PREFIX = "models.segmentation.interactive.sam2."


def _remap_sam2_targets(cfg):
    """
    Recursively remap ``_target_`` values from the upstream ``sam2.xxx``
    module paths to the local ``models.segmentation.interactive.sam2.xxx``
    paths so that Hydra ``instantiate`` can locate the classes.
    """
    if isinstance(cfg, dict):
        for k, v in cfg.items():
            if k == "_target_" and isinstance(v, str) and v.startswith(_SAM2_TARGET_PREFIX):
                cfg[k] = _LOCAL_TARGET_PREFIX + v[len(_SAM2_TARGET_PREFIX):]
            else:
                _remap_sam2_targets(v)
    elif isinstance(cfg, (list, tuple)):
        for item in cfg:
            _remap_sam2_targets(item)


def build_sam2(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    **kwargs,
):

    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        ]
    # Read config and init model
    cfg = _compose_hydra_config(config_file, overrides=hydra_overrides_extra)
    OmegaConf.resolve(cfg)
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    _remap_sam2_targets(model_cfg)
    model = instantiate(model_cfg, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def build_sam2_video_predictor(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    vos_optimized=False,
    **kwargs,
):
    hydra_overrides = [
        "++model._target_=models.segmentation.interactive.sam2.sam2_video_predictor.SAM2VideoPredictor",
    ]
    if vos_optimized:
        hydra_overrides = [
            "++model._target_=models.segmentation.interactive.sam2.sam2_video_predictor.SAM2VideoPredictorVOS",
            "++model.compile_image_encoder=True",  # Let sam2_base handle this
        ]

    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            # the sigmoid mask logits on interacted frames with clicks in the memory encoder so that the encoded masks are exactly as what users see from clicking
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            # fill small holes in the low-res masks up to `fill_hole_area` (before resizing them to the original video resolution)
            "++model.fill_hole_area=8",
        ]
    hydra_overrides.extend(hydra_overrides_extra)

    # Read config and init model
    cfg = _compose_hydra_config(config_file, overrides=hydra_overrides)
    OmegaConf.resolve(cfg)
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    _remap_sam2_targets(model_cfg)
    model = instantiate(model_cfg, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model

