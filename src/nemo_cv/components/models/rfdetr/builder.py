"""US-RFDETR model builder.

Wires together the SAM2 backbone, RF-DETR transformer, segmentation
head (optional), criterion and post-processor into a single dict
returned by :func:`build_rfdetr`. The dict is what the recipe class
unpacks during ``setup()``.

Ported from ``USSam/projects/DenseUS/dense_models/rfdetr_builder.py``
with import-path edits and a clarifying docstring update; the
hyperparameter defaults match the DenseUS reference run that produced
the published numbers.
"""

from __future__ import annotations

import logging
from argparse import Namespace
from typing import Dict, List, Optional

import torch.nn as nn

from nemo_cv.components.models.rfdetr.lwdetr import (
    LWDETR,
    PostProcess,
    SetCriterion,
)
from nemo_cv.components.models.rfdetr.matcher import build_matcher
from nemo_cv.components.models.rfdetr.segmentation_head import (
    SegmentationHead,
)
from nemo_cv.components.models.rfdetr.transformer import (
    build_transformer,
)
from nemo_cv.components.models.rfdetr.sam2_backbone import (
    build_sam2_rfdetr_backbone,
)

logger = logging.getLogger(__name__)


def _make_args(**overrides) -> Namespace:
    """Build a minimal ``Namespace`` carrying RF-DETR's expected attributes.

    The original RF-DETR codebase wires its sub-modules via a single
    ``argparse.Namespace`` (a pre-Hydra design). We honour that here so
    we can reuse RF-DETR's transformer / matcher / criterion builders
    unchanged.
    """
    defaults = dict(
        # Transformer
        hidden_dim=256,
        sa_nheads=8,
        ca_nheads=16,
        dec_layers=3,
        dec_n_points=2,
        dropout=0.0,
        dim_feedforward=2048,
        num_queries=300,
        num_select=300,
        group_detr=13,
        two_stage=True,
        num_feature_levels=3,
        lite_refpoint_refine=True,
        bbox_reparam=True,
        decoder_norm="LN",
        # Loss
        num_classes=1,
        focal_alpha=0.25,
        cls_loss_coef=1.0,
        bbox_loss_coef=5.0,
        giou_loss_coef=2.0,
        aux_loss=True,
        use_varifocal_loss=False,
        use_position_supervised_loss=False,
        ia_bce_loss=True,
        set_cost_class=2.0,
        set_cost_bbox=5.0,
        set_cost_giou=2.0,
        # Segmentation
        segmentation_head=False,
        mask_downsample_ratio=4,
        mask_ce_loss_coef=5.0,
        mask_dice_loss_coef=5.0,
        mask_point_sample_ratio=16,
        # Device hint (not used directly here)
        device="cuda",
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def build_rfdetr(
    image_encoder: nn.Module,
    num_classes: int = 1,
    # ----- Backbone -----
    pretrained_ckpt: Optional[str] = None,
    frozen_backbone: bool = False,
    use_feature_levels: Optional[List[int]] = None,
    # ----- Transformer -----
    hidden_dim: int = 256,
    dec_layers: int = 3,
    sa_nheads: int = 8,
    ca_nheads: int = 16,
    dec_n_points: int = 2,
    dim_feedforward: int = 2048,
    dropout: float = 0.0,
    num_queries: int = 300,
    num_select: int = 100,
    group_detr: int = 13,
    two_stage: bool = True,
    bbox_reparam: bool = True,
    lite_refpoint_refine: bool = True,
    # ----- Segmentation -----
    segmentation_head: bool = False,
    mask_downsample_ratio: int = 4,
    # ----- Loss -----
    aux_loss: bool = True,
    ia_bce_loss: bool = True,
    focal_alpha: float = 0.25,
    cls_loss_coef: float = 1.0,
    bbox_loss_coef: float = 5.0,
    giou_loss_coef: float = 2.0,
    mask_ce_loss_coef: float = 5.0,
    mask_dice_loss_coef: float = 5.0,
) -> Dict[str, nn.Module]:
    """Build the US-RFDETR model + criterion + postprocessor.

    Returns
    -------
    dict
        ``{"model": LWDETR, "criterion": SetCriterion, "postprocess": PostProcess}``.
        The recipe unpacks this in ``setup()``.

    Notes
    -----
    * ``num_classes`` is the foreground count; the RF-DETR convention
      adds +1 for the no-object class internally.
    * If ``segmentation_head=True``, an extra :class:`SegmentationHead`
      is attached and the criterion gets ``loss_mask_ce + loss_mask_dice``
      in its weight dict.
    * ``frozen_backbone=True`` is supported but uncommon — we typically
      want at least the FPN neck trainable so the encoder adapts to
      detection features.
    """
    if use_feature_levels is None:
        use_feature_levels = [1, 2, 3]
    num_feature_levels = len(use_feature_levels)

    # ----- Backbone -----
    backbone = build_sam2_rfdetr_backbone(
        image_encoder=image_encoder,
        hidden_dim=hidden_dim,
        position_embedding="sine",
        pretrained_ckpt=pretrained_ckpt,
        frozen=frozen_backbone,
        use_feature_levels=use_feature_levels,
    )

    # ----- Transformer -----
    args = _make_args(
        hidden_dim=hidden_dim,
        sa_nheads=sa_nheads,
        ca_nheads=ca_nheads,
        dec_layers=dec_layers,
        dec_n_points=dec_n_points,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        num_queries=num_queries,
        group_detr=group_detr,
        two_stage=two_stage,
        num_feature_levels=num_feature_levels,
        lite_refpoint_refine=lite_refpoint_refine,
        bbox_reparam=bbox_reparam,
        num_classes=num_classes,
        aux_loss=aux_loss,
        ia_bce_loss=ia_bce_loss,
        focal_alpha=focal_alpha,
        cls_loss_coef=cls_loss_coef,
        bbox_loss_coef=bbox_loss_coef,
        giou_loss_coef=giou_loss_coef,
        segmentation_head=segmentation_head,
        mask_downsample_ratio=mask_downsample_ratio,
        mask_ce_loss_coef=mask_ce_loss_coef,
        mask_dice_loss_coef=mask_dice_loss_coef,
    )

    transformer = build_transformer(args)

    # ----- Segmentation head (optional) -----
    seg_head = (
        SegmentationHead(
            hidden_dim, dec_layers, downsample_ratio=mask_downsample_ratio
        )
        if segmentation_head
        else None
    )

    # ----- LWDETR model -----
    model = LWDETR(
        backbone=backbone,
        transformer=transformer,
        segmentation_head=seg_head,
        num_classes=num_classes + 1,  # RF-DETR convention: +1 for no-object
        num_queries=num_queries,
        aux_loss=aux_loss,
        group_detr=group_detr,
        two_stage=two_stage,
        lite_refpoint_refine=lite_refpoint_refine,
        bbox_reparam=bbox_reparam,
    )

    # ----- Criterion -----
    args.num_select = num_select
    args.set_cost_class = 2.0
    args.set_cost_bbox = 5.0
    args.set_cost_giou = 2.0
    args.mask_point_sample_ratio = 16
    matcher = build_matcher(args)

    weight_dict = {
        "loss_ce": cls_loss_coef,
        "loss_bbox": bbox_loss_coef,
        "loss_giou": giou_loss_coef,
    }
    if segmentation_head:
        weight_dict["loss_mask_ce"] = mask_ce_loss_coef
        weight_dict["loss_mask_dice"] = mask_dice_loss_coef

    if aux_loss:
        aux_weight_dict = {}
        for i in range(dec_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        if two_stage:
            aux_weight_dict.update({k + "_enc": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ["labels", "boxes", "cardinality"]
    if segmentation_head:
        losses.append("masks")

    criterion = SetCriterion(
        num_classes + 1,
        matcher=matcher,
        weight_dict=weight_dict,
        focal_alpha=focal_alpha,
        losses=losses,
        group_detr=group_detr,
        use_varifocal_loss=False,
        use_position_supervised_loss=False,
        ia_bce_loss=ia_bce_loss,
        mask_point_sample_ratio=16 if segmentation_head else 16,
    )

    postprocess = PostProcess(num_select=num_select)

    return {
        "model": model,
        "criterion": criterion,
        "postprocess": postprocess,
    }
