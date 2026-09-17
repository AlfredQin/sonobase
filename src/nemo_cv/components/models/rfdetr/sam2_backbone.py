"""SAM2 backbone adapter for RF-DETR.

Replaces RF-DETR's DINOv2 + MultiScaleProjector with the pretrained SAM2
``ImageEncoder`` (Hiera or TriBranchTrunk + FPN neck). The FPN already
produces multi-scale 256-dim features, so no extra projector is needed.

The adapter converts SAM2's FPN output into ``List[NestedTensor]`` +
position encodings — the exact format expected by RF-DETR's transformer.

Ported from ``USSam/projects/DenseUS/dense_models/rfdetr_backbone.py``
with a single import-path edit (DenseUS's
``models.segmentation.interactive.sam2.modeling.backbones.image_encoder``
→ sonobase's ``nemo_cv.components.models.sam2.modeling.backbones.image_encoder``).
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo_cv.components.models.rfdetr.position_encoding import (
    build_position_encoding,
)
from nemo_cv.components.models.rfdetr.util.misc import NestedTensor
from nemo_cv.components.models.sam2.modeling.backbones.image_encoder import (
    ImageEncoder,
)

logger = logging.getLogger(__name__)


class SAM2RFDETRBackbone(nn.Module):
    """Wraps a SAM2 ``ImageEncoder`` so its output matches RF-DETR's
    backbone interface: ``forward(NestedTensor) -> List[NestedTensor]``.

    The ImageEncoder's FPN produces 4 feature levels (strides 4, 8, 16,
    32) at ``d_model`` channels (typically 256). This adapter selects
    which levels to pass to the transformer via ``use_feature_levels``.

    Parameters
    ----------
    image_encoder : ImageEncoder
        SAM2 image encoder (trunk + FPN neck) with ``scalp=0``.
    pretrained_ckpt : str, optional
        Path to a single-file ``.pt`` checkpoint. Accepts SAM2 release
        format (``{"model": state_dict}``), MedSAM2 release format, and
        the converted SonoBase format produced by
        ``nemo_cv.recipes.benchmarks.convert_sonobase_dcp_to_pt``.
    frozen : bool
        Freeze backbone parameters (no gradient updates during training).
    use_feature_levels : list[int], optional
        Which FPN output levels to use (``0`` = stride-4, ``1`` =
        stride-8, …). Default ``[1, 2, 3]`` ≈ P3/P4/P5 (skip stride-4
        for efficiency — RF-DETR's deformable attention only needs three
        scales).
    """

    def __init__(
        self,
        image_encoder: ImageEncoder,
        pretrained_ckpt: Optional[str] = None,
        frozen: bool = False,
        use_feature_levels: Optional[List[int]] = None,
        d_model: int = 256,
    ):
        super().__init__()
        self.encoder = image_encoder
        self.out_channels = d_model

        if use_feature_levels is None:
            use_feature_levels = [1, 2, 3]
        self.use_feature_levels = list(use_feature_levels)

        if pretrained_ckpt is not None:
            self._load_pretrained(pretrained_ckpt)

        if frozen:
            for p in self.encoder.parameters():
                p.requires_grad = False
            logger.info("SAM2RFDETRBackbone: encoder frozen.")

    def forward(self, tensor_list: NestedTensor) -> List[NestedTensor]:
        """Run the encoder + select FPN levels.

        Parameters
        ----------
        tensor_list : NestedTensor
            ``.tensors``: ``(B, 3, H, W)``, ``.mask``: ``(B, H, W)`` padding mask.

        Returns
        -------
        List[NestedTensor]
            One entry per requested FPN level, each with ``.tensors``
            ``(B, C, H_i, W_i)`` and a per-level downsampled padding mask.
        """
        output = self.encoder(tensor_list.tensors)
        all_fpn_feats: List[torch.Tensor] = output["backbone_fpn"]

        out: List[NestedTensor] = []
        for lvl_idx in self.use_feature_levels:
            feat = all_fpn_feats[lvl_idx]
            mask = tensor_list.mask
            mask = F.interpolate(
                mask[None].float(), size=feat.shape[-2:]
            ).to(torch.bool)[0]
            out.append(NestedTensor(feat, mask))
        return out

    def _load_pretrained(self, ckpt_path: str) -> None:
        """Load image-encoder weights from a single-file ``.pt``.

        Strips ``model.image_encoder.`` / ``image_encoder.`` prefixes if
        present (so the same loader handles SAM2 release format,
        MedSAM2, and the converted SonoBase format) then calls
        ``load_state_dict(strict=False)``. Mismatches are logged.

        Note: ``weights_only=False`` is intentional. Some older
        checkpoints store ``omegaconf.DictConfig`` objects in their
        provenance / hyperparams that the strict ``weights_only=True``
        unpickler rejects; we trust the source path here.
        """
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        if isinstance(raw, dict) and "model" in raw and isinstance(raw["model"], dict):
            state_dict = raw["model"]
        elif isinstance(raw, dict) and "state_dict" in raw:
            state_dict = raw["state_dict"]
        else:
            state_dict = raw

        ie_prefix = "image_encoder."
        model_prefix = "model.image_encoder."
        if any(k.startswith(model_prefix) for k in state_dict):
            ie_sd = {
                k[len(model_prefix):]: v
                for k, v in state_dict.items()
                if k.startswith(model_prefix)
            }
        elif any(k.startswith(ie_prefix) for k in state_dict):
            ie_sd = {
                k[len(ie_prefix):]: v
                for k, v in state_dict.items()
                if k.startswith(ie_prefix)
            }
        else:
            ie_sd = state_dict

        msg = self.encoder.load_state_dict(ie_sd, strict=False)
        logger.info(f"SAM2RFDETRBackbone: loaded weights from {ckpt_path}")
        if msg.missing_keys:
            logger.warning(
                f"  Missing ({len(msg.missing_keys)}): {msg.missing_keys[:5]}..."
            )
        if msg.unexpected_keys:
            logger.warning(
                f"  Unexpected ({len(msg.unexpected_keys)}): {msg.unexpected_keys[:5]}..."
            )


class SAM2Joiner(nn.Sequential):
    """Pairs the SAM2 backbone with RF-DETR's position encoding.

    Mirrors ``rfdetr.models.backbone.Joiner`` but uses our SAM2 backbone.
    Returns ``(features, position_encodings)``.
    """

    def __init__(self, backbone: SAM2RFDETRBackbone, position_embedding: nn.Module):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        features = self[0](tensor_list)
        pos = []
        for feat in features:
            pos.append(
                self[1](feat, align_dim_orders=False).to(feat.tensors.dtype)
            )
        return features, pos


def build_sam2_rfdetr_backbone(
    image_encoder: ImageEncoder,
    hidden_dim: int = 256,
    position_embedding: str = "sine",
    pretrained_ckpt: Optional[str] = None,
    frozen: bool = False,
    use_feature_levels: Optional[List[int]] = None,
) -> SAM2Joiner:
    """Build a SAM2-backed backbone module compatible with RF-DETR's ``LWDETR``.

    Returns
    -------
    SAM2Joiner
        ``nn.Sequential(SAM2RFDETRBackbone, PositionEmbedding)``.
    """
    pos_enc = build_position_encoding(hidden_dim, position_embedding)
    backbone = SAM2RFDETRBackbone(
        image_encoder=image_encoder,
        pretrained_ckpt=pretrained_ckpt,
        frozen=frozen,
        use_feature_levels=use_feature_levels,
    )
    return SAM2Joiner(backbone, pos_enc)
