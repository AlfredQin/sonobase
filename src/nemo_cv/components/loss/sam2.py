from collections import defaultdict
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo_cv.components.training.utils import CORE_LOSS_KEY, get_world_size, is_dist_avail_and_initialized


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_objects: float,
    loss_on_multimask: bool = False,
) -> torch.Tensor:
    """Compute the DICE loss, similar to generalized IoU for masks.

    Args:
        inputs: Predicted logits of arbitrary shape.
        targets: Binary ground-truth labels with same shape as inputs.
        num_objects: Number of objects (for normalization).
        loss_on_multimask: If True, inputs/targets are [N, M, H, W].
    """
    inputs = inputs.sigmoid()
    if loss_on_multimask:
        assert inputs.dim() == 4 and targets.dim() == 4
        inputs = inputs.flatten(2)
        targets = targets.flatten(2)
        numerator = 2 * (inputs * targets).sum(-1)
    else:
        inputs = inputs.flatten(1)
        numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    if loss_on_multimask:
        return loss / num_objects
    return loss.sum() / num_objects


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_objects: float,
    alpha: float = 0.25,
    gamma: float = 2,
    loss_on_multimask: bool = False,
) -> torch.Tensor:
    """Focal loss from RetinaNet (https://arxiv.org/abs/1708.02002).

    Args:
        inputs: Predicted logits of arbitrary shape.
        targets: Binary ground-truth labels with same shape as inputs.
        num_objects: Number of objects (for normalization).
        alpha: Weighting factor for positive/negative balance.
        gamma: Modulating factor exponent.
        loss_on_multimask: If True, inputs are [N, M, H, W].
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if loss_on_multimask:
        assert loss.dim() == 4
        return loss.flatten(2).mean(-1) / num_objects
    return loss.mean(1).sum() / num_objects


def iou_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    pred_ious: torch.Tensor,
    num_objects: float,
    loss_on_multimask: bool = False,
    use_l1_loss: bool = False,
) -> torch.Tensor:
    """IoU prediction loss (MSE or L1 between predicted and actual IoU).

    Args:
        inputs: Predicted mask logits [N, M, H, W].
        targets: Ground-truth masks [N, M, H, W].
        pred_ious: Predicted IoU scores per mask.
        num_objects: Number of objects (for normalization).
        loss_on_multimask: If True, return per-mask losses.
        use_l1_loss: Use L1 instead of MSE.
    """
    assert inputs.dim() == 4 and targets.dim() == 4
    pred_mask = inputs.flatten(2) > 0
    gt_mask = targets.flatten(2) > 0
    area_i = torch.sum(pred_mask & gt_mask, dim=-1).float()
    area_u = torch.sum(pred_mask | gt_mask, dim=-1).float()
    actual_ious = area_i / torch.clamp(area_u, min=1.0)

    if use_l1_loss:
        loss = F.l1_loss(pred_ious, actual_ious, reduction="none")
    else:
        loss = F.mse_loss(pred_ious, actual_ious, reduction="none")
    if loss_on_multimask:
        return loss / num_objects
    return loss.sum() / num_objects


class MultiStepMultiMasksAndIous(nn.Module):
    """Multi-step multi-mask and IoU loss for SAM2.

    Computes focal, dice, and IoU losses across multiple prediction steps,
    selecting the best mask channel (lowest focal+dice) for backpropagation.
    """

    def __init__(
        self,
        weight_dict: Dict[str, float],
        focal_alpha: float = 0.25,
        focal_gamma: float = 2,
        supervise_all_iou: bool = False,
        iou_use_l1_loss: bool = False,
        pred_obj_scores: bool = False,
        focal_gamma_obj_score: float = 0.0,
        focal_alpha_obj_score: float = -1,
    ):
        super().__init__()
        self.weight_dict = weight_dict
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        assert "loss_mask" in self.weight_dict
        assert "loss_dice" in self.weight_dict
        assert "loss_iou" in self.weight_dict
        if "loss_class" not in self.weight_dict:
            self.weight_dict["loss_class"] = 0.0

        self.focal_alpha_obj_score = focal_alpha_obj_score
        self.focal_gamma_obj_score = focal_gamma_obj_score
        self.supervise_all_iou = supervise_all_iou
        self.iou_use_l1_loss = iou_use_l1_loss
        self.pred_obj_scores = pred_obj_scores

    def forward(self, outs_batch: List[Dict], targets_batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        assert len(outs_batch) == len(targets_batch)
        num_objects = torch.tensor(
            targets_batch.shape[1], device=targets_batch.device, dtype=torch.float
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_objects)
        num_objects = torch.clamp(num_objects / get_world_size(), min=1).item()

        losses = defaultdict(int)
        for outs, targets in zip(outs_batch, targets_batch):
            cur_losses = self._forward(outs, targets, num_objects)
            for k, v in cur_losses.items():
                losses[k] += v

        return losses

    def _forward(self, outputs: Dict, targets: torch.Tensor, num_objects: float) -> Dict[str, torch.Tensor]:
        target_masks = targets.unsqueeze(1).float()
        assert target_masks.dim() == 4

        src_masks_list = outputs["multistep_pred_multimasks_high_res"]
        ious_list = outputs["multistep_pred_ious"]
        object_score_logits_list = outputs["multistep_object_score_logits"]

        assert len(src_masks_list) == len(ious_list)
        assert len(object_score_logits_list) == len(ious_list)

        losses = {"loss_mask": 0, "loss_dice": 0, "loss_iou": 0, "loss_class": 0}
        for src_masks, ious, object_score_logits in zip(src_masks_list, ious_list, object_score_logits_list):
            self._update_losses(losses, src_masks, target_masks, ious, num_objects, object_score_logits)
        losses[CORE_LOSS_KEY] = self.reduce_loss(losses)
        return losses

    def _update_losses(
        self,
        losses: Dict[str, torch.Tensor],
        src_masks: torch.Tensor,
        target_masks: torch.Tensor,
        ious: torch.Tensor,
        num_objects: float,
        object_score_logits: torch.Tensor,
    ):
        target_masks = target_masks.expand_as(src_masks)
        loss_multimask = sigmoid_focal_loss(
            src_masks, target_masks, num_objects,
            alpha=self.focal_alpha, gamma=self.focal_gamma, loss_on_multimask=True,
        )
        loss_multidice = dice_loss(src_masks, target_masks, num_objects, loss_on_multimask=True)

        if not self.pred_obj_scores:
            loss_class = torch.tensor(0.0, dtype=loss_multimask.dtype, device=loss_multimask.device)
            target_obj = torch.ones(
                loss_multimask.shape[0], 1, dtype=loss_multimask.dtype, device=loss_multimask.device,
            )
        else:
            target_obj = torch.any((target_masks[:, 0] > 0).flatten(1), dim=-1)[..., None].float()
            loss_class = sigmoid_focal_loss(
                object_score_logits, target_obj, num_objects,
                alpha=self.focal_alpha_obj_score, gamma=self.focal_gamma_obj_score,
            )

        loss_multiiou = iou_loss(
            src_masks, target_masks, ious, num_objects,
            loss_on_multimask=True, use_l1_loss=self.iou_use_l1_loss,
        )

        assert loss_multimask.dim() == 2
        assert loss_multidice.dim() == 2
        assert loss_multiiou.dim() == 2

        if loss_multimask.size(1) > 1:
            loss_combo = loss_multimask * self.weight_dict["loss_mask"] + loss_multidice * self.weight_dict["loss_dice"]
            best_loss_inds = torch.argmin(loss_combo, dim=-1)
            batch_inds = torch.arange(loss_combo.size(0), device=loss_combo.device)
            loss_mask = loss_multimask[batch_inds, best_loss_inds].unsqueeze(1)
            loss_dice = loss_multidice[batch_inds, best_loss_inds].unsqueeze(1)
            if self.supervise_all_iou:
                loss_iou = loss_multiiou.mean(dim=-1).unsqueeze(1)
            else:
                loss_iou = loss_multiiou[batch_inds, best_loss_inds].unsqueeze(1)
        else:
            loss_mask = loss_multimask
            loss_dice = loss_multidice
            loss_iou = loss_multiiou

        loss_mask = loss_mask * target_obj
        loss_dice = loss_dice * target_obj
        loss_iou = loss_iou * target_obj

        losses["loss_mask"] += loss_mask.sum()
        losses["loss_dice"] += loss_dice.sum()
        losses["loss_iou"] += loss_iou.sum()
        losses["loss_class"] += loss_class

    def reduce_loss(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        reduced_loss = 0.0
        for loss_key, weight in self.weight_dict.items():
            if loss_key not in losses:
                raise ValueError(f"{type(self)} doesn't compute {loss_key}")
            if weight != 0:
                reduced_loss += losses[loss_key] * weight
        return reduced_loss
