import torch
import torch.nn.functional as F
from typing import List, Dict
from torchmetrics.segmentation import DiceScore, MeanIoU

from nemo_cv.components.datasets.sam2.data_utils import BatchedVideoDatapoint


class Sam2VOSMeanIoU(MeanIoU):
    """Mean IoU metric for SAM2 Video Object Segmentation.

    Handles the output format from SAM2 video training where:
    - outputs is a List[Dict] with one dict per frame
    - each dict contains 'pred_masks_high_res' with shape (N, 1, H, W)
    - inputs.masks has shape (T, O, H, W) where T=num_frames, O=num_objects
    """

    def update(self, outputs: List[Dict], inputs: BatchedVideoDatapoint):
        gt_masks = inputs.masks
        num_frames = len(outputs)

        for frame_idx in range(num_frames):
            frame_output = outputs[frame_idx]
            pred_masks_binary = (frame_output["pred_masks_high_res"].squeeze(1) > 0.0).long()
            gt_frame = gt_masks[frame_idx].long()

            pred_h, pred_w = pred_masks_binary.shape[-2:]
            gt_h, gt_w = gt_frame.shape[-2:]

            if pred_h != gt_h or pred_w != gt_w:
                pred_masks_binary = F.interpolate(
                    pred_masks_binary.unsqueeze(1).float(),
                    size=(gt_h, gt_w),
                    mode="nearest",
                ).squeeze(1).long()

            super().update(pred_masks_binary, gt_frame)


class Sam2VOSDiceScore(DiceScore):
    """Dice Score metric for SAM2 Video Object Segmentation.

    Handles the output format from SAM2 video training where:
    - outputs is a List[Dict] with one dict per frame
    - each dict contains 'multistep_pred_masks_high_res' with shape (N, num_steps, H, W)
    - inputs.masks has shape (T, O, H, W) where T=num_frames, O=num_objects
    """

    def update(self, outputs: List[Dict], inputs: BatchedVideoDatapoint):
        gt_masks = inputs.masks
        num_frames = len(outputs)

        for frame_idx in range(num_frames):
            frame_output = outputs[frame_idx]
            pred_masks_high_res = frame_output["pred_masks_high_res"]
            if pred_masks_high_res.dim() == 4:
                pred_masks_high_res = pred_masks_high_res.squeeze(1)

            pred_masks_binary = (pred_masks_high_res > 0.0).long()
            gt_frame = gt_masks[frame_idx].long()

            pred_h, pred_w = pred_masks_binary.shape[-2:]
            gt_h, gt_w = gt_frame.shape[-2:]

            if pred_h != gt_h or pred_w != gt_w:
                pred_masks_binary = F.interpolate(
                    pred_masks_binary.unsqueeze(1).float(),
                    size=(gt_h, gt_w),
                    mode="nearest",
                ).squeeze(1).long()

            super().update(pred_masks_binary, gt_frame)
