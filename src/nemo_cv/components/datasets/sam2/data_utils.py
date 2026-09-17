from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from PIL import Image as PILImage


@dataclass
class BatchedVideoMetaData:
    """Metadata about a batch of videos.

    Attributes:
        unique_objects_identifier: [TxOx3] tensor (video_id, obj_id, frame_id).
        frame_orig_size: [TxOx2] tensor with original frame sizes.
    """

    unique_objects_identifier: torch.LongTensor
    frame_orig_size: torch.LongTensor

    def to(self, *args, **kwargs):
        return BatchedVideoMetaData(
            unique_objects_identifier=self.unique_objects_identifier.to(*args, **kwargs),
            frame_orig_size=self.frame_orig_size.to(*args, **kwargs),
        )

    def pin_memory(self, device=None):
        return BatchedVideoMetaData(
            unique_objects_identifier=self.unique_objects_identifier.pin_memory(device=device),
            frame_orig_size=self.frame_orig_size.pin_memory(device=device),
        )


@dataclass
class BatchedVideoDatapoint:
    """A batch of videos with annotations and metadata.

    Attributes:
        img_batch: [TxBxCxHxW] tensor of image data.
        obj_to_frame_idx: [TxOx2] tensor mapping objects to frames.
        masks: [TxOxHxW] tensor of binary masks.
        metadata: BatchedVideoMetaData instance.
        dict_key: string key identifying the batch (used for loss dispatch).
    """

    img_batch: torch.FloatTensor
    obj_to_frame_idx: torch.IntTensor
    masks: torch.BoolTensor
    metadata: BatchedVideoMetaData

    dict_key: str
    dataset_names: Optional[List[str]] = None
    video_names: Optional[List[str]] = None

    @property
    def num_frames(self) -> int:
        return self.img_batch.shape[0]

    @property
    def num_videos(self) -> int:
        return self.img_batch.shape[1]

    @property
    def flat_obj_to_img_idx(self) -> torch.IntTensor:
        """Flattened object-to-image index for a [(T*B)xCxHxW] img_batch."""
        frame_idx, video_idx = self.obj_to_frame_idx.unbind(dim=-1)
        return video_idx * self.num_frames + frame_idx

    @property
    def flat_img_batch(self) -> torch.FloatTensor:
        """Flattened img_batch of shape [(B*T)xCxHxW]."""
        return self.img_batch.transpose(0, 1).flatten(0, 1)

    def to(self, *args, **kwargs):
        return BatchedVideoDatapoint(
            img_batch=self.img_batch.to(*args, **kwargs),
            obj_to_frame_idx=self.obj_to_frame_idx.to(*args, **kwargs),
            masks=self.masks.to(*args, **kwargs),
            metadata=self.metadata.to(*args, **kwargs),
            dict_key=self.dict_key,
            dataset_names=self.dataset_names,
            video_names=self.video_names,
        )

    def pin_memory(self, device=None):
        return BatchedVideoDatapoint(
            img_batch=self.img_batch.pin_memory(device=device),
            obj_to_frame_idx=self.obj_to_frame_idx.pin_memory(device=device),
            masks=self.masks.pin_memory(device=device),
            metadata=self.metadata.pin_memory(device=device),
            dict_key=self.dict_key,
            dataset_names=self.dataset_names,
            video_names=self.video_names,
        )


@dataclass
class Object:
    object_id: int
    frame_index: int
    segment: Union[torch.Tensor, dict]


@dataclass
class Frame:
    data: Union[torch.Tensor, PILImage.Image]
    objects: List[Object]


@dataclass
class VideoDatapoint:
    """An image/video and all its annotations."""

    frames: List[Frame]
    video_id: int
    size: Tuple[int, int]


def collate_fn(
    batch: List[VideoDatapoint],
    dict_key: str,
) -> BatchedVideoDatapoint:
    img_batch = []
    for video in batch:
        img_batch.append(torch.stack([frame.data for frame in video.frames], dim=0))

    img_batch = torch.stack(img_batch, dim=0).permute((1, 0, 2, 3, 4))
    T = img_batch.shape[0]

    step_t_objects_identifier = [[] for _ in range(T)]
    step_t_frame_orig_size = [[] for _ in range(T)]
    step_t_masks = [[] for _ in range(T)]
    step_t_obj_to_frame_idx = [[] for _ in range(T)]

    for video_idx, video in enumerate(batch):
        orig_video_id = video.video_id
        orig_frame_size = video.size
        for t, frame in enumerate(video.frames):
            for obj in frame.objects:
                step_t_obj_to_frame_idx[t].append(torch.tensor([t, video_idx], dtype=torch.int))
                step_t_masks[t].append(obj.segment.to(torch.bool))
                step_t_objects_identifier[t].append(
                    torch.tensor([orig_video_id, obj.object_id, obj.frame_index])
                )
                step_t_frame_orig_size[t].append(torch.tensor(orig_frame_size))

    obj_to_frame_idx = torch.stack(
        [torch.stack(indices, dim=0) for indices in step_t_obj_to_frame_idx],
        dim=0,
    )
    masks = torch.stack([torch.stack(m, dim=0) for m in step_t_masks], dim=0)
    objects_identifier = torch.stack(
        [torch.stack(ids, dim=0) for ids in step_t_objects_identifier], dim=0
    )
    frame_orig_size = torch.stack(
        [torch.stack(sizes, dim=0) for sizes in step_t_frame_orig_size], dim=0
    )
    return BatchedVideoDatapoint(
        img_batch=img_batch,
        obj_to_frame_idx=obj_to_frame_idx,
        masks=masks,
        metadata=BatchedVideoMetaData(
            unique_objects_identifier=objects_identifier,
            frame_orig_size=frame_orig_size,
        ),
        dict_key=dict_key,
    )
