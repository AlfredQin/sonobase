import logging
import math
import os
import random
import time
from contextlib import nullcontext
from omegaconf import DictConfig
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn


class Phase:
    TRAIN = "train"
    VAL = "val"


CORE_LOSS_KEY = "core_loss"


def unwrap_ddp_if_wrapped(model: nn.Module) -> nn.Module:
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    return model


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def barrier() -> None:
    if is_dist_avail_and_initialized():
        dist.barrier()


def all_reduce_max(tensor: torch.Tensor) -> torch.Tensor:
    if is_dist_avail_and_initialized():
        if not tensor.is_cuda:
            tensor = tensor.cuda()
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        tensor = tensor.cpu()
    return tensor


def seed_everything(seed: int) -> None:
    """Seed every RNG path used by training/eval recipes.

    Use this in recipes that do NOT already manage RNG via
    `nemo_automodel.components.training.rng.StatefulRNG` /
    `ScopedRNG` (which seed the same three RNGs with optional
    rank-offset and add checkpoint state_dict support). The
    sonobase pretrain recipe uses StatefulRNG and doesn't call
    this; us_rfdetr uses StatefulRNG too. few_shot / save_predictions
    use this helper because they're single-GPU recipes (no
    rank-offset needed).

    DDP rank-offset, if desired, is the caller's responsibility.
    """
    logging.info(f"seed_everything: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    """DataLoader `worker_init_fn` that reseeds numpy / Python random per worker.

    PyTorch already reseeds `torch`'s RNG per worker (derived from
    `torch.initial_seed()`), but does NOT reseed `numpy.random` or Python's
    `random` — augmentation code in `datasets/sam2/transforms.py` uses both,
    so without this hook every worker process inherits the launch state and
    augmentation sequences are non-deterministic across runs even at fixed
    main-process seed.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_amp_type(amp_type: Optional[str] = None) -> Optional[torch.dtype]:
    if amp_type is None:
        return None
    assert amp_type in ["bfloat16", "float16"], f"Invalid AMP type: {amp_type}"
    return torch.bfloat16 if amp_type == "bfloat16" else torch.float16


def build_autocast_context(
    optim_cfg: DictConfig,
    device_type: str = "cuda",
):
    """Build an autocast context manager.

    AutoModel convention: prefer native bf16 via FSDP MixedPrecisionPolicy or
    explicit dtype casting. torch.autocast is used only as an opt-in for specific
    recipes. GradScaler is NOT used (bf16 does not require loss scaling).

    For SAM2 backward-compatibility, fp16 + GradScaler can still be enabled via
    config but is not the default path.
    """
    amp_enabled = optim_cfg.amp.get("enabled", False)
    amp_dtype_str = optim_cfg.amp.get("amp_dtype", "bfloat16")
    if not amp_enabled:
        return nullcontext()
    dtype = get_amp_type(amp_dtype_str)
    return torch.autocast(device_type=device_type, dtype=dtype)


def build_grad_scaler(
    optim_cfg: DictConfig,
    device: str = "cuda",
) -> torch.amp.GradScaler:
    """Build a GradScaler, disabled by default (AutoModel convention).

    Only enable when using fp16 AMP — bf16 does not need loss scaling.
    """
    amp_enabled = optim_cfg.amp.get("enabled", False)
    amp_dtype_str = optim_cfg.amp.get("amp_dtype", "bfloat16")
    return torch.amp.GradScaler(device, enabled=amp_enabled and amp_dtype_str == "float16")


def human_readable_time(time_seconds: float) -> str:
    t = int(time_seconds)
    minutes, seconds = divmod(t, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    return f"{days:02d}d {hours:02d}h {minutes:02d}m"


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self, name: str, device: torch.device, fmt: str = ":f"):
        self.name = name
        self.fmt = fmt
        self.device = device
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name}: {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class MemMeter:
    """Tracks peak GPU memory usage per iteration."""

    def __init__(self, name: str, device: torch.device, fmt: str = ":f"):
        self.name = name
        self.fmt = fmt
        self.device = device
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.peak = 0
        self.sum = 0
        self.count = 0

    def update(self, n: int = 1, reset_peak_usage: bool = True):
        self.val = torch.cuda.max_memory_allocated() // 1e9
        self.sum += self.val * n
        self.count += n
        self.avg = self.sum / self.count
        self.peak = max(self.peak, self.val)
        if reset_peak_usage:
            torch.cuda.reset_peak_memory_stats()

    def __str__(self):
        fmtstr = "{name}: {val" + self.fmt + "} ({avg" + self.fmt + "}/{peak" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class DurationMeter:
    """Tracks elapsed time."""

    def __init__(self, name: str, device: torch.device, fmt: str = ":f"):
        self.name = name
        self.device = device
        self.fmt = fmt
        self.val = 0

    def reset(self):
        self.val = 0

    def update(self, val: float):
        self.val = val

    def add(self, val: float):
        self.val += val

    def __str__(self):
        return f"{self.name}: {human_readable_time(self.val)}"


class ProgressMeter:
    """Displays training progress with multiple meters."""

    def __init__(
        self,
        num_batches: int,
        meters: list,
        real_meters: dict,
        prefix: str = "",
    ):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.real_meters = real_meters
        self.prefix = prefix

    def display(self, batch: int):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        entries += [
            " | ".join(
                [f"{os.path.join(name, subname)}: {val:.4f}" for subname, val in meter.compute().items()]
            )
            for name, meter in self.real_meters.items()
        ]
        logging.info(" | ".join(entries))

    def _get_batch_fmtstr(self, num_batches: int) -> str:
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def print_model_summary(model: nn.Module) -> None:
    if get_rank() != 0:
        return
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    non_trainable = total - trainable
    logging.info("==" * 10)
    logging.info(f"Summary for model {type(model)}")
    logging.info(f"\tTotal parameters {_human_readable_count(total)}")
    logging.info(f"\tTrainable parameters {_human_readable_count(trainable)}")
    logging.info(f"\tNon-Trainable parameters {_human_readable_count(non_trainable)}")
    logging.info("==" * 10)


_PARAM_UNITS = [" ", "K", "M", "B", "T"]


def _human_readable_count(number: int) -> str:
    assert number >= 0
    labels = _PARAM_UNITS
    num_digits = int(np.floor(np.log10(number)) + 1 if number > 0 else 1)
    num_groups = int(np.ceil(num_digits / 3))
    num_groups = min(num_groups, len(labels))
    shift = -3 * (num_groups - 1)
    number = number * (10**shift)
    index = num_groups - 1
    if index < 1 or number >= 100:
        return f"{int(number):,d} {labels[index]}"
    return f"{number:,.1f} {labels[index]}"
