"""US-RFDETR training recipe — per-dataset detector training (DDP).

Trains the RF-DETR transformer on top of a SAM2-style image encoder
(SAM2-no-ft, MedSAM2, or SonoBase). The detector is trained PER-DATASET,
not jointly across datasets — each `(dataset × backbone)` cell produces
its own ``best.pt`` ranked by validation bbox-mAP.

Pipeline (matches the existing sonobase recipes' structure)
-----------------------------------------------------------

1. ``setup()``
     * Initialise distributed (``initialize_distributed``).
     * Build the chosen ``data_module`` via Hydra → train / val / test
       Datasets, wrap them in ``DistributedSampler``-aware DataLoaders.
     * Build the SAM2 ``image_encoder`` via Hydra (recursive) →
       ``build_rfdetr`` → ``{model, criterion, postprocess}``.
     * DDP-wrap the model (``find_unused_parameters=True`` because
       group-DETR has unused queries depending on the batch).
     * Build the AdamW optimiser with TWO param groups (head at ``lr``,
       backbone at ``lr_encoder``) + a per-step
       ``timm.scheduler.CosineLRScheduler`` with linear warmup.
     * Instantiate per-dataset ``MeanAveragePrecision`` (bbox + segm)
       trackers — one entry per validation/test loader.
     * Allocate ``training.jsonl`` / ``validation.jsonl`` /
       ``test.jsonl`` metric loggers (rank 0).

2. ``run()`` → ``run_train_loop()`` → ``run_val_loop()`` per epoch.
   Every epoch:
     * Train: forward → criterion → weighted-sum loss → backward →
       grad clip → optimiser.step → scheduler.step_update.
     * Validate: forward → postprocess → metric.update; at end of
       loader, compute() syncs across DDP ranks and logs result.
     * If `bbox_mAP` improves on rank 0, atomically write
       ``${experiment_dir}/best.pt`` (single-file checkpoint, model
       state only — no DCP).
3. After training, load ``best.pt`` (rank 0, broadcast) and run
   ``run_test_loop()``; write ``test_metrics.json``.

CLI
---
::

    uv run torchrun --nproc_per_node=4 -m nemo_cv.recipes.us_rfdetr.train \\
        -c ./configs/us_rfdetr -cn train \\
        experiment=us_rfdetr_on_acouslic backbone=sonobase \\
        model.pretrained_ckpt=./experiments/sonobase/.../LATEST.pt
"""

from __future__ import annotations

import json
import logging
import math
import os
import pathlib
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from hydra.utils import instantiate as hydra_instantiate
from omegaconf import DictConfig, OmegaConf

from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.distributed.init_utils import (
    DistInfo,
    initialize_distributed,
)
from nemo_automodel.components.loggers.log_utils import setup_logging
from nemo_automodel.components.loggers.metric_logger import (
    MetricsSample,
    build_metric_logger,
)

from nemo_automodel.components.training.rng import ScopedRNG, StatefulRNG

from nemo_cv.components.datasets.us_rfdetr import detection_collate_fn
from nemo_cv.components.models.rfdetr import build_rfdetr
from nemo_cv.components.models.rfdetr.util.misc import (
    nested_tensor_from_tensor_list,
)
from nemo_cv.components.training.utils import (
    barrier,
    build_autocast_context,
    build_grad_scaler,
    seed_worker,
    unwrap_ddp_if_wrapped,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Helpers (module-level, stateless)
# ---------------------------------------------------------------------------


def _ensure_distributed(dist_cfg: DictConfig) -> DistInfo:
    """Initialise distributed if needed (works for ``world_size == 1`` too).

    Mirrors the pattern used by ``sonobase.pretrain.build_distributed`` so
    multi-GPU `torchrun` and single-GPU `torchrun --nproc_per_node=1` (or
    plain `python -m`) both reach a working ``DistInfo``.
    """
    backend = dist_cfg.get("backend", "nccl")
    timeout = dist_cfg.get("timeout_minutes", 30)
    return initialize_distributed(backend=backend, timeout_minutes=timeout)


def _convert_targets_to_detr(
    targets: List[Dict[str, torch.Tensor]],
    image_sizes: List[Tuple[int, int]],
) -> List[Dict[str, torch.Tensor]]:
    """Convert torchvision-style targets (xyxy abs boxes) to RF-DETR format
    (cxcywh normalised boxes, per-instance masks, ``orig_size`` tensor).

    The conversion is in-place free — returns fresh dicts so the original
    targets remain available for torchmetrics' GT side.
    """
    out: List[Dict[str, torch.Tensor]] = []
    for t, (h, w) in zip(targets, image_sizes):
        boxes = t["boxes"].float()
        cx = (boxes[:, 0] + boxes[:, 2]) / 2.0 / w
        cy = (boxes[:, 1] + boxes[:, 3]) / 2.0 / h
        bw = (boxes[:, 2] - boxes[:, 0]) / w
        bh = (boxes[:, 3] - boxes[:, 1]) / h
        dt: Dict[str, torch.Tensor] = {
            "boxes": torch.stack([cx, cy, bw, bh], dim=-1),
            "labels": t["labels"],
            "orig_size": torch.tensor([h, w], device=boxes.device),
        }
        if "masks" in t:
            dt["masks"] = t["masks"].float()
        out.append(dt)
    return out


def _detr_results_to_torchmetrics(
    results: List[Dict[str, torch.Tensor]],
    score_threshold: float,
) -> List[Dict[str, torch.Tensor]]:
    """Filter PostProcess output by score and reshape for torchmetrics."""
    out: List[Dict[str, torch.Tensor]] = []
    for r in results:
        scores = r["scores"].float()
        keep = scores > score_threshold
        d: Dict[str, torch.Tensor] = {
            "boxes": r["boxes"][keep].float(),
            "scores": scores[keep],
            "labels": r["labels"][keep],
        }
        if "masks" in r:
            d["masks"] = r["masks"][keep].squeeze(1).to(torch.uint8)
        out.append(d)
    return out


def _targets_to_torchmetrics(
    targets: List[Dict[str, torch.Tensor]],
) -> List[Dict[str, torch.Tensor]]:
    out: List[Dict[str, torch.Tensor]] = []
    for t in targets:
        d: Dict[str, torch.Tensor] = {
            "boxes": t["boxes"].float(),
            "labels": t["labels"],
        }
        if "masks" in t:
            d["masks"] = t["masks"].to(torch.uint8)
        out.append(d)
    return out


def _build_loader(
    dataset,
    batch_size: int,
    num_workers: int,
    *,
    is_train: bool,
    dist_env: DistInfo,
    seed: int = 42,
) -> torch.utils.data.DataLoader:
    """Construct a DDP-aware DataLoader for one detection dataset."""
    if dist.is_available() and dist.is_initialized() and dist_env.world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=dist_env.world_size,
            rank=dist_env.rank,
            shuffle=is_train,
            seed=seed,
            # train drops trailing partial batches. For eval drop_last=False
            # keeps every sample, but DistributedSampler then PADS the dataset
            # with up to world_size-1 duplicates so each rank gets an equal
            # count — and torchmetrics MeanAveragePrecision pools detections
            # without image_id keying, so those duplicates are double-counted.
            # Run the authoritative eval on a single GPU (see the warning in
            # _setup_datamodule); multi-GPU val mAP is approximate.
            drop_last=is_train,
        )
        shuffle_arg = False
    else:
        sampler = None
        shuffle_arg = is_train
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_arg,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=detection_collate_fn,
        pin_memory=True,
        drop_last=is_train,
        persistent_workers=(num_workers > 0),
        # Reseed numpy + Python random per DataLoader worker. PyTorch already
        # reseeds torch RNG per worker; np/random would otherwise inherit the
        # main-process state once and never refresh.
        worker_init_fn=seed_worker,
    )


def _build_optimizer(
    model: nn.Module,
    criterion: Optional[nn.Module],
    *,
    lr: float,
    lr_encoder: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """AdamW with two param groups: backbone (``lr_encoder``) + head (``lr``).

    Norm / bias / position-embedding parameters get zero weight decay
    (standard transformer hygiene). Criterion parameters (rare — typically
    none) flow into the head group so they get the higher LR.
    """
    head_decay, head_no_decay = [], []
    bb_decay, bb_no_decay = [], []

    def _is_no_decay(name: str) -> bool:
        return any(k in name for k in ("bias", "norm", "pos_embed", "gamma"))

    unwrapped = unwrap_ddp_if_wrapped(model)
    for n, p in unwrapped.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("backbone."):
            (bb_no_decay if _is_no_decay(n) else bb_decay).append(p)
        else:
            (head_no_decay if _is_no_decay(n) else head_decay).append(p)

    if criterion is not None:
        for n, p in criterion.named_parameters():
            if not p.requires_grad:
                continue
            (head_no_decay if _is_no_decay(n) else head_decay).append(p)

    param_groups = [
        {"params": head_decay, "lr": lr, "weight_decay": weight_decay},
        {"params": head_no_decay, "lr": lr, "weight_decay": 0.0},
        {"params": bb_decay, "lr": lr_encoder, "weight_decay": weight_decay},
        {"params": bb_no_decay, "lr": lr_encoder, "weight_decay": 0.0},
    ]
    # Drop empty groups so the LR scheduler index stays sensible.
    param_groups = [g for g in param_groups if len(g["params"]) > 0]
    return torch.optim.AdamW(param_groups)


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
    lr_min: float,
    warmup_lr_init: float,
):
    """timm `CosineLRScheduler` with linear warmup, stepped per-batch."""
    from timm.scheduler.cosine_lr import CosineLRScheduler

    return CosineLRScheduler(
        optimizer,
        t_initial=total_steps,
        lr_min=lr_min,
        warmup_t=warmup_steps,
        warmup_lr_init=warmup_lr_init,
        warmup_prefix=True,
        t_in_epochs=False,
        cycle_limit=1,
    )


# ---------------------------------------------------------------------------
#  Recipe
# ---------------------------------------------------------------------------


class TrainUSRFDETRRecipe:
    """Per-dataset US-RFDETR training recipe (DDP-aware).

    Dual-config convention (matches ``TrainSonobaseRecipe``):

    * ``self.cfg`` — :class:`ConfigNode`, snapshotted alongside checkpoints.
    * ``self.hydra_cfg`` — :class:`omegaconf.DictConfig`, used by
      ``hydra.utils.instantiate`` for recursive object construction.
    """

    def __init__(self, cfg: ConfigNode, hydra_cfg: DictConfig):
        self.cfg = cfg
        self.hydra_cfg = hydra_cfg

    # ------------------ build phase ------------------
    def setup(self) -> None:
        hcfg = self.hydra_cfg

        # Distributed + logging
        self.dist_env: DistInfo = _ensure_distributed(hcfg.get("dist_env", {}))
        setup_logging()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        self.seed = int(hcfg.get("seed", 42))
        self.device = self.dist_env.device

        # RNG: seed once at recipe entry, before any model / data construction.
        # `ranked=True` adds dist.get_rank() so each DDP rank gets a distinct
        # sequence. State_dict / load_state_dict is unused here (us_rfdetr does
        # not support mid-run resume) but harmless.
        self.rng = StatefulRNG(seed=self.seed, ranked=True)

        # CUDA flags
        cuda_cfg = hcfg.get("cuda", {})
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = cuda_cfg.get(
                "cudnn_deterministic", False
            )
            torch.backends.cudnn.benchmark = cuda_cfg.get("cudnn_benchmark", True)
            allow_tf32 = cuda_cfg.get("allow_tf32", False)
            torch.backends.cuda.matmul.allow_tf32 = cuda_cfg.get(
                "matmul_allow_tf32", allow_tf32
            )
            torch.backends.cudnn.allow_tf32 = cuda_cfg.get(
                "cudnn_allow_tf32", allow_tf32
            )

        # Output dirs (rank 0 mkdirs; other ranks just compute the paths)
        save_dir = hcfg.get("checkpoint", {}).get(
            "save_dir", "./experiments/us_rfdetr/default"
        )
        self.output_dir = pathlib.Path(save_dir)
        self.results_dir = self.output_dir / "results"
        self.checkpoints_dir = self.output_dir / "checkpoints"
        if self.dist_env.is_main:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.results_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        # Data module — recipe-side DataLoader construction
        self._setup_datamodule(hcfg)

        # Model + criterion + postprocess
        self._setup_model(hcfg)

        # Optimizer + LR scheduler + AMP scaler
        self._setup_optim(hcfg)

        # Per-dataset MeanAveragePrecision metrics
        self._setup_metrics(hcfg)

        # Metric loggers (rank 0 only)
        if self.dist_env.is_main:
            self.metric_logger_train = build_metric_logger(
                str(self.output_dir / "training.jsonl"), flush=True
            )
            self.metric_logger_train.buffer_size = 1
            self.metric_logger_valid = build_metric_logger(
                str(self.output_dir / "validation.jsonl"), flush=True
            )
            self.metric_logger_valid.buffer_size = 1
            self.metric_logger_test = build_metric_logger(
                str(self.output_dir / "test.jsonl"), flush=True
            )
            self.metric_logger_test.buffer_size = 1
        else:
            self.metric_logger_train = None
            self.metric_logger_valid = None
            self.metric_logger_test = None

        # Best-mAP tracker (rank 0 owns the bookkeeping; broadcast decisions)
        self.best_val_map: float = -1.0
        self.best_epoch: int = -1

        # Logging cadence
        self.log_freq = int(hcfg.get("logging", {}).get("log_freq", 10))

        barrier()
        if self.dist_env.is_main:
            logger.info(
                f"TrainUSRFDETRRecipe ready — "
                f"dataset={self.dataset_name}, num_classes={self.num_classes}, "
                f"backbone={self._backbone_label}, "
                f"world_size={self.dist_env.world_size}, "
                f"output={self.output_dir}"
            )

    # ------ setup helpers ------
    def _setup_datamodule(self, hcfg: DictConfig) -> None:
        """Instantiate the data module + build train/val/test DataLoaders."""
        dm = hydra_instantiate(hcfg.data_module, _convert_="all")
        dm.setup()
        self.data_module = dm
        self.dataset_name = getattr(dm, "dataset_name", "unknown")
        self.num_classes = int(getattr(dm, "num_classes", 1))

        scratch = hcfg.get("scratch", {})
        bs = int(scratch.get("batch_size", hcfg.get("batch_size", 4)))
        nw = int(scratch.get("num_workers", hcfg.get("num_workers", 8)))

        if dm.train_dataset is None:
            raise ValueError(
                f"DataModule {type(dm).__name__} produced no training dataset. "
                "Check that the split file exists under DET_ANNOTATION_DIR."
            )
        self.train_loader = _build_loader(
            dm.train_dataset, batch_size=bs, num_workers=nw,
            is_train=True, dist_env=self.dist_env, seed=self.seed,
        )

        self.val_loaders: List[torch.utils.data.DataLoader] = [
            _build_loader(
                d, batch_size=bs, num_workers=nw,
                is_train=False, dist_env=self.dist_env, seed=self.seed,
            )
            for d in dm.val_datasets
        ]
        self.val_dataset_names: List[str] = list(dm.val_dataset_names)

        self.test_loaders: List[torch.utils.data.DataLoader] = [
            _build_loader(
                d, batch_size=bs, num_workers=nw,
                is_train=False, dist_env=self.dist_env, seed=self.seed,
            )
            for d in dm.test_datasets
        ]
        self.test_dataset_names: List[str] = list(dm.test_dataset_names)

        if self.dist_env.is_main:
            logger.info(
                f"Data: train batches/rank={len(self.train_loader)}, "
                f"val loaders={len(self.val_loaders)} ({self.val_dataset_names}), "
                f"test loaders={len(self.test_loaders)} ({self.test_dataset_names})"
            )
            if self.dist_env.world_size > 1 and (self.val_loaders or self.test_loaders):
                ws = self.dist_env.world_size
                logger.warning(
                    "Eval is running under world_size=%d. DistributedSampler "
                    "pads each eval loader with up to %d duplicate sample(s) so "
                    "every rank gets an equal count, and torchmetrics "
                    "MeanAveragePrecision pools detections without image_id "
                    "keying — so those duplicates are double-counted. Reported "
                    "val/test mAP is therefore biased by <=%d duplicated "
                    "image(s) and is non-deterministic across runs. Run the "
                    "authoritative test pass on a single GPU (world_size=1) for "
                    "an exact, reproducible number.",
                    ws, ws - 1, ws - 1,
                )

    def _setup_model(self, hcfg: DictConfig) -> None:
        """Instantiate the SAM2 image encoder + RF-DETR via ``build_rfdetr``."""
        model_cfg = hcfg.model
        # Cosmetic label for logs — derived from image_encoder._target_
        ie_target = model_cfg.image_encoder._target_
        self._backbone_label = ie_target.rsplit(".", 1)[-1]

        # Wrap model construction in ScopedRNG so head init / dropout init /
        # any randomness inside the encoder/detector builders draws from a
        # deterministic, rank-adjusted seed regardless of upstream RNG state.
        # Mirrors the pattern used by sonobase/pretrain.py build_model.
        with ScopedRNG(seed=self.seed, ranked=True):
            # Recursively build the image encoder via Hydra
            image_encoder = hydra_instantiate(
                model_cfg.image_encoder, _recursive_=True, _convert_="all"
            )

            # build_rfdetr wires backbone + transformer + head + criterion + postprocess.
            # `num_classes` is sourced from the data module (overrides any
            # default in the model config) so the YAML stays dataset-agnostic.
            builder_kwargs = OmegaConf.to_container(model_cfg, resolve=True)
            builder_kwargs.pop("_target_", None)
            builder_kwargs.pop("image_encoder", None)
            # User-facing override wins; otherwise use data module's num_classes
            if "num_classes" not in builder_kwargs or builder_kwargs["num_classes"] is None:
                builder_kwargs["num_classes"] = self.num_classes

            components = build_rfdetr(image_encoder=image_encoder, **builder_kwargs)
        model = components["model"]
        self.criterion: nn.Module = components["criterion"]
        self.postprocess: nn.Module = components["postprocess"]

        model.to(self.device)
        self.criterion.to(self.device)

        # DDP wrap (find_unused_parameters=True for group-DETR)
        if dist.is_available() and dist.is_initialized() and self.dist_env.world_size > 1:
            dist_cfg = hcfg.get("distributed", {})
            # With `model.segmentation_head=true` AND two_stage, the seg head
            # runs twice per forward — once over the decoder outputs, once over
            # the two-stage encoder outputs (`skip_blocks=True`, so a subset of
            # its parameters). The find_unused_parameters reducer marks each
            # grad ready exactly once and raises "Expected to mark a variable
            # ready only once" on the second use. static_graph is the supported
            # fix: it allows both repeated *and* unused parameters, and the
            # graph here is fixed across iterations. Opt-in, so bbox-only runs
            # keep the exact DDP behaviour their published numbers were run
            # with.
            static_graph = dist_cfg.get("static_graph", False)
            self.model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[int(os.environ.get("LOCAL_RANK", 0))] if torch.cuda.is_available() else None,
                find_unused_parameters=(
                    False if static_graph
                    else dist_cfg.get("find_unused_parameters", True)
                ),
                static_graph=static_graph,
            )
        else:
            self.model = model

        if self.dist_env.is_main:
            n_total = sum(p.numel() for p in model.parameters())
            n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(
                f"Model built — total params {n_total/1e6:.1f}M, "
                f"trainable {n_train/1e6:.1f}M "
                f"(backbone={self._backbone_label}, num_classes={self.num_classes})"
            )

    def _setup_optim(self, hcfg: DictConfig) -> None:
        """Build optimiser + per-step CosineLRScheduler + GradScaler."""
        opt_cfg = hcfg.get("optim", {})
        lr = float(opt_cfg.get("lr", 1e-4))
        lr_encoder = float(opt_cfg.get("lr_encoder", 1e-5))
        weight_decay = float(opt_cfg.get("weight_decay", 0.1))

        self.optimizer = _build_optimizer(
            self.model, self.criterion,
            lr=lr, lr_encoder=lr_encoder, weight_decay=weight_decay,
        )

        # Total steps = (steps per epoch on this rank) * max_epochs
        steps_per_epoch = max(1, len(self.train_loader))
        self.max_epochs = int(hcfg.get("trainer", {}).get("max_epochs", 24))
        self.total_steps = steps_per_epoch * self.max_epochs
        self.warmup_epochs = float(opt_cfg.get("warmup_epochs", 0.0))
        warmup_steps = int(steps_per_epoch * self.warmup_epochs)

        self.lr_scheduler = _build_scheduler(
            self.optimizer,
            total_steps=self.total_steps,
            warmup_steps=warmup_steps,
            lr_min=float(opt_cfg.get("lr_min", 1e-6)),
            warmup_lr_init=float(opt_cfg.get("warmup_lr_init", 1e-6)),
        )

        # AMP scaler — bf16 on most modern GPUs needs no scaling, but fp16
        # would. The shared `build_grad_scaler` honours the optim AMP cfg.
        self.scaler = build_grad_scaler(opt_cfg, device="cuda")

        # Gradient clip
        self.gradient_clip_val = float(
            hcfg.get("trainer", {}).get("gradient_clip_val", 0.1)
        )

        if self.dist_env.is_main:
            logger.info(
                f"Optim: AdamW (head_lr={lr}, encoder_lr={lr_encoder}, wd={weight_decay}); "
                f"total_steps={self.total_steps}, warmup_steps={warmup_steps}, "
                f"clip={self.gradient_clip_val}"
            )

    def _setup_metrics(self, hcfg: DictConfig) -> None:
        """One ``MeanAveragePrecision`` (bbox + segm) per val/test loader."""
        from torchmetrics.detection.mean_ap import MeanAveragePrecision

        def _make_metrics(n: int) -> List[Dict[str, MeanAveragePrecision]]:
            return [
                {
                    "bbox": MeanAveragePrecision(iou_type="bbox"),
                    "segm": MeanAveragePrecision(iou_type="segm"),
                }
                for _ in range(max(n, 0))
            ]

        self.val_metrics = _make_metrics(len(self.val_loaders))
        self.test_metrics = _make_metrics(len(self.test_loaders))

        # Inference-time score threshold for filtering low-confidence
        # detections before passing to torchmetrics. 0 = no filtering
        # (matches reviewer expectations for COCO-style mAP).
        self.score_threshold = float(
            hcfg.get("eval", {}).get("score_threshold", 0.0)
        )

    # ------------------ main entry ------------------
    def run(self) -> None:
        """Run training (with per-epoch validation) then final test."""
        self.run_train_loop()
        # After training, load best.pt and run the test loop on it.
        # If no validation happened (e.g. no val list) we just test the
        # final-epoch weights as-is.
        if self.dist_env.is_main and self.best_val_map >= 0:
            logger.info(
                f"Loading best checkpoint from epoch {self.best_epoch} "
                f"(val_bbox_mAP={self.best_val_map:.4f}) for test."
            )
        self._load_best_for_test()
        self.run_test_loop()

        if self.metric_logger_train is not None:
            self.metric_logger_train.close()
            self.metric_logger_valid.close()
            self.metric_logger_test.close()
        barrier()

    # ------------------ training loop ------------------
    def run_train_loop(self) -> None:
        """DDP per-epoch training loop with periodic validation."""
        for epoch in range(self.max_epochs):
            if hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(epoch)

            self.model.train()
            self.criterion.train()
            t_start = time.perf_counter()
            running_loss = 0.0
            n_seen = 0

            for step, (images, targets) in enumerate(self.train_loader):
                global_step = epoch * len(self.train_loader) + step
                images = [img.to(self.device, non_blocking=True) for img in images]
                targets = [
                    {k: (v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                     for k, v in t.items()}
                    for t in targets
                ]

                loss, loss_dict = self._forward_and_loss(images, targets)
                if not math.isfinite(loss.item()):
                    raise FloatingPointError(
                        f"Loss is {loss.item()} at epoch={epoch} step={step}; aborting."
                    )

                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                if self.gradient_clip_val > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        unwrap_ddp_if_wrapped(self.model).parameters(),
                        self.gradient_clip_val,
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.lr_scheduler.step_update(global_step)

                running_loss += float(loss.item()) * len(images)
                n_seen += len(images)

                if (
                    self.dist_env.is_main
                    and (step % self.log_freq == 0 or step + 1 == len(self.train_loader))
                ):
                    cur_lr = self.optimizer.param_groups[0]["lr"]
                    elapsed = time.perf_counter() - t_start
                    logger.info(
                        f"[train] epoch {epoch:3d} step {step:4d}/{len(self.train_loader)} "
                        f"loss={loss.item():.4f} "
                        f"lr={cur_lr:.2e} "
                        f"elapsed={elapsed:.1f}s"
                    )
                    self.metric_logger_train.log(
                        MetricsSample(
                            step=int(global_step),
                            epoch=int(epoch),
                            metrics={
                                "loss": float(loss.item()),
                                "lr": float(cur_lr),
                                **{f"loss/{k}": float(v.item())
                                   for k, v in loss_dict.items() if torch.is_tensor(v)},
                            },
                        )
                    )

            barrier()
            avg_loss = running_loss / max(n_seen, 1)
            if self.dist_env.is_main:
                logger.info(
                    f"[train] epoch {epoch} done — avg_loss={avg_loss:.4f} "
                    f"({time.perf_counter() - t_start:.1f}s)"
                )

            # Per-epoch validation (when a val loader exists)
            if self.val_loaders:
                val_map = self.run_val_loop(epoch)
                # Best-checkpoint decision is made on rank 0; broadcast a
                # 0/1 flag so every rank knows whether to barrier on the
                # write.
                self._maybe_save_best(val_map, epoch)

    # ------------------ forward + loss ------------------
    def _forward_and_loss(
        self,
        images: List[torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass + criterion → weighted-sum scalar loss + raw loss dict."""
        samples = nested_tensor_from_tensor_list(images)
        image_sizes = [(img.shape[-2], img.shape[-1]) for img in images]
        detr_targets = _convert_targets_to_detr(targets, image_sizes)

        autocast_ctx = build_autocast_context(self.hydra_cfg.get("optim", {}))
        with autocast_ctx:
            outputs = self.model(samples, detr_targets)
            loss_dict = self.criterion(outputs, detr_targets)
            weight_dict = self.criterion.weight_dict
            total_loss = sum(
                loss_dict[k] * weight_dict[k]
                for k in loss_dict
                if k in weight_dict
            )
        return total_loss, loss_dict

    # ------------------ validation / test loops ------------------
    @torch.no_grad()
    def run_val_loop(self, epoch: int) -> float:
        """Per-epoch validation loop. Returns the macro bbox-mAP across loaders."""
        return self._run_eval_loop(
            self.val_loaders, self.val_dataset_names,
            self.val_metrics, prefix="val", epoch=epoch,
        )

    @torch.no_grad()
    def run_test_loop(self) -> float:
        """Final test loop. Returns the macro bbox-mAP across test loaders."""
        return self._run_eval_loop(
            self.test_loaders, self.test_dataset_names,
            self.test_metrics, prefix="test", epoch=-1, write_json=True,
        )

    def _run_eval_loop(
        self,
        loaders: List[torch.utils.data.DataLoader],
        names: List[str],
        metrics: List[Dict[str, Any]],
        *,
        prefix: str,
        epoch: int,
        write_json: bool = False,
    ) -> float:
        """Shared eval loop body for val + test.

        For each loader, walks all batches with the model in eval mode,
        runs ``postprocess`` to convert raw outputs into per-sample
        boxes/scores/labels (+ masks if seg head is present), updates
        the per-loader ``MeanAveragePrecision`` (which all-reduces
        across DDP ranks internally on ``compute()``), and logs
        ``{prefix}/{name}/{bbox,segm}_mAP`` plus a macro average.
        """
        if not loaders:
            return float("nan")

        unwrapped_model = unwrap_ddp_if_wrapped(self.model)
        unwrapped_model.eval()
        self.criterion.eval()

        per_dataset_results: Dict[str, Dict[str, float]] = {}
        for ds_name, loader, mdict in zip(names, loaders, metrics):
            barrier()
            if self.dist_env.is_main:
                logger.info(f"[{prefix}] {ds_name} — start")

            if hasattr(loader.sampler, "set_epoch"):
                loader.sampler.set_epoch(epoch)

            t_start = time.perf_counter()
            n_seen_local = 0
            for step, (images, targets) in enumerate(loader):
                images = [img.to(self.device, non_blocking=True) for img in images]
                targets_on_dev = [
                    {k: (v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                     for k, v in t.items()}
                    for t in targets
                ]
                samples = nested_tensor_from_tensor_list(images)

                autocast_ctx = build_autocast_context(self.hydra_cfg.get("optim", {}))
                with autocast_ctx:
                    outputs = unwrapped_model(samples)

                # Sizes used for postprocessing (model returns predictions
                # in normalised coords; PostProcess scales to orig image).
                orig_sizes = torch.stack([
                    torch.tensor([img.shape[-2], img.shape[-1]], device=self.device)
                    for img in images
                ])
                results = self.postprocess(outputs, orig_sizes)

                preds = _detr_results_to_torchmetrics(results, self.score_threshold)
                gts = _targets_to_torchmetrics(targets_on_dev)
                mdict["bbox"].update(preds, gts)
                if any("masks" in p for p in preds):
                    mdict["segm"].update(preds, gts)

                n_seen_local += len(images)
                if self.dist_env.is_main and step % self.log_freq == 0:
                    elapsed = time.perf_counter() - t_start
                    logger.info(
                        f"[{prefix}] {ds_name} step {step:4d}/{len(loader)} "
                        f"local_samples={n_seen_local} elapsed={elapsed:.1f}s"
                    )

            # MeanAveragePrecision.compute() syncs across DDP ranks
            bbox_res = mdict["bbox"].compute()
            mdict["bbox"].reset()
            ds_record: Dict[str, float] = {
                "bbox_mAP": float(bbox_res["map"].item()),
                "bbox_mAP_50": float(bbox_res["map_50"].item()),
                "bbox_mAP_75": float(bbox_res["map_75"].item()),
            }
            if mdict["segm"].update_called:
                segm_res = mdict["segm"].compute()
                ds_record["segm_mAP"] = float(segm_res["map"].item())
                ds_record["segm_mAP_50"] = float(segm_res["map_50"].item())
                ds_record["segm_mAP_75"] = float(segm_res["map_75"].item())
            mdict["segm"].reset()
            per_dataset_results[ds_name] = ds_record

            if self.dist_env.is_main:
                metric_str = " | ".join(
                    f"{k}={v:.4f}" for k, v in ds_record.items()
                )
                logger.info(
                    f"[{prefix}] {ds_name} done — {metric_str} "
                    f"(elapsed={time.perf_counter() - t_start:.1f}s)"
                )

        # Macro average across loaders (use NaN-safe mean)
        bbox_maps = [
            v["bbox_mAP"] for v in per_dataset_results.values()
            if not math.isnan(v.get("bbox_mAP", float("nan")))
        ]
        macro_bbox = sum(bbox_maps) / len(bbox_maps) if bbox_maps else float("nan")

        if self.dist_env.is_main:
            logger.info(f"[{prefix}] macro bbox_mAP={macro_bbox:.4f}")
            self.metric_logger_valid.log(
                MetricsSample(
                    step=int(epoch),
                    epoch=int(epoch),
                    metrics={
                        f"{prefix}/macro_bbox_mAP": macro_bbox,
                        **{f"{prefix}/{k}/{m}": v
                           for k, rec in per_dataset_results.items()
                           for m, v in rec.items()},
                    },
                )
            ) if prefix == "val" else self.metric_logger_test.log(
                MetricsSample(
                    step=int(epoch),
                    epoch=int(epoch),
                    metrics={
                        f"{prefix}/macro_bbox_mAP": macro_bbox,
                        **{f"{prefix}/{k}/{m}": v
                           for k, rec in per_dataset_results.items()
                           for m, v in rec.items()},
                    },
                )
            )
            if write_json:
                blob = {
                    "datasets": per_dataset_results,
                    "macro_bbox_mAP": macro_bbox,
                    "best_epoch": self.best_epoch,
                    "best_val_bbox_mAP": self.best_val_map,
                }
                out = self.results_dir / "test_metrics.json"
                with out.open("w") as f:
                    json.dump(blob, f, indent=2)
                logger.info(f"Wrote {out}")

        barrier()
        return macro_bbox

    # ------------------ best-checkpoint bookkeeping ------------------
    def _maybe_save_best(self, val_map: float, epoch: int) -> None:
        """Atomically promote the current weights to ``best.pt`` if mAP improved.

        Decision is made on rank 0 (which owns ``self.best_val_map``);
        a 0/1 flag is broadcast so every rank synchronises on the
        ``barrier()`` after.
        """
        flag = torch.zeros(1, device=self.device)
        if self.dist_env.is_main:
            if not math.isnan(val_map) and val_map > self.best_val_map:
                self.best_val_map = float(val_map)
                self.best_epoch = int(epoch)
                self._write_best_pt(epoch=epoch, val_map=val_map)
                flag.fill_(1)

        if dist.is_available() and dist.is_initialized() and self.dist_env.world_size > 1:
            dist.broadcast(flag, src=0)
        barrier()

    def _write_best_pt(self, epoch: int, val_map: float) -> None:
        """Write ``best.pt`` (single-file, model state only). Rank-0 only."""
        unwrapped = unwrap_ddp_if_wrapped(self.model)
        state_dict = unwrapped.state_dict()
        out = self.checkpoints_dir / "best.pt"
        tmp = out.with_suffix(".pt.tmp")
        torch.save({
            "model": state_dict,
            "epoch": int(epoch),
            "val_bbox_mAP": float(val_map),
            "_provenance": {
                "torch_version": str(torch.__version__),
                "experiment_name": str(self.hydra_cfg.get("scratch", {}).get(
                    "experiment_name", "unknown"
                )),
                "backbone": self._backbone_label,
                "dataset": self.dataset_name,
                "num_classes": int(self.num_classes),
            },
        }, tmp)
        os.replace(tmp, out)
        logger.info(f"[best] epoch {epoch} val_bbox_mAP={val_map:.4f} -> {out}")

    def _load_best_for_test(self) -> None:
        """Load ``best.pt`` (if it exists) so the test loop runs on the best weights."""
        path = self.checkpoints_dir / "best.pt"
        if not path.is_file():
            if self.dist_env.is_main:
                logger.warning(
                    f"No best.pt at {path}; testing with final-epoch weights."
                )
            return

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        unwrapped = unwrap_ddp_if_wrapped(self.model)
        msg = unwrapped.load_state_dict(ckpt["model"], strict=True)
        if self.dist_env.is_main:
            logger.info(
                f"Loaded best.pt for test (epoch={ckpt.get('epoch')}, "
                f"val_mAP={ckpt.get('val_bbox_mAP')}; "
                f"missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)})"
            )
        barrier()


# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point — Hydra compose then ``recipe.setup()`` + ``recipe.run()``."""
    import sys

    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    config_dir: Optional[str] = None
    config_name: str = "train"
    overrides: List[str] = []
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] in ("-c", "--config-dir") and i + 1 < len(args):
            config_dir = str(pathlib.Path(args[i + 1]).resolve())
            i += 2
        elif args[i] in ("-cn", "--config-name") and i + 1 < len(args):
            config_name = args[i + 1]
            i += 2
        elif "=" in args[i] or args[i].startswith("+") or args[i].startswith("~"):
            overrides.append(args[i])
            i += 1
        else:
            i += 1

    if config_dir is None:
        config_dir = str(pathlib.Path(__file__).parent.resolve() / "configs")

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=config_dir, version_base=None)
    hydra_cfg = compose(config_name=config_name, overrides=overrides)
    OmegaConf.resolve(hydra_cfg)

    cfg_dict = OmegaConf.to_container(hydra_cfg, resolve=True)
    cfg_node = ConfigNode(cfg_dict)

    recipe = TrainUSRFDETRRecipe(cfg_node, hydra_cfg)
    recipe.setup()
    recipe.run()


if __name__ == "__main__":
    main()
