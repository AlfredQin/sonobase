import gc
import logging
import math
import os
import pathlib
import time
import wandb
from wandb import Settings
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
from hydra.utils import instantiate as hydra_instantiate
from omegaconf import DictConfig, OmegaConf

from nemo_automodel.components.checkpoint.checkpointing import Checkpointer, CheckpointingConfig
from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.distributed.init_utils import DistInfo, initialize_distributed
from nemo_automodel.components.loggers.log_utils import setup_logging
from nemo_automodel.components.loggers.metric_logger import MetricsSample, build_metric_logger
from nemo_automodel.components.training.rng import ScopedRNG, StatefulRNG
from nemo_automodel.components.training.step_scheduler import StepScheduler
from nemo_automodel.recipes.base_recipe import BaseRecipe

from nemo_cv.components.optim.optimizer import GradientClipper, SAM2Optimizer, construct_optimizer
from nemo_cv.components.datasets.sam2.data_utils import BatchedVideoDatapoint
from nemo_cv.components.training.utils import (
    CORE_LOSS_KEY,
    DurationMeter,
    Phase,
    barrier,
    build_autocast_context,
    build_grad_scaler,
    get_amp_type,
    human_readable_time,
    print_model_summary,
    unwrap_ddp_if_wrapped,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Stateless builder functions (all use hydra.utils.instantiate)
# ---------------------------------------------------------------------------


def build_distributed(cfg_dist: DictConfig) -> DistInfo:
    """Initialize distributed backend."""
    backend = cfg_dist.get("backend", "nccl")
    timeout = cfg_dist.get("timeout_minutes", 1)
    return initialize_distributed(backend=backend, timeout_minutes=timeout)


def build_model(
    cfg_model: DictConfig,
    device: torch.device,
    seed: int = 42,
    pretrained_ckpt_path: Optional[str] = None,
) -> nn.Module:
    """Build SAM2 model via hydra.utils.instantiate and move to device.

    Hydra recursively instantiates all nested _target_ entries (image_encoder,
    memory_attention, memory_encoder, etc.).

    If pretrained_ckpt_path is provided, loads SAM2 pretrained weights with
    strict=False. This initializes memory_attention, memory_encoder, mask_decoder,
    and prompt_encoder from a SAM2 checkpoint. Image encoder weights that don't
    match (e.g. TriBranchTrunk vs standard Hiera) are silently skipped since the
    image encoder handles its own pretrained loading via ckpt_path0 / timm.

    On resume, BaseRecipe.load_checkpoint() overwrites all weights from the DCP
    checkpoint, so this pretrained loading is harmless.
    """
    with ScopedRNG(seed=seed, ranked=True):
        model = hydra_instantiate(cfg_model, _recursive_=True, _convert_="all")

        if pretrained_ckpt_path is not None and os.path.isfile(pretrained_ckpt_path):
            sd = torch.load(pretrained_ckpt_path, map_location="cpu", weights_only=True)
            if "model" in sd:
                sd = sd["model"]
            missing, unexpected = model.load_state_dict(sd, strict=False)
            logger.info(
                f"Loaded SAM2 pretrained weights from {pretrained_ckpt_path} "
                f"(missing={len(missing)}, unexpected={len(unexpected)})"
            )
        elif pretrained_ckpt_path is not None:
            logger.warning(f"Pretrained checkpoint not found: {pretrained_ckpt_path}")

        print_model_summary(model)
        model.to(device)
    return model


def build_loss_fn(cfg_loss: DictConfig) -> nn.ModuleDict:
    """Build loss functions as nn.ModuleDict keyed by dataset name."""
    loss_dict = {}
    for key in cfg_loss:
        loss_dict[key] = hydra_instantiate(cfg_loss[key])
    return nn.ModuleDict(loss_dict)


def build_datasets(cfg_ds: DictConfig, mode: str = "train") -> dict:
    """Build train and val dataset objects via hydra.utils.instantiate.

    Training: single TorchTrainMixedDataset (batches can mix sub-datasets).
    Validation: dict of {dataset_name: VOSDataset} — one per sub-dataset for
    per-dataset metric tracking. Each is wrapped in its own DataLoader.
    """
    datasets = {}
    if mode in ["train", "train_only"]:
        datasets["train"] = hydra_instantiate(cfg_ds.train, _recursive_=True, _convert_="all")
    if mode in ["train", "val"]:
        val_cfg = OmegaConf.select(cfg_ds, "val", default=None)
        if val_cfg is not None:
            val_datasets = {}
            for ds_name in val_cfg:
                val_datasets[ds_name] = hydra_instantiate(val_cfg[ds_name], _recursive_=True, _convert_="all")
            datasets["val"] = val_datasets
    return datasets


def build_optimizer(cfg_opt: DictConfig, model: nn.Module) -> SAM2Optimizer:
    """Build SAM2 optimizer with per-param-group schedulers."""
    return construct_optimizer(
        model=model,
        optimizer_conf=cfg_opt.optimizer,
        options_conf=cfg_opt.get("options", None),
        param_group_modifiers_conf=cfg_opt.get("param_group_modifiers", None),
    )


def build_gradient_clipper(cfg_opt: DictConfig) -> Optional[GradientClipper]:
    """Build gradient clipper from config."""
    clip_cfg = cfg_opt.get("gradient_clip", None)
    if clip_cfg is None:
        return None
    return hydra_instantiate(clip_cfg)


def build_checkpoint_config(cfg_ckpt: DictConfig) -> CheckpointingConfig:
    """Build checkpoint configuration."""
    save_dir = cfg_ckpt.get("save_dir", "./checkpoints")
    ckpt_kwargs = dict(
        enabled=cfg_ckpt.get("enabled", True),
        checkpoint_dir=save_dir,
        model_save_format=cfg_ckpt.get("model_save_format", "torch_save"),
        save_consolidated=cfg_ckpt.get("save_consolidated", False),
        model_cache_dir=save_dir,
        model_repo_id="",
        is_peft=False,
    )
    return CheckpointingConfig(**ckpt_kwargs)


def build_step_scheduler(cfg_step_sched: DictConfig, dataloader, max_epochs: int, dp_size: int = 1) -> StepScheduler:
    """Build StepScheduler for training iteration bookkeeping."""
    kwargs = OmegaConf.to_container(cfg_step_sched, resolve=True) if cfg_step_sched else {}
    defaults = dict(
        num_epochs=max_epochs,
        global_batch_size=1,
        local_batch_size=1,
        dp_size=dp_size,
        ckpt_every_steps=len(dataloader) if dataloader is not None else 1000,
        dataloader=dataloader,
    )
    defaults.update(kwargs)
    defaults["num_epochs"] = max_epochs
    return StepScheduler(**defaults)


def build_wandb(cfg: DictConfig):
    """Initialize W&B if configured."""
    kwargs = OmegaConf.to_container(cfg.wandb, resolve=True)
    # if kwargs.get("name", "") == "":
    #     kwargs["name"] = "_".join(_get_model_name(cfg.model).split("/")[-2:])
    return wandb.init(**kwargs, config=OmegaConf.to_container(cfg, resolve=True), settings=Settings(silent=True))


def setup_ddp(model: nn.Module, dist_env: DistInfo, cfg_dist: DictConfig) -> nn.Module:
    """Wrap model in DistributedDataParallel."""
    find_unused = cfg_dist.get("find_unused_parameters", False)
    comms_dtype = cfg_dist.get("comms_dtype", None)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device_ids = [local_rank] if torch.cuda.is_available() else []
    model = nn.parallel.DistributedDataParallel(
        model, device_ids=device_ids, find_unused_parameters=find_unused,
    )

    if comms_dtype is not None:
        from torch.distributed.algorithms import ddp_comm_hooks

        amp_type = get_amp_type(comms_dtype)
        hook = (
            ddp_comm_hooks.default_hooks.bf16_compress_hook
            if amp_type == torch.bfloat16
            else ddp_comm_hooks.default_hooks.fp16_compress_hook
        )
        model.register_comm_hook(None, hook)
        logger.info(f"Enabled {comms_dtype} gradient communication hook")

    return model


# ---------------------------------------------------------------------------
#  Trainer class – orchestration only
# ---------------------------------------------------------------------------


class TrainSonobaseRecipe(BaseRecipe):
    """Training recipe for SAM2 video segmentation models.

    Uses dual config objects:
    - self.cfg (ConfigNode): tracked by BaseRecipe for checkpoint config.yaml snapshots
    - self.hydra_cfg (OmegaConf DictConfig): used by hydra.utils.instantiate in builders

    SAM2-specific adaptations:
    - Per-epoch DataLoader via dataset.get_loader(epoch)
    - Per-dataset-key loss dispatch via batch.dict_key
    - "where"-based LR scheduling (fractional epoch progress)
    """

    EPSILON = 1e-8

    def __init__(self, cfg: ConfigNode, hydra_cfg: DictConfig):
        self.cfg = cfg
        self.hydra_cfg = hydra_cfg

    # ------------------ build phase ------------------
    def setup(self):
        """Build all components needed for training."""
        hcfg = self.hydra_cfg

        torch.cuda.reset_peak_memory_stats()
        self.dist_env = build_distributed(hcfg.get("dist_env", {}))
        setup_logging()

        self.mode = hcfg.get("mode", "train")
        self.max_epochs = hcfg.get("max_epochs", 10)
        self.val_epoch_freq = hcfg.get("val_epoch_freq", 1)
        self.seed = hcfg.get("seed", 42)
        self.rng = StatefulRNG(seed=self.seed, ranked=True)
        self.device = self.dist_env.device

        # W&B
        self.wandb_run = None
        if self.dist_env.is_main and hasattr(hcfg, "wandb"):
            self.wandb_run = build_wandb(hcfg)
            if self.wandb_run is not None:
                logger.info(f"W&B run: {self.wandb_run.url}")

        self._log_experiment_details()
        self._log_library_versions()

        # RNG seeding for pretrain is owned by `StatefulRNG(seed, ranked=True)`
        # above — it calls init_all_rng on construction (rank-adjusted, with
        # checkpoint state_dict support for resume). Don't call
        # `seed_everything` here: it would overwrite the rank adjustment.

        # CUDA configuration — nested under `cuda:` in the YAML (matches the
        # save_predictions / test_sam2 / few_shot recipes). Defaults match the
        # YAML's stated intent: explicit non-determinism (benchmark=true) for
        # throughput; flip to (cudnn_deterministic=true, cudnn_benchmark=false)
        # in the YAML for bit-deterministic pretrain.
        cuda_cfg = hcfg.get("cuda", {})
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = cuda_cfg.get("cudnn_deterministic", False)
            torch.backends.cudnn.benchmark = cuda_cfg.get("cudnn_benchmark", True)
            allow_tf32 = cuda_cfg.get("allow_tf32", False)
            torch.backends.cuda.matmul.allow_tf32 = cuda_cfg.get("matmul_allow_tf32", allow_tf32)
            torch.backends.cudnn.allow_tf32 = cuda_cfg.get("cudnn_allow_tf32", allow_tf32)

        # Loss
        self.loss_fn = build_loss_fn(hcfg.loss_fn)
        if self.loss_fn is not None:
            self.loss_fn.to(self.device)

        # Checkpoint
        checkpoint_config = build_checkpoint_config(hcfg.get("checkpoint", {}))
        self.checkpointer = Checkpointer(
            config=checkpoint_config, dp_rank=self.dist_env.rank, tp_rank=0, pp_rank=0, moe_mesh=None,
        )

        # Model (hydra.utils.instantiate handles recursive _target_ resolution)
        # pretrained_ckpt_path loads SAM2 weights for memory/decoder parts on fresh training;
        # on resume, load_checkpoint() overwrites everything from the DCP checkpoint.
        model = build_model(hcfg.model, self.device, seed=self.seed, pretrained_ckpt_path=hcfg.get("pretrained_ckpt_path", None))

        # Optimizer
        self.optim = build_optimizer(hcfg.optim, model)
        self.optimizer = self.optim.optimizer
        self.gradient_clipper = build_gradient_clipper(hcfg.optim)
        # AMP
        self.scaler = build_grad_scaler(hcfg.optim, device="cuda")
        self.peft_config = None

        # DDP wrapping — set self.model only once (BaseRecipe tracks it)
        self.model = setup_ddp(model, self.dist_env, hcfg.get("distributed", {}))

        # Datasets
        # Training: single TorchTrainMixedDataset
        # Validation: dict of {dataset_name: VOSDataset} for per-dataset metrics
        self.datasets = build_datasets(hcfg.data, self.mode)
        self.train_dataset = self.datasets.get("train", None)
        self.val_datasets = self.datasets.get("val", {})

        # StepScheduler
        self.step_scheduler = build_step_scheduler(
            hcfg.get("step_scheduler", None),
            self.train_dataset.get_loader(epoch=0) if self.train_dataset else None,
            self.max_epochs,
            dp_size=self._get_dp_group_size(),
        )
        self._setup_garbage_collection(self.step_scheduler)

        # Metric loggers
        ckpt_dir = pathlib.Path(self.checkpointer.config.checkpoint_dir)
        os.makedirs(ckpt_dir, exist_ok=True)
        self.metric_logger_train = build_metric_logger(ckpt_dir / "training.jsonl", flush=True)
        self.metric_logger_train.buffer_size = 1
        self.metric_logger_valid = build_metric_logger(ckpt_dir / "validation.jsonl", flush=True)
        self.metric_logger_valid.buffer_size = 1

        # Validation metrics (aggregate + per-dataset)
        val_metrics_cfg = hcfg.get("val_metrics", None)
        self.val_metrics = {}
        self.per_dataset_val_metrics = {}
        if val_metrics_cfg:
            for name in val_metrics_cfg:
                self.val_metrics[name] = hydra_instantiate(val_metrics_cfg[name]).to(self.device)
            for ds_name in self.val_datasets:
                self.per_dataset_val_metrics[ds_name] = {
                    name: hydra_instantiate(val_metrics_cfg[name]).to(self.device)
                    for name in val_metrics_cfg
                }

        # Resume from checkpoint
        self.load_checkpoint(hcfg.checkpoint.get("restore_from", None))

        # Timers
        self.start_time = time.time()
        self.est_epoch_time = {Phase.TRAIN: 0, Phase.VAL: 0}
        self.time_elapsed_meter = DurationMeter("Time Elapsed", self.device, ":.2f")

        # Logging config
        self.log_freq = hcfg.logging.get("log_freq", 20)

        self._log_step_scheduler_details(self.step_scheduler)

        barrier()
        logger.info("TrainSam2Recipe setup complete.")

    # ------------------ main loop ------------------
    def run_train_validation_loop(self):
        """Run the training and validation loop."""
        assert self.mode in ["train", "train_only", "val"]

        if self.mode == "val":
            self._run_validation()
            return

        start_epoch = self.step_scheduler.epoch
        if self.mode == "train" and start_epoch > 0:
            logger.info(f"Resuming training from epoch: {start_epoch}")
            if self._is_intermediate_val_epoch(start_epoch - 1):
                val_log_data = self._run_validation()
                self.log_val_metrics(val_log_data)

        self.model.train()
        self.timestamp = time.perf_counter()

        for epoch in self.step_scheduler.epochs:
            train_loader = self.train_dataset.get_loader(epoch=int(epoch))
            self.step_scheduler.dataloader = train_loader
            barrier()

            iters_per_epoch = len(train_loader)
            epoch_loss_sum = 0.0
            epoch_steps = 0

            # for data_iter, batch in enumerate(train_loader):
            for data_iter, batch in enumerate(self.step_scheduler):
                # TODO Need to implement the gradient accumulation
                assert isinstance(batch, List) and len(batch) == 1, "Expected a single batch"
                batch = batch[0]
                batch = batch.to(self.device, non_blocking=True)

                log_data = self._run_train_optim_step(batch, epoch, data_iter, iters_per_epoch)
                self.log_train_metrics(log_data)

                epoch_loss_sum += log_data.metrics["loss"]
                epoch_steps += 1

                # Validation + checkpoint (VLM pattern: val before ckpt so val_loss is available)
                val_loss = {}
                if self.mode == "train" and self.step_scheduler.is_val_step and self.val_datasets:
                    val_log_data = self._run_validation()
                    if val_log_data is not None:
                        val_loss["val_loss"] = val_log_data.metrics["val_loss"]
                        if "miou" in val_log_data.metrics:
                            val_loss["miou"] = val_log_data.metrics["miou"]
                        self.log_val_metrics(val_log_data)
                    self.model.train()

                if self.step_scheduler.is_ckpt_step:
                    avg_loss = epoch_loss_sum / max(epoch_steps, 1)
                    self.save_checkpoint(
                        epoch=epoch,
                        step=self.step_scheduler.step,
                        train_loss=avg_loss,
                        val_loss=val_loss if val_loss else None,
                        best_metric_key="val_loss",
                    )

                self._maybe_collect_garbage()

            avg_epoch_loss = epoch_loss_sum / max(epoch_steps, 1)
            logger.info(f"Epoch {epoch} avg_loss: {avg_epoch_loss:.4f}")

            del train_loader
            gc.collect()

        self.metric_logger_train.close()
        self.metric_logger_valid.close()
        self.checkpointer.close()

    # ------------------ forward / backward / optim ------------------
    def _forward_backward_step(self, batch: BatchedVideoDatapoint) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward + loss + backward (no optimizer step)."""
        autocast_ctx = build_autocast_context(self.hydra_cfg.optim)
        with autocast_ctx:
            outputs = self.model(batch)
            targets = batch.masks

            key = batch.dict_key
            loss = self.loss_fn[key](outputs, targets)

            extra_losses = {}
            if isinstance(loss, dict):
                extra_losses = {f"{key}_{k}": v for k, v in loss.items() if k != CORE_LOSS_KEY}
                loss = loss[CORE_LOSS_KEY]

        if not math.isfinite(loss.item()):
            raise FloatingPointError(f"Loss is {loss.item()}, stopping training")

        self.scaler.scale(loss).backward()

        return loss, extra_losses

    def _run_train_optim_step(
        self, batch: BatchedVideoDatapoint, epoch: int, data_iter: int, iters_per_epoch: int,
    ) -> MetricsSample:
        """Execute a single training step: zero_grad -> forward/backward -> optim."""
        self.optim.zero_grad(set_to_none=True)

        loss, extra_losses = self._forward_backward_step(batch)

        exact_epoch = epoch + float(data_iter) / iters_per_epoch
        where = float(exact_epoch) / self.max_epochs
        assert where <= 1 + self.EPSILON
        if where < 1.0:
            self.optim.step_schedulers(where, step=int(exact_epoch * iters_per_epoch))

        if self.gradient_clipper is not None:
            self.scaler.unscale_(self.optim.optimizer)
            self.gradient_clipper(model=self.model)

        self.scaler.step(self.optim.optimizer)
        self.scaler.update()

        self.time_elapsed_meter.update(time.time() - self.start_time)

        t = time.perf_counter()
        time_delta = t - self.timestamp
        self.timestamp = t

        metrics = {
            "loss": loss.item(),
            "lr": self.optim.optimizer.param_groups[0]["lr"],
            "mem": torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0,
            "where": where,
            "batch_time": time_delta,
        }
        for k, v in extra_losses.items():
            metrics[k] = v.item()

        return MetricsSample(step=self.step_scheduler.step, epoch=epoch, metrics=metrics)

    # ------------------ validation ------------------
    @torch.no_grad()
    def _run_validation(self) -> Optional[MetricsSample]:
        """Run validation across all val datasets with per-dataset metrics.

        Each val dataset gets its own DataLoader. Aggregate and per-dataset
        metrics (mIoU, Dice) are tracked separately via self.val_metrics and
        self.per_dataset_val_metrics (both created in setup()).
        """
        if not self.val_datasets:
            return None

        with ScopedRNG(seed=1, ranked=True):
            self.model.eval()

            total_loss = 0.0
            total_samples = 0
            val_epoch = int(self.step_scheduler.epoch)

            for ds_name, val_dataset in self.val_datasets.items():
                val_loader = val_dataset.get_loader(epoch=val_epoch)

                for data_iter, batch in enumerate(val_loader):
                    batch = batch.to(self.device, non_blocking=True)

                    autocast_ctx = build_autocast_context(self.hydra_cfg.optim)
                    with autocast_ctx:
                        outputs = self.model(batch)
                        targets = batch.masks
                        batch_size = batch.num_videos

                        key = batch.dict_key
                        loss = self.loss_fn[key](outputs, targets)
                        if isinstance(loss, dict):
                            loss = loss[CORE_LOSS_KEY]

                    total_loss += loss.item() * batch_size
                    total_samples += batch_size

                    # Update aggregate metrics
                    for m in self.val_metrics.values():
                        m.update(outputs, batch)

                    # Update per-dataset metrics
                    if ds_name in self.per_dataset_val_metrics:
                        for m in self.per_dataset_val_metrics[ds_name].values():
                            m.update(outputs, batch)

                    if data_iter % 10 == 0:
                        barrier()

                del val_loader

        # Aggregate loss across ranks
        total_loss_t = torch.tensor([total_loss], dtype=torch.float, device=self.device)
        total_samples_t = torch.tensor([total_samples], dtype=torch.long, device=self.device)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(total_loss_t)
            torch.distributed.all_reduce(total_samples_t)
        val_loss = total_loss_t.item() / max(total_samples_t.item(), 1)

        # Compute metrics (torchmetrics auto-syncs across GPUs)
        metrics_dict = {
            "val_loss": val_loss,
            "lr": self.optim.optimizer.param_groups[0]["lr"],
            "mem": torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0,
        }

        for name, m in self.val_metrics.items():
            metrics_dict[name] = m.compute().item()
            m.reset()

        for ds_name, ds_metrics in self.per_dataset_val_metrics.items():
            for name, m in ds_metrics.items():
                metrics_dict[f"{ds_name}/{name}"] = m.compute().item()
                m.reset()

        gc.collect()

        return MetricsSample(
            step=self.step_scheduler.step,
            epoch=self.step_scheduler.epoch,
            metrics=metrics_dict,
        )

    # ------------------ logging ------------------
    def log_train_metrics(self, log_data: MetricsSample):
        if not self.dist_env.is_main:
            return
        if self.wandb_run is not None:
            wandb.log(log_data.to_dict(), step=log_data.step)
        self.metric_logger_train.log(log_data)
        logger.info(
            "step {} | epoch {} | loss {:.4f} | lr {:.2e} | mem {:.2f} GiB | where {:.4f}".format(
                log_data.step, log_data.epoch, log_data.metrics["loss"],
                log_data.metrics["lr"], log_data.metrics["mem"], log_data.metrics["where"],
            )
        )
        torch.cuda.reset_peak_memory_stats()

    def log_val_metrics(self, log_data: Optional[MetricsSample]):
        if not self.dist_env.is_main or log_data is None:
            return
        if self.wandb_run is not None:
            wandb.log(log_data.to_dict(), step=log_data.step)
        self.metric_logger_valid.log(log_data)

        m = log_data.metrics
        parts = [
            f"[val] step {log_data.step}",
            f"epoch {log_data.epoch}",
            f"val_loss {m['val_loss']:.4f}",
            f"lr {m['lr']:.2e}",
        ]
        if "miou" in m:
            parts.append(f"miou {m['miou']:.4f}")
        if "dice" in m:
            parts.append(f"dice {m['dice']:.4f}")
        for k, v in m.items():
            if "/" in k:
                parts.append(f"{k} {v:.4f}")
        logger.info(" | ".join(parts))

    # ------------------ helpers ------------------
    def _is_intermediate_val_epoch(self, epoch: int) -> bool:
        return epoch % self.val_epoch_freq == 0 and epoch < self.max_epochs - 1

    def _get_dp_group_size(self, include_cp: bool = False) -> int:
        """Get data parallel world size, handling DDP mode where device_mesh is None."""
        # In DDP mode, device_mesh is None, so use torch.distributed directly
        device_mesh = getattr(self, "device_mesh", None)
        if device_mesh is None:
            return dist.get_world_size() if dist.is_initialized() else 1
        # Otherwise, use the parent implementation
        return super()._get_dp_group_size(include_cp=include_cp)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    """Main entry point for SAM2 training.

    Uses Hydra for config composition and resolution, then creates both
    OmegaConf DictConfig (for hydra.utils.instantiate) and ConfigNode
    (for BaseRecipe checkpoint tracking).

    Usage:
        uv run python -m nemo_cv.recipes.sonobase.pretrain -c ./configs/pretrain
        uv run python -m nemo_cv.recipes.sonobase.pretrain -c ./configs/pretrain experiment=pretrain_hiera_b_conv_s_conv_t_on_38_pt_8_bm
        uv run torchrun --nproc-per-node=8 -m nemo_cv.recipes.sonobase.pretrain -c ./configs/pretrain scratch.num_epochs=10
    """
    import sys

    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    config_dir = None
    config_name = "train"
    overrides = []
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

    recipe = TrainSonobaseRecipe(cfg_node, hydra_cfg)
    recipe.setup()
    recipe.run_train_validation_loop()



if __name__ == "__main__":
    main()
