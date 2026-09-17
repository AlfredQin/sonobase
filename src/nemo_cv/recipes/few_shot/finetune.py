"""Few-shot fine-tuning recipe (single-GPU, configurable per-module freeze).

Implements the few-shot fine-tuning protocol:

  * Single-GPU (no DDP).
  * Configurable per-module freeze (`freeze.*` in finetune.yaml): by
    default only `mask_decoder` trains; `image_encoder`, `memory_encoder`
    and `prompt_encoder` are frozen. Frozen modules still run in forward()
    — they just accumulate no gradient.
  * Reuses the existing pretrain stack as much as possible:
      - `SAM2Train.forward()` for the forward pass.
      - `MultiStepMultiMasksAndIous` for the loss (the same one pretrain uses).
      - `SaUSRawDataset` / `JSONRawDataset` + `collate_fn` for data loading,
        with a custom `file_list_txt` materialized by
        `nemo_cv.components.few_shot.splits.materialize_subset` from the
        deterministic (dataset, N, seed) recipe.
      - The existing pretrain transforms (HFlip + RandomAffine), but with
        the milder `degrees=15, scale=[0.9,1.1]` overridden in the
        few-shot scratch config.
  * Optimizer: AdamW(lr=1e-4, wd=0.01); cosine LR to 0; grad clip = 1.0.
  * Epoch counts depend on N (50 for N≤5, 30 for N∈{10,20}, 20 for N=50).
    No early stopping.
  * Saves the final model state_dict as a `.pt` file in the same shape
    Meta's released SAM2 checkpoints use (`{"model": <sd>, ...}`), so it
    plugs straight into the existing `save_predictions.py` recipe at eval
    time without any conversion.

Usage (typically from `scripts/few_shot/_finetune_one.sh`):

    cd src
    uv run python -m nemo_cv.recipes.few_shot.finetune \\
      -c ./configs/few_shot -cn finetune \\
      data=ACOUSLIC \\
      ckpt_path=<base .pt> \\
      scratch.dataset_name=ACOUSLIC \\
      scratch.model_label=sonobase \\
      scratch.N=5 \\
      scratch.fewshot_seed=42 \\
      +model/image_encoder@model.image_encoder=hiera_b_conv_s_conv_t

CLI mirrors the analysis Stage-1 conventions; the recipe runs once per
(dataset, model, N, seed) combination.
"""

from __future__ import annotations

import json
import logging
import math
import os
import pathlib
import sys
import time
from contextlib import nullcontext
from typing import List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate as hydra_instantiate
from omegaconf import DictConfig, OmegaConf

from nemo_cv.components.few_shot.splits import materialize_subset
from nemo_cv.components.training.utils import (
    CORE_LOSS_KEY,
    build_autocast_context,
    seed_everything,
)
from nemo_cv.recipes.sonobase.pretrain import build_loss_fn, build_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# N-aware epoch schedule
# ---------------------------------------------------------------------------


def epochs_for_n(N: int, schedule: DictConfig) -> int:
    """Map N to the protocol's training-epoch count.

    schedule yaml fragment:
      small:  {N_max: 5,   epochs: 50}
      medium: {N_max: 20,  epochs: 30}
      large:  {N_max: 100, epochs: 20}
    """
    for tier in ("small", "medium", "large"):
        bucket = schedule.get(tier)
        if bucket is None:
            continue
        if N <= int(bucket["N_max"]):
            return int(bucket["epochs"])
    raise ValueError(f"N={N} exceeds every bucket in epoch_schedule.")


# ---------------------------------------------------------------------------
# Per-module freeze
# ---------------------------------------------------------------------------

# Maps a `freeze.<name>` config flag to the SAM2Base submodule attribute it
# controls. Submodules NOT listed here (memory_attention, obj-ptr
# projections, and the bare SAM2 nn.Parameters) are always frozen — they are
# not part of the few-shot tuning surface.
_FREEZE_MODULES = {
    "image_encoder": "image_encoder",
    "mask_decoder": "sam_mask_decoder",
    "memory_encoder": "memory_encoder",
    "prompt_encoder": "sam_prompt_encoder",
}

# Defaults used when a `freeze.<name>` flag is absent: the few-shot protocol
# fine-tunes the mask decoder only.
_FREEZE_DEFAULTS = {
    "image_encoder": True,
    "mask_decoder": False,
    "memory_encoder": True,
    "prompt_encoder": True,
}


def apply_module_freeze(model: nn.Module, freeze_cfg: DictConfig) -> dict:
    """Freeze / unfreeze SAM2 submodules per the few-shot `freeze` config.

    Every parameter is frozen first; then each module whose `freeze.<name>`
    flag resolves to False is unfrozen. Submodules outside `_FREEZE_MODULES`
    (memory_attention, obj-ptr projections, bare SAM2 parameters) stay
    frozen — they are not part of the few-shot tuning surface.

    Frozen modules still **run** in forward(); they just accumulate no
    gradient. Returns a ``{flag_name: trainable_bool}`` report.
    """
    # 1. Freeze everything.
    for p in model.parameters():
        p.requires_grad = False

    # 2. Unfreeze each module whose freeze flag resolves to False.
    report: dict = {}
    for flag, attr in _FREEZE_MODULES.items():
        do_freeze = bool(freeze_cfg.get(flag, _FREEZE_DEFAULTS[flag]))
        sub = getattr(model, attr, None)
        if sub is None:
            raise AttributeError(
                f"Model has no `{attr}` attribute — cannot apply the "
                f"`freeze.{flag}` setting. Check that you're loading "
                "SAM2Train, not an unwrapped sub-component."
            )
        if not do_freeze:
            for p in sub.parameters():
                p.requires_grad = True
        report[flag] = not do_freeze

    # 3. At least one module must train, else the run is a no-op.
    if not any(report.values()):
        raise RuntimeError(
            "Every module is frozen — fine-tuning would be a no-op. "
            "Set at least one `freeze.*` flag to false "
            "(e.g. freeze.mask_decoder=false)."
        )
    return report


# ---------------------------------------------------------------------------
# Recipe
# ---------------------------------------------------------------------------


class FewShotFinetuneRecipe:
    """Single-GPU, decoder-only fine-tuning of SAM2-style models on N samples."""

    def __init__(self, hydra_cfg: DictConfig):
        self.hydra_cfg = hydra_cfg

    # ------------------ setup ------------------
    @staticmethod
    def _ensure_single_process_group() -> None:
        """Init a 1-rank gloo process group so DistributedSampler doesn't crash.

        `TorchTrainMixedDataset.get_loader` always wraps with
        DistributedSampler, which calls `dist.get_world_size()` even on
        single-GPU. Same trick `convert_sonobase_dcp_to_pt.py` uses.
        """
        if dist.is_initialized():
            return
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        # Pick a free port to avoid clashing with other processes.
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        os.environ.setdefault("MASTER_PORT", str(port))
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        dist.init_process_group(backend="gloo", world_size=1, rank=0)

    def setup(self) -> None:
        hcfg = self.hydra_cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._ensure_single_process_group()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            cuda_cfg = hcfg.get("cuda", {})
            torch.backends.cudnn.deterministic = cuda_cfg.get("cudnn_deterministic", False)
            torch.backends.cudnn.benchmark = cuda_cfg.get("cudnn_benchmark", True)
            allow_tf32 = cuda_cfg.get("allow_tf32", False)
            torch.backends.cuda.matmul.allow_tf32 = cuda_cfg.get("matmul_allow_tf32", allow_tf32)
            torch.backends.cudnn.allow_tf32 = cuda_cfg.get("cudnn_allow_tf32", allow_tf32)

        # ---- Resolve key params from the experiment ----
        self.dataset_name = str(hcfg.scratch.dataset_name)
        self.model_label = str(hcfg.scratch.model_label)
        self.N = int(hcfg.scratch.N)
        self.fewshot_seed = int(hcfg.scratch.fewshot_seed)
        self.training_unit = str(hcfg.scratch.get("training_unit", "video"))
        self.experiment_name = str(hcfg.scratch.experiment_name)

        if self.dataset_name == "ACOUSLIC" and self.training_unit == "frame":
            raise NotImplementedError(
                "training_unit=frame for ACOUSLIC is reserved for a follow-up. "
                "Pass scratch.training_unit=video (default) or implement the "
                "frame-level adapter."
            )

        # Reproducibility — these seeds drive both the subset selection AND
        # any data-augmentation RNG inside the train loop. Few-shot is
        # single-GPU, so no rank-offset needed.
        seed_everything(self.fewshot_seed)

        # batch_size = min(N, 4) for image datasets; videos are always 1 per
        # step. The YAML caps at the maximum (4 for images,
        # 1 for videos); this clamps further when N is below the cap so the
        # reported batch_size matches what the loader actually yields.
        #
        # `main()` calls OmegaConf.resolve(hcfg) before setup(), which
        # eagerly substitutes `${data.batch_size}` in
        # `data.train_dataset.batch_sizes` to the literal YAML value.
        # Mutating `hcfg.data.batch_size` alone would leave the already-
        # resolved nested list stale, so the loader would still receive
        # the pre-clamp value. Update both fields explicitly.
        original_bs = int(hcfg.data.batch_size)
        new_bs = min(original_bs, self.N)
        hcfg.data.batch_size = new_bs
        if "train_dataset" in hcfg.data and "batch_sizes" in hcfg.data.train_dataset:
            for i in range(len(hcfg.data.train_dataset.batch_sizes)):
                hcfg.data.train_dataset.batch_sizes[i] = new_bs
        if new_bs != original_bs:
            logger.info(
                f"few-shot batch_size clamped from {original_bs} to "
                f"{new_bs} = min(YAML, N={self.N}); "
                f"both data.batch_size and data.train_dataset.batch_sizes updated"
            )

        # ---- Materialize the deterministic subset ----
        # All 3 models running with the same (dataset, N, seed) write the
        # same .txt content, so the data loader sees IDENTICAL training
        # samples.
        annotation_dir = pathlib.Path(hcfg.scratch.annotation_dir).expanduser()
        train_list_txt = str(annotation_dir / self.dataset_name / "train_list.txt")
        splits_dir = pathlib.Path(hcfg.scratch.splits_dir).expanduser().resolve()
        self.split_txt_path = materialize_subset(
            splits_dir=splits_dir,
            dataset=self.dataset_name,
            N=self.N,
            seed=self.fewshot_seed,
            train_list_txt=train_list_txt,
            training_unit=self.training_unit,
        )
        logger.info(f"Subset txt: {self.split_txt_path}")

        # ---- Build dataset ----
        # `data.file_list_txt` resolves to ${scratch.splits_dir}/${experiment_name}.txt
        # which was just materialized. Hydra will pick it up cleanly.
        self.train_dataset = hydra_instantiate(
            hcfg.data.train_dataset, _recursive_=True, _convert_="all"
        )

        # ---- Build model + load base checkpoint ----
        ckpt_path = pathlib.Path(hcfg.ckpt_path).expanduser().resolve()
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Base checkpoint not found: {ckpt_path}")
        logger.info(f"Loading base ckpt: {ckpt_path}")
        model = build_model(hcfg.model, self.device, seed=self.fewshot_seed,
                            pretrained_ckpt_path=None)
        sd = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        if "model" in sd:
            sd = sd["model"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        logger.info(
            f"Loaded base weights — missing={len(missing)}, unexpected={len(unexpected)}"
        )
        if missing[:3]:
            logger.info(f"  first 3 missing: {missing[:3]}")
        if unexpected[:3]:
            logger.info(f"  first 3 unexpected: {unexpected[:3]}")
        del sd

        # ---- Per-module freeze (configurable via `freeze.*`) ----
        freeze_report = apply_module_freeze(model, hcfg.get("freeze", {}))
        trainable_modules = sorted(k for k, v in freeze_report.items() if v)
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        logger.info(
            f"Module freeze applied — trainable modules: {trainable_modules}; "
            f"{n_trainable / 1e6:.2f} M trainable / {n_frozen / 1e6:.2f} M frozen"
        )

        model.to(self.device)
        # NOTE: we leave the model in `train()` but the encoder's BatchNorm /
        # dropout will see no gradient anyway. The mask_decoder / prompt_encoder
        # need train() for any dropout layers.
        model.train()
        self.model = model

        # ---- Loss ----
        self.loss_fn = build_loss_fn(hcfg.loss_fn).to(self.device)

        # ---- Optimizer + scheduler ----
        # AdamW(lr=1e-4, wd=0.01), cosine LR to 0, grad clip 1.0.
        # Use plain torch.optim — no need for the pretrain SAM2Optimizer's
        # per-param-group modifiers when only a small slice trains.
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        opt_cfg = hcfg.optim
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=float(opt_cfg.lr),
            weight_decay=float(opt_cfg.weight_decay),
        )

        # Iters-per-epoch determines the cosine schedule's T_max.
        loader_probe = self.train_dataset.get_loader(epoch=0)
        self.iters_per_epoch = len(loader_probe)
        del loader_probe

        self.epochs = epochs_for_n(self.N, hcfg.epoch_schedule)
        total_steps = max(self.epochs * self.iters_per_epoch, 1)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=float(opt_cfg.cosine_min_lr),
        )
        self.grad_clip_norm = float(opt_cfg.grad_clip_norm)

        # AMP context factory — bf16 by default, matches pretrain.
        self._autocast = self._build_autocast(opt_cfg)

        # ---- Output dirs + log file ----
        output_root = pathlib.Path(hcfg.output_root).expanduser().resolve()
        self.ckpt_dir = output_root / "checkpoints" / self.experiment_name
        self.log_dir = output_root / "logs"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.train_log_path = self.ckpt_dir / "train_log.jsonl"

        logger.info(
            f"FewShotFinetuneRecipe setup complete: "
            f"experiment={self.experiment_name}, N={self.N}, seed={self.fewshot_seed}, "
            f"iters/epoch={self.iters_per_epoch}, epochs={self.epochs}, "
            f"total_steps={total_steps}, ckpt_dir={self.ckpt_dir}"
        )

    @staticmethod
    def _build_autocast(opt_cfg: DictConfig):
        if "amp" not in opt_cfg or not bool(opt_cfg.amp.get("enabled", False)):
            return nullcontext
        return lambda: build_autocast_context(opt_cfg)

    # ------------------ training loop ------------------
    def run(self) -> None:
        hcfg = self.hydra_cfg
        log_freq = int(hcfg.logging.get("log_freq", 5))

        # Resume short-circuit: if final.pth already exists, skip the run.
        # Reproducibility comes from the deterministic seeds + same subset.
        final_ckpt = self.ckpt_dir / "final.pth"
        if final_ckpt.is_file():
            logger.info(f"final.pth already exists — skipping fine-tune ({final_ckpt})")
            return

        log_fp = self.train_log_path.open("a")
        try:
            global_step = 0
            t_start = time.perf_counter()
            for epoch in range(self.epochs):
                loader = self.train_dataset.get_loader(epoch=epoch)
                epoch_loss_sum = 0.0
                epoch_steps = 0

                for batch_pack in loader:
                    # Pretrain iterates via `step_scheduler` which packs each
                    # step in a 1-element list. Iterating the loader
                    # directly (single-GPU, no step scheduler) yields the
                    # BatchedVideoDatapoint unwrapped. Handle both shapes.
                    if isinstance(batch_pack, list):
                        assert len(batch_pack) == 1, f"Unexpected batch list size: {len(batch_pack)}"
                        batch = batch_pack[0]
                    else:
                        batch = batch_pack
                    batch = batch.to(self.device, non_blocking=True)

                    self.optimizer.zero_grad(set_to_none=True)
                    with self._autocast():
                        outputs = self.model(batch)
                        targets = batch.masks
                        loss = self.loss_fn[batch.dict_key](outputs, targets)
                        if isinstance(loss, dict):
                            loss = loss[CORE_LOSS_KEY]

                    if not math.isfinite(loss.item()):
                        raise FloatingPointError(
                            f"Non-finite loss at epoch {epoch} step {global_step}: {loss.item()}"
                        )

                    loss.backward()
                    if self.grad_clip_norm and self.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in self.model.parameters() if p.requires_grad],
                            self.grad_clip_norm,
                        )
                    self.optimizer.step()
                    self.lr_scheduler.step()

                    epoch_loss_sum += loss.item()
                    epoch_steps += 1
                    global_step += 1

                    if global_step % log_freq == 0:
                        lr_now = self.optimizer.param_groups[0]["lr"]
                        rec = {
                            "epoch": epoch,
                            "step": global_step,
                            "loss": float(loss.item()),
                            "lr": float(lr_now),
                            "elapsed_s": time.perf_counter() - t_start,
                        }
                        log_fp.write(json.dumps(rec) + "\n")
                        log_fp.flush()
                        logger.info(
                            f"epoch {epoch:3d} step {global_step:5d} "
                            f"loss={loss.item():.4f} lr={lr_now:.2e}"
                        )

                avg_loss = epoch_loss_sum / max(epoch_steps, 1)
                rec = {
                    "epoch": epoch, "step": global_step,
                    "epoch_avg_loss": float(avg_loss),
                    "elapsed_s": time.perf_counter() - t_start,
                }
                log_fp.write(json.dumps(rec) + "\n")
                log_fp.flush()
                logger.info(f"epoch {epoch:3d} avg_loss={avg_loss:.4f}")

        finally:
            log_fp.close()

        self._save_final_checkpoint(final_ckpt)
        elapsed = time.perf_counter() - t_start
        logger.info(
            f"Fine-tune complete: {self.experiment_name} ({self.epochs} epochs, "
            f"{elapsed:.1f} s, final ckpt: {final_ckpt})"
        )

    # ------------------ checkpoint save ------------------
    def _save_final_checkpoint(self, path: pathlib.Path) -> None:
        """Save the model state_dict in the same shape Meta's SAM2 .pt files use.

        That makes the checkpoint a drop-in input for the existing
        `save_predictions.py` recipe — no conversion needed at eval time.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        sd = {k: v.cpu() for k, v in self.model.state_dict().items()}

        provenance = {
            "experiment_name": self.experiment_name,
            "dataset_name": self.dataset_name,
            "model_label": self.model_label,
            "N": int(self.N),
            "fewshot_seed": int(self.fewshot_seed),
            "training_unit": self.training_unit,
            "epochs": int(self.epochs),
            "split_txt_path": str(self.split_txt_path),
            "torch_version": str(torch.__version__),
        }
        torch.save({"model": sd, "_provenance": provenance}, path)
        logger.info(f"Wrote checkpoint: {path} ({sum(v.numel() for v in sd.values()) / 1e6:.1f} M params)")


# ---------------------------------------------------------------------------
# Entry point — same Hydra arg conventions as save_predictions.py
# ---------------------------------------------------------------------------


def main() -> None:
    config_dir = None
    config_name = "finetune"
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

    recipe = FewShotFinetuneRecipe(hydra_cfg)
    recipe.setup()
    recipe.run()


if __name__ == "__main__":
    main()
