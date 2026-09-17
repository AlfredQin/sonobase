"""Convert a sonobase DCP-format pretrain checkpoint to a single-file `.pt`.

The sonobase pretraining recipe (`nemo_cv.recipes.sonobase.pretrain`) writes
checkpoints in PyTorch's distributed-checkpoint (DCP) format: a directory
with one shard per DDP rank under `<ckpt_dir>/model/`. The benchmark test
recipe (`nemo_cv.recipes.benchmarks.test_sam2`) expects a single `.pt` file
in the SAM2 release format (`{"model": state_dict}`) so all three benchmarked
models — SAM2-no-ft, MedSAM2, sonobase — load through the same code path.

This converter bridges the two formats. It runs **single-process** (no DDP)
and produces a `.pt` file that any downstream script can `torch.load(...)`.

Why a separate converter and not just on-the-fly DCP loading in the test
recipe?

- Sharding is a training concern: DCP shards are co-tied to the DDP world
  size and rank layout used at save time. Decoupling produces an artifact
  that's portable across machines and rank counts.
- Comparison scripts that aggregate results from sam2-no-ft / medsam2 /
  sonobase shouldn't have to know which one came from DCP and which came
  from a `.pt` file — uniformity simplifies the rest of the pipeline.
- The conversion happens once per pretrain checkpoint; subsequent test
  runs (different seeds, different test sets) reuse the same `.pt`.

Usage:

    cd src
    uv run python -m nemo_cv.recipes.benchmarks.convert_sonobase_dcp_to_pt \\
        ./experiments/sonobase/pretrain/.../epoch_N_step_M \\
        ./experiments/sonobase/pretrain/.../epoch_N_step_M.pt

The DCP directory must contain a `config.yaml` (the resolved Hydra config
from the training run, written by `BaseRecipe.save_checkpoint`) and a
`model/` subdirectory with DCP shards (`.metadata` + `__N_0.distcp` files).
"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib
import socket
import subprocess
import sys
import time

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from nemo_automodel.components.checkpoint.checkpointing import Checkpointer, CheckpointingConfig
from nemo_automodel.components.loggers.log_utils import setup_logging

logger = logging.getLogger(__name__)


def _free_tcp_port() -> int:
    """Find an unused TCP port on localhost (race-free enough for one-shot use)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _init_single_process_group() -> None:
    """Initialize torch.distributed with world_size=1 so DCP load can run.

    PyTorch's DCP API uses distributed collectives even on single rank; they
    no-op when world_size=1, but the process group must still be initialized.
    We use the gloo backend (CPU) since the converter doesn't need GPU ops.
    """
    if dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(_free_tcp_port()))
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    dist.init_process_group(backend="gloo", world_size=1, rank=0)


def _git_hash_or_none() -> str | None:
    """Return the current repo git HEAD hash, or None if not in a repo."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        return out or None
    except Exception:
        return None


def convert(dcp_dir: str, out_pt: str) -> dict:
    """Convert one DCP checkpoint dir to a single `.pt` file.

    Args:
        dcp_dir: Path to the checkpoint directory (e.g. `epoch_1_step_387`).
            Must contain `config.yaml` and `model/` subdirectory.
        out_pt: Destination path for the `.pt` file. Parent directories will
            be created if needed.

    Returns:
        The provenance dict that was attached to the saved checkpoint
        (also under the `_provenance` key inside the `.pt`).
    """
    dcp_dir = str(pathlib.Path(dcp_dir).expanduser().resolve())
    out_pt = str(pathlib.Path(out_pt).expanduser().resolve())

    # Sanity-check input
    if not os.path.isdir(dcp_dir):
        raise FileNotFoundError(f"DCP checkpoint directory does not exist: {dcp_dir}")
    cfg_path = os.path.join(dcp_dir, "config.yaml")
    model_path = os.path.join(dcp_dir, "model")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(
            f"Missing {cfg_path}. The DCP dir must contain config.yaml "
            "(the resolved Hydra config saved at training time)."
        )
    if not os.path.isdir(model_path):
        raise FileNotFoundError(
            f"Missing {model_path}. Expected a `model/` subdirectory with DCP shards."
        )

    # Load config and rebuild the model
    logger.info(f"Reading config from {cfg_path}")
    hcfg = OmegaConf.load(cfg_path)
    OmegaConf.resolve(hcfg)

    # Import here so this script can be imported without requiring the full
    # sonobase recipe stack to be loadable in some contexts (e.g. doc gen).
    from nemo_cv.recipes.sonobase.pretrain import build_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Building model on device={device}")
    # We don't need pretrained_ckpt_path here because the DCP load below
    # overwrites all parameters anyway.
    model = build_model(hcfg.model, device, seed=int(hcfg.get("seed", 42)), pretrained_ckpt_path=None)
    model.eval()

    # Initialize single-process group so DCP collectives work
    _init_single_process_group()

    # Build a minimal Checkpointer just to use its load_model method
    ckpt_cfg = CheckpointingConfig(
        enabled=False,                  # we never save here
        checkpoint_dir=dcp_dir,
        model_save_format="torch_save",
        save_consolidated=False,
        model_cache_dir=dcp_dir,
        model_repo_id="",
        is_peft=False,
    )
    checkpointer = Checkpointer(
        config=ckpt_cfg, dp_rank=0, tp_rank=0, pp_rank=0, moe_mesh=None,
    )

    logger.info(f"Loading DCP shards from {model_path}")
    t0 = time.perf_counter()
    checkpointer.load_model(model, model_path)
    t_load = time.perf_counter() - t0
    logger.info(f"DCP load complete in {t_load:.1f}s")

    # Materialize the state dict and save in SAM2 release format
    sd = model.state_dict()
    n_params = len(sd)
    bytes_total = sum(t.nelement() * t.element_size() for t in sd.values())
    logger.info(f"State dict: {n_params} tensors, {bytes_total / 1024**2:.1f} MiB total")

    # NOTE: every value here is a plain str/int so the resulting `.pt` can be
    # loaded with `weights_only=True`. `torch.__version__` is a `TorchVersion`
    # (a str subclass) that PyTorch's restricted unpickler rejects, so we cast.
    provenance = {
        "source_dcp_dir": dcp_dir,
        "converted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": socket.gethostname(),
        "git_hash": _git_hash_or_none() or "",
        "torch_version": str(torch.__version__),
        "n_params": n_params,
    }

    out_dir = os.path.dirname(out_pt)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    payload = {"model": sd, "_provenance": provenance}
    logger.info(f"Saving {out_pt}")
    torch.save(payload, out_pt)

    if dist.is_initialized():
        dist.destroy_process_group()

    return provenance


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(
        description=(
            "Convert a sonobase DCP-format pretrain checkpoint to a single-file "
            ".pt suitable for the benchmark test recipe."
        )
    )
    parser.add_argument(
        "dcp_dir",
        help=(
            "Path to a sonobase pretrain checkpoint dir, e.g. "
            "`./experiments/sonobase/pretrain/.../epoch_1_step_387`. "
            "Must contain `config.yaml` and a `model/` subdir with DCP shards."
        ),
    )
    parser.add_argument(
        "out_pt",
        help=(
            "Destination path for the `.pt` file. Convention: place it next to "
            "the source DCP dir with a `.pt` suffix, e.g. "
            "`./experiments/.../epoch_1_step_387.pt`."
        ),
    )
    args = parser.parse_args()

    provenance = convert(args.dcp_dir, args.out_pt)
    logger.info("=" * 60)
    logger.info("CONVERSION COMPLETE")
    logger.info(f"  Source:   {provenance['source_dcp_dir']}")
    logger.info(f"  Output:   {args.out_pt}")
    logger.info(f"  N tensors: {provenance['n_params']}")
    logger.info(f"  Git hash: {provenance['git_hash']}")
    logger.info(f"  Time:     {provenance['converted_at']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
