"""US-RFDETR test recipe — eval-only on a saved checkpoint.

Single-GPU evaluation by default (the typical
``torchrun --nproc_per_node=1`` invocation in the bundled shell wrappers
gives a 1-rank DDP world); however the underlying code path is
DDP-aware, so multi-GPU eval works too if you ever need it.

Subclasses :class:`TrainUSRFDETRRecipe` to reuse all of the heavy
``setup()`` machinery (data module + model + criterion + postprocess +
metric instances), then overrides ``setup()`` to additionally load the
checkpoint at ``ckpt_path``, and overrides ``run()`` to skip training
and call ``run_test_loop()`` only.

CLI
---
::

    uv run torchrun --nproc_per_node=1 -m nemo_cv.recipes.us_rfdetr.test \\
        -c ./configs/us_rfdetr -cn test \\
        experiment=us_rfdetr_on_acouslic backbone=sonobase \\
        ckpt_path=./experiments/us_rfdetr/acouslic/sonobase/checkpoints/best.pt
"""

from __future__ import annotations

import logging
import os
import pathlib
from typing import List, Optional

import torch
from omegaconf import DictConfig, OmegaConf

from nemo_automodel.components.config.loader import ConfigNode

from nemo_cv.components.training.utils import unwrap_ddp_if_wrapped
from nemo_cv.recipes.us_rfdetr.train import TrainUSRFDETRRecipe

logger = logging.getLogger(__name__)


class TestUSRFDETRRecipe(TrainUSRFDETRRecipe):
    """Eval-only US-RFDETR recipe.

    Differs from :class:`TrainUSRFDETRRecipe` in two places only:

    * ``setup()`` additionally loads ``ckpt_path`` (which must point at a
      single-file ``.pt`` produced by training, normally ``best.pt``).
    * ``run()`` skips ``run_train_loop()`` and goes straight to
      ``run_test_loop()``.
    """

    def setup(self) -> None:
        super().setup()
        ckpt_path = self.hydra_cfg.get("ckpt_path", None)
        if ckpt_path in (None, "", "???"):
            raise ValueError(
                "test recipe requires `ckpt_path=<path-to-best.pt>` on the CLI"
            )
        ckpt_path = str(pathlib.Path(ckpt_path).expanduser().resolve())
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"ckpt_path does not exist: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        unwrapped = unwrap_ddp_if_wrapped(self.model)
        msg = unwrapped.load_state_dict(sd, strict=True)
        if self.dist_env.is_main:
            logger.info(
                f"Loaded eval checkpoint from {ckpt_path} "
                f"(missing={len(msg.missing_keys)}, "
                f"unexpected={len(msg.unexpected_keys)})"
            )

    def run(self) -> None:
        """Skip training; run only the test loop."""
        self.run_test_loop()
        if self.metric_logger_test is not None:
            self.metric_logger_test.close()


# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for eval-only runs."""
    import sys

    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    config_dir: Optional[str] = None
    config_name: str = "test"
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

    recipe = TestUSRFDETRRecipe(cfg_node, hydra_cfg)
    recipe.setup()
    recipe.run()


if __name__ == "__main__":
    main()
