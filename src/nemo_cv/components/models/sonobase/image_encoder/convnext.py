import torch
import torch.nn as nn
import timm
from pprint import pprint


class ConvNeXtWrapper(nn.Module):
    """
    Wrapper around timm's ConvNeXt backbone to match the HieraWrapper interface.

    Provides forward_patch_embed() and forward_blocks() methods so it can be used
    as a drop-in replacement for HieraWrapper in TBHiera / TBHieraEncoder.

    The internal ConvNeXt stages are flattened into a single block list:
      - Stage 0: blocks only (no downsample since the stem already handles stride-4)
      - Stages 1-3: downsample layer + blocks

    Features at the interface boundaries are in channels-last format (B, H, W, C)
    to match the Hiera convention used by the interaction blocks.

    ConvNeXt variants and their block counts (after flattening):
      convnext_tiny  : depths=[3,3, 9,3] -> 3 + (1+3) + (1+9) + (1+3) = 21 blocks
      convnext_small : depths=[3,3,27,3] -> 3 + (1+3) + (1+27)+ (1+3) = 39 blocks
      convnext_base  : depths=[3,3,27,3], dims=[128,256,512,1024] -> 39 blocks
    """

    def __init__(
        self,
        model_name: str = "convnext_small",
        pretrained: bool = True,
        drop_path_rate: float = 0.0,
    ):
        """
        Args:
            model_name: timm model name for the ConvNeXt variant
                        (e.g. "convnext_tiny", "convnext_small",
                         "convnext_small.fb_in22k_ft_in1k", etc.)
            pretrained: Whether to load pretrained weights via timm.
            drop_path_rate: Stochastic depth rate.
        """
        super().__init__()

        # Create the full timm model (we will disassemble it)
        model = timm.create_model(
            model_name,
            pretrained=pretrained,
            drop_path_rate=drop_path_rate,
        )

        # Stem  –  equivalent to Hiera's patch_embed (stride 4)
        self.patch_embed = model.stem

        # Flatten every stage into a single ModuleList so that
        # forward_blocks(x, [start, end]) can slice arbitrary ranges.
        #   stage 0  : blocks only           (identity downsample is skipped)
        #   stage 1-3: downsample + blocks   (downsample counted as one "block")
        self.blocks = nn.ModuleList()
        for stage_idx, stage in enumerate(model.stages):
            if stage_idx > 0:
                self.blocks.append(stage.downsample)
            for block in stage.blocks:
                self.blocks.append(block)

    # ------------------------------------------------------------------
    # Public interface (matches HieraWrapper)
    # ------------------------------------------------------------------

    def forward_patch_embed(self, x):
        """
        Process the raw image through the ConvNeXt stem.

        Args:
            x: (B, 3, H, W)

        Returns:
            (B, H/4, W/4, C)  – channels-last, spatial resolution / 4
        """
        x = self.patch_embed(x)          # (B, C, H/4, W/4)
        x = x.permute(0, 2, 3, 1)       # (B, H/4, W/4, C)
        return x

    def forward_blocks(self, x, index):
        """
        Run a contiguous slice of the flattened block list.

        Args:
            x:     (B, H, W, C)  – channels-last
            index: [start_idx, end_idx)

        Returns:
            (B, H', W', C') – channels-last.  Resolution and channel dim
            may change when the slice contains a downsample block.
        """
        start_idx, end_idx = index
        x = x.permute(0, 3, 1, 2)       # (B, C, H, W)
        for i in range(start_idx, end_idx):
            x = self.blocks[i](x)
        x = x.permute(0, 2, 3, 1)       # (B, H', W', C')
        return x
