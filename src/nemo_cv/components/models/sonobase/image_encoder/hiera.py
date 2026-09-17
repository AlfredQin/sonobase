
from nemo_cv.components.models.sam2.modeling.backbones.hieradet import Hiera


class HieraWrapper(Hiera):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward_patch_embed(self, x):
        x = self.patch_embed(x)
        # x: (B, H, W, C)

        # Add pos embed
        x = x + self._get_pos_embed(x.shape[1:3])

        return x

    def forward_blocks(self, x, index,):
        start_idx, end_idx = index
        blks = self.blocks[start_idx:end_idx]
        for blk in blks:
            x = blk(x)

        return x