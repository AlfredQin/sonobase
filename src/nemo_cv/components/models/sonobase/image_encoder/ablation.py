import torch.nn as nn
import torch
import torch.nn.functional as F
from pprint import pprint
from typing import List, Dict
from copy import deepcopy
from timm.models import group_parameters
from timm.layers import LayerScale2d, LayerNorm2d, LayerNorm, DropPath, LayerScale

from nemo_cv.components.models.sam2.modeling.backbones.hieradet import Hiera, MLP
from nemo_cv.components.models.sonobase.image_encoder.ms_deform_attn.modules.rf_detr_ms_deform_attn import MSDeformAttn
from nemo_cv.components.models.sonobase.image_encoder.utils.deform_attn import get_reference_points
from nemo_cv.components.models.sonobase.image_encoder.utils.position_encoding import PositionEmbeddingSine
from nemo_cv.components.models.sonobase.image_encoder.utils.init_weights import _init_weights


class TriBranchTrunk(nn.Module):
    """
    wrapper of sam2 image encoder
    """
    def __init__(
            self,
            branch0: nn.Module,
            branch1: nn.Module,
            branch2: nn.Module,
            ckpt_path0: str = None,
            ckpt_path1: str = None,
            ckpt_path2: str = None,
            img_size0: int = None,
            img_size1: int = None,
            img_size2: int = 1024,
            branch0_indices = None,
            branch1_indices = None,
            branch2_indices = None,
            return_indices = None,
            branch0_dims = None,
            branch1_dims = None,
            branch2_dims = None,
            num_heads: List[int] = None,
            with_cross_branch_interaction: bool = False,
            init_values=0.,
            interact_type: str = "deform_cross_attn",
            merge_location: str = "before_interact",
            image_scales: List[float] = None,
            ffn: str = None,
            with_pos: bool = True,
            n_points: int = 4,
            dropout_attn=0.,
            drop_path_attn=0.,
            drop_path_ffn=0.,
            expand_ratio=4,
            interaction_indices: List[int] = None,
    ):
        """
        :param branch0:
        :param branch1:
        :param branch2:
        :param fusion_block:
        :param fusion_indices: List of dictionaries, each dictionary contains the starting and ending block index of each
        branch.
        """
        super().__init__()
        self.branch0 = branch0
        self.branch1 = branch1
        self.branch2 = branch2
        self.img_size0 = img_size0
        self.img_size1 = img_size1
        self.img_size2 = img_size2
        self.image_scales = image_scales

        if ckpt_path0 is not None:
            state_dict = torch.load(ckpt_path0)['model']
            state_dict = {k.replace("image_encoder.trunk.", ""): v for k, v in state_dict.items() if k.startswith("image_encoder.trunk")}
            msg = self.branch0.load_state_dict(state_dict, strict=False)
            pprint(f"Load sam2 encoder from {ckpt_path0} for branch 0, {msg}")
        if ckpt_path1 is not None:
            state_dict = torch.load(ckpt_path1)['model']
            state_dict = {k.replace("image_encoder.trunk.", ""): v for k, v in state_dict.items() if k.startswith("image_encoder.trunk")}
            msg = self.branch1.load_state_dict(state_dict, strict=False)
            pprint(f"Load sam2 encoder from {ckpt_path1} for branch 1, {msg}")
        if ckpt_path2 is not None:
            state_dict = torch.load(ckpt_path2)['model']
            state_dict = {k.replace("image_encoder.trunk.", ""): v for k, v in state_dict.items() if k.startswith("image_encoder.trunk")}
            msg = self.branch2.load_state_dict(state_dict, strict=False)
            pprint(f"Load sam2 encoder from {ckpt_path2} for branch 2, {msg}")

        self.branch0_indices = branch0_indices
        self.branch1_indices = branch1_indices
        self.branch2_indices = branch2_indices
        self.return_indices = return_indices
        self.interaction_indices = interaction_indices

        self.with_cross_branch_interaction = with_cross_branch_interaction
        self.interaction_blocks = nn.ModuleList()
        self.merge_location = merge_location
        if self.with_cross_branch_interaction:
            for idx, (dim0, dim1, dim2) in enumerate(zip(branch0_dims, branch1_dims, branch2_dims)):
                if self.merge_location == "before_interact" and idx == return_indices[-1]:
                    continue  # we skip the last layer for interaction when merging before interaction
                if interact_type == "deform_self_attn":
                    self.interaction_blocks.append(
                        MSDeformSelfAttnCrossBranchBlock(
                            dim=dim0,
                            branch_dims=[dim0, dim1, dim2],
                            n_points=n_points,
                            expand_ratio=expand_ratio,
                            num_layers=1,
                            drop_path_ffn=drop_path_ffn,
                            num_heads=num_heads[idx],
                            init_values=init_values,
                            ffn=ffn,
                        )
                    )

        self.branch0_merge_blocks = nn.ModuleList()
        self.branch1_merge_blocks = nn.ModuleList()
        self.branch2_merge_blocks = nn.ModuleList()
        self.branch0_weights = nn.ModuleList()
        self.branch1_weights = nn.ModuleList()
        self.branch2_weights = nn.ModuleList()
        for idx, (dim0, dim1, dim2) in enumerate(zip(branch0_dims, branch1_dims, branch2_dims)):
            if idx in self.return_indices:
                self.branch0_merge_blocks.append(nn.Sequential(
                    nn.Identity()
                ))
                self.branch0_weights.append(LayerScale2d(dim0, init_values=1.))
                self.branch1_merge_blocks.append(nn.Sequential(
                    nn.Conv2d(dim1, dim0, kernel_size=3, stride=1, padding=1),
                ))
                self.branch1_weights.append(LayerScale2d(dim0, init_values=1.))
                self.branch2_merge_blocks.append(nn.Sequential(
                    nn.Conv2d(dim2, dim0, kernel_size=3, stride=1, padding=1),
                ))
                self.branch2_weights.append(LayerScale2d(dim0, init_values=1.))

        self.branch0_merge_blocks.apply(_init_weights)
        self.branch1_merge_blocks.apply(_init_weights)
        self.branch2_merge_blocks.apply(_init_weights)

    def forward(self, images: torch.Tensor) -> List[torch.Tensor]:
        """
        :param images: input images with resolution of 1024 x 1024
        :return:
        """
        image_pyramids = [
            F.interpolate(images, size=(self.img_size0, self.img_size0), mode='bilinear', align_corners=False),
            F.interpolate(images, size=(self.img_size1, self.img_size1), mode='bilinear', align_corners=False),
            F.interpolate(images, size=(self.img_size2, self.img_size2), mode='bilinear', align_corners=False),
        ]

        x0 = self.branch0.forward_patch_embed(image_pyramids[0])
        x1 = self.branch1.forward_patch_embed(image_pyramids[1])
        x2 = self.branch2.forward_patch_embed(image_pyramids[2])
        branch0_features, branch1_features, branch2_features = [], [], []
        for i, (indices0, indices1, indices2) in enumerate(zip(self.branch0_indices, self.branch1_indices, self.branch2_indices)):
            x0 = self.branch0.forward_blocks(x0, indices0)
            x1 = self.branch1.forward_blocks(x1, indices1)
            x2 = self.branch2.forward_blocks(x2, indices2)

            # intermediate feature
            if self.merge_location == "before_interact":
                if i in self.return_indices:
                    branch0_features.append(x0.permute(0, 3, 1, 2))
                    branch1_features.append(x1.permute(0, 3, 1, 2))
                    branch2_features.append(x2.permute(0, 3, 1, 2))
                # fusion
                if self.with_cross_branch_interaction and i != self.return_indices[-1] and i in self.interaction_indices:  # we dont need to do fusion for the last block
                    x0, x1, x2 = self.interaction_blocks[i](x0, x1, x2)

            elif self.merge_location == "after_interact":
                # fusion
                if self.with_cross_branch_interaction and i in self.interaction_indices:
                    x0, x1, x2 = self.interaction_blocks[i](x0, x1, x2)

                # intermediate feature
                if i in self.return_indices:
                    branch0_features.append(x0.permute(0, 3, 1, 2))
                    branch1_features.append(x1.permute(0, 3, 1, 2))
                    branch2_features.append(x2.permute(0, 3, 1, 2))

        out = []
        for s, (f0, f1, f2) in enumerate(zip(branch0_features, branch1_features, branch2_features)):
            h, w = f2.shape[-2:]
            f0 = F.interpolate(self.branch0_merge_blocks[s](f0), size=(h, w), mode='bilinear', align_corners=False)
            f1 = F.interpolate(self.branch1_merge_blocks[s](f1), size=(h, w), mode='bilinear', align_corners=False)
            f2 = self.branch2_merge_blocks[s](f2)
            w_sum = self.branch0_weights[s](f0) + self.branch1_weights[s](f1) + self.branch2_weights[s](f2)
            out.append(w_sum)

        return out

    def get_num_layers(self) -> int:
        """
        Return the total number of layers for layer decay calculation.
        Uses the largest branch (branch0) as reference.
        """
        return len(self.branch0.blocks)

    def get_layer_id(self, layer_name: str) -> int:
        """
        Map parameter names to layer ids for layer-wise learning rate decay.

        Args:
            layer_name: Name of the parameter

        Returns:
            Layer id for the parameter (0 = earliest layers, num_layers = last layers)
        """
        num_layers = self.get_num_layers()

        # Position embeddings get layer 0 (lowest lr)
        if "pos_embed" in layer_name:
            return 0

        # Relative position biases get the last layer id
        if "rel_pos" in layer_name:
            return num_layers + 1

        # Patch embedding gets layer 0
        if "patch_embed" in layer_name:
            return 0

        # Map branch blocks to layer ids
        # Extract block index from layer names like "branch0.blocks.5.attn.weight"
        for branch_name in ["branch0", "branch1", "branch2"]:
            if branch_name in layer_name and "blocks" in layer_name:
                try:
                    # Parse: branch0.blocks.X.* or branch0.blocks.X.*
                    parts = layer_name.split("blocks.")
                    if len(parts) > 1:
                        block_idx = int(parts[1].split(".")[0])
                        # Normalize block indices across branches
                        # branch0 has the most blocks, others may have fewer
                        if branch_name == "branch0":
                            return block_idx + 1
                        elif branch_name == "branch1":
                            # Scale branch1 block indices to match branch0 range
                            branch1_blocks = len(self.branch1.blocks)
                            scale = num_layers / branch1_blocks if branch1_blocks > 0 else 1
                            return int(block_idx * scale) + 1
                        elif branch_name == "branch2":
                            # Scale branch2 block indices to match branch0 range
                            branch2_blocks = len(self.branch2.blocks)
                            scale = num_layers / branch2_blocks if branch2_blocks > 0 else 1
                            return int(block_idx * scale) + 1
                except (ValueError, IndexError):
                    pass

        # Neck, merge blocks, interaction blocks, etc. get the last layer id
        return num_layers + 1

    @property
    def channel_list(self):
        return self.branch0.channel_list


class MSDeformSelfAttnCrossBranchLayer(nn.Module):
    """
    Cross branch features fusion block with multi-scale deformable attention
    """
    def __init__(
            self,
            dim,
            branch_dims,
            num_heads,
            n_points,
            dropout_attn=0.,
            drop_path_attn=0.,
            drop_path_ffn=0.,
            expand_ratio=4,
            query_norm=False,
            init_values=0.,
            ffn: str = None,
    ):
        super().__init__()
        self.query_norm = nn.LayerNorm(dim) if query_norm else nn.Identity()
        self.value_norm = nn.LayerNorm(dim)
        self.self_attn = MSDeformAttn(d_model=dim, n_levels=len(branch_dims), n_heads=num_heads, n_points=n_points)
        self.dropout_attns = nn.ModuleList([nn.Dropout(dropout_attn) if dropout_attn > 0. else nn.Identity() for _ in branch_dims])
        self.drop_path_attns = nn.ModuleList([DropPath(drop_path_attn) if drop_path_attn > 0. else nn.Identity() for _ in branch_dims])
        self.ls_attns = nn.ModuleList([LayerScale(dim, init_values=init_values) for dim in branch_dims])

        self.input_proj_layers = nn.ModuleList()
        self.output_proj_layers = nn.ModuleList()
        for bdim in branch_dims:
            if bdim != dim:
                self.input_proj_layers.append(nn.Linear(bdim, dim))
                self.output_proj_layers.append(nn.Linear(dim, bdim))
            else:
                self.input_proj_layers.append(nn.Identity())
                self.output_proj_layers.append(nn.Identity())

        self.ffn_type = ffn
        if ffn is not None:
            self.ffn_norms = nn.ModuleList([nn.LayerNorm(dim) for dim in branch_dims])
            if ffn == "mlp":
                self.ffns = nn.ModuleList([MLP(dim, int(dim * expand_ratio), dim, num_layers=2, activation=nn.GELU) for dim in branch_dims])
            elif ffn == "conv_ffn":
                self.ffns = nn.ModuleList([ConvFFN(in_features=dim, hidden_features=int(dim * expand_ratio), drop=0.) for dim in branch_dims])
            self.drop_path_ffns = nn.ModuleList([DropPath(drop_path_ffn) if drop_path_ffn > 0. else nn.Identity() for _ in branch_dims])
            self.ls_ffns = nn.ModuleList([LayerScale(dim, init_values=init_values) for dim in branch_dims])

    def forward(self, mb_features, reference_points, spatial_shapes, level_start_index, pos_embeds=None):
        """

        Args:
            mb_features: multi_branch_features: List of B x H x W x C features
            reference_points:
            spatial_shapes:
            level_start_index:
            pos:

        Returns:

        """
        query = torch.cat([proj(feat.flatten(1, 2)) for feat, proj in zip(mb_features, self.input_proj_layers)], dim=1)
        query = self.self_attn(
            query=self.with_pos_embed(self.query_norm(query), pos_embeds),
            reference_points=reference_points,
            input_flatten=self.value_norm(query),
            input_spatial_shapes=spatial_shapes,
            input_level_start_index=level_start_index,
        )
        # query = query + self.drop_path_attn(self.dropout_attn(self.ls_attn(attn_out)))
        B = query.shape[0]
        query = torch.split(query, [shape[0] * shape[1] for shape in spatial_shapes], dim=1)
        query = [feat.view(B, *shape, -1) for feat, shape in zip(query, spatial_shapes)]
        output = []
        for i in range(len(mb_features)):
            out_feat = mb_features[i] + self.drop_path_attns[i](self.dropout_attns[i](
                self.ls_attns[i](self.output_proj_layers[i](query[i]))
            ))
            if hasattr(self, "ffns"):
                out_feat = out_feat + self.drop_path_ffns[i](self.ls_ffns[i](self.ffns[i](self.ffn_norms[i](out_feat))))
            output.append(out_feat)

        return output

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos


class MSDeformSelfAttnCrossBranchBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            branch_dims,
            n_points: int = 4,
            num_layers: int = 1,
            expand_ratio: int = 4,
            drop_path_attn=0.,
            drop_path_ffn=0.,
            num_heads=None,
            init_values=0.,
            ffn: str = None,
    ):
        super().__init__()
        fusion_layer = MSDeformSelfAttnCrossBranchLayer(
            dim,
            branch_dims,
            num_heads,
            n_points=n_points,
            dropout_attn=0.,
            drop_path_attn=drop_path_attn,
            drop_path_ffn=drop_path_ffn,
            expand_ratio=expand_ratio,
            query_norm=False,
            init_values=init_values,
            ffn=ffn
        )
        self.layers = nn.ModuleList([deepcopy(fusion_layer) for _ in range(num_layers)])
        self.dim = dim

        self.level_embed = nn.ParameterList([nn.Parameter(torch.randn(dim)) for _ in range(len(branch_dims))])
        self.pos_encoding = PositionEmbeddingSine(num_pos_feats=dim // 2, normalize=True)

        self._init_weights()

    def _init_weights(self):
        self.apply(_init_weights)
        def _init_deform_attn(m):
            if isinstance(m, (MSDeformAttn)):
                m._reset_parameters()
        self.apply(_init_deform_attn)

    def forward(self, *multi_branch_features: List[torch.Tensor]):
        """
        @param multi_branch_features: List of B x H x W x C features
        @return:
        """
        pos_embeds = []
        for feat, lvl_emb in zip(multi_branch_features, self.level_embed):
            dummy_feat = feat.new_zeros(feat.shape[0], self.dim, feat.shape[1], feat.shape[2])
            pos = self.pos_encoding(dummy_feat)
            lvl_pos = pos + lvl_emb.view(1, -1, 1, 1)
            pos_embeds.append(lvl_pos.permute(0, 2, 3, 1))
        pos_embeds = torch.cat([pos.flatten(1, 2) for pos in pos_embeds], dim=1)
        reference_points, spatial_shapes, level_start_index = self.get_deform_attn_inputs(multi_branch_features)

        for layer in self.layers:
            multi_branch_features = layer(multi_branch_features, reference_points, spatial_shapes, level_start_index, pos_embeds)

        return multi_branch_features

    def get_deform_attn_inputs(self, multi_branch_features):
        device = multi_branch_features[-1].device
        spatial_shapes = [feat.shape[1:3] for feat in multi_branch_features]
        reference_points = get_reference_points(spatial_shapes, device)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        return reference_points, spatial_shapes, level_start_index

class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """

        Args:
            x: B, H, W, C

        Returns:

        """
        x = self.fc1(x)
        x = self.dwconv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
