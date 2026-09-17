import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

import math
import warnings
from .ms_deform_attn import MSDeformAttnFunction, _is_power_of_2


class TemporalMSDeformAttnBase(nn.Module):
    def __init__(
            self,
            d_model=256,
            n_levels=4,
            t_window=2,
            n_heads=8,
            n_curr_points=4,
            n_temporal_points=2
    ):
        super(TemporalMSDeformAttnBase, self).__init__()
        """
        Multi-Scale Deformable Attention Module
        :param d_model          hidden dimension
        :param n_levels         number of feature levels
        :param n_heads          number of attention heads
        :param n_curr_points    number of sampling points per attention head per feature level from
                                each query corresponding frame
        :param n_temporal_points    number of sampling points per attention head per feature level
                                    from temporal frames
        """
        if d_model % n_heads != 0:
            raise ValueError(
                'd_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        if not _is_power_of_2(_d_per_head):
            warnings.warn(
                "You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation.")

        self.im2col_step = 64
        self.d_model = d_model
        self.n_levels = n_levels
        self.t_window = t_window
        self.n_heads = n_heads
        self.n_curr_points = n_curr_points
        self.n_temporal_points = n_temporal_points

        # Used for sampling and attention in the current frame
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_curr_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_curr_points)

        # Used for sampling and attention in the prev or post frames
        self.temporal_sampling_offsets = nn.Linear(d_model, n_heads * n_levels * t_window * n_temporal_points * 2)
        self.temporal_attention_weights = nn.Linear(d_model, n_heads * n_levels * t_window * n_temporal_points)

        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        # sampling offset initialized weight to 0, so at initial iterations the bias is the only that matters at all
        constant_(self.temporal_sampling_offsets.weight.data, 0.)

        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)

        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0])

        # curr_frame init
        curr_grid_init = grid_init.view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels,
                                                                      self.n_curr_points, 1)
        for i in range(self.n_curr_points):
            curr_grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(curr_grid_init.reshape(-1))

        # temporal init
        temporal_grid_init = grid_init.view(self.n_heads, 1, 1, 1, 2).repeat(1, self.n_levels,
                                                                             self.t_window,
                                                                             self.n_temporal_points,
                                                                             1)

        for i in range(self.n_temporal_points):
            temporal_grid_init[:, :, :, i, :] *= i + 1

        with torch.no_grad():
            self.temporal_sampling_offsets.bias = nn.Parameter(temporal_grid_init.reshape(-1))

        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        constant_(self.temporal_attention_weights.weight.data, 0.)
        constant_(self.temporal_attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(
            self,
            query,
            reference_points,
            input_flatten,
            input_spatial_shapes,
            input_level_start_index,
            input_padding_mask
    ):
        """
        Args:
            query:
            reference_points:
            input_flatten:
            input_spatial_shapes:
            input_level_start_index:
            input_padding_mask:

        Returns:

        """
        raise NotImplementedError

    # Computes current/temporal sampling offsets and attention weights,
    # which are treated different for the encoder and decoder later on
    def _compute_deformable_attention(self, query, input_flatten):
        T_, Len_q, _ = query.shape
        T_, Len_in, _ = input_flatten.shape

        value = self.value_proj(input_flatten)
        value = value.view(T_, Len_in, self.n_heads, self.d_model // self.n_heads)

        temporal_sampling_offsets = self.temporal_sampling_offsets(query).view(T_,
                                                                               Len_q,
                                                                               self.n_heads,
                                                                               self.t_window,
                                                                               self.n_levels,
                                                                               self.n_temporal_points,
                                                                               2)
        temporal_sampling_offsets = temporal_sampling_offsets.flatten(3, 4)  # (T_, L, N_H, t*N_L, N_P, 2)
        temporal_attention_weights = self.temporal_attention_weights(query)
        temporal_attention_weights = temporal_attention_weights.view(T_,
                                                                     Len_q,
                                                                     self.n_heads,
                                                                     self.t_window * self.n_levels * self.n_temporal_points
                                                                     )
        curr_frame_attention_weights = self.attention_weights(query).view(T_,
                                                                          Len_q,
                                                                          self.n_heads,
                                                                          self.n_levels * self.n_curr_points
                                                                          )
        attention_weights_curr_temporal = torch.cat(
            [curr_frame_attention_weights, temporal_attention_weights], dim=3)
        attention_weights_curr_temporal = F.softmax(attention_weights_curr_temporal, -1)
        attention_weights_curr = attention_weights_curr_temporal[:, :, :, :self.n_levels * self.n_curr_points]
        attention_weights_temporal = attention_weights_curr_temporal[:, :, :, self.n_levels * self.n_curr_points:]
        attention_weights_curr = attention_weights_curr.view(T_,
                                                             Len_q,
                                                             self.n_heads,
                                                             self.n_levels,
                                                             self.n_curr_points
                                                             ).contiguous()
        attention_weights_temporal = attention_weights_temporal.view(T_,
                                                                     Len_q,
                                                                     self.n_heads,
                                                                     self.t_window * self.n_levels,
                                                                     self.n_temporal_points
                                                                     ).contiguous()

        curr_frame_sampling_offsets = self.sampling_offsets(query).view(T_,
                                                                        Len_q,
                                                                        self.n_heads,
                                                                        self.n_levels,
                                                                        self.n_curr_points,
                                                                        2)

        return value, curr_frame_sampling_offsets, temporal_sampling_offsets, attention_weights_curr, attention_weights_temporal


class TemporalMSDeformAttnEncoder(TemporalMSDeformAttnBase):
    def forward(
            self,
            query,
            reference_points,
            input_flatten,
            input_spatial_shapes,
            input_level_start_index,
            temporal_offsets
    ):
        output = []
        input_current_spatial_shapes, input_temporal_spatial_shapes = input_spatial_shapes
        input_current_level_start_index, input_temporal_level_start_index = input_level_start_index
        T_, Len_q, _ = query.shape
        T_, Len_in, _ = input_flatten.shape
        assert reference_points.shape[-1] == 2

        (
            value,
            curr_frame_sampling_offsets,
            temporal_sampling_offsets,
            attention_weights_curr,
            attention_weights_temporal
        ) = super()._compute_deformable_attention(query, input_flatten)

        offset_normalizer = torch.stack(
            [input_current_spatial_shapes[..., 1], input_current_spatial_shapes[..., 0]], -1)
        temporal_offsets_normalizer = offset_normalizer.repeat(self.t_window, 1)

        for t in range(T_):
            current_frame_values = value[t][None]
            sampling_locations = reference_points[t][None, :, None, :, None] \
                                 + curr_frame_sampling_offsets[t][None] / offset_normalizer[None,
                                                                          None, None, :, None, :]

            output_curr = MSDeformAttnFunction.apply(
                current_frame_values, input_current_spatial_shapes, input_current_level_start_index,
                sampling_locations, attention_weights_curr[t][None], self.im2col_step)

            temporal_frames = temporal_offsets[t] + t
            temporal_frames_values = value[temporal_frames].flatten(0, 1)[None]
            temporal_ref_points = reference_points[t, :, 0][None, :, None, None, None]

            temporal_sampling_locations = temporal_ref_points \
                                          + temporal_sampling_offsets[t][
                                              None] / temporal_offsets_normalizer[None, None, None,
                                                      :, None, :]

            output_temporal = MSDeformAttnFunction.apply(
                temporal_frames_values, input_temporal_spatial_shapes,
                input_temporal_level_start_index, temporal_sampling_locations,
                attention_weights_temporal[t][None], self.im2col_step)

            frame_output = output_curr + output_temporal
            output.append(frame_output)

        output = torch.cat(output, dim=0)
        output = self.output_proj(output)

        return output


class TemporalMSDeformAttnDecoder(TemporalMSDeformAttnBase):
    def __init__(
            self,
            d_model=256,
            n_levels=4,
            t_window=2,
            n_heads=8,
            n_curr_points=4,
            n_temporal_points=2,
            dec_instance_aware_att=True
    ):
        super(TemporalMSDeformAttnDecoder, self).__init__(
            d_model=d_model,
            n_levels=n_levels,
            t_window=t_window,
            n_heads=n_heads,
            n_curr_points=n_curr_points,
            n_temporal_points=n_temporal_points)
        self.dec_instance_aware_att = dec_instance_aware_att

    def forward(
            self,
            query,
            reference_points,
            input_flatten,
            input_spatial_shapes,
            input_level_start_index,
            input_padding_mask,
            temporal_offsets
    ):
        output = []
        input_current_spatial_shapes, input_temporal_spatial_shapes = input_spatial_shapes
        input_current_level_start_index, input_temporal_level_start_index = input_level_start_index

        T_ = input_flatten.shape[0]
        embd_per_frame = query.shape[1] // T_  # maybe num_query_per_frame
        query = query.reshape([T_, embd_per_frame, query.shape[-1]])

        if reference_points.shape[0] != T_:
            reference_points = reference_points.reshape(
                (T_, embd_per_frame) + reference_points.shape[-2:])

        _, Len_q, _ = query.shape
        _, Len_in, _ = input_flatten.shape

        (
            value,
            curr_frame_sampling_offsets,
            temporal_sampling_offsets,
            attention_weights_curr,
            attention_weights_temporal
        ) = super()._compute_deformable_attention(query, input_flatten)
        if input_padding_mask is not None:
            input_padding_mask = input_padding_mask.astype(value.dtype).unsqueeze(-1)
            value *= input_padding_mask
        # To add hook for att maps visualization
        current_sampling_locations_for_att_maps, temporal_sampling_locations_for_att_maps = [], []
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [input_current_spatial_shapes[..., 1], input_current_spatial_shapes[..., 0]], -1)
            temporal_offsets_normalizer = offset_normalizer.repeat(self.t_window, 1)

            for t in range(T_):
                current_frame_values = value[t][None]
                sampling_locations = reference_points[t][None, :, None, :, None] \
                                     + curr_frame_sampling_offsets[t][None] / offset_normalizer[
                                                                              None, None, None, :,
                                                                              None, :]

                current_sampling_locations_for_att_maps.append(sampling_locations)
                output_curr = MSDeformAttnFunction.apply(
                    current_frame_values, input_current_spatial_shapes,
                    input_current_level_start_index,
                    sampling_locations, attention_weights_curr[t][None], self.im2col_step
                )

                temporal_frames = temporal_offsets[t] + t
                temporal_frames_values = value[temporal_frames].flatten(0, 1)[None]

                if self.dec_instance_aware_att:
                    temporal_ref_points = reference_points[temporal_frames].transpose(0, 1).flatten(
                        1, 2)[None, :, None, :, None]
                else:
                    temporal_ref_points = reference_points[t].repeat(1, self.t_window, 1)[None, :,
                                          None, :, None]

                temporal_sampling_locations = temporal_ref_points \
                                              + temporal_sampling_offsets[t][
                                                  None] / temporal_offsets_normalizer[None, None,
                                                          None, :, None, :]

                temporal_sampling_locations_for_att_maps.append(temporal_sampling_locations)

                # In order to avoid a for loop that computes the attention for each temporal
                # frame, we STACK them all on the resolution level axis.
                output_temporal = MSDeformAttnFunction.apply(
                    temporal_frames_values, input_temporal_spatial_shapes,
                    input_temporal_level_start_index, temporal_sampling_locations,
                    attention_weights_temporal[t][None], self.im2col_step)

                frame_output = output_curr + output_temporal
                output.append(frame_output)

        elif reference_points.shape[-1] == 4:
            for t in range(T_):
                current_frame_values = value[t: t + 1]  # 1, L, N_H, D_H
                current_frame_ref_points = reference_points[t: t + 1]  # 1, L, N_L, 4
                current_sampling_offsets = curr_frame_sampling_offsets[t: t + 1]  # 1, L, N_H, N_L, N_P, 2
                sampling_locations = current_frame_ref_points[:, :, None, :, None, :2] \
                                     + (current_sampling_offsets / self.n_curr_points) * \
                                     current_frame_ref_points[:, :, None, :, None, 2:] * 0.5

                current_sampling_locations_for_att_maps.append(sampling_locations)
                output_curr = MSDeformAttnFunction.apply(
                    current_frame_values, input_current_spatial_shapes,
                    input_current_level_start_index, sampling_locations,
                    attention_weights_curr[t: t + 1], self.im2col_step
                )

                temporal_frames = temporal_offsets[t] + t
                temporal_frames_values = value[temporal_frames].flatten(0, 1)[None]  # 1, L * (T_ - 1), N_H, D_H

                if self.dec_instance_aware_att:
                    if reference_points.shape[2] == 1:  # means num of level = 1
                        reference_points = reference_points.repeat(1, 1, self.n_levels, 1)
                    temporal_ref_points = reference_points[temporal_frames].transpose(0, 1).flatten(
                        1, 2)[None, :, None, :, None]
                else:
                    if reference_points.shape[2] == 1:  # means num of level = 1
                        temporal_ref_points = current_frame_ref_points[:, :, None, :, None]
                    else:
                        temporal_ref_points = reference_points[t: t + 1].repeat(1, 1, self.t_window, 1)[:, :, None, :,
                                              None]

                temporal_sampling_locations = temporal_ref_points[..., :2] \
                                              + (temporal_sampling_offsets[t: t + 1] / self.n_temporal_points) * \
                                              temporal_ref_points[..., 2:] * 0.5

                temporal_sampling_locations_for_att_maps.append(temporal_sampling_locations)
                output_temporal = MSDeformAttnFunction.apply(
                    temporal_frames_values, input_temporal_spatial_shapes,
                    input_temporal_level_start_index, temporal_sampling_locations,
                    attention_weights_temporal[t: t + 1], self.im2col_step
                )

                frame_output = output_curr + output_temporal
                output.append(frame_output)

        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(
                    reference_points.shape[-1]))

        output = torch.cat(output, dim=0).flatten(0, 1)[None]
        output = self.output_proj(output)

        return (output, current_sampling_locations_for_att_maps, temporal_sampling_locations_for_att_maps,
                attention_weights_curr, attention_weights_temporal)


class TemporalMSDeformAttnDecoderCurrFrame(TemporalMSDeformAttnBase):
    def __init__(
            self,
            d_model=256,
            n_levels=4,
            t_window=2,
            n_heads=8,
            n_curr_points=4,
            n_temporal_points=2,
            dec_instance_aware_att=True
    ):
        super().__init__(
            d_model=d_model,
            n_levels=n_levels,
            t_window=t_window,
            n_heads=n_heads,
            n_curr_points=n_curr_points,
            n_temporal_points=n_temporal_points)
        self.dec_instance_aware_att = dec_instance_aware_att

    def forward(
            self,
            query,
            reference_points,
            input_flatten,
            input_spatial_shapes,
            input_level_start_index,
            input_padding_mask,
            temporal_offsets
    ):
        """
        :param query: 1, N_q, D
        :param reference_points:
        :param input_flatten:
        :param input_spatial_shapes:
        :param input_level_start_index:
        :param input_padding_mask:
        :param temporal_offsets:
        :return:
        """
        output = []
        input_current_spatial_shapes, input_temporal_spatial_shapes = input_spatial_shapes
        input_current_level_start_index, input_temporal_level_start_index = input_level_start_index

        T_ = input_flatten.shape[0]
        _, Len_q, _ = query.shape
        _, Len_in, _ = input_flatten.shape

        (
            value,
            curr_frame_sampling_offsets,
            temporal_sampling_offsets,
            attention_weights_curr,
            attention_weights_temporal
        ) = self._compute_deformable_attention(query, input_flatten)
        if input_padding_mask is not None:
            input_padding_mask = input_padding_mask.astype(value.dtype).unsqueeze(-1)
            value *= input_padding_mask
        # To add hook for att maps visualization
        current_sampling_locations_for_att_maps, temporal_sampling_locations_for_att_maps = [], []
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [input_current_spatial_shapes[..., 1], input_current_spatial_shapes[..., 0]], -1)
            temporal_offsets_normalizer = offset_normalizer.repeat(self.t_window, 1)

            for t in range(T_):
                current_frame_values = value[t][None]
                sampling_locations = reference_points[t][None, :, None, :, None] \
                                     + curr_frame_sampling_offsets[t][None] / offset_normalizer[
                                                                              None, None, None, :,
                                                                              None, :]

                current_sampling_locations_for_att_maps.append(sampling_locations)
                output_curr = MSDeformAttnFunction.apply(
                    current_frame_values, input_current_spatial_shapes,
                    input_current_level_start_index,
                    sampling_locations, attention_weights_curr[t][None], self.im2col_step
                )

                temporal_frames = temporal_offsets[t] + t
                temporal_frames_values = value[temporal_frames].flatten(0, 1)[None]

                if self.dec_instance_aware_att:
                    temporal_ref_points = reference_points[temporal_frames].transpose(0, 1).flatten(
                        1, 2)[None, :, None, :, None]
                else:
                    temporal_ref_points = reference_points[t].repeat(1, self.t_window, 1)[None, :,
                                          None, :, None]

                temporal_sampling_locations = temporal_ref_points \
                                              + temporal_sampling_offsets[t][
                                                  None] / temporal_offsets_normalizer[None, None,
                                                          None, :, None, :]

                temporal_sampling_locations_for_att_maps.append(temporal_sampling_locations)

                # In order to avoid a for loop that computes the attention for each temporal
                # frame, we STACK them all on the resolution level axis.
                output_temporal = MSDeformAttnFunction.apply(
                    temporal_frames_values, input_temporal_spatial_shapes,
                    input_temporal_level_start_index, temporal_sampling_locations,
                    attention_weights_temporal[t][None], self.im2col_step)

                frame_output = output_curr + output_temporal
                output.append(frame_output)

        elif reference_points.shape[-1] == 4:
            current_frame_values = value[0: 1]  # 1, L, N_H, D_H
            current_frame_ref_points = reference_points[0: 1]  # 1, L, N_L, 4
            current_sampling_offsets = curr_frame_sampling_offsets  # 1, L, N_H, N_L, N_P, 2
            sampling_locations = current_frame_ref_points[:, :, None, :, None, :2] \
                                 + (current_sampling_offsets / self.n_curr_points) * \
                                 current_frame_ref_points[:, :, None, :, None, 2:] * 0.5

            current_sampling_locations_for_att_maps.append(sampling_locations)
            output_curr = MSDeformAttnFunction.apply(
                current_frame_values, input_current_spatial_shapes,
                input_current_level_start_index, sampling_locations,
                attention_weights_curr, self.im2col_step
            )

            temporal_frames = temporal_offsets[0]
            temporal_frames_values = value[temporal_frames].flatten(0, 1)[None]  # 1, L * (T_ - 1), N_H, D_H

            if self.dec_instance_aware_att:
                if reference_points.shape[2] == 1:  # means num of level = 1
                    reference_points = reference_points.repeat(1, 1, self.n_levels, 1)
                temporal_ref_points = reference_points[temporal_frames].transpose(0, 1).flatten(
                    1, 2)[None, :, None, :, None]
            else:
                if reference_points.shape[2] == 1:  # means num of level = 1
                    temporal_ref_points = current_frame_ref_points[:, :, None, :, None]
                else:
                    temporal_ref_points = reference_points.repeat(1, 1, self.t_window, 1)[:, :, None, :, None]

            temporal_sampling_locations = temporal_ref_points[..., :2] \
                                          + (temporal_sampling_offsets / self.n_temporal_points) * \
                                          temporal_ref_points[..., 2:] * 0.5

            temporal_sampling_locations_for_att_maps.append(temporal_sampling_locations)
            output_temporal = MSDeformAttnFunction.apply(
                temporal_frames_values, input_temporal_spatial_shapes,
                input_temporal_level_start_index, temporal_sampling_locations,
                attention_weights_temporal, self.im2col_step
            )

            frame_output = output_curr + output_temporal
            output.append(frame_output)

        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(
                    reference_points.shape[-1]))

        output = torch.cat(output, dim=0).flatten(0, 1)[None]
        output = self.output_proj(output)

        return (output, current_sampling_locations_for_att_maps, temporal_sampling_locations_for_att_maps,
                attention_weights_curr, attention_weights_temporal)

    def _compute_deformable_attention(self, query, input_flatten):
        _, Len_q, _ = query.shape
        T_, Len_in, _ = input_flatten.shape

        value = self.value_proj(input_flatten)
        value = value.view(T_, Len_in, self.n_heads, self.d_model // self.n_heads)

        temporal_sampling_offsets = self.temporal_sampling_offsets(query).view(-1,
                                                                               Len_q,
                                                                               self.n_heads,
                                                                               self.t_window,
                                                                               self.n_levels,
                                                                               self.n_temporal_points,
                                                                               2)
        temporal_sampling_offsets = temporal_sampling_offsets.flatten(3, 4)  # (T_, L, N_H, t*N_L, N_P, 2)
        temporal_attention_weights = self.temporal_attention_weights(query)
        temporal_attention_weights = temporal_attention_weights.view(-1,
                                                                     Len_q,
                                                                     self.n_heads,
                                                                     self.t_window * self.n_levels * self.n_temporal_points
                                                                     )
        curr_frame_attention_weights = self.attention_weights(query).view(-1,
                                                                          Len_q,
                                                                          self.n_heads,
                                                                          self.n_levels * self.n_curr_points
                                                                          )
        attention_weights_curr_temporal = torch.cat(
            [curr_frame_attention_weights, temporal_attention_weights], dim=3)
        attention_weights_curr_temporal = F.softmax(attention_weights_curr_temporal, -1)
        attention_weights_curr = attention_weights_curr_temporal[:, :, :, :self.n_levels * self.n_curr_points]
        attention_weights_temporal = attention_weights_curr_temporal[:, :, :, self.n_levels * self.n_curr_points:]
        attention_weights_curr = attention_weights_curr.view(-1,
                                                             Len_q,
                                                             self.n_heads,
                                                             self.n_levels,
                                                             self.n_curr_points
                                                             ).contiguous()
        attention_weights_temporal = attention_weights_temporal.view(-1,
                                                                     Len_q,
                                                                     self.n_heads,
                                                                     self.t_window * self.n_levels,
                                                                     self.n_temporal_points
                                                                     ).contiguous()

        curr_frame_sampling_offsets = self.sampling_offsets(query).view(-1,
                                                                        Len_q,
                                                                        self.n_heads,
                                                                        self.n_levels,
                                                                        self.n_curr_points,
                                                                        2)

        return value, curr_frame_sampling_offsets, temporal_sampling_offsets, attention_weights_curr, attention_weights_temporal


class BatchTemporalMSDeformAttnDecoder(TemporalMSDeformAttnBase):
    def __init__(
            self,
            d_model=256,
            n_levels=4,
            t_window=2,
            n_heads=8,
            n_curr_points=4,
            n_temporal_points=2,
            dec_instance_aware_att=True
    ):
        super().__init__(
            d_model=d_model,
            n_levels=n_levels,
            t_window=t_window,
            n_heads=n_heads,
            n_curr_points=n_curr_points,
            n_temporal_points=n_temporal_points)
        self.dec_instance_aware_att = dec_instance_aware_att

    def forward(
            self,
            query,
            reference_points,
            input_flatten,
            input_spatial_shapes,
            input_level_start_index,
            input_padding_mask,
            temporal_offsets
    ):
        """
        :param query: T, L, C
        :param reference_points: T, L, N_L, 4
        :param input_flatten:
        :param input_spatial_shapes:
        :param input_level_start_index:
        :param input_padding_mask:
        :param temporal_offsets:
        :return:
        """
        output = []
        input_current_spatial_shapes, input_temporal_spatial_shapes = input_spatial_shapes
        input_current_level_start_index, input_temporal_level_start_index = input_level_start_index

        T_ = input_flatten.shape[0]
        query_per_frame = query.shape[1] // T_
        query = query.reshape([-1, T_, query_per_frame, query.shape[-1]])

        if reference_points.shape[0] != T_:
            reference_points = reference_points.reshape((T_, query_per_frame) + reference_points.shape[-2:])

        T_q, _, Len_q, _ = query.shape
        _, Len_in, _ = input_flatten.shape

        (
            value,
            curr_frame_sampling_offsets,
            temporal_sampling_offsets,
            attention_weights_curr,
            attention_weights_temporal
        ) = self._compute_deformable_attention(query, input_flatten)
        if input_padding_mask is not None:
            input_padding_mask = input_padding_mask.astype(value.dtype).unsqueeze(-1)
            value *= input_padding_mask
        # To add hook for att maps visualization
        current_sampling_locations_for_att_maps, temporal_sampling_locations_for_att_maps = [], []
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [input_current_spatial_shapes[..., 1], input_current_spatial_shapes[..., 0]], -1)
            temporal_offsets_normalizer = offset_normalizer.repeat(self.t_window, 1)

            for t in range(T_):
                current_frame_values = value[t][None]
                sampling_locations = reference_points[t][None, :, None, :, None] \
                                     + curr_frame_sampling_offsets[t][None] / offset_normalizer[
                                                                              None, None, None, :,
                                                                              None, :]

                current_sampling_locations_for_att_maps.append(sampling_locations)
                output_curr = MSDeformAttnFunction.apply(
                    current_frame_values, input_current_spatial_shapes,
                    input_current_level_start_index,
                    sampling_locations, attention_weights_curr[t][None], self.im2col_step
                )

                temporal_frames = temporal_offsets[t] + t
                temporal_frames_values = value[temporal_frames].flatten(0, 1)[None]

                if self.dec_instance_aware_att:
                    temporal_ref_points = reference_points[temporal_frames].transpose(0, 1).flatten(
                        1, 2)[None, :, None, :, None]
                else:
                    temporal_ref_points = reference_points[t].repeat(1, self.t_window, 1)[None, :,
                                          None, :, None]

                temporal_sampling_locations = temporal_ref_points \
                                              + temporal_sampling_offsets[t][
                                                  None] / temporal_offsets_normalizer[None, None,
                                                          None, :, None, :]

                temporal_sampling_locations_for_att_maps.append(temporal_sampling_locations)

                # In order to avoid a for loop that computes the attention for each temporal
                # frame, we STACK them all on the resolution level axis.
                output_temporal = MSDeformAttnFunction.apply(
                    temporal_frames_values, input_temporal_spatial_shapes,
                    input_temporal_level_start_index, temporal_sampling_locations,
                    attention_weights_temporal[t][None], self.im2col_step)

                frame_output = output_curr + output_temporal
                output.append(frame_output)

        elif reference_points.shape[-1] == 4:
            for t_q in range(T_q):
                for t in range(T_):
                    current_frame_values = value[t: t + 1]  # 1, L, N_H, D_H
                    current_frame_ref_points = reference_points[t: t + 1]  # 1, L, N_L, 4
                    current_sampling_offsets = curr_frame_sampling_offsets[t: t + 1]  # 1, L, N_H, N_L, N_P, 2
                    sampling_locations = current_frame_ref_points[:, :, None, :, None, :2] \
                                         + (current_sampling_offsets / self.n_curr_points) * \
                                         current_frame_ref_points[:, :, None, :, None, 2:] * 0.5

                    current_sampling_locations_for_att_maps.append(sampling_locations)
                    output_curr = MSDeformAttnFunction.apply(
                        current_frame_values, input_current_spatial_shapes,
                        input_current_level_start_index, sampling_locations,
                        attention_weights_curr[t: t + 1], self.im2col_step
                    )

                    temporal_frames = temporal_offsets[t] + t
                    temporal_frames_values = value[temporal_frames].flatten(0, 1)[None]  # 1, L * (T_ - 1), N_H, D_H

                    if self.dec_instance_aware_att:
                        if reference_points.shape[2] == 1:  # means num of level = 1
                            reference_points = reference_points.repeat(1, 1, self.n_levels, 1)
                        temporal_ref_points = reference_points[temporal_frames].transpose(0, 1).flatten(
                            1, 2)[None, :, None, :, None]
                    else:
                        if reference_points.shape[2] == 1:  # means num of level = 1
                            temporal_ref_points = current_frame_ref_points[:, :, None, :, None]
                        else:
                            temporal_ref_points = reference_points[t: t + 1].repeat(1, 1, self.t_window, 1)[:, :, None,
                                                  :, None]

                    temporal_sampling_locations = temporal_ref_points[..., :2] \
                                                  + (temporal_sampling_offsets[t: t + 1] / self.n_temporal_points) * \
                                                  temporal_ref_points[..., 2:] * 0.5

                    temporal_sampling_locations_for_att_maps.append(temporal_sampling_locations)
                    output_temporal = MSDeformAttnFunction.apply(
                        temporal_frames_values, input_temporal_spatial_shapes,
                        input_temporal_level_start_index, temporal_sampling_locations,
                        attention_weights_temporal[t: t + 1], self.im2col_step
                    )

                    frame_output = output_curr + output_temporal
                    output.append(frame_output)

        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(
                    reference_points.shape[-1]))

        output = torch.cat(output, dim=0).flatten(0, 1)[None]
        output = self.output_proj(output)

        return (output, current_sampling_locations_for_att_maps, temporal_sampling_locations_for_att_maps,
                attention_weights_curr, attention_weights_temporal)

    def _compute_deformable_attention(self, query, input_flatten):
        T_q, _, Len_q, _ = query.shape
        T_, Len_in, _ = input_flatten.shape

        value = self.value_proj(input_flatten)
        value = value.view(T_, Len_in, self.n_heads, self.d_model // self.n_heads)

        temporal_sampling_offsets = self.temporal_sampling_offsets(query).view(T_q,
                                                                               -1,
                                                                               Len_q,
                                                                               self.n_heads,
                                                                               self.t_window,
                                                                               self.n_levels,
                                                                               self.n_temporal_points,
                                                                               2)
        temporal_sampling_offsets = temporal_sampling_offsets.flatten(3, 4)  # (T_, L, N_H, t*N_L, N_P, 2)
        temporal_attention_weights = self.temporal_attention_weights(query)
        temporal_attention_weights = temporal_attention_weights.view(T_q,
                                                                     -1,
                                                                     Len_q,
                                                                     self.n_heads,
                                                                     self.t_window * self.n_levels * self.n_temporal_points
                                                                     )
        curr_frame_attention_weights = self.attention_weights(query).view(T_q, -1,
                                                                          Len_q,
                                                                          self.n_heads,
                                                                          self.n_levels * self.n_curr_points
                                                                          )
        attention_weights_curr_temporal = torch.cat(
            [curr_frame_attention_weights, temporal_attention_weights], dim=3)
        attention_weights_curr_temporal = F.softmax(attention_weights_curr_temporal, -1)
        attention_weights_curr = attention_weights_curr_temporal[:, :, :, :self.n_levels * self.n_curr_points]
        attention_weights_temporal = attention_weights_curr_temporal[:, :, :, self.n_levels * self.n_curr_points:]
        attention_weights_curr = attention_weights_curr.view(T_q, -1,
                                                             Len_q,
                                                             self.n_heads,
                                                             self.n_levels,
                                                             self.n_curr_points
                                                             ).contiguous()
        attention_weights_temporal = attention_weights_temporal.view(T_q, -1,
                                                                     Len_q,
                                                                     self.n_heads,
                                                                     self.t_window * self.n_levels,
                                                                     self.n_temporal_points
                                                                     ).contiguous()

        curr_frame_sampling_offsets = self.sampling_offsets(query).view(T_q, -1,
                                                                        Len_q,
                                                                        self.n_heads,
                                                                        self.n_levels,
                                                                        self.n_curr_points,
                                                                        2)

        return value, curr_frame_sampling_offsets, temporal_sampling_offsets, attention_weights_curr, attention_weights_temporal


class TemporalMSDeformAttnDecoder2(TemporalMSDeformAttnBase):
    def __init__(
            self,
            d_model=256,
            n_levels=4,
            t_window=2,
            n_heads=8,
            n_curr_points=4,
            n_temporal_points=2,
            dec_instance_aware_att=True
    ):
        super().__init__(
            d_model=d_model,
            n_levels=n_levels,
            t_window=t_window,
            n_heads=n_heads,
            n_curr_points=n_curr_points,
            n_temporal_points=n_temporal_points)
        self.dec_instance_aware_att = dec_instance_aware_att

    def forward(
            self,
            query,
            reference_points,
            input_flatten,
            input_spatial_shapes,
            input_level_start_index,
            input_padding_mask,
            temporal_offsets
    ):
        output = []
        input_current_spatial_shapes, input_temporal_spatial_shapes = input_spatial_shapes
        input_current_level_start_index, input_temporal_level_start_index = input_level_start_index

        T_ = input_flatten.shape[0]
        _, Len_q, _ = query.shape
        _, Len_in, _ = input_flatten.shape

        (
            value,
            curr_frame_sampling_offsets,
            temporal_sampling_offsets,
            attention_weights_curr,
            attention_weights_temporal
        ) = super()._compute_deformable_attention(query, input_flatten)
        if input_padding_mask is not None:
            input_padding_mask = input_padding_mask.astype(value.dtype).unsqueeze(-1)
            value *= input_padding_mask
        # To add hook for att maps visualization
        current_sampling_locations_for_att_maps, temporal_sampling_locations_for_att_maps = [], []
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [input_current_spatial_shapes[..., 1], input_current_spatial_shapes[..., 0]], -1)
            temporal_offsets_normalizer = offset_normalizer.repeat(self.t_window, 1)

            for t in range(T_):
                current_frame_values = value[t][None]
                sampling_locations = reference_points[t][None, :, None, :, None] \
                                     + curr_frame_sampling_offsets[t][None] / offset_normalizer[
                                                                              None, None, None, :,
                                                                              None, :]

                current_sampling_locations_for_att_maps.append(sampling_locations)
                output_curr = MSDeformAttnFunction.apply(
                    current_frame_values, input_current_spatial_shapes,
                    input_current_level_start_index,
                    sampling_locations, attention_weights_curr[t][None], self.im2col_step
                )

                temporal_frames = temporal_offsets[t] + t
                temporal_frames_values = value[temporal_frames].flatten(0, 1)[None]

                if self.dec_instance_aware_att:
                    temporal_ref_points = reference_points[temporal_frames].transpose(0, 1).flatten(
                        1, 2)[None, :, None, :, None]
                else:
                    temporal_ref_points = reference_points[t].repeat(1, self.t_window, 1)[None, :,
                                          None, :, None]

                temporal_sampling_locations = temporal_ref_points \
                                              + temporal_sampling_offsets[t][
                                                  None] / temporal_offsets_normalizer[None, None,
                                                          None, :, None, :]

                temporal_sampling_locations_for_att_maps.append(temporal_sampling_locations)

                # In order to avoid a for loop that computes the attention for each temporal
                # frame, we STACK them all on the resolution level axis.
                output_temporal = MSDeformAttnFunction.apply(
                    temporal_frames_values, input_temporal_spatial_shapes,
                    input_temporal_level_start_index, temporal_sampling_locations,
                    attention_weights_temporal[t][None], self.im2col_step)

                frame_output = output_curr + output_temporal
                output.append(frame_output)

        elif reference_points.shape[-1] == 4:
            for t in range(T_):
                current_frame_values = value[t: t + 1]  # 1, L, N_H, D_H
                current_frame_ref_points = reference_points[t: t + 1]  # 1, L, N_L, 4
                current_sampling_offsets = curr_frame_sampling_offsets[t: t + 1]  # 1, L, N_H, N_L, N_P, 2
                sampling_locations = current_frame_ref_points[:, :, None, :, None, :2] \
                                     + (current_sampling_offsets / self.n_curr_points) * \
                                     current_frame_ref_points[:, :, None, :, None, 2:] * 0.5

                current_sampling_locations_for_att_maps.append(sampling_locations)
                output_curr = MSDeformAttnFunction.apply(
                    current_frame_values, input_current_spatial_shapes,
                    input_current_level_start_index, sampling_locations,
                    attention_weights_curr[t: t + 1], self.im2col_step
                )

                temporal_frames = temporal_offsets[t] + t
                temporal_frames_values = value[temporal_frames].flatten(0, 1)[None]  # 1, L * (T_ - 1), N_H, D_H

                if self.dec_instance_aware_att:
                    if reference_points.shape[2] == 1:  # means num of level = 1
                        reference_points = reference_points.repeat(1, 1, self.n_levels, 1)
                    temporal_ref_points = reference_points[temporal_frames].transpose(0, 1).flatten(
                        1, 2)[None, :, None, :, None]
                else:
                    if reference_points.shape[2] == 1:  # means num of level = 1
                        temporal_ref_points = current_frame_ref_points[:, :, None, :, None]
                    else:
                        temporal_ref_points = reference_points[t: t + 1].repeat(1, 1, self.t_window, 1)[:, :, None, :,
                                              None]

                temporal_sampling_locations = temporal_ref_points[..., :2] \
                                              + (temporal_sampling_offsets[t: t + 1] / self.n_temporal_points) * \
                                              temporal_ref_points[..., 2:] * 0.5

                temporal_sampling_locations_for_att_maps.append(temporal_sampling_locations)
                output_temporal = MSDeformAttnFunction.apply(
                    temporal_frames_values, input_temporal_spatial_shapes,
                    input_temporal_level_start_index, temporal_sampling_locations,
                    attention_weights_temporal[t: t + 1], self.im2col_step
                )

                frame_output = output_curr + output_temporal
                output.append(frame_output)

        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(
                    reference_points.shape[-1]))

        output = torch.cat(output, dim=0)
        output = self.output_proj(output)

        return (output, current_sampling_locations_for_att_maps, temporal_sampling_locations_for_att_maps,
                attention_weights_curr, attention_weights_temporal)


from ..functions.ms_deform_attn_func import ms_deform_attn_core_pytorch_key_aware
import torch.utils.checkpoint as cp

class TemporalMSDeformAttnKeyAware(nn.Module):
    def __init__(
            self,
            d_model=256,
            n_levels=4,
            t_window=2,
            n_heads=8,
            n_curr_points=4,
            n_temporal_points=2,
            use_checkpoint=True,
            separate_attn=False,
    ):
        super().__init__()
        """
        Multi-Scale Deformable Attention Module
        :param d_model          hidden dimension
        :param n_levels         number of feature levels
        :param n_heads          number of attention heads
        :param n_curr_points    number of sampling points per attention head per feature level from
                                each query corresponding frame
        :param n_temporal_points    number of sampling points per attention head per feature level
                                    from temporal frames
        """
        if d_model % n_heads != 0:
            raise ValueError(
                'd_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        if not _is_power_of_2(_d_per_head):
            warnings.warn(
                "You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation.")

        self.im2col_step = 64
        self.d_model = d_model
        self.n_levels = n_levels
        self.t_window = t_window
        self.n_heads = n_heads
        self.n_curr_points = n_curr_points
        self.n_temporal_points = n_temporal_points
        self.use_checkpoint = use_checkpoint
        self.separate_attn = separate_attn

        # Used for sampling and attention in the current frame
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_curr_points * 2)

        # Used for sampling and attention in the prev or post frames
        self.temporal_sampling_offsets = nn.Linear(d_model, n_heads * n_levels * t_window * n_temporal_points * 2)

        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.query_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        # sampling offset initialized weight to 0, so at initial iterations the bias is the only that matters at all
        constant_(self.temporal_sampling_offsets.weight.data, 0.)

        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)

        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0])

        # curr_frame init
        curr_grid_init = grid_init.view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels,
                                                                      self.n_curr_points, 1)
        for i in range(self.n_curr_points):
            curr_grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(curr_grid_init.reshape(-1))

        # temporal init
        temporal_grid_init = grid_init.view(self.n_heads, 1, 1, 1, 2).repeat(1, self.n_levels,
                                                                             self.t_window,
                                                                             self.n_temporal_points,
                                                                             1)

        for i in range(self.n_temporal_points):
            temporal_grid_init[:, :, :, i, :] *= i + 1

        with torch.no_grad():
            self.temporal_sampling_offsets.bias = nn.Parameter(temporal_grid_init.reshape(-1))

        xavier_uniform_(self.key_proj.weight.data)
        constant_(self.key_proj.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)
        xavier_uniform_(self.query_proj.weight.data)
        constant_(self.query_proj.bias.data, 0.)

    def forward(
            self,
            query,
            reference_points,
            input_flatten,
            input_spatial_shapes,
            input_level_start_index,
            temporal_offsets
    ):
        output = []
        input_current_spatial_shapes, input_temporal_spatial_shapes = input_spatial_shapes
        T_, Len_q, _ = query.shape
        T_, Len_in, _ = input_flatten.shape
        assert reference_points.shape[-1] == 2

        (query_proj,
         key,
         value,
         curr_frame_sampling_offsets,
         temporal_sampling_offsets
         ) = self._compute_deformable_attention(query, input_flatten)

        offset_normalizer = torch.stack(
            [input_current_spatial_shapes[..., 1], input_current_spatial_shapes[..., 0]], -1)
        temporal_offsets_normalizer = offset_normalizer.repeat(self.t_window, 1)

        for t in range(T_):
            current_frame_values = value[t: t + 1]
            current_frame_keys = key[t: t + 1]
            sampling_locations = reference_points[t][None, :, None, :, None] \
                                 + curr_frame_sampling_offsets[t: t + 1] / offset_normalizer[None,
                                                                          None, None, :, None, :]
            temporal_frames = temporal_offsets[t] + t
            temporal_frames_values = value[temporal_frames].flatten(0, 1)[None]
            temporal_ref_points = reference_points[t, :, 0][None, :, None, None, None]

            temporal_sampling_locations = temporal_ref_points \
                                          + temporal_sampling_offsets[t: t + 1] / temporal_offsets_normalizer[None, None, None,
                                                      :, None, :]
            temporal_frames_keys = key[temporal_frames].flatten(0, 1)[None]

            if self.separate_attn:
                if not self.use_checkpoint:
                    output_curr = ms_deform_attn_core_pytorch_key_aware(
                        query=query_proj[t: t + 1],
                        value=current_frame_values,
                        key=current_frame_keys,
                        value_spatial_shapes=input_current_spatial_shapes,
                        sampling_locations=sampling_locations,
                    )
                else:
                    output_curr = cp.checkpoint(
                        ms_deform_attn_core_pytorch_key_aware,
                        query_proj[t: t + 1],
                        current_frame_values,
                        current_frame_keys,
                        input_current_spatial_shapes,
                        sampling_locations,
                        use_reentrant=False
                    )

                if not self.use_checkpoint:
                    output_temporal = ms_deform_attn_core_pytorch_key_aware(
                        query=query_proj[t: t + 1],
                        value=temporal_frames_values,
                        key=temporal_frames_keys,
                        value_spatial_shapes=input_temporal_spatial_shapes,
                        sampling_locations=temporal_sampling_locations,
                    )
                else:
                    output_temporal = cp.checkpoint(
                        ms_deform_attn_core_pytorch_key_aware,
                        query_proj[t: t + 1],
                        temporal_frames_values,
                        temporal_frames_keys,
                        input_temporal_spatial_shapes,
                        temporal_sampling_locations,
                        use_reentrant=False
                    )
            else:
                if not self.use_checkpoint:
                    output_curr, output_temporal = self._key_aware_attention_combine(
                        query_proj[t: t + 1], current_frame_values, current_frame_keys, input_current_spatial_shapes, sampling_locations,
                        temporal_frames_values, temporal_frames_keys, input_temporal_spatial_shapes, temporal_sampling_locations
                    )
                else:
                    output_curr, output_temporal = cp.checkpoint(
                        self._key_aware_attention_combine,
                        query_proj[t: t + 1],
                        current_frame_values,
                        current_frame_keys,
                        input_current_spatial_shapes,
                        sampling_locations,
                        temporal_frames_values,
                        temporal_frames_keys,
                        input_temporal_spatial_shapes,
                        temporal_sampling_locations,
                        use_reentrant=False,
                    )

            frame_output = output_curr + output_temporal
            output.append(frame_output)

        output = torch.cat(output, dim=0)
        output = self.output_proj(output)

        return output

    # Computes current/temporal sampling offsets and attention weights,
    # which are treated different for the encoder and decoder later on
    def _compute_deformable_attention(self, query, input_flatten):
        T_, Len_q, _ = query.shape
        T_, Len_in, _ = input_flatten.shape

        query_proj = self.query_proj(query)
        key = self.key_proj(input_flatten)
        key = key.view(T_, Len_in, self.n_heads, self.d_model // self.n_heads)
        value = self.value_proj(input_flatten)
        value = value.view(T_, Len_in, self.n_heads, self.d_model // self.n_heads)

        temporal_sampling_offsets = self.temporal_sampling_offsets(query).view(T_,
                                                                               Len_q,
                                                                               self.n_heads,
                                                                               self.t_window,
                                                                               self.n_levels,
                                                                               self.n_temporal_points,
                                                                               2)
        temporal_sampling_offsets = temporal_sampling_offsets.flatten(3, 4)  # (T_, L, N_H, t*N_L, N_P, 2)

        curr_frame_sampling_offsets = self.sampling_offsets(query).view(T_,
                                                                        Len_q,
                                                                        self.n_heads,
                                                                        self.n_levels,
                                                                        self.n_curr_points,
                                                                        2)

        return query_proj, key, value, curr_frame_sampling_offsets, temporal_sampling_offsets


    def _key_aware_attention_combine(
            self, query, current_frame_values, current_frame_keys, input_current_spatial_shapes, sampling_locations,
            temporal_frames_values, temporal_frames_keys, input_temporal_spatial_shapes, temporal_sampling_locations
    ):
        N_, Lq_, M_, L_, P_, _ = sampling_locations.shape
        current_frame_values, attention_weights_curr = self._compute_attention_weights_and_values(
            query, current_frame_values, current_frame_keys, input_current_spatial_shapes, sampling_locations
        )
        temporal_frames_values, attention_weights_temporal = self._compute_attention_weights_and_values(
            query, temporal_frames_values, temporal_frames_keys, input_temporal_spatial_shapes, temporal_sampling_locations
        )
        attention_weights_curr_temporal = torch.cat([attention_weights_curr, attention_weights_temporal], dim=-1)
        attention_weights_curr_temporal = F.softmax(attention_weights_curr_temporal, -1)
        attention_weights_curr = attention_weights_curr_temporal[..., :attention_weights_curr.shape[-1]]
        attention_weights_temporal = attention_weights_curr_temporal[..., attention_weights_curr.shape[-1]:]

        output_curr = attention_weights_curr.matmul(current_frame_values)
        output_curr = output_curr.squeeze(-2).view(N_, M_, Lq_, -1).permute(0, 2, 1, 3)
        output_curr = output_curr.flatten(2).contiguous()
        output_temporal = attention_weights_temporal.matmul(temporal_frames_values)
        output_temporal = output_temporal.squeeze(-2).view(N_, M_, Lq_, -1).permute(0, 2, 1, 3)
        output_temporal = output_temporal.flatten(2).contiguous()

        return output_curr, output_temporal

    def _compute_attention_weights_and_values(self, query, value, key, value_spatial_shapes, sampling_locations,):
        # N: batch szie; S_: total value num;   M_: head num 8; mD: 256/M (32)
        # Lq_: len q;  L_: num levels (4); P_: sample point per-level (4)
        N_, S_, M_, D_ = value.shape
        _, Lq_, M_, L_, P_, _ = sampling_locations.shape
        value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
        key_list = key.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
        sampling_grids = 2 * sampling_locations - 1
        sampling_value_list = []
        sampling_key_list = []
        for lid_, (H_, W_) in enumerate(value_spatial_shapes):
            # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
            value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_ * M_, D_, H_, W_)
            key_l_ = key_list[lid_].flatten(2).transpose(1, 2).reshape(N_ * M_, D_, H_, W_)
            # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
            sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
            # N_*M_, D_, Lq_, P_
            sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                              mode='bilinear', padding_mode='zeros', align_corners=False)
            sampling_value_list.append(sampling_value_l_)

            # N_*M_, D_, Lq_, P_
            sampling_key_l__ = F.grid_sample(key_l_, sampling_grid_l_,
                                             mode='bilinear', padding_mode='zeros', align_corners=False)
            sampling_key_list.append(sampling_key_l__)
        # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
        key = torch.stack(sampling_key_list, dim=-2).flatten(-2)
        value = torch.stack(sampling_value_list, dim=-2).flatten(-2)
        # N_*M_, D_, Lq_, P_ -> N_*M_, D_, Lq_, L_, P_ -> N_*M_, D_, Lq_, L_*P_

        key = key.permute(0, 2, 3, 1).flatten(0, 1)  # N_*M_, D_, Lq_, L_*P_ -> N*M, Lq, L*P, D -> N*M*Lq, L*P, D

        N_, Lq, DD_ = query.shape
        query = query.view(N_, Lq, M_, DD_ // M_)
        query = query.permute(0, 2, 1, 3).flatten(0, 2)  # N, Lq, M, D -> N, M, Lq, D -> N*M*Lq, D

        query = query.unsqueeze(-2)  # N*M*Lq, D-> N*M*Lq, 1, D
        dk = query.size()[-1]

        attention_weights = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(dk)
        # attention_weights = F.softmax(attention_weights, -1)

        value = value.permute(0, 2, 3, 1).flatten(0, 1)  # N*M*Lq, L*P, D

        # output = attention_weights.matmul(value)  # N*M, Lq, 1,  L*P x N*M*Lq, L*P, D -> N*M, Lq, 1,  D
        #
        # output = output.squeeze(-2).view(N_, M_, Lq_, D_).permute(0, 2, 1, 3)  # N*M, Lq, 1,  D -> N, Lq, M,  D
        #
        # output = output.flatten(2)
        #
        # return output.contiguous()

        return value, attention_weights
