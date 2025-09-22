# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/position_encoding.py
# Copyright (c) Medical AI Lab, Alibaba DAMO Academy

"""
Various positional encodings for the transformer.
"""
import math

import numpy as np
import torch
from torch import nn


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    Adapted for 2D data, add pos_cache to save time
    """

    def __init__(self, num_total_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        # For 2D, split features equally between x and y dimensions
        self.num_pos_feats = num_total_pos_feats // 2
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        self.pos_cache = None

    def forward(self, x, mask=None):
        if self.pos_cache is not None and x.shape == self.pos_cache.shape and mask is None:
            return self.pos_cache
        if mask is None:
            # For 2D data: (batch, height, width)
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        # For 2D: only y and x dimensions
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        # Same dimension features for both x and y
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        # Concatenate y and x position encodings
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        self.pos_cache = pos
        return pos
