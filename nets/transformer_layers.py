# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
# Copyright (c) Medical AI Lab, Alibaba DAMO Academy

import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F, Conv2d
from torch.amp.autocast_mode import autocast


class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="gelu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)  # query_feat + query_emb
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="gelu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        # print(f"#######################")
        # print(f"CrossAttentionLayer, tgt.shape: {tgt.shape}, memory.shape: {memory.shape}, memory_mask.shape: {memory_mask.shape}, pos.shape: {pos.shape}, query_pos.shape: {query_pos.shape}")
        
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),  # learnable query_feat + query_emb
                                   key=self.with_pos_embed(memory, pos),  # image ft
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="gelu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class kMaxCrossAttentionLayer(nn.Module):
    """
    Kmax-Deeplab风格的k-means cross attention，输入输出严格对齐CrossAttentionLayer：
    输入：
        tgt: (L, N, C)
        memory: (S, N, C)
        memory_mask: (L, S) or None
        memory_key_padding_mask: (N, S) or None
        pos: (S, N, C) or None
        query_pos: (L, N, C) or None
    输出：
        (L, N, C)  # 与tgt同shape
        mask_logits: (N, L, H, W)  # 可选
    """
    def __init__(self, d_model, nhead, dropout=0.0, base_filters=128, bottleneck_expansion=2, key_expansion=1, value_expansion=2):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self._bottleneck_channels = int(round(base_filters * bottleneck_expansion))
        self._total_key_depth = int(round(base_filters * key_expansion))
        self._total_value_depth = int(round(base_filters * value_expansion))
        # 1x1 conv for memory (pixel) and tgt (query)
        self.pixel_proj = nn.Linear(d_model, self._bottleneck_channels)
        self.query_proj = nn.Linear(d_model, self._bottleneck_channels)
        self.pixel_value_proj = nn.Linear(self._bottleneck_channels, self._total_value_depth)
        self.query_out_proj = nn.Linear(self._total_value_depth, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        # 初始化
        nn.init.xavier_uniform_(self.pixel_proj.weight)
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.pixel_value_proj.weight)
        nn.init.xavier_uniform_(self.query_out_proj.weight)

    def forward(self, tgt, memory,
                memory_mask=None,
                memory_key_padding_mask=None,
                pos=None,
                query_pos=None,
                return_mask_logits=False):
        # tgt: (L, N, C), memory: (S, N, C)
        L, N, C = tgt.shape
        S, N2, C2 = memory.shape
        assert N == N2 and C == C2
        # 加位置编码
        q = tgt if query_pos is None else tgt + query_pos  # (L, N, C)
        k = memory if pos is None else memory + pos        # (S, N, C)
        # 变换到bottleneck
        q_proj = self.query_proj(q)    # (L, N, bottleneck)
        k_proj = self.pixel_proj(k)    # (S, N, bottleneck)
        v_proj = self.pixel_value_proj(k_proj)  # (S, N, value_depth)
        # 归一化
        k_proj_norm = F.normalize(k_proj, p=2, dim=-1)    # (S, N, bottleneck)
        # k-means assignment: mask logits
        # (N, L, S)
        mask_logits = torch.einsum('lnc,snc->nls', q_proj, k_proj_norm)
        # 聚类分配
        with torch.no_grad():
            clustering_result = mask_logits.detach()
            index = clustering_result.max(-1, keepdim=True)[1]  # (N, L, 1)
            clustering_result = torch.zeros_like(clustering_result).scatter_(-1, index, 1.0)  # (N, L, S)
        # k-means update
        with autocast('cuda', enabled=False):
            # (N, L, value_depth)
            kmeans_update = torch.einsum('nls,snd->nld', clustering_result.float(), v_proj.float())
        # (L, N, value_depth)
        kmeans_update = kmeans_update.permute(1, 0, 2)
        kmeans_update = self.query_out_proj(kmeans_update)  # (L, N, C)
        out = tgt + self.dropout(kmeans_update)
        out = self.norm(out)
        if return_mask_logits:
            # mask_logits: (N, L, S) 可reshape为 (N, L, H, W) 由上层决定
            return out, mask_logits
        else:
            return out
