# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
# Copyright (c) Medical AI Lab, Alibaba DAMO Academy

# Modified by Ke Yan for 2D adaptation

import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F
from .transformer_layers import SelfAttentionLayer, CrossAttentionLayer, FFNLayer, MLP, kMaxCrossAttentionLayer
from .position_embedding import PositionEmbeddingSine
from logger import log_info

class MultiExpertsDecoder(nn.Module):
    def __init__(
            self,
            in_channels=[128, 64, 32],
            num_classes: int = 3,
            hidden_dim: int = 64,
            num_experts: int = 6,
            num_extra_queries: int = 0,
            nheads: int = 8,
            dim_feedforward: int = 384,
            dec_layers: int = 3,
            pre_norm: bool = False,
            num_feature_levels: int = 3,
            use_pos_emb: bool = False,
            non_object: bool = True,
            mask_threshold: float = 0.5,
            mask_embed_type: str = 'sum_mask',  # 'sum_mask' or 'edl_uncertainty'
            cls_target_type: str = 'single',
            **kwargs
    ):
        super().__init__()
        unused_kwargs = kwargs
        if unused_kwargs:
            log_info(f"unused kwargs: {unused_kwargs}")

        self.use_pos_emb = use_pos_emb
        self.hidden_dim = hidden_dim
        self.non_object = non_object
        self.num_classes = num_classes
        self.in_channels = in_channels

        self.mask_threshold = mask_threshold
        assert mask_embed_type in ['sum_mask', 'edl_uncertainty']
        self.mask_embed_type = mask_embed_type

        # positional encoding
        N_steps = hidden_dim
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before= \
                        pre_norm,
                )
            )
        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_experts = num_experts
        self.num_queries = self.num_experts + num_extra_queries
        # learnable query features
        self.query_feat = nn.Embedding(self.num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(self.num_queries, hidden_dim)

        # level embedding (we always use 3 scales)
        self.num_feature_levels = num_feature_levels
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()

        # Replace the single Conv3d with two Conv2d layers in input_proj
        self.input_proj = nn.ModuleList()
        for l in range(self.num_feature_levels):
            self.input_proj.append(nn.Conv2d(in_channels[l], hidden_dim, kernel_size=1))
            weight_init.c2_xavier_fill(self.input_proj[-1])

        # output
        assert cls_target_type in ['single', 'multi']
        self.cls_target_type = cls_target_type
        # 直接输出单个类别的证据头
        self.single_target_embed = nn.Linear(hidden_dim, self.num_experts + int(non_object))
        # 用于多类别的beta分布双分支证据头
        self.multi_target_alpha_head = nn.Linear(hidden_dim, self.num_experts)
        self.multi_target_beta_head = nn.Linear(hidden_dim, self.num_experts)

        # mask output
        self.mask_embed = MLP(hidden_dim, hidden_dim, hidden_dim, 3)
        self.mask_experts_head = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):

        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.output_cls_embed(decoder_output)

        mask_embed = self.mask_embed(decoder_output)
        _mask_features = torch.einsum("bqc,bchw->bqchw", mask_embed, mask_features)
        
        # 处理 5D 张量：(b, q, c, h, w) -> (b*q, c, h, w)
        b, q, c, h, w = _mask_features.shape
        _mask_features_reshaped = _mask_features.view(b * q, c, h, w)
        # 应用 Conv2d：(b*q, c, h, w) -> (b*q, num_classes, h, w)
        outputs_mask_reshaped = self.mask_experts_head(_mask_features_reshaped)
        # 重塑回 5D：(b*q, num_classes, h, w) -> (b, q, num_classes, h, w)
        outputs_mask = outputs_mask_reshaped.view(b, q, self.num_classes, h, w)

        attn_mask = self.attn_mask_embed(outputs_mask, attn_mask_target_size)
        return outputs_class, outputs_mask, attn_mask

    def attn_mask_embed(self, output_mask, attn_mask_target_size):
        if self.mask_embed_type == 'edl_uncertainty':
            # Uncertainty of edl
            alpha = F.softplus(output_mask) + 1  # (B, Q, C, H, W)
            S = torch.sum(alpha, dim=2)       # (B, Q, H, W) 
            uncertainty = self.num_classes / S  # (B, Q, H, W)
            # 插值到目标大小
            attn_mask = F.interpolate(uncertainty, size=attn_mask_target_size, mode="bilinear", align_corners=False)  # (b, q, H', W')
            attn_mask = (attn_mask
                        .flatten(2)                    # (b, q, H'*W')
                        .unsqueeze(1)                  # (b, 1, q, H'*W')
                        .repeat(1, self.num_heads, 1, 1)  # (b, nheads, q, H'*W')
                        .flatten(0, 1) > self.mask_threshold).bool()   # (b*nheads, q, H'*W')
        elif self.mask_embed_type == 'sum_mask':
            attn_mask = output_mask.sum(dim=2)  # (b, q, h, w)
            # 2) 插值到目标大小 (H', W')
            attn_mask = F.interpolate(attn_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)  # (b, q, H', W')
            # 3) 转为 (bs * nheads, q, H'*W') 的布尔掩码
            attn_mask = (attn_mask.sigmoid()
                        .flatten(2)                    # (b, q, H'*W')
                        .unsqueeze(1)                  # (b, 1, q, H'*W')
                        .repeat(1, self.num_heads, 1, 1)  # (b, nheads, q, H'*W')
                        .flatten(0, 1) < 0.5).bool()   # (b*nheads, q, H'*W')
        attn_mask = attn_mask.detach()
        return attn_mask
    
    def output_cls_embed(self, decoder_output):
        if self.cls_target_type == 'single':
            # (b, num_queries, num_experts (+ non_object))
            return self.single_target_embed(decoder_output) 
        else:
            # (2, b, num_queries, num_experts)
            return torch.stack([self.multi_target_alpha_head(decoder_output), self.multi_target_beta_head(decoder_output)])
        
    def forward(self, x, mask_features, mask=None):
        if self.num_feature_levels > 1 and not isinstance(x, torch.Tensor):
            assert len(x) == self.num_feature_levels, "x {} num_feature_levels {} ".format(x.shape,
                                                                                           self.num_feature_levels)
        else:
            x = [x]

        src = []
        pos = []
        size_list = []

        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)  # p.e.
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        predictions_class = []
        predictions_mask = []

        # prediction heads on learnable query features
        first_attn_target_size = size_list[0]
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
            output,
            mask_features,
            attn_mask_target_size=first_attn_target_size,
        )
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)
        # meta for initial prediction (before any cross-attention)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # Cross Attention
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,
                pos=pos[level_index] if not self.use_pos_emb else None,
                query_pos=query_embed
            )
            # Self Attention
            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )
            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )
            next_attn_target_size = size_list[(i + 1) % self.num_feature_levels]
            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
                output,
                mask_features,
                attn_mask_target_size=next_attn_target_size,
            )
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
            # print(f"outputs_mask.shape:{outputs_mask.shape}")


        assert len(predictions_class) == self.num_layers + 1
        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class, predictions_mask
            )
        }

        return out

    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        outs = []
        for idx, (a, b) in enumerate(zip(outputs_class[:-1], outputs_seg_masks[:-1])):
            d = {"pred_logits": a, "pred_masks": b}
            outs.append(d)
        return outs
