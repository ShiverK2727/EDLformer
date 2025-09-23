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


class MultiExpertsDecoderMulti(nn.Module):
    def __init__(
            self,
            in_channels=[128, 64, 32],
            num_seg_classes: int = 2,
            hidden_dim: int = 64,
            num_queries: int = 20,
            num_cls_classes: int = 12,
            nheads: int = 8,
            dim_feedforward: int = 384,
            dec_layers: int = 3,
            pre_norm: bool = False,
            num_feature_levels: int = 3,
            use_pos_emb: bool = False,
            non_object: bool = True,
            mask_threshold: float = 0.5,
            **kwargs
    ):
        super().__init__()
        unused_kwargs = kwargs
        if unused_kwargs:
            log_info(f"unused kwargs: {unused_kwargs}")

        self.use_pos_emb = use_pos_emb
        self.hidden_dim = hidden_dim
        self.non_object = non_object
        self.num_cls_classes = num_cls_classes
        self.num_seg_classes = num_seg_classes
        self.num_experts = num_queries  # 专家数量等于查询数量
        self.in_channels = in_channels

        self.mask_threshold = mask_threshold

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

        self.num_queries = num_queries
        
        # learnable query features
        self.query_feat = nn.Embedding(self.num_queries, hidden_dim)
        # learnable query p.e. 
        self.query_embed = nn.Embedding(self.num_queries, hidden_dim)

        # level embedding (we always use 3 scales)
        self.num_feature_levels = num_feature_levels
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()

        self.input_proj = nn.ModuleList()
        for l in range(self.num_feature_levels):
            self.input_proj.append(nn.Conv2d(in_channels[l], hidden_dim, kernel_size=1))
            weight_init.c2_xavier_fill(self.input_proj[-1])

        self.cls_embed = nn.Linear(hidden_dim, self.num_cls_classes)
        # mask output
        self.mask_embed = MLP(hidden_dim, hidden_dim, hidden_dim, 3)
        self.mask_experts_head = nn.Conv2d(hidden_dim, self.num_seg_classes, kernel_size=1)

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
        outputs_mask = outputs_mask_reshaped.view(b, q, self.num_seg_classes, h, w)
        _outputs_mask = outputs_mask.sum(dim=2)

        attn_mask = self.attn_mask_embed(_outputs_mask, attn_mask_target_size)

        return outputs_class, outputs_mask, attn_mask

    def attn_mask_embed(self, output_mask, attn_mask_target_size):
        attn_mask = output_mask
        # --- MODIFICATION END ---
        
        # 2) 插值到目标大小 (H', W')
        attn_mask = F.interpolate(attn_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)  # (b, q, H', W')
        # 3) 转为 (bs * nheads, q, H'*W') 的布尔掩码
        attn_mask = (attn_mask.sigmoid()
                                .flatten(2)                      # (b, q, H'*W')
                                .unsqueeze(1)                    # (b, 1, q, H'*W')
                                .repeat(1, self.num_heads, 1, 1)     # (b, nheads, q, H'*W')
                                .flatten(0, 1) < 0.5).bool()   # (b*nheads, q, H'*W')
        attn_mask = attn_mask.detach()
        return attn_mask
    
    def output_cls_embed(self, decoder_output):
        return self.cls_embed(decoder_output)
        
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



class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class MultiExpertsDecoderMultiV2(MultiExpertsDecoderMulti):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 重新定义 mask_experts_head 为 3x3 卷积 + SEBlock
        # 首先移除原来的 mask_experts_head
        delattr(self, 'mask_experts_head')
        
        # 创建新的 mask_experts_head：3x3 conv + SEBlock + 1x1 conv to final classes
        self.mask_experts_se = SEBlock(self.hidden_dim)
        self.mask_experts_conv3x3 = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1, groups=self.hidden_dim)  # Depthwise convolution
        self.mask_experts_final = nn.Conv2d(self.hidden_dim, self.num_seg_classes, kernel_size=1)
        
        # 初始化权重
        weight_init.c2_xavier_fill(self.mask_experts_conv3x3)
        weight_init.c2_xavier_fill(self.mask_experts_final)

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.output_cls_embed(decoder_output)

        mask_embed = self.mask_embed(decoder_output)
        _mask_features = torch.einsum("bqc,bchw->bqchw", mask_embed, mask_features)

        # 处理 5D 张量：(b, q, c, h, w) -> (b*q, c, h, w)
        b, q, c, h, w = _mask_features.shape
        _mask_features_reshaped = _mask_features.view(b * q, c, h, w)
        
        # 应用新的 mask_experts_head：3x3 conv + SEBlock + 1x1 conv
        x = self.mask_experts_conv3x3(_mask_features_reshaped)
        x = F.relu(x, inplace=True)  # 添加激活函数
        x = self.mask_experts_se(x)
        outputs_mask_reshaped = self.mask_experts_final(x)
        
        # 重塑回 5D：(b*q, num_classes, h, w) -> (b, q, num_classes, h, w)
        outputs_mask = outputs_mask_reshaped.view(b, q, self.num_seg_classes, h, w)
        _outputs_mask = outputs_mask.sum(dim=2)

        attn_mask = self.attn_mask_embed(_outputs_mask, attn_mask_target_size)

        return outputs_class, outputs_mask, attn_mask
    