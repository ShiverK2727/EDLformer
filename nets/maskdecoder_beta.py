# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
# Copyright (c) Medical AI Lab, Alibaba DAMO Academy

# Modified by Ke Yan for 2D adaptation
# Further modified based on user request for flexible EDL attention and classification heads

import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F
from .transformer_layers import SelfAttentionLayer, CrossAttentionLayer, FFNLayer, MLP, kMaxCrossAttentionLayer
from .position_embedding import PositionEmbeddingSine
from logger import log_info


class MultiExpertsDecodeEDL(nn.Module):
    def __init__(
            self,
            in_channels=[128, 64, 32],
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
            # --- 注意力掩码配置 ---
            attn_mask_type: str = 'mask',           # 'mask', 'uncertainty', 或 'uncertainty_ratio'
            mask_threshold: float = 0.5,            # 'mask' 模式下的前景阈值
            uncertainty_threshold: float = 0.5,     # 'uncertainty' 模式下的固定不确定性阈值
            uncertainty_focus_ratio: float = 0.2,   # 新增: 'uncertainty_ratio' 模式下的关注百分比
            # --- 分类头配置 ---
            cls_type: str = 'single',               # 'single' (单标签) 或 'multi' (多标签)
            **kwargs
    ):
        super().__init__()
        unused_kwargs = kwargs
        if unused_kwargs:
            log_info(f"unused kwargs: {unused_kwargs}")

        self.use_pos_emb = use_pos_emb
        self.hidden_dim = hidden_dim
        self.non_object = non_object
        self.in_channels = in_channels
        self.num_queries = num_queries
        self.num_cls_classes = num_cls_classes

        # --- 保存配置参数 ---
        self.mask_threshold = mask_threshold
        self.uncertainty_threshold = uncertainty_threshold
        self.uncertainty_focus_ratio = uncertainty_focus_ratio # 新增
        self.cls_type = cls_type
        self.attn_mask_type = attn_mask_type
        # 修改: assert断言中加入新的不确定性模式
        assert self.attn_mask_type in ['mask', 'uncertainty_threshold', 'uncertainty_ratio']
        assert self.cls_type in ['single', 'multi']

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
                    d_model=hidden_dim, nhead=nheads, dropout=0.0, normalize_before=pre_norm
                )
            )
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim, nhead=nheads, dropout=0.0, normalize_before=pre_norm
                )
            )
            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim, dim_feedforward=dim_feedforward, dropout=0.0, normalize_before=pre_norm
                )
            )
        self.decoder_norm = nn.LayerNorm(hidden_dim)

        # learnable query features
        self.query_feat = nn.Embedding(self.num_queries, hidden_dim)
        self.query_embed = nn.Embedding(self.num_queries, hidden_dim)

        # level embedding
        self.num_feature_levels = num_feature_levels
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        
        self.input_proj = nn.ModuleList()
        for l in range(self.num_feature_levels):
            self.input_proj.append(nn.Conv2d(in_channels[l], hidden_dim, kernel_size=1))
            weight_init.c2_xavier_fill(self.input_proj[-1])

        # 分类头: 两个头分别用于alpha和beta，根据cls_type决定如何使用
        # 分类头输出维度 = num_cls_classes + (1 if non_object else 0)
        cls_output_dim = self.num_cls_classes + (1 if self.non_object else 0)
        self.alpha_cls_embed = nn.Linear(hidden_dim, cls_output_dim)
        self.beta_cls_embed = nn.Linear(hidden_dim, cls_output_dim)

        # 分割头: 两个MLP头分别用于alpha和beta
        self.alpha_mask_embed = MLP(hidden_dim, hidden_dim, hidden_dim, 3)
        self.beta_mask_embed = MLP(hidden_dim, hidden_dim, hidden_dim, 3)

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        
        # 根据cls_type决定分类头的输出
        alpha_cls, beta_cls = self.output_cls_embed(decoder_output)

        alpha_mask_embed = self.alpha_mask_embed(decoder_output)
        beta_mask_embed = self.beta_mask_embed(decoder_output)

        alpha_mask = torch.einsum('bqc,bchw->bqhw', alpha_mask_embed, mask_features)
        beta_mask = torch.einsum('bqc,bchw->bqhw', beta_mask_embed, mask_features)
        
        # 根据attn_mask_type决定注意力掩码的生成方式
        attn_mask = self.generate_attn_mask(alpha_mask, beta_mask, attn_mask_target_size)
        
        return [alpha_cls, beta_cls], [alpha_mask, beta_mask], attn_mask

    def generate_attn_mask(self, alpha_mask_logits, beta_mask_logits, attn_mask_target_size):
        # 核心修改: 将attn_mask的生成逻辑封装到此方法中
        alpha = F.softplus(alpha_mask_logits) + 1
        beta = F.softplus(beta_mask_logits) + 1

        if self.attn_mask_type == 'mask':
            pred_prob = alpha / (alpha + beta)
            attn_mask = F.interpolate(pred_prob, size=attn_mask_target_size, mode="bilinear", align_corners=False)
            attn_mask = attn_mask < self.mask_threshold # 直接比较概率值

        elif self.attn_mask_type == 'uncertainty_threshold':
            total_evidence = alpha + beta
            uncertainty = 2.0 / (total_evidence + 1e-6)
            attn_mask = F.interpolate(uncertainty, size=attn_mask_target_size, mode="bilinear", align_corners=False)
            attn_mask = attn_mask < self.uncertainty_threshold # 不确定性低的区域被mask

        elif self.attn_mask_type == 'uncertainty_ratio':
            # 新增: 基于不确定性百分比的动态阈值方法
            total_evidence = alpha + beta
            uncertainty = 2.0 / (total_evidence + 1e-6)
            attn_mask = F.interpolate(uncertainty, size=attn_mask_target_size, mode="bilinear", align_corners=False)
            
            b, q, h, w = attn_mask.shape
            flat_attn_mask = attn_mask.flatten(2) # 展平为 [b, q, h*w]
            
            # 要保留的像素数量 (不确定性最高的20%)
            num_pixels_to_keep = int(self.uncertainty_focus_ratio * h * w)
            if num_pixels_to_keep == 0: num_pixels_to_keep = 1
            if num_pixels_to_keep > h * w: num_pixels_to_keep = h * w
            
            # 找到作为分界点的阈值
            # torch.kthvalue找到第k小的元素，所以我们需要找第(总数-保留数)小的元素
            threshold_vals, _ = torch.kthvalue(
                flat_attn_mask, 
                k=h * w - num_pixels_to_keep, 
                dim=-1, 
                keepdim=True
            )
            
            # 不确定性小于动态阈值的像素将被忽略
            attn_mask = flat_attn_mask < threshold_vals
        
        else:
            raise ValueError(f"Unknown attn_mask_type: {self.attn_mask_type}")

        # --- 通用的掩码格式化步骤 ---
        # 调整形状以匹配多头注意力的输入要求
        attn_mask = (attn_mask.detach()                # (b, q, H'*W')
                     .unsqueeze(1)                     # (b, 1, q, H'*W')
                     .repeat(1, self.num_heads, 1, 1)   # (b, nheads, q, H'*W')
                     .flatten(0, 1))                   # (b*nheads, q, H'*W')
        return attn_mask.bool()
    
    def output_cls_embed(self, decoder_output):
        # 核心修改: 根据cls_type决定返回alpha, beta还是只有alpha
        if self.cls_type == 'single':
            # 单标签模式: 只返回alpha头，它将被解释为狄利克雷分布的证据向量
            return self.alpha_cls_embed(decoder_output), None
        elif self.cls_type == 'multi':
            # 多标签模式: 返回alpha和beta头，用于构建独立的Beta分布
            return self.alpha_cls_embed(decoder_output), self.beta_cls_embed(decoder_output)
        else:
            raise ValueError(f"Unknown cls_type: {self.cls_type}")
            
    def forward(self, x, mask_features, mask=None):
        if self.num_feature_levels > 1 and not isinstance(x, torch.Tensor):
            assert len(x) == self.num_feature_levels, f"x length {len(x)} num_feature_levels {self.num_feature_levels}"
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

        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        predictions_class = []
        predictions_mask = []

        first_attn_target_size = size_list[0]
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
            output, mask_features, attn_mask_target_size=first_attn_target_size
        )
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,
                pos=pos[level_index] if not self.use_pos_emb else None,
                query_pos=query_embed
            )
            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )
            output = self.transformer_ffn_layers[i](output)
            
            next_attn_target_size = size_list[(i + 1) % self.num_feature_levels]
            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
                output, mask_features, attn_mask_target_size=next_attn_target_size
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
        return [{"pred_logits": a, "pred_masks": b} 
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])]