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


class MultiExpertsDecoderSingle(nn.Module):
    def __init__(
            self,
            in_channels=[128, 64, 32],
            num_classes: int = 2,
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
            force_matching: bool = False,  # 强制匹配模式：queries直接对应专家
            expert_classification: bool = False,  # 新增：专家分类模式开关
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
        self.force_matching = force_matching  # 新增：强制匹配模式
        self.expert_classification = expert_classification  # 新增：专家分类模式

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

        self.num_experts = num_experts
        self.num_queries = self.num_experts + num_extra_queries
        
        # 强制匹配模式下，queries数量必须匹配专家标注数量
        if self.force_matching:
            if self.expert_classification:
                # 专家分类模式：需要2N个queries对应2N个专家标注
                required_queries = self.num_experts * 2  # 2N
                if self.num_queries != required_queries:
                    log_info(f"Warning: force_matching + expert_classification but num_queries({self.num_queries}) != 2*num_experts({required_queries})")
                    log_info(f"Setting num_queries = 2*num_experts = {required_queries}")
                    self.num_queries = required_queries
            else:
                # Mask分类模式：理论上不应该使用固定匹配，但保持兼容性
                log_info(f"Warning: force_matching=True with mask classification is not recommended")
                if self.num_queries != self.num_experts:
                    log_info(f"Setting num_queries = num_experts = {self.num_experts}")
                    self.num_queries = self.num_experts
            
            # 强制匹配模式下不需要non_object
            if self.non_object:
                log_info(f"Warning: force_matching=True, setting non_object=False")
                self.non_object = False
        
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

        # 根据匹配模式调整输出头
        if self.force_matching:
            # 强制匹配模式：根据expert_classification决定输出类型
            if self.expert_classification:
                # 专家分类模式：输出2N个专家标签 (N个disc + N个cup)
                expert_output_dim = self.num_experts * 2  # 2N个专家标签
                self.class_embed = nn.Linear(hidden_dim, expert_output_dim)
                log_info(f"Expert classification mode: outputting {expert_output_dim} expert labels (2N expert annotations)")
            else:
                # Mask类别分类模式：输出mask类别 (如disc, cup)
                self.class_embed = nn.Linear(hidden_dim, self.num_classes)
                log_info(f"Mask classification mode: outputting {self.num_classes} mask classes")
        else:
            # 匈牙利匹配模式：原有的single_target_embed
            if self.expert_classification:
                # 匈牙利匹配 + 专家分类：输出2N个专家标签 + 背景类
                expert_output_dim = self.num_experts * 2  # 2N个专家标签
                self.single_target_embed = nn.Linear(hidden_dim, expert_output_dim + int(self.non_object))
                log_info(f"Hungarian + Expert classification mode: outputting {expert_output_dim} expert labels + {int(self.non_object)} background")
            else:
                # 匈牙利匹配 + mask分类：输出mask类别 + 背景类
                self.single_target_embed = nn.Linear(hidden_dim, self.num_classes + int(self.non_object))
                log_info(f"Hungarian + Mask classification mode: outputting {self.num_classes} mask classes + {int(self.non_object)} background")
        
        # mask output
        self.mask_embed = MLP(hidden_dim, hidden_dim, hidden_dim, 3)
        
        # 最终参数汇总日志
        log_info(f"MultiExpertsDecoderSingle configuration summary:")
        log_info(f"  Number of experts: {self.num_experts}")
        log_info(f"  Number of queries: {self.num_queries}")
        log_info(f"  Force matching: {self.force_matching}")
        log_info(f"  Expert classification: {self.expert_classification}")
        log_info(f"  Non-object enabled: {self.non_object}")
        if hasattr(self, 'class_embed'):
            log_info(f"  Classification output dimension: {self.class_embed.out_features}")
        elif hasattr(self, 'single_target_embed'):
            log_info(f"  Classification output dimension: {self.single_target_embed.out_features}")
        log_info(f"  Input channels: {self.in_channels}")
        log_info(f"  Hidden dimension: {self.hidden_dim}")


    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.output_cls_embed(decoder_output)

        mask_embed = self.mask_embed(decoder_output)

        outputs_mask = torch.einsum('bqc,bchw->bqhw', mask_embed, mask_features)

        attn_mask = self.attn_mask_embed(outputs_mask, attn_mask_target_size)
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
        if self.force_matching:
            # 强制匹配模式：根据expert_classification决定输出含义
            if self.expert_classification:
                # 专家分类模式：每个query输出专家ID概率 (b, num_queries, num_experts)
                return self.class_embed(decoder_output)
            else:
                # Mask类别分类模式：每个query输出mask类别概率 (b, num_queries, num_classes)
                return self.class_embed(decoder_output)
        else:
            # 匈牙利匹配模式：保持原有逻辑
            if hasattr(self, 'single_target_embed'):
                return self.single_target_embed(decoder_output)
            else:
                # 兼容性处理
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
