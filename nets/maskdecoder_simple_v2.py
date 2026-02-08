import torch
from torch import nn
from .transformer_layers import SelfAttentionLayer, CrossAttentionLayer, FFNLayer, MLP
from .position_embedding import PositionEmbeddingSine

class FactorizedQueryGenerator(nn.Module):
    """ Q_expert + Q_class """
    def __init__(self, num_experts, num_classes, hidden_dim):
        super().__init__()
        self.expert_embed = nn.Embedding(num_experts, hidden_dim)
        self.class_embed = nn.Embedding(num_classes, hidden_dim)

    def forward(self):
        # [N, 1, D] + [1, K, D] -> [N, K, D]
        q_final = self.expert_embed.weight.unsqueeze(1) + self.class_embed.weight.unsqueeze(0)
        return q_final.flatten(0, 1)

class CascadedDualSourceLayer(nn.Module):
    """单层 Transformer：可选 Bridge 交互 + Pixel 交互 + FFN。"""
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1, use_bridge=True):
        super().__init__()
        self.use_bridge = use_bridge
        self.self_attn = SelfAttentionLayer(d_model, nhead, dropout=dropout)
        if use_bridge:
            self.cross_attn_semantic = CrossAttentionLayer(d_model, nhead, dropout=dropout)
            self.norm_sem = nn.LayerNorm(d_model)
        self.cross_attn_spatial = CrossAttentionLayer(d_model, nhead, dropout=dropout)
        self.ffn = FFNLayer(d_model, dim_feedforward, dropout=dropout)

    def forward(self, tgt, query_pos, pixel_src=None, pixel_pos=None, bridge_src=None, bridge_pos=None):
        # 1) Semantic cross-attention (only when use_bridge=True and source provided)
        if self.use_bridge and bridge_src is not None:
            tgt_sem = self.cross_attn_semantic(tgt, memory=bridge_src, pos=bridge_pos, query_pos=query_pos)
            tgt = self.norm_sem(tgt + tgt_sem)
        # 2) Spatial cross-attention (always when pixel_src provided)
        if pixel_src is not None:
            tgt = self.cross_attn_spatial(tgt, memory=pixel_src, pos=pixel_pos, query_pos=query_pos, memory_mask=None)
        # 3) Self-attention
        tgt = self.self_attn(tgt, query_pos=query_pos)
        # 4) FFN
        tgt = self.ffn(tgt)
        return tgt

class DualSourceTransformerDecoder(nn.Module):
    def __init__(
        self,
        in_channels_pixel_list,     # List of Pixel Feature Channels
        in_channels_bridge,    # Bridge Feature Channel (1536)
        hidden_dim=64,
        num_experts=4,
        num_classes=2,
        nheads=4,
        dim_feedforward=256,
        use_bridge: bool = False,
        bridge_layers_indices: list = [0], # Layers that interact with bridge
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_bridge = use_bridge
        self.bridge_layers_indices = bridge_layers_indices if bridge_layers_indices is not None else []
        
        # --- Projections ---
        # 支持多尺度特征投影：为每个尺度创建一个 1x1 卷积
        self.pixel_projs = nn.ModuleList([
            nn.Conv2d(in_ch, hidden_dim, kernel_size=1) 
            for in_ch in in_channels_pixel_list
        ])
        
        # 对 Bridge 特征进行重卷积，使其适合 Transformer 交互
        if self.use_bridge:
            self.bridge_proj = nn.Sequential(
                nn.Conv2d(in_channels_bridge, hidden_dim, kernel_size=1),
                nn.GroupNorm(32, hidden_dim),
                nn.ReLU()
            )
        else:
            self.bridge_proj = None
        
        # --- Components ---
        self.query_generator = FactorizedQueryGenerator(num_experts, num_classes, hidden_dim)
        self.pe_layer = PositionEmbeddingSine(hidden_dim, normalize=True)
        
        # --- Cascaded Layers ---
        # Always equal to number of pixel scales
        self.num_layers = len(in_channels_pixel_list)
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            # Check if this layer should interact with bridge
            use_bridge_layer = self.use_bridge and (i in self.bridge_layers_indices)
            self.layers.append(CascadedDualSourceLayer(hidden_dim, nheads, dim_feedforward, use_bridge=use_bridge_layer))
        
        # --- Heads ---
        # Mask Predictor (Evidence Head) 输出维度与 hidden_dim 同步
        self.mask_embed = MLP(hidden_dim, hidden_dim, hidden_dim, 3)
        self.nheads = nheads  # Store nheads

    def forward(self, pixel_features, bridge_feature, mask_features):
        """
        pixel_features: List of tensors [B, Ci, Hi, Wi] (Multi-scale)
        bridge_feature: [B, C_bri, H_bri, W_bri] (Low Res, Aligned)
        mask_features: [B, D, H, W] (High Res for final mask generation)
        """
        B = pixel_features[0].shape[0]
        
        # 1. Prepare Sources
        # Semantic Source (Bridge)
        if self.use_bridge and bridge_feature is not None:
            b_feat = self.bridge_proj(bridge_feature)
            b_pos = self.pe_layer(b_feat)
            b_src = b_feat.flatten(2).permute(2, 0, 1) # [HW, B, D]
            b_pos = b_pos.flatten(2).permute(2, 0, 1)
        else:
            b_src = None
            b_pos = None
        
        # 2. Prepare Query
        # [N*K, B, D]
        query_embed = self.query_generator().unsqueeze(1).repeat(1, B, 1)
        tgt = torch.zeros_like(query_embed)
        
        # 3. Cascaded Loop
        predictions_mask = []
        # layer assignment: Iterate through defined layers (aligned with pixel scales)
        for i, layer in enumerate(self.layers):
            # A. Prepare Pixel Features (Always present for this layer index)
            src_feat = pixel_features[i]
            p_feat = self.pixel_projs[i](src_feat)
            p_pos = self.pe_layer(p_feat)
            p_src = p_feat.flatten(2).permute(2, 0, 1)
            p_pos = p_pos.flatten(2).permute(2, 0, 1)

            # B. Check for Bridge Interaction
            layer_use_bridge = self.use_bridge and (i in self.bridge_layers_indices)

            tgt = layer(
                tgt=tgt,
                query_pos=query_embed,
                pixel_src=p_src,
                pixel_pos=p_pos,
                bridge_src=b_src if layer_use_bridge else None,
                bridge_pos=b_pos if layer_use_bridge else None,
            )
            
            # --- Generate Predictions for this layer (Deep Supervision) ---
            tgt_transposed = tgt.transpose(0, 1)

            # Mask Embedding: [B, N*K, mask_dim]
            curr_mask_embed = self.mask_embed(tgt_transposed)
            
            # Mask Logits: Generated at HIGH RESOLUTION using mask_features
            # [B, N*K, mask_dim] @ [B, mask_dim, H_high, W_high] -> [B, N*K, H_high, W_high]
            mask_logits = torch.einsum("bqc,bchw->bqhw", curr_mask_embed, mask_features)
            predictions_mask.append(mask_logits)
            
        # Return all intermediate outputs
        out = {
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(predictions_mask)
        }
        return out

    def _set_aux_loss(self, outputs_seg_masks):
        outs = []
        for b in outputs_seg_masks[:-1]:
            outs.append({"pred_masks": b})
        return outs