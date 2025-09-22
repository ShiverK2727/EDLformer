# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
# Copyright (c) Medical AI Lab, Alibaba DAMO Academy

# Modified by Ke Yan for 2D adaptation

import torch
from torch import nn
from torch.nn import functional as F
from .smp_unet_backbone import SMPUnetEncoder, CustomUnetDecoder
from .edl_multiexperts_decoder import MultiExpertsDecoder
from logger import log_info
from typing import List, Optional


class EDLMaskFormer(nn.Module):
    """
    Complete EDL MaskFormer with integrated SMPUnetEncoder and CustomUnetDecoder.
    This class replaces the part_factory pattern with direct instantiation.
    """
    def __init__(
            self,
            in_channels: int = 3,
            num_classes: int = 3,
            num_experts: int = 6,
            num_extra_queries: int = 0,
            # Encoder configuration
            backbone_encoder_name: str = "resnet34",
            backbone_encoder_weights: str = "imagenet", 
            # Decoder configuration
            backbone_decoder_channels: List[int] = [256, 128, 64, 32, 16],
            backbone_use_batchnorm: bool = True,
            backbone_upsample_mode: str = 'interp',
            # Mask features configuration
            consensus_uncertainty_threshold: float = 0.5,
            mask_dim: int = 64,
            mask_pixel_idxs: Optional[List[int]] = None,
            # Predictor configuration
            decoder_nheads: int = 8,
            decoder_dim_feedforward: int = 384,
            decoder_dec_layers: int = 3,
            decoder_pre_norm: bool = False,
            decoder_num_feature_levels: int = 3,
            decoder_use_pos_emb: bool = False,
            decoder_non_object: bool = True,
            decoder_mask_threshold: float = 0.5,
            decoder_mask_embed_type: str = 'sum_mask',
            decoder_cls_target_type: str = 'single',
            **kwargs
    ):
        super().__init__()

        self.num_classes = num_classes
        self.consensus_uncertainty_threshold = consensus_uncertainty_threshold
        self.decoder_cls_target_type = decoder_cls_target_type

        if mask_pixel_idxs is None:
            mask_pixel_idxs = [-4, -3, -2]  # Default from config
        
        self.mask_pixel_idxs = mask_pixel_idxs
        
        # Initialize encoder
        self.pixel_encoder = SMPUnetEncoder(
            encoder_name=backbone_encoder_name,
            encoder_weights=backbone_encoder_weights,
            in_channels=in_channels
        )
        
        # Initialize decoder
        encoder_channels = self.pixel_encoder.encoder.out_channels
        self.pixel_decoder = CustomUnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=backbone_decoder_channels,
            use_batchnorm=backbone_use_batchnorm,
            upsample_mode=backbone_upsample_mode
        )

        # Calculate input channels for predictor from selected decoder features
        predictor_in_channels = [backbone_decoder_channels[i] for i in mask_pixel_idxs]
        
        # Segmentation head
        self.segmentation_head = nn.Conv2d(
            backbone_decoder_channels[-1], num_classes, kernel_size=1, stride=1, padding=0
        )
        self.deep_supervision_heads = nn.ModuleList([
            nn.Conv2d(in_ch, num_classes, kernel_size=1, stride=1, padding=0)
            for in_ch in predictor_in_channels
        ])

        # Linear projection for latent mask features
        self.linear_mask_features = nn.Conv2d(
            backbone_decoder_channels[-1], mask_dim, 
            kernel_size=1, stride=1, padding=0
        )
        
        # Initialize predictor
        # 过滤掉不属于MultiExpertsDecoder的参数
        predictor_kwargs = {}
        predictor_allowed_params = {
            'mask_dim', 'mask_pixel_idxs', 'consensus_uncertainty_threshold'
        }
        for k, v in kwargs.items():
            if k in predictor_allowed_params:
                predictor_kwargs[k] = v
        
        self.predictor = MultiExpertsDecoder(
            in_channels=predictor_in_channels,
            num_classes=num_classes,
            hidden_dim=mask_dim,
            num_experts=num_experts,
            num_extra_queries=num_extra_queries,
            nheads=decoder_nheads,
            dim_feedforward=decoder_dim_feedforward,
            dec_layers=decoder_dec_layers,
            pre_norm=decoder_pre_norm,
            num_feature_levels=decoder_num_feature_levels,
            use_pos_emb=decoder_use_pos_emb,
            non_object=decoder_non_object,
            mask_threshold=decoder_mask_threshold,
            mask_embed_type=decoder_mask_embed_type,
            cls_target_type=decoder_cls_target_type,
            **predictor_kwargs
        )
        
        log_info(f"EDLMaskFormer initialized with:")
        log_info(f"  Encoder: {backbone_encoder_name} ({backbone_encoder_weights})")
        log_info(f"  Encoder channels: {encoder_channels}")
        log_info(f"  Decoder channels: {backbone_decoder_channels}")
        log_info(f"  Predictor input channels: {predictor_in_channels}")
        log_info(f"  Mask pixel indices: {mask_pixel_idxs}")
        log_info(f"  Num experts: {num_experts}, Num classes: {num_classes}")
        log_info(f"  Classification target type: {decoder_cls_target_type}")

    def forward(self, input_image):
        """
        Forward pass through the complete EDL MaskFormer.
        
        现在直接返回网络输出logits作为证据，EDL处理移到loss计算中进行。
        这样做的好处：
        1. 保持梯度流的完整性
        2. 在loss中灵活控制EDL参数
        3. 减少模型前向传播的计算量
        
        Args:
            input_image: Input tensor of shape (B, C, H, W)
            
        Returns:
            predictions: Dictionary containing raw logits (evidence) and EDL config
        """
        # Encoder forward pass
        en_feats = self.pixel_encoder(input_image)
        assert isinstance(en_feats, List), "pixel encoder must return a list"
        
        # Decoder forward pass
        de_feats = self.pixel_decoder(en_feats)
        assert isinstance(de_feats, List), "pixel decoder must return a list"
        
        # Select features for predictor input
        mask_pixel_feats = [de_feats[i] for i in self.mask_pixel_idxs]
        
        # Generate mask features from the last decoder output
        mask_features = self.linear_mask_features(de_feats[-1])
        
        # Predictor forward pass
        predictions = self.predictor(mask_pixel_feats, mask_features, mask=None)

        # 1) Segmentation head on final decoder feature (B, num_classes, H, W)
        pix_pred_mask = self.segmentation_head(de_feats[-1])  # (B, num_classes, H, W)

        # 2) Deep supervision heads for selected decoder features
        aux_pix_pred_masks = []
        for idx, (head, mask_idx) in enumerate(zip(self.deep_supervision_heads, self.mask_pixel_idxs)):
            feat = de_feats[mask_idx]  # (B, C_in, H_i, W_i)
            ds_mask = head(feat)  # (B, num_classes, H_i, W_i)
            predictions['aux_outputs'][idx]['pix_pred_masks'] = ds_mask
            aux_pix_pred_masks.append(ds_mask)

        # Merge pixel-level outputs into predictor outputs dictionary
        predictions['pix_pred_masks'] = pix_pred_mask
        
        return predictions


