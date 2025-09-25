# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
# Copyright (c) Medical AI Lab, Alibaba DAMO Academy

# Modified by Ke Yan for 2D adaptation
# Integrated with a flexible, configurable Evidential Deep Learning (EDL) decoder
# Removed direct pixel-wise segmentation heads to create a pure Transformer-based predictor architecture.

import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Optional

# 假设这些模块位于同一目录下或已正确安装
from .smp_unet_backbone import SMPUnetEncoder, CustomUnetDecoder
from .maskdecoder_beta import MultiExpertsDecodeEDL # 导入我们强大的新版解码头
from logger import log_info


class FlexibleEDLMaskFormer(nn.Module):
    """
    一个灵活的、基于证据学习的MaskFormer变体。
    该模型集成了可配置的MultiExpertsDecodeEDL解码器，允许在初始化时
    轻松切换分类模式（单标签/多标签）和注意力掩码生成策略。
    此版本移除了辅助的像素级分割头，完全依赖Transformer进行预测。
    """
    def __init__(
            self,
            in_channels: int = 3,
            # --- 分类与查询配置 ---
            num_cls_classes: int = 12,      # 分类任务的类别总数
            num_queries: int = 20,          # Transformer解码器的查询数量
            # --- 编码器配置 ---
            backbone_encoder_name: str = "resnet34",
            backbone_encoder_weights: str = "imagenet",
            # --- 解码器(像素级别)配置 ---
            backbone_decoder_channels: List[int] = [256, 128, 64, 32, 16],
            backbone_use_batchnorm: bool = True,
            backbone_upsample_mode: str = 'interp',
            # --- 掩码特征配置 ---
            mask_dim: int = 64,
            mask_pixel_idxs: Optional[List[int]] = None,
            # --- 预测头(Transformer)配置 ---
            predictor_nheads: int = 8,
            predictor_dim_feedforward: int = 384,
            predictor_dec_layers: int = 3,
            predictor_pre_norm: bool = False,
            predictor_use_pos_emb: bool = False,
            predictor_non_object: bool = True,
            # --- 新增的灵活配置 ---
            predictor_cls_type: str = 'multi',           # 'single' 或 'multi'
            predictor_attn_mask_type: str = 'uncertainty_ratio', # 'mask', 'uncertainty', 'uncertainty_ratio'
            predictor_mask_threshold: float = 0.5,
            predictor_uncertainty_threshold: float = 0.5,
            predictor_uncertainty_focus_ratio: float = 0.2,
    ):
        """
        初始化 FlexibleEDLMaskFormer 模型。
        
        Args:
            num_cls_classes (int): 分类头的类别数。
            num_queries (int): 查询数量, 对应模型能同时检测的最大实例数。
            predictor_cls_type (str): 分类头模式。 'single' (单标签互斥) 或 'multi' (多标签独立)。
            predictor_attn_mask_type (str): 注意力掩码生成策略。
                'mask': 基于前景概率。
                'uncertainty': 基于固定的不确定性阈值。
                'uncertainty_ratio': 基于动态的不确定性百分比。
            ... 其他参数 ...
        """
        super().__init__()

        self.num_cls_classes = num_cls_classes
        
        if mask_pixel_idxs is None:
            mask_pixel_idxs = [-4, -3, -2]  # 默认使用U-Net解码器最后三层的特征
        
        self.mask_pixel_idxs = mask_pixel_idxs
        
        # 1. 初始化像素编码器 (Pixel Encoder - Backbone)
        self.pixel_encoder = SMPUnetEncoder(
            encoder_name=backbone_encoder_name,
            encoder_weights=backbone_encoder_weights,
            in_channels=in_channels
        )
        
        # 2. 初始化像素解码器 (Pixel Decoder - U-Net Neck)
        encoder_channels = self.pixel_encoder.encoder.out_channels
        self.pixel_decoder = CustomUnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=backbone_decoder_channels,
            use_batchnorm=backbone_use_batchnorm,
            upsample_mode=backbone_upsample_mode
        )

        # 从像素解码器中选择哪些层级的特征送入Transformer
        predictor_in_channels = [backbone_decoder_channels[i] for i in self.mask_pixel_idxs]

        # 4. 用于生成掩码特征的线性投射层
        self.linear_mask_features = nn.Conv2d(
            backbone_decoder_channels[-1], mask_dim, kernel_size=1
        )
        
        # 5. 初始化我们强大的、可配置的Transformer预测头
        self.predictor = MultiExpertsDecodeEDL(
            in_channels=predictor_in_channels,
            hidden_dim=mask_dim,
            num_queries=num_queries,
            num_cls_classes=self.num_cls_classes,
            nheads=predictor_nheads,
            dim_feedforward=predictor_dim_feedforward,
            dec_layers=predictor_dec_layers,
            pre_norm=predictor_pre_norm,
            num_feature_levels=len(predictor_in_channels),
            use_pos_emb=predictor_use_pos_emb,
            non_object=predictor_non_object,
            # 将配置参数直接传入
            cls_type=predictor_cls_type,
            attn_mask_type=predictor_attn_mask_type,
            mask_threshold=predictor_mask_threshold,
            uncertainty_threshold=predictor_uncertainty_threshold,
            uncertainty_focus_ratio=predictor_uncertainty_focus_ratio,
        )
        
        log_info(f"--- FlexibleEDLMaskFormer Initialized (Pure Predictor Version) ---")
        log_info(f"  Encoder: {backbone_encoder_name} ({backbone_encoder_weights})")
        log_info(f"  Predictor Queries: {num_queries}, Classification Classes: {num_cls_classes}")
        log_info(f"  Classification Head Mode (cls_type): '{predictor_cls_type}'")
        log_info(f"  Attention Mask Mode (attn_mask_type): '{predictor_attn_mask_type}'")
        log_info(f"  NOTE: Auxiliary pixel-wise heads have been removed.")
        log_info(f"="*50)

    def forward(self, input_image):
        """
        模型的前向传播过程。
        
        Args:
            input_image: 输入图像张量, 形状 (B, C, H, W)
            
        Returns:
            predictions: 包含模型所有输出的字典
        """
        # 1. 编码器前向传播，提取多尺度特征
        en_feats = self.pixel_encoder(input_image)
        
        # 2. 解码器前向传播，融合特征并上采样
        de_feats = self.pixel_decoder(en_feats)
        
        # 3. 准备送入Transformer预测头的输入
        # a) 多尺度的像素特征 (用于Cross-Attention)
        mask_pixel_feats = [de_feats[i] for i in self.mask_pixel_idxs]
        # b) 高分辨率的掩码特征 (用于最终生成掩码)
        mask_features = self.linear_mask_features(de_feats[-1])
        
        # 4. Transformer预测头前向传播
        # 这是模型唯一的输出源
        predictions = self.predictor(mask_pixel_feats, mask_features, mask=None)

        
        return predictions

