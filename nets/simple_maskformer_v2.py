import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoConfig
from typing import List

from .smp_unet_backbone_simple import SMPUnetEncoder, CustomUnetDecoder
from .maskdecoder_simple_v2 import DualSourceTransformerDecoder

class SemanticBridge(nn.Module):
    """ 将 ResNet 特征投影到 DINO 维度 """
    def __init__(self, in_channels=512, out_channels=1536):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.GELU()
        )
    def forward(self, x): return self.projector(x)
    
class SimpleMaskFormerV2(nn.Module):
    def __init__(
        self,
        in_channels=3,
        num_classes=2,
        num_experts=4,
        backbone_name="resnet34",
        dino_model_path=None,
        hidden_dim=64,
        use_dino_align: bool = False,
        bridge_layers_indices: List[int] = [0],
        # Decoder configurations to align with SimpleMaskFormer
        backbone_decoder_channels: List[int] = [128, 64, 32, 16, 8],
        backbone_use_batchnorm: bool = True,
        backbone_upsample_mode: str = 'interp',
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_experts = num_experts
        self.use_dino_align = use_dino_align and dino_model_path is not None
        self.bridge_layers_indices = bridge_layers_indices
        
        # 1. Encoder
        self.backbone = SMPUnetEncoder(
            encoder_name=backbone_name, 
            encoder_weights="imagenet",
            in_channels=in_channels
        )
        # encoder returns [c1, c2, c3, c4, c5]
        # CustomUnetDecoder expects the full channel list (including input/dummy at idx 0)
        # to correctly calculate skip connections.
        enc_channels = self.backbone.encoder.out_channels 
        self.bottleneck_dim = enc_channels[-1]
        
        # 2. DINOv3 Setup
        self.dino_model_path = dino_model_path if self.use_dino_align else None
        if self.dino_model_path:
            # print(f"Loading DINOv3 from: {dino_model_path}")
            config = AutoConfig.from_pretrained(self.dino_model_path, trust_remote_code=True)
            self.dino_model = AutoModel.from_pretrained(self.dino_model_path, config=config, trust_remote_code=True)
            for param in self.dino_model.parameters(): param.requires_grad = False
            self.dino_model.eval()
            self.dino_dim = self.dino_model.config.hidden_sizes[-1] if hasattr(self.dino_model.config, 'hidden_sizes') else 1536
        else:
            self.dino_model = None
            self.dino_dim = 1536
            
        # 3. Semantic Bridge (only when DINO align is enabled)
        self.bridge = SemanticBridge(self.bottleneck_dim, self.dino_dim) if self.use_dino_align else None
        
        # 4. Pixel Decoder (Reverted to traditional UNetDecoder)
        self.pixel_decoder = CustomUnetDecoder(
            encoder_channels=enc_channels,
            decoder_channels=backbone_decoder_channels,
            use_batchnorm=backbone_use_batchnorm,
            upsample_mode=backbone_upsample_mode
        )
        
        # Projection for mask features (last layer of decoder) - use hidden_dim to align with transformer
        self.linear_mask_features = nn.Conv2d(
            backbone_decoder_channels[-1], hidden_dim,
            kernel_size=1, stride=1, padding=0
        )
        
        # Auxiliary Segmentation Head (Traditional)
        # Assuming backbone_decoder_channels[-1] is the channel count of the highest resolution feature
        self.segmentation_head = nn.Conv2d(
            backbone_decoder_channels[-1], num_classes, kernel_size=1
        )
        
        # 5. Dual-Source Decoder
        # Select features to send to transformer. 
        # CustomUnetDecoder returns features from Deep to Shallow order of blocks:
        # Index 0: Stride 16 (ch=128)
        # Index 1: Stride 8 (ch=64)
        # Index 2: Stride 4 (ch=32)
        # Index 3: Stride 2 (ch=16)
        # Index 4: Stride 1 (ch=8)
        
        # User requested 8, 4, 2 strides (Indices 1, 2, 3)
        self.transformer_scale_indices = [1, 2, 3]
        trans_in_channels = [backbone_decoder_channels[i] for i in self.transformer_scale_indices]
        # Pixel-decoder deep supervision heads aligned with transformer scales
        self.pixel_aux_heads = nn.ModuleList([
            nn.Conv2d(backbone_decoder_channels[i], num_classes, kernel_size=1)
            for i in self.transformer_scale_indices
        ])
        
        self.transformer_decoder = DualSourceTransformerDecoder(
            in_channels_pixel_list=trans_in_channels,
            in_channels_bridge=self.dino_dim, # Input from Bridge
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            num_classes=num_classes,
            use_bridge=self.use_dino_align,
            bridge_layers_indices=self.bridge_layers_indices,
        )

    def forward(self, x):
        img_size = x.shape[2:]
        
        # A. Backbone
        features = self.backbone(x) # [c1, c2, c3, c4, c5]
        bottleneck = features[-1]
        
        # B. Bridge (Source 1: Semantic)
        bridge_feat = self.bridge(bottleneck) if self.bridge is not None else None
        
        # DINO (Teacher)
        dino_feat = None
        if self.dino_model is not None:
            with torch.no_grad():
                out = self.dino_model(x, output_hidden_states=True)
                dino_feat = out.last_hidden_state
                # Handle 3D output [B, L, D] from transformers
                if len(dino_feat.shape) == 3:
                    B, L, D = dino_feat.shape
                    # Attempt to reshape to [B, D, H, W]
                    side = int(L ** 0.5)
                    if side * side == L:
                        dino_feat = dino_feat.permute(0, 2, 1).contiguous().view(B, D, side, side)
                    else:
                        # Check for CLS token (L = side*side + 1)
                        side_cls = int((L - 1) ** 0.5)
                        if side_cls * side_cls == (L - 1):
                             # Remove CLS (assuming it's first)
                             feat_no_cls = dino_feat[:, 1:, :] 
                             dino_feat = feat_no_cls.permute(0, 2, 1).contiguous().view(B, D, side_cls, side_cls)
        
        # C. Pixel Decoder (Source 2: Spatial)
        de_feats = self.pixel_decoder(features)
        
        # Standard MaskFormer takes the last feature (Highest resolution)
        # de_feats is a list of decoder features. Last one is usually highest res.
        mask_features = self.linear_mask_features(de_feats[-1]) # [B, hidden_dim, H, W]
        
        # Select Multi-scale features for Transformer (Stride 8, 4, 2)
        ms_pixel_feats = [de_feats[i] for i in self.transformer_scale_indices]

        # Pixel decoder deep supervision (aligned with transformer scales)
        pixel_aux_masks = []
        for head, feat in zip(self.pixel_aux_heads, ms_pixel_feats):
            aux_logits = head(feat)
            pixel_aux_masks.append(aux_logits)
        
        # D. Transformer (Cascaded Interaction) - mask-only outputs
        transformer_out = self.transformer_decoder(ms_pixel_feats, bridge_feat, mask_features)
        
        # E. Auxiliary Pixel Segmentation
        pix_pred_mask = self.segmentation_head(de_feats[-1])
        
        # F. Format Output for DSEBridgeLoss (Evidence + Bridge + DINO)
        final_mask_logits = transformer_out['pred_masks']
             
        # Evidence Generation
        # [B, N*K, H, W] -> [B, N, K, H, W]
        # Query Order: Expert 1 (Class 1...K), Expert 2 (Class 1...K)
        B_sz = final_mask_logits.shape[0]
        evidence = F.softplus(final_mask_logits)
        evidence = evidence.view(B_sz, self.num_experts, self.num_classes, final_mask_logits.shape[-2], final_mask_logits.shape[-1])
        
        return {
            "pred_evidence": evidence, # For DSEBridgeLoss task_loss
            "aux_outputs": transformer_out['aux_outputs'], # Deep supervision (mask only)
            "pixel_final_mask": pix_pred_mask, # Final pixel-decoder head
            "pix_pred_masks": pixel_aux_masks, # Pixel-decoder deep supervision aligned with transformer scales
            "bridge_feat": bridge_feat, # For DSEBridgeLoss align
            "dino_feat": dino_feat, # For DSEBridgeLoss align
        }