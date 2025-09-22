import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.encoders import get_encoder


class SMPUnetEncoder(nn.Module):
    def __init__(self, encoder_name="resnet34", encoder_weights="imagenet", in_channels=3):
        super().__init__()
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            weights=encoder_weights,
        )
    def forward(self, x):
        features = self.encoder(x)
        features = features[1:]
        # print(f"Encoder features shape: {[f.shape for f in features]}")
        return features


class CustomDecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, use_batchnorm=True, upsample_mode='interp'):
        super().__init__()
        self.upsample_mode = upsample_mode
        
        if upsample_mode == 'transposed':
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        else:
            self.up = None  # 用F.interpolate

        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        self.relu2 = nn.ReLU(inplace=True)


    def forward(self, x, skip=None):
        if self.upsample_mode == 'transposed':
            assert self.up is not None  # 运行时保证
            x = self.up(x)
        elif self.upsample_mode == 'interp':
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        else:
            raise ValueError(f"Unknown upsample_mode: {self.upsample_mode}")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x
    
    
class CustomUnetDecoder(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, use_batchnorm=True, upsample_mode='interp'):
        super().__init__()
        # encoder_channels from resnet encoder (e.g. [3, 64, 64, 128, 256, 512])
        encoder_channels = encoder_channels[1:][::-1]
        in_channels = [encoder_channels[0]] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        self.blocks = nn.ModuleList([
            CustomDecoderBlock(in_ch, skip_ch, out_ch, use_batchnorm, upsample_mode)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, decoder_channels)
        ])

    def forward(self, features):
        # features: list, 从浅到深（已去掉第一个特征）
        features = features[::-1]  # 反转，从深到浅
        x = features[0]
        skips = features[1:]
        decoder_features = []
        for i, block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = block(x, skip)
            decoder_features.append(x)
        return decoder_features  # list, 从深到浅
    

class UnetBackbone(nn.Module):
    def __init__(self, encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, decoder_channels=[256, 128, 64, 32, 16], use_batchnorm=True):
        super().__init__()
        self.encoder = SMPUnetEncoder(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels
        )
        # print(f"Encoder输出通道: {self.encoder.encoder.out_channels}")
        self.decoder = CustomUnetDecoder(
            encoder_channels=self.encoder.encoder.out_channels,
            decoder_channels=decoder_channels,
            use_batchnorm=use_batchnorm
        )
    def forward(self, x):
        features = self.encoder(x)
        decoder_features = self.decoder(features)
        return decoder_features


class UnetWithClassHead(nn.Module):
    def __init__(self, encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, decoder_channels=[256, 128, 64, 32, 16], num_classes=2, use_batchnorm=True):
        super().__init__()
        self.unet = UnetBackbone(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            decoder_channels=decoder_channels,
            use_batchnorm=use_batchnorm
        )
        # 最后一层输出特征通道
        final_channels = decoder_channels[-1]
        self.mask_head = nn.Conv2d(final_channels, num_classes, kernel_size=1)
    def forward(self, x):
        feats = self.unet(x)
        out = self.mask_head(feats[-1])  # [B, num_classes, H, W]
        return out

if __name__ == "__main__":
    model = UnetWithClassHead()
    inp = torch.randn(2, 3, 256, 256)
    out = model(inp)
    print("Output shape:", out.shape)