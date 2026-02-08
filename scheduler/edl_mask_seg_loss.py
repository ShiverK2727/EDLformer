import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict


# 基础 EDL 组件

def beta_ace_loss(alpha: torch.Tensor, beta: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Binary ACE for Beta-EDL (per-channel independent)."""
    S = alpha + beta
    loss = target * (torch.digamma(S) - torch.digamma(alpha)) + (1 - target) * (torch.digamma(S) - torch.digamma(beta))
    return loss.mean()


def beta_kl_regularizer(alpha: torch.Tensor, beta: torch.Tensor, epoch: int, annealing_step: int = 10) -> torch.Tensor:
    """KL to Beta(1,1) prior, per-channel."""
    alpha_flat = alpha.flatten(0, -4)  # [B*N*C*H*W]
    beta_flat = beta.flatten(0, -4)
    S = alpha_flat + beta_flat
    kl = (
        (alpha_flat - 1) * (torch.digamma(alpha_flat) - torch.digamma(S)) +
        (beta_flat - 1) * (torch.digamma(beta_flat) - torch.digamma(S)) -
        (torch.lgamma(alpha_flat) + torch.lgamma(beta_flat) - torch.lgamma(S)) +
        torch.lgamma(torch.tensor(2.0, device=alpha.device))
    )
    anneal = min(1.0, float(epoch) / float(max(1, annealing_step)))
    return (anneal * kl).mean()


def soft_dice_loss(prob: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    计算 Soft Dice Loss。
    兼容 Soft Target (概率) 和 Hard Target (掩码)。
    prob, target: [B, N, C, H, W]
    计算方式：按 Spatial 维度 (H, W) 计算 Dice，保留 Batch, Expert, Class 维度，最后取平均。
    """
    b, n, c, h, w = prob.shape
    prob = prob.view(b, n, c, -1)
    target = target.view(b, n, c, -1)
    
    intersection = (prob * target).sum(dim=-1)  # Sum over Spatial
    denom = prob.sum(dim=-1) + target.sum(dim=-1)
    
    dice = (2 * intersection + smooth) / (denom + smooth)
    return 1 - dice.mean()


def soft_dice_loss_4d(prob: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    计算 Soft Dice Loss (4D 版本)。
    prob, target: [B, C, H, W]
    计算方式：按 Spatial 维度 (H, W) 计算 Dice，保留 Batch, Class 维度，最后取平均。
    """
    b, c, h, w = prob.shape
    prob = prob.view(b, c, -1)
    target = target.view(b, c, -1)
    
    intersection = (prob * target).sum(dim=-1)  # Sum over Spatial
    denom = prob.sum(dim=-1) + target.sum(dim=-1)
    dice = (2 * intersection + smooth) / (denom + smooth)
    return 1 - dice.mean()


def compute_gram_matrix(x: torch.Tensor) -> torch.Tensor:
    """
    x: [B, C, H, W]
    返回: [B, C, C]
    """
    B, C, H, W = x.shape
    feat = x.view(B, C, H * W)
    gram = torch.bmm(feat, feat.transpose(1, 2)) / float(H * W)
    return gram


class MaskSegEDLLoss(nn.Module):
    """针对 SimpleMaskFormerV2 的无分类版 EDL 分割损失。"""

    def __init__(
        self,
        num_classes: int,
        num_experts: int,
        # 主分支权重
        weight_ace: float = 1.0,
        weight_kl: float = 0.1,
        weight_dice: float = 1.0,
        kl_annealing_step: int = 10,
        # Pixel decoder 主分支权重
        weight_pixel_bce: float = 1.0,
        weight_pixel_dice: float = 1.0,
        # 深监督配置
        use_pixel_aux: bool = False,
        use_transformer_aux: bool = False,
        pixel_aux_weights: Optional[List[float]] = None,
        transformer_aux_weights: Optional[List[float]] = None,
        # Dice 计算模式
        dice_use_evidence_softmax: bool = False,
        # Pixel Decoder 监督模式 ('mean', 'intersection', 'union')
        pixel_consensus_type: str = 'mean',
        # DINO 对齐
        use_dino_align: bool = False,
        weight_dino_align: float = 1.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_experts = num_experts
        self.weight_ace = weight_ace
        self.weight_kl = weight_kl
        self.weight_dice = weight_dice
        self.kl_annealing_step = kl_annealing_step
        self.weight_pixel_bce = weight_pixel_bce
        self.weight_pixel_dice = weight_pixel_dice
        self.use_pixel_aux = use_pixel_aux
        self.use_transformer_aux = use_transformer_aux
        self.pixel_aux_weights = pixel_aux_weights or []
        self.transformer_aux_weights = transformer_aux_weights or []
        self.dice_use_evidence_softmax = dice_use_evidence_softmax
        self.pixel_consensus_type = pixel_consensus_type
        self.use_dino_align = use_dino_align
        self.weight_dino_align = weight_dino_align
        self.bce_loss = nn.BCEWithLogitsLoss()

    def _ensure_one_hot(self, targets: torch.Tensor) -> torch.Tensor:
        """将输入转为 one-hot；若已为 one-hot 则直接返回。"""
        if targets.dtype in (torch.float, torch.float16, torch.float32, torch.float64):
            # 假设已是 one-hot 或概率，直接返回
            return targets
        # 若为整型，转换为 one-hot
        b, n, h, w = targets.shape
        return F.one_hot(targets.long(), num_classes=self.num_classes).permute(0, 1, 4, 2, 3).float()

    def _get_consensus_target(self, targets: torch.Tensor) -> torch.Tensor:
        """
        根据 pixel_consensus_type 生成 Pixel Decoder 的监督目标。
        targets: [B, N, C, H, W]
        Returns: [B, C, H, W]
        """
        if self.pixel_consensus_type == 'intersection':
            # 严格交集：仅当所有专家都标注为 1 时才为 1 (Hard Mask)
            # targets 假设为 0/1 或 logits，用 >0.5 判定是否激活
            return (targets.min(dim=1)[0] > 0.5).float()
        elif self.pixel_consensus_type == 'union':
            # 并集：任意专家标注为 1 则为 1 (Hard Mask)
            return (targets.max(dim=1)[0] > 0.5).float()
        else:
            # 默认 'mean': 平均概率 (Soft Probability, 0~1)
            return targets.mean(dim=1)

    def forward(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor, epoch: int = 0):
        """
        outputs: 来自 SimpleMaskFormerV2 的输出字典。
        targets: [B, N, C, H, W] (one-hot 或 {0,1})
        """
        loss_dict = {}
        targets = self._ensure_one_hot(targets)

        # === 主 Evidence 分支 ===
        logits = outputs['pred_logits']           # [B, N, C, 2, H, W]
        alpha = F.softplus(logits[:, :, :, 0]) + 1.0  # [B, N, C, H, W]
        beta  = F.softplus(logits[:, :, :, 1]) + 1.0
        if self.dice_use_evidence_softmax:
            # DEviS 风格：对证据做 softmax，得到锐化后的概率
            dice_prob_main = torch.softmax(F.softplus(logits[:, :, :, 0]), dim=2)
        else:
            dice_prob_main = alpha / (alpha + beta)

        prob = alpha / (alpha + beta)

        loss_ace = beta_ace_loss(alpha, beta, targets)
        loss_kl = beta_kl_regularizer(alpha, beta, epoch, self.kl_annealing_step)
        loss_dice = soft_dice_loss(dice_prob_main, targets)

        total_loss = (
            self.weight_ace * loss_ace +
            self.weight_kl * loss_kl +
            self.weight_dice * loss_dice
        )

        loss_dict['main_ace'] = loss_ace.detach()
        loss_dict['main_kl'] = loss_kl.detach()
        loss_dict['main_dice'] = loss_dice.detach()

        # === Transformer 深监督 ===
        if self.use_transformer_aux:
            aux_outputs = outputs.get('aux_outputs', []) or []
            for idx, aux in enumerate(aux_outputs):
                logits_aux = aux['pred_masks']  # [B, N, C, 2, H, W]
                if logits_aux.shape[-2:] != targets.shape[-2:]:
                    logits_aux = F.interpolate(logits_aux, size=targets.shape[-2:], mode='bilinear', align_corners=False)
                alpha_aux = F.softplus(logits_aux[:, :, :, 0]) + 1.0
                beta_aux  = F.softplus(logits_aux[:, :, :, 1]) + 1.0
                if self.dice_use_evidence_softmax:
                    dice_prob_aux = torch.softmax(F.softplus(logits_aux[:, :, :, 0]), dim=2)
                else:
                    dice_prob_aux = alpha_aux / (alpha_aux + beta_aux)

                ace_aux = beta_ace_loss(alpha_aux, beta_aux, targets)
                dice_aux = soft_dice_loss(dice_prob_aux, targets)
                weight = self.transformer_aux_weights[idx] if idx < len(self.transformer_aux_weights) else 1.0
                total_loss = total_loss + weight * (ace_aux + dice_aux)
                loss_dict[f'trans_aux_{idx}_ace'] = ace_aux.detach()
                loss_dict[f'trans_aux_{idx}_dice'] = dice_aux.detach()

        # === Pixel Decoder 主分支 ===
        pixel_logits = outputs.get('pixel_final_mask')  # [B, C, H, W]
        if pixel_logits is not None:
            consensus_target = self._get_consensus_target(targets)  # [B, C, H, W]
            pixel_prob = torch.sigmoid(pixel_logits)
            pixel_bce = self.bce_loss(pixel_logits, consensus_target)
            pixel_dice = soft_dice_loss_4d(pixel_prob, consensus_target)
            total_loss = total_loss + self.weight_pixel_bce * pixel_bce + self.weight_pixel_dice * pixel_dice
            loss_dict['pixel_bce'] = pixel_bce.detach()
            loss_dict['pixel_dice'] = pixel_dice.detach()

        # === Pixel Decoder 深监督 ===
        if self.use_pixel_aux:
            pix_aux = outputs.get('pix_pred_masks', []) or []
            # 共识目标只计算一次
            consensus_target = self._get_consensus_target(targets)
            for idx, aux_logits in enumerate(pix_aux):
                if aux_logits.shape[-2:] != consensus_target.shape[-2:]:
                    aux_logits = F.interpolate(aux_logits, size=consensus_target.shape[-2:], mode='bilinear', align_corners=False)
                prob_aux = torch.sigmoid(aux_logits)
                bce_aux = self.bce_loss(aux_logits, consensus_target)
                dice_aux = soft_dice_loss_4d(prob_aux, consensus_target)
                weight = self.pixel_aux_weights[idx] if idx < len(self.pixel_aux_weights) else 1.0
                total_loss = total_loss + weight * (bce_aux + dice_aux)
                loss_dict[f'pix_aux_{idx}_bce'] = bce_aux.detach()
                loss_dict[f'pix_aux_{idx}_dice'] = dice_aux.detach()

        # === DINO Gram 对齐 ===
        if self.use_dino_align:
            bridge_feat = outputs.get('bridge_feat')
            dino_feat = outputs.get('dino_feat')
            if bridge_feat is not None and dino_feat is not None:
                gram_bridge = compute_gram_matrix(bridge_feat)
                gram_dino = compute_gram_matrix(dino_feat.detach())
                align_loss = F.mse_loss(gram_bridge, gram_dino)
                total_loss = total_loss + self.weight_dino_align * align_loss
                loss_dict['dino_align'] = align_loss.detach()

        loss_dict['total'] = total_loss.detach()
        return total_loss, loss_dict
