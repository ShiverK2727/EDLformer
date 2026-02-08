import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict


# 基础 EDL 组件

def edl_digamma_ace_loss(alpha: torch.Tensor, y_one_hot: torch.Tensor) -> torch.Tensor:
    """
    Alpha shape: [B, N, C, H, W]
    y_one_hot shape: [B, N, C, H, W]
    """
    S = torch.sum(alpha, dim=2, keepdim=True)  # [B, N, 1, H, W]
    loss = torch.sum(y_one_hot * (torch.digamma(S) - torch.digamma(alpha)), dim=2)  # [B, N, H, W]
    return loss.mean()


def edl_kl_regularizer(alpha: torch.Tensor, y_one_hot: torch.Tensor, epoch: int, annealing_step: int = 10) -> torch.Tensor:
    """
    KL 正则，按像素展开。
    """
    num_classes = alpha.shape[2]
    alpha_flat = alpha.permute(0, 1, 3, 4, 2).contiguous().view(-1, num_classes)
    y_flat = y_one_hot.permute(0, 1, 3, 4, 2).contiguous().view(-1, num_classes)
    evidence = alpha_flat - 1
    alp = evidence * (1 - y_flat) + 1
    beta = torch.ones((1, num_classes), device=alpha.device)
    S_alpha = torch.sum(alp, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB_alpha = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alp), dim=1, keepdim=True)
    lnB_beta = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg_alpha_sum = torch.digamma(S_alpha)
    dg_alpha = torch.digamma(alp)
    kl = torch.sum((alp - beta) * (dg_alpha - dg_alpha_sum), dim=1, keepdim=True) + lnB_alpha + lnB_beta
    anneal = min(1.0, float(epoch) / float(max(1, annealing_step)))
    return (anneal * kl).mean()


def soft_dice_loss(prob: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    prob, target: [B, N, C, H, W]
    """
    prob_flat = prob.flatten(0, 1).flatten(2)  # [B*N*C, HW]
    target_flat = target.flatten(0, 1).flatten(2)
    intersection = (prob_flat * target_flat).sum(dim=1)
    denom = prob_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice = (2 * intersection + smooth) / (denom + smooth)
    return (1 - dice).mean()


def soft_dice_loss_4d(prob: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    prob, target: [B, C, H, W]
    """
    prob_flat = prob.flatten(1).flatten(1)  # [B*C, HW]
    target_flat = target.flatten(1).flatten(1)
    intersection = (prob_flat * target_flat).sum(dim=1)
    denom = prob_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice = (2 * intersection + smooth) / (denom + smooth)
    return (1 - dice).mean()


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

    def forward(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor, epoch: int = 0):
        """
        outputs: 来自 SimpleMaskFormerV2 的输出字典。
        targets: [B, N, C, H, W] (one-hot 或 {0,1})
        """
        loss_dict = {}
        targets = self._ensure_one_hot(targets)

        # === 主 Evidence 分支 ===
        evidence = outputs['pred_evidence']  # [B, N, C, H, W]
        alpha = evidence + 1.0
        prob = alpha / torch.sum(alpha, dim=2, keepdim=True)

        loss_ace = edl_digamma_ace_loss(alpha, targets)
        loss_kl = edl_kl_regularizer(alpha, targets, epoch, self.kl_annealing_step)
        loss_dice = soft_dice_loss(prob, targets)

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
                logits_aux = aux['pred_masks']  # [B, N*C, H, W]
                if logits_aux.shape[-2:] != targets.shape[-2:]:
                    logits_aux = F.interpolate(logits_aux, size=targets.shape[-2:], mode='bilinear', align_corners=False)
                bsz = logits_aux.shape[0]
                aux_reshaped = logits_aux.view(bsz, self.num_experts, self.num_classes, *logits_aux.shape[-2:])
                alpha_aux = F.softplus(aux_reshaped) + 1.0
                prob_aux = alpha_aux / torch.sum(alpha_aux, dim=2, keepdim=True)

                ace_aux = edl_digamma_ace_loss(alpha_aux, targets)
                dice_aux = soft_dice_loss(prob_aux, targets)
                weight = self.transformer_aux_weights[idx] if idx < len(self.transformer_aux_weights) else 1.0
                total_loss = total_loss + weight * (ace_aux + dice_aux)
                loss_dict[f'trans_aux_{idx}_ace'] = ace_aux.detach()
                loss_dict[f'trans_aux_{idx}_dice'] = dice_aux.detach()

        # === Pixel Decoder 主分支 ===
        pixel_logits = outputs.get('pixel_final_mask')  # [B, C, H, W]
        if pixel_logits is not None:
            consensus_target = targets.mean(dim=1)  # [B, C, H, W]
            pixel_prob = torch.sigmoid(pixel_logits)
            pixel_bce = self.bce_loss(pixel_logits, consensus_target)
            pixel_dice = soft_dice_loss_4d(pixel_prob, consensus_target)
            total_loss = total_loss + self.weight_pixel_bce * pixel_bce + self.weight_pixel_dice * pixel_dice
            loss_dict['pixel_bce'] = pixel_bce.detach()
            loss_dict['pixel_dice'] = pixel_dice.detach()

        # === Pixel Decoder 深监督 ===
        if self.use_pixel_aux:
            pix_aux = outputs.get('pix_pred_masks', []) or []
            for idx, aux_logits in enumerate(pix_aux):
                consensus_target = targets.mean(dim=1)
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
