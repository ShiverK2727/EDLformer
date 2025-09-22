import torch
import torch.nn.functional as F
from torch import nn
from typing import Dict, Tuple

from logger import log_info
# --- [修复] 导入正确的匹配器模块 ---
from .edl_single_matcher import EDLHungarianMatcher

# ==============================================================================
# 核心 EDL 和 标准 损失函数 (这些辅助函数保持不变)
# ==============================================================================

def edl_digamma_ace_loss(alpha: torch.Tensor, y_one_hot: torch.Tensor, reduction: str = 'mean'):
    S = torch.sum(alpha, dim=1, keepdim=True)
    loss = torch.sum(y_one_hot * (torch.digamma(S) - torch.digamma(alpha)), dim=1)
    return loss.mean() if reduction == 'mean' else loss.sum()

def kl_divergence(alpha: torch.Tensor, num_classes: int):
    beta = torch.ones((1, num_classes), device=alpha.device)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB_alpha = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_beta = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg_alpha_sum = torch.digamma(S_alpha)
    dg_alpha = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg_alpha - dg_alpha_sum), dim=1, keepdim=True) + lnB_alpha + lnB_beta
    return kl

def edl_kl_regularizer(alpha: torch.Tensor, y_one_hot: torch.Tensor, epoch: int, annealing_step: int = 10):
    num_classes = alpha.shape[1]
    alpha_flat = alpha.permute(0, 2, 3, 1).contiguous().view(-1, num_classes)
    y_one_hot_flat = y_one_hot.permute(0, 2, 3, 1).contiguous().view(-1, num_classes)
    # 注意：这里的 'evidence' 是从 alpha 反推的，仅用于KL散度计算
    evidence = alpha_flat - 1
    alp = evidence * (1 - y_one_hot_flat) + 1
    kl_div = kl_divergence(alp, num_classes)
    annealing_coef = min(1.0, epoch / annealing_step)
    loss = annealing_coef * kl_div
    return loss.mean()

def compute_soft_dice_loss(pred_probs: torch.Tensor, targets_one_hot: torch.Tensor):
    pred_flat = pred_probs.flatten(2)
    target_flat = targets_one_hot.flatten(2)
    intersection = torch.sum(pred_flat * target_flat, dim=2)
    pred_vol = torch.sum(pred_flat, dim=2)
    target_vol = torch.sum(target_flat, dim=2)
    dice_score = (2.0 * intersection + 1e-5) / (pred_vol + target_vol + 1.0 + 1e-5)
    dice_loss = -torch.log(dice_score)
    return dice_loss.mean()

# ==============================================================================
# 混合输入版的主损失类 (Hybrid Input Version)
# ==============================================================================
class EDLSingleLoss(nn.Module):
    def __init__(
        self,
        matcher_cost_class: float = 2.0,
        matcher_cost_mask: float = 5.0,
        matcher_cost_dice: float = 5.0,
        # --- [重构] 新的四权重控制系统 ---
        loss_weight_cls: float = 1.0,              # 分类头权重
        loss_weight_mask: float = 1.0,             # 证据学习分割权重  
        loss_weight_dice: float = 1.0,             # Dice损失权重
        loss_weight_consensus_mask: float = 1.0,   # 共识证据学习权重
        loss_weight_consensus_dice: float = 1.0,   # 共识Dice损失权重
        kl_annealing_step: int = 10,
        # --- [新增] dice计算基础控制开关 ---
        dice_on_evidence: bool = False,            # True: dice基于证据学习概率, False: dice基于原始logits
        # --- [保留] 一致性监督损失的参数 ---
        loss_weight_consistency: float = 0.5,
        consistency_uncertainty_threshold: float = 0.2,
        # ---
        **kwargs
    ):
        super().__init__()
        self.matcher = EDLHungarianMatcher(
            cost_class=matcher_cost_class, cost_mask=matcher_cost_mask, cost_dice=matcher_cost_dice
        )
        # --- [重构] 保存新的权重参数 ---
        self.loss_weight_cls = loss_weight_cls
        self.loss_weight_mask = loss_weight_mask
        self.loss_weight_dice = loss_weight_dice
        self.loss_weight_consensus_mask = loss_weight_consensus_mask
        self.loss_weight_consensus_dice = loss_weight_consensus_dice
        self.kl_annealing_step = kl_annealing_step
        self.dice_on_evidence = dice_on_evidence
        # --- [新增] 保存新参数 ---
        self.loss_weight_consistency = loss_weight_consistency
        self.consistency_uncertainty_threshold = consistency_uncertainty_threshold
        # ---
        log_info("EDLSingleLoss (Hybrid Input Version with Consistency Loss) initialized.", print_message=True)

    def forward(
        self, 
        # --- 用于匹配的原始模型输出 ---
        raw_cls_logits: torch.Tensor,
        raw_mask_logits: torch.Tensor,
        # --- 用于损失计算的、经过EDL处理的输出 ---
        expert_alpha: torch.Tensor,
        cls_alpha: torch.Tensor,
        pixel_alpha: torch.Tensor,
        pixel_logits: torch.Tensor, # 用于共识部分的Dice Loss
        # --- 真值标签 ---
        target_masks: torch.Tensor,
        target_labels: torch.Tensor,
        consensus_target: torch.Tensor,
        epoch: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # 1. 匹配过程：使用原始logits进行匹配，保证稳定性
        indices = self.matcher(raw_cls_logits, raw_mask_logits, target_masks, target_labels)
        
        # 2. 专家损失：分离为分类、证据学习掩码、Dice三个部分
        loss_cls, loss_exp_mask, loss_exp_dice = self._compute_expert_loss(
            cls_alpha, expert_alpha, raw_mask_logits, 
            target_labels, target_masks, indices, epoch
        )
        
        # 3. 共识损失：分离为证据学习和Dice两部分
        loss_con_mask, loss_con_dice = self._compute_consensus_loss(pixel_alpha, pixel_logits, consensus_target, epoch)

        # --- [保留] 4. 一致性监督损失 ---
        loss_consistency = self._compute_consistency_loss(
            expert_alpha, pixel_alpha, indices
        )
        # ---

        # --- [重构] 5. 加权合并总损失 - 使用新的四权重系统 ---
        total_loss = (self.loss_weight_cls * loss_cls + 
                      self.loss_weight_mask * loss_exp_mask + 
                      self.loss_weight_dice * loss_exp_dice +
                      self.loss_weight_consensus_mask * loss_con_mask +
                      self.loss_weight_consensus_dice * loss_con_dice +
                      self.loss_weight_consistency * loss_consistency)
        
        return total_loss, loss_cls, loss_exp_mask, loss_exp_dice, loss_con_mask, loss_con_dice, loss_consistency
        # ---

    def _compute_expert_loss(self, cls_alpha, expert_alpha, raw_mask_logits, target_labels, target_masks, indices, epoch):
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        
        if len(src_idx[0]) == 0:
            device = cls_alpha.device
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

        # --- 1. 分类损失 (基于预处理的alpha) ---
        matched_cls_alpha = cls_alpha[src_idx]
        matched_cls_labels = target_labels[tgt_idx]
        cls_y_one_hot = F.one_hot(matched_cls_labels, num_classes=matched_cls_alpha.shape[-1]).float()
        loss_cls = edl_digamma_ace_loss(matched_cls_alpha, cls_y_one_hot)
        
        # --- 2. 证据学习掩码损失 (仅包含ACE和KL正则化) ---
        matched_expert_alpha = expert_alpha[src_idx]
        target_masks_one_hot = target_masks[tgt_idx]
        
        loss_mask_ace = edl_digamma_ace_loss(matched_expert_alpha, target_masks_one_hot)
        loss_mask_kl = edl_kl_regularizer(matched_expert_alpha, target_masks_one_hot, epoch, self.kl_annealing_step)
        loss_expert_mask = loss_mask_ace + loss_mask_kl
        
        # --- 3. Dice损失 (根据dice_on_evidence开关选择计算基础) ---
        if self.dice_on_evidence:
            # 基于证据学习的概率计算dice
            evidence = matched_expert_alpha - 1
            S = torch.sum(matched_expert_alpha, dim=1, keepdim=True)
            dice_input = matched_expert_alpha / S  # 使用Dirichlet分布的期望概率
        else:
            # 基于原始logits的softmax概率计算dice
            matched_mask_logits = raw_mask_logits[src_idx]
            dice_input = F.softmax(matched_mask_logits, dim=1)
            
        loss_dice = compute_soft_dice_loss(dice_input, target_masks_one_hot)
        
        return loss_cls, loss_expert_mask, loss_dice

    def _compute_consensus_loss(self, pixel_alpha, pixel_logits, consensus_target, epoch):
        if (self.loss_weight_consensus_mask <= 0 and self.loss_weight_consensus_dice <= 0) or pixel_alpha is None:
            device = consensus_target.device
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
        
        consensus_target_one_hot = consensus_target.float()
        
        # --- 1. 证据学习部分 (仅ACE + KL正则化) ---
        loss_con_mask = torch.tensor(0.0, device=consensus_target.device)
        if self.loss_weight_consensus_mask > 0:
            loss_mask_ace = edl_digamma_ace_loss(pixel_alpha, consensus_target_one_hot)
            loss_mask_kl = edl_kl_regularizer(pixel_alpha, consensus_target_one_hot, epoch, self.kl_annealing_step)
            loss_con_mask = loss_mask_ace + loss_mask_kl
        
        # --- 2. Dice损失部分 (根据dice_on_evidence开关选择计算基础) ---
        loss_con_dice = torch.tensor(0.0, device=consensus_target.device)
        if self.loss_weight_consensus_dice > 0:
            if self.dice_on_evidence:
                # 基于证据学习的概率计算dice
                S = torch.sum(pixel_alpha, dim=1, keepdim=True)
                dice_input = pixel_alpha / S  # 使用Dirichlet分布的期望概率
            else:
                # 基于原始logits的softmax概率计算dice
                dice_input = F.softmax(pixel_logits, dim=1)
            
            loss_con_dice = compute_soft_dice_loss(dice_input, consensus_target_one_hot)
        
        return loss_con_mask, loss_con_dice

    # --- [新增] 计算一致性损失的函数 ---
    def _compute_consistency_loss(self, expert_alpha, pixel_alpha, indices):
        # 如果一致性损失的权重为0，则不进行计算，节省资源
        if self.loss_weight_consistency <= 0 or pixel_alpha is None:
            return torch.tensor(0.0, device=expert_alpha.device)
            
        num_classes = pixel_alpha.shape[1]
        
        # 1. 计算共识预测的整体不确定性
        S_pixel = torch.sum(pixel_alpha, dim=1, keepdim=True)
        U_pixel = num_classes / (S_pixel + 1e-8) # [B, 1, H, W]
        
        # 2. 根据阈值创建高置信度区域的掩码
        confident_mask = (U_pixel < self.consistency_uncertainty_threshold).float()
        
        # 如果没有任何区域是高置信度的，则不计算损失
        if confident_mask.sum() == 0:
            return torch.tensor(0.0, device=expert_alpha.device)

        # 3. 获取匹配上的专家预测和对应的共识预测
        src_idx = self._get_src_permutation_idx(indices)
        
        # 如果没有匹配，不计算损失
        if len(src_idx[0]) == 0:
            return torch.tensor(0.0, device=expert_alpha.device)
            
        matched_expert_alpha = expert_alpha[src_idx] # [num_matched, C, H, W]
        
        # 使用src_idx中的batch索引来选择对应的共识alpha和置信掩码
        pixel_alpha_for_matched = pixel_alpha[src_idx[0]]
        confident_mask_for_matched = confident_mask[src_idx[0]]
        
        # 4. 计算在置信区域内的L1损失
        # 我们只惩罚在置信区域内的差异
        numerator = F.l1_loss(matched_expert_alpha * confident_mask_for_matched,
                              pixel_alpha_for_matched * confident_mask_for_matched,
                              reduction='sum')
                              
        # 通过置信区域的像素总数进行归一化，以保证损失的尺度稳定
        denominator = confident_mask_for_matched.sum() + 1e-8
        
        loss_consistency = numerator / denominator
        
        return loss_consistency
    # ---

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

