import torch
import torch.nn.functional as F
from torch import nn
from typing import Dict, Tuple, List, Literal

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
    # 确保张量在计算前是连续的
    alpha_flat = alpha.permute(0, 2, 3, 1).contiguous().view(-1, num_classes)
    y_one_hot_flat = y_one_hot.permute(0, 2, 3, 1).contiguous().view(-1, num_classes)
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

def compute_bce_loss(pred_logits: torch.Tensor, targets_one_hot: torch.Tensor):
    return F.binary_cross_entropy_with_logits(pred_logits, targets_one_hot, reduction='mean')

# ==============================================================================
# 全新设计的、功能更灵活的主损失类
# ==============================================================================
class FlexibleEDLSingleLoss(nn.Module):
    """
    一个灵活的、可配置的损失模块，集成了多种新功能：
    1. 可选的匹配策略：支持匈牙利匹配或固定的顺序匹配。
    2. 可切换的损失函数：
        - 分类头：可选择 'edl' (证据学习) 或 'ce' (标准交叉熵)。
        - 分割头：可选择 'edl' (证据学习+Dice) 或 'bce_dice' (BCE+Dice)。
    3. 保留了原有的专家、共识和一致性损失结构。
    """
    def __init__(
        self,
        # --- [新增] 核心功能开关 ---
        use_matcher: bool = True,
        cls_loss_type: Literal['edl', 'ce'] = 'edl',
        mask_loss_type: Literal['edl', 'bce_dice'] = 'edl',
        # --- 匹配器参数 (仅当 use_matcher=True 时有效) ---
        matcher_cost_class: float = 2.0,
        matcher_cost_mask: float = 5.0,
        matcher_cost_dice: float = 5.0,
        # --- 损失权重 ---
        loss_weight_cls: float = 2.0,
        loss_weight_mask: float = 5.0, # 在 bce_dice 模式下，这是 BCE 的权重
        loss_weight_dice: float = 5.0,
        loss_weight_consensus_mask: float = 1.0,
        loss_weight_consensus_dice: float = 1.0,
        loss_weight_consistency: float = 0.5,
        # --- EDL 相关参数 ---
        kl_annealing_step: int = 10,
        dice_on_evidence: bool = False,
        # --- 一致性损失参数 ---
        consistency_uncertainty_threshold: float = 0.2,
        **kwargs
    ):
        super().__init__()
        
        # 保存所有开关和参数
        self.use_matcher = use_matcher
        self.cls_loss_type = cls_loss_type
        self.mask_loss_type = mask_loss_type
        self.dice_on_evidence = dice_on_evidence

        if self.use_matcher:
            self.matcher = EDLHungarianMatcher(
                cost_class=matcher_cost_class, cost_mask=matcher_cost_mask, cost_dice=matcher_cost_dice
            )
        else:
            self.matcher = None
        
        # 保存所有权重
        self.loss_weight_cls = loss_weight_cls
        self.loss_weight_mask = loss_weight_mask
        self.loss_weight_dice = loss_weight_dice
        self.loss_weight_consensus_mask = loss_weight_consensus_mask
        self.loss_weight_consensus_dice = loss_weight_consensus_dice
        self.loss_weight_consistency = loss_weight_consistency
        self.kl_annealing_step = kl_annealing_step
        self.consistency_uncertainty_threshold = consistency_uncertainty_threshold
        
        # 初始化交叉熵损失（如果需要）
        if self.cls_loss_type == 'ce':
            self.ce_loss_fn = nn.CrossEntropyLoss(reduction='mean')

        log_info(f"FlexibleEDLSingleLoss initialized with settings:", print_message=True)
        log_info(f"  - Use Matcher: {self.use_matcher}", print_message=True)
        log_info(f"  - Classification Loss: '{self.cls_loss_type}'", print_message=True)
        log_info(f"  - Segmentation Loss: '{self.mask_loss_type}'", print_message=True)

    def forward(
        self,
        # --- 原始 logits (用于匹配和传统损失) ---
        raw_cls_logits: torch.Tensor,
        raw_mask_logits: torch.Tensor,
        # --- EDL 处理后的 alpha (用于EDL损失和一致性) ---
        expert_alpha: torch.Tensor,
        cls_alpha: torch.Tensor,
        pixel_alpha: torch.Tensor,
        pixel_logits: torch.Tensor, # 共识部分的原始 logits
        # --- 真值 ---
        target_masks: torch.Tensor,
        target_labels: torch.Tensor,
        consensus_target: torch.Tensor,
        epoch: int
    ) -> Tuple[torch.Tensor, ...]:
        
        # 1. 匹配过程：根据开关选择匹配策略
        if self.use_matcher:
            indices = self.matcher(raw_cls_logits, raw_mask_logits, target_masks, target_labels)
        else:
            indices = self._get_fixed_indices(raw_cls_logits, target_labels)
        
        # 2. 专家损失
        loss_cls, loss_exp_mask, loss_exp_dice = self._compute_expert_loss(
            cls_alpha, expert_alpha, raw_cls_logits, raw_mask_logits,
            target_labels, target_masks, indices, epoch
        )
        
        # 3. 共识损失
        loss_con_mask, loss_con_dice = self._compute_consensus_loss(
            pixel_alpha, pixel_logits, consensus_target, epoch
        )

        # 4. 一致性监督损失 (逻辑保持不变)
        loss_consistency = self._compute_consistency_loss(
            expert_alpha, pixel_alpha, indices
        )

        # 5. 加权合并总损失
        total_loss = (self.loss_weight_cls * loss_cls + 
                      self.loss_weight_mask * loss_exp_mask + 
                      self.loss_weight_dice * loss_exp_dice +
                      self.loss_weight_consensus_mask * loss_con_mask +
                      self.loss_weight_consensus_dice * loss_con_dice +
                      self.loss_weight_consistency * loss_consistency)
        
        return total_loss, loss_cls, loss_exp_mask, loss_exp_dice, loss_con_mask, loss_con_dice, loss_consistency

    def _compute_expert_loss(self, cls_alpha, expert_alpha, raw_cls_logits, raw_mask_logits, target_labels, target_masks, indices, epoch):
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        
        if len(src_idx[0]) == 0:
            device = cls_alpha.device
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

        # --- 1. 分类损失 (根据开关选择) ---
        if self.cls_loss_type == 'edl':
            matched_cls_alpha = cls_alpha[src_idx]
            matched_cls_labels = target_labels[tgt_idx]
            cls_y_one_hot = F.one_hot(matched_cls_labels, num_classes=matched_cls_alpha.shape[-1]).float()
            loss_cls = edl_digamma_ace_loss(matched_cls_alpha, cls_y_one_hot)
        elif self.cls_loss_type == 'ce':
            matched_cls_logits = raw_cls_logits[src_idx]
            matched_cls_labels = target_labels[tgt_idx]
            loss_cls = self.ce_loss_fn(matched_cls_logits, matched_cls_labels)
        else:
            raise ValueError(f"Unknown cls_loss_type: {self.cls_loss_type}")

        # --- 2. 分割损失 (根据开关选择) ---
        target_masks_one_hot = target_masks[tgt_idx]
        
        if self.mask_loss_type == 'edl':
            matched_expert_alpha = expert_alpha[src_idx]
            # EDL-ACE + KL
            loss_mask_ace = edl_digamma_ace_loss(matched_expert_alpha, target_masks_one_hot)
            loss_mask_kl = edl_kl_regularizer(matched_expert_alpha, target_masks_one_hot, epoch, self.kl_annealing_step)
            loss_expert_mask = loss_mask_ace + loss_mask_kl
            
            # Dice on EDL evidence or raw logits
            if self.dice_on_evidence:
                S = torch.sum(matched_expert_alpha, dim=1, keepdim=True)
                dice_input = matched_expert_alpha / S
            else:
                matched_mask_logits = raw_mask_logits[src_idx]
                dice_input = F.softmax(matched_mask_logits, dim=1)
            loss_expert_dice = compute_soft_dice_loss(dice_input, target_masks_one_hot)

        elif self.mask_loss_type == 'bce_dice':
            matched_mask_logits = raw_mask_logits[src_idx]
            # loss_expert_mask in this mode is BCE loss
            loss_expert_mask = compute_bce_loss(matched_mask_logits, target_masks_one_hot)
            # loss_expert_dice is standard Dice loss
            dice_input = torch.sigmoid(matched_mask_logits)
            loss_expert_dice = compute_soft_dice_loss(dice_input, target_masks_one_hot)
        else:
            raise ValueError(f"Unknown mask_loss_type: {self.mask_loss_type}")
        
        return loss_cls, loss_expert_mask, loss_expert_dice

    def _compute_consensus_loss(self, pixel_alpha, pixel_logits, consensus_target, epoch):
        if (self.loss_weight_consensus_mask <= 0 and self.loss_weight_consensus_dice <= 0):
            device = consensus_target.device
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
        
        # 在多分类场景下，共识目标通常已经是 one-hot 格式
        consensus_target_one_hot = consensus_target.float()
        
        if self.mask_loss_type == 'edl':
            # 证据学习部分 (ACE + KL)
            loss_con_mask = torch.tensor(0.0, device=consensus_target.device)
            if self.loss_weight_consensus_mask > 0 and pixel_alpha is not None:
                loss_mask_ace = edl_digamma_ace_loss(pixel_alpha, consensus_target_one_hot)
                loss_mask_kl = edl_kl_regularizer(pixel_alpha, consensus_target_one_hot, epoch, self.kl_annealing_step)
                loss_con_mask = loss_mask_ace + loss_mask_kl
            
            # Dice损失部分
            loss_con_dice = torch.tensor(0.0, device=consensus_target.device)
            if self.loss_weight_consensus_dice > 0 and pixel_alpha is not None:
                if self.dice_on_evidence:
                    S = torch.sum(pixel_alpha, dim=1, keepdim=True)
                    dice_input = pixel_alpha / S
                else:
                    dice_input = F.softmax(pixel_logits, dim=1)
                loss_con_dice = compute_soft_dice_loss(dice_input, consensus_target_one_hot)

        elif self.mask_loss_type == 'bce_dice':
            # BCE部分
            loss_con_mask = torch.tensor(0.0, device=consensus_target.device)
            if self.loss_weight_consensus_mask > 0 and pixel_logits is not None:
                loss_con_mask = compute_bce_loss(pixel_logits, consensus_target_one_hot)
            
            # Dice部分
            loss_con_dice = torch.tensor(0.0, device=consensus_target.device)
            if self.loss_weight_consensus_dice > 0 and pixel_logits is not None:
                dice_input = torch.sigmoid(pixel_logits)
                loss_con_dice = compute_soft_dice_loss(dice_input, consensus_target_one_hot)
        else:
            raise ValueError(f"Unknown mask_loss_type: {self.mask_loss_type}")
            
        return loss_con_mask, loss_con_dice

    def _compute_consistency_loss(self, expert_alpha, pixel_alpha, indices):
        if self.loss_weight_consistency <= 0 or pixel_alpha is None:
            return torch.tensor(0.0, device=expert_alpha.device)
            
        num_classes = pixel_alpha.shape[1]
        S_pixel = torch.sum(pixel_alpha, dim=1, keepdim=True)
        U_pixel = num_classes / (S_pixel + 1e-8)
        confident_mask = (U_pixel < self.consistency_uncertainty_threshold).float()
        
        if confident_mask.sum() == 0:
            return torch.tensor(0.0, device=expert_alpha.device)

        src_idx = self._get_src_permutation_idx(indices)
        if len(src_idx[0]) == 0:
            return torch.tensor(0.0, device=expert_alpha.device)
            
        matched_expert_alpha = expert_alpha[src_idx]
        pixel_alpha_for_matched = pixel_alpha[src_idx[0]]
        confident_mask_for_matched = confident_mask[src_idx[0]]
        
        numerator = F.l1_loss(matched_expert_alpha * confident_mask_for_matched,
                                pixel_alpha_for_matched * confident_mask_for_matched,
                                reduction='sum')
                                
        denominator = confident_mask_for_matched.sum() + 1e-8
        return numerator / denominator

    @staticmethod
    def _get_fixed_indices(preds: torch.Tensor, targets: List[torch.Tensor]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        生成固定的、按顺序的匹配索引。
        假设 preds 的数量 (preds.shape[1]) 大于或等于每个 batch item 的 target 数量。
        """
        indices = []
        device = preds.device
        for i, t in enumerate(targets):
            num_targets = len(t)
            if num_targets == 0:
                src = torch.tensor([], dtype=torch.long, device=device)
                tgt = torch.tensor([], dtype=torch.long, device=device)
            else:
                # 将前 num_targets 个预测与 num_targets 个真值按顺序匹配
                src = torch.arange(num_targets, device=device)
                tgt = torch.arange(num_targets, device=device)
            indices.append((src, tgt))
        return indices

    @staticmethod
    def _get_src_permutation_idx(indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    @staticmethod
    def _get_tgt_permutation_idx(indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx
