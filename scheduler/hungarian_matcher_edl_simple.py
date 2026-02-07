# -*- coding: utf-8 -*-
"""
用于EDL MaskFormer的匈牙利匹配器 V3。
- 使用更贴近EDL思想的“证据错配”代价。
- 提供可配置的分类代价计算方式 (ce / evidence)。
- 各代价项可通过设置权重为0来禁用。
"""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.amp.autocast_mode import autocast
import numpy as np


# --- 代价函数 ---


def batch_dice_loss(inputs_prob: torch.Tensor, targets: torch.Tensor):
    """
    基于概率计算Dice代价。
    """
    inputs = inputs_prob.flatten(1)
    targets = targets.flatten(1)
    
    # 确保数值稳定性
    inputs = torch.clamp(inputs, min=1e-8, max=1.0)
    targets = torch.clamp(targets, min=0.0, max=1.0)
    
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    
    # 防止除零
    denominator = torch.clamp(denominator, min=1e-8)
    loss = 1 - (numerator + 1) / (denominator + 1)
    
    # 检查并修复无效值
    loss = torch.where(torch.isnan(loss) | torch.isinf(loss), torch.tensor(1.0, device=loss.device), loss)
    loss = torch.clamp(loss, min=0.0, max=1.0)
    
    return loss


def batch_evidence_misalignment_cost(alpha: torch.Tensor, beta: torch.Tensor, targets: torch.Tensor):
    """
    计算证据错配代价。
    代价 = sum(beta * y + alpha * (1-y)) / num_pixels
    """
    alpha = alpha.flatten(1)  # (num_queries, H*W)
    beta = beta.flatten(1)    # (num_queries, H*W)
    targets = targets.flatten(1) # (num_targets, H*W)
    
    # 确保数值稳定性
    alpha = torch.clamp(alpha, min=1e-8, max=1e6)
    beta = torch.clamp(beta, min=1e-8, max=1e6)
    targets = torch.clamp(targets, min=0.0, max=1.0)
    
    # y=1时，代价是beta；y=0时，代价是alpha
    # cost_matrix[q, t] = sum_{pixels} (beta_q * target_t + alpha_q * (1-target_t))
    cost = torch.einsum("nh,mh->nm", beta, targets) + torch.einsum("nh,mh->nm", alpha, (1 - targets))
    
    # 按像素数量归一化
    num_pixels = targets.shape[1]
    cost = cost / max(num_pixels, 1)  # 防除零
    
    # 检查并修复无效值
    cost = torch.where(torch.isnan(cost) | torch.isinf(cost), torch.tensor(1e6, device=cost.device), cost)
    cost = torch.clamp(cost, min=0.0, max=1e6)
    
    return cost


class HungarianMatcherEDL_V3(nn.Module):
    """
    此类使用证据错配代价和Dice代价来计算匹配。
    """

    def __init__(self, cost_class: float = 1.0, cost_dice: float = 1.0, cost_evidence: float = 1.0, cost_class_type: str = 'evidence'):
        super().__init__()
        self.cost_class = cost_class
        self.cost_dice = cost_dice
        self.cost_evidence = cost_evidence
        self.cost_class_type = cost_class_type
        assert self.cost_class_type in ['ce', 'evidence'], "cost_class_type must be 'ce' or 'evidence'"
        # 确保至少有一个代价项被启用
        assert cost_class != 0 or cost_dice != 0 or cost_evidence != 0, "all costs for matcher cannot be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        # pred_logits对于single模式是(alpha, None)，对于multi模式是(alpha, beta)
        # 我们只关心alpha部分用于分类
        pred_alpha_cls, _ = outputs["pred_logits"]
        bs, num_queries = pred_alpha_cls.shape[:2]
        indices = []

        for b in range(bs):
            cost_class = torch.tensor(0.0, device=pred_alpha_cls.device)
            if self.cost_class > 0:
                # --- 分类代价 ---
                tgt_ids = targets[b]["labels"]
                # 根据cost_class_type选择代价计算方式
                if self.cost_class_type == 'ce':
                    # 基于交叉熵的代价：使用softmax概率
                    out_prob = pred_alpha_cls[b].softmax(-1)
                elif self.cost_class_type == 'evidence':
                    # 基于证据的代价：使用Dirichlet分布的期望概率
                    evidence = F.softplus(pred_alpha_cls[b])
                    evidence = torch.clamp(evidence, min=1e-8, max=1e6)  # 数值稳定性
                    alpha = evidence + 1
                    total_alpha = torch.sum(alpha, dim=-1, keepdim=True)
                    total_alpha = torch.clamp(total_alpha, min=1e-8)  # 防除零
                    out_prob = alpha / total_alpha
                else:
                    raise ValueError("Invalid cost_class_type")
                
                # 确保概率的数值稳定性
                out_prob = torch.clamp(out_prob, min=1e-8, max=1.0)
                # 代价是负的目标概率
                cost_class = -out_prob[:, tgt_ids]
                
                # 检查并修复无效值
                cost_class = torch.where(torch.isnan(cost_class) | torch.isinf(cost_class), 
                                       torch.tensor(1.0, device=cost_class.device), cost_class)

            # --- 分割代价 ---
            pred_alpha_mask, pred_beta_mask = outputs["pred_masks"]
            out_alpha_mask_logit = pred_alpha_mask[b]
            out_beta_mask_logit = pred_beta_mask[b]
            
            # 添加数值稳定性检查
            out_alpha_mask_logit = torch.clamp(out_alpha_mask_logit, min=-10, max=10)
            out_beta_mask_logit = torch.clamp(out_beta_mask_logit, min=-10, max=10)
            
            alpha_mask = F.softplus(out_alpha_mask_logit) + 1
            beta_mask = F.softplus(out_beta_mask_logit) + 1
            
            # 确保alpha和beta在合理范围内
            alpha_mask = torch.clamp(alpha_mask, min=1e-8, max=1e6)
            beta_mask = torch.clamp(beta_mask, min=1e-8, max=1e6)
            
            # 确保目标张量在正确设备上，且数据类型一致
            tgt_mask = (targets[b]["masks"] > 0).float().to(device=alpha_mask.device, dtype=alpha_mask.dtype)

            with autocast(device_type="cuda", enabled=False):
                cost_evidence = torch.tensor(0.0, device=alpha_mask.device)
                if self.cost_evidence > 0:
                    # 1. 证据错配代价
                    cost_evidence = batch_evidence_misalignment_cost(alpha_mask, beta_mask, tgt_mask)
                
                cost_dice = torch.tensor(0.0, device=alpha_mask.device)
                if self.cost_dice > 0:
                    # 2. Dice 代价
                    total_evidence = alpha_mask + beta_mask
                    total_evidence = torch.clamp(total_evidence, min=1e-8)  # 防除零
                    prob_mask = alpha_mask / total_evidence
                    prob_mask = torch.clamp(prob_mask, min=1e-8, max=1.0)  # 确保概率范围
                    cost_dice = batch_dice_loss(prob_mask, tgt_mask)

            # --- 最终代价矩阵 ---
            # 各代价项通过权重控制其是否启用
            C = (
                self.cost_evidence * cost_evidence
                + self.cost_class * cost_class
                + self.cost_dice * cost_dice
            )
            
            # 数值稳定性检查和修复
            C = torch.where(torch.isnan(C) | torch.isinf(C), torch.tensor(1e6, device=C.device), C)
            C = torch.clamp(C, min=0.0, max=1e6)
            
            C = C.cpu()
            
            # 额外的numpy级别检查（如果仍有问题）
            C_numpy = C.numpy()
            if not np.isfinite(C_numpy).all():
                print(f"警告: 代价矩阵包含无效值，使用默认匹配")
                # 使用对角线匹配作为后备方案
                num_queries, num_targets = C.shape
                min_size = min(num_queries, num_targets)
                row_indices = np.arange(min_size)
                col_indices = np.arange(min_size)
            else:
                row_indices, col_indices = linear_sum_assignment(C_numpy)
            
            indices.append((row_indices, col_indices))

        return [
            (torch.as_tensor(i, dtype=torch.int64, device='cpu'), torch.as_tensor(j, dtype=torch.int64, device='cpu'))
            for i, j in indices
        ]
