# Multi版本的SimpleMaskFormer损失函数
# 基于现有SimpleMaskformerLoss架构，适配BN2HW数据格式

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
from logger import log_info
from .hungarian_matcher_simple import HungarianMatcher
try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


def cat(tensors: List[torch.Tensor], dim: int = 0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    num_masks = len(inputs)

    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


def sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    return loss.mean(1).sum() / len(inputs)


class SimpleMaskFormerMultiLoss(nn.Module):
    """
    SimpleMaskFormerMulti的损失函数，基于SimpleMaskformerLoss架构。
    
    关键适配：
    1. 支持BN2HW数据格式（B个样本，N个专家，2个分割类别，H×W）
    2. 基于现有SimpleMaskformerLoss的完整配置参数
    3. 支持匈牙利匹配和固定匹配模式
    4. 支持深监督和EOS处理
    
    输入格式：
    - outputs: 模型输出，包含 pred_logits [B, Q, 6] 和 pred_masks [B, Q, 2, H, W]
    - targets: 目标列表，每个元素包含 expert_masks [N, 2, H, W], expert_labels [N]
    """
    
    def __init__(self, 
                 num_classes=2,              # 分割类别数 (disc, cup)
                 num_experts=6,              # 专家数量
                 force_matching=False,       # 是否使用固定匹配
                 
                 # 匈牙利匹配器配置
                 matcher_cost_class=5.0,     # 分类代价权重
                 matcher_cost_mask=2.0,      # mask代价权重  
                 matcher_cost_dice=2.0,      # dice代价权重
                 
                 # 损失权重
                 eos_coef=1,                 # EOS系数
                 cost_weight=[5.0, 2.0, 2.0], # [cls, bce, dice]
                 non_object=True,            # 是否支持non-object
                 no_object_weight=None,      # non-object权重
                 
                 # 深监督配置
                 ds_loss_weights=None,       # 深监督权重 [1.0, 0.8, 0.6, 0.4]
                 disable_ds=False,           # 是否禁用深监督
                 ds_avg_type='v0',          # 深监督平均类型
                 cls_type=None):             # 分类类型
        super().__init__()
        
        self.num_classes = num_classes
        self.num_experts = num_experts
        self.force_matching = force_matching
        self.non_object = non_object
        self.eos_coef = eos_coef
        self.cost_weight = cost_weight if isinstance(cost_weight, list) else [cost_weight, 1.0, 1.0]
        self.no_object_weight = no_object_weight
        self.ds_loss_weights = ds_loss_weights
        self.disable_ds = disable_ds
        self.ds_avg_type = ds_avg_type
        
        # 构建匈牙利匹配器（即使在force_matching模式下也需要，因为可能有aux_outputs）
        if not force_matching:
            self.matcher = HungarianMatcher(
                cost_class=matcher_cost_class,
                cost_mask=matcher_cost_mask, 
                cost_dice=matcher_cost_dice
            )
        
        # 权重获取
        self.weight_cls = self.cost_weight[0] if len(self.cost_weight) > 0 else 5.0
        self.weight_ce = self.cost_weight[1] if len(self.cost_weight) > 1 else 2.0
        self.weight_dice = self.cost_weight[2] if len(self.cost_weight) > 2 else 2.0
        
        log_info(f"SimpleMaskFormerMultiLoss initialized:")
        log_info(f"  - Force matching: {force_matching}")
        log_info(f"  - Classes: seg={num_classes}, experts={num_experts}")
        log_info(f"  - Matcher costs: cls={matcher_cost_class}, mask={matcher_cost_mask}, dice={matcher_cost_dice}")
        log_info(f"  - Loss weights: cls={self.weight_cls}, ce={self.weight_ce}, dice={self.weight_dice}")
        log_info(f"  - Deep supervision: disabled={disable_ds}, weights={ds_loss_weights}")
        log_info(f"  - Non-object: enabled={non_object}, eos_coef={eos_coef}")

    def forward(self, outputs, targets):
        """
        This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
                      
        Multi版本特殊处理：
        - pred_masks: [B, Q, 2, H, W] - Q个queries，每个产生2类分割
        - expert_masks: [N, 2, H, W] - N个专家，每个有2类分割标注
        """
        total_loss = None
        if self.ds_loss_weights is not None:
            ds_loss_weights = self.ds_loss_weights
        else:
            len_ds = 1 + len(outputs['aux_outputs']) if isinstance(outputs, dict) and 'aux_outputs' in outputs else 1
            ds_loss_weights = [1] * len_ds

        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        loss_list = []
        
        if self.force_matching:
            loss_final = self.compute_loss_force_matching(outputs_without_aux, targets)
        else:
            loss_final = self.compute_loss_hungarian(outputs_without_aux, targets)
        
        loss_list.append(loss_final * ds_loss_weights[0])
        
        if not self.disable_ds and 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"][::-1]):  # reverse order
                if self.force_matching:
                    loss_aux = self.compute_loss_force_matching(aux_outputs, targets)
                else:
                    loss_aux = self.compute_loss_hungarian(aux_outputs, targets)
                loss_list.append(ds_loss_weights[i + 1] * loss_aux)
            
            if self.ds_avg_type == 'v0':
                total_loss = sum(loss_list) / len(loss_list)
            elif self.ds_avg_type == 'v1':
                total_loss = (loss_list[0] + sum(loss_list[1:]) / len(loss_list[1:])) / 2
            elif self.ds_avg_type == 'v2':
                total_loss = sum(loss_list)
        else:
            total_loss = loss_final
    
        return total_loss

    def compute_loss_force_matching(self, outputs, targets):
        """
        固定匹配模式：queries直接对应专家，不需要匹配算法
        
        Multi版本适配：
        - pred_masks: [B, Q, 2, H, W] 
        - expert_masks: [N, 2, H, W] - 直接按顺序匹配前N个queries
        """
        pred_logits = outputs['pred_logits']  # [B, Q, 6] - 专家分类
        pred_masks = outputs['pred_masks']    # [B, Q, 2, H, W] - 分割预测
        
        batch_size, num_queries, num_seg_classes, H, W = pred_masks.shape
        total_loss = 0.0
        
        for b in range(batch_size):
            target = targets[b]
            expert_masks = target['expert_masks']    # [N, 2, H, W]
            expert_labels = target['expert_labels']  # [N] - 专家ID (0-5)
            
            num_experts = expert_masks.shape[0]
            
            # 取前N个queries直接匹配
            pred_masks_matched = pred_masks[b][:num_experts]      # [N, 2, H, W]
            pred_logits_matched = pred_logits[b][:num_experts]    # [N, 6]
            
            # 分割损失：对每个专家的2类分割分别计算
            mask_loss = 0.0
            dice_loss_val = 0.0
            
            for expert_idx in range(num_experts):
                for seg_class in range(num_seg_classes):  # disc, cup
                    pred_mask_cls = pred_masks_matched[expert_idx, seg_class]  # [H, W]
                    target_mask_cls = expert_masks[expert_idx, seg_class]      # [H, W]
                    
                    # CE损失
                    mask_loss += sigmoid_ce_loss(
                        pred_mask_cls.unsqueeze(0).unsqueeze(0),  # [1, 1, H, W]
                        target_mask_cls.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                    )
                    
                    # Dice损失
                    dice_loss_val += dice_loss(
                        pred_mask_cls.unsqueeze(0).unsqueeze(0),  # [1, 1, H, W]
                        target_mask_cls.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                    )
            
            # 分类损失：专家ID分类
            cls_targets = F.one_hot(expert_labels, num_classes=self.num_experts + (1 if self.non_object else 0)).float()
            if not self.non_object:
                cls_targets = cls_targets[:, :-1]  # 移除non-object类
            
            cls_loss = F.cross_entropy(pred_logits_matched, cls_targets.argmax(dim=-1))
            
            # 合并损失
            sample_loss = (self.weight_cls * cls_loss + 
                          self.weight_ce * mask_loss + 
                          self.weight_dice * dice_loss_val)
            total_loss += sample_loss
        
        return total_loss / batch_size

    def compute_loss_hungarian(self, outputs, targets):
        """
        匈牙利匹配模式：使用匈牙利算法进行最优匹配
        
        Multi版本适配：
        - 计算query与expert之间的匹配代价矩阵
        - 对每个分割类别分别计算代价
        """
        if linear_sum_assignment is None:
            raise ImportError("Hungarian matching requires scipy. Please install: pip install scipy")
            
        pred_logits = outputs['pred_logits']  # [B, Q, 6] 
        pred_masks = outputs['pred_masks']    # [B, Q, 2, H, W]
        
        batch_size, num_queries, num_seg_classes, H, W = pred_masks.shape
        total_loss = 0.0
        
        for b in range(batch_size):
            target = targets[b]
            expert_masks = target['expert_masks']    # [N, 2, H, W]
            expert_labels = target['expert_labels']  # [N]
            
            num_experts = expert_masks.shape[0]
            
            # 计算代价矩阵
            cost_matrix = self._compute_cost_matrix(
                pred_logits[b], pred_masks[b], expert_masks, expert_labels
            )  # [Q, N]
            
            # 匈牙利匹配 - 使用detach()切断梯度连接
            query_indices, expert_indices = linear_sum_assignment(cost_matrix.detach().cpu().numpy())
            
            # 计算匹配后的损失
            matched_loss = 0.0
            
            for i, (q_idx, e_idx) in enumerate(zip(query_indices, expert_indices)):
                if e_idx >= num_experts:  # 跳过无效匹配
                    continue
                    
                # 分割损失
                pred_mask_matched = pred_masks[b, q_idx]      # [2, H, W]
                target_mask_matched = expert_masks[e_idx]     # [2, H, W]
                
                for seg_class in range(num_seg_classes):
                    pred_mask_cls = pred_mask_matched[seg_class]    # [H, W]
                    target_mask_cls = target_mask_matched[seg_class] # [H, W]
                    
                    # CE损失
                    matched_loss += self.weight_ce * sigmoid_ce_loss(
                        pred_mask_cls.unsqueeze(0).unsqueeze(0),
                        target_mask_cls.unsqueeze(0).unsqueeze(0)
                    )
                    
                    # Dice损失
                    matched_loss += self.weight_dice * dice_loss(
                        pred_mask_cls.unsqueeze(0).unsqueeze(0),
                        target_mask_cls.unsqueeze(0).unsqueeze(0)
                    )
                
                # 分类损失
                pred_logit_matched = pred_logits[b, q_idx]   # [6]
                target_label_matched = expert_labels[e_idx]   # scalar
                
                cls_loss = F.cross_entropy(
                    pred_logit_matched.unsqueeze(0), 
                    target_label_matched.unsqueeze(0)
                )
                matched_loss += self.weight_cls * cls_loss
            
            total_loss += matched_loss
        
        return total_loss / batch_size
    
    def _compute_cost_matrix(self, pred_logits, pred_masks, expert_masks, expert_labels):
        """
        计算query与expert之间的代价矩阵
        
        Args:
            pred_logits: [Q, 6] - 预测的专家分类
            pred_masks: [Q, 2, H, W] - 预测的分割masks  
            expert_masks: [N, 2, H, W] - 目标expert masks
            expert_labels: [N] - 目标expert标签
            
        Returns:
            cost_matrix: [Q, N] - 代价矩阵
        """
        num_queries = pred_logits.shape[0]
        num_experts = expert_masks.shape[0]
        
        cost_matrix = torch.zeros(num_queries, num_experts, device=pred_logits.device)
        
        for q in range(num_queries):
            for e in range(num_experts):
                # 分类代价
                pred_cls_prob = F.softmax(pred_logits[q], dim=-1)
                target_cls = expert_labels[e]
                cls_cost = -pred_cls_prob[target_cls]  # 负对数似然
                
                # 分割代价 (所有类别的平均)
                mask_cost = 0.0
                dice_cost = 0.0
                
                for seg_class in range(pred_masks.shape[1]):  # 2个分割类别
                    pred_mask = pred_masks[q, seg_class]      # [H, W]
                    target_mask = expert_masks[e, seg_class]  # [H, W]
                    
                    # BCE代价 - 使用autocast安全的版本
                    pred_logits_flat = pred_mask.flatten()
                    target_flat = target_mask.flatten()
                    mask_cost += F.binary_cross_entropy_with_logits(pred_logits_flat, target_flat)
                    
                    # Dice代价 - 使用sigmoid后的概率
                    pred_flat = pred_mask.sigmoid().flatten()
                    numerator = 2 * (pred_flat * target_flat).sum()
                    denominator = pred_flat.sum() + target_flat.sum()
                    dice_cost += 1 - (numerator + 1) / (denominator + 1)
                
                # 综合代价
                total_cost = (self.matcher.cost_class * cls_cost + 
                             self.matcher.cost_mask * mask_cost + 
                             self.matcher.cost_dice * dice_cost)
                cost_matrix[q, e] = total_cost
        
        return cost_matrix


# 为了向后兼容，保留简化的接口
class SimpleMaskFormerMultiLossFixed(SimpleMaskFormerMultiLoss):
    """固定匹配版本的Multi损失函数"""
    def __init__(self, **kwargs):
        kwargs['force_matching'] = True
        super().__init__(**kwargs)


class SimpleMaskFormerMultiLossHungarian(SimpleMaskFormerMultiLoss):
    """匈牙利匹配版本的Multi损失函数"""
    def __init__(self, **kwargs):
        kwargs['force_matching'] = False
        super().__init__(**kwargs)