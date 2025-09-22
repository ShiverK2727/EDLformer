"""
Multi-Expert Matcher for Multi-Annotator Segmentation

This module implements a one-to-many matching strategy for multi-expert annotations,
where each query can be matched to multiple similar expert annotations.
"""

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.amp.autocast_mode import autocast
from typing import List, Dict, Tuple, Optional
import numpy as np


def batch_dice_loss_full(inputs: torch.Tensor, targets: torch.Tensor):
    """
    计算完整的DICE损失 (不使用点采样)
    Args:
        inputs: [N, H*W] 预测掩码
        targets: [M, H*W] 目标掩码
    Returns:
        cost_matrix: [N, M] 成本矩阵
    """
    inputs = inputs.sigmoid()
    # 计算交集和并集
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


def batch_sigmoid_ce_loss_full(inputs: torch.Tensor, targets: torch.Tensor):
    """
    计算完整的BCE损失 (不使用点采样)
    Args:
        inputs: [N, H*W] 预测掩码logits
        targets: [M, H*W] 目标掩码
    Returns:
        cost_matrix: [N, M] 成本矩阵
    """
    hw = inputs.shape[1]
    
    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )
    
    # 计算成本矩阵
    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )
    
    return loss / hw


def compute_expert_similarity(expert_masks: torch.Tensor, 
                            similarity_threshold: float = 0.7) -> torch.Tensor:
    """
    计算专家标注之间的相似性
    Args:
        expert_masks: [num_experts, H*W] 专家标注掩码
        similarity_threshold: 相似性阈值
    Returns:
        similarity_matrix: [num_experts, num_experts] 相似性矩阵
    """
    # 计算IoU矩阵
    intersection = torch.einsum("ec,fc->ef", expert_masks, expert_masks)
    union = expert_masks.sum(-1)[:, None] + expert_masks.sum(-1)[None, :] - intersection
    iou_matrix = intersection / (union + 1e-6)
    
    # 基于阈值生成相似性矩阵
    similarity_matrix = (iou_matrix > similarity_threshold).float()
    
    return similarity_matrix


class MultiExpertMatcher(nn.Module):
    """
    多专家一对多匹配器
    
    核心思想:
    1. 允许一个query匹配多个相似的专家标注
    2. 使用全图计算，不进行点采样
    3. 基于专家标注相似性进行分组
    4. 最小化总体匹配成本
    """
    
    def __init__(self, 
                 cost_class: float = 1.0,
                 cost_mask: float = 1.0, 
                 cost_dice: float = 1.0,
                 similarity_threshold: float = 0.7,
                 max_matches_per_query: int = 3,
                 expert_weight_strategy: str = 'uniform'):
        """
        Args:
            cost_class: 分类成本权重
            
            cost_mask: 掩码BCE成本权重
            cost_dice: Dice成本权重
            similarity_threshold: 专家相似性阈值
            max_matches_per_query: 每个query最大匹配数
            expert_weight_strategy: 专家权重策略 ('uniform', 'quality', 'confidence')
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.similarity_threshold = similarity_threshold
        self.max_matches_per_query = max_matches_per_query
        self.expert_weight_strategy = expert_weight_strategy
        
        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, "所有成本不能都为0"
    
    def compute_expert_groups(self, expert_masks: torch.Tensor) -> List[List[int]]:
        """
        基于相似性将专家标注分组
        Args:
            expert_masks: [num_experts, H*W] 专家掩码
        Returns:
            expert_groups: 专家分组列表，每组包含相似的专家索引
        """
        num_experts = expert_masks.shape[0]
        similarity_matrix = compute_expert_similarity(expert_masks, self.similarity_threshold)
        
        # 使用连通分量进行分组
        visited = [False] * num_experts
        expert_groups = []
        
        def dfs(expert_idx, current_group):
            visited[expert_idx] = True
            current_group.append(expert_idx)
            
            for neighbor_idx in range(num_experts):
                if not visited[neighbor_idx] and similarity_matrix[expert_idx, neighbor_idx] > 0:
                    dfs(neighbor_idx, current_group)
        
        for expert_idx in range(num_experts):
            if not visited[expert_idx]:
                current_group = []
                dfs(expert_idx, current_group)
                expert_groups.append(current_group)
        
        return expert_groups
    
    def compute_group_representative(self, 
                                   expert_masks: torch.Tensor,
                                   expert_labels: torch.Tensor,
                                   group_indices: List[int]) -> Tuple[torch.Tensor, int]:
        """
        计算专家组的代表性掩码和标签
        Args:
            expert_masks: [num_experts, H*W] 专家掩码
            expert_labels: [num_experts] 专家标签
            group_indices: 组内专家索引列表
        Returns:
            representative_mask: [H*W] 代表性掩码
            representative_label: 代表性标签
        """
        group_masks = expert_masks[group_indices]  # [group_size, H*W]
        group_labels = expert_labels[group_indices]  # [group_size]
        
        if self.expert_weight_strategy == 'uniform':
            # 简单平均
            representative_mask = group_masks.mean(dim=0)
        elif self.expert_weight_strategy == 'quality':
            # 基于掩码质量加权 (可以基于掩码的清晰度等指标)
            mask_quality = group_masks.sum(dim=-1)  # 简单的质量指标
            weights = F.softmax(mask_quality, dim=0)
            representative_mask = torch.einsum('g,gh->h', weights, group_masks)
        else:
            # 默认uniform
            representative_mask = group_masks.mean(dim=0)
        
        # 标签使用众数
        representative_label = torch.mode(group_labels)[0].item()
        
        return representative_mask, representative_label
    
    def compute_one_to_many_assignment(self, 
                                     cost_matrix: torch.Tensor,
                                     expert_groups: List[List[int]]) -> List[Tuple[int, List[int]]]:
        """
        计算一对多分配
        Args:
            cost_matrix: [num_queries, num_expert_groups] 成本矩阵
            expert_groups: 专家分组
        Returns:
            assignments: [(query_idx, [expert_indices])...] 分配结果
        """
        num_queries, num_groups = cost_matrix.shape
        
        # 使用贪心策略进行一对多匹配
        assignments = []
        used_groups = set()
        
        # 按成本排序所有可能的匹配
        query_group_costs = []
        for q in range(num_queries):
            for g in range(num_groups):
                if g not in used_groups:
                    query_group_costs.append((cost_matrix[q, g].item(), q, g))
        
        query_group_costs.sort()  # 按成本升序排列
        
        query_matches = {q: [] for q in range(num_queries)}
        
        # 贪心分配
        for cost, query_idx, group_idx in query_group_costs:
            if len(query_matches[query_idx]) < self.max_matches_per_query and group_idx not in used_groups:
                query_matches[query_idx].append(group_idx)
                used_groups.add(group_idx)
        
        # 转换为最终格式
        for query_idx, group_indices in query_matches.items():
            if group_indices:  # 只包含有匹配的query
                expert_indices = []
                for group_idx in group_indices:
                    expert_indices.extend(expert_groups[group_idx])
                assignments.append((query_idx, expert_indices))
        
        return assignments
    
    @torch.no_grad()
    def forward(self, outputs: Dict[str, torch.Tensor], targets: List[Dict]) -> List[List[Tuple[int, List[int]]]]:
        """
        执行多专家匹配
        Args:
            outputs: 模型输出
                - "pred_logits": [batch_size, num_queries, num_classes] 分类logits
                - "pred_masks": [batch_size, num_queries, H, W] 预测掩码
            targets: 目标列表，每个目标包含多个专家标注
                - "labels": [num_experts] 专家标签
                - "masks": [num_experts, H, W] 专家掩码
        Returns:
            batch_assignments: 每个样本的分配结果 [[(query_idx, [expert_indices])...], ...]
        """
        batch_size, num_queries = outputs["pred_logits"].shape[:2]
        batch_assignments = []
        
        for b in range(batch_size):
            # 提取当前样本的预测和目标
            pred_logits = outputs["pred_logits"][b]  # [num_queries, num_classes]
            pred_masks = outputs["pred_masks"][b]    # [num_queries, H, W]
            
            expert_labels = targets[b]["labels"]     # [num_experts]
            expert_masks = targets[b]["masks"]       # [num_experts, H, W]
            
            # 检查是否有专家标注
            if len(expert_labels) == 0:
                # 没有专家标注，返回空分配
                batch_assignments.append([])
                continue
            
            # 展平掩码用于计算
            pred_masks_flat = pred_masks.reshape(num_queries, -1)     # [num_queries, H*W]
            expert_masks_flat = expert_masks.reshape(len(expert_labels), -1)  # [num_experts, H*W]
            
            # 步骤1: 将专家标注分组
            expert_groups = self.compute_expert_groups(expert_masks_flat)
            
            # 步骤2: 为每组计算代表性掩码和标签
            group_representatives = []
            group_labels = []
            
            for group_indices in expert_groups:
                repr_mask, repr_label = self.compute_group_representative(
                    expert_masks_flat, expert_labels, group_indices
                )
                group_representatives.append(repr_mask)
                group_labels.append(repr_label)
            
            if not group_representatives:
                # 如果没有专家组，返回空分配
                batch_assignments.append([])
                continue
            
            # 转换为张量
            group_masks = torch.stack(group_representatives)  # [num_groups, H*W]
            group_labels_tensor = torch.tensor(group_labels, device=pred_logits.device)
            
            # 步骤3: 计算成本矩阵
            pred_probs = pred_logits.softmax(-1)  # [num_queries, num_classes]
            
            # 分类成本
            cost_class = -pred_probs[:, group_labels_tensor]  # [num_queries, num_groups]
            
            # 掩码成本 (使用完整图像)
            with autocast(device_type="cuda", enabled=False):
                pred_masks_float = pred_masks_flat.float()
                group_masks_float = group_masks.float()
                
                cost_mask = batch_sigmoid_ce_loss_full(pred_masks_float, group_masks_float)
                cost_dice = batch_dice_loss_full(pred_masks_float, group_masks_float)
            
            # 总成本
            total_cost = (
                self.cost_class * cost_class +
                self.cost_mask * cost_mask +
                self.cost_dice * cost_dice
            )  # [num_queries, num_groups]
            
            # 步骤4: 执行一对多分配
            assignments = self.compute_one_to_many_assignment(total_cost, expert_groups)
            batch_assignments.append(assignments)
        
        return batch_assignments
    
    def __repr__(self, _repr_indent=4):
        head = "MultiExpertMatcher " + self.__class__.__name__
        body = [
            "cost_class: {}".format(self.cost_class),
            "cost_mask: {}".format(self.cost_mask),
            "cost_dice: {}".format(self.cost_dice),
            "similarity_threshold: {}".format(self.similarity_threshold),
            "max_matches_per_query: {}".format(self.max_matches_per_query),
            "expert_weight_strategy: {}".format(self.expert_weight_strategy),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)


class AdaptiveMultiExpertMatcher(MultiExpertMatcher):
    """
    自适应多专家匹配器
    
    增强功能:
    1. 动态调整相似性阈值
    2. 考虑专家置信度
    3. 支持软匹配权重
    """
    
    def __init__(self, 
                 cost_class: float = 1.0,
                 cost_mask: float = 1.0,
                 cost_dice: float = 1.0,
                 initial_similarity_threshold: float = 0.7,
                 adaptive_threshold: bool = True,
                 confidence_weight: float = 0.1,
                 soft_matching: bool = True):
        """
        Args:
            initial_similarity_threshold: 初始相似性阈值
            adaptive_threshold: 是否自适应调整阈值
            confidence_weight: 置信度权重
            soft_matching: 是否使用软匹配
        """
        super().__init__(
            cost_class=cost_class,
            cost_mask=cost_mask, 
            cost_dice=cost_dice,
            similarity_threshold=initial_similarity_threshold,
            expert_weight_strategy='confidence'
        )
        
        self.adaptive_threshold = adaptive_threshold
        self.confidence_weight = confidence_weight
        self.soft_matching = soft_matching
        self.threshold_history = []
    
    def estimate_expert_confidence(self, expert_masks: torch.Tensor) -> torch.Tensor:
        """
        估计专家标注的置信度
        Args:
            expert_masks: [num_experts, H*W] 专家掩码
        Returns:
            confidences: [num_experts] 置信度分数
        """
        # 基于掩码的清晰度和一致性估计置信度
        
        # 1. 掩码清晰度 (接近0或1的程度)
        clarity = 1 - 4 * expert_masks * (1 - expert_masks)  # 对于0和1值，这个值接近1
        clarity_score = clarity.mean(dim=-1)
        
        # 2. 掩码完整性 (非空程度)
        completeness = expert_masks.sum(dim=-1) / expert_masks.shape[-1]
        
        # 3. 边界清晰度 (简化版本)
        # 可以计算梯度幅值等更复杂的指标
        
        # 组合置信度分数
        confidence = 0.6 * clarity_score + 0.4 * completeness
        
        return confidence
    
    def adaptive_similarity_threshold(self, expert_masks: torch.Tensor) -> float:
        """
        自适应调整相似性阈值
        Args:
            expert_masks: [num_experts, H*W] 专家掩码
        Returns:
            adjusted_threshold: 调整后的阈值
        """
        if not self.adaptive_threshold:
            return self.similarity_threshold
        
        # 计算专家间的平均IoU
        intersection = torch.einsum("ec,fc->ef", expert_masks, expert_masks)
        union = expert_masks.sum(-1)[:, None] + expert_masks.sum(-1)[None, :] - intersection
        iou_matrix = intersection / (union + 1e-6)
        
        # 移除对角线元素
        mask = ~torch.eye(iou_matrix.shape[0], dtype=torch.bool, device=iou_matrix.device)
        mean_iou = iou_matrix[mask].mean().item()
        
        # 基于平均IoU调整阈值
        if mean_iou > 0.8:
            adjusted_threshold = min(0.9, self.similarity_threshold + 0.1)
        elif mean_iou < 0.4:
            adjusted_threshold = max(0.3, self.similarity_threshold - 0.1)
        else:
            adjusted_threshold = self.similarity_threshold
        
        self.threshold_history.append(adjusted_threshold)
        return adjusted_threshold
    
    @torch.no_grad()
    def forward(self, outputs: Dict[str, torch.Tensor], targets: List[Dict]) -> List[List[Tuple[int, List[int]]]]:
        """重写forward方法以包含自适应特性"""
        batch_size, num_queries = outputs["pred_logits"].shape[:2]
        batch_assignments = []
        
        for b in range(batch_size):
            pred_logits = outputs["pred_logits"][b]
            pred_masks = outputs["pred_masks"][b]
            expert_labels = targets[b]["labels"]
            expert_masks = targets[b]["masks"]
            
            # 检查是否有专家标注
            if len(expert_labels) == 0:
                batch_assignments.append([])
                continue
            
            pred_masks_flat = pred_masks.reshape(num_queries, -1)
            expert_masks_flat = expert_masks.reshape(len(expert_labels), -1)
            
            # 自适应阈值调整
            current_threshold = self.adaptive_similarity_threshold(expert_masks_flat)
            
            # 估计专家置信度
            expert_confidences = self.estimate_expert_confidence(expert_masks_flat)
            
            # 使用当前阈值进行分组
            self.similarity_threshold = current_threshold
            expert_groups = self.compute_expert_groups(expert_masks_flat)
            
            # 计算组代表（考虑置信度）
            group_representatives = []
            group_labels = []
            group_confidences = []
            
            for group_indices in expert_groups:
                group_masks = expert_masks_flat[group_indices]
                group_labels_subset = expert_labels[group_indices]
                group_conf = expert_confidences[group_indices]
                
                # 基于置信度的加权平均
                if self.soft_matching:
                    weights = F.softmax(group_conf, dim=0)
                    repr_mask = torch.einsum('g,gh->h', weights, group_masks)
                else:
                    repr_mask = group_masks.mean(dim=0)
                
                repr_label = torch.mode(group_labels_subset)[0].item()
                group_conf_mean = group_conf.mean().item()
                
                group_representatives.append(repr_mask)
                group_labels.append(repr_label)
                group_confidences.append(group_conf_mean)
            
            if not group_representatives:
                batch_assignments.append([])
                continue
            
            group_masks = torch.stack(group_representatives)
            group_labels_tensor = torch.tensor(group_labels, device=pred_logits.device)
            group_conf_tensor = torch.tensor(group_confidences, device=pred_logits.device)
            
            # 计算成本（考虑置信度）
            pred_probs = pred_logits.softmax(-1)
            cost_class = -pred_probs[:, group_labels_tensor]
            
            with autocast(device_type="cuda", enabled=False):
                pred_masks_float = pred_masks_flat.float()
                group_masks_float = group_masks.float()
                cost_mask = batch_sigmoid_ce_loss_full(pred_masks_float, group_masks_float)
                cost_dice = batch_dice_loss_full(pred_masks_float, group_masks_float)
            
            # 置信度调整成本
            confidence_bonus = self.confidence_weight * group_conf_tensor[None, :]  # [1, num_groups]
            
            total_cost = (
                self.cost_class * cost_class +
                self.cost_mask * cost_mask +
                self.cost_dice * cost_dice -
                confidence_bonus
            )
            
            assignments = self.compute_one_to_many_assignment(total_cost, expert_groups)
            batch_assignments.append(assignments)
        
        return batch_assignments


def create_multi_expert_matcher(matcher_type: str = 'basic', **kwargs) -> MultiExpertMatcher:
    """
    创建多专家匹配器的工厂函数
    Args:
        matcher_type: 'basic' 或 'adaptive'
        **kwargs: 匹配器参数
    Returns:
        matcher: 多专家匹配器实例
    """
    if matcher_type == 'basic':
        return MultiExpertMatcher(**kwargs)
    elif matcher_type == 'adaptive':
        return AdaptiveMultiExpertMatcher(**kwargs)
    else:
        raise ValueError(f"未知的匹配器类型: {matcher_type}")
