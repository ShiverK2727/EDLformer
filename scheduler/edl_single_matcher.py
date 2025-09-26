# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import torch.nn.functional as F

def batch_dice_cost(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss

def batch_sigmoid_ce_cost(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    hw = inputs.shape[1]
    pos = F.binary_cross_entropy_with_logits(inputs, torch.ones_like(inputs), reduction="none")
    neg = F.binary_cross_entropy_with_logits(inputs, torch.zeros_like(inputs), reduction="none")

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum("nc,mc->nm", neg, (1 - targets))

    return loss / hw


class EDLHungarianMatcher(nn.Module):
    """
    [OPTIMIZED]
    This class computes an assignment between the targets and the predictions of the network.
    It is optimized to compute the cost matrix in a batch-wise manner.

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_mask: float = 1, cost_dice: float = 1):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the L1 error of the predicted masks in the matching cost
            cost_dice: This is the relative weight of the dice loss of the predicted masks in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, out_probs, out_masks, target_masks, target_labels):
        """ 
        [OPTIMIZED] Performs the matching
        
        Args:
            out_probs: (torch.Tensor) A tensor of shape [B, num_queries, num_classes] with the classification probabilities.
            out_masks: (torch.Tensor) A tensor of shape [B, num_queries, H, W] with the predicted masks.
            target_masks: (torch.Tensor) A tensor of shape [B, num_experts, H, W] containing the target masks.
            target_labels: (torch.Tensor) A tensor of shape [B, num_experts] containing the target labels.

        Returns:
            A list of size `batch_size`, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_experts)
        """
        bs, num_queries = out_probs.shape[:2]
        num_experts = target_labels.shape[1]

        # 1. Classification cost (batch-wise)
        # We flatten to compute the cost matrices in a batch
        # cost_class shape: [B, num_queries, num_experts]
        prob = out_probs.flatten(0, 1).softmax(-1)  # [B*num_queries, num_classes]
        # The target labels are repeated for each query to perform batch-wise gathering
        tgt_ids_repeated = target_labels.unsqueeze(1).repeat(1, num_queries, 1)  # [B, num_queries, num_experts]
        tgt_ids_flat = tgt_ids_repeated.flatten(0, 1)  # [B*num_queries, num_experts]
        
        # 修复：使用正确的gather操作
        # prob: [B*num_queries, num_classes], tgt_ids_flat: [B*num_queries, num_experts]
        # 我们需要为每个(query, expert)对收集对应类别的概率
        num_pairs = tgt_ids_flat.shape[0]
        num_experts_per_pair = tgt_ids_flat.shape[1]
        
        # 扩展prob以匹配expert数量维度
        prob_expanded = prob.unsqueeze(1).expand(-1, num_experts_per_pair, -1)  # [B*num_queries, num_experts, num_classes]
        
        # 使用gather收集目标类别的概率
        prob_for_target = prob_expanded.gather(2, tgt_ids_flat.unsqueeze(2)).squeeze(2)  # [B*num_queries, num_experts]
        
        # Reshape and compute final class cost
        cost_class = -prob_for_target.view(bs, num_queries, num_experts)
        
        # 2. L1 and Dice cost for masks (batch-wise)
        # Reshape for batch-wise computation
        out_masks_flat = out_masks.flatten(2) # [B, num_queries, H*W]
        tgt_masks_flat = target_masks.flatten(2) # [B, num_experts, H*W]

        # 计算L1成本矩阵 - 使用广播机制
        # out_masks_flat: [B, num_queries, H*W] -> [B, num_queries, 1, H*W]
        # tgt_masks_flat: [B, num_experts, H*W] -> [B, 1, num_experts, H*W]
        out_masks_exp = out_masks_flat.unsqueeze(2) # [B, num_queries, 1, H*W]
        tgt_masks_exp = tgt_masks_flat.unsqueeze(1) # [B, 1, num_experts, H*W]
        
        # L1 cost: 计算每对(query, expert)之间的L1距离
        # cost_mask shape: [B, num_queries, num_experts]
        cost_mask = torch.abs(out_masks_exp - tgt_masks_exp).sum(-1)  # [B, num_queries, num_experts]


        # Dice cost
        # cost_dice shape: [B, num_queries, num_experts]
        inputs = out_masks_exp.repeat(1, 1, num_experts, 1)
        targets = tgt_masks_exp.repeat(1, num_queries, 1, 1)
        numerator = 2 * (inputs * targets).sum(-1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        cost_dice = 1 - (numerator + 1) / (denominator + 1)

        # 3. Final cost matrix
        # C shape: [B, num_queries, num_experts]
        C = self.cost_mask * cost_mask + self.cost_class * cost_class + self.cost_dice * cost_dice
        
        # 4. Perform linear sum assignment for each sample in the batch
        # This part still requires a loop as scipy.optimize.linear_sum_assignment is not batched
        indices = [linear_sum_assignment(c.cpu()) for c in C]
        
        return [(torch.as_tensor(i, dtype=torch.int64, device='cpu'), torch.as_tensor(j, dtype=torch.int64, device='cpu')) for i, j in indices]
