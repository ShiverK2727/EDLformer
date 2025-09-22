# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/matcher.py
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.amp.autocast_mode import autocast
from logger import log_info
import numpy as np


def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
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
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
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

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )

    return loss / hw


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_mask: float = 1, cost_dice: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, "all costs cant be 0"

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        """More memory-friendly matching"""
        bs, num_queries = outputs["pred_logits"].shape[:2]

        indices = []

        # Iterate through batch size
        for b in range(bs):

            out_prob = outputs["pred_logits"][b].softmax(-1)  # [num_queries, num_classes]
            tgt_ids = targets[b]["labels"]

            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -out_prob[:, tgt_ids]
            
            # 添加分类成本日志
            if b == 0:  # 只记录第一个batch的信息以避免过多日志
                log_info(f"[Hungarian Matcher] Batch {b}: queries={num_queries}, targets={len(tgt_ids)}")
                log_info(f"[Hungarian Matcher] Target IDs: {tgt_ids.tolist()[:10]}...")  # 只显示前10个
                log_info(f"[Hungarian Matcher] Pred probs shape: {out_prob.shape}, max prob: {out_prob.max():.4f}")
                log_info(f"[Hungarian Matcher] Class cost range: [{cost_class.min():.4f}, {cost_class.max():.4f}]")

            out_mask = outputs["pred_masks"][b]  # [num_queries, H_pred, W_pred]
            # gt masks are already padded when preparing target
            tgt_mask = targets[b]["masks"].to(out_mask)

            out_mask = out_mask[:, None]
            tgt_mask = tgt_mask[:, None]
            
            # Use full image for computation instead of point sampling
            with autocast(device_type="cuda", enabled=False):
                out_mask = out_mask.float().reshape(out_mask.shape[0], -1)   # [num_queries, H*W]
                tgt_mask = tgt_mask.float().reshape(tgt_mask.shape[0], -1)   # [num_targets, H*W]
                cost_dice = batch_dice_loss(out_mask, tgt_mask)
                cost_mask = batch_sigmoid_ce_loss(out_mask, tgt_mask)
            
            # Final cost matrix
            C = (
                self.cost_mask * cost_mask
                + self.cost_class * cost_class
                + self.cost_dice * cost_dice
            )
            C = C.reshape(num_queries, -1).cpu()
            
            # 添加成本矩阵日志
            if b == 0:  # 只记录第一个batch的信息
                log_info(f"[Hungarian Matcher] Mask cost range: [{cost_mask.min():.4f}, {cost_mask.max():.4f}]")
                log_info(f"[Hungarian Matcher] Dice cost range: [{cost_dice.min():.4f}, {cost_dice.max():.4f}]")
                log_info(f"[Hungarian Matcher] Total cost matrix shape: {C.shape}")
                log_info(f"[Hungarian Matcher] Total cost range: [{C.min():.4f}, {C.max():.4f}]")
                
                # 记录成本权重
                log_info(f"[Hungarian Matcher] Cost weights: class={self.cost_class}, mask={self.cost_mask}, dice={self.cost_dice}")
            
            # 执行匈牙利算法
            row_indices, col_indices = linear_sum_assignment(C)
            
            if b == 0:  # 记录匹配结果
                log_info(f"[Hungarian Matcher] Matched {len(row_indices)} pairs")
                log_info(f"[Hungarian Matcher] Matched queries: {row_indices[:10].tolist()}...")  # 前10个
                log_info(f"[Hungarian Matcher] Matched targets: {col_indices[:10].tolist()}...")  # 前10个
                
                # 计算匹配的平均成本
                matched_costs = C[row_indices, col_indices]
                log_info(f"[Hungarian Matcher] Matched costs: mean={matched_costs.mean():.4f}, std={matched_costs.std():.4f}")
            
            indices.append((row_indices, col_indices))

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        return self.memory_efficient_forward(outputs, targets)

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_class: {}".format(self.cost_class),
            "cost_mask: {}".format(self.cost_mask),
            "cost_dice: {}".format(self.cost_dice),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)


