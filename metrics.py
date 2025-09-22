import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


# ================== RIGA ================== #

def convert_ring_cup_to_disc_cup(pred_ring_cup, thresh=0.5, is_label=False):
    """
    将模型预测的 3 通道 one-hot 输出（背景、视环、视杯）转换为 2 通道形式（视盘、视杯）。
    
    Args:
        pred_ring_cup: 5D tensor (B, N, 3, H, W) - 输入的ring/cup格式数据
        thresh: 阈值，仅对预测数据有效 (默认0.5)
        is_label: 是否为标签数据。True时忽略thresh，直接使用二值数据
    
    Returns:
        5D tensor (B, N, 2, H, W) - disc/cup格式数据
    """
    if pred_ring_cup.dim() != 5:
        raise ValueError(f"Input must be 5D tensor (B, N, 3, H, W), got {pred_ring_cup.dim()}D")
    
    B, N, C, H, W = pred_ring_cup.shape
    if C != 3:
        raise ValueError(f"Input must have 3 channels (bg, ring, cup), got {C}")
    
    if is_label:
        # 标签数据：直接使用二值数据，不应用阈值
        ring = pred_ring_cup[:, :, 1, :, :]  # (B, N, H, W)
        cup = pred_ring_cup[:, :, 2, :, :]   # (B, N, H, W)
    else:
        # 预测数据：应用阈值进行二值化
        pred_binary = (pred_ring_cup > thresh).float()
        ring = pred_binary[:, :, 1, :, :]
        cup = pred_binary[:, :, 2, :, :]
    
    # disc = ring + cup (视盘包含视环和视杯)
    disc = torch.clamp(ring + cup, 0, 1)
    
    # 构建 (B, N, 2, H, W) 格式：[disc, cup]
    disc_cup = torch.stack([disc, cup], dim=2)
    
    return disc_cup


def convert_labels_ring_cup_to_disc_cup(labels_ring_cup):
    """
    专用于标签转换的便利函数，避免阈值混乱。
    
    Args:
        labels_ring_cup: 5D tensor (B, N, 3, H, W) - ring/cup格式的标签数据
    
    Returns:
        5D tensor (B, N, 2, H, W) - disc/cup格式的标签数据
    """
    return convert_ring_cup_to_disc_cup(labels_ring_cup, is_label=True)

def convert_ring_cup_probs_to_disc_cup_probs(ring_cup_probs: torch.Tensor) -> torch.Tensor:
    """
    将 (..., bg, ring, cup) 概率张量转换为 (..., disc, cup) 概率张量。
    """
    if ring_cup_probs.shape[-3] != 3:
        raise ValueError(f"Input tensor must have 3 channels (bg, ring, cup), but got {ring_cup_probs.shape[-3]}")

    ring_probs = ring_cup_probs[..., 1, :, :]
    cup_probs = ring_cup_probs[..., 2, :, :]
    disc_probs = torch.clamp(ring_probs + cup_probs, 0, 1)

    return torch.stack([disc_probs, cup_probs], dim=-3)


# ===================================================================
# 核心辅助函数
# ===================================================================

def _compute_iou_torch(pred: torch.Tensor, label: torch.Tensor, smooth: float = 1e-8) -> torch.Tensor:
    """计算两个Tensor之间的IoU。"""
    intersection = (pred * label).sum()
    union = pred.sum() + label.sum() - intersection
    return (intersection + smooth) / (union + smooth)

def _compute_dice_torch(pred: torch.Tensor, label: torch.Tensor, smooth: float = 1e-8) -> torch.Tensor:
    """计算两个Tensor之间的Dice相似系数。"""
    intersection = (pred * label).sum()
    return (2. * intersection + smooth) / (pred.sum() + label.sum() + smooth)

# ===================================================================
# 最终评估指标计算函数
# ===================================================================

def calculate_soft_dice(labels: torch.Tensor, preds: torch.Tensor, class_names: list = None) -> dict:
    """计算软Dice分数，可以灵活处理任意类别。"""
    if labels.dim() != preds.dim() or labels.shape[-3] != preds.shape[-3]:
         raise ValueError("Labels and preds must have same dimensions and channel count.")

    num_classes = labels.shape[-3]
    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]
    
    batch_dice_scores_per_class = {name: [] for name in class_names}
    
    for i in range(labels.shape[0]):
        pred_mean = preds[i].mean(0)
        label_mean = labels[i].mean(0)

        for c, class_name in enumerate(class_names):
            thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
            thres_dice_scores = []
            for thresh in thresholds:
                pred_binary = (pred_mean[c] > thresh).float().view(-1)
                label_binary = (label_mean[c] > thresh).float().view(-1)
                dice = _compute_dice_torch(pred_binary, label_binary)
                thres_dice_scores.append(dice)
            
            avg_dice_for_class = torch.stack(thres_dice_scores).mean().item()
            batch_dice_scores_per_class[class_name].append(avg_dice_for_class)

    final_scores = {name: np.mean(scores) if scores else 0.0 for name, scores in batch_dice_scores_per_class.items()}
    mean_score = np.mean(list(final_scores.values()))
    final_scores['mean'] = mean_score
    return final_scores


def calculate_ged(labels: torch.Tensor, preds: torch.Tensor, class_names: list = None) -> float:
    """
    [通用版本] 计算广义能量距离 (GED)，支持任意通道数的多类别分割。
    
    Args:
        labels: 5D tensor (B, N, C, H, W) - 真实标签
        preds: 5D tensor (B, N, C, H, W) - 预测结果
        class_names: 类别名称列表，用于标识各通道含义 (可选)
    
    Returns:
        float: GED值
    """
    if labels.dim() != 5 or preds.dim() != 5 or labels.shape[2] != preds.shape[2]:
        raise ValueError("Input tensors must be 5D (B, N, C, H, W) with matching channel dimensions")

    num_classes = labels.shape[2]
    if class_names is not None and len(class_names) != num_classes:
        raise ValueError(f"class_names length ({len(class_names)}) must match number of classes ({num_classes})")

    batch_ged = []

    for i in range(labels.shape[0]): # 遍历Batch
        pred_masks_i = preds[i]   # (N, C, H, W)
        label_masks_i = labels[i] # (N, C, H, W)
        
        def _get_avg_dist(masks1, masks2):
            total_dist = 0
            count = 0
            for m1_idx in range(masks1.shape[0]):
                for m2_idx in range(masks2.shape[0]):
                    if masks1 is masks2 and m1_idx == m2_idx: continue
                    
                    mask1 = masks1[m1_idx] # (C, H, W)
                    mask2 = masks2[m2_idx] # (C, H, W)

                    # 计算所有类别的IoU并取平均
                    class_ious = []
                    for c in range(num_classes):
                        iou_c = _compute_iou_torch(mask1[c].view(-1), mask2[c].view(-1))
                        class_ious.append(iou_c)
                    
                    avg_iou = torch.stack(class_ious).mean()
                    total_dist += (1 - avg_iou)
                    count += 1
            return total_dist / count if count > 0 else torch.tensor(0.0, device=labels.device)

        dist_pred_label = _get_avg_dist(pred_masks_i, label_masks_i)
        dist_pred_pred = _get_avg_dist(pred_masks_i, pred_masks_i)
        dist_label_label = _get_avg_dist(label_masks_i, label_masks_i)
        
        ged = 2 * dist_pred_label - dist_pred_pred - dist_label_label
        batch_ged.append(ged.item())

    return np.mean(batch_ged) if batch_ged else 0.0


def calculate_personalization_metrics(labels: torch.Tensor, preds: torch.Tensor, is_test: bool = True, class_names: list = None) -> dict:
    """
    [通用版本] 计算所有个性化指标，支持任意数据集的多类别分割。
    
    Args:
        labels: 5D tensor (B, N, C, H, W) - 多专家真实标签
        preds: 5D tensor (B, N, C, H, W) - 多专家预测结果  
        is_test: 是否为测试模式（影响专家匹配策略）
        class_names: 类别名称列表，如 ["disc", "cup"] 或 ["lesion", "normal"] 等
    
    Returns:
        dict: 包含 dice_max, dice_match, dice_per_expert 的详细指标
        
    支持的数据集示例:
        - RIGA: class_names=["disc", "cup"]
        - 皮肤病变: class_names=["lesion", "normal"]
        - 肺部分割: class_names=["left_lung", "right_lung", "heart"]
        - 脑肿瘤: class_names=["necrotic", "edema", "enhancing"]
    """
    if labels.dim() != 5 or preds.dim() != 5 or labels.shape[1] != preds.shape[1]:
        raise ValueError("Input tensors must be 5D (B, N, C, H, W) and N must be equal")

    num_experts = labels.shape[1]
    num_classes = labels.shape[2]
    
    if class_names is None:
        class_names = [f"class_{c}" for c in range(num_classes)]
    elif len(class_names) != num_classes:
        raise ValueError("Length of class_names must match the number of classes")

    batch_dice_max, batch_dice_match, batch_per_expert = [], [], []

    for i in range(labels.shape[0]):
        pred_masks = (preds[i] > 0.5).float()
        label_masks = labels[i].float()
        
        dice_matrices_per_class = torch.zeros((num_classes, num_experts, num_experts), device=preds.device)
        for c in range(num_classes):
            for rater_idx in range(num_experts):
                for pred_idx in range(num_experts):
                    dice_matrices_per_class[c, rater_idx, pred_idx] = _compute_dice_torch(
                        pred_masks[pred_idx, c].view(-1),
                        label_masks[rater_idx, c].view(-1)
                    )
        
        mean_dice_matrix = dice_matrices_per_class.mean(dim=0)
        
        if is_test:
            matched_indices = (torch.arange(num_experts), torch.arange(num_experts))
        else:
            cost_matrix_np = (1 - mean_dice_matrix).cpu().numpy()
            matched_indices = linear_sum_assignment(cost_matrix_np)

        sample_max = {name: dice_matrices_per_class[c].max(dim=1)[0].mean().item() for c, name in enumerate(class_names)}
        sample_match = {name: dice_matrices_per_class[c, matched_indices[0], matched_indices[1]].mean().item() for c, name in enumerate(class_names)}
        sample_per_expert = {name: dice_matrices_per_class[c, matched_indices[0], matched_indices[1]].tolist() for c, name in enumerate(class_names)}
        
        batch_dice_max.append(sample_max)
        batch_dice_match.append(sample_match)
        batch_per_expert.append(sample_per_expert)

    final_dice_max = {name: np.mean([d[name] for d in batch_dice_max]) for name in class_names}
    final_dice_match = {name: np.mean([d[name] for d in batch_dice_match]) for name in class_names}
    final_per_expert = {name: np.mean([d[name] for d in batch_per_expert], axis=0).tolist() for name in class_names}

    final_dice_max['overall'] = np.mean(list(final_dice_max.values()))
    final_dice_match['overall'] = np.mean(list(final_dice_match.values()))
    
    return {
        'dice_max': final_dice_max,
        'dice_match': final_dice_match,
        'dice_per_expert': final_per_expert
    }


# ===================================================================
# 数据集特定的便利函数
# ===================================================================

def calculate_riga_metrics(labels: torch.Tensor, preds: torch.Tensor, is_test: bool = True) -> dict:
    """
    RIGA数据集专用的便利函数，预设disc/cup类别名称。
    
    Args:
        labels: 5D tensor (B, N, 2, H, W) - RIGA多专家标签
        preds: 5D tensor (B, N, 2, H, W) - RIGA多专家预测
        is_test: 是否为测试模式
    
    Returns:
        dict: 包含所有RIGA相关指标的完整结果
    """
    class_names = ["disc", "cup"]
    
    results = {}
    
    # 计算软Dice分数
    results['soft_dice'] = calculate_soft_dice(labels, preds, class_names=class_names)
    
    # 计算GED
    results['ged'] = calculate_ged(labels, preds, class_names=class_names)
    
    # 计算个性化指标
    personalization = calculate_personalization_metrics(labels, preds, is_test=is_test, class_names=class_names)
    results.update(personalization)
    
    return results
