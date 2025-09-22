from typing import List
import torch
import torch.nn.functional as F
from torch import nn
from .hungarian_matcher_simple import HungarianMatcher
from logger import log_info, log_error


def cat(tensors: List[torch.Tensor], dim: int = 0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
):
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


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
):
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
    num_masks = len(inputs)
    loss = F.binary_cross_entropy_with_logits(inputs.flatten(1), targets.flatten(1).float(), reduction="none")
    loss = loss.mean(1).sum() / num_masks
    return loss


def multi_cls_focal_loss(logits, targets):
    gamma = 2
    ce_loss = F.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = ((1 - pt) ** gamma * ce_loss).mean()
    return focal_loss


class SimpleMaskformerLoss(nn.Module):
    """
    基础MaskFormer损失函数，支持强制匹配和匈牙利匹配两种模式。
    """

    def __init__(self,
                 num_classes,
                 force_matching=False,  # 新增：强制匹配模式
                 expert_classification=False,  # 新增：专家分类模式
                 num_experts=6,  # 新增：专家数量，用于专家分类模式
                 matcher_cost_class=1,
                 matcher_cost_mask=1,
                 matcher_cost_dice=1,
                 eos_coef=1,
                 cost_weight=[2.0, 5.0, 5.0],
                 non_object=True,
                 no_object_weight=None,
                 ds_loss_weights=None,
                 disable_ds=False,
                 ds_avg_type='v0',
                 cls_type=None,
                 **kwargs
                 ):
        """
        Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            force_matching: if True, use force matching (queries directly correspond to experts)
            matcher: module able to compute a matching between targets and proposals
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        unused_kwargs = kwargs
        if unused_kwargs:
            log_info(f"unused kwargs: {unused_kwargs}")

        self.num_classes = num_classes
        self.force_matching = force_matching
        self.expert_classification = expert_classification
        self.num_experts = num_experts
        
        # 强制匹配模式下不需要匈牙利匹配器
        if not self.force_matching:
            self.matcher = HungarianMatcher(
                cost_class=matcher_cost_class,
                cost_mask=matcher_cost_mask,
                cost_dice=matcher_cost_dice,
            )
        else:
            self.matcher = None
            # 强制匹配模式下通常不需要non_object
            if non_object:
                log_info("Warning: force_matching=True but non_object=True, consider setting non_object=False")
        
        self.eos_coef = eos_coef
        self.non_object = non_object
        self.no_object_weight = no_object_weight

        self.ds_loss_weights = ds_loss_weights if ds_loss_weights is not None else [1.0, 0.8, 0.6, 0.4]
        self.ds_avg_type = ds_avg_type
        self.cls_type = cls_type

        if not self.force_matching:
            empty_weight = torch.ones(self.num_classes + 1)
            empty_weight[-1] = self.eos_coef
            self.register_buffer("empty_weight", empty_weight)

        self.disable_ds = disable_ds
        if self.disable_ds:
            self.ds_loss_weights[0] = 1
            self.ds_loss_weights[1:] = [0] * len(self.ds_loss_weights[1:]) 

        if cost_weight is None:
            self.cost_weight = [2.0, 5.0, 5.0]
        else:
            self.cost_weight = cost_weight

    def forward(self, outputs, targets):
        """
        This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        total_loss, smooth, do_fg = None, 1e-5, False
        if self.ds_loss_weights is not None:
            ds_loss_weights = self.ds_loss_weights
        else:
            len_ds = 1 + len(outputs['aux_outputs']) if isinstance(outputs, dict) else len(outputs)
            ds_loss_weights = [1] * len_ds

        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        loss_list = []
        
        if self.force_matching:
            loss_final = self.compute_loss_force_matching(outputs_without_aux, targets)
        else:
            loss_final = self.compute_loss_hungarian(outputs_without_aux, targets)
        
        loss_list.append(loss_final * ds_loss_weights[0])
        
        if not self.disable_ds:
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
        强制匹配模式：queries直接对应专家，不需要匈牙利匹配
        支持四种分类方式：
        1. expert_classification=True: 输出专家ID
        2. expert_classification=False: 输出mask类别
        适配2N数据格式：N个专家 x 2个类别(disc, cup)
        """
        # 获取预测和目标
        src_masks = outputs["pred_masks"]  # (B, num_queries, H, W)
        src_logits = outputs["pred_logits"]  # (B, num_queries, output_classes)
        
        batch_size, num_queries = src_masks.shape[:2]
        
        # 收集所有target masks和labels
        target_masks_list = []
        target_expert_labels_list = []
        target_mask_labels_list = []
        
        for batch_idx, target in enumerate(targets):
            t_masks = target["expert_masks"]  # (2N, H, W) - N个disc + N个cup
            t_expert_labels = target["expert_labels"]  # (2N,) - 专家ID序列
            t_mask_labels = target["mask_labels"]  # (2N,) - mask类别标签
            
            target_masks_list.append(t_masks)
            target_expert_labels_list.append(t_expert_labels)
            target_mask_labels_list.append(t_mask_labels)
        
        # 计算损失
        total_mask_loss = 0
        total_cls_loss = 0
        
        for batch_idx in range(batch_size):
            target_masks = target_masks_list[batch_idx]  # (2N, H, W)
            target_expert_labels = target_expert_labels_list[batch_idx]  # (2N,)
            target_mask_labels = target_mask_labels_list[batch_idx]  # (2N,)
            
            num_targets = min(num_queries, target_masks.shape[0])  # 取较小值
            
            # 类别损失 - 根据expert_classification模式选择不同的目标
            pred_cls = src_logits[batch_idx, :num_targets]  # (num_targets, output_classes)
            
            if self.expert_classification:
                # 专家分类模式：目标是专家ID
                target_cls = target_expert_labels[:num_targets]
            else:
                # Mask分类模式：目标是mask类别标签
                target_cls = target_mask_labels[:num_targets]
            
            if self.cls_type == 'focal':
                cls_loss = multi_cls_focal_loss(pred_cls, target_cls)
            else:
                cls_loss = F.cross_entropy(pred_cls, target_cls)
            
            total_cls_loss += cls_loss
            
            # 掩码损失 - 使用二元分割损失 (BCE + Dice)
            pred_masks = src_masks[batch_idx, :num_targets]  # (num_targets, H, W)
            target_masks_batch = target_masks[:num_targets]  # (num_targets, H, W)
            
            # 转换为二元mask（前景 vs 背景）
            target_masks_binary = (target_masks_batch > 0).float()  # (num_targets, H, W)
            
            # 计算BCE和Dice损失
            mask_ce_loss = sigmoid_ce_loss(pred_masks, target_masks_binary)
            mask_dice_loss = dice_loss(pred_masks, target_masks_binary)
            
            total_mask_loss += mask_ce_loss + mask_dice_loss
        
        # 平均损失
        total_cls_loss = total_cls_loss / batch_size
        total_mask_loss = total_mask_loss / batch_size
        
        # 组合损失（类似MaskFormer的模式）
        loss = (total_cls_loss * (self.cost_weight[0] / 10) + 
                total_mask_loss * (self.cost_weight[1] / 10))
        
        return loss

    def compute_loss_hungarian(self, outputs, targets):
        """
        匈牙利匹配模式：原有的匈牙利匹配逻辑，适配2N数据格式
        """
        # 将targets转换为匈牙利匹配器期望的格式
        matcher_targets = []
        for target in targets:
            if self.expert_classification:
                # 专家分类模式：使用专家ID作为标签
                target_labels = target["expert_labels"]
            else:
                # Mask分类模式：使用mask类别作为标签
                target_labels = target["mask_labels"]
            
            matcher_target = {
                "labels": target_labels,  # (2N,)
                "masks": target["expert_masks"]  # (2N, H, W)
            }
            matcher_targets.append(matcher_target)
        
        indices = self.matcher(outputs, matcher_targets)
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        
        # 匈牙利匹配不能保证所有targets都被匹配，特别是当queries < targets时
        total_target_masks = sum([len(t["expert_masks"]) for t in targets])
        total_queries = outputs["pred_masks"].shape[1]
        max_possible_matches = min(total_queries * len(targets), total_target_masks)
        actual_matches = len(tgt_idx[0])
        
        if actual_matches != max_possible_matches:
            log_info(f"[Hungarian Loss] Matching: {actual_matches}/{total_target_masks} targets matched "
                    f"(queries={total_queries}, max_possible={max_possible_matches})")
        
        # 添加匹配质量分析
        if hasattr(self, '_log_counter'):
            self._log_counter += 1
        else:
            self._log_counter = 1
            
        # 每50次迭代记录一次详细信息
        if self._log_counter % 50 == 0:
            log_info(f"[Hungarian Loss] Detailed matching analysis (iteration {self._log_counter}):")
            for batch_idx, (src_batch_idx, tgt_batch_idx) in enumerate(zip(src_idx[0], tgt_idx[0])):
                if batch_idx < 5:  # 只记录前5个匹配
                    log_info(f"  Batch {src_batch_idx}: Query {src_idx[1][batch_idx]} -> Target {tgt_idx[1][batch_idx]}")

        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]  # 根据匹配结果选择预测masks
        
        # 收集所有目标masks
        target_masks = torch.cat([t["expert_masks"] for t in targets], dim=0)  # (总数, H, W)
        target_masks = target_masks[tgt_idx[1]]  # 根据匹配结果选择目标masks
        
        # 计算mask损失 - BCE + Dice
        target_masks_binary = (target_masks > 0).float()  # 转换为二元mask
        mask_ce_loss = sigmoid_ce_loss(src_masks, target_masks_binary)
        mask_dice_loss = dice_loss(src_masks, target_masks_binary)
        
        # 添加mask损失分析
        if self._log_counter % 50 == 0:
            log_info(f"[Hungarian Loss] Mask loss analysis:")
            log_info(f"  Target masks stats: min={target_masks.min():.4f}, max={target_masks.max():.4f}, mean={target_masks.mean():.4f}")
            log_info(f"  Pred masks stats: min={src_masks.min():.4f}, max={src_masks.max():.4f}, mean={src_masks.mean():.4f}")
            log_info(f"  BCE loss: {mask_ce_loss:.4f}, Dice loss: {mask_dice_loss:.4f}")

        # 计算分类损失
        src_logits = outputs["pred_logits"].float()
        idx = self._get_src_permutation_idx(indices)
        
        if self.expert_classification:
            # 专家分类模式：目标是expert_labels (0-11)，对应12个专家类别
            # 标签0-5: 专家0-5的disc
            # 标签6-11: 专家0-5的cup
            target_classes_o = torch.cat([t["expert_labels"][J] for t, (_, J) in zip(targets, indices)])
            
            # 添加调试信息
            if hasattr(self, '_log_counter') and self._log_counter % 50 == 0:
                log_info(f"[Expert Classification] Expert labels: {target_classes_o[:10].tolist()}...")
                # 分析标签分布
                disc_labels = target_classes_o[target_classes_o < self.num_experts]  # 0-5 (disc)
                cup_labels = target_classes_o[target_classes_o >= self.num_experts]  # 6-11 (cup)
                log_info(f"[Expert Classification] Disc labels count: {len(disc_labels)}, Cup labels count: {len(cup_labels)}")
        else:
            # Mask分类模式：目标是mask类别
            target_classes_o = torch.cat([t["mask_labels"][J] for t, (_, J) in zip(targets, indices)])
        
        # 为未匹配的queries设置背景类（如果启用non_object）
        if self.non_object:
            if self.expert_classification:
                background_class = self.num_experts * 2  # 专家分类模式下，背景类是2N（因为有2N个专家标签）
            else:
                background_class = self.num_classes  # Mask分类模式下，背景类是num_classes
        else:
            # 如果没有non_object，使用第一个类作为默认
            background_class = 0
            
        target_classes = torch.full(
            src_logits.shape[:2], background_class, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        if self.cls_type == 'focal':
            loss_cls = multi_cls_focal_loss(outputs["pred_logits"].transpose(1, 2), target_classes)
        else:
            if self.no_object_weight is not None:
                if self.expert_classification:
                    num_classes_total = self.num_experts * 2  # 2N个专家标签
                else:
                    num_classes_total = self.num_classes  # mask类别数
                empty_weight = torch.ones(num_classes_total + int(self.non_object)).to(outputs["pred_logits"].device)
                empty_weight[-1] = self.eos_coef
                loss_cls = F.cross_entropy(outputs["pred_logits"].transpose(1, 2), target_classes, empty_weight)
            else:
                loss_cls = F.cross_entropy(outputs["pred_logits"].transpose(1, 2), target_classes)

        # 组合损失
        loss = (loss_cls * (self.cost_weight[0] / 10) + 
                mask_ce_loss * (self.cost_weight[1] / 10) + 
                mask_dice_loss * (self.cost_weight[2] / 10))
        return loss

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx