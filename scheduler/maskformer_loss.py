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


class MaskformerLoss(nn.Module):
    """
    This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self,
                 num_classes,
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
            matcher: module able to compute a matching between targets and proposals
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        unused_kwargs = kwargs
        if unused_kwargs:
            log_info(f"unused kwargs: {unused_kwargs}")

        self.num_classes = num_classes
        self.matcher = HungarianMatcher(
            cost_class=matcher_cost_class,
            cost_mask=matcher_cost_mask,
            cost_dice=matcher_cost_dice,
        )
        self.eos_coef = eos_coef
        self.non_object = non_object
        self.no_object_weight = no_object_weight

        self.ds_loss_weights = ds_loss_weights if ds_loss_weights is not None else [1.0, 0.8, 0.6, 0.4]
        self.ds_avg_type = ds_avg_type
        self.cls_type = cls_type

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
        loss_final = self.compute_loss_hungarian(outputs_without_aux, targets)
        # print(f"loss: {loss_final}")
        loss_list.append(loss_final * ds_loss_weights[0])
        if not self.disable_ds:
            for i, aux_outputs in enumerate(outputs["aux_outputs"][::-1]):  # reverse order
                loss_aux = self.compute_loss_hungarian(aux_outputs, targets)
                loss_list.append(ds_loss_weights[i + 1] * loss_aux)
            # print(f"ds_avg_type:{self.ds_avg_type}, loss: {loss_list}")
            if self.ds_avg_type == 'v0':
                # print(f"in v0")
                total_loss = sum(loss_list) / len(loss_list)
                # print(f"total_loss: {total_loss}")
            elif self.ds_avg_type == 'v1':
                total_loss = (loss_list[0] + sum(loss_list[1:]) / len(loss_list[1:])) / 2
            elif self.ds_avg_type == 'v2':
                total_loss = sum(loss_list)
        else:
            total_loss = loss_final
    
        return total_loss


    def compute_loss_hungarian(self, outputs, targets):
        indices = self.matcher(outputs, targets)
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        assert len(tgt_idx[0]) == sum(
            [len(t["masks"]) for t in targets])  # verify that all masks of (K1, K2, ..) are used

        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        target_masks = torch.cat([t["masks"] for t in targets], dim=0)  # (K1+K2+..., D, H, W) actually
        print(f"src_masks.shape: {src_masks.shape}, target_masks.shape: {target_masks.shape}")
        
        # Direct mask loss calculation without point sampling
        src_masks = src_masks.flatten(1)  # [K..., D*H*W]
        target_masks = target_masks.flatten(1)  # [K..., D*H*W]
        
        # print(f"src_masks.shape: {src_masks.shape}, target_masks.shape: {target_masks.shape}")
        mask_ce_loss = sigmoid_ce_loss(src_masks, target_masks)
        mask_dice_loss = dice_loss(src_masks, target_masks)

        src_logits = outputs["pred_logits"].float()
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        if self.cls_type == 'focal':
            loss_cls = multi_cls_focal_loss(outputs["pred_logits"].transpose(1, 2), target_classes)
        else:
            if self.no_object_weight is not None:
                empty_weight = torch.ones(self.num_classes + self.non_object).to(outputs["pred_logits"].device)
                empty_weight[-1] = self.eos_coef
                loss_cls = F.cross_entropy(outputs["pred_logits"].transpose(1, 2), target_classes, empty_weight)
            else:
                loss_cls = F.cross_entropy(outputs["pred_logits"].transpose(1, 2), target_classes)

        loss = (loss_cls * (self.cost_weight[0] / 10) + mask_ce_loss * (self.cost_weight[1] / 10) + mask_dice_loss * (
                    self.cost_weight[2] / 10))
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

