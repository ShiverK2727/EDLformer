# -*- coding: utf-8 -*-
"""
EDL MaskFormer 的损失函数实现 V2。
- 使用V3版本的匹配器。
- 对分类和分割分支使用独立的权重进行加权。
- 各损失项可通过设置权重为0来禁用。
- 更新EDL损失为使用Digamma函数。
"""
import torch
import torch.nn.functional as F
from torch import nn
from .hungarian_matcher_edl_simple import HungarianMatcherEDL_V3
from logger import log_info
import collections


# --- 辅助函数 (与之前版本相同) ---

def dice_loss_from_prob(inputs_prob: torch.Tensor, targets: torch.Tensor):
    """
    从概率计算DICE损失 (用于单通道二元分割)。
    """
    inputs_prob = inputs_prob.flatten(1)
    targets = targets.flatten(1)
    num_masks = len(inputs_prob)
    if num_masks == 0:
        return torch.tensor(0.0, device=inputs_prob.device)
    
    numerator = 2 * (inputs_prob * targets).sum(-1)
    denominator = inputs_prob.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks

def dice_loss_from_logits(logits: torch.Tensor, targets_one_hot: torch.Tensor):
    """
    从Logits计算DICE损失 (用于多类别分割)。
    """
    probs = torch.softmax(logits, dim=1)
    num_masks, num_classes = probs.shape[:2]
    if num_masks == 0:
        return torch.tensor(0.0, device=logits.device)
        
    probs = probs.flatten(2)
    targets_one_hot = targets_one_hot.flatten(2)

    # 忽略背景类别 (channel 0)
    numerator = 2 * (probs[:, 1:] * targets_one_hot[:, 1:]).sum(-1)
    denominator = probs[:, 1:].sum(-1) + targets_one_hot[:, 1:].sum(-1)
    
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


def edl_beta_loss(alpha: torch.Tensor, beta: torch.Tensor, targets: torch.Tensor):
    """
    计算Beta分布的证据损失 (Type-II ML)，使用Digamma函数。
    """
    S = alpha + beta
    alpha_flat, beta_flat = alpha.flatten(1), beta.flatten(1)
    S_flat, targets_flat = S.flatten(1), targets.flatten(1)
    
    # 使用Digamma函数替换log函数
    loss = targets_flat * (torch.digamma(S_flat) - torch.digamma(alpha_flat)) + \
           (1 - targets_flat) * (torch.digamma(S_flat) - torch.digamma(beta_flat))
    
    return loss.mean()

def pixelwise_edl_dirichlet_loss(evidence: torch.Tensor, targets_one_hot: torch.Tensor):
    """
    计算像素级的Dirichlet证据损失，使用Digamma函数。
    """
    alpha = F.softplus(evidence) + 1
    S = torch.sum(alpha, dim=1, keepdim=True)
    # 使用Digamma函数替换log函数
    loss = torch.sum(targets_one_hot * (torch.digamma(S) - torch.digamma(alpha)), dim=1)
    return loss.mean()


def edl_dirichlet_loss(alpha: torch.Tensor, targets_one_hot: torch.Tensor):
    """
    计算分类的Dirichlet证据损失 (Type-II ML)，使用Digamma函数。
    """
    S = torch.sum(alpha, dim=1, keepdim=True)
    # 使用Digamma函数替换log函数
    loss = torch.sum(targets_one_hot * (torch.digamma(S) - torch.digamma(alpha)), dim=1)
    return loss.mean()


def kl_divergence_dirichlet(alpha: torch.Tensor, targets_one_hot: torch.Tensor):
    """
    计算KL散度正则化项，仅用于惩罚错误类别的证据。
    """
    beta_prior = torch.ones_like(alpha)
    alpha_regularized = alpha * (1 - targets_one_hot) + targets_one_hot
    
    S_alpha = torch.sum(alpha_regularized, dim=1, keepdim=True)
    S_beta = torch.sum(beta_prior, dim=1, keepdim=True)

    lnB_alpha = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha_regularized), dim=1, keepdim=True)
    lnB_beta = torch.sum(torch.lgamma(beta_prior), dim=1, keepdim=True) - torch.lgamma(S_beta)
    
    dg_S_alpha = torch.digamma(S_alpha)
    dg_alpha = torch.digamma(alpha_regularized)
    
    kl_div = torch.sum((alpha_regularized - beta_prior) * (dg_alpha - dg_S_alpha), dim=1, keepdim=True) + lnB_alpha + lnB_beta
    return kl_div.mean()



class SimpleEDLMaskformerLossV2(nn.Module):
    """
    此类用于计算SimpleEDLMaskformer的损失。
    """

    def __init__(self,
                 num_cls_classes,  # 分类头的类别数（不包含背景类）
                 # 注意：分割部分固定为二元分割（前景/背景），与 num_cls_classes 无关
                 # Matcher weights
                 matcher_cost_class=1.0,
                 matcher_cost_dice=1.0,
                 matcher_cost_evidence=1.0,
                 matcher_cost_class_type='evidence',
                 # Loss options and weights
                 cls_loss_type='edl',
                 seg_edl_as_dirichlet=False,
                 dice_on_logits=False,
                 branch_weight_cls=1.0,
                 branch_weight_seg=1.0,
                 # Sub-weights for classification branch
                 loss_weight_cls=1.0,
                 loss_weight_kl=1.0,
                 # Sub-weights for segmentation branch
                 loss_weight_dice=1.0,
                 loss_weight_edl_mask=1.0,
                 # Other params
                 eos_coef=0.1,
                 annealing_steps=10000,
                 non_object=True,
                 ds_loss_weights=None,
                 log_interval=100,
                 use_direct_matching=False,
                 use_deep_supervision=True,  # 新增: 是否启用深监督
                 ds_avg_type='sum',      # 新增: 深监督损失聚合方式 ('sum', 'mean', 'v1_style')
                 ):
        super().__init__()
        
        # num_cls_classes: 仅用于分类头，表示真实的分类类别数（不包含背景类）
        # 分割部分固定为二元分割，与num_cls_classes无关
        self.num_cls_classes = num_cls_classes
        self.cls_loss_type = cls_loss_type
        self.seg_edl_as_dirichlet = seg_edl_as_dirichlet
        self.dice_on_logits = dice_on_logits
        
        self.use_direct_matching = use_direct_matching
        if not self.use_direct_matching:
            self.matcher = HungarianMatcherEDL_V3(
                cost_class=matcher_cost_class,
                cost_dice=matcher_cost_dice,
                cost_evidence=matcher_cost_evidence,
                cost_class_type=matcher_cost_class_type
            )
            
        self.eos_coef, self.non_object = eos_coef, non_object
        if self.non_object:
            # 分类头的总类别数 = 真实类别数 + 1(背景类)
            empty_weight = torch.ones(self.num_cls_classes + 1)
            empty_weight[-1] = self.eos_coef
            self.register_buffer("empty_weight", empty_weight)

        self.ds_loss_weights = ds_loss_weights if ds_loss_weights is not None else [1.0] * 4
        
        # Loss branch weights
        self.branch_weight_cls = branch_weight_cls
        self.branch_weight_seg = branch_weight_seg
        # Loss sub-weights
        self.loss_weight_cls = loss_weight_cls
        self.loss_weight_kl = loss_weight_kl
        self.loss_weight_dice = loss_weight_dice
        self.loss_weight_edl_mask = loss_weight_edl_mask

        self.annealing_steps = annealing_steps
        
        # 日志记录器
        self.log_interval = log_interval
        self._log_counter = 0
        
        # 新增深监督控制参数
        self.use_deep_supervision = use_deep_supervision
        self.ds_avg_type = ds_avg_type
        assert self.ds_avg_type in ['sum', 'mean_all', 'mean_aux'], "ds_avg_type must be 'sum', 'mean_all', or 'mean_aux'"

    def loss_labels(self, outputs, targets, indices, current_step, should_log=False):
        # outputs["pred_logits"] is (alpha, None) for 'single' mode
        src_logits_alpha, _ = outputs["pred_logits"]
        idx = self._get_src_permutation_idx(indices)
        
        num_matches = len(idx[0])
        if num_matches == 0: # No matches
            if should_log:
                log_info("[Loss Labels] No queries matched.")
            return torch.tensor(0.0, device=src_logits_alpha.device), torch.tensor(0.0, device=src_logits_alpha.device)

        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        
        if self.use_direct_matching:
            # 直接匹配模式：每个样本都有固定数量的目标(N*C=12个专家-类别组合)
            # 不需要背景类，直接计算匹配的查询-目标对的损失
            if self.cls_loss_type == 'ce':
                # 对匹配的查询使用交叉熵损失
                src_logits_matched = src_logits_alpha[idx]  # [num_matches, num_cls_classes]
                loss_cls = F.cross_entropy(src_logits_matched, target_classes_o)
                loss_kl = torch.tensor(0.0, device=src_logits_alpha.device)
            else: # 'edl'
                evidence = F.softplus(src_logits_alpha)
                alpha = evidence + 1
                alpha_matched = alpha[idx]  # [num_matches, num_cls_classes]
                # 使用num_cls_classes作为类别数（不包含背景类）
                target_one_hot_matched = F.one_hot(target_classes_o, num_classes=self.num_cls_classes).float()
                
                loss_cls = edl_dirichlet_loss(alpha_matched, target_one_hot_matched)
                annealing_coef = min(1.0, current_step / self.annealing_steps)
                loss_kl = annealing_coef * kl_divergence_dirichlet(alpha_matched, target_one_hot_matched)
        else:
            # 匈牙利匹配模式：需要处理背景类
            background_class = self.num_cls_classes
            target_classes = torch.full(
                src_logits_alpha.shape[:2], background_class, dtype=torch.int64, device=src_logits_alpha.device
            )
            target_classes[idx] = target_classes_o

            # 分类头的总类别数 = 真实类别数 + 背景类
            num_total_classes = self.num_cls_classes + (1 if self.non_object else 0)
            
            if self.cls_loss_type == 'ce':
                loss_cls = F.cross_entropy(src_logits_alpha.transpose(1, 2), target_classes, self.empty_weight if self.non_object else None)
                loss_kl = torch.tensor(0.0, device=src_logits_alpha.device)
            else: # 'edl'
                evidence = F.softplus(src_logits_alpha)
                alpha = evidence + 1
                alpha_matched = alpha[idx]
                target_one_hot_matched = F.one_hot(target_classes_o, num_classes=num_total_classes).float()
                
                loss_cls = edl_dirichlet_loss(alpha_matched, target_one_hot_matched)
                annealing_coef = min(1.0, current_step / self.annealing_steps)
                loss_kl = annealing_coef * kl_divergence_dirichlet(alpha_matched, target_one_hot_matched)

        if should_log:
            matching_mode = "Direct" if self.use_direct_matching else "Hungarian"
            log_info(f"[Loss Labels - {matching_mode}] Matched {num_matches} queries. "
                     f"loss_cls: {loss_cls.item():.4f}, loss_kl: {loss_kl.item():.4f}")

        return loss_cls, loss_kl

    def loss_masks(self, outputs, targets, indices, should_log=False):
        src_alpha_mask, src_beta_mask = outputs["pred_masks"]
        src_idx = self._get_src_permutation_idx(indices)
        
        num_matches = len(src_idx[0])
        if num_matches == 0: # No matches
            if should_log:
                log_info("[Loss Masks] No masks matched.")
            return torch.tensor(0.0, device=src_alpha_mask.device), torch.tensor(0.0, device=src_alpha_mask.device)

        src_alpha_mask = src_alpha_mask[src_idx]
        src_beta_mask = src_beta_mask[src_idx]

        target_masks = torch.cat([t["masks"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_masks = (target_masks > 0).float()

        if self.seg_edl_as_dirichlet:
            # Channel 0: background (beta), Channel 1: foreground (alpha)
            evidence = torch.stack([src_beta_mask, src_alpha_mask], dim=1)
            target_masks_one_hot = F.one_hot(target_masks.long(), num_classes=2).permute(0, 3, 1, 2).float()
            
            loss_edl_mask = pixelwise_edl_dirichlet_loss(evidence, target_masks_one_hot)

            if self.dice_on_logits:
                loss_dice = dice_loss_from_logits(evidence, target_masks_one_hot)
            else:
                alpha = F.softplus(src_alpha_mask) + 1
                beta = F.softplus(src_beta_mask) + 1
                prob_mask = alpha / (alpha + beta)
                loss_dice = dice_loss_from_prob(prob_mask, target_masks)
        else:
            alpha = F.softplus(src_alpha_mask) + 1
            beta = F.softplus(src_beta_mask) + 1
            loss_edl_mask = edl_beta_loss(alpha, beta, target_masks)
            
            if self.dice_on_logits:
                logits = src_alpha_mask - src_beta_mask
                prob_mask_from_logits = logits.sigmoid()
                loss_dice = dice_loss_from_prob(prob_mask_from_logits, target_masks)
            else:
                prob_mask = alpha / (alpha + beta)
                loss_dice = dice_loss_from_prob(prob_mask, target_masks)
        
        if should_log:
            log_info(f"[Loss Masks] Matched {num_matches} masks. "
                     f"loss_dice: {loss_dice.item():.4f}, loss_edl_mask: {loss_edl_mask.item():.4f}")

        return loss_dice, loss_edl_mask

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def compute_loss(self, outputs, targets, current_step, should_log=False):
        device = outputs["pred_logits"][0].device

        # 根据开关选择匹配策略
        if self.use_direct_matching:
            # 直接匹配：每个样本的前N*C个查询直接与N*C个目标一对一匹配
            indices = []
            for i, t in enumerate(targets):
                num_targets = len(t["labels"])  # 应该是N*C=12
                num_queries = outputs["pred_logits"][0].shape[1]  # 模型的查询数量
                
                if num_queries < num_targets:
                    raise ValueError(
                        f"Direct matching requires num_queries ({num_queries}) >= num_targets ({num_targets}). "
                        f"Sample {i} has {num_targets} targets but model only has {num_queries} queries."
                    )
                
                # 直接匹配：前num_targets个查询对应num_targets个目标
                src_idx = torch.arange(num_targets, device=device)
                tgt_idx = torch.arange(num_targets, device=device)
                indices.append((src_idx, tgt_idx))
                
                if should_log and i == 0:  # 只在第一个样本时记录
                    log_info(f"[Direct Matching] Sample {i}: {num_targets} targets matched to first {num_targets} queries")
        else:
            # 匈牙利匹配：使用匹配器寻找最优分配
            indices = self.matcher(outputs, targets)
            if should_log:
                total_matches = sum(len(idx[0]) for idx in indices)
                log_info(f"[Hungarian Matching] Total matches across batch: {total_matches}")

        loss_cls, loss_kl = self.loss_labels(outputs, targets, indices, current_step, should_log)
        loss_dice, loss_edl_mask = self.loss_masks(outputs, targets, indices, should_log)
        
        # Calculate branch-wise losses
        loss_cls_branch = self.loss_weight_cls * loss_cls + self.loss_weight_kl * loss_kl
        loss_seg_branch = self.loss_weight_dice * loss_dice + self.loss_weight_edl_mask * loss_edl_mask
        
        # Calculate final weighted loss
        total_loss = (self.branch_weight_cls * loss_cls_branch + 
                      self.branch_weight_seg * loss_seg_branch)
        
        # 创建用于日志记录的字典，存储未加权的损失分量
        loss_dict = {
            'loss_cls': loss_cls.detach(),
            'loss_kl': loss_kl.detach(),
            'loss_dice': loss_dice.detach(),
            'loss_edl_mask': loss_edl_mask.detach()
        }
        
        if should_log:
            log_info(f"[Compute Loss] cls_branch_loss: {loss_cls_branch.item():.4f}, "
                     f"seg_branch_loss: {loss_seg_branch.item():.4f} -> total: {total_loss.item():.4f}")
        
        return total_loss, loss_dict

    def forward(self, outputs, targets, current_step):
        self._log_counter += 1
        should_log = (self.log_interval > 0 and self._log_counter % self.log_interval == 0)

        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        
        if should_log:
            log_info(f"--- Loss Calculation (Step: {self._log_counter}) ---")
        
        # 1. 计算主输出的损失
        loss_final, loss_dict = self.compute_loss(outputs_without_aux, targets, current_step, should_log)
        
        # 初始化用于日志的字典，并记录主损失分量
        losses_to_log = {k: v for k, v in loss_dict.items()}
        
        # 创建损失列表，用于后续聚合
        loss_list = [loss_final * self.ds_loss_weights[0]]
        losses_to_log['total_loss_main'] = loss_list[0].detach()

        # 2. 如果启用深监督，则计算辅助损失
        if self.use_deep_supervision and "aux_outputs" in outputs:
            # 按照参考代码，反向遍历辅助输出
            for i, aux_outputs in enumerate(outputs["aux_outputs"][::-1]):
                loss_aux, loss_dict_aux = self.compute_loss(aux_outputs, targets, current_step, should_log=False)
                
                # 记录未加权的辅助损失分量
                for k, v in loss_dict_aux.items():
                    losses_to_log[f'{k}_aux_{i}'] = v
                
                # 计算加权后的辅助损失
                weighted_loss_aux = loss_aux * self.ds_loss_weights[i + 1]
                loss_list.append(weighted_loss_aux)
                
                # 记录加权后的辅助层总损失
                losses_to_log[f'total_loss_aux_{i}'] = weighted_loss_aux.detach()
        
        # 3. 根据聚合类型计算最终总损失
        if len(loss_list) > 1: # 仅当存在辅助损失时才需要聚合
            if self.ds_avg_type == 'sum': # 对应 v2
                total_loss = torch.sum(torch.stack(loss_list))
            elif self.ds_avg_type == 'mean_all': # 对应 v0
                total_loss = torch.mean(torch.stack(loss_list))
            elif self.ds_avg_type == 'mean_aux': # 对应 v1
                if len(loss_list) > 1:
                    total_loss = (loss_list[0] + torch.mean(torch.stack(loss_list[1:]))) / 2
                else: # 只有一个损失，直接用
                    total_loss = loss_list[0]
            else:
                raise ValueError(f"Unknown ds_avg_type: '{self.ds_avg_type}'")
        else: # 没有辅助损失或禁用了深监督
            total_loss = loss_list[0]

        if should_log:
            log_info(f"--- Final Total Loss (DS type: '{self.ds_avg_type}'): {total_loss.item():.4f} ---")

        return total_loss, losses_to_log

