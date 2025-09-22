import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, List

def apply_evidence_activation(logits: torch.Tensor, activation: str = 'softplus') -> torch.Tensor:
    """Applies an activation function to convert logits to non-negative evidence."""
    if activation == 'softplus':
        return F.softplus(logits)
    elif activation == 'elu':
        # elu(x) + 1 ensures non-negativity, as elu can be -1.
        return F.elu(logits) + 1.0
    elif activation == 'exp':
        return torch.exp(logits)
    else:
        raise ValueError(f"Unsupported activation function: {activation}")

def compute_dirichlet_predictions(
    evidence: torch.Tensor,
    dirichlet_prior: float = 1.0,
    epsilon: float = 1e-10
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes predictions and uncertainty from evidence based on the Dirichlet distribution.
    
    Returns:
        Tuple: (probabilities, alpha, uncertainty, total_strength, belief_masses)
    """
    num_classes = evidence.shape[-1]
    alpha = evidence + dirichlet_prior
    
    total_strength = torch.sum(alpha, dim=-1, keepdim=True)
    
    probabilities = alpha / (total_strength + epsilon)
    uncertainty = num_classes / (total_strength + epsilon)
    belief_masses = evidence / (total_strength + epsilon)

    return probabilities, alpha, uncertainty, total_strength, belief_masses

def compute_beta_predictions(
    pos_evidence: torch.Tensor,
    neg_evidence: torch.Tensor,
    beta_prior_weight: float = 2.0,
    epsilon: float = 1e-10
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Computes predictions and uncertainty from evidence based on the Beta distribution.
    
    Returns:
        Tuple: (probabilities, [alpha, beta], uncertainty, total_strength, [belief, disbelief])
    """
    half_prior = beta_prior_weight / 2.0
    alpha = pos_evidence + half_prior
    beta = neg_evidence + half_prior
    
    total_strength = alpha + beta
    
    probabilities = alpha / (total_strength + epsilon)
    
    prior_weight_tensor = torch.full_like(total_strength, beta_prior_weight)
    uncertainty = prior_weight_tensor / (total_strength + epsilon)
    
    # Belief and disbelief masses
    belief_masses = torch.clamp((alpha - half_prior) / (total_strength - prior_weight_tensor + epsilon), min=0.0)
    disbelief_masses = torch.clamp((beta - half_prior) / (total_strength - prior_weight_tensor + epsilon), min=0.0)
    
    return probabilities, (alpha, beta), uncertainty, total_strength, (belief_masses, disbelief_masses)

def _process_one_output_set(
    outputs: Dict[str, torch.Tensor],
    cls_target_type: str,
    activation: str,
    dirichlet_prior: float,
    beta_prior_weight: float,
    epsilon: float
) -> Dict[str, Tuple]:
    """辅助函数，用于处理单组预测（主输出或一个辅助输出）。"""
    
    processed_results = {}

    # --- 1. 处理专家级分割掩码 (pred_masks) ---
    if 'pred_masks' in outputs:
        expert_logits = outputs['pred_masks'] # (B, Q, C, H, W)
        # 为了进行逐像素计算，将类别维度移到最后
        expert_logits_spatial = expert_logits.permute(0, 1, 3, 4, 2) # (B, Q, H, W, C)
        expert_evidence = apply_evidence_activation(expert_logits_spatial, activation)
        # 计算Dirichlet分布结果
        exp_probs, exp_alpha, exp_unc, exp_S, exp_belief = compute_dirichlet_predictions(
            expert_evidence, dirichlet_prior, epsilon
        )
        # 将结果维度恢复原状
        processed_results['expert_masks'] = (
            expert_evidence.permute(0, 1, 4, 2, 3), # evidence
            exp_probs.permute(0, 1, 4, 2, 3),       # probabilities
            exp_alpha.permute(0, 1, 4, 2, 3),       # distribution_params (alpha)
            exp_unc.permute(0, 1, 4, 2, 3),         # uncertainty
        )

    # --- 2. 处理分类Logits (pred_logits) ---
    if 'pred_logits' in outputs:
        cls_logits = outputs['pred_logits']
        if cls_target_type == 'single':
            # 单标签分类 (Dirichlet)
            cls_evidence = apply_evidence_activation(cls_logits, activation)
            cls_probs, cls_alpha, cls_unc, _, _ = compute_dirichlet_predictions(
                cls_evidence, dirichlet_prior, epsilon
            )
            processed_results['class_predictions'] = (cls_evidence, cls_probs, cls_alpha, cls_unc)
        else:
            # 多标签分类 (Beta)
            pos_evidence = apply_evidence_activation(cls_logits[0], activation)
            neg_evidence = apply_evidence_activation(cls_logits[1], activation)
            cls_probs, cls_alpha_beta, cls_unc, _, _ = compute_beta_predictions(
                pos_evidence, neg_evidence, beta_prior_weight, epsilon
            )
            cls_evidence = (pos_evidence, neg_evidence)
            processed_results['class_predictions'] = (cls_evidence, cls_probs, cls_alpha_beta, cls_unc)

    # --- 3. 处理像素级分割掩码 (pix_pred_masks) ---
    if 'pix_pred_masks' in outputs:
        pixel_logits = outputs['pix_pred_masks'] # (B, C, H, W)
        pixel_logits_spatial = pixel_logits.permute(0, 2, 3, 1) # (B, H, W, C)
        pixel_evidence = apply_evidence_activation(pixel_logits_spatial, activation)
        pix_probs, pix_alpha, pix_unc, _, _ = compute_dirichlet_predictions(
            pixel_evidence, dirichlet_prior, epsilon
        )
        processed_results['pixel_masks'] = (
            pixel_evidence.permute(0, 3, 1, 2), # evidence
            pix_probs.permute(0, 3, 1, 2),      # probabilities
            pix_alpha.permute(0, 3, 1, 2),      # distribution_params (alpha)
            pix_unc.permute(0, 3, 1, 2),        # uncertainty
        )
        
    return processed_results

def process_edlmaskformer_outputs(
    model_outputs: Dict[str, Any],
    cls_target_type: str,
    activation: str = 'softplus',
    dirichlet_prior: float = 1.0,
    beta_prior_weight: float = 2.0,
    epsilon: float = 1e-10,
    process_aux: bool = True
) -> Dict[str, Any]:
    """
    一个高级接口函数，用于完整处理EDLMaskFormer模型的输出字典。
    它将原始logits转换为证据，并计算概率、分布参数和不确定性。

    Args:
        model_outputs (Dict[str, Any]): EDLMaskFormer模型前向传播的原始输出。
        cls_target_type (str): 分类任务的类型, 'single' 或 'multi'。
        activation (str): 用于将logits转换为证据的激活函数。
        dirichlet_prior (float): Dirichlet分布的先验参数。
        beta_prior_weight (float): Beta分布的先验权重 (W)。
        epsilon (float): 用于防止除零错误的数值稳定常数。
        process_aux (bool): 是否也处理深度监督的辅助输出。

    Returns:
        Dict[str, Any]: 一个结构与输入相似的字典，但原始logits被替换为
                        包含(evidence, probabilities, distribution_params, uncertainty)
                        的元组。
    """
    # 处理主输出
    final_results = _process_one_output_set(
        model_outputs, cls_target_type, activation, dirichlet_prior, beta_prior_weight, epsilon
    )
    
    # 如果需要，处理辅助输出
    if process_aux and 'aux_outputs' in model_outputs:
        aux_results = []
        for aux_output in model_outputs['aux_outputs']:
            aux_processed = _process_one_output_set(
                aux_output, cls_target_type, activation, dirichlet_prior, beta_prior_weight, epsilon
            )
            aux_results.append(aux_processed)
        final_results['aux_outputs'] = aux_results
        
    return final_results

