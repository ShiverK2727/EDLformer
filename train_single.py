import sys
import os
import yaml
import torch
import shutil
import collections
import json
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler

# 假设所有必要的模块都位于正确的路径下
from logger import setup_logger, log_info
from nets.edl_process import process_edlmaskformer_outputs
from utils import (
    set_seed, load_config,
    build_optimizer, build_scheduler, build_dataloaders,
    build_edl_components, save_checkpoint,
    apply_args_override, setup_training_args_parser, save_override_record,
)
# --- [关键] 确保从新的metrics文件中导入 ---
from metrics import (
    convert_ring_cup_to_disc_cup, convert_labels_ring_cup_to_disc_cup,
    calculate_personalization_metrics, calculate_ged, calculate_soft_dice, calculate_riga_metrics
)

# =============================================================================
# 辅助函数
# =============================================================================

class ProcessedOutputIndex:
    """定义从process_edlmaskformer_outputs返回的元组的常量索引，避免魔术数字。"""
    EVIDENCE = 0
    PROBABILITIES = 1
    ALPHA = 2
    UNCERTAINTY = 3

def extract_processed_outputs(processed_outputs):
    """
    辅助函数：安全地从处理后的输出中提取所需的张量。
    """
    result = {}
    if 'expert_masks' in processed_outputs:
        expert_tuple = processed_outputs['expert_masks']
        result['expert_alpha'] = expert_tuple[ProcessedOutputIndex.ALPHA]
        result['expert_probs'] = expert_tuple[ProcessedOutputIndex.PROBABILITIES]
    if 'class_predictions' in processed_outputs:
        cls_tuple = processed_outputs['class_predictions']
        result['cls_alpha'] = cls_tuple[ProcessedOutputIndex.ALPHA]
        result['cls_probs'] = cls_tuple[ProcessedOutputIndex.PROBABILITIES]
    pixel_tuple = processed_outputs.get('pixel_masks', (None,)*4)
    result['pixel_alpha'] = pixel_tuple[ProcessedOutputIndex.ALPHA]
    return result

# =============================================================================
# 核心训练与验证循环
# =============================================================================

def train_loop(model, 
               train_dataloader, 
               val_dataloader, 
               optimizer, 
               loss_fn, 
               scaler, 
               config, 
               last_checkpoints, 
               start_epoch, 
               best_metric, 
               scheduler=None):
    """
    主训练循环。
    """
    max_checkpoints = config.get('max_checkpoints', 3)
    total_epochs = config['epochs']
    dataset_type = config.get('dataset_type', 'RIGA')

    for epoch in range(start_epoch, total_epochs):
        log_info(f"--- Starting Epoch {epoch + 1}/{total_epochs} ---", print_message=True)
        
        model.train()
        epoch_losses = collections.defaultdict(float)
        num_batches_processed = 0
        
        pbar_train = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{total_epochs} [Train]', leave=True, ncols=120, file=sys.stdout)
        
        for i, batch in enumerate(pbar_train):
            if not batch or batch['image'].size(0) == 0:
                continue

            optimizer.zero_grad(set_to_none=True)
            
            # [证据学习监督] 使用3通道one-hot标签进行训练
            images = batch['image'].cuda(non_blocking=True)
            target_masks = batch['expert_masks'].cuda(non_blocking=True)      # (B, N, 3, H, W) [bg, ring, cup]
            target_labels = batch['expert_labels'].cuda(non_blocking=True)     # 类别标签
            consensus_mask = batch['consensus_mask'].cuda(non_blocking=True)    # 共识掩码

            with autocast(device_type='cuda', enabled=(scaler is not None)):
                raw_outputs = model(images)
                processed_outputs = process_edlmaskformer_outputs(
                    model_outputs=raw_outputs,
                    cls_target_type=config['model'].get('decoder_cls_target_type', 'single'),
                    activation=config['model'].get('edl_activation', 'softplus'),
                    process_aux=True
                )
                extracted = extract_processed_outputs(processed_outputs)
                
                loss_total, loss_cls, loss_exp, loss_dice, loss_con, loss_con_dice, loss_consistency = loss_fn(
                    raw_cls_logits=raw_outputs['pred_logits'],
                    raw_mask_logits=raw_outputs['pred_masks'],
                    expert_alpha=extracted['expert_alpha'],
                    cls_alpha=extracted['cls_alpha'],
                    pixel_alpha=extracted['pixel_alpha'],
                    pixel_logits=raw_outputs.get('pix_pred_masks'),
                    target_masks=target_masks,
                    target_labels=target_labels,
                    consensus_target=consensus_mask, 
                    epoch=epoch + 1
                )
            
            if scaler is not None:
                scaler.scale(loss_total).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('grad_clip', 1.0))
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('grad_clip', 1.0))
                optimizer.step()

            for k, v in [('total', loss_total), ('cls', loss_cls), ('exp', loss_exp), ('dice', loss_dice), ('con', loss_con), ('con_dice', loss_con_dice), ('consistency', loss_consistency)]:
                epoch_losses[k] += v.item()
            num_batches_processed += 1
            
            pbar_train.set_postfix({
                'loss': f"{loss_total.item():.4f}", 'cls': f"{loss_cls.item():.4f}",
                'exp': f"{loss_exp.item():.4f}", 'dice': f"{loss_dice.item():.4f}",
                'con': f"{loss_con.item():.4f}", 'con_d': f"{loss_con_dice.item():.4f}",
                'consis': f"{loss_consistency.item():.4f}", 'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
            })
        
        if num_batches_processed > 0:
            for key in epoch_losses:
                epoch_losses[key] /= num_batches_processed
        
        log_info(f"Epoch {epoch+1} Training Summary:", print_message=True)
        log_info(f"  - Avg Train Loss: Total={epoch_losses['total']:.6f} | Cls={epoch_losses['cls']:.6f} | Exp={epoch_losses['exp']:.6f} | Dice={epoch_losses['dice']:.6f} | Con={epoch_losses['con']:.6f} | ConDice={epoch_losses['con_dice']:.6f} | Consis={epoch_losses['consistency']:.6f}", print_message=True)

        # --- 验证与模型保存 ---
        log_info(f"Starting validation for Epoch {epoch+1}...", print_message=True)
        val_metrics = validate(model, val_dataloader, config, dataset_type)
        
        current_metric_score = val_metrics['soft_dice']['mean']
        
        log_info(f"Epoch {epoch+1} Validation Results:", print_message=True)
        log_info(f"  - Samples: {val_metrics.get('num_samples', 'N/A')}, Experts: {val_metrics.get('num_experts', 'N/A')}", print_message=True)
        log_info("-" * 70, print_message=True)
        
        # --- [修复] 统一的disc/cup指标显示 ---
        soft_dice = val_metrics['soft_dice']
        log_info(f"  [BEST METRIC] Soft Dice Mean: {soft_dice['mean']:.6f} (越高越好)", print_message=True)
        
        # 适配新的personalization数据结构
        if dataset_type == 'RIGA':
            # RIGA数据集：显示disc/cup指标
            log_info(f"    - Per Class: Disc={soft_dice.get('disc', 0):.4f}, Cup={soft_dice.get('cup', 0):.4f}", print_message=True)
            
            p = val_metrics['personalization']  # 这是riga_results的内容
            log_info(f"  [Personalization Metrics - Disc/Cup]", print_message=True)
            log_info(f"    - Dice Max   (Overall): {p['dice_max']['overall']:.4f} | (Disc: {p['dice_max'].get('disc', 0):.4f}, Cup: {p['dice_max'].get('cup', 0):.4f})", print_message=True)
            log_info(f"    - Dice Match (Overall): {p['dice_match']['overall']:.4f} | (Disc: {p['dice_match'].get('disc', 0):.4f}, Cup: {p['dice_match'].get('cup', 0):.4f})", print_message=True)
            for class_name, scores in p['dice_per_expert'].items():
                scores_str = ", ".join([f"{s:.4f}" for s in scores])
                log_info(f"    - Dice Per Expert ({class_name.capitalize()}): [{scores_str}]", print_message=True)
            
            # 显示RIGA专用的disc/cup软dice指标
            riga_soft_dice = p['soft_dice']
            log_info(f"  [RIGA Detailed Metrics]", print_message=True)
            log_info(f"    - Internal Soft Dice: Mean={riga_soft_dice['mean']:.4f} | Disc={riga_soft_dice.get('disc', 0):.4f}, Cup={riga_soft_dice.get('cup', 0):.4f}", print_message=True)
        else:
            # 其他数据集：使用标准personalization结构
            if 'background' in soft_dice:
                log_info(f"    - Per Class: BG={soft_dice.get('background', 0):.4f}, Others=..." , print_message=True)
            p = val_metrics['personalization']
            log_info(f"  [Personalization Metrics]", print_message=True)
            log_info(f"    - Dice Max   (Overall): {p['dice_max']['overall']:.4f}", print_message=True)
            log_info(f"    - Dice Match (Overall): {p['dice_match']['overall']:.4f}", print_message=True)
            for class_name, scores in p['dice_per_expert'].items():
                scores_str = ", ".join([f"{s:.4f}" for s in scores])
                log_info(f"    - Dice Per Expert ({class_name.capitalize()}): [{scores_str}]", print_message=True)
            
        log_info(f"  [Diversity] GED Score: {val_metrics['ged']:.6f} (越低越好)", print_message=True)
        log_info("-" * 70, print_message=True)
        
        is_best = current_metric_score > best_metric
        if is_best:
            best_metric = current_metric_score
            log_info(f"  *** New best model found! (based on Soft Dice Mean). Best: {best_metric:.6f} ***", print_message=True)
        
        save_checkpoint({ 'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict() if scaler else None, 'best_metric': best_metric, 'val_metrics': val_metrics, 'train_avg_loss': dict(epoch_losses)}, is_best, config['save_dir'], epoch+1, last_checkpoints, max_checkpoints)
        
        if scheduler is not None:
            scheduler.step()
        log_info(f"--- Epoch {epoch + 1}/{total_epochs} finished. Current LR: {optimizer.param_groups[0]['lr']:.8f} ---\n", print_message=True)


def validate(model, dataloader, config, dataset_type):
    """验证循环，适配新的指标返回格式。"""
    def prediction_to_masks(model_outputs, config, dataset_type):
        """
        将SimpleMaskFormer的输出转换为评估所需的格式
        
        Args:
            model_outputs: SimpleMaskFormer的输出，包含pred_logits和pred_masks
            config: 配置文件
            dataset_type: 数据集类型
            
        Returns:
            5D tensor (B, N, C, H, W) - 预测的专家级别mask
        """
        pred_masks = model_outputs["pred_masks"]  # (B, num_queries, H, W)
        pred_logits = model_outputs["pred_logits"]  # (B, num_queries, num_classes)
        
        batch_size, num_queries, H, W = pred_masks.shape
        num_experts = config['model']['num_experts']
        
        # 获取类别预测
        pred_probs = torch.softmax(pred_logits, dim=-1)  # (B, num_queries, num_classes)
        
        # Sigmoid mask概率
        mask_probs = torch.sigmoid(pred_masks)  # (B, num_queries, H, W)
        
        if config['model']['expert_classification']:
            # 专家分类模式：queries预测专家ID，需要重组为专家级输出
            # 在这种模式下，每个query对应一个专家标注
            num_classes = 2  # RIGA的disc和cup
            if num_queries == 2 * num_experts:
                # 2N个queries：前N个是disc，后N个是cup
                semseg = torch.zeros(batch_size, num_experts, num_classes, H, W, device=pred_masks.device)
                
                for batch_idx in range(batch_size):
                    for query_idx in range(num_queries):
                        # 获取这个query最可能的专家ID
                        expert_id = torch.argmax(pred_probs[batch_idx, query_idx]).item()
                        if expert_id < num_experts:  # 有效的专家ID
                            class_id = 0 if query_idx < num_experts else 1  # disc或cup
                            semseg[batch_idx, expert_id, class_id] = mask_probs[batch_idx, query_idx]
            else:
                log_info(f"Warning: Expected {2*num_experts} queries for expert classification, got {num_queries}")
                # 退回到简单分配
                semseg = torch.zeros(batch_size, num_experts, num_classes, H, W, device=pred_masks.device)
                for i in range(min(num_queries, num_experts * num_classes)):
                    expert_idx = i % num_experts
                    class_idx = i // num_experts
                    if class_idx < num_classes:
                        semseg[:, expert_idx, class_idx] = mask_probs[:, i]
        else:
            # 语义分类模式：queries预测语义类别，需要复制到所有专家
            num_classes = pred_logits.shape[-1]
            if config['model'].get('non_object', True):
                num_classes -= 1  # 排除背景类
            
            semseg = torch.zeros(batch_size, num_experts, num_classes, H, W, device=pred_masks.device)
            
            for batch_idx in range(batch_size):
                for query_idx in range(min(num_queries, num_classes * num_experts)):
                    # 获取这个query最可能的语义类别
                    class_id = torch.argmax(pred_probs[batch_idx, query_idx]).item()
                    if class_id < num_classes:  # 有效的语义类别
                        # 将这个mask分配给所有专家的对应类别
                        expert_idx = query_idx % num_experts
                        semseg[batch_idx, expert_idx, class_id] = mask_probs[batch_idx, query_idx]
        
        # 对于RIGA，确保输出是2通道（disc, cup）
        if dataset_type == 'RIGA' and semseg.shape[2] != 2:
            log_info(f"Warning: Expected 2 classes for RIGA, got {semseg.shape[2]}")
            # 如果是3通道，转换为2通道
            if semseg.shape[2] == 3:
                # 假设是 [background, disc, cup] -> [disc, cup]
                semseg = semseg[:, :, 1:3, :, :]
        
        return semseg
    
    model.eval()
    all_preds_disc_cup, all_labels_disc_cup = [], []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating', leave=False, ncols=120, file=sys.stdout):
            images = batch['image'].cuda(non_blocking=True)
            # [验证指标对齐] 使用2通道disc/cup格式与其他方法对齐
            labels_for_validation = batch['val_masks'].cuda(non_blocking=True)  # (B, N, 2, H, W) [disc, cup] - 已经是正确格式
            
            preds_for_validation = prediction_to_masks(model(images), config, dataset_type)  # 获取专家级预测
            
            # 收集验证数据 - 全部使用2通道格式
            all_preds_disc_cup.append(preds_for_validation)  # 2通道预测
            all_labels_disc_cup.append(labels_for_validation)  # 2通道标签（无需转换）

    all_preds_disc_cup = torch.cat(all_preds_disc_cup, dim=0)
    all_labels_disc_cup = torch.cat(all_labels_disc_cup, dim=0)

    # [修复] 统一使用2通道disc/cup格式计算所有指标
    if dataset_type == 'RIGA':
        # 使用RIGA专用便利函数计算所有指标
        riga_results = calculate_riga_metrics(all_labels_disc_cup, all_preds_disc_cup, is_test=True)
        
        return {
            "soft_dice": riga_results['soft_dice'],  # 统一的disc/cup指标
            "ged": riga_results['ged'],
            "personalization": riga_results,  # 包含dice_max, dice_match, dice_per_expert
            "num_samples": len(all_labels_disc_cup), 
            "num_experts": all_labels_disc_cup.shape[1]
        }
    else:
        # 其他数据集的通用处理
        soft_dice_results = calculate_soft_dice(all_labels_disc_cup, all_preds_disc_cup)
        ged_score = calculate_ged(all_labels_disc_cup, all_preds_disc_cup)
        personalization_results = calculate_personalization_metrics(labels=all_labels_disc_cup, preds=all_preds_disc_cup, is_test=True)
        
        return {
            "soft_dice": soft_dice_results,
            "ged": ged_score,
            "personalization": personalization_results,
            "num_samples": len(all_labels_disc_cup), 
            "num_experts": all_labels_disc_cup.shape[1]
        }

def main(config, args, override_params):
    """主函数。"""
    set_seed(config.get('seed', 42))
    
    os.makedirs(config['save_dir'], exist_ok=True)
    shutil.copy(args.config_path, os.path.join(config['save_dir'], 'original_train_config.yaml'))
    shutil.copy(args.dataset_yaml, os.path.join(config['save_dir'], 'original_dataset_config.yaml'))
    
    final_config_path = os.path.join(config['save_dir'], 'final_train_config.yaml')
    with open(final_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)
    
    setup_logger(config['save_dir'], log_name='train_log.txt')
    save_override_record(override_params, args, config['save_dir'])

    log_info("="*80 + "\nEDL MaskFormer Training Session Started (Detailed Logging v3)\n" + "="*80, print_message=True)
    
    dataloader, val_dataloader = build_dataloaders(config)
    model, loss_fn = build_edl_components(config)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    scaler = GradScaler(device='cuda') if config.get('use_amp', False) else None
    
    last_checkpoints, start_epoch, best_metric = collections.deque(), 0, -np.inf
    
    resume_path = config.get('resume')
    if resume_path and os.path.isfile(resume_path):
        checkpoint = torch.load(resume_path, map_location='cuda')
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if scaler and 'scaler' in checkpoint and checkpoint.get('scaler'):
            scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_metric = checkpoint.get('best_metric', -np.inf)
        log_info(f"✓ Resumed training from checkpoint: {resume_path} (epoch {start_epoch})", print_message=True)
    else:
        log_info("Starting training from scratch.", print_message=True)

    train_loop(model, dataloader, val_dataloader, optimizer, loss_fn, scaler, config, 
               last_checkpoints, start_epoch, best_metric, scheduler)

if __name__ == '__main__':
    parser = setup_training_args_parser()
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    config = load_config(args.config_path)
    config['dataset_yaml'] = args.dataset_yaml
    config, override_params = apply_args_override(config, args)
    
    main(config, args, override_params)
