import os
import sys
import yaml
import torch
import collections
import numpy as np
import shutil
from tqdm import tqdm
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler

# 假设所有必要的模块都位于正确的路径下
from logger import setup_logger, log_info
from utils import (
    set_seed, load_config, build_optimizer, build_scheduler, 
    build_edl_components, process_batch_for_expert_class_combination,
    save_checkpoint, apply_args_override, setup_training_args_parser, 
    save_override_record,
)
# --- 简化的指标计算 ---
from metrics import (
    calculate_soft_dice, calculate_riga_metrics
)

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
               best_metrics,  # 新增参数
               scheduler=None):
    """
    主训练循环 - 基础MaskFormer版本。
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
            
            # 基础MaskFormer训练 - 适配2N数据格式
            images = batch['image'].cuda(non_blocking=True) # (B, C, H, W)
            expert_masks = batch['expert_masks'].cuda(non_blocking=True)  # (B, N, 2, H, W)
            expert_targets, _ = process_batch_for_expert_class_combination(batch)

            with autocast(device_type='cuda', enabled=(scaler is not None)):
                outputs = model(images)
                loss_total = loss_fn(outputs, expert_targets)
            
            if scaler is not None:
                scaler.scale(loss_total).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_total.backward()
                optimizer.step()

            # 记录损失
            epoch_losses['total'] += loss_total.item()
            num_batches_processed += 1
            
            pbar_train.set_postfix({
                'loss': f"{loss_total.item():.4f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
            })
        
        # 平均训练损失
        if num_batches_processed > 0:
            for key in epoch_losses:
                epoch_losses[key] /= num_batches_processed
        
        log_info(f"Epoch {epoch+1} Training Summary:", print_message=True)
        log_info(f"  - Avg Train Loss: Total={epoch_losses['total']:.6f}", print_message=True)

        # --- 验证与模型保存 ---
        log_info(f"Starting validation for Epoch {epoch+1}...", print_message=True)
        val_metrics = validate(model, val_dataloader, config, dataset_type)
        
        current_metric_score = val_metrics['soft_dice']['mean']
        
        log_info(f"Epoch {epoch+1} Validation Results:", print_message=True)
        log_info(f"  - Samples: {val_metrics.get('num_samples', 'N/A')}, Experts: {val_metrics.get('num_experts', 'N/A')}", print_message=True)
        log_info("-" * 70, print_message=True)
        
        # 统一的指标显示
        soft_dice = val_metrics['soft_dice']
        log_info(f"  [BEST METRIC] Soft Dice Mean: {soft_dice['mean']:.6f} (越高越好)", print_message=True)
        
        if dataset_type == 'RIGA':
            # RIGA数据集：显示disc/cup指标
            log_info(f"    - Per Class: Disc={soft_dice.get('disc', 0):.4f}, Cup={soft_dice.get('cup', 0):.4f}", print_message=True)
        else:
            # 其他数据集
            if 'background' in soft_dice:
                log_info(f"    - Per Class: BG={soft_dice.get('background', 0):.4f}, Ring={soft_dice.get('ring', 0):.4f}, Cup={soft_dice.get('cup', 0):.4f}", print_message=True)
        
        log_info("-" * 70, print_message=True)
        
        # 更新best指标的判断和记录
        is_best = current_metric_score > best_metrics['soft_dice_mean']
        if is_best:
            best_metrics['soft_dice_mean'] = current_metric_score
            best_metrics['epoch'] = epoch
            best_metrics['full_metrics'] = val_metrics.copy()  # 保存完整的指标
            
            # 更新各个best分类指标
            if dataset_type == 'RIGA':
                best_metrics['soft_dice_disc'] = soft_dice.get('disc', 0)
                best_metrics['soft_dice_cup'] = soft_dice.get('cup', 0)
            
            # 与旧的best_metric保持同步
            best_metric = current_metric_score
            
            log_info(f"  *** New best model found! (Epoch {epoch+1}) ***", print_message=True)
            log_info(f"      Best Soft Dice Mean: {best_metrics['soft_dice_mean']:.6f}", print_message=True)
            if dataset_type == 'RIGA':
                log_info(f"      Best Disc: {best_metrics['soft_dice_disc']:.6f}, Best Cup: {best_metrics['soft_dice_cup']:.6f}", print_message=True)
        
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict() if scaler else None,
            'best_metric': best_metric,
            'best_metrics': best_metrics,  # 保存详细的best指标
            'val_metrics': val_metrics,
            'train_avg_loss': dict(epoch_losses)
        }, is_best, config['save_dir'], epoch+1, last_checkpoints, max_checkpoints)
        
        if scheduler is not None:
            scheduler.step()
        log_info(f"--- Epoch {epoch + 1}/{total_epochs} finished. Current LR: {optimizer.param_groups[0]['lr']:.8f} ---\n", print_message=True)

    # ===== 训练完成后的最终指标输出和保存 =====
    log_info("\n" + "="*80, print_message=True)
    log_info("TRAINING COMPLETED - FINAL RESULTS SUMMARY", print_message=True)
    log_info("="*80, print_message=True)
    
    # 输出最终的best指标结果
    log_info(f"BEST EPOCH: {best_metrics['epoch'] + 1}", print_message=True)
    log_info(f"BEST SOFT DICE MEAN: {best_metrics['soft_dice_mean']:.6f}", print_message=True)
    if dataset_type == 'RIGA' and best_metrics['full_metrics']:
        best_soft_dice = best_metrics['full_metrics']['soft_dice']
        log_info(f"BEST DETAILED METRICS:", print_message=True)
        log_info(f"  - Disc Soft Dice: {best_soft_dice.get('disc', 0):.6f}", print_message=True)
        log_info(f"  - Cup Soft Dice: {best_soft_dice.get('cup', 0):.6f}", print_message=True)
    
    # 输出最后一次训练的指标结果
    log_info(f"LAST EPOCH METRIC: {current_metric_score:.6f} (Soft Dice Mean)", print_message=True)
    
    # 详细的最后一次验证指标
    log_info("\nLAST EPOCH DETAILED METRICS:", print_message=True)
    log_info(f"  - Samples: {val_metrics.get('num_samples', 'N/A')}, Experts: {val_metrics.get('num_experts', 'N/A')}", print_message=True)
    
    last_soft_dice = val_metrics['soft_dice']
    log_info(f"  - Soft Dice Mean: {last_soft_dice['mean']:.6f}", print_message=True)
    
    if dataset_type == 'RIGA':
        # RIGA数据集：显示disc/cup指标
        log_info(f"  - Per Class: Disc={last_soft_dice.get('disc', 0):.4f}, Cup={last_soft_dice.get('cup', 0):.4f}", print_message=True)
    else:
        # 其他数据集
        if 'background' in last_soft_dice:
            log_info(f"  - Per Class: BG={last_soft_dice.get('background', 0):.4f}, Ring={last_soft_dice.get('ring', 0):.4f}, Cup={last_soft_dice.get('cup', 0):.4f}", print_message=True)
    
    # 保存最终指标到txt文件
    save_dir = config['save_dir']
    final_results_path = os.path.join(save_dir, 'final_training_results.txt')
    
    with open(final_results_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("TRAINING COMPLETED - FINAL RESULTS SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Training Configuration:\n")
        f.write(f"  - Total Epochs: {total_epochs}\n")
        f.write(f"  - Dataset Type: {dataset_type}\n")
        f.write(f"  - Batch Size: {config.get('batch_size', 'N/A')}\n")
        f.write(f"  - Model: Simple MaskFormer\n")
        f.write(f"  - Force Matching: {config.get('model', {}).get('force_matching', 'N/A')}\n")
        f.write(f"  - Expert Classification: {config.get('model', {}).get('expert_classification', 'N/A')}\n\n")
        
        f.write(f"BEST EPOCH: {best_metrics['epoch'] + 1}\n\n")
        
        f.write(f"BEST SOFT DICE MEAN: {best_metrics['soft_dice_mean']:.6f}\n\n")
        
        if dataset_type == 'RIGA' and best_metrics['full_metrics']:
            best_soft_dice = best_metrics['full_metrics']['soft_dice']
            f.write("BEST DETAILED METRICS:\n")
            f.write(f"  - Best Disc Soft Dice: {best_soft_dice.get('disc', 0):.6f}\n")
            f.write(f"  - Best Cup Soft Dice: {best_soft_dice.get('cup', 0):.6f}\n\n")
        
        f.write(f"LAST EPOCH METRIC: {current_metric_score:.6f} (Soft Dice Mean)\n\n")
        
        f.write("LAST EPOCH DETAILED METRICS:\n")
        f.write(f"  - Total Samples: {val_metrics.get('num_samples', 'N/A')}\n")
        f.write(f"  - Number of Experts: {val_metrics.get('num_experts', 'N/A')}\n")
        f.write(f"  - Soft Dice Mean: {last_soft_dice['mean']:.6f}\n")
        
        if dataset_type == 'RIGA':
            f.write(f"  - Disc Soft Dice: {last_soft_dice.get('disc', 0):.6f}\n")
            f.write(f"  - Cup Soft Dice: {last_soft_dice.get('cup', 0):.6f}\n")
        else:
            if 'background' in last_soft_dice:
                f.write(f"  - Background Soft Dice: {last_soft_dice.get('background', 0):.6f}\n")
                f.write(f"  - Ring Soft Dice: {last_soft_dice.get('ring', 0):.6f}\n")
                f.write(f"  - Cup Soft Dice: {last_soft_dice.get('cup', 0):.6f}\n")
        
        f.write(f"\n")
        f.write(f"Training completed successfully!\n")
        f.write(f"Results saved to: {save_dir}\n")
    
    log_info(f"\nFinal results saved to: {final_results_path}", print_message=True)
    log_info("="*80, print_message=True)
    
    return best_metrics


def validate(model, dataloader, config, dataset_type):
    """验证循环 - 基础MaskFormer版本。"""
    def semantic_inference(output, current_dataset_type, none_object, expert_classification=False):
        # 基础MaskFormer语义推理
        pred_logits = output['pred_logits']  # (B, num_queries, num_classes) 
        pred_masks = output['pred_masks']    # (B, num_queries, H, W)
        
        # 简单处理：使用所有queries的平均预测作为最终结果
        # 或者可以根据confidence选择最佳query
        pred_probs = torch.softmax(pred_logits, dim=-1)  # (B, num_queries, num_classes)
        
        # 在专家分类模式下，需要特别处理non_object
        if none_object and not expert_classification:
            # 只在非专家分类模式下去掉non_object类别
            pred_probs = pred_probs[..., :-1]
        elif none_object and expert_classification:
            # 专家分类模式下，non_object是第13个类别，需要保留前12个专家类别
            pred_probs = pred_probs[..., :-1]  # 去掉non_object，保留12个专家类别
        
        pred_masks = pred_masks.sigmoid()
        semseg = torch.einsum("bqc,bqhw->bchw", pred_probs, pred_masks)
        return semseg  # (B, num_classes, H, W)
    
    # 从配置中获取模式参数
    model_config = config.get('model', {})
    force_matching = model_config.get('force_matching', True)
    expert_classification = model_config.get('expert_classification', False)
    non_object = model_config.get('decoder_non_object', True)
    
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating', leave=False, ncols=120, file=sys.stdout):
            images = batch['image'].cuda(non_blocking=True)
            # 使用验证标签: [B, N, 2, H, W] - 保持专家维度用于指标计算
            labels_for_validation = batch['val_masks'].cuda(non_blocking=True)  # [B, N, 2, H, W]
            
            # 模型推理
            outputs = model(images)
            
            # 根据模式进行不同的推理处理
            num_experts = labels_for_validation.shape[1]  # N
            
            if force_matching and expert_classification:
                # 固定匹配 + 专家分类：预测专家ID（0-2N）
                # 模型输出应该是 [B, 2N, H, W]，对应专家0-N的disc和专家0-N的cup
                preds_semantic = semantic_inference(outputs, dataset_type, non_object, expert_classification=True)  # [B, 2N, H, W]
                B, num_out, H, W = preds_semantic.shape
                
                if num_out == 2 * num_experts:
                    # 按照expert_labels的组织方式重排：0-N是disc，N-2N是cup
                    disc_preds = preds_semantic[:, :num_experts, :, :]      # [B, N, H, W]
                    cup_preds = preds_semantic[:, num_experts:, :, :]       # [B, N, H, W]
                    preds_for_validation = torch.stack([disc_preds, cup_preds], dim=2)  # [B, N, 2, H, W]
                else:
                    raise ValueError(f"Expert classification mode expects {2 * num_experts} output channels, got {num_out}")
                    
            elif not force_matching and expert_classification:
                # 匈牙利匹配 + 专家分类：动态匹配专家预测
                # 输出0-11的12个专家类别，按预测类别索引构建N2hw标签
                preds_semantic = semantic_inference(outputs, dataset_type, non_object, expert_classification=True)  # [B, 12, H, W]
                B, num_out, H, W = preds_semantic.shape
                
                # 确保输出维度至少为12个专家类别
                expected_expert_classes = 2 * num_experts  # 12个专家类别 (6个disc + 6个cup)
                if num_out >= expected_expert_classes:
                    # 方案：按预测类别索引构建专家预测 
                    # 类别0-5对应专家0-5的disc，类别6-11对应专家0-5的cup
                    
                    # 获取每个专家类别的预测结果
                    expert_class_preds = preds_semantic[:, :expected_expert_classes, :, :]  # [B, 12, H, W]
                    
                    # 重新组织为[专家, 类别]格式
                    disc_expert_preds = expert_class_preds[:, :num_experts, :, :]      # [B, 6, H, W] - 专家0-5的disc
                    cup_expert_preds = expert_class_preds[:, num_experts:expected_expert_classes, :, :]  # [B, 6, H, W] - 专家0-5的cup
                    
                    preds_for_validation = torch.stack([disc_expert_preds, cup_expert_preds], dim=2)  # [B, N, 2, H, W]
                else:
                    raise ValueError(f"Hungarian + expert classification mode expects at least {expected_expert_classes} expert classes, got {num_out}")
                    
            else:
                # mask分类模式（固定匹配或匈牙利匹配）：预测mask类别（disc, cup）
                # 这种模式下不区分专家，输出应该是语义级别的预测
                preds_semantic = semantic_inference(outputs, dataset_type, non_object, expert_classification=False)  # [B, num_classes, H, W]
                B, num_out, H, W = preds_semantic.shape
                
                if num_out >= 2:
                    # 为mask分类模式实现真正的语义分割
                    # 从所有queries中提取最佳的语义预测
                    disc_pred = preds_semantic[:, 0, :, :].unsqueeze(1)  # [B, 1, H, W] 
                    cup_pred = preds_semantic[:, 1, :, :].unsqueeze(1)   # [B, 1, H, W]
                    
                    # 为每个专家生成相同的语义预测（因为不区分专家）
                    disc_preds = disc_pred.expand(B, num_experts, H, W)  # [B, N, H, W]
                    cup_preds = cup_pred.expand(B, num_experts, H, W)    # [B, N, H, W]
                    preds_for_validation = torch.stack([disc_preds, cup_preds], dim=2)  # [B, N, 2, H, W]
                else:
                    raise ValueError(f"Mask classification mode expects at least 2 output channels, got {num_out}")
            
            # 收集验证数据
            all_preds.append(preds_for_validation)
            all_labels.append(labels_for_validation)

    all_preds = torch.cat(all_preds, dim=0)  # [Total_B, N, 2, H, W]
    all_labels = torch.cat(all_labels, dim=0)  # [Total_B, N, 2, H, W]

    # 计算指标
    if dataset_type == 'RIGA':
        # 使用RIGA专用便利函数计算所有指标
        riga_results = calculate_riga_metrics(all_labels, all_preds, is_test=True)
        
        return {
            "soft_dice": riga_results['soft_dice'],
            "num_samples": len(all_labels), 
            "num_experts": all_labels.shape[1]  # 报告专家数量
        }
    else:
        # 其他数据集的通用处理
        soft_dice_results = calculate_soft_dice(all_labels, all_preds)
        
        return {
            "soft_dice": soft_dice_results,
            "num_samples": len(all_labels), 
            "num_experts": all_labels.shape[1]  # 报告专家数量
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
    
    setup_logger(config['save_dir'], log_name='train_log')
    save_override_record(override_params, args, config['save_dir'])

    log_info("="*80 + "\nSimple MaskFormer Training Session Started\n" + "="*80, print_message=True)
    
    dataloader, val_dataloader = build_dataloaders(config)
    model, loss_fn = build_edl_components(config)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    scaler = GradScaler(device='cuda') if config.get('use_amp', False) else None
    
    last_checkpoints, start_epoch, best_metric = collections.deque(), 0, -np.inf
    # 初始化详细的best指标记录
    best_metrics = {
        'soft_dice_mean': -np.inf,
        'soft_dice_disc': -np.inf,
        'soft_dice_cup': -np.inf,
        'epoch': 0,
        'full_metrics': None
    }
    
    resume_path = config.get('resume')
    if resume_path and os.path.isfile(resume_path):
        checkpoint = torch.load(resume_path, map_location='cuda')
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if scaler and 'scaler' in checkpoint and checkpoint.get('scaler'):
            scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_metric = checkpoint.get('best_metric', -np.inf)
        # 恢复详细的best指标
        best_metrics = checkpoint.get('best_metrics', {
            'soft_dice_mean': best_metric,
            'soft_dice_disc': -np.inf,
            'soft_dice_cup': -np.inf,
            'epoch': start_epoch - 1,
            'full_metrics': None
        })
        log_info(f"✓ Resumed training from checkpoint: {resume_path} (epoch {start_epoch})", print_message=True)
    else:
        log_info("Starting training from scratch.", print_message=True)

    final_best_metrics = train_loop(model, dataloader, val_dataloader, optimizer, loss_fn, scaler, config, 
               last_checkpoints, start_epoch, best_metric, best_metrics, scheduler)
    
    log_info(f"Training session completed with final best soft dice mean: {final_best_metrics['soft_dice_mean']:.6f} (Epoch {final_best_metrics['epoch'] + 1})", print_message=True)

if __name__ == '__main__':
    parser = setup_training_args_parser()
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    config = load_config(args.config_path)
    config['dataset_yaml'] = args.dataset_yaml
    config, override_params = apply_args_override(config, args)
    
    main(config, args, override_params)