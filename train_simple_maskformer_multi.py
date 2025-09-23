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
    set_seed, load_config,
    build_optimizer, build_scheduler, build_dataloaders_multi,
    build_simple_maskformer_multi_components, save_checkpoint,
    apply_args_override, setup_training_args_parser, save_override_record,
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
    主训练循环 - Multi版本MaskFormer。
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
            
            # Multi版本MaskFormer训练 - 适配BN2HW数据格式
            images = batch['image'].cuda(non_blocking=True)
            
            # 新的Multi数据格式适配 - 来自RIGADatasetSimpleMulti
            target_expert_masks = batch['expert_masks'].cuda(non_blocking=True)    # (B, N, 2, H, W) - N个专家，每个专家有disc+cup
            target_expert_labels = batch['expert_labels'].cuda(non_blocking=True)  # (B, N) - 专家ID序列
            target_mask_labels = batch.get('mask_labels', None)  # Multi版本可能不需要mask_labels

            with autocast(device_type='cuda', enabled=(scaler is not None)):
                # 前向传播
                outputs = model(images)
                
                # 准备目标数据格式 - 适配Multi版本的损失函数
                targets = []
                batch_size = images.shape[0]
                
                for b in range(batch_size):
                    batch_targets = {
                        'expert_masks': target_expert_masks[b],    # (N, 2, H, W)
                        'expert_labels': target_expert_labels[b],  # (N,)
                    }
                    targets.append(batch_targets)
                
                # 计算损失
                loss_total = loss_fn(outputs, targets)
            
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
        f.write(f"  - Model: Simple MaskFormer Multi\n")
        f.write(f"  - Model Type: {config.get('model', {}).get('model_type', 'multi')}\n")
        f.write(f"  - Loss Type: {config.get('loss', {}).get('loss_type', 'auto')}\n\n")
        
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
    """验证循环 - Multi版本MaskFormer。"""
    def semantic_inference(output, none_object):
        # Multi版本MaskFormer语义推理
        pred_logits = output['pred_logits']  # (B, num_queries, num_cls_classes) 
        pred_masks = output['pred_masks']    # (B, num_queries, num_seg_classes, H, W)

        if none_object:
            # 去掉non_object类别，但保持专家数量一致
            pred_probs = pred_logits[..., :-1].softmax(dim=-1)  # (B, num_queries, num_experts)
        else:
            pred_probs = pred_logits.softmax(dim=-1)  # (B, num_queries, num_experts)
        
        pred_masks = pred_masks.sigmoid()
        # 严格按照用户要求的einstein求和：torch.einsum("bqn,bqchw->bnchw", pred_probs, pred_masks)
        # 这里 n=num_experts(应该与标签匹配), c=num_seg_classes(2)
        semseg = torch.einsum("bqn,bqchw->bnchw", pred_probs, pred_masks)
        
        # 确保输出与标签的专家数量一致（6个专家）
        if semseg.shape[1] != 6:
            # 如果模型输出的专家数少于期望的6个，用零填充
            batch_size, _, num_classes, height, width = semseg.shape
            padded_semseg = torch.zeros(batch_size, 6, num_classes, height, width, 
                                      device=semseg.device, dtype=semseg.dtype)
            padded_semseg[:, :semseg.shape[1]] = semseg
            semseg = padded_semseg
        
        return semseg  # (B, 6, 2, H, W) - 确保6个专家
    
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating', leave=False, ncols=120, file=sys.stdout):
            images = batch['image'].cuda(non_blocking=True)
            # Multi版本使用验证标签: [B, N, 2, H, W] - 保持专家维度用于指标计算
            labels_for_validation = batch['val_masks'].cuda(non_blocking=True)  # [B, N, 2, H, W]
            
            # 模型推理
            outputs = model(images)
            
            # Multi版本推理处理：使用einstein求和获取语义分割结果
            # 严格按照用户提供的验证过程执行
            preds_semantic = semantic_inference(outputs, none_object=config.get('model', {}).get('decoder_non_object', True))  # [B, N, C, H, W]
            
            # preds_semantic 现在是 [B, 6, 2, H, W] 格式，直接与 val_masks [B, N, 2, H, W] 计算指标
            preds_for_validation = preds_semantic  # [B, N, 2, H, W]
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
    
    setup_logger(config['save_dir'], log_name='train_log.txt')
    save_override_record(override_params, args, config['save_dir'])

    log_info("="*80 + "\nSimple MaskFormer Multi Training Session Started\n" + "="*80, print_message=True)
    
    dataloader, val_dataloader = build_dataloaders_multi(config)
    model, loss_fn = build_simple_maskformer_multi_components(config)
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