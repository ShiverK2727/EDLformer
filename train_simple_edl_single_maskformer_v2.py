import os
import sys
import yaml
import torch
import torch.nn.functional as F
import collections
import numpy as np
import shutil
from tqdm import tqdm
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler

# 导入重构后的、功能更全面的utils模块
from utils_v2 import (
    set_seed, load_config, build_complete_training_setup,
    process_batch_for_expert_class_combination,
    save_checkpoint, apply_args_override, setup_training_args_parser,
    save_override_record, TensorboardLogger, log_epoch_summary, save_final_results
)
from metrics import calculate_riga_metrics
from logger import setup_logger, log_info


def train_one_epoch(model, train_loader, optimizer, loss_fn, scaler, logger, epoch, total_epochs):
    """一个训练轮次的循环。"""
    model.train()
    epoch_losses = collections.defaultdict(float)
    
    pbar_train = tqdm(
        train_loader, desc=f'Epoch {epoch+1}/{total_epochs} [Train]', 
        leave=True, ncols=150, file=sys.stdout)
    
    for i, batch in enumerate(pbar_train):
        if not batch or batch['image'].size(0) == 0:
            continue

        global_step = epoch * len(train_loader) + i
        optimizer.zero_grad(set_to_none=True)
        
        images = batch['image'].cuda(non_blocking=True)
        # 使用优化版的数据处理器
        targets, batch_metadata = process_batch_for_expert_class_combination(batch)

        with autocast(device_type='cuda', enabled=(scaler is not None)):
            outputs = model(images)
            total_loss, loss_dict = loss_fn(outputs, targets, global_step)
        
        if scaler is not None:
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()

        # 更新并记录损失 - 确保所有损失都被记录
        loss_dict['total_loss'] = total_loss.detach()
        
        # 记录所有损失分量到TensorBoard
        logger.log_training_losses(loss_dict, global_step)
        
        # 累计epoch损失
        for k, v in loss_dict.items():
            epoch_losses[k] += v.item()

        # 显示详细的损失信息
        loss_components = ', '.join([f"{k}: {v.item():.4f}" for k, v in loss_dict.items() if k != 'total_loss'])
        pbar_train.set_postfix({
            'total': f"{total_loss.item():.4f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.2e}",
            'sub': loss_components[:50] + '...' if len(loss_components) > 50 else loss_components
        })
        
    # 计算平均损失
    num_batches = len(train_loader)
    avg_epoch_losses = {k: v / num_batches for k, v in epoch_losses.items()}
    
    # 记录epoch平均损失到日志
    log_info(f"Epoch {epoch+1} 平均训练损失:", print_message=True)
    for loss_name, loss_value in avg_epoch_losses.items():
        log_info(f"  {loss_name}: {loss_value:.6f}", print_message=True)
    
    return avg_epoch_losses


def validate(model, dataloader, dataset_metadata, config):
    """验证函数，计算模型在验证集上的性能指标。
    
    Args:
        model: 训练好的模型
        dataloader: 验证数据加载器
        dataset_metadata: 全局数据集元数据（由build_complete_training_setup返回）
        config: 训练配置
    """
    
    def semantic_inference(outputs, metadata, config):
        """
        根据模型配置和输出进行语义推理，将查询预测转换为专家x类别的语义分割图。
        
        Args:
            outputs: 模型输出字典，包含 'pred_logits' 和 'pred_masks'
            metadata: 数据集元数据，包含专家和类别信息
            config: 训练配置，用于确定推理模式
            
        Returns:
            semseg: (B, num_experts, num_classes, H, W) 的语义分割结果
        """
        # 从配置中获取损失函数设置，确定推理模式
        loss_config = config.get('loss', {})
        cls_loss_type = loss_config.get('cls_loss_type', 'edl')
        seg_edl_as_dirichlet = loss_config.get('seg_edl_as_dirichlet', False)
        non_object = loss_config.get('non_object', True)
        
        # 提取模型输出
        pred_logits_list = outputs['pred_logits']  # [alpha_cls, beta_cls] 或 [alpha_cls, None]
        pred_masks_list = outputs['pred_masks']    # [alpha_mask, beta_mask]
        
        alpha_cls, beta_cls = pred_logits_list[0], pred_logits_list[1]
        alpha_mask, beta_mask = pred_masks_list[0], pred_masks_list[1]
        
        # 1. 处理分类分支的推理
        if cls_loss_type == 'ce':
            # CE模式：直接使用logits进行softmax
            pred_probs = torch.softmax(alpha_cls, dim=-1)
        else:  # 'edl'
            # EDL模式：使用Dirichlet分布
            # 对于分类分支，只使用alpha参数构成Dirichlet分布
            evidence = F.softplus(alpha_cls)
            alpha = evidence + 1
            S = torch.sum(alpha, dim=-1, keepdim=True)
            pred_probs = alpha / S
        
        # 移除背景类（如果存在）
        num_total_classes = metadata['total_combined_classes']
        if non_object and pred_probs.shape[-1] > num_total_classes:
            # 移除最后一个背景类
            pred_probs = pred_probs[..., :num_total_classes]
        
        # 2. 处理分割分支的推理
        if seg_edl_as_dirichlet:
            # Dirichlet模式：对每个查询位置，将alpha和beta evidence堆叠为2通道
            # alpha_mask: (B, Q, H, W), beta_mask: (B, Q, H, W)
            # 堆叠为 (B, Q, 2, H, W)，其中通道0=背景evidence(beta)，通道1=前景evidence(alpha)
            evidence = torch.stack([beta_mask, alpha_mask], dim=2)  # (B, Q, 2, H, W)
            alpha_dirichlet = F.softplus(evidence) + 1  # (B, Q, 2, H, W)
            S = torch.sum(alpha_dirichlet, dim=2, keepdim=True)  # (B, Q, 1, H, W)
            # 取前景通道（channel 1）的概率
            pred_masks_prob = alpha_dirichlet[:, :, 1:2, :, :] / S  # (B, Q, 1, H, W)
            pred_masks_prob = pred_masks_prob.squeeze(2)  # (B, Q, H, W)
        else:
            # Beta分布模式：alpha和beta直接构成Beta分布
            alpha_seg = F.softplus(alpha_mask) + 1
            beta_seg = F.softplus(beta_mask) + 1
            pred_masks_prob = alpha_seg / (alpha_seg + beta_seg)
        
        # 3. 语义推理：将查询预测转换为语义分割图
        # pred_probs: (B, num_queries, total_combined_classes)
        # pred_masks_prob: (B, num_queries, H, W)
        semseg = torch.einsum("bqc,bqhw->bchw", pred_probs, pred_masks_prob)
        
        # 4. 将组合类别转换回专家x类别结构
        # semseg: (B, total_combined_classes, H, W) -> (B, num_experts, num_classes, H, W)
        batch_size, _, height, width = semseg.shape
        num_experts = metadata['num_experts']
        num_classes = metadata['num_seg_classes']
        
        semseg_reshaped = semseg.view(batch_size, num_experts, num_classes, height, width)
        
        return semseg_reshaped
    
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating', leave=False, ncols=120, file=sys.stdout):
            images = batch['image'].cuda(non_blocking=True)
            labels_for_validation = batch['expert_masks'].cuda(non_blocking=True)
            # 不需要targets，直接使用全局dataset_metadata避免重复计算
            
            outputs = model(images)
            # 使用全局dataset_metadata进行语义推理
            semseg_reshaped = semantic_inference(outputs, dataset_metadata, config)
            
            # 添加到结果列表
            all_preds.append(semseg_reshaped.cpu())
            all_labels.append(labels_for_validation.cpu())
    
    # 连接所有批次的结果
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    return calculate_riga_metrics(all_labels, all_preds)


def train_loop(config, args=None):
    """主训练循环。"""
    # 1. 构建所有组件
    setup_result = build_complete_training_setup(config, args)
    model, loss_fn, train_loader, val_loader, optimizer, scheduler, dataset_metadata, final_config, override_params = setup_result
    
    # 2. 初始化训练状态和工具
    save_dir = config['experiment']['save_dir']
    logger = TensorboardLogger(save_dir)
    scaler = GradScaler(device='cuda') if config['training']['use_amp'] else None
    
    start_epoch = 0
    best_metric_score = -np.inf
    last_checkpoints = collections.deque()

    # 恢复训练功能
    if args and args.resume:
        if os.path.exists(args.resume):
            try:
                from utils_v2 import load_checkpoint
                start_epoch, best_metric_score = load_checkpoint(
                    args.resume, model, optimizer, scaler
                )
                log_info(f"成功恢复训练，从第 {start_epoch + 1} 轮开始", print_message=True)
            except Exception as e:
                log_info(f"恢复训练失败: {str(e)}", print_message=True)
                log_info("将从头开始训练", print_message=True)
        else:
            log_info(f"检查点文件不存在: {args.resume}", print_message=True)
            log_info("将从头开始训练", print_message=True)

    # 3. 开始训练循环
    total_epochs = config['training']['epochs']
    best_metrics = {'epoch': -1, 'metrics': {'soft_dice': {'mean': -np.inf}}}
    last_metrics = {'epoch': -1, 'metrics': {'soft_dice': {'mean': 0}}}
    
    for epoch in range(start_epoch, total_epochs):
        
        # 训练
        avg_train_losses = train_one_epoch(model, train_loader, optimizer, loss_fn, scaler, logger, epoch, total_epochs)
        
        # 验证
        val_metrics = validate(model, val_loader, dataset_metadata, config)
        
        # 记录日志
        current_lr = optimizer.param_groups[0]['lr']
        logger.log_validation_metrics(val_metrics, avg_train_losses, current_lr, epoch)
        log_epoch_summary(epoch, total_epochs, avg_train_losses, val_metrics)
        
        # 保存模型
        current_metric_score = val_metrics['soft_dice']['mean']
        is_best = current_metric_score > best_metric_score
        if is_best:
            best_metric_score = current_metric_score
            best_metrics = {'epoch': epoch, 'metrics': val_metrics}
            log_info(f"*** New best model found! Epoch {epoch+1}, Soft Dice Mean: {best_metric_score:.6f} ***", print_message=True)
        
        # 更新最后一轮epoch的指标
        last_metrics = {'epoch': epoch, 'metrics': val_metrics}

        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict() if scaler else None,
            'best_metric_score': best_metric_score,
            'config': config,
            'dataset_metadata': dataset_metadata
        }, is_best, save_dir, epoch + 1, last_checkpoints, config['experiment']['max_checkpoints'])
        
        if scheduler:
            scheduler.step()
    
    # 保存最终结果摘要
    save_final_results(save_dir, best_metrics, last_metrics, config)
    
    logger.close()
    log_info("="*80, print_message=True)
    log_info("训练完成！", print_message=True)
    log_info(f"最佳模型保存在: {os.path.join(save_dir, 'best_model.pth')}", print_message=True)
    
    # 详细的最佳和最后epoch指标对比
    log_info("", print_message=True)  # 空行
    log_info("--- 最终指标摘要 ---", print_message=True)
    
    # 最佳epoch指标
    best_epoch_metrics = best_metrics['metrics']
    log_info(f"最佳 Epoch {best_metrics['epoch'] + 1}:", print_message=True)
    log_info(f"  - Soft Dice: {best_epoch_metrics['soft_dice']['mean']:.6f} (disc: {best_epoch_metrics['soft_dice'].get('disc', 0):.4f}, cup: {best_epoch_metrics['soft_dice'].get('cup', 0):.4f})", print_message=True)
    
    if 'dice_max' in best_epoch_metrics:
        dice_max = best_epoch_metrics['dice_max']
        log_info(f"  - Dice Max: {dice_max.get('overall', 0):.6f} (disc: {dice_max.get('disc', 0):.4f}, cup: {dice_max.get('cup', 0):.4f})", print_message=True)
    
    if 'dice_match' in best_epoch_metrics:
        dice_match = best_epoch_metrics['dice_match']
        log_info(f"  - Dice Match: {dice_match.get('overall', 0):.6f} (disc: {dice_match.get('disc', 0):.4f}, cup: {dice_match.get('cup', 0):.4f})", print_message=True)
    
    # 最后epoch指标
    last_epoch_metrics = last_metrics['metrics']
    log_info(f"最后 Epoch {last_metrics['epoch'] + 1}:", print_message=True)
    log_info(f"  - Soft Dice: {last_epoch_metrics['soft_dice']['mean']:.6f} (disc: {last_epoch_metrics['soft_dice'].get('disc', 0):.4f}, cup: {last_epoch_metrics['soft_dice'].get('cup', 0):.4f})", print_message=True)
    
    if 'dice_max' in last_epoch_metrics:
        dice_max = last_epoch_metrics['dice_max']
        log_info(f"  - Dice Max: {dice_max.get('overall', 0):.6f} (disc: {dice_max.get('disc', 0):.4f}, cup: {dice_max.get('cup', 0):.4f})", print_message=True)
    
    if 'dice_match' in last_epoch_metrics:
        dice_match = last_epoch_metrics['dice_match']
        log_info(f"  - Dice Match: {dice_match.get('overall', 0):.6f} (disc: {dice_match.get('disc', 0):.4f}, cup: {dice_match.get('cup', 0):.4f})", print_message=True)
    
    log_info("", print_message=True)  # 空行
    log_info(f"TensorBoard 日志保存在: {save_dir}", print_message=True)
    log_info(f"最终结果摘要保存在: {os.path.join(save_dir, 'final_training_summary.txt')}", print_message=True)
    log_info("="*80, print_message=True)


if __name__ == '__main__':
    parser = setup_training_args_parser()
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # 加载和处理配置
    config = load_config(args.config_path)
    config, override_params = apply_args_override(config, args)
    
    # 设置随机种子（确保在所有操作之前）
    set_seed(config['training'].get('seed', 42))

    # 初始化保存目录和日志
    save_dir = config['experiment']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    setup_logger(save_dir, log_name='train_log.txt')
    
    # 保存完整的文件集合：原始配置、实际训练参数、覆写记录
    log_info("保存配置文件...", print_message=True)
    
    # 1. 原始配置文件
    shutil.copy(args.config_path, os.path.join(save_dir, 'original_config.yaml'))
    log_info(f"原始配置已保存: {os.path.join(save_dir, 'original_config.yaml')}", print_message=True)
    
    # 2. 实际训练参数文件
    with open(os.path.join(save_dir, 'final_config.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False, indent=2)
    log_info(f"实际训练参数已保存: {os.path.join(save_dir, 'final_config.yaml')}", print_message=True)
    
    # 3. 覆写参数记录
    if override_params:
        save_override_record(config, override_params, args, save_dir)
    else:
        log_info("无参数覆写。", print_message=True)
    
    # 记录训练开始信息
    log_info("="*80, print_message=True)
    log_info("开始 EDL MaskFormer v2 训练...", print_message=True)
    log_info(f"实验目录: {save_dir}", print_message=True)
    log_info(f"随机种子: {config['training'].get('seed', 42)}", print_message=True)
    log_info(f"训练轮数: {config['training']['epochs']}", print_message=True)
    log_info(f"批处理大小: {config['training']['batch_size']}", print_message=True)
    log_info(f"学习率: {config['training']['learning_rate']}", print_message=True)
    log_info("="*80, print_message=True)
        
    train_loop(config, args)
