import os
import sys
import yaml
import torch
import random
import numpy as np
import datetime
from typing import List, Dict, Tuple, Any
from torch.utils.data import DataLoader
from collections import deque
import shutil
import json
import argparse

# --- [核心修改] 导入所有需要的模型，包括新的统一模型和旧模型 ---
# from nets.simple_maskformer import SimpleMaskFormer, SimpleMaskFormerMulti
from nets.simple_edl_maskformer import FlexibleEDLMaskFormer

# --- [核心修改] 导入两种模式对应的EDL损失函数 ---
# from scheduler.edl_single_loss import EDLSingleLoss
# from scheduler.simple_maskformer_loss import SimpleMaskformerLoss
# from scheduler.simple_maskformer_multi_loss import SimpleMaskFormerMultiLoss
from scheduler.simple_edl_single_maskformer_loss import SimpleEDLMaskformerLossV2

# 导入数据集和其他辅助模块
from datasets import RIGADatasetSimpleV2
from logger import log_info, log_error
from torch.utils.tensorboard import SummaryWriter

# ==============================================================================
# 基础设置 (Setup)
# ==============================================================================

def set_seed(seed):
    """设置随机种子以确保实验的可重复性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    log_info(f"随机种子已设置为: {seed}", print_message=False)

def load_config(yaml_path):
    """加载YAML配置文件。"""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

# ==============================================================================
# [核心修改] 统一的组件构建器
# ==============================================================================

def build_edl_components(config):
    """根据配置构建模型、损失函数和数据集，并返回元数据。"""
    model_config = config['model']
    loss_config = config.get('loss', {})
    training_config = config.get('training', {})
    dataset_config = config.get('dataset', {})
    dataset_yaml = dataset_config.get('config_path')
    # 修复相对路径问题
    if dataset_yaml and not os.path.isabs(dataset_yaml):
        dataset_yaml = os.path.join('/app/MultiAnn/EDLformer', dataset_yaml.lstrip('./'))
    if not dataset_yaml or not os.path.exists(dataset_yaml):
        raise FileNotFoundError(f"数据集配置文件不存在: {dataset_yaml}")
    train_dataset = RIGADatasetSimpleV2(config_path=dataset_yaml, is_train=True)
    val_dataset = RIGADatasetSimpleV2(config_path=dataset_yaml, is_train=False)
    train_loader = DataLoader(train_dataset, batch_size=training_config.get('batch_size', 8), shuffle=True, num_workers=training_config.get('num_workers', 4), pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=training_config.get('num_workers', 4), pin_memory=True)
    sample_data = train_dataset[0]
    expert_masks_shape = sample_data["expert_masks"].shape
    log_info(f"数据集expert_masks形状: {expert_masks_shape}", print_message=True)
    
    if len(expert_masks_shape) == 4:
        # 形状: (num_experts, num_seg_classes, H, W)
        num_experts, num_seg_classes, _, _ = expert_masks_shape
    elif len(expert_masks_shape) == 5:
        # 形状: (1, num_experts, num_seg_classes, H, W)
        _, num_experts, num_seg_classes, _, _ = expert_masks_shape
    else:
        raise ValueError(f"不支持的expert_masks形状: {expert_masks_shape}")
    total_combined_classes = num_experts * num_seg_classes
    dataset_metadata = {'num_experts': num_experts, 'num_seg_classes': num_seg_classes, 'total_combined_classes': total_combined_classes}
    log_info(f"数据集构建完成。动态推断: {num_experts}专家, {num_seg_classes}分割类别, {total_combined_classes}组合类别。", print_message=True)
    if model_config.get('num_cls_classes') != total_combined_classes:
        log_info(f"警告: 模型配置'num_cls_classes'({model_config.get('num_cls_classes')})与推断值({total_combined_classes})不匹配。", print_message=True)
    model = FlexibleEDLMaskFormer(**model_config)
    log_info("模型构建完成。", print_message=True)
    loss_params = {k: v for k, v in loss_config.items()}
    loss_params['num_cls_classes'] = total_combined_classes
    loss_fn = SimpleEDLMaskformerLossV2(**loss_params)
    log_info("损失函数构建完成。", print_message=True)
    return model.cuda(), loss_fn, train_loader, val_loader, dataset_metadata

def build_complete_training_setup(config, args):
    """构建完整的训练设置，并应用参数覆写。"""
    log_info("="*80, print_message=True)
    log_info("开始构建完整训练设置...", print_message=True)
    
    # [关键改动] 应用命令行参数覆写
    config, override_params = apply_args_override(config, args)
    if override_params:
        print_override_summary(override_params)
    
    model, loss_fn, train_loader, val_loader, dataset_metadata = build_edl_components(config)
    optimizer, scheduler = build_optimizer_and_scheduler(model, config)
    
    log_info("完整训练设置构建完成。", print_message=True)
    log_info(f"🎯 实验保存目录: {config.get('experiment', {}).get('save_dir', 'N/A')}", print_message=True)
    log_info("="*80, print_message=True)
    
    # [关键改动] 返回覆写记录，供主函数保存
    return model, loss_fn, train_loader, val_loader, optimizer, scheduler, dataset_metadata, config, override_params



def build_optimizer_and_scheduler(model, config):
    """统一构建优化器和学习率调度器，使用单一学习率配置。"""
    training_cfg = config.get('training', {})
    
    # 统一学习率配置 - 强制使用单个学习率声明
    base_lr = float(training_cfg.get('learning_rate', 1e-3))
    log_info(f"使用统一学习率配置: {base_lr}", print_message=True)
    
    # 优化器配置
    optimizer_type = training_cfg.get('optimizer_type', 'AdamW')
    weight_decay = float(training_cfg.get('weight_decay', 1e-4))
    
    # 构建优化器
    if optimizer_type.lower() == 'adamw':
        betas = training_cfg.get('betas', [0.9, 0.999])
        if isinstance(betas, list) and len(betas) == 2:
            betas = (float(betas[0]), float(betas[1]))
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=base_lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=float(training_cfg.get('eps', 1e-8)),
            amsgrad=bool(training_cfg.get('amsgrad', False))
        )
        log_info(f"构建AdamW优化器: lr={base_lr}, weight_decay={weight_decay}", print_message=False)
        
    elif optimizer_type.lower() == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=base_lr,
            weight_decay=weight_decay
        )
        log_info(f"构建Adam优化器: lr={base_lr}, weight_decay={weight_decay}", print_message=False)
        
    elif optimizer_type.lower() == 'sgd':
        momentum = float(training_cfg.get('momentum', 0.9))
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=base_lr,
            weight_decay=weight_decay,
            momentum=momentum
        )
        log_info(f"构建SGD优化器: lr={base_lr}, weight_decay={weight_decay}, momentum={momentum}", print_message=False)
    else:
        raise ValueError(f"未知的优化器类型: {optimizer_type}")
    
    # 学习率调度器配置
    use_scheduler = training_cfg.get('use_cosine_scheduler', False)
    if not use_scheduler:
        log_info("不使用学习率调度器。", print_message=False)
        return optimizer, None
    
    # 构建余弦退火调度器
    # 优先从 training 节点获取 epochs，否则从顶级节点获取
    total_epochs = training_cfg.get('epochs', config.get('epochs', 100))
    warmup_epochs = int(training_cfg.get('warmup_epochs', 0))
    min_lr = float(training_cfg.get('min_learning_rate', 1e-6))
    
    def lr_lambda(epoch):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            # 线性warmup
            return float(epoch + 1) / float(max(1, warmup_epochs))
        
        # 余弦退火
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1. + np.cos(np.pi * progress))
        return cosine * (1.0 - min_lr / base_lr) + min_lr / base_lr
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    log_info(f"构建余弦退火调度器: warmup_epochs={warmup_epochs}, min_lr={min_lr}", print_message=False)
    
    return optimizer, scheduler


def process_batch_for_expert_class_combination(
    batch: Dict[str, torch.Tensor]
) -> Tuple[List[Dict[str, torch.Tensor]], Dict[str, Any]]:
    """
    处理来自多专家标注数据集的批次数据，将(专家, 类别)组合视为新类别。

    此函数旨在为类 MaskFormer 模型准备 'targets'，同时生成用于结果重构的元数据。

    Args:
        batch (Dict[str, torch.Tensor]): 
            从 DataLoader 输出的批次字典，至少需要包含:
            - 'expert_masks': torch.Tensor, 形状为 [B, N, C, H, W]
              B: 批次大小, N: 专家数, C: 原始类别数, H, W: 掩码高宽。
              掩码应为二值 (0 或 1)。

    Returns:
        Tuple[List[Dict[str, torch.Tensor]], Dict[str, Any]]:
        - targets (List[Dict[str, torch.Tensor]]):
            一个列表，长度为B。每个元素是一个字典，包含:
            - 'labels': 1D Tensor, 形状为 [N*C]，包含新生成的唯一类别ID。
            - 'masks': 3D Tensor, 形状为 [N*C, H, W]，包含对应的二值掩码。
            此格式可直接用于 DETR/MaskFormer 类的损失函数。

        - metadata (Dict[str, Any]):
            一个元数据字典，用于后续结果解析和重构，包含:
            - 'num_experts' (int): 原始专家数 N。
            - 'num_classes' (int): 原始类别数 C。
            - 'total_combined_classes' (int): N * C，组合后的总类别数。
            - 'mapping_tensor': 2D Tensor, 形状为 [N*C, 2]。
              每一行 `mapping_tensor[new_label]` 包含了 `[expert_id, class_id]`，
              可以将新的组合标签反向映射回原始的专家和类别ID。
    """
    # 1. 从批次中提取数据并获取维度信息
    # 假设 expert_masks 已经是 float 类型的二值 (0/1) Tensor
    expert_masks = batch['expert_masks']
    device = expert_masks.device
    
    # 动态获取维度，使其能够自适应不同数据集
    B, N, C, H, W = expert_masks.shape
    
    # 2. 生成新的组合类别标签
    # 创建一个从 0 到 N*C-1 的连续整数作为新的标签
    # 形状为 (B, N*C)
    total_combined_classes = N * C
    new_labels = torch.arange(total_combined_classes, device=device).repeat(B, 1)
    # print(f"new_labels: {new_labels}")

    # 3. 重塑掩码以匹配新标签
    # 将专家和类别维度合并
    # [B, N, C, H, W] -> [B, N*C, H, W]
    reshaped_masks = expert_masks.view(B, total_combined_classes, H, W)

    # 4. 构建损失函数所需的 `targets` 列表
    targets = []
    for i in range(B):
        targets.append({
            "labels": new_labels[i],
            "masks": reshaped_masks[i]
        })

    # 5. 构建用于反向映射和重构的元数据
    # 创建一个映射张量，大小为 (N*C, 2)
    # mapping_tensor[new_label] = [expert_id, class_id]
    
    # [0, 0, ..., 1, 1, ..., N-1, N-1, ...] (每个重复C次)
    expert_ids_map = torch.arange(N, device=device).view(N, 1).repeat(1, C).view(-1)
    # print(f"expert_ids_map: {expert_ids_map}")
    
    # [0, 1, ..., C-1, 0, 1, ..., C-1, ...] (重复N次)
    class_ids_map = torch.arange(C, device=device).repeat(N)
    # print(f"class_ids_map: {class_ids_map}")

    # 将它们堆叠成 [N*C, 2] 的映射关系
    mapping_tensor = torch.stack([expert_ids_map, class_ids_map], dim=1)
    # print(f"mapping_tensor: {mapping_tensor}")

    metadata = {
        'num_experts': N,
        'num_classes': C,
        'total_combined_classes': total_combined_classes,
        'mapping_tensor': mapping_tensor
    }

    return targets, metadata


# ==============================================================================
# 命令行参数处理
# ==============================================================================


def setup_training_args_parser():
    parser = argparse.ArgumentParser(description='Flexible EDL MaskFormer Training')
    
    # 基础配置
    parser.add_argument('--config_path', type=str, required=True, help='训练配置文件路径')
    parser.add_argument('--gpu', type=str, default='0', help='GPU设备ID')
    
    # 实验配置覆写
    parser.add_argument('--save_dir', type=str, help='覆写保存目录')
    parser.add_argument('--max_checkpoints', type=int, help='覆写最大检查点数量')
    
    # 数据集配置覆写
    parser.add_argument('--dataset_config_path', type=str, help='覆写数据集配置文件路径')
    parser.add_argument('--dataset_type', type=str, help='覆写数据集类型')
    
    # 训练配置覆写
    parser.add_argument('--seed', type=int, help='覆写随机种子')
    parser.add_argument('--epochs', type=int, help='覆写训练轮数')
    parser.add_argument('--batch_size', type=int, help='覆写批处理大小')
    parser.add_argument('--use_amp', type=bool, help='覆写是否使用混合精度')
    
    # 学习率和优化器配置覆写
    parser.add_argument('--learning_rate', '--lr', type=float, help='覆写学习率')
    parser.add_argument('--min_learning_rate', type=float, help='覆写最小学习率')
    parser.add_argument('--optimizer_type', type=str, choices=['AdamW', 'Adam', 'SGD'], help='覆写优化器类型')
    parser.add_argument('--weight_decay', type=float, help='覆写权重衰减')
    parser.add_argument('--warmup_epochs', type=int, help='覆写预热轮数')
    parser.add_argument('--use_cosine_scheduler', type=bool, help='覆写是否使用余弦调度器')
    
    # 其他
    parser.add_argument('--resume', type=str, help='从检查点恢复训练的路径')

    return parser


def apply_args_override(config, args):
    """应用命令行参数覆写YAML配置，支持嵌套配置结构。"""
    override_params = {}
    original_values = {}
    
    # 定义参数到配置节点的映射
    param_mapping = {
        # 实验配置
        'save_dir': ('experiment', 'save_dir'),
        'max_checkpoints': ('experiment', 'max_checkpoints'),
        
        # 数据集配置
        'dataset_config_path': ('dataset', 'config_path'),
        'dataset_type': ('dataset', 'type'),
        
        # 训练配置
        'seed': ('training', 'seed'),
        'epochs': ('training', 'epochs'),
        'batch_size': ('training', 'batch_size'),
        'use_amp': ('training', 'use_amp'),
        'learning_rate': ('training', 'learning_rate'),
        'min_learning_rate': ('training', 'min_learning_rate'),
        'optimizer_type': ('training', 'optimizer_type'),
        'weight_decay': ('training', 'weight_decay'),
        'warmup_epochs': ('training', 'warmup_epochs'),
        'use_cosine_scheduler': ('training', 'use_cosine_scheduler'),
    }
    
    for arg_name, arg_value in vars(args).items():
        if arg_value is not None:
            # 处理learning_rate的别名
            if arg_name == 'lr':
                arg_name = 'learning_rate'
            
            # 跳过非配置参数
            if arg_name in ['config_path', 'gpu', 'resume']:
                continue
            
            # 获取参数映射
            if arg_name in param_mapping:
                section, key = param_mapping[arg_name]
                
                # 确保配置节存在
                if section not in config:
                    config[section] = {}
                
                # 记录原值和覆写
                config_path = f'{section}.{key}'
                if key in config[section]:
                    original_values[config_path] = config[section][key]
                    if config[section][key] != arg_value:
                        override_params[config_path] = {
                            'original': config[section][key],
                            'override': arg_value
                        }
                
                # 应用覆写
                config[section][key] = arg_value
            else:
                # 处理顶级参数（向后兼容）
                if arg_name in config and config[arg_name] != arg_value:
                    override_params[arg_name] = {
                        'original': config[arg_name],
                        'override': arg_value
                    }
                config[arg_name] = arg_value
    
    return config, override_params

def save_override_record(config, override_params, args, save_dir):
    """保存详细的参数覆写记录到文件。"""
    if not override_params:
        log_info("无参数覆写，跳过覆写记录保存。", print_message=False)
        return
    
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    override_path = os.path.join(save_dir, 'args_override_record.txt')
    
    with open(override_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("EDLformer 训练参数覆写记录\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"覆写时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总覆写参数数量: {len(override_params)}\n\n")
        
        f.write("覆写详情:\n")
        f.write("-" * 60 + "\n")
        
        for config_path, change_info in override_params.items():
            original_val = change_info['original']
            override_val = change_info['override']
            f.write(f"参数路径: {config_path}\n")
            f.write(f"  原始值: {original_val} ({type(original_val).__name__})\n")
            f.write(f"  覆写值: {override_val} ({type(override_val).__name__})\n")
            f.write(f"  变更: {original_val} -> {override_val}\n")
            f.write("-" * 40 + "\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("完整的命令行参数:\n")
        f.write("="*80 + "\n")
        for arg_name, arg_value in vars(args).items():
            if arg_value is not None:
                f.write(f"--{arg_name}: {arg_value}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("覆写后的完整配置摘要:\n")
        f.write("="*80 + "\n")
        
        # 输出关键配置信息
        training_cfg = config.get('training', {})
        f.write("训练配置:\n")
        for key in ['seed', 'epochs', 'batch_size', 'learning_rate', 'optimizer_type']:
            if key in training_cfg:
                f.write(f"  {key}: {training_cfg[key]}\n")
        
        dataset_cfg = config.get('dataset', {})
        if dataset_cfg:
            f.write("数据集配置:\n")
            for key, value in dataset_cfg.items():
                f.write(f"  {key}: {value}\n")
        
        experiment_cfg = config.get('experiment', {})
        if experiment_cfg:
            f.write("实验配置:\n")
            for key, value in experiment_cfg.items():
                f.write(f"  {key}: {value}\n")
    
    log_info(f"参数覆写记录已保存到: {override_path}", print_message=True)

def print_override_summary(override_params):
    """在控制台打印参数覆写摘要。"""
    log_info("="*60, print_message=True)
    log_info(f"🔧 检测到 {len(override_params)} 个参数覆写:", print_message=True)
    for config_path, change_info in override_params.items():
        log_info(f"  - {config_path}: {change_info['original']} → {change_info['override']}", print_message=True)
    log_info("="*60, print_message=True)
            

# ==============================================================================
# 模型保存与加载 (Checkpointing)
# ==============================================================================


def save_checkpoint(state, is_best, save_dir, epoch, last_checkpoints: deque, max_checkpoints=3):
    """保存模型检查点。"""
    filename = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(state, filename)
    last_checkpoints.append(filename)
    if is_best:
        best_path = os.path.join(save_dir, 'best_model.pth')
        shutil.copyfile(filename, best_path)
    while len(last_checkpoints) > max_checkpoints:
        old_ckpt = last_checkpoints.popleft()
        if os.path.exists(old_ckpt):
            os.remove(old_ckpt)
    log_info(f"已保存检查点: {filename}", print_message=False)


def load_checkpoint(checkpoint_path, model, optimizer=None, scaler=None):
    """加载模型检查点。"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
    
    log_info(f"正在加载检查点: {checkpoint_path}", print_message=True)
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    
    # 加载模型状态
    model.load_state_dict(checkpoint['state_dict'])
    
    # 加载优化器状态
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        log_info("优化器状态已加载", print_message=True)
    
    # 加载激活函数缩放器状态
    if scaler is not None and 'scaler' in checkpoint and checkpoint['scaler'] is not None:
        scaler.load_state_dict(checkpoint['scaler'])
        log_info("AMP Scaler状态已加载", print_message=True)
    
    start_epoch = checkpoint.get('epoch', 0)
    best_metric_score = checkpoint.get('best_metric_score', -np.inf)
    
    log_info(f"检查点加载完成: epoch={start_epoch}, best_score={best_metric_score:.6f}", print_message=True)
    
    return start_epoch, best_metric_score



# ==============================================================================
# TensorBoard 日志记录器
# ==============================================================================

class TensorboardLogger:
    """一个封装了TensorBoard SummaryWriter的类，简化日志记录。"""
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_training_losses(self, loss_dict, global_step):
        """记录训练过程中的所有损失分量。"""
        for loss_name, loss_value in loss_dict.items():
            self.writer.add_scalar(f'Loss/{loss_name}', loss_value, global_step)

    def log_validation_metrics(self, metrics, train_losses, learning_rate, epoch):
        """记录验证指标、平均训练损失和学习率。"""
        # 记录平均训练损失（所有分量）
        for loss_name, loss_value in train_losses.items():
            self.writer.add_scalar(f'Loss/train_avg_{loss_name}', loss_value, epoch)

        # 记录软 Dice 分数（按类别）
        soft_dice = metrics.get('soft_dice', {})
        for metric_name, value in soft_dice.items():
            self.writer.add_scalar(f'Metrics/soft_dice_{metric_name}', value, epoch)
        
        # 记录个性化指标 - dice_per_expert
        dice_per_expert = metrics.get('dice_per_expert', {})
        for class_name, expert_scores in dice_per_expert.items():
            if isinstance(expert_scores, list):
                for expert_idx, score in enumerate(expert_scores):
                    self.writer.add_scalar(f'Personalization/dice_per_expert_{class_name}_expert_{expert_idx}', score, epoch)
                # 记录平均值
                avg_score = sum(expert_scores) / len(expert_scores) if expert_scores else 0.0
                self.writer.add_scalar(f'Personalization/dice_per_expert_{class_name}_avg', avg_score, epoch)
        
        # 记录其他个性化指标
        dice_max = metrics.get('dice_max', {})
        dice_match = metrics.get('dice_match', {})
        for class_name in ['disc', 'cup', 'overall']:
            if class_name in dice_max:
                self.writer.add_scalar(f'Personalization/dice_max_{class_name}', dice_max[class_name], epoch)
            if class_name in dice_match:
                self.writer.add_scalar(f'Personalization/dice_match_{class_name}', dice_match[class_name], epoch)
        
        # 记录GED指标
        if 'ged' in metrics:
            self.writer.add_scalar('Metrics/ged', metrics['ged'], epoch)
        
        # 记录学习率
        self.writer.add_scalar('Learning_Rate', learning_rate, epoch)
    
    def close(self):
        self.writer.close()



# ==============================================================================
# 日志打印辅助函数
# ==============================================================================

def log_epoch_summary(epoch, total_epochs, train_losses, val_metrics):
    """在控制台打印每个epoch的训练和验证摘要。"""
    log_info(f"--- Epoch {epoch + 1}/{total_epochs} Summary ---", print_message=True)
    
    # 记录训练损失（所有分量）
    log_info(f"  - Avg Train Loss (Total): {train_losses.get('total_loss', 0):.6f}", print_message=True)
    for loss_name, loss_value in train_losses.items():
        if loss_name != 'total_loss':
            log_info(f"    - {loss_name}: {loss_value:.6f}", print_message=True)
    
    # 记录软 Dice 分数（按类别）
    soft_dice = val_metrics.get('soft_dice', {})
    mean_dice = soft_dice.get('mean', 0)
    disc_dice = soft_dice.get('disc', 0)
    cup_dice = soft_dice.get('cup', 0)
    
    log_info(f"  - Validation Soft Dice Mean: {mean_dice:.6f}", print_message=True)
    log_info(f"    - Disc: {disc_dice:.4f}, Cup: {cup_dice:.4f}", print_message=True)
    
    # 记录个性化指标
    dice_per_expert = val_metrics.get('dice_per_expert', {})
    dice_max = val_metrics.get('dice_max', {})
    dice_match = val_metrics.get('dice_match', {})
    
    if dice_per_expert:
        log_info(f"  - Dice Per Expert:", print_message=True)
        for class_name, expert_scores in dice_per_expert.items():
            if isinstance(expert_scores, list) and expert_scores:
                avg_score = sum(expert_scores) / len(expert_scores)
                scores_str = ', '.join([f"{score:.4f}" for score in expert_scores])
                log_info(f"    - {class_name}: [{scores_str}] (avg: {avg_score:.4f})", print_message=True)
    
    if dice_max:
        log_info(f"  - Dice Max:", print_message=True)
        for class_name, score in dice_max.items():
            log_info(f"    - {class_name}: {score:.4f}", print_message=True)
    
    if dice_match:
        log_info(f"  - Dice Match:", print_message=True)
        for class_name, score in dice_match.items():
            log_info(f"    - {class_name}: {score:.4f}", print_message=True)
    
    # 记录GED指标
    if 'ged' in val_metrics:
        log_info(f"  - GED: {val_metrics['ged']:.6f}", print_message=True)
    
    log_info("-" * 40, print_message=True)


def save_final_results(save_dir, best_metrics, last_metrics, config):
    """将最终的训练结果摘要保存到文本文件。"""
    path = os.path.join(save_dir, 'final_training_summary.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("训练完成 - 最终结果摘要\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"实验目录: {save_dir}\n")
        f.write(f"总训练轮数: {config['training']['epochs']}\n\n")

        f.write("--- 最佳Epoch指标 ---\n")
        f.write(f"最佳轮次: {best_metrics.get('epoch', -1) + 1}\n")
        
        # Soft Dice 指标
        best_dice = best_metrics.get('metrics', {}).get('soft_dice', {})
        f.write(f"  - Soft Dice Mean: {best_dice.get('mean', 0):.6f}\n")
        f.write(f"  - Soft Dice Disc: {best_dice.get('disc', 0):.6f}\n")
        f.write(f"  - Soft Dice Cup: {best_dice.get('cup', 0):.6f}\n")
        
        # 个性化指标
        best_metrics_data = best_metrics.get('metrics', {})
        best_dice_max = best_metrics_data.get('dice_max', {})
        best_dice_match = best_metrics_data.get('dice_match', {})
        best_dice_per_expert = best_metrics_data.get('dice_per_expert', {})
        
        if best_dice_max:
            f.write(f"  - Dice Max Overall: {best_dice_max.get('overall', 0):.6f}\n")
            f.write(f"  - Dice Max Disc: {best_dice_max.get('disc', 0):.6f}\n")
            f.write(f"  - Dice Max Cup: {best_dice_max.get('cup', 0):.6f}\n")
        
        if best_dice_match:
            f.write(f"  - Dice Match Overall: {best_dice_match.get('overall', 0):.6f}\n")
            f.write(f"  - Dice Match Disc: {best_dice_match.get('disc', 0):.6f}\n")
            f.write(f"  - Dice Match Cup: {best_dice_match.get('cup', 0):.6f}\n")
        
        if best_dice_per_expert:
            f.write(f"  - Dice Per Expert:\n")
            for class_name, expert_scores in best_dice_per_expert.items():
                if isinstance(expert_scores, list) and expert_scores:
                    avg_score = sum(expert_scores) / len(expert_scores)
                    scores_str = ', '.join([f"{score:.4f}" for score in expert_scores])
                    f.write(f"    - {class_name}: [{scores_str}] (avg: {avg_score:.4f})\n")
        
        if 'ged' in best_metrics_data:
            f.write(f"  - GED: {best_metrics_data['ged']:.6f}\n")
        f.write("\n")
        
        f.write("--- 最后Epoch指标 ---\n")
        f.write(f"最后轮次: {last_metrics.get('epoch', -1) + 1}\n")
        
        # Soft Dice 指标
        last_dice = last_metrics.get('metrics', {}).get('soft_dice', {})
        f.write(f"  - Soft Dice Mean: {last_dice.get('mean', 0):.6f}\n")
        f.write(f"  - Soft Dice Disc: {last_dice.get('disc', 0):.6f}\n")
        f.write(f"  - Soft Dice Cup: {last_dice.get('cup', 0):.6f}\n")
        
        # 个性化指标
        last_metrics_data = last_metrics.get('metrics', {})
        last_dice_max = last_metrics_data.get('dice_max', {})
        last_dice_match = last_metrics_data.get('dice_match', {})
        last_dice_per_expert = last_metrics_data.get('dice_per_expert', {})
        
        if last_dice_max:
            f.write(f"  - Dice Max Overall: {last_dice_max.get('overall', 0):.6f}\n")
            f.write(f"  - Dice Max Disc: {last_dice_max.get('disc', 0):.6f}\n")
            f.write(f"  - Dice Max Cup: {last_dice_max.get('cup', 0):.6f}\n")
        
        if last_dice_match:
            f.write(f"  - Dice Match Overall: {last_dice_match.get('overall', 0):.6f}\n")
            f.write(f"  - Dice Match Disc: {last_dice_match.get('disc', 0):.6f}\n")
            f.write(f"  - Dice Match Cup: {last_dice_match.get('cup', 0):.6f}\n")
        
        if last_dice_per_expert:
            f.write(f"  - Dice Per Expert:\n")
            for class_name, expert_scores in last_dice_per_expert.items():
                if isinstance(expert_scores, list) and expert_scores:
                    avg_score = sum(expert_scores) / len(expert_scores)
                    scores_str = ', '.join([f"{score:.4f}" for score in expert_scores])
                    f.write(f"    - {class_name}: [{scores_str}] (avg: {avg_score:.4f})\n")
        
        if 'ged' in last_metrics_data:
            f.write(f"  - GED: {last_metrics_data['ged']:.6f}\n")
        f.write("\n")
    log_info(f"最终结果摘要已保存至: {path}", print_message=True)