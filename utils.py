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
    """
    [统一构建器] 根据新的配置结构构建模型、损失函数和数据集。
    """
    model_config = config['model']
    loss_config = config.get('loss', {})
    training_config = config.get('training', {})
    dataset_config = config.get('dataset', {})
    
    # 参数验证
    required_model_params = ['num_cls_classes', 'num_queries']
    for param in required_model_params:
        if param not in model_config:
            raise ValueError(f"模型配置中缺少必需参数: {param}")
    
    # 1. 构建模型
    log_info("="*80, print_message=True)
    log_info("构建 FlexibleEDLMaskFormer 模型...", print_message=True)
    log_info(f"模型配置: {json.dumps(model_config, indent=2)}", print_message=False)
    model = FlexibleEDLMaskFormer(**model_config)
    log_info("模型构建完成。", print_message=True)

    # 2. 构建损失函数
    log_info("="*80, print_message=True)
    log_info("构建 SimpleEDLMaskformerLossV2 损失函数...", print_message=True)
    loss_params = {k: v for k, v in loss_config.items()}
    log_info(f"损失函数配置: {json.dumps(loss_params, indent=2)}", print_message=False)
    loss_fn = SimpleEDLMaskformerLossV2(**loss_params)
    log_info("损失函数构建完成。", print_message=True)

    # 3. 构建数据集
    log_info("="*80, print_message=True)
    log_info("构建数据集...", print_message=True)
    
    # 从新的配置结构中获取数据集配置
    dataset_yaml = dataset_config.get('config_path') or config.get('dataset_yaml')  # 兼容旧配置
    if not dataset_yaml or not os.path.exists(dataset_yaml):
        raise FileNotFoundError(f"数据集配置文件不存在: {dataset_yaml}")
    
    train_dataset = RIGADatasetSimpleV2(config_path=dataset_yaml, is_train=True)
    val_dataset = RIGADatasetSimpleV2(config_path=dataset_yaml, is_train=False)
    
    # 从training配置中获取批处理大小
    batch_size = training_config.get('batch_size', 8)
    num_workers = training_config.get('num_workers', 4)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)
    log_info(f"数据集构建完成。训练集: {len(train_dataset)}样本, 验证集: {len(val_dataset)}样本。", print_message=True)
    log_info(f"批处理大小: {batch_size}, 工作线程数: {num_workers}", print_message=True)
    log_info("="*80, print_message=True)

    return model.cuda(), loss_fn, train_loader, val_loader


def build_complete_training_setup(config, args=None, save_override_record_flag=True):
    """
    构建完整的训练设置，包括模型、损失函数、数据集、优化器和调度器。
    
    Args:
        config: 配置字典
        args: 命令行参数（可选）
        save_override_record_flag: 是否保存覆写记录
    
    Returns:
        tuple: (model, loss_fn, train_loader, val_loader, optimizer, scheduler)
    """
    log_info("="*80, print_message=True)
    log_info("构建完整训练设置", print_message=True)
    log_info("="*80, print_message=True)
    
    # 处理参数覆写
    override_params = {}
    if args is not None:
        original_config = config.copy()
        config, override_params = apply_args_override(config, args)
        
        # 显示覆写摘要
        print_override_summary(override_params)
        
        # 保存覆写记录
        if save_override_record_flag and override_params:
            save_dir = config.get('experiment', {}).get('save_dir', './exp/default')
            save_override_record(config, override_params, args, save_dir)
    
    # 设置随机种子
    training_config = config.get('training', {})
    seed = training_config.get('seed', 42)
    set_seed(seed)
    
    # 1. 构建模型、损失函数和数据集
    model, loss_fn, train_loader, val_loader = build_edl_components(config)
    
    # 2. 构建优化器和调度器
    log_info("构建优化器和学习率调度器...", print_message=True)
    optimizer, scheduler = build_optimizer_and_scheduler(model, config)
    
    log_info("="*80, print_message=True)
    log_info("完整训练设置构建完成", print_message=True)
    log_info(f"🎯 实验保存目录: {config.get('experiment', {}).get('save_dir', 'N/A')}", print_message=True)
    log_info("="*80, print_message=True)
    
    return model, loss_fn, train_loader, val_loader, optimizer, scheduler


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
    total_epochs = config.get('epochs', 100)
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
    print(f"new_labels: {new_labels}")

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
    print(f"expert_ids_map: {expert_ids_map}")
    
    # [0, 1, ..., C-1, 0, 1, ..., C-1, ...] (重复N次)
    class_ids_map = torch.arange(C, device=device).repeat(N)
    print(f"class_ids_map: {class_ids_map}")

    # 将它们堆叠成 [N*C, 2] 的映射关系
    mapping_tensor = torch.stack([expert_ids_map, class_ids_map], dim=1)
    print(f"mapping_tensor: {mapping_tensor}")

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


def identify_overridden_params(args):
    """识别哪些参数通过命令行进行了覆写。"""
    overridden_params = []
    
    for arg_name, arg_value in vars(args).items():
        if arg_value is not None and arg_name not in ['config_path', 'gpu', 'resume']:
            overridden_params.append(arg_name)
    
    return overridden_params


def print_override_summary(override_params):
    """打印参数覆写摘要。"""
    if not override_params:
        log_info("✅ 未检测到参数覆写，使用配置文件中的默认值。", print_message=True)
        return
    
    log_info("="*60, print_message=True)
    log_info(f"🔧 检测到 {len(override_params)} 个参数覆写:", print_message=True)
    log_info("="*60, print_message=True)
    
    for config_path, change_info in override_params.items():
        original_val = change_info['original']
        override_val = change_info['override']
        log_info(f"📝 {config_path}: {original_val} → {override_val}", print_message=True)
    
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


# ==============================================================================
# 使用示例和测试
# ==============================================================================
if __name__ == '__main__':
    
    def simulate_and_test(B, N, C, H, W):
        print(f"\n--- 测试数据集配置: B={B}, N={N}, C={C} ---")
        
        # 1. 模拟一个来自 RIGADatasetSimpleV2 的 batch
        #    每个掩码都是随机的二值图像
        mock_expert_masks = torch.randint(0, 2, (B, N, C, H, W), dtype=torch.float32).cuda()
        mock_batch = {"expert_masks": mock_expert_masks}
        
        # 2. 调用处理函数
        targets, metadata = process_batch_for_expert_class_combination(mock_batch)
        
        # 3. 验证输出
        print("元数据 (metadata):")
        print(f"  - 专家数 (N): {metadata['num_experts']}")
        print(f"  - 原始类别数 (C): {metadata['num_classes']}")
        print(f"  - 组合后总类别数 (N*C): {metadata['total_combined_classes']}")
        print(f"  - 映射张量形状: {metadata['mapping_tensor'].shape}")
        
        # 打印部分映射关系以供检查
        print("\n部分映射关系 (new_label -> [expert_id, class_id]):")
        for i in list(range(min(5, N*C))) + list(range(max(0, N*C-5), N*C)):
             new_label = i
             expert_id = metadata['mapping_tensor'][new_label, 0].item()
             class_id = metadata['mapping_tensor'][new_label, 1].item()
             print(f"  - 组合标签 {new_label:2d} -> 专家 {expert_id}, 类别 {class_id}")

        print("\nTargets 格式检查:")
        print(f"  - `targets` 列表长度: {len(targets)} (应为 B={B})")
        assert len(targets) == B
        
        first_sample_target = targets[0]
        print(f"  - 第一个样本 'labels' 形状: {first_sample_target['labels'].shape} (应为 [{N*C}])")
        assert first_sample_target['labels'].shape[0] == N * C
        
        print(f"  - 第一个样本 'masks' 形状: {first_sample_target['masks'].shape} (应为 [{N*C}, {H}, {W}])")
        assert first_sample_target['masks'].shape == (N * C, H, W)
        
        # 验证数据内容是否一致
        original_mask_sample = mock_batch['expert_masks'][0] # [N, C, H, W]
        processed_mask_sample = first_sample_target['masks'].view(N, C, H, W)
        assert torch.equal(original_mask_sample, processed_mask_sample)
        print("  - 数据内容一致性检查通过!")
        print("-" * (40))

    # --- 运行不同配置的测试 ---
    # RIGA 数据集情况
    simulate_and_test(B=4, N=6, C=2, H=64, W=64)
    
    # 其他可能的数据集情况
    simulate_and_test(B=8, N=4, C=1, H=32, W=32)
    simulate_and_test(B=2, N=4, C=3, H=48, W=48)
    
    print("\n" + "="*80)
    print("统一训练配置使用示例:")
    print("="*80)
    
    # 示例配置
    example_config = {
        'epochs': 100,
        'batch_size': 8,
        'dataset_yaml': './codes/configs/RIGA.yaml',
        'model': {
            'num_cls_classes': 12,
            'num_queries': 12,
            'in_channels': 3
        },
        'loss': {
            'num_classes': 2
        },
        'training': {
            'learning_rate': 1e-3,
            'optimizer_type': 'AdamW',
            'weight_decay': 1e-4,
            'use_cosine_scheduler': True,
            'warmup_epochs': 10
        }
    }
    
    print("配置示例:")
    print("training:")
    print("  learning_rate: 1e-3      # 统一学习率声明")
    print("  optimizer_type: 'AdamW'  # 优化器类型")
    print("  weight_decay: 1e-4       # 权重衰减")
    print("  use_cosine_scheduler: true")
    print("  warmup_epochs: 10")
    print("\n使用方法:")
    print("model, loss_fn, train_loader, val_loader, optimizer, scheduler = build_complete_training_setup(config)")
    print("="*80)