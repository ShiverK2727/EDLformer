import os
import sys
import yaml
import torch
import random
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import deque
import shutil
import json
import argparse

# 从项目中导入必要的模块
from datasets import RIGADatasetSimple as RIGADataset
from datasets import RIGADatasetSimpleMulti
from nets.edl_maskformer import EDLMaskFormer
from nets.simple_maskformer import SimpleMaskFormer, SimpleMaskFormerMulti, SimpleMaskFormerMultiV2
# --- [修复] 导入正确的损失函数模块 ---
from scheduler.edl_single_loss import EDLSingleLoss
from scheduler.simple_maskformer_loss import SimpleMaskformerLoss
from scheduler.simple_maskformer_multi_loss import (
    SimpleMaskFormerMultiLoss, 
    SimpleMaskFormerMultiLossFixed, 
    SimpleMaskFormerMultiLossHungarian
)
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
# 组件构建器 (Builders / Factories)
# ==============================================================================

def build_simple_maskformer_components(config):
    """根据配置构建并返回基础MaskFormer模型和损失函数。"""
    model_config = config['model']
    loss_config = config.get('loss', {})
    
    # 记录模型配置参数
    log_info("="*80, print_message=True)
    log_info("构建SimpleMaskFormer模型...", print_message=True)
    log_info("模型配置参数:", print_message=True)
    import json
    for key, value in model_config.items():
        log_info(f"  {key}: {value}", print_message=True)
    log_info("="*80, print_message=True)
    
    # 构建SimpleMaskFormer模型
    model = SimpleMaskFormer(**model_config)
    log_info("SimpleMaskFormer 模型构建完成。", print_message=True)

    # 记录损失函数配置参数
    log_info("="*80, print_message=True)
    log_info("构建SimpleMaskformerLoss损失函数...", print_message=True)
    log_info("损失函数配置参数:", print_message=True)
    for key, value in loss_config.items():
        log_info(f"  {key}: {value}", print_message=True)
    log_info("="*80, print_message=True)

    # 构建SimpleMaskformerLoss损失函数
    loss_params = {k: v for k, v in loss_config.items() if k != 'loss_name'}
    loss_fn = SimpleMaskformerLoss(**loss_params)
    log_info("SimpleMaskformerLoss 损失函数构建完成。", print_message=True)
    
    return model.cuda(), loss_fn

def build_simple_maskformer_multi_components(config):
    """根据配置构建并返回Multi版本MaskFormer模型和损失函数。"""
    model_config = config['model']
    loss_config = config.get('loss', {})
    
    # 记录模型配置参数
    log_info("="*80, print_message=True)
    log_info("构建SimpleMaskFormerMulti模型...", print_message=True)
    log_info("模型配置参数:", print_message=True)
    import json
    for key, value in model_config.items():
        log_info(f"  {key}: {value}", print_message=True)
    log_info("="*80, print_message=True)
    
    # 根据模型类型选择构建函数
    model_type = model_config.get('model_type', 'multi')  # 'multi' or 'multi_v2'
    
    if model_type == 'multi_v2':
        model = SimpleMaskFormerMultiV2(**{k: v for k, v in model_config.items() if k != 'model_type'})
        log_info("SimpleMaskFormerMultiV2 模型构建完成（增强SEBlock版本）。", print_message=True)
    else:
        model = SimpleMaskFormerMulti(**{k: v for k, v in model_config.items() if k != 'model_type'})
        log_info("SimpleMaskFormerMulti 模型构建完成。", print_message=True)

    # 记录损失函数配置参数
    log_info("="*80, print_message=True)
    log_info("构建SimpleMaskFormerMultiLoss损失函数...", print_message=True)
    log_info("损失函数配置参数:", print_message=True)
    for key, value in loss_config.items():
        log_info(f"  {key}: {value}", print_message=True)
    log_info("="*80, print_message=True)

    # 构建Multi版本损失函数
    loss_type = loss_config.get('loss_type', 'auto')  # 'fixed', 'hungarian', 'auto'
    loss_params = {k: v for k, v in loss_config.items() if k not in ['loss_type', 'loss_name']}
    
    if loss_type == 'fixed':
        loss_params['force_matching'] = True
        loss_fn = SimpleMaskFormerMultiLossFixed(**loss_params)
        log_info("SimpleMaskFormerMultiLossFixed 损失函数构建完成（固定匹配）。", print_message=True)
    elif loss_type == 'hungarian':
        loss_params['force_matching'] = False  
        loss_fn = SimpleMaskFormerMultiLossHungarian(**loss_params)
        log_info("SimpleMaskFormerMultiLossHungarian 损失函数构建完成（匈牙利匹配）。", print_message=True)
    else:
        # 自动模式：使用配置中的force_matching值（已在loss_params中）
        force_matching = loss_config.get('force_matching', True)
        loss_fn = SimpleMaskFormerMultiLoss(**loss_params)
        match_type = "固定匹配" if force_matching else "匈牙利匹配"
        log_info(f"SimpleMaskFormerMultiLoss 损失函数构建完成（自动模式：{match_type}）。", print_message=True)
    
    return model.cuda(), loss_fn

def build_edl_components(config):
    """根据配置构建并返回 EDL MaskFormer 模型和损失函数。"""
    model_config = config['model']
    loss_config = config.get('loss', {})
    
    model = EDLMaskFormer(**model_config)
    log_info("EDL MaskFormer 模型构建完成。", print_message=False)

    # --- [核心修改] 支持新的 FlexibleEDLSingleLoss ---
    # 1. 检查损失函数类型
    loss_name = loss_config.get('loss_name', 'edl_single_loss')
    loss_params = {k: v for k, v in loss_config.items() if k != 'loss_name'}
    
    # 2. 根据损失函数类型选择不同的实现
    if loss_name == 'flexible_edl_single_loss':
        from scheduler.edl_fixed_single_loss import FlexibleEDLSingleLoss
        loss_fn = FlexibleEDLSingleLoss(**loss_params)
        log_info("FlexibleEDLSingleLoss 损失函数构建完成。", print_message=False)
    else:
        # 默认使用原始的 EDLSingleLoss
        loss_fn = EDLSingleLoss(**loss_params)
        log_info("EDLSingleLoss 损失函数构建完成。", print_message=False)
    
    # 3. [增强日志] 记录传递给损失函数的参数，便于调试和追溯
    log_info(f"损失函数类型: {loss_name}", print_message=False)
    log_info("损失函数参数:", print_message=False)
    log_info(json.dumps(loss_params, indent=2), print_message=False)
    # ---
    
    return model.cuda(), loss_fn

def build_optimizer(model, config):
    """构建优化器。学习率统一从scheduler配置中读取。"""
    opt_cfg = config.get('optimizer', {})
    opt_type = opt_cfg.get('type', 'AdamW')
    sched_cfg = config.get('schelduler', {})
    
    # 学习率统一从scheduler配置中读取
    base_lr = float(sched_cfg.get('base_lr', 5e-4))
    
    # 过滤并转换优化器参数，确保类型正确
    # 注意：不再从optimizer配置中读取lr，统一使用scheduler的base_lr
    opt_params = {}
    for k, v in opt_cfg.items():
        if k not in ['type', 'lr']:  # 排除lr，使用scheduler的base_lr
            opt_params[k] = v
    
    if opt_type.lower() == 'adamw':
        # 确保AdamW参数类型正确
        adamw_params = {
            'lr': base_lr,  # 使用scheduler的base_lr
            'weight_decay': float(opt_params.get('weight_decay', 1e-4)),
            'eps': float(opt_params.get('eps', 1e-8)),
            'amsgrad': bool(opt_params.get('amsgrad', False))
        }
        # 处理betas参数（列表）
        if 'betas' in opt_params:
            betas = opt_params['betas']
            if isinstance(betas, list) and len(betas) == 2:
                adamw_params['betas'] = (float(betas[0]), float(betas[1]))
            else:
                adamw_params['betas'] = (0.9, 0.999)  # 默认值
        else:
            adamw_params['betas'] = (0.9, 0.999)
        
        log_info(f"使用AdamW优化器，学习率: {base_lr} (来自scheduler配置)", print_message=False)
        return torch.optim.AdamW(model.parameters(), **adamw_params)
    elif opt_type.lower() == 'adam':
        # 确保Adam参数类型正确
        adam_params = {
            'lr': base_lr,  # 使用scheduler的base_lr
            'eps': float(opt_params.get('eps', 1e-8)),
            'amsgrad': bool(opt_params.get('amsgrad', False))
        }
        if 'betas' in opt_params:
            betas = opt_params['betas']
            if isinstance(betas, list) and len(betas) == 2:
                adam_params['betas'] = (float(betas[0]), float(betas[1]))
        log_info(f"使用Adam优化器，学习率: {base_lr} (来自scheduler配置)", print_message=False)
        return torch.optim.Adam(model.parameters(), **adam_params)
    elif opt_type.lower() == 'sgd':
        sgd_params = {
            'lr': base_lr,  # 使用scheduler的base_lr
            'momentum': float(opt_params.get('momentum', 0.9)),
            'weight_decay': float(opt_params.get('weight_decay', 1e-4))
        }
        log_info(f"使用SGD优化器，学习率: {base_lr} (来自scheduler配置)", print_message=False)
        return torch.optim.SGD(model.parameters(), **sgd_params)
    else:
        raise ValueError(f"未知的优化器类型: {opt_type}")

def build_scheduler(optimizer, config):
    """构建学习率调度器。"""
    sched_cfg = config.get('schelduler', {})
    if not sched_cfg.get('use_cosine_scheduler', False):
        return None
    
    total_epochs = config.get('epochs', 100)
    warmup_epochs = int(sched_cfg.get('warmup', 0))
    base_lr = float(sched_cfg.get('base_lr', 5e-4))
    min_lr = float(sched_cfg.get('min_lr', 1e-6))
    
    def lr_lambda(epoch):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1. + np.cos(np.pi * progress))
        return cosine * (1.0 - min_lr / base_lr) + min_lr / base_lr
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def build_dataloaders(config, dataset_type=None):
    """构建数据加载器。"""
    if dataset_type is None:
        dataset_type = config.get('dataset_type', 'RIGA')
    
    if dataset_type == "RIGA":
        # 假设 RIGADataset 接受 config_path 和 is_train
        dataset = RIGADataset(config_path=config['dataset_yaml'], is_train=True)
        val_dataset = RIGADataset(config_path=config['dataset_yaml'], is_train=False)
    else:
        raise ValueError(f"未知的数据集类型: {dataset_type}")

    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    
    return dataloader, val_dataloader

def build_dataloaders_multi(config, dataset_type=None):
    """构建Multi版本的数据加载器，使用RIGADatasetSimpleMulti。"""
    if dataset_type is None:
        dataset_type = config.get('dataset_type', 'RIGA')
    
    if dataset_type == "RIGA":
        # 使用Multi版本的数据集
        dataset = RIGADatasetSimpleMulti(config_path=config['dataset_yaml'], is_train=True)
        val_dataset = RIGADatasetSimpleMulti(config_path=config['dataset_yaml'], is_train=False)
        log_info("使用RIGADatasetSimpleMulti数据集（BN2HW格式）", print_message=True)
    else:
        raise ValueError(f"未知的数据集类型: {dataset_type}")

    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    
    return dataloader, val_dataloader

# ==============================================================================
# 命令行参数处理
# ==============================================================================

def setup_training_args_parser():
    parser = argparse.ArgumentParser(description='EDL MaskFormer Training (Final Version)')
    
    parser.add_argument('--config_path', type=str, required=True, help='训练配置文件路径')
    parser.add_argument('--dataset_yaml', type=str, required=True, help='数据集配置文件路径')
    parser.add_argument('--gpu', type=str, default='0', help='GPU设备ID')
    parser.add_argument('--save_dir', type=str, default='./exp', help='保存训练输出的目录')
    # 添加其他所有需要的命令行参数...
    parser.add_argument('--lr', type=float, help='覆写学习率')
    parser.add_argument('--batch_size', type=int, help='覆写批处理大小')
    parser.add_argument('--epochs', type=int, help='覆写总训练轮数')
    parser.add_argument('--resume', type=str, help='从检查点恢复训练的路径')

    return parser

def apply_args_override(config, args):
    """应用命令行参数覆写YAML配置。"""
    override_params = {}
    for arg in vars(args):
        arg_value = getattr(args, arg)
        if arg_value is not None:
            # 简单处理，实际可能需要更复杂的逻辑来定位配置中的键
            keys = arg.split('.')
            d = config
            for key in keys[:-1]:
                d = d.setdefault(key, {})
            if keys[-1] in d and d[keys[-1]] != arg_value:
                override_params[arg] = arg_value
            d[keys[-1]] = arg_value
    
    # 特殊处理非直接映射的参数
    if args.save_dir: config['save_dir'] = args.save_dir
    if args.resume: config['resume'] = args.resume

    return config, override_params

def save_override_record(override_params, args, save_dir):
    """保存参数覆写记录。"""
    if not override_params: return
    override_path = os.path.join(save_dir, 'override_args.txt')
    with open(override_path, 'w', encoding='utf-8') as f:
        # ... 写入逻辑 ...
        f.write("Command Line Overrides:\n")
        for k, v in override_params.items():
            f.write(f"  {k}: {v}\n")

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
