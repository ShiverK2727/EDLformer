import os
import json
import yaml
import torch
import random
import datetime
import argparse
import numpy as np
from collections import deque
from typing import Tuple, Dict, Any
from torch.utils.tensorboard import SummaryWriter

from logger import log_info


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_config(yaml_path: str) -> Dict[str, Any]:
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def setup_training_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='SimpleMaskFormerV2 Beta-EDL Training')
    parser.add_argument('--config_path', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--gpu', type=str, default='0', help='CUDA device id, e.g., "0" or "0,1"')
    # Lightweight overrides for common hyperparameters
    parser.add_argument('--save_dir', type=str, help='Override experiment.save_dir')
    parser.add_argument('--epochs', type=int, help='Override training.epochs')
    parser.add_argument('--batch_size', type=int, help='Override training.batch_size')
    parser.add_argument('--learning_rate', '--lr', dest='learning_rate', type=float, help='Override training.learning_rate')
    parser.add_argument('--num_workers', type=int, help='Override training.num_workers')
    parser.add_argument('--use_amp', action='store_true', help='Enable AMP regardless of config')
    parser.add_argument('--no_amp', action='store_true', help='Disable AMP regardless of config')
    parser.add_argument('--optimizer_type', type=str, choices=['AdamW', 'Adam', 'SGD'], help='Override training.optimizer_type')
    parser.add_argument('--weight_decay', type=float, help='Override training.weight_decay')
    parser.add_argument('--warmup_epochs', type=int, help='Override training.warmup_epochs')
    parser.add_argument('--use_cosine_scheduler', action='store_true', help='Force enable cosine LR scheduler')
    parser.add_argument('--no_cosine_scheduler', action='store_true', help='Force disable cosine LR scheduler')
    parser.add_argument('--min_learning_rate', type=float, help='Override training.min_learning_rate')
    return parser


def apply_args_override(config: Dict[str, Any], args: argparse.Namespace) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Apply a minimal set of CLI overrides to the loaded config."""
    overrides = {}
    training_cfg = config.setdefault('training', {})
    experiment_cfg = config.setdefault('experiment', {})

    if args.save_dir:
        overrides['experiment.save_dir'] = {'original': experiment_cfg.get('save_dir'), 'override': args.save_dir}
        experiment_cfg['save_dir'] = args.save_dir
    if args.epochs is not None:
        overrides['training.epochs'] = {'original': training_cfg.get('epochs'), 'override': args.epochs}
        training_cfg['epochs'] = args.epochs
    if args.batch_size is not None:
        overrides['training.batch_size'] = {'original': training_cfg.get('batch_size'), 'override': args.batch_size}
        training_cfg['batch_size'] = args.batch_size
    if args.learning_rate is not None:
        overrides['training.learning_rate'] = {'original': training_cfg.get('learning_rate'), 'override': args.learning_rate}
        training_cfg['learning_rate'] = args.learning_rate
    if args.num_workers is not None:
        overrides['training.num_workers'] = {'original': training_cfg.get('num_workers'), 'override': args.num_workers}
        training_cfg['num_workers'] = args.num_workers
    if args.optimizer_type is not None:
        overrides['training.optimizer_type'] = {'original': training_cfg.get('optimizer_type'), 'override': args.optimizer_type}
        training_cfg['optimizer_type'] = args.optimizer_type
    if args.weight_decay is not None:
        overrides['training.weight_decay'] = {'original': training_cfg.get('weight_decay'), 'override': args.weight_decay}
        training_cfg['weight_decay'] = args.weight_decay
    if args.warmup_epochs is not None:
        overrides['training.warmup_epochs'] = {'original': training_cfg.get('warmup_epochs'), 'override': args.warmup_epochs}
        training_cfg['warmup_epochs'] = args.warmup_epochs
    if args.min_learning_rate is not None:
        overrides['training.min_learning_rate'] = {'original': training_cfg.get('min_learning_rate'), 'override': args.min_learning_rate}
        training_cfg['min_learning_rate'] = args.min_learning_rate

    if args.use_amp:
        overrides['training.use_amp'] = {'original': training_cfg.get('use_amp'), 'override': True}
        training_cfg['use_amp'] = True
    if args.no_amp:
        overrides['training.use_amp'] = {'original': training_cfg.get('use_amp'), 'override': False}
        training_cfg['use_amp'] = False

    if args.use_cosine_scheduler:
        overrides['training.use_cosine_scheduler'] = {'original': training_cfg.get('use_cosine_scheduler'), 'override': True}
        training_cfg['use_cosine_scheduler'] = True
    if args.no_cosine_scheduler:
        overrides['training.use_cosine_scheduler'] = {'original': training_cfg.get('use_cosine_scheduler'), 'override': False}
        training_cfg['use_cosine_scheduler'] = False

    return config, overrides


def save_override_record(overrides: Dict[str, Any], args: argparse.Namespace, save_dir: str) -> None:
    if not overrides:
        return
    os.makedirs(save_dir, exist_ok=True)
    record_path = os.path.join(save_dir, 'args_override_record.txt')
    with open(record_path, 'w', encoding='utf-8') as f:
        f.write("Overrides applied:\n")
        for k, v in overrides.items():
            f.write(f"{k}: {v['original']} -> {v['override']}\n")
        f.write("\nCommand args:\n")
        for arg_name, arg_val in vars(args).items():
            f.write(f"{arg_name}: {arg_val}\n")


class TensorboardLogger:
    def __init__(self, save_dir: str):
        self.log_dir = os.path.join(save_dir, 'logs')
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def _to_scalar(self, value: Any):
        """Convert tensors/arrays/lists to a scalar for TensorBoard; otherwise return None."""
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, (np.ndarray,)):
            return float(np.mean(value))
        if torch.is_tensor(value):
            return float(value.detach().mean().item())
        if isinstance(value, list) and value and all(isinstance(x, (int, float)) for x in value):
            return float(np.mean(value))
        return None

    def log_validation_metrics(self, metrics: Dict[str, Any], train_losses: Dict[str, float], lr: float, epoch: int):
        # Log validation metrics safely (lists/arrays -> mean scalar)
        for k, v in metrics.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    if isinstance(sub_v, dict):
                        for leaf_k, leaf_v in sub_v.items():
                            scalar = self._to_scalar(leaf_v)
                            if scalar is not None:
                                self.writer.add_scalar(f"val/{k}_{sub_k}_{leaf_k}", scalar, epoch)
                    else:
                        scalar = self._to_scalar(sub_v)
                        if scalar is not None:
                            self.writer.add_scalar(f"val/{k}_{sub_k}", scalar, epoch)
            else:
                scalar = self._to_scalar(v)
                if scalar is not None:
                    self.writer.add_scalar(f"val/{k}", scalar, epoch)
        # Log training losses
        for k, v in train_losses.items():
            scalar = self._to_scalar(v)
            if scalar is not None:
                self.writer.add_scalar(f"train/{k}", scalar, epoch)
        self.writer.add_scalar("train/lr", lr, epoch)

    def close(self):
        self.writer.close()


# def log_epoch_summary(epoch: int, total_epochs: int, train_losses: Dict[str, float], val_metrics: Dict[str, Any]) -> None:
#     log_info(f"Epoch {epoch+1}/{total_epochs} summary:", print_message=True)
#     loss_msg = ', '.join([f"{k}: {v:.4f}" for k, v in train_losses.items()])
#     log_info(f"  Train losses: {loss_msg}", print_message=True)
#     log_info(f"  Val metrics: {json.dumps(val_metrics)}", print_message=True)

    
def log_epoch_summary(epoch: int, total_epochs: int, train_losses: Dict[str, float], val_metrics: Dict[str, Any]) -> None:
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

    # 记录 EDL 加权 Soft Dice
    soft_dice_edl = val_metrics.get('soft_dice_edl', {})
    if soft_dice_edl:
        log_info(f"  - EDL Weighted Soft Dice Mean: {soft_dice_edl.get('mean', 0):.6f}", print_message=True)
        log_info(f"    - Disc: {soft_dice_edl.get('disc', 0):.4f}, Cup: {soft_dice_edl.get('cup', 0):.4f}", print_message=True)

    soft_dice_ds = val_metrics.get('soft_dice_ds', {})
    if soft_dice_ds:
        log_info(f"  - DS Fused Soft Dice Mean: {soft_dice_ds.get('mean', 0):.6f}", print_message=True)
        log_info(f"    - Disc: {soft_dice_ds.get('disc', 0):.4f}, Cup: {soft_dice_ds.get('cup', 0):.4f}", print_message=True)
    
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

    dice_per_expert_mean = val_metrics.get('dice_per_expert_mean', {})
    if dice_per_expert_mean:
        log_info(f"  - Dice Per Expert Mean:", print_message=True)
        for class_name, score in dice_per_expert_mean.items():
            log_info(f"    - {class_name}: {score:.4f}", print_message=True)
    
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
        ged_val = val_metrics['ged']
        if isinstance(ged_val, dict):
            overall = ged_val.get('overall', 0)
            log_info(f"  - GED Overall: {overall:.6f}", print_message=True)
            for cls_name, v in ged_val.items():
                if cls_name == 'overall':
                    continue
                log_info(f"    - GED {cls_name}: {v:.6f}", print_message=True)
        else:
            log_info(f"  - GED: {ged_val:.6f}", print_message=True)
    
    log_info("-" * 40, print_message=True)


def save_checkpoint(state: Dict[str, Any], is_best: bool, save_dir: str, epoch: int, last_checkpoints: deque, max_checkpoints: int) -> None:
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f"checkpoint_{epoch}.pth")
    torch.save(state, filename)
    last_checkpoints.append(filename)
    if len(last_checkpoints) > max_checkpoints:
        old = last_checkpoints.popleft()
        if os.path.exists(old):
            os.remove(old)
    if is_best:
        best_path = os.path.join(save_dir, 'best_model.pth')
        torch.save(state, best_path)


def save_final_results(save_dir: str, best_metrics: Dict[str, Any], last_metrics: Dict[str, Any], config: Dict[str, Any]) -> None:
    os.makedirs(save_dir, exist_ok=True)
    result_path = os.path.join(save_dir, 'final_results.json')
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump({
            'best': best_metrics,
            'last': last_metrics,
            'config': config
        }, f, indent=2)


def build_optimizer_and_scheduler(model: torch.nn.Module, config: Dict[str, Any]):
    training_cfg = config.get('training', {})
    base_lr = float(training_cfg.get('learning_rate', 1e-3))
    weight_decay = float(training_cfg.get('weight_decay', 1e-4))
    optimizer_type = training_cfg.get('optimizer_type', 'AdamW')

    if optimizer_type.lower() == 'adamw':
        betas = training_cfg.get('betas', [0.9, 0.999])
        betas = (float(betas[0]), float(betas[1])) if isinstance(betas, (list, tuple)) else (0.9, 0.999)
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay, betas=betas, eps=float(training_cfg.get('eps', 1e-8)))
    elif optimizer_type.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'sgd':
        momentum = float(training_cfg.get('momentum', 0.9))
        optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, weight_decay=weight_decay, momentum=momentum)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    use_scheduler = bool(training_cfg.get('use_cosine_scheduler', False))
    if not use_scheduler:
        return optimizer, None

    total_epochs = training_cfg.get('epochs', config.get('epochs', 100))
    warmup_epochs = int(training_cfg.get('warmup_epochs', 0))
    min_lr = float(training_cfg.get('min_learning_rate', 1e-6))

    def lr_lambda(epoch):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1. + np.cos(np.pi * progress))
        return cosine * (1.0 - min_lr / base_lr) + min_lr / base_lr

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler
