import os
import sys
import yaml
import torch
import collections
import numpy as np
from tqdm import tqdm
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader

from datasets import RIGADatasetSimpleV2
from logger import setup_logger, log_info
from metrics import calculate_riga_metrics

from utils_v2 import (
    set_seed, load_config, apply_args_override, setup_training_args_parser,
    save_override_record, TensorboardLogger, log_epoch_summary,
    save_checkpoint, save_final_results, build_optimizer_and_scheduler
)

from nets.simple_maskformer_v2 import SimpleMaskFormerV2
from scheduler.edl_mask_seg_loss import MaskSegEDLLoss


def _resolve_dataset_yaml(dataset_cfg: dict) -> str:
    yaml_path = dataset_cfg.get('config_path')
    if yaml_path and not os.path.isabs(yaml_path):
        yaml_path = os.path.join('/app/MultiAnn/EDLformer', yaml_path.lstrip('./'))
    if not yaml_path or not os.path.exists(yaml_path):
        raise FileNotFoundError(f"数据集配置文件不存在: {yaml_path}")
    return yaml_path


def build_components(config):
    # 数据集与 DataLoader
    dataset_yaml = _resolve_dataset_yaml(config['dataset'])
    train_dataset = RIGADatasetSimpleV2(config_path=dataset_yaml, is_train=True)
    val_dataset = RIGADatasetSimpleV2(config_path=dataset_yaml, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training'].get('batch_size', 8),
        shuffle=True,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True,
    )

    sample = train_dataset[0]
    expert_masks_shape = sample['expert_masks'].shape  # [N, C, H, W]
    num_experts, num_classes, _, _ = expert_masks_shape
    dataset_metadata = {
        'num_experts': num_experts,
        'num_seg_classes': num_classes,
        'total_combined_classes': num_experts * num_classes,
    }
    log_info(f"数据集: {num_experts} 专家, {num_classes} 类别, mask 形状 {expert_masks_shape}", print_message=True)

    # 模型
    model_cfg = config['model']
    model = SimpleMaskFormerV2(
        in_channels=model_cfg.get('in_channels', 3),
        num_classes=num_classes,
        num_experts=num_experts,
        backbone_name=model_cfg.get('backbone_name', 'resnet34'),
        dino_model_path=model_cfg.get('dino_model_path'),
        hidden_dim=model_cfg.get('hidden_dim', 256),
        use_dino_align=model_cfg.get('use_dino_align', False),
        bridge_layers_indices=model_cfg.get('bridge_layers_indices', [0]),
        backbone_decoder_channels=model_cfg.get('backbone_decoder_channels', [128, 64, 32, 16, 8]),
        backbone_use_batchnorm=model_cfg.get('backbone_use_batchnorm', True),
        backbone_upsample_mode=model_cfg.get('backbone_upsample_mode', 'interp'),
    ).cuda()
    log_info("模型构建完成。", print_message=True)

    # 损失
    loss_cfg = config.get('loss', {})
    loss_fn = MaskSegEDLLoss(
        num_classes=num_classes,
        num_experts=num_experts,
        weight_ace=loss_cfg.get('weight_ace', 1.0),
        weight_kl=loss_cfg.get('weight_kl', 0.1),
        weight_dice=loss_cfg.get('weight_dice', 1.0),
        kl_annealing_step=loss_cfg.get('kl_annealing_step', 10),
        weight_pixel_bce=loss_cfg.get('weight_pixel_bce', 1.0),
        weight_pixel_dice=loss_cfg.get('weight_pixel_dice', 1.0),
        use_pixel_aux=loss_cfg.get('use_pixel_aux', False),
        use_transformer_aux=loss_cfg.get('use_transformer_aux', False),
        pixel_aux_weights=loss_cfg.get('pixel_aux_weights'),
        transformer_aux_weights=loss_cfg.get('transformer_aux_weights'),
        use_dino_align=model_cfg.get('use_dino_align', False),
        weight_dino_align=loss_cfg.get('weight_dino_align', 1.0),
    ).cuda()
    log_info("损失函数构建完成。", print_message=True)

    # 优化器 & 调度器
    optimizer, scheduler = build_optimizer_and_scheduler(model, config)

    return model, loss_fn, train_loader, val_loader, optimizer, scheduler, dataset_metadata


def train_one_epoch(model, loader, optimizer, loss_fn, scaler, epoch, total_epochs):
    model.train()
    epoch_losses = collections.defaultdict(float)
    pbar = tqdm(loader, desc=f'Epoch {epoch+1}/{total_epochs} [Train]', leave=True, ncols=150, file=sys.stdout)

    for i, batch in enumerate(pbar):
        if not batch or batch['image'].size(0) == 0:
            continue

        images = batch['image'].cuda(non_blocking=True)
        targets = batch['expert_masks'].cuda(non_blocking=True)  # [B, N, C, H, W]

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type='cuda', enabled=(scaler is not None)):
            outputs = model(images)
            total_loss, loss_dict = loss_fn(outputs, targets, epoch)

        if scaler is not None:
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()

        loss_dict['total_loss'] = total_loss.detach()
        for k, v in loss_dict.items():
            epoch_losses[k] += v.item()

        loss_components = ', '.join([f"{k}: {v.item():.4f}" for k, v in loss_dict.items() if k != 'total_loss'])
        pbar.set_postfix({
            'total': f"{total_loss.item():.4f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.2e}",
            'sub': loss_components[:60] + '...' if len(loss_components) > 60 else loss_components
        })

    num_batches = len(loader)
    avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}
    return avg_losses


def validate(model, loader, dataset_metadata, config):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc='Validating', leave=False, ncols=120, file=sys.stdout):
            images = batch['image'].cuda(non_blocking=True)
            labels = batch['expert_masks'].cuda(non_blocking=True)

            outputs = model(images)
            evidence = outputs['pred_evidence']  # [B, N, C, H, W]
            alpha = evidence + 1.0
            prob = alpha / torch.sum(alpha, dim=2, keepdim=True)

            all_preds.append(prob.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return calculate_riga_metrics(all_labels, all_preds)


def train_loop(config, args=None):
    model, loss_fn, train_loader, val_loader, optimizer, scheduler, dataset_metadata = build_components(config)

    save_dir = config['experiment']['save_dir']
    logger = TensorboardLogger(save_dir)
    scaler = GradScaler(device='cuda') if config['training'].get('use_amp', False) else None

    start_epoch = 0
    best_metric_score = -np.inf
    best_metrics = {'epoch': -1, 'metrics': {'soft_dice': {'mean': -np.inf}}}
    last_metrics = {'epoch': -1, 'metrics': {'soft_dice': {'mean': 0}}}
    last_checkpoints = collections.deque()

    total_epochs = config['training']['epochs']

    for epoch in range(start_epoch, total_epochs):
        avg_train_losses = train_one_epoch(model, train_loader, optimizer, loss_fn, scaler, epoch, total_epochs)
        val_metrics = validate(model, val_loader, dataset_metadata, config)

        current_lr = optimizer.param_groups[0]['lr']
        logger.log_validation_metrics(val_metrics, avg_train_losses, current_lr, epoch)
        log_epoch_summary(epoch, total_epochs, avg_train_losses, val_metrics)

        current_metric_score = val_metrics['soft_dice']['mean']
        is_best = current_metric_score > best_metric_score
        if is_best:
            best_metric_score = current_metric_score
            best_metrics = {'epoch': epoch, 'metrics': val_metrics}
            log_info(f"*** New best model found! Epoch {epoch+1}, Soft Dice Mean: {best_metric_score:.6f} ***", print_message=True)

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

    save_final_results(save_dir, best_metrics, last_metrics, config)
    logger.close()
    log_info("训练完成。", print_message=True)


if __name__ == '__main__':
    parser = setup_training_args_parser()
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    config = load_config(args.config_path)
    config, override_params = apply_args_override(config, args)

    set_seed(config['training'].get('seed', 42))

    save_dir = config['experiment']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    setup_logger(save_dir, log_name='train_log.txt')

    # 保存配置
    with open(os.path.join(save_dir, 'final_config.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False, indent=2)
    save_override_record(config, override_params, args, save_dir)

    log_info("开始 SimpleMaskFormerV2 EDL 分割训练...", print_message=True)
    log_info(f"实验目录: {save_dir}", print_message=True)

    train_loop(config, args)
