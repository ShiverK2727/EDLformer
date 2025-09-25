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
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

# 只导入必要的模块用于debug
try:
    from logger import setup_logger, log_info
except ImportError:
    def log_info(msg, print_message=True):
        if print_message:
            print(msg)

try:
    from utils import set_seed, load_config, apply_args_override, setup_training_args_parser
except ImportError:
    import argparse
    def set_seed(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    def load_config(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def apply_args_override(config, args):
        # 简化版本，直接返回配置和空的override
        if hasattr(args, 'save_dir') and args.save_dir:
            config['save_dir'] = args.save_dir
        if hasattr(args, 'batch_size') and args.batch_size:
            config['batch_size'] = args.batch_size
        return config, {}
    
    def setup_training_args_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument('--config_path', type=str, required=True)
        parser.add_argument('--dataset_yaml', type=str, required=True)
        parser.add_argument('--gpu', type=str, default='0')
        parser.add_argument('--save_dir', type=str, default='./debug_output')
        parser.add_argument('--batch_size', type=int, default=2)
        return parser

from datasets import RIGADatasetSimpleV2

# =============================================================================
# 数据可视化函数
# =============================================================================

def denormalize_image(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    将标准化的图像张量反标准化为可显示的格式
    Args:
        image_tensor: [3, H, W] 标准化的图像张量
        mean, std: 标准化参数
    Returns:
        numpy array: [H, W, 3] 范围在[0,1]的图像
    """
    # 将张量转换为numpy并转置
    if torch.is_tensor(image_tensor):
        image_np = image_tensor.cpu().numpy()
    else:
        image_np = image_tensor
    
    # 从[3, H, W]转换为[H, W, 3]
    image_np = image_np.transpose(1, 2, 0)
    
    # 反标准化
    mean = np.array(mean)
    std = np.array(std)
    image_np = image_np * std + mean
    
    # 限制到[0,1]范围
    image_np = np.clip(image_np, 0, 1)
    
    return image_np

def visualize_batch_data(batch, batch_idx, save_dir):
    """
    可视化一个batch的数据
    Args:
        batch: 数据加载器返回的batch
        batch_idx: batch索引
        save_dir: 保存路径
    """
    batch_size = batch['image'].size(0)
    n_experts = batch['expert_masks'].size(1)  # [B, N, 2, H, W]
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 为每个样本创建可视化
    for sample_idx in range(batch_size):
        # 获取单个样本数据
        image = batch['image'][sample_idx]  # [3, H, W]
        expert_masks = batch['expert_masks'][sample_idx]  # [N, 2, H, W]
        expert_labels = batch['expert_labels'][sample_idx]  # [N]
        sample_name = batch['name'][sample_idx]
        
        # 反标准化图像
        image_np = denormalize_image(image)
        
        # 创建子图：原图 + N个专家的2类掩码
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, n_experts + 1, figure=fig, hspace=0.3, wspace=0.2)
        
        # 显示原图
        ax_orig = fig.add_subplot(gs[:, 0])
        ax_orig.imshow(image_np)
        ax_orig.set_title(f'Original Image\n{sample_name}', fontsize=12, fontweight='bold')
        ax_orig.axis('off')
        
        # 显示每个专家的掩码
        for expert_idx in range(n_experts):
            expert_label = expert_labels[expert_idx].item()
            disc_mask = expert_masks[expert_idx, 0].cpu().numpy()  # [H, W]
            cup_mask = expert_masks[expert_idx, 1].cpu().numpy()   # [H, W]
            
            # 专家disc掩码
            ax_disc = fig.add_subplot(gs[0, expert_idx + 1])
            ax_disc.imshow(image_np, alpha=0.7)
            ax_disc.imshow(disc_mask, alpha=0.6, cmap='Reds')
            ax_disc.set_title(f'Expert {expert_label}\nDisc Mask', fontsize=10)
            ax_disc.axis('off')
            
            # 专家cup掩码
            ax_cup = fig.add_subplot(gs[1, expert_idx + 1])
            ax_cup.imshow(image_np, alpha=0.7)
            ax_cup.imshow(cup_mask, alpha=0.6, cmap='Blues')
            ax_cup.set_title(f'Expert {expert_label}\nCup Mask', fontsize=10)
            ax_cup.axis('off')
            
            # 叠加显示两个掩码
            ax_combined = fig.add_subplot(gs[2, expert_idx + 1])
            ax_combined.imshow(image_np, alpha=0.7)
            ax_combined.imshow(disc_mask, alpha=0.4, cmap='Reds', label='Disc')
            ax_combined.imshow(cup_mask, alpha=0.4, cmap='Blues', label='Cup')
            ax_combined.set_title(f'Expert {expert_label}\nCombined', fontsize=10)
            ax_combined.axis('off')
        
        # 添加图例
        red_patch = patches.Patch(color='red', alpha=0.6, label='Disc Mask')
        blue_patch = patches.Patch(color='blue', alpha=0.6, label='Cup Mask')
        fig.legend(handles=[red_patch, blue_patch], loc='upper right', fontsize=12)
        
        # 添加总标题
        fig.suptitle(f'Batch {batch_idx + 1} - Sample {sample_idx + 1}\n'
                    f'Expert Labels: {expert_labels.cpu().numpy().tolist()}', 
                    fontsize=16, fontweight='bold')
        
        # 保存图像
        save_path = os.path.join(save_dir, f'batch_{batch_idx + 1}_sample_{sample_idx + 1}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved visualization: {save_path}")

def debug_dataset_visualization(config, num_batches=3):
    """
    Debug函数：可视化数据集的前几个批次
    Args:
        config: 配置字典
        num_batches: 要可视化的批次数量
    """
    log_info("Starting dataset visualization debug...", print_message=True)
    
    # 创建数据集和数据加载器
    dataset_config_path = config.get('dataset_yaml', 'codes/configs/RIGA.yaml')
    dataset = RIGADatasetSimpleV2(dataset_config_path, is_train=True)
    
    # 创建数据加载器
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.get('batch_size', 4),
        shuffle=True,
        num_workers=config.get('num_workers', 2),
        drop_last=False
    )
    
    # 创建保存目录
    save_dir = os.path.join(config.get('save_dir', './debug_output'), 'dataset_visualization')
    os.makedirs(save_dir, exist_ok=True)
    
    log_info(f"Visualization images will be saved to: {save_dir}", print_message=True)
    
    # 采样并可视化指定数量的批次
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
            
        log_info(f"Processing batch {batch_idx + 1}/{num_batches}...", print_message=True)
        log_info(f"Batch info: image shape={batch['image'].shape}, "
                f"expert_masks shape={batch['expert_masks'].shape}, "
                f"expert_labels shape={batch['expert_labels'].shape}", print_message=True)
        
        # 可视化当前批次
        visualize_batch_data(batch, batch_idx, save_dir)
    
    log_info(f"Dataset visualization completed! Images saved to: {save_dir}", print_message=True)

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
            # expert_masks (B, N, C, H, W) -> (B, N*C, H, W)
            # RIGA: C=2 (disc, cup)
            target_expert_masks = batch['expert_masks'].cuda(non_blocking=True)    # (B, N, 2, H, W)
            num_classes = target_expert_masks.size(2)  # C
            target_expert_masks = target_expert_masks.view(target_expert_masks.size(0), -1, target_expert_masks.size(3), target_expert_masks.size(4))  # (B, N*C, H, W)
            # expert_masks (B, N) -> (B, N*C)
            target_expert_labels = batch['expert_labels'].cuda(non_blocking=True)  # (B, N)
            target_expert_labels = target_expert_labels.unsqueeze(2).repeat(1, 1, num_classes).view(target_expert_labels.size(0), -1)  # (B, N*C)

# =============================================================================
# 主函数
# =============================================================================

def main(config, args, override_params):
    """
    主函数 - Debug版本，专门用于数据可视化
    """
    log_info("=" * 80, print_message=True)
    log_info("Starting Dataset Debug Mode", print_message=True)
    log_info("=" * 80, print_message=True)
    
    # 设置随机种子
    set_seed(config.get('seed', 42))
    
    # 记录配置信息
    log_info(f"Config: {config}", print_message=True)
    log_info(f"Override params: {override_params}", print_message=True)
    
    # 开始数据集可视化调试
    debug_dataset_visualization(config, num_batches=3)
    
    log_info("Dataset debug completed successfully!", print_message=True)

            
if __name__ == '__main__':
    parser = setup_training_args_parser()
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    config = load_config(args.config_path)
    config['dataset_yaml'] = args.dataset_yaml
    config, override_params = apply_args_override(config, args)
    
    main(config, args, override_params)