import torch
import yaml
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
import warnings
import argparse

# --- 忽略 Matplotlib 的特定警告 ---
warnings.filterwarnings("ignore", message="The figure layout has changed to tight")

# ==============================================================================
# 步骤 1: 导入您提供的 RIGADataset 代码
# 请确保您的 dataset.py 文件与此脚本放在同一目录中
# ==============================================================================
try:
    # 假设 dataset.py 中有 BaseRIGADataset 和 RIGADataset 的实现
    from edl_datasets import RIGADataset
except ImportError:
    print("错误：无法导入 'dataset.py'。")
    print("请将包含 RIGADataset 和 BaseRIGADataset 类的文件命名为 'dataset.py' 并与此脚本放在同一目录下。")
    # 为了让代码能继续运行，定义一个临时的空类结构
    class RIGADataset(torch.utils.data.Dataset):
        def __init__(self, *args, **kwargs):
            print("警告: 正在使用临时的 RIGADataset。这个类无法加载真实数据。")
        def __getitem__(self, index):
            raise NotImplementedError("请提供真实的 RIGADataset 实现。")
        def __len__(self):
            return 0

# ==============================================================================
# 步骤 2: 可视化函数 (修改为保存图片)
# ==============================================================================
def visualize_batch(batch, batch_num, config_path, output_dir="visualizations"):
    """将一个批次的数据可视化并保存为图片文件"""
    images = batch['image']
    expert_masks = batch['expert_masks'] # (B, N, 3, H, W) [bg, rim, cup]
    consensus_masks = batch['consensus_mask'] # (B, 3, H, W) [bg, rim, cup]
    batch_size = images.shape[0]
    num_experts_to_show = expert_masks.shape[1]  # 显示所有专家

    # 设置总的图窗大小
    fig, axes = plt.subplots(batch_size, 2 + num_experts_to_show, figsize=(12, 3 * batch_size))
    fig.suptitle(f"Visualization of Batch #{batch_num} (from {os.path.basename(config_path)})", fontsize=16)

    for i in range(batch_size):
        # --- 显示原始图像 ---
        ax_img = axes[i, 0] if batch_size > 1 else axes[0]
        # 反归一化 (如果需要) - 假设使用了标准 ImageNet 均值和标准差
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = images[i] * std + mean
        img = to_pil_image(img_tensor.clamp(0, 1))
        
        ax_img.imshow(img)
        ax_img.set_title(f"Sample {i+1}\nOriginal Image")
        ax_img.axis('off')

        # --- 显示专家掩码 ---
        for n in range(num_experts_to_show):
            ax_expert = axes[i, 1 + n] if batch_size > 1 else axes[1 + n]
            expert_rim = expert_masks[i, n, 1, :, :]
            expert_cup = expert_masks[i, n, 2, :, :]
            overlay = torch.stack([expert_cup, expert_rim, torch.zeros_like(expert_cup)], dim=0)
            overlay_img = to_pil_image(overlay)
            ax_expert.imshow(img, alpha=0.6)
            ax_expert.imshow(overlay_img, alpha=0.5)
            ax_expert.set_title(f"Expert #{n+1} Mask\n(Red=Cup, Green=Rim)")
            ax_expert.axis('off')

        # --- 显示共识掩码 ---
        ax_consensus = axes[i, 1 + num_experts_to_show] if batch_size > 1 else axes[1 + num_experts_to_show]
        consensus_rim = consensus_masks[i, 1, :, :]
        consensus_cup = consensus_masks[i, 2, :, :]
        overlay = torch.stack([consensus_cup, consensus_rim, torch.zeros_like(consensus_cup)], dim=0)
        overlay_img = to_pil_image(overlay)
        ax_consensus.imshow(img, alpha=0.6)
        ax_consensus.imshow(overlay_img, alpha=0.5)
        ax_consensus.set_title(f"Consensus Mask\n(Red=Cup, Green=Rim)")
        ax_consensus.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # --- 保存图片而不是显示 ---
    save_path = os.path.join(output_dir, f"batch_{batch_num:02d}.png")
    plt.savefig(save_path)
    plt.close(fig) # 关闭图窗以释放内存
    print(f"    - Visualization saved to: {save_path}")


# ==============================================================================
# 步骤 3: 主执行逻辑 (更新以处理更多批次并创建目录)
# ==============================================================================
def main():
    """主函数，用于解析参数、加载数据和启动可视化。"""
    parser = argparse.ArgumentParser(description="基于指定的YAML配置文件对RIGA数据集进行采样和可视化。")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="指向数据集YAML配置文件的路径 (例如：/path/to/your/RIGA.yaml)。"
    )
    args = parser.parse_args()

    try:
        # --- 加载数据集 ---
        print(f"\n正在从配置文件 '{args.config}' 初始化 RIGADataset...")
        # 假设数据集使用训练集配置和变换 (is_train=True)
        dataset = RIGADataset(config_path=args.config, is_train=True)
        
        if len(dataset) == 0:
             raise RuntimeError("数据集初始化失败，长度为0。请检查 'dataset.py' 的实现和配置文件路径。")
        print(f"数据集加载成功，共找到 {len(dataset)} 个样本。")

        # --- 创建 DataLoader ---
        data_loader = DataLoader(
            dataset,
            batch_size=3,  # 每个批次可视化的样本数
            shuffle=True,
            num_workers=0  # 在Windows上建议为0，避免多进程问题
        )
        
        # --- 创建输出目录 ---
        output_dir = "visualizations"
        os.makedirs(output_dir, exist_ok=True)
        print(f"可视化图片将保存在 '{output_dir}/' 目录下。")

        # --- 迭代并可视化 (增加批次数) ---
        num_batches_to_show = 5 # 要可视化的批次数
        print(f"\n将从数据加载器中取样并可视化 {num_batches_to_show} 个批次...")
        
        for i, sample_batch in enumerate(data_loader):
            if i >= num_batches_to_show:
                break
            
            print(f"\n--- 正在处理批次 #{i+1} ---")
            print("批次数据维度:")
            for key, value in sample_batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"  - {key}: {value.shape}")
            
            visualize_batch(sample_batch, i + 1, args.config, output_dir)

    except FileNotFoundError:
        print(f"\n错误：找不到配置文件 '{args.config}'。请确认路径是否正确。")
    except Exception as e:
        print(f"\n脚本执行时发生严重错误: {e}")
        print("请检查：")
        print("1. 'dataset.py' 是否存在且包含能够解析您YAML文件的 'BaseRIGADataset' 实现。")
        print("2. 您的数据集文件路径是否正确，并且程序有权限访问。")
        print("3. 数据文件格式是否与 'BaseRIGADataset' 中的加载逻辑匹配。")


if __name__ == "__main__":
    main()

