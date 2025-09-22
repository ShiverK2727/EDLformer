import torch
import torch.nn.functional as F
from datasets import BaseRIGADataset


# ==============================================================================
# EDL 数据集 (继承类) - 这是您需要使用的最终版本
# ==============================================================================
class RIGADataset(BaseRIGADataset):
    def __init__(self, config_path=None, is_train=True):
        super(RIGADataset, self).__init__(config_path, is_train)

    def __getitem__(self, index):
        base_sample = super(RIGADataset, self).__getitem__(index)
        
        image = base_sample["image"]          # [3, H, W]
        masks_disc_cup = base_sample["masks_disc_cup"]  # [N, 3, H, W]
        n_experts, num_classes, h, w = masks_disc_cup.shape
        name = base_sample["name"]

        # ==============================================================================
        # 采用您的新方案：分别处理 Disc 和 Cup 的共识，然后重组
        # ==============================================================================

        # 2. 分别计算 Disc 和 Cup 的平均投票
        # masks_disc_cup 的通道1是 disc, 通道2是 cup
        all_disc_masks = masks_disc_cup[:, 1, :, :]  # [N, H, W]
        all_cup_masks = masks_disc_cup[:, 2, :, :]   # [N, H, W]

        # 3. 生成共识掩码：所有专家都同意的区域 (交集)
        #    如果一个像素在所有专家的掩码中值都为1，那么它的最小值为1。
        #    这是一种高效计算交集的方法。
        consensus_disc_binary = torch.min(all_disc_masks, dim=0).values
        consensus_cup_binary = torch.min(all_cup_masks, dim=0).values

        # 4. 强制保证拓扑结构：共识 Disc 必须包含共识 Cup
        #    如果一个像素被认为是 cup，那么它也必须被认为是 disc。
        #    使用 logical_or (并集) 操作可以确保这一点。
        final_consensus_disc = torch.logical_or(consensus_disc_binary, consensus_cup_binary).float()
        final_consensus_cup = consensus_cup_binary

        # 5. 重组为 [bg, rim, cup] 格式的 one-hot 编码
        #    通过减法得到 rim 区域，这样保证了 rim 和 cup 之间没有重叠和空隙。
        final_consensus_rim = final_consensus_disc - final_consensus_cup
        final_consensus_bg = 1.0 - final_consensus_disc # 背景是 disc 之外的区域

        # 堆叠成最终的 one-hot 共识掩码
        consensus_mask = torch.stack([
            final_consensus_bg, 
            final_consensus_rim, 
            final_consensus_cup
        ], dim=0)

        expert_labels = torch.arange(n_experts, dtype=torch.long)  # [N_experts]

        # print(f"unique mask: {torch.unique(consensus_mask)} in sample {name}")
        # print(f"unique expert masks: {torch.unique(base_sample['masks_ring_cup']).tolist()} in sample {name}")
        sample = {
            "image": image,
            "expert_masks": base_sample["masks_ring_cup"], # [N, 3, H, W] one-hot的专家标注
            "expert_labels": expert_labels,    # [N_experts]
            "consensus_mask": consensus_mask, # [3, H, W] one-hot的共识标注
            "val_masks": base_sample["masks_disc_cup"][:, 1:, :, :] , # [N, 2, H, W] # 非one-hot的专家标注
            "name": name
        }
        
        return sample



