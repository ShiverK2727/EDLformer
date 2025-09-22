import os
import yaml

import numpy as np
import pandas as pd
import torch
from torch.utils import data
import torchvision.transforms.functional as F
import PIL.Image
from logger import log_info, log_error  



class BaseRIGADataset(data.Dataset):
    def __init__(self, config_path, is_train=True):
        super(BaseRIGADataset, self).__init__()
        
        self.config = self._load_config(config_path)
        self.root = self.config.get('root', '')
        self.datasets = self.config.get('train_list', []) if is_train else self.config.get('valid_list', [])
        self.scale_size = 256

        transform_key = 'train_transform' if is_train else 'valid_transform'
        transform_config = self.config.get(transform_key, []) # 预期为列表
        self.transform = self._build_transform(transform_config)
        print(f"Transforms for {'train' if is_train else 'valid'} set: {self.transform}")
        
        self.dataframe = pd.DataFrame()
        for split in self.datasets:
            try:
                df_all = pd.read_csv(os.path.join(self.root, f'Glaucoma_multirater_{split}.csv'), encoding='gbk')
                df_filtered = df_all[df_all['rater'] == 0].copy()
                self.dataframe = pd.concat([self.dataframe, df_filtered])
            except FileNotFoundError:
                print(f"Warning: CSV file for dataset '{split}' not found.")
        
        self.dataframe.reset_index(drop=True, inplace=True)
        
        self.PIXEL_DISC_REGION = 255

    def _load_config(self, config_path):
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        raise FileNotFoundError(f"Config file not found at {config_path}")

    def _build_transform(self, transform_config):
        transform_list = []
        if not transform_config or not isinstance(transform_config, list):
            return None
            
        for transform_item in transform_config:
            if not isinstance(transform_item, dict) or 'name' not in transform_item:
                print(f"Warning: Invalid transform item format: {transform_item}. Skipping.")
                continue

            name = transform_item['name']
            params = transform_item.get('params')
            
            transform_class = globals().get(name)
            if transform_class:
                try:
                    if params:
                        # 传递字典参数
                        transform_list.append(transform_class(**params))
                    else:
                        # 无参数
                        transform_list.append(transform_class())
                except Exception as e:
                    print(f"Error initializing transform '{name}' with params {params}: {e}")
            else:
                print(f"Warning: Transform '{name}' not found in globals and will be skipped.")
        
        if not transform_list:
            return None
        
        return Compose(transform_list)


    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        img_name = row['imgName'].replace('\\', '/')
        seg_name_template = row['maskName']

        img_path = os.path.join(self.root, img_name)
        img_pil = PIL.Image.open(img_path).convert('RGB').resize((self.scale_size, self.scale_size))
        img_np = np.array(img_pil)

        masks_disc_cup = []
        masks_ring_cup = []
        for n in range(1, 7):
            n_seg_path = seg_name_template.replace('FinalLabel', f'Rater{n}')
            n_seg_path = os.path.join(self.root, n_seg_path).replace('\\', '/')
            try:
                n_seg_pil = PIL.Image.open(n_seg_path).convert('L').resize((self.scale_size, self.scale_size), PIL.Image.NEAREST)
                n_seg_np = np.array(n_seg_pil)

                is_cup = (n_seg_np == self.PIXEL_DISC_REGION)
                is_ring = (n_seg_np > 0) & (~is_cup)
                is_disc = is_cup | is_ring

                background_mask = (n_seg_np == 0)

                cup_mask = is_cup.astype(np.float32)
                disc_mask = is_disc.astype(np.float32)
                ring_mask = is_ring.astype(np.float32)
                
                disc_cup_mask = np.stack([background_mask, disc_mask, cup_mask], axis=0)
                ring_cup_mask = np.stack([background_mask, ring_mask, cup_mask], axis=0)
                
                masks_disc_cup.append(disc_cup_mask)
                masks_ring_cup.append(ring_cup_mask)

            except FileNotFoundError:
                log_error(f"Mask file not found: {n_seg_path}. Skipping this mask.")
                pass
    
        masks_disc_cup = np.stack(masks_disc_cup, axis=0) # [N, 3, H, W]
        masks_ring_cup = np.stack(masks_ring_cup, axis=0) # [N, 3, H, W]

        sample = {"image": img_np, 
                  "masks_disc_cup": masks_disc_cup,
                  "masks_ring_cup": masks_ring_cup,
                  "name": img_name.split('.')[0]}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
class RIGADatasetSimple(BaseRIGADataset):
    def __init__(self, config_path=None, is_train=True):
        super(RIGADatasetSimple, self).__init__(config_path, is_train)

    def __getitem__(self, index):
        base_sample = super(RIGADatasetSimple, self).__getitem__(index)
        
        image = base_sample["image"]          # [3, H, W]
        masks_disc_cup = base_sample["masks_disc_cup"]  # [N, 3, H, W]
        n_experts, num_classes, h, w = masks_disc_cup.shape
        name = base_sample["name"]

        # 2. 分别计算 Disc 和 Cup 的平均投票
        # masks_disc_cup 的通道1是 disc, 通道2是 cup
        all_disc_masks = masks_disc_cup[:, 1, :, :]  # [N, H, W]
        all_cup_masks = masks_disc_cup[:, 2, :, :]   # [N, H, W]

        all_expert_masks = torch.cat([all_disc_masks, all_cup_masks], dim=0)  # [2N, H, W]

        expert_labels = torch.arange(n_experts * 2, dtype=torch.long)  # [2N_experts]
        mask_labels = [0] * n_experts + [1] * n_experts  # Disc=0, Cup=1

        val_masks = torch.stack([all_disc_masks, all_cup_masks], dim=1).float()  # [N, 2, H, W]

        sample = {
            "image": image,
            "expert_masks": all_expert_masks, # [2N, H, W]
            "expert_labels": expert_labels,   # [2N]
            "mask_labels": torch.tensor(mask_labels, dtype=torch.long), # [2N], Disc=0, Cup=1
            "val_masks": val_masks,           # [N, 2, H, W]
            "name": name
        }
        return sample

# =================================================================================
#  数据增强 Transform 类的定义 (已重构为 Tensor-first)
#  为了使代码自包含，我们将这些类直接定义在这里。
# =================================================================================

class Compose:
    """将多个 transform 组合在一起。"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample
    
    def __repr__(self):
        # 打印transform管道，方便调试
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string


class ToTensor:
    """将样本中的Numpy数组转换为PyTorch张量。这应该是管道的第一步。"""
    def __call__(self, sample):
        image = sample['image']
        # 转换图像: HWC -> CHW, 并从 [0, 255] 归一化到 [0, 1]
        sample['image'] = torch.from_numpy(image.transpose((2, 0, 1))).float().div(255)

        # 转换掩码
        for key in sample:
            if 'masks' in key:
                sample[key] = torch.from_numpy(sample[key]).float()
        
        return sample


class Normalize:
    """对图像张量进行标准化。"""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        sample['image'] = F.normalize(sample['image'], self.mean, self.std)
        return sample


class RandomHorizontalFlip:
    """以一定概率水平翻转图像和所有掩码 (Tensor版本)。"""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if torch.rand(1) < self.p:
            sample['image'] = F.hflip(sample['image'])
            for key in sample:
                if 'masks' in key:
                    sample[key] = F.hflip(sample[key])
        return sample


class RandomVerticalFlip:
    """以一定概率垂直翻转图像和所有掩码 (Tensor版本)。"""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if torch.rand(1) < self.p:
            sample['image'] = F.vflip(sample['image'])
            for key in sample:
                if 'masks' in key:
                    sample[key] = F.vflip(sample[key])
        return sample


class RandomColorJitter:
    """以一定概率随机改变图像的亮度、对比度。只作用于图像。"""
    def __init__(self, p=0.5, brightness=0, contrast=0):
        self.p = p
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, sample):
        if torch.rand(1) < self.p:
            # 从 [max(0, 1 - brightness), 1 + brightness] 中随机选择一个值
            if self.brightness > 0:
                brightness_factor = torch.tensor(1.0).uniform_(1 - self.brightness, 1 + self.brightness).item()
                sample['image'] = F.adjust_brightness(sample['image'], brightness_factor)
            
            # 从 [max(0, 1 - contrast), 1 + contrast] 中随机选择一个值
            if self.contrast > 0:
                contrast_factor = torch.tensor(1.0).uniform_(1 - self.contrast, 1 + self.contrast).item()
                sample['image'] = F.adjust_contrast(sample['image'], contrast_factor)
            
        return sample


class RandomGamma:
    """以一定概率随机进行Gamma校正。只作用于图像。"""
    def __init__(self, p=0.5, gamma_range=(0.8, 1.2)):
        self.p = p
        self.gamma_min, self.gamma_max = gamma_range

    def __call__(self, sample):
        if torch.rand(1) < self.p:
            gamma = torch.tensor(1.0).uniform_(self.gamma_min, self.gamma_max).item()
            sample['image'] = F.adjust_gamma(sample['image'], gamma)
        return sample


