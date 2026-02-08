import torch
import torch.nn as nn
import torch.nn.functional as F

class DSEBridgeLoss(nn.Module):
    def __init__(self, num_classes, annealing_step=10):
        super().__init__()
        self.num_classes = num_classes
        self.annealing_step = annealing_step
        self.epoch_num = 0 

    def update_epoch(self, epoch):
        self.epoch_num = epoch

    def compute_gram_matrix(self, x):
        B, C, H, W = x.shape
        features = x.view(B, C, H * W)
        gram = torch.bmm(features, features.transpose(1, 2)) / (H * W)
        return gram

    def edl_loss(self, evidence, targets):
        """
        Bayes Risk Loss (MSE) + KL Reg
        evidence: [B, N, K, H, W]
        targets: [B, N, H, W]
        """
        B, N, K, H, W = evidence.shape
        evidence = evidence.view(B * N, K, H, W)
        targets = targets.view(B * N, H, W)
        
        alpha = evidence + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        
        # One-hot target
        # Assuming targets in [0, K-1]
        y = F.one_hot(targets, num_classes=K).permute(0, 3, 1, 2).float()

        # Risk (MSE)
        prob = alpha / S
        diff = (y - prob) ** 2
        var = alpha * (S - alpha) / (S * S * (S + 1))
        loss_mse = torch.sum(diff + var, dim=1, keepdim=True)
        
        # KL Divergence
        # Use torch.lgamma instead of lgamma
        annealing_coef = min(1, self.epoch_num / self.annealing_step)
        alpha_tilde = y + (1 - y) * alpha 
        kl_loss = torch.lgamma(alpha_tilde).sum(dim=1) \
                  - torch.lgamma(torch.tensor(1.0, device=evidence.device)) * K \
                  - torch.lgamma(torch.sum(alpha_tilde, dim=1)) \
                  + torch.lgamma(torch.tensor(K * 1.0, device=evidence.device))
        
        return torch.mean(loss_mse + annealing_coef * kl_loss)

    def forward(self, outputs, targets):
        # targets: [B, N, H, W]
        evidence = outputs["pred_evidence"]
        
        # 1. Task Loss
        task_loss = self.edl_loss(evidence, targets)
        
        # 2. Gram Alignment Loss
        align_loss = torch.tensor(0.0, device=evidence.device)
        bridge_feat = outputs.get("bridge_feat")
        dino_feat = outputs.get("dino_feat")
        
        if dino_feat is not None and bridge_feat is not None:
            g_bridge = self.compute_gram_matrix(bridge_feat)
            g_dino = self.compute_gram_matrix(dino_feat.detach())
            align_loss = F.mse_loss(g_bridge, g_dino)
            
        return task_loss, align_loss
