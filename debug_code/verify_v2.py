import os
import sys
import torch
import importlib.util

# Explicitly load DELoss from path
deloss_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../scheduler/DELoss.py'))
spec = importlib.util.spec_from_file_location("DELoss", deloss_path)
DELoss = importlib.util.module_from_spec(spec)
sys.modules["DELoss"] = DELoss
spec.loader.exec_module(DELoss)

print(f"Loaded DELoss from: {DELoss.__file__}")
print(f"DELoss dir: {dir(DELoss)}")

try:
    DSEBridgeLoss = DELoss.DSEBridgeLoss
except AttributeError:
    print("DSEBridgeLoss not found in DELoss module!")
    with open(deloss_path, 'r') as f:
        print(f"File content:\n{f.read()}")
    sys.exit(1)

# Add project root to path for nets import
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from nets.simple_maskformer_v2 import SimpleMaskFormerV2

def verify_v2():
    print("Initializing SimpleMaskFormerV2...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Path to DINOv3 (from previously discovered path)
    dino_path = "/app/huggingface/models--facebook--dinov3-convnext-large-pretrain-lvd1689m/snapshots/e959efa74c867491dcfe3ec3e4f97382e39025b3"
    
    try:
        model = SimpleMaskFormerV2(
            in_channels=3,
            num_classes=2,
            num_experts=4,
            backbone_name="resnet34",
            dino_model_path=dino_path,
            hidden_dim=256,
            use_dino_align=True,
            bridge_layers_indices=[0],
        ).to(device)
    except Exception as e:
        print(f"Warning: Failed to load DINO: {e}")
        print("Initializing WITHOUT DINO...")
        model = SimpleMaskFormerV2(
            in_channels=3,
            num_classes=2,
            num_experts=4,
            backbone_name="resnet34",
            dino_model_path=None,
            hidden_dim=256,
            use_dino_align=False,
            bridge_layers_indices=[],
        ).to(device)
    
    print("Model initialized.")
    if model.dino_model is not None:
        print("DINOv3 loaded successfully.")
    else:
        print("DINOv3 NOT loaded (check path).")

    # Mock Input
    B, C, H, W = 2, 3, 256, 256
    x = torch.randn(B, C, H, W).to(device)
    print(f"Input shape: {x.shape}")

    # Forward Pass
    print("Running forward pass...")
    outputs = model(x)
    
    # Parse outputs
    pred_evidence = outputs['pred_evidence'] # [B, N, K, H, W]
    bridge_feat = outputs['bridge_feat']
    dino_feat = outputs['dino_feat']
    pixel_final_mask = outputs['pixel_final_mask']
    pix_pred_masks = outputs['pix_pred_masks']
    
    print("-" * 30)
    print(f"Pred Evidence: {pred_evidence.shape} (Expected: [B, Experts, Classes, H, W])")
    print(f"Pixel Final Mask: {pixel_final_mask.shape}")
    print(f"Number of transformer layers outputs: {len(pix_pred_masks)}")
    assert len(pix_pred_masks) == 3, f"Expected 3 transformer output maps (one per scale), but got {len(pix_pred_masks)}"
    
    for i, m in enumerate(pix_pred_masks):
        print(f"  Layer {i} mask shape: {m.shape}")
    
    if bridge_feat is not None:
         print(f"Bridge Feat: {bridge_feat.shape}")
    else:
         print("Bridge Feat: None")

    print("-" * 30)
    print("Verification Successful!")

if __name__ == "__main__":
    verify_v2()
    print(f"Total Queries: {model.num_experts * model.num_classes}")
    print("Query index i maps to: Expert = i // num_classes, Class = i % num_classes (primary dim is Expert, inner dim is Class)")
    print("Example: Query 0 is Expert 0, Class 0. Query 1 is Expert 0, Class 1...")

    # Loss Calculation
    print("\nCalculating Loss...")
    criterion = DSEBridgeLoss(num_classes=model.num_classes)
    criterion.update_epoch(0)
    
    # Mock Targets [B, N, H, W] (Expert labels)
    targets = torch.randint(0, model.num_classes, (B, model.num_experts, H, W)).to(device)
    
    task_loss, align_loss = criterion(outputs, targets)
    
    print(f"Task Loss: {task_loss.item():.4f}")
    print(f"Align Loss: {align_loss.item():.4f}")
    print(f"Total Loss: {(task_loss + align_loss).item():.4f}")
    
    print("\nVerification Passed!")

if __name__ == "__main__":
    verify_v2()