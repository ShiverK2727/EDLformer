import torch
from transformers import AutoModel, AutoConfig

# 不要用 ID，直接用本地绝对路径
# 根据你之前的 ls 输出，这是 snapshot 的完整路径
local_model_path = "/app/huggingface/models--facebook--dinov3-convnext-large-pretrain-lvd1689m/snapshots/e959efa74c867491dcfe3ec3e4f97382e39025b3"

try:
    # 加载配置
    config = AutoConfig.from_pretrained(local_model_path)
    # 加载模型
    model = AutoModel.from_pretrained(local_model_path, config=config)
    print("模型加载成功！")
    print(model)
except Exception as e:
    print(f"加载失败: {e}")