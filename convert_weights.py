# convert_weights_to_jittor.py

import torch
import jittor as jt
import numpy as np
from model.MSHNet import MSHNet  # Jittor 版模型

# 加载 PyTorch 权重
pytorch_weights = torch.load("weights/iou_67.86_IRSTD-1k.pkl", map_location='cpu')
state_dict = pytorch_weights['state_dict'] if 'state_dict' in pytorch_weights else pytorch_weights

# 初始化 Jittor 模型
model = MSHNet(input_channels=1)  # 或 input_channels=3，如果你的是RGB图像
jt_model_dict = model.state_dict()

converted_weights = {}

missing_keys = []
matched_keys = []

for name in jt_model_dict.keys():
    if name in state_dict:
        # 将 torch.tensor 转为 numpy 后变成 jt.array
        converted_weights[name] = jt.array(state_dict[name].cpu().numpy())
        matched_keys.append(name)
    else:
        missing_keys.append(name)

# 加载已转换权重
model.load_parameters(converted_weights)

# 保存为 .npz 文件
jt.save(model.state_dict(), "weights/iou_67.86_IRSTD-1k_jittor.npz")
print("✅ 权重转换完成，已保存为 weights/iou_67.86_IRSTD-1k_jittor.npz")
print(f"✅ 成功匹配参数数量: {len(matched_keys)}")
print(f"⚠️ 未匹配的参数数量: {len(missing_keys)}")
if missing_keys:
    print("未匹配的键名如下（可能为缓冲变量或不必要的）:")
    for k in missing_keys:
        print(" -", k)
