# -*- coding: utf-8 -*-
"""
@Time ： 2025/6/26 1:07
@Auth ： 孙殿芳
"""
import torch
import jittor as jt
import numpy as np

# 加载模型的权重文件
torch_weight_path = 'weights/iou_67.86_IRSTD-1k.pkl'
jittor_weight_path = 'weights/iou_67.86_IRSTD-1k_jittor.npz'

# 加载 PyTorch 权重
checkpoint = torch.load(torch_weight_path, map_location=torch.device('cpu'))
torch_state = checkpoint['net'] if 'net' in checkpoint else checkpoint  # 看是完整 checkpoint 还是 state_dict

# 加载 Jittor 权重
ckpt = jt.load(jittor_weight_path)

# 打印 PyTorch 和 Jittor 权重的键数
print(f"\n📦 PyTorch 参数数: {len(torch_state)}")
print(f"📦 Jittor 参数数: {len(ckpt)}\n")

# 检查转换是否成功
print("=== 🔍 权重键对比 ===")
mismatch_count = 0
for k in torch_state:
    if k in ckpt:
        torch_shape = tuple(torch_state[k].shape)
        jittor_shape = tuple(ckpt[k].shape)
        if torch_shape == jittor_shape:
            print(f"[✓] {k} | 形状一致: {torch_shape}")
        else:
            print(f"[×] {k} | 🔺 形状不一致: torch {torch_shape} vs jittor {jittor_shape}")
            mismatch_count += 1
    else:
        print(f"[×] {k} | ⛔ 在 Jittor 权重中缺失")
        mismatch_count += 1

# 检查 Jittor 权重中是否有 PyTorch 中不存在的参数（多余项）
extra_keys = set(ckpt.keys()) - set(torch_state.keys())
if extra_keys:
    print("\n⚠️ 以下参数存在于 Jittor 权重中，但不在 PyTorch 权重中：")
    for k in extra_keys:
        print(" -", k)

# 总结
print("\n=== 🧾 总结 ===")
if mismatch_count == 0 and not extra_keys:
    print("🎉 权重转换成功！所有键和形状均一致。")
else:
    print(f"🔧 存在 {mismatch_count} 个未匹配项，或形状不一致。请检查转换逻辑是否存在 bug。")
