# 必要ライブラリ
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# データ生成
x_vals = np.linspace(-2 * np.pi, 2 * np.pi, 200)
y_sin = np.sin(x_vals)
y_cos = np.cos(x_vals)

x_data = np.concatenate([x_vals, x_vals])
y_data = np.concatenate([y_sin, y_cos])
x_tensor = torch.tensor(x_data, dtype=torch.float32).unsqueeze(1)
y_tensor = torch.tensor(y_data, dtype=torch.float32).unsqueeze(1)
dataset = TensorDataset(x_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# RBF層とネットワーク
class RBFLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(out_features, in_features))
        self.log_sigmas = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        x = x.unsqueeze(1)
        d = torch.sum((x - self.centers.unsqueeze(0)) ** 2, dim=2)
        sigmas = torch.exp(self.log_sigmas).unsqueeze(0)
        return torch.exp(-d / (2 * sigmas ** 2))

class RBFNetwork(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.rbf = RBFLayer(in_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, sigma_mask):
        h = self.rbf(x) * sigma_mask
        return self.linear(h)

# 動的mask管理
class DynamicMaskManager(nn.Module):
    def __init__(self, hidden_dim, max_masks=10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.masks = nn.ParameterList()
        self.max_masks = max_masks
        self.add_mask()

    def add_mask(self):
        self.masks.append(nn.Parameter(torch.ones(1, self.hidden_dim) * 0.9))

    def remove_mask(self, idx):
        new_masks = [p for i, p in enumerate(self.masks) if i != idx]
        self.masks = nn.ParameterList(new_masks)

    def get_masks(self):
        return [torch.clamp(m, 0.0, 1.0) for m in self.masks]

    def num_masks(self):
        return len(self.masks)

# 初期化
Nh = 20
net = RBFNetwork(1, Nh, 1)
mask_manager = DynamicMaskManager(hidden_dim=Nh)
optimizer = optim.Adam(list(net.parameters()) + list(mask_manager.parameters()), lr=0.01)
loss_fn = nn.MSELoss()
lambda_l1 = 1e-3
similarity_threshold = 0.98

# 学習ループ
for epoch in range(1500):
    losses_per_mask = [0.0 for _ in range(mask_manager.num_masks())]
    usage_counts = [0 for _ in range(mask_manager.num_masks())]

    for x_batch, y_batch in loader:
        masks = mask_manager.get_masks()
        preds = [net(x_batch, m) for m in masks]
        errors = [torch.abs(pred - y_batch) for pred in preds]
        total_errors = torch.stack([e.mean(dim=1) for e in errors], dim=1)
        assign = torch.argmin(total_errors, dim=1)

        optimizer.zero_grad()
        for i, m in enumerate(masks):
            idxs = (assign == i)
            if idxs.sum().item() == 0:
                continue
            x_sub, y_sub = x_batch[idxs], y_batch[idxs]
            pred = net(x_sub, m)
            loss = loss_fn(pred, y_sub) + lambda_l1 * m.abs().sum()
            loss.backward(retain_graph=True)
            losses_per_mask[i] += loss.item()
            usage_counts[i] += len(x_sub)
        optimizer.step()

    # 条件に応じてmask追加
    if epoch % 10 == 0 and mask_manager.num_masks() < mask_manager.max_masks:
        if max(losses_per_mask) > 0.1 and min(usage_counts) > 10:
            mask_manager.add_mask()

    # 重複mask削除
    with torch.no_grad():
        masks = mask_manager.get_masks()
        to_remove = set()
        for i in range(len(masks)):
            for j in range(i+1, len(masks)):
                sim = torch.cosine_similarity(masks[i], masks[j])
                if sim > similarity_threshold:
                    to_remove.add(j)
        for idx in sorted(to_remove, reverse=True):
            mask_manager.remove_mask(idx)

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: masks={mask_manager.num_masks()}, usage={usage_counts}")

# 推論・プロット
x_test = torch.linspace(-2 * np.pi, 2 * np.pi, 500).unsqueeze(1)
with torch.no_grad():
    masks = mask_manager.get_masks()
    preds = [net(x_test, m).squeeze().numpy() for m in masks]

plt.plot(x_vals, y_sin, 'r--', label='sin(x)')
plt.plot(x_vals, y_cos, 'b--', label='cos(x)')

#plt.xlim(-2 * np.pi, 2 * np.pi)

colors = ['g', 'm', 'c', 'orange', 'purple']
for i, y in enumerate(preds):
    plt.plot(x_test.numpy(), y, color=colors[i % len(colors)], label=f'mask {i}')
plt.title("Dynamic Masked Subnetworks")
plt.legend()
plt.grid(True)
plt.show()