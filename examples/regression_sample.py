import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("../")

from nnn import noise #import nnn.noise as noise
from nnn import activation
from nnn import layer
from nnn import model

# データセットの作成
x = np.linspace(-2 * np.pi, 2 * np.pi, 1000).reshape(-1, 1)
y = np.sin(x)

# PyTorchのテンソルに変換
x_tensor = torch.tensor(x, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# モデルのインスタンス化
net = model.SimpleNNNSample()

# 損失関数と最適化アルゴリズム
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)

# 学習
epochs = 1000
print("Training starts")
for epoch in range(epochs):
    optimizer.zero_grad()
    output = net(x_tensor)
    loss = criterion(output, y_tensor)
    print("\r"+f"loss: {loss}",end="")
    loss.backward()
    optimizer.step()
print(".")
print("Training ends")

# 予測
y_pred = net(x_tensor).detach().numpy()

# 可視化
plt.figure(figsize=(8, 6))
plt.plot(x, y, label="True sin(x)", color='blue')
plt.plot(x, y_pred, label="NN Approximation", color='red', linestyle="dashed")
plt.legend()
plt.title("Approximation of sin(x) using a Feedforward Neural Network")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.grid()
plt.show()