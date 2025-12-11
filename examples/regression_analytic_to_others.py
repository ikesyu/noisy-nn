import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("../")

import nnn.noise as noise #from nnn import noise
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
model_analytic = model.SimpleNNNAnalytic()

# 損失関数と最適化アルゴリズム
criterion = nn.MSELoss()
optimizer = optim.Adam(model_analytic.parameters(), lr=0.01)

# 学習
epochs = 5000
print("Training starts")
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model_analytic(x_tensor)
    loss = criterion(output, y_tensor)
    print("\r"+f"loss: {loss}",end="")
    loss.backward()
    optimizer.step()
print(".")
print("Training ends")

# サンプリングモデルのインスタンス化と重みのコピー
model_sample = model.SimpleNNNSample()
for i in range(len(model_analytic.fcs)):
  model_sample.fcs[i].load_state_dict(model_analytic.fcs[i].state_dict())

""" # モデルパラメータの緩和
std=1e-3
with torch.no_grad():
  for param in model_sample.parameters():
    if param.requires_grad:
      noise = torch.randn_like(param) * std
      param.add_(noise) """

# 損失関数と最適化アルゴリズム
criterion = nn.MSELoss()
optimizer = optim.Adam(model_sample.parameters(), lr=0.01)

# 重みの「馴らし」
epochs = 2000
print("Training starts")
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model_sample(x_tensor)
    loss = criterion(output, y_tensor)
    print("\r"+f"loss: {loss}",end="")
    loss.backward()
    optimizer.step()
print(".")
print("Training ends")


# 統計量モデルのインスタンス化と重みのコピー
model_statistic = model.SimpleNNNStatistic()
for i in range(len(model_analytic.fcs)):
  model_statistic.fcs[i].load_state_dict(model_analytic.fcs[i].state_dict())

""" # モデルパラメータの緩和
std=1e-3
with torch.no_grad():
  for param in model_sample.parameters():
    if param.requires_grad:
      noise = torch.randn_like(param) * std
      param.add_(noise)  
 """
# 損失関数と最適化アルゴリズム
criterion = nn.MSELoss()
optimizer = optim.Adam(model_statistic.parameters(), lr=0.01)

# 重みの「馴らし」
epochs = 2000
print("Training starts")
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model_statistic(x_tensor)
    loss = criterion(output, y_tensor)
    print("\r"+f"loss: {loss}",end="")
    loss.backward()
    optimizer.step()
print(".")
print("Training ends")
  
# 予測
y_pred = model_analytic(x_tensor).detach().numpy()
y_pred_sample = model_sample(x_tensor).detach().numpy()
y_pred_statistic = model_statistic(x_tensor).detach().numpy()

# 可視化
plt.figure(figsize=(8, 6))
plt.plot(x, y, label="True sin(x)", color='black')
plt.plot(x, y_pred_sample, label="NN Approximation (Sampling-base)", color='red', alpha=0.3, linestyle="dashed")
plt.plot(x, y_pred_statistic, label="NN Approximation (Stats-base)", color='blue', alpha=0.3, linestyle="dashed")
plt.plot(x, y_pred, label="NN Approximation", color='orange', linestyle="dashed")
plt.legend()
plt.title("Approximation of sin(x) using a Feedforward Neural Network")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.grid()
plt.show()
