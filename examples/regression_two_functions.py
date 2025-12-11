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
x = np.linspace(-2 * np.pi, 2 * np.pi, 200).reshape(-1, 1)
y_sin = np.sin(x)
x = np.linspace(-2 * np.pi, 2 * np.pi, 200).reshape(-1, 1)
y_cos = np.cos(x)

# PyTorchのテンソルに変換
x_tensor = torch.tensor(x, dtype=torch.float32)
y_sin_tensor = torch.tensor(y_sin, dtype=torch.float32)
y_cos_tensor = torch.tensor(y_cos, dtype=torch.float32)

# モデルのインスタンス化
structure = [1, 50, 50, 1]
model_analytic = model.SimpleNNNAnalytic(structure)

# ２種類のstdvecの生成
stdvec_sin = noise.gen_stdvec(50,  0, 25, on_std=0.5, off_std=0.0)
stdvec_cos = noise.gen_stdvec(50, 25, 50, on_std=0.5, off_std=0.0)
stdvecs_sin = [stdvec_sin, stdvec_sin]
stdvecs_cos = [stdvec_cos, stdvec_cos]

# 損失関数と最適化アルゴリズム
criterion = nn.MSELoss()
optimizer = optim.Adam(model_analytic.parameters(), lr=0.01)

# 学習
epochs = 1500
print("Training starts")
for epoch in range(epochs):
    # sin(x)の学習
    optimizer.zero_grad()
    output = model_analytic(x_tensor, stdvecs_sin)
    loss = criterion(output, y_sin_tensor)
    print("\r"+f"loss: {loss}",end="")
    loss.backward()
    optimizer.step()
    # cos(x)の学習
    # fc層のバイアス次第では完全に分けてもOK
    optimizer.zero_grad()
    output = model_analytic(x_tensor, stdvecs_cos)
    loss = criterion(output, y_cos_tensor)
    print("\r"+f"loss: {loss}",end="")
    loss.backward()
    optimizer.step()
print(".")
print("Training ends")

# サンプリングモデルのインスタンス化と重みのコピー
model_sample = model.SimpleNNNSample(structure)
for i in range(len(model_analytic.fcs)):
    model_sample.fcs[i].load_state_dict(model_analytic.fcs[i].state_dict())

# 損失関数と最適化アルゴリズム
criterion = nn.MSELoss()
optimizer = optim.Adam(model_sample.parameters(), lr=0.01)

# 馴らし
epochs = 1000
print("Habituation starts")
for epoch in range(epochs):
    # sin(x)の学習
    optimizer.zero_grad()
    output = model_sample(x_tensor, stdvecs_sin)
    loss = criterion(output, y_sin_tensor)
    print("\r"+f"loss: {loss}",end="")
    loss.backward()
    optimizer.step()
    # cos(x)の学習
    # fc層のバイアス次第では完全に分けてもOK
    optimizer.zero_grad()
    output = model_sample(x_tensor, stdvecs_cos)
    loss = criterion(output, y_cos_tensor)
    print("\r"+f"loss: {loss}",end="")
    loss.backward()
    optimizer.step()
print(".")
print("Habituation ends")


# 統計量モデルのインスタンス化と重みのコピー
model_statistic = model.SimpleNNNStatistic(structure)
for i in range(len(model_analytic.fcs)):
    model_statistic.fcs[i].load_state_dict(model_analytic.fcs[i].state_dict())

# 損失関数と最適化アルゴリズム
criterion = nn.MSELoss()
optimizer = optim.Adam(model_statistic.parameters(), lr=0.01)

# 馴らし
epochs = 1000
print("Habituation starts")
for epoch in range(epochs):
    # sin(x)の学習
    optimizer.zero_grad()
    output = model_statistic(x_tensor, stdvecs_sin)
    loss = criterion(output, y_sin_tensor)
    print("\r"+f"loss: {loss}",end="")
    loss.backward()
    optimizer.step()
    # cos(x)の学習
    # fc層のバイアス次第では完全に分けてもOK
    optimizer.zero_grad()
    output = model_statistic(x_tensor, stdvecs_cos)
    loss = criterion(output, y_cos_tensor)
    print("\r"+f"loss: {loss}",end="")
    loss.backward()
    optimizer.step()
print(".")
print("Habituation ends")

  
# 予測
y_pred_sin = model_analytic(x_tensor, stdvecs_sin).detach().numpy()
y_pred_sample_sin = model_sample(x_tensor, stdvecs_sin).detach().numpy()
y_pred_statistic_sin = model_statistic(x_tensor, stdvecs_sin).detach().numpy()

y_pred_cos = model_analytic(x_tensor, stdvecs_cos).detach().numpy()
y_pred_sample_cos = model_sample(x_tensor, stdvecs_cos).detach().numpy()
y_pred_statistic_cos = model_statistic(x_tensor, stdvecs_cos).detach().numpy()

# 可視化
plt.rcParams["font.size"] = 12
plt.figure(figsize=(8, 6))
plt.plot(x, y_sin, label="True sin(x)", color='black')
plt.plot(x, y_pred_sample_sin, label="NNN (Sampling)", color='red', alpha=0.3, linestyle="dashed")
plt.plot(x, y_pred_statistic_sin, label="NNN (Statistic)", color='blue', alpha=0.3, linestyle="dashed")
plt.plot(x, y_pred_sin, label="NNN (Analytic)", color='orange', linestyle="dashed")

plt.plot(x, y_cos, label="True cos(x)", color='black')
plt.plot(x, y_pred_sample_cos, label="NNN (Sampling)", color='red', alpha=0.3, linestyle="dashed")
plt.plot(x, y_pred_statistic_cos, label="NN (Statistic)", color='blue', alpha=0.3, linestyle="dashed")
plt.plot(x, y_pred_cos, label="NNN (Analytic)", color='orange', linestyle="dashed")

plt.legend(loc='upper right')
#plt.title("Approximation of sin(x) and cos(x) using a NNN")
plt.xlabel("x")
plt.ylabel("sin(x) / cos(x)")
plt.xlim(-1.5*np.pi, 1.5*np.pi)
plt.ylim(-1.5, 1.5)
plt.grid()
plt.show()
