import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import sys
sys.path.append("../")

from nnn import noise  # import nnn.noise as noise
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

# プロットの準備
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x[249:750], y[249:750], label="True sin(x)", color='blue')
line, = ax.plot(x[249:750], np.zeros_like(y)[249:750], label="NN Approximation", color='red', linestyle="dashed")
ax.legend()
ax.set_title("Approximation of sin(x) using a Feedforward Neural Network")
ax.set_xlabel("x")
ax.set_ylabel("sin(x)")
ax.grid()

epochs = 150
history = []

def train_step(epoch):
    optimizer.zero_grad()
    output = net(x_tensor)
    loss = criterion(output, y_tensor)
    loss.backward()
    optimizer.step()
    history.append(output.detach().numpy()[249:750])
    line.set_ydata(history[-1])
    ax.set_title(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}")
    return line,

# アニメーションの作成
ani = animation.FuncAnimation(fig, train_step, frames=epochs, interval=50, blit=True)

#ani.save("anim.mp4",writer='ffmpeg')
plt.show()
