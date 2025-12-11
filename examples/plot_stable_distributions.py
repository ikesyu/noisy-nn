import torch
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append("../")
from nnn import noise

# Parameters
alpha = torch.Tensor([2.0])
beta = torch.Tensor([0])
gamma = torch.Tensor([1.0])

# Sampling noise
sample = torch.tensor(np.zeros(1000), dtype=torch.float32)
gen = noise.stable_noise_like(alpha, beta, gamma)
sample = gen(sample)
data = sample.to('cpu').detach().numpy().copy()

# PDF
x = np.linspace(-5, 5, 100)
x_tensor = torch.tensor(x, dtype=torch.float32)
pdf = noise.stable_pdf_torch(alpha, beta, gamma)
y_tensor = pdf(x_tensor)
y = y_tensor.to('cpu').detach().numpy().copy()

# Create 2 ax
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))  # 1行2列の配置

# Histogram
ax1.hist(data, bins=30, range=(-5,5), color='skyblue', edgecolor='black')
ax1.set_title("Histogram")
ax1.set_xlabel("Value")
ax1.set_xlim([-5,5])
ax1.set_ylabel("Frequency")

# Line
ax2.plot(x, y, color='red')
ax2.set_title("Line Graph")
ax2.set_xlabel("X")
ax2.set_ylabel("Y")

# Layout & Show
plt.tight_layout()
plt.show()