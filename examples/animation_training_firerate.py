"""Animated training with firing-rate visualization of hidden-layer activity.

Based on animation_training.py, extended with firing-rate maps that show the
mean CrossingSample activation of every neuron in each hidden layer.
Uses SimpleNNNSample (Monte-Carlo sampling with T samples per input).

    Top panel   : prediction (y vs. x) updated each epoch.
    Lower panels: one firing-rate map per hidden layer.

Firing-rate map axes interpretation:
    x-axis  — input x value  (each of the N training points; from −2π to +2π)
    y-axis  — neuron index within the hidden layer
    shade   — mean CrossingSample output over T samples (firing rate).
               White = 0 (never fired), black = 1 (fired every T-step).
               (CrossingSample output ∈ {0, 0.5, 1} per T-step;
                XOR is computed along the T dimension)

Note: this is a firing-rate map, not a raster plot.
      For a true raster (binary spike events), see animation_training_raster.py.

Noise settings are collected at the top of the file and can be toggled
by commenting / uncommenting one line at a time.

Run from the project root:

    python examples/animation_training_firerate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.animation as animation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from nnn import model


# ============================================================
# NOISE / STRUCTURE SETTINGS — toggle by commenting out lines
# ============================================================

# Model structure: [input_dim, hidden1, hidden2, ..., output_dim]
STRUCTURE = [1, 25, 25, 1]
# STRUCTURE = [1, 50, 50, 1]   # wider hidden layers

# Global Gaussian noise std (used when TRAIN_STDS / EVAL_STDS are None)
STD = 0.5
# STD = 0.1   # low noise  → sparse, localized spikes
# STD = 1.0   # high noise → denser, more distributed spikes

# Per-layer std at TRAINING time  (None = use STD for all layers)
TRAIN_STDS = None
# TRAIN_STDS = [0.5, 0.3]   # stronger noise in first layer
# TRAIN_STDS = [0.5, 0.0]   # noise only in the first hidden layer

# Per-layer std at VISUALIZATION time  (None = same as training, noisy raster)
# std=0.0 → no noise injected → near-deterministic crossing pattern
EVAL_STDS = None
# EVAL_STDS = [0.0, 0.0]   # noiseless raster  (cf. regression_noiseless_result.py)
# EVAL_STDS = [0.5, 0.0]   # noise only in first layer during visualization

# Training hyper-parameters
EPOCHS = 300
LR = 0.01

# Display: number of input points used for x-axis of raster
N_INPUTS = 1000

# ============================================================


# --- Dataset ---
x_np = np.linspace(-2 * np.pi, 2 * np.pi, N_INPUTS).reshape(-1, 1)
y_np = np.sin(x_np)
x_tensor = torch.tensor(x_np, dtype=torch.float32)
y_tensor = torch.tensor(y_np, dtype=torch.float32)

# --- Model ---
net = model.SimpleNNNSample(structure=STRUCTURE, std=STD)  # default t=10
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=LR)

# --- Forward hooks: capture each hidden layer's crossing output ---
n_hidden = len(STRUCTURE) - 2
hidden_dims = [STRUCTURE[i + 1] for i in range(n_hidden)]
hidden_acts: dict[int, np.ndarray] = {}   # idx → [N_INPUTS, H_i]


def make_hook(idx: int):
    def hook(module, input, output):
        hidden_acts[idx] = output.detach().cpu().numpy()
    return hook


for _i, _gc in enumerate(net.gaussian_crossing):
    _gc.register_forward_hook(make_hook(_i))

# Initialize hidden_acts with zeros so imshow starts with a valid array
for _i, _H in enumerate(hidden_dims):
    hidden_acts[_i] = np.zeros((N_INPUTS, _H))

# --- Figure layout: 1 prediction + n_hidden rasters ---
height_ratios = [3] + [1] * n_hidden
fig, axes = plt.subplots(
    1 + n_hidden, 1,
    figsize=(14, 5 + 2.5 * n_hidden),
    gridspec_kw={'height_ratios': height_ratios},
)
ax_pred = axes[0]
ax_rasters = list(axes[1:]) if n_hidden > 1 else [axes[1]]

# --- Prediction subplot ---
DISP = slice(249, 750)   # middle 501 points for display clarity
ax_pred.plot(x_np[DISP], y_np[DISP], label='True sin(x)',
             color='steelblue', lw=2, zorder=3)
pred_line, = ax_pred.plot(
    x_np[DISP], np.zeros_like(y_np[DISP]),
    color='tomato', linestyle='dashed', lw=1.5, label='NN output', zorder=4,
)
ax_pred.set_xlim(float(x_np[DISP][0]), float(x_np[DISP][-1]))
ax_pred.set_ylim(-1.6, 1.6)
ax_pred.set_xlabel('x')
ax_pred.set_ylabel('y')
ax_pred.legend(loc='upper right', fontsize=9)
ax_pred.grid(alpha=0.35)
pred_title = ax_pred.set_title(f'Epoch 0/{EPOCHS}  |  Loss: —')

# --- Raster subplots (imshow; updated via set_data each frame) ---
# Raster matrix shape: [H, N_INPUTS] — rows = neurons, cols = inputs.
# origin='lower' → neuron 1 at bottom, neuron H at top.
# extent maps col/row indices to x-axis input values and y-axis neuron numbers.
x_extent = [float(x_np[0]), float(x_np[-1])]   # actual x values for x-axis
raster_ims = []
for i, (ax_r, H) in enumerate(zip(ax_rasters, hidden_dims)):
    init_mat = np.zeros((H, N_INPUTS))
    im = ax_r.imshow(
        init_mat,
        aspect='auto',
        cmap='binary',           # white = silent (0), black = spike (1)
        vmin=0, vmax=1,
        origin='lower',
        extent=[x_extent[0], x_extent[1], 0.5, H + 0.5],
        interpolation='nearest',
    )
    ax_r.set_xlabel('x  (input value)')
    ax_r.set_ylabel('Neuron')
    ax_r.set_title(
        f'Hidden layer {i + 1}  ({H} neurons)  —  CrossingSample firing rate (mean over T)  '
        f'[std={STD if EVAL_STDS is None else EVAL_STDS[i]}]'
    )
    tick_step = max(1, H // 5)
    ax_r.set_yticks(np.arange(1, H + 1, tick_step))
    raster_ims.append(im)

plt.tight_layout()


# --- Animation update ---
def train_step(epoch: int):
    # 1. Training forward + backward
    net.train()
    optimizer.zero_grad()
    train_out = net(x_tensor, stds=TRAIN_STDS)
    loss = criterion(train_out, y_tensor)
    loss.backward()
    optimizer.step()

    # 2. Visualization forward pass — hooks fill hidden_acts
    net.eval()
    with torch.no_grad():
        vis_out = net(x_tensor, stds=EVAL_STDS)

    # 3. Update prediction line
    pred_line.set_ydata(vis_out.numpy()[DISP])
    pred_title.set_text(f'Epoch {epoch + 1}/{EPOCHS}  |  Loss: {loss.item():.6f}')

    # 4. Update raster images
    # hidden_acts[i] is [N_INPUTS, T, H]; CrossingSample output ∈ {0, 0.5, 1}.
    # Average over T → [N_INPUTS, H] firing rate in [0, 1]; imshow expects [H, N_INPUTS].
    for i, im in enumerate(raster_ims):
        fire_rate = hidden_acts[i].mean(axis=1)   # [N_INPUTS, T, H] → [N_INPUTS, H]
        im.set_data(fire_rate.T)                  # [H, N_INPUTS]

    return [pred_line, pred_title] + raster_ims


ani = animation.FuncAnimation(
    fig, train_step,
    frames=EPOCHS,
    interval=50,
    blit=False,    # suptitle-level Text objects have axes=None; blit=True would crash
    repeat=False,
)

# ani.save('animation_training_raster.mp4', writer='ffmpeg')
plt.show()
