"""Animate smooth noise-field transition between two learned sub-networks.

Two sub-networks share the same linear weight matrices but are separated
by their noise injection region (noise field):

    Sin sub-network : neurons 0 .. H/2-1 receive Gaussian noise (std = ON_STD)
    Cos sub-network : neurons H/2 .. H-1 receive Gaussian noise (std = ON_STD)

Training alternates minimising MSE(net(x, stdvecs_sin), sin(x)) and
MSE(net(x, stdvecs_cos), cos(x)) in each epoch, identical in spirit to
regression_two_functions.py.

After training the animation sweeps α ∈ [0, 1] using
    noise.interpolate_stdvecs(stdvec_sin, stdvec_cos, rate=α)
to shift the active noise region smoothly from the sin half to the cos half
and back, forming a cyclic loop.

Panels (updated every animation frame):
    Top      : prediction curve  y vs. x
    Middle   : noise field — std per neuron shown as colour intensity
    Lower ×2 : firing-rate map per hidden layer
               (mean CrossingSample activation over T samples;
                same visualisation as animation_training_firerate.py)

Model: SimpleNNNSample (Monte-Carlo, default t=10 samples per input)

Run from the project root:

    python examples/animation_2funcs_firerate_sample.py
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

from nnn import noise, model


# ============================================================
# SETTINGS
# ============================================================

STRUCTURE = [1, 50, 50, 1]   # [input_dim, hidden1, hidden2, output_dim]
ON_STD    = 0.5              # noise std in the active region
OFF_STD   = 0.0              # noise std in the inactive region
N_INPUTS  = 200              # number of training / display points
EPOCHS    = 1500             # training epochs (sin + cos alternated per epoch)
LR        = 0.01

# Animation
N_HALF    = 60               # frames per half-sweep (sin→cos or cos→sin)
N_HOLD    = 30               # frames to hold at each pure mode (α=0 or α=1)
INTERVAL  = 60               # ms per frame

# ============================================================


# --- Dataset ---
x_np   = np.linspace(-2 * np.pi, 2 * np.pi, N_INPUTS).reshape(-1, 1)
sin_np = np.sin(x_np)
cos_np = np.cos(x_np)
x_tensor   = torch.tensor(x_np,   dtype=torch.float32)
sin_tensor = torch.tensor(sin_np, dtype=torch.float32)
cos_tensor = torch.tensor(cos_np, dtype=torch.float32)

# --- Noise field definition ---
H        = STRUCTURE[1]          # hidden dim (assumed equal across hidden layers)
HALF     = H // 2
n_hidden = len(STRUCTURE) - 2

stdvec_sin  = noise.gen_stdvec(H, 0,    HALF, on_std=ON_STD, off_std=OFF_STD)
stdvec_cos  = noise.gen_stdvec(H, HALF, H,    on_std=ON_STD, off_std=OFF_STD)
stdvecs_sin = [stdvec_sin] * n_hidden   # same field applied to every hidden layer
stdvecs_cos = [stdvec_cos] * n_hidden

# --- Model ---
net       = model.SimpleNNNSample(structure=STRUCTURE)
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=LR)

# --- Training ---
print(f"Training  (structure={STRUCTURE}, ON_STD={ON_STD}, epochs={EPOCHS}) …")
for epoch in range(1, EPOCHS + 1):
    net.train()
    # sin sub-network
    optimizer.zero_grad()
    loss_sin = criterion(net(x_tensor, stds=stdvecs_sin), sin_tensor)
    loss_sin.backward()
    optimizer.step()
    # cos sub-network
    optimizer.zero_grad()
    loss_cos = criterion(net(x_tensor, stds=stdvecs_cos), cos_tensor)
    loss_cos.backward()
    optimizer.step()
    if epoch == 1 or epoch % 300 == 0:
        print(f"  epoch {epoch:5d} | sin {loss_sin.item():.4e}  cos {loss_cos.item():.4e}")

net.eval()

# Accuracy check
with torch.no_grad():
    rmse_sin = float(torch.sqrt(((net(x_tensor, stds=stdvecs_sin) - sin_tensor) ** 2).mean()))
    rmse_cos = float(torch.sqrt(((net(x_tensor, stds=stdvecs_cos) - cos_tensor) ** 2).mean()))
print(f"\nPost-training RMSE  sin: {rmse_sin:.5f}   cos: {rmse_cos:.5f}\n")

# --- Forward hooks: capture hidden-layer CrossingSample outputs ---
hidden_acts: dict[int, np.ndarray] = {}
for _i in range(n_hidden):
    hidden_acts[_i] = np.zeros((N_INPUTS, net.t, H))


def make_hook(idx: int):
    def hook(module, input, output):
        hidden_acts[idx] = output.detach().cpu().numpy()   # [N, T, H]
    return hook


for _i, _gc in enumerate(net.gaussian_crossing):
    _gc.register_forward_hook(make_hook(_i))

# --- Alpha sweep: hold at 0 → transition → hold at 1 → transition → …, cyclic ---
alphas = np.concatenate([
    np.zeros(N_HOLD),                    # hold at sin mode  (α=0)
    np.linspace(0.0, 1.0, N_HALF),       # transition sin → cos
    np.ones(N_HOLD),                     # hold at cos mode  (α=1)
    np.linspace(1.0, 0.0, N_HALF),       # transition cos → sin
])
n_frames = len(alphas)

# --- Figure ---
height_ratios = [3, 0.5] + [1.2] * n_hidden
fig, axes = plt.subplots(
    2 + n_hidden, 1,
    figsize=(14, 5 + 2.5 * n_hidden),
    gridspec_kw={'height_ratios': height_ratios},
)
ax_pred  = axes[0]
ax_noise = axes[1]
ax_rasters = list(axes[2:])

# --- Prediction subplot ---
ax_pred.plot(x_np.ravel(), sin_np.ravel(),
             color='steelblue', lw=1.5, ls='--', alpha=0.55, label='sin(x)  [sin region]')
ax_pred.plot(x_np.ravel(), cos_np.ravel(),
             color='darkorange', lw=1.5, ls=':',  alpha=0.55, label='cos(x)  [cos region]')
pred_line, = ax_pred.plot(x_np.ravel(), np.zeros(N_INPUTS),
                          color='black', lw=2.0, label='network output')
ax_pred.set_xlim(float(x_np[0]), float(x_np[-1]))
ax_pred.set_ylim(-1.6, 1.6)
ax_pred.set_xlabel('x')
ax_pred.set_ylabel('y')
ax_pred.legend(loc='upper right', fontsize=9)
ax_pred.grid(alpha=0.35)
pred_title = ax_pred.set_title('')

# --- Noise field subplot ---
noise_im = ax_noise.imshow(
    stdvec_sin.numpy().reshape(1, -1),
    aspect='auto',
    cmap='Reds',
    vmin=0, vmax=ON_STD,
    extent=[-0.5, H - 0.5, 0, 1],
    interpolation='nearest',
)
ax_noise.axvline(HALF - 0.5, color='gray', lw=1.0, ls=':')
ax_noise.set_xlim(-0.5, H - 0.5)
ax_noise.set_yticks([])
ax_noise.set_xlabel('Neuron index')
ax_noise.set_title('Noise field  (Gaussian std per neuron)')

# Static region labels (use blended transform: x=data, y=axes fraction)
_trans = ax_noise.get_xaxis_transform()
ax_noise.text(HALF // 2 - 0.5,        0.5, 'sin region',
              ha='center', va='center', fontsize=8, color='dimgray',
              transform=_trans)
ax_noise.text(HALF + HALF // 2 - 0.5, 0.5, 'cos region',
              ha='center', va='center', fontsize=8, color='dimgray',
              transform=_trans)

# --- Firing-rate map subplots ---
raster_ims = []
for i, ax_r in enumerate(ax_rasters):
    im = ax_r.imshow(
        np.zeros((H, N_INPUTS)),
        aspect='auto',
        cmap='binary',      # white = 0 (silent), black = 1 (always firing)
        vmin=0, vmax=1,
        origin='lower',
        extent=[float(x_np[0]), float(x_np[-1]), 0.5, H + 0.5],
        interpolation='nearest',
    )
    ax_r.axhline(HALF + 0.5, color='steelblue', lw=0.8, ls=':')  # region boundary
    ax_r.set_xlabel('x  (input value)')
    ax_r.set_ylabel('Neuron')
    ax_r.set_title(
        f'Hidden layer {i + 1}  ({H} neurons)  —  firing rate (mean over T={net.t} samples)'
    )
    tick_step = max(1, H // 5)
    ax_r.set_yticks(np.arange(1, H + 1, tick_step))
    # Annotate region boundary on y-axis
    ax_r.text(float(x_np[-1]), HALF + 0.5, ' ← sin | cos →',
              va='center', fontsize=7, color='steelblue')
    raster_ims.append(im)

fig.suptitle(
    f'Sub-network switching via noise-field interpolation  '
    f'(SimpleNNNSample, structure={STRUCTURE})',
    fontsize=10,
)
plt.tight_layout()

# --- Animation update ---
def update(frame: int):
    alpha  = float(alphas[frame])
    stdvec = noise.interpolate_stdvecs(stdvec_sin, stdvec_cos, rate=alpha)
    stds   = [stdvec] * n_hidden

    # Forward pass — hooks fill hidden_acts
    with torch.no_grad():
        vis_out = net(x_tensor, stds=stds)

    # Prediction line
    pred_line.set_ydata(vis_out.numpy().ravel())

    # Title
    if alpha < 0.02:
        mode = 'sin mode  (sin region fully active)'
    elif alpha > 0.98:
        mode = 'cos mode  (cos region fully active)'
    else:
        mode = f'transition  α = {alpha:.2f}  ({(1-alpha)*100:.0f}% sin  /  {alpha*100:.0f}% cos)'
    pred_title.set_text(mode)

    # Noise field
    noise_im.set_data(stdvec.numpy().reshape(1, -1))

    # Firing-rate maps: mean over T dimension
    for i, im in enumerate(raster_ims):
        fire_rate = hidden_acts[i].mean(axis=1)   # [N, T, H] → [N, H]
        im.set_data(fire_rate.T)                  # → [H, N]

    return [pred_line, pred_title, noise_im] + raster_ims


ani = animation.FuncAnimation(
    fig, update,
    frames=n_frames,
    interval=INTERVAL,
    blit=False,
    repeat=True,
)

# ani.save('animation_two_functions_firerate.mp4', writer='ffmpeg')
plt.show()
