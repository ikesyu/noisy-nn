"""Train sin / cos sub-networks in one NNN and save the transition as a GIF.

Two sub-networks share the same linear weights but are separated by the
noise field (per-neuron Gaussian noise std):

    sin sub-network : neurons 0 .. H/2-1 receive noise (std = ON_STD)
    cos sub-network : neurons H/2 .. H-1 receive noise (std = ON_STD)

Training alternates MSE(net(x, stds_sin), sin(x)) and
MSE(net(x, stds_cos), cos(x)) per epoch (same recipe as
regression_two_functions.py / animation_2funcs_firerate_sample.py).

After training, the noise field is swept with
    noise.interpolate_stdvecs(stdvec_sin, stdvec_cos, rate=alpha)
for a cyclic alpha in [0, 1], and every frame shows:

    Top      : regression curve (network output vs. sin / cos targets)
    Middle   : noise field — Gaussian std per neuron
    Lower ×2 : raster plot per hidden layer — sample-level crossing
               spikes (one stochastic draw; values 0 / 0.5 / 1) over the
               input sweep

Model: SimpleNNNSample. Training uses t=T_TRAIN samples; the animation
uses a separate copy with t=T_VIS for a smoother prediction curve.

Run from the project root:

    python examples/animation_sincos_transition.py

Output: examples/out/animation_sincos_transition.gif
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib
matplotlib.use("Agg")
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
T_TRAIN   = 10               # samples per input during training
T_VIS     = 64               # samples per input during visualization
SEED      = 0

# Animation
N_HALF    = 45               # frames per half-sweep (sin→cos or cos→sin)
N_HOLD    = 15               # frames to hold at each pure mode
FPS       = 15               # GIF frame rate

OUT_DIR  = Path(__file__).resolve().parent / "out"
OUT_PATH = OUT_DIR / "animation_sincos_transition.gif"

# ============================================================

torch.manual_seed(SEED)
np.random.seed(SEED)

# --- Dataset ---
x_np   = np.linspace(-2 * np.pi, 2 * np.pi, N_INPUTS).reshape(-1, 1)
X_MIN, X_MAX = float(x_np[0, 0]), float(x_np[-1, 0])
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
stdvecs_sin = [stdvec_sin] * n_hidden
stdvecs_cos = [stdvec_cos] * n_hidden

# --- Training ---
net       = model.SimpleNNNSample(structure=STRUCTURE, t=T_TRAIN)
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=LR)

print(f"Training  (structure={STRUCTURE}, ON_STD={ON_STD}, epochs={EPOCHS}) …")
for epoch in range(1, EPOCHS + 1):
    net.train()
    optimizer.zero_grad()
    loss_sin = criterion(net(x_tensor, stds=stdvecs_sin), sin_tensor)
    loss_sin.backward()
    optimizer.step()

    optimizer.zero_grad()
    loss_cos = criterion(net(x_tensor, stds=stdvecs_cos), cos_tensor)
    loss_cos.backward()
    optimizer.step()
    if epoch == 1 or epoch % 300 == 0:
        print(f"  epoch {epoch:5d} | sin {loss_sin.item():.4e}  cos {loss_cos.item():.4e}")

# --- Visualization model: same weights, more samples per input ---
vis_net = model.SimpleNNNSample(structure=STRUCTURE, t=T_VIS)
for fc_vis, fc in zip(vis_net.fcs, net.fcs):
    fc_vis.load_state_dict(fc.state_dict())
vis_net.eval()

with torch.no_grad():
    rmse_sin = float(torch.sqrt(((vis_net(x_tensor, stds=stdvecs_sin) - sin_tensor) ** 2).mean()))
    rmse_cos = float(torch.sqrt(((vis_net(x_tensor, stds=stdvecs_cos) - cos_tensor) ** 2).mean()))
print(f"\nPost-training RMSE  sin: {rmse_sin:.5f}   cos: {rmse_cos:.5f}\n")

# --- Forward hooks: capture hidden-layer crossing samples [N, T, H] ---
hidden_acts: dict[int, np.ndarray] = {
    i: np.zeros((N_INPUTS, T_VIS, H)) for i in range(n_hidden)
}


def make_hook(idx: int):
    def hook(module, inputs, output):
        hidden_acts[idx] = output.detach().cpu().numpy()
    return hook


for _i, _gc in enumerate(vis_net.gaussian_crossing):
    _gc.register_forward_hook(make_hook(_i))

# --- Alpha sweep: hold → sin→cos → hold → cos→sin, cyclic ---
alphas = np.concatenate([
    np.zeros(N_HOLD),
    np.linspace(0.0, 1.0, N_HALF),
    np.ones(N_HOLD),
    np.linspace(1.0, 0.0, N_HALF),
])
n_frames = len(alphas)

# --- Figure ---
plt.rcParams.update({
    "font.size":        15,
    "axes.titlesize":   17,
    "axes.labelsize":   16,
    "xtick.labelsize":  14,
    "ytick.labelsize":  14,
    "legend.fontsize":  13,
})

height_ratios = [3, 0.5] + [1.6] * n_hidden
fig, axes = plt.subplots(
    2 + n_hidden, 1,
    figsize=(12, 4.5 + 2.4 * n_hidden),
    gridspec_kw={"height_ratios": height_ratios},
)
ax_pred  = axes[0]
ax_noise = axes[1]
ax_rasters = list(axes[2:])

# --- Regression curve subplot ---
ax_pred.plot(x_np.ravel(), sin_np.ravel(),
             color="steelblue", lw=1.5, ls="--", alpha=0.55, label="sin(x)  [sin region]")
ax_pred.plot(x_np.ravel(), cos_np.ravel(),
             color="darkorange", lw=1.5, ls=":", alpha=0.55, label="cos(x)  [cos region]")
pred_line, = ax_pred.plot(x_np.ravel(), np.zeros(N_INPUTS),
                          color="black", lw=2.0, label="network output")
ax_pred.set_xlim(X_MIN, X_MAX)
ax_pred.set_ylim(-1.6, 1.6)
ax_pred.set_xticks([-6, -3, 0, 3, 6])
ax_pred.set_yticks([-1, 0, 1])
ax_pred.set_xlabel("x")
ax_pred.set_ylabel("y")
ax_pred.legend(loc="upper right")
ax_pred.grid(alpha=0.35)
# Non-empty placeholder so tight_layout reserves room for the animated title
pred_title = ax_pred.set_title("sin mode  (sin region fully active)")

# --- Noise field subplot ---
noise_im = ax_noise.imshow(
    stdvec_sin.numpy().reshape(1, -1),
    aspect="auto",
    cmap="Reds",
    vmin=0, vmax=ON_STD,
    extent=[-0.5, H - 0.5, 0, 1],
    interpolation="nearest",
)
ax_noise.axvline(HALF - 0.5, color="gray", lw=1.0, ls=":")
ax_noise.set_xlim(-0.5, H - 0.5)
ax_noise.set_yticks([])
ax_noise.set_xticks([0, HALF - 1, H - 1])
ax_noise.set_xticklabels([1, HALF, H])
ax_noise.set_xlabel("Neuron index")
ax_noise.set_title("Noise field  (Gaussian std per neuron)")

_trans = ax_noise.get_xaxis_transform()
ax_noise.text(HALF // 2 - 0.5,        0.5, "sin region",
              ha="center", va="center", fontsize=13, color="dimgray", transform=_trans)
ax_noise.text(HALF + HALF // 2 - 0.5, 0.5, "cos region",
              ha="center", va="center", fontsize=13, color="dimgray", transform=_trans)

# --- Raster subplots: sample-level crossing spikes over the input sweep ---
raster_ims = []
for i, ax_r in enumerate(ax_rasters):
    im = ax_r.imshow(
        np.zeros((H, N_INPUTS)),
        aspect="auto",
        cmap="binary",      # white = silent, black = crossing spike
        vmin=0, vmax=1,
        origin="lower",
        extent=[X_MIN, X_MAX, 0.5, H + 0.5],
        interpolation="nearest",
    )
    ax_r.axhline(HALF + 0.5, color="steelblue", lw=0.8, ls=":")
    ax_r.set_xlabel("x  (input value)")
    ax_r.set_ylabel("Neuron")
    ax_r.set_title(f"Hidden layer {i + 1}  raster")
    ax_r.set_xticks([-6, -3, 0, 3, 6])
    ax_r.set_yticks([1, HALF, H])
    raster_ims.append(im)

fig.tight_layout()


# --- Animation update ---
def update(frame: int):
    alpha  = float(alphas[frame])
    stdvec = noise.interpolate_stdvecs(stdvec_sin, stdvec_cos, rate=alpha)
    stds   = [stdvec] * n_hidden

    with torch.no_grad():
        vis_out = vis_net(x_tensor, stds=stds)   # hooks fill hidden_acts

    pred_line.set_ydata(vis_out.numpy().ravel())

    if alpha < 0.02:
        mode = "sin mode  (sin region fully active)"
    elif alpha > 0.98:
        mode = "cos mode  (cos region fully active)"
    else:
        mode = f"transition  α = {alpha:.2f}  ({(1-alpha)*100:.0f}% sin  /  {alpha*100:.0f}% cos)"
    pred_title.set_text(mode)

    noise_im.set_data(stdvec.numpy().reshape(1, -1))

    for i, im in enumerate(raster_ims):
        spikes = hidden_acts[i][:, 0, :]   # one stochastic draw: [N, H]
        im.set_data(spikes.T)              # → [H, N]

    return [pred_line, pred_title, noise_im] + raster_ims


ani = animation.FuncAnimation(
    fig, update,
    frames=n_frames,
    blit=False,
    repeat=True,
)

OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Saving GIF ({n_frames} frames) to {OUT_PATH} …")
ani.save(OUT_PATH, writer=animation.PillowWriter(fps=FPS))
plt.close(fig)
print("Done.")
