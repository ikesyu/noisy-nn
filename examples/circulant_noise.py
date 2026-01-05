import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

import sys  # noqa
sys.path.append("../")  # noqa


from nnn import model  # noqa
from nnn import layer  # noqa
from nnn import activation  # noqa

device = torch.device("cuda")
torch.set_default_device(device)


# データセットの作成
x = np.linspace(-2 * np.pi, 2 * np.pi, 200).reshape(-1, 1)
x_tensor = torch.tensor(x, dtype=torch.float32)

structure = [1, 200, 200, 1]
model_analytic = model.SimpleNNNAnalytic(structure)


# 損失関数と最適化アルゴリズム
criterion = nn.MSELoss()
optimizer = optim.Adam(model_analytic.parameters(), lr=1E-4, weight_decay=0.0)


def triangular(t):
    f = 1/(2*np.pi)
    y = 4*np.abs(((f*t) % 1) - 0.5) - 1
    return y


def square(t):
    return np.where(np.sin(t) > 0, 1, -1)


funcs = {
    "sin": np.sin,
    "cos": np.cos,
    "triangular": triangular,
    "square": square
}
ys = {n: torch.tensor(f(x), dtype=torch.float32) for n, f in funcs.items()}


def noise_pattern(phase):
    stdvecs = [
        torch.tensor(
            np.sin(
                np.linspace(0, 2*np.pi, n)-phase
            ).clip(min=0), dtype=torch.float32
        )
        for n in structure[1:-1]]
    return stdvecs


def inference(phase):
    with torch.no_grad():
        y = model_analytic(x_tensor, noise_pattern(phase)
                           ).detach().cpu().numpy()
    return y


def set_colors(N):
    h = np.linspace(2/3, 0, N)              # hue: blue (2/3) -> red (0)
    colors = hsv_to_rgb(np.column_stack([h, np.ones(N), np.ones(N)]))
    plt.gca().set_prop_cycle(color=colors)


epochs = 20000
func_epochs = 4  # how many epochs the same function is presented to the learning
# period over which the noise pattern is shifted (in epochs)
noise_period = len(ys) * func_epochs

losses = []
print("Training starts")
for epoch in range(epochs):
    # sin(x)の学習
    optimizer.zero_grad()

    stdvecs = noise_pattern(epoch*2*np.pi/noise_period)
    y_sel = list(ys.values())[(epoch//func_epochs) % len(ys)]
    output = model_analytic(x_tensor, stdvecs)
    loss = criterion(output, y_sel)
    print("\r"+f"loss: {loss}", end="")

    # plt.figure()
    # plt.plot(x_tensor, y_sel, "r")
    # plt.title(f"{epoch=}")
    # plt.figure()
    # plt.title(f"{epoch=}")
    # for i in range(len(stdvecs)):
    #    plt.plot(stdvecs[i], label=f"{i=}")

    losses.append(loss.item())
    loss.backward()
    optimizer.step()
print(".")
print("Training ends")

plt.semilogy(losses)


plt.figure()
plt.title("Noise std patterns")
N_noise_pattern = func_epochs*len(funcs.keys())
set_colors(N_noise_pattern)
for i in range(N_noise_pattern):
    v = noise_pattern(i*2*np.pi/N_noise_pattern)
    plt.plot(v[0].cpu().numpy())
plt.xlabel("unit number")
plt.ylabel("noise std")


plt.figure()
plt.title("Functions")
set_colors(len(ys))
for yname, yval in ys.items():
    plt.plot(x, yval.detach().cpu(), label=yname)
plt.legend()
plt.grid(True)


noise_step = 360/noise_period
set_colors(func_epochs)
fig, axes = plt.subplots(len(funcs), 1, figsize=(
    8, 6), sharex=True, sharey=True)
for ifn, (fn, ax) in enumerate(zip(funcs.keys(), axes.flat)):
    for i in range(func_epochs):
        phase = (ifn*func_epochs+i)*noise_step
        y = inference(np.radians(phase))
        ax.plot(x, y, label=f"phase={float(phase)}")
    plt.xlabel("x")
    plt.ylabel("y")
    ax.grid(True)
    ax.legend()


plt.figure()
plt.title("Best matching phases")
set_colors(len(ys))
for phase in np.linspace(0, 360, len(ys), endpoint=False)+360/(2*len(ys)):
    y = model_analytic(x_tensor, noise_pattern(
        np.radians(phase))).detach().cpu().numpy()
    plt.plot(x, y, "--", label=f"phase={float(phase)}")

plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)


plt.show()
