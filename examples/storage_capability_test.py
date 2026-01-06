import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import os

import sys  # noqa
sys.path.append("../")  # noqa


from nnn import model  # noqa
from nnn import layer  # noqa
from nnn import activation  # noqa

# device = torch.device("cuda")
# torch.set_default_device(device)


# データセットの作成
x = np.linspace(-2 * np.pi, 2 * np.pi, 200).reshape(-1, 1)
x_tensor = torch.tensor(x, dtype=torch.float32)

structure = [1, 100, 100, 1]


def triangular(t):
    f = 1/(2*np.pi)
    y = 4*np.abs(((f*t) % 1) - 0.5) - 1
    return y


def square(t):
    return np.where(np.sin(t) > 0, 1, -1)


def noise_pattern(phase):
    stdvecs = [
        torch.tensor(
            np.sin(
                np.linspace(0, 2*np.pi, n)-phase
            ).clip(min=0), dtype=torch.float32
        )
        for n in structure[1:-1]]
    return stdvecs


def inference(m, phase):
    with torch.no_grad():
        y = m(x_tensor, noise_pattern(phase)
              ).detach().cpu().numpy()
    return y


def set_colors(N):
    h = np.linspace(2/3, 0, N)              # hue: blue (2/3) -> red (0)
    colors = hsv_to_rgb(np.column_stack([h, np.ones(N), np.ones(N)]))
    plt.gca().set_prop_cycle(color=colors)


def train_net(phases):
    model_analytic = model.SimpleNNNAnalytic(structure)

    # 損失関数と最適化アルゴリズム
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model_analytic.parameters(),
                           lr=1E-4, weight_decay=0.0)

    ys = [torch.tensor(np.cos(x+phase), dtype=torch.float32)
          for phase in phases]

    epochs = 20000
    n_phases = len(phases)

    # print("Training starts")
    for epoch in range(epochs):
        # sin(x)の学習
        optimizer.zero_grad()

        stdvecs = noise_pattern(epoch*2*np.pi/n_phases)
        y_sel = ys[epoch % n_phases]
        output = model_analytic(x_tensor, stdvecs)
        loss = criterion(output, y_sel)
        # print("\r"+f"loss: {loss}", end="")
        loss.backward()
        optimizer.step()
    # print(".")
    # print("Training ends")

    losses = []
    with torch.no_grad():
        for i, y in enumerate(ys):
            stdvecs = noise_pattern(i*2*np.pi/n_phases)
            output = model_analytic(x_tensor, stdvecs)
            loss = criterion(output, y)
            losses.append(float(loss.item()))

    return model_analytic, losses


def phase_set(n, shuffle=False):
    a = np.linspace(0, 2*np.pi, n, endpoint=False)
    if shuffle:
        np.random.shuffle(a)
    return a


def worker(n_phases, sh, threads_per_run):
    # Force CPU-only and cap per-process threads to avoid oversubscription
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("OMP_NUM_THREADS", str(threads_per_run))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads_per_run))
    torch.set_num_threads(threads_per_run)
    return train_net(phase_set(n_phases, sh))


def run_batch(n_phases_list, sh):
    print(f"{os.cpu_count()=}")
    n_jobs = (os.cpu_count()-2)//2
    threads_per_run = 2

    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        # ex.map preserves order; repeat() passes constants to all calls
        results = list(
            ex.map(worker, n_phases_list, repeat(sh), repeat(threads_per_run)))
    return results


# threads_per_run = max(1, os.cpu_count()-2)


def plot_inferences(m, n_phases):
    set_colors(n_phases)
    for ph in phase_set(n_phases):
        plt.plot(x, inference(m, ph))


def compare_storage():
    n_phases_list = list(range(1, 101, 3))
    losses = [run_batch(n_phases_list, sh) for sh in [False, True]]

    plt.plot(n_phases_list, [np.max(l) for m, l in losses[0]], label="ordered")
    plt.plot(n_phases_list, [np.max(l)
             for m, l in losses[1]], label="shuffled")
    plt.grid(True)
    plt.xlabel("Number of functions")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


compare_storage()
