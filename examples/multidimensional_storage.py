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

noise_structure = [100]

# [8*8*8]  # [8, 8, 8]
structure = [1] + [np.prod(noise_structure)]*2+[1]


def gaussian_fill(shape, center, sigma, amplitude=1.0, normalize=False):
    # print(f"{shape=} {center=}")
    """
    shape: [n0, n1, ..., nD]) with an N-D Gaussian:
      G(x) = amplitude * exp(-0.5 * sum_d ((x_d - center[d]) / sigma[d])**2)

    - center: list/tuple/1D tensor of length out.ndim in [0,1]
    - sigma:  scalar or sequence of length len(shape)
    - normalize=True will rescale so out.sum()==1 (discrete normalization)
    """
    D = len(shape)
    out = torch.empty(*shape, dtype=torch.float32)
    assert len(center) == D, "center must have one entry per dimension"

    device, dtype = out.device, out.dtype

    center = torch.as_tensor(center, dtype=dtype, device=device)
    sigma = torch.as_tensor(sigma, dtype=dtype, device=device)
    if sigma.numel() == 1:
        sigma = sigma.expand(D)

    out.fill_(amplitude)
    # Use separability: product of 1D Gaussians equals N-D Gaussian with diagonal covariance
    for d, n in enumerate(out.shape):
        coords = torch.linspace(0, 1, n, device=device, dtype=dtype)
        g1 = torch.exp(-0.5 *
                       ((coords - center[d]) / sigma[d]) ** 2)
        # g1 = torch.where(g1 >= 0.1, g1, 0)
        shape = [1] * D
        shape[d] = n
        out.mul_(g1.view(shape))
    if normalize:
        s = out.sum()
        if s > 0:
            out.div_(s)
    return out


def noise_pattern(center):
    stdvecs = gaussian_fill(shape=noise_structure,
                            center=center,
                            sigma=[0.1 for n in noise_structure],
                            amplitude=1,
                            normalize=False
                            ).reshape(-1)
    stdvecs = torch.where(stdvecs >= 0.1, stdvecs, 0)
    return [stdvecs]*2


def noise_pattern_table(shape):
    noise_patterns = np.empty(shape, dtype=object)
    # print(f"noise pattern_table {shape=}")
    for idx in np.ndindex(shape):
        noise_patterns[idx] = noise_pattern(fractional_index(idx, shape))
    return noise_patterns


def set_colors(N):
    # hue: blue (2/3) -> red (0)
    h = np.linspace(2/3, 0, N)
    colors = hsv_to_rgb(np.column_stack([h, np.ones(N), np.ones(N)]))
    plt.gca().set_prop_cycle(color=colors)


def fractional_index(idx, shape):
    fidx = tuple(i / (n - 1) if n > 1 else 0.0 for i, n in zip(idx, shape))
    return fidx


def train_net(ys):
    # print(f"Net stucture {structure}")
    model_analytic = model.SimpleNNNAnalytic(structure)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model_analytic.parameters(),
                           lr=1E-4, weight_decay=0.0)

    yshape = ys.shape[:-2]
    noise_patterns = noise_pattern_table(yshape)
    # print(f"{yshape=}")
    epochs = int(np.ceil(20000/np.prod(yshape)))
    # epochs = 10000
    # epochs = 2
    print(f"using {epochs=}")
    indices = list(np.ndindex(yshape))
    for epoch in range(epochs):
        optimizer.zero_grad()
        for idx in indices:
            output = model_analytic(x_tensor, noise_patterns[idx])
            loss = criterion(output, ys[idx])
            loss.backward()
        optimizer.step()
        # if epoch == 0:
        #    plt.show()

    losses = []
    with torch.no_grad():
        for idx in indices:
            output = model_analytic(x_tensor, noise_patterns[idx])
            loss = criterion(output, ys[idx])
            losses.append(float(loss.item()))

    return model_analytic, losses


def sine_pfa(n_phase=1, n_freq=1, n_amp=1, shuffle=False):
    phase = np.linspace(0, 2*np.pi, n_phase, endpoint=False)
    freq = np.linspace(1, 2, n_freq)
    amp = np.linspace(1, 2, n_amp)
    # print(f"{phase=} {freq=} {amp=}")

    vals = amp[None, None, :, None, None] * np.sin(
        freq[None, :, None, None, None] * x[None, None, None, :, :] + phase[:, None, None, None, None])

    assert np.min(vals.shape) > 0, "did you pass shuffle by value?"

    tosqueeze = [dim for dim in range(1, 3) if vals.shape[dim] == 1]
    # print(f"before squeeze {vals.shape}")
    if len(tosqueeze) > 0:
        vals = np.squeeze(vals, axis=tuple(tosqueeze))
        # print(f"after squeeze {vals.shape}")
    if shuffle:
        shape = tuple(vals.shape)
        vals = vals.reshape(-1, shape[-2], shape[-1])
        p = np.random.permutation(vals.shape[0])
        vals = vals[p, :, :]
        vals = vals.reshape(shape)

    return torch.tensor(vals, dtype=torch.float32)


def plot_noise(shape):
    #
    noise_patterns = noise_pattern_table(shape)

    dims = np.sum([1 if n > 1 else 0 for n in shape])
    if dims == 1:
        set_colors(np.prod(shape))
        for idx in np.ndindex(shape):
            plt.plot(noise_patterns[idx][0].detach().cpu(), label=f"{idx=}")
    if dims == 2:
        R, C = [d for d in shape if d > 1]
        fig, axes = plt.subplots(R, C, figsize=(3*C, 3*R))
        for idx in np.ndindex(shape):
            ax = axes[idx[-2:]]
            ax.imshow(noise_patterns[idx]
                      [0].detach().cpu().reshape(noise_structure[-2:]))
            ax.set_title(f"idx={idx}")
            ax.axis("off")

        plt.tight_layout()


def select_indices(shape):
    indices = list(np.ndindex(shape))
    n = len(indices)
    if n > 27:
        step = int(np.ceil(n/27))
        indices = indices[::step]
    return indices


def plot_ys(ys):
    shape = ys.shape[:-2]
    # print(f"{ys.shape=}{shape=}")
    indices = select_indices(shape)
    set_colors(len(indices))
    for idx in indices:
        plt.plot(x, ys[idx].detach().cpu().numpy(), label=f"{idx=}")
    plt.legend()


def plot_inference(shape, m):
    indices = select_indices(shape)
    set_colors(len(indices))
    for idx in indices:
        stdvecs = noise_pattern(fractional_index(idx, shape))
        output = m(x_tensor, stdvecs).detach().cpu().numpy()
        plt.plot(x, output, ".--", label=f"{idx=}")
    plt.legend()


def test_shape(shape, shuffle=False):
    shape = tuple(shape)
    plot_noise(shape)

    # plt.legend()
    ys = sine_pfa(*shape, shuffle=shuffle)
    # print(f"testing {shape=} {ys.shape=}")
    plt.figure()
    plot_ys(ys)

    print("training begin")
    m, l = train_net(ys)
    print("training end")
    plt.figure()
    plot_inference(shape, m)
    plt.show()


def worker(shape, shuffle, threads_per_run):
    # Force CPU-only and cap per-process threads to avoid oversubscription
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("OMP_NUM_THREADS", str(threads_per_run))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads_per_run))
    torch.set_num_threads(threads_per_run)
    ys = sine_pfa(*shape, shuffle=shuffle)
    model, losses = train_net(ys)
    return losses


def run_batch(shapes, shuffle):
    cpus = os.cpu_count()-1
    n_jobs = min(cpus, len(shapes))
    threads_per_run = cpus//n_jobs

    print(f"{os.cpu_count()=} {n_jobs=} {threads_per_run=}")
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        # ex.map preserves order; repeat() passes constants to all calls
        results = list(
            ex.map(worker, shapes, repeat(shuffle), repeat(threads_per_run)))
    res = {tuple(s): r for s, r in zip(shapes, results)}
    return res

    # threads_per_run = max(1, os.cpu_count()-2)


def compute_storage_comparison(shapes, label=""):
    results = {"ordered": run_batch(shapes, shuffle=False),
               "shuffled": run_batch(shapes, shuffle=True)
               }
    torch.save(results, f"multidimensional_storage_{label}.pt")


def plot_storage_comparisons(label=""):
    results = torch.load(f"multidimensional_storage_{label}.pt",
                         map_location="cpu", weights_only=False)

    def get_props(res):
        x = []
        y = []
        for shape, losses in res.items():
            x.append(shape[0])
            y.append(np.max(losses))
        return x, y

    for name, res in results.items():
        x, y = get_props(res)
        plt.plot(x, y, ".-", label=name)

    plt.xlabel("Functions per dimension")
    plt.ylabel("Loss")
    plt.grid(True)

    plt.legend()
    plt.show()


def active_count(dims, sigma):
    size = int(np.ceil(np.pow(256, 1.0/dims)))
    # size = 256
    shape = [size]*dims
    # print(f"{shape=}")
    center = [0.5]*dims
    g = gaussian_fill(shape, center, sigma, amplitude=1.0, normalize=False)
    count = torch.count_nonzero(g > 1E-3).item()
    return count/np.prod(shape)


def compare_active_count():
    sigmas = np.linspace(0, 1, 100)
    for d in range(1, 4):
        a = [active_count(d, sigma) for sigma in sigmas]
        plt.plot(sigmas, a, ".-", label=f"{d=}")
    plt.legend()
    plt.grid(True)

    plt.figure()
    ds = range(1, 3)
    a = [active_count(d, np.sqrt(d)*0.1) for d in ds]
    plt.plot(ds, a)
    plt.show()


compare_active_count()

# plt.plot(noise_pattern([0, 0, 0])[0].detach().cpu().numpy())
# plt.figure()
# plt.show()
# test_shape([2, 2, 3])
# test_shape([2, 2, 3], shuffle=True)

# shapes = [[i, i, i] for i in range(1, 6, 1)]

# shapes = [[i] for i in range(1, 101, 1)]
# compute_storage_comparison(shapes, "unidim")
# plot_storage_comparisons("unidim")


# test_shape([3, 3], shuffle=True)
