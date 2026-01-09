import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import gc


import sys  # noqa
sys.path.append("../")  # noqa


from nnn import model  # noqa
from nnn import layer  # noqa
from nnn import activation  # noqa


def set_cuda(active):
    if active:
        device = torch.device("cuda")

    else:
        device = torch.device("cpu")
    torch.set_default_device(device)

# データセットの作成


def all_live_tensors():
    tensors = []
    for obj in gc.get_objects():
        try:
            if isinstance(obj, torch.Tensor):
                tensors.append(obj)
        except Exception:
            pass  # Some proxies can raise on isinstance
    return tensors


def list_tensors():
    tensors = all_live_tensors()
    print(f"Found {len(tensors)} tensors")
    for t in tensors[:50]:  # limit output
        try:
            size_bytes = t.numel() * t.element_size()
            print(f"device={t.device}, shape={tuple(t.shape)}, dtype={t.dtype}, "
                  f"requires_grad={t.requires_grad}, size={size_bytes/1e6:.3f} MB")
        except Exception as e:
            print(f"tensor print error: {e}")

    # Aggregate by device
    from collections import Counter
    counts = Counter(t.device for t in tensors)
    print("Counts by device:", counts)
    bytes_by_device = {}
    for t in tensors:
        try:
            bytes_by_device[t.device] = bytes_by_device.get(
                t.device, 0) + t.numel() * t.element_size()
        except Exception:
            pass
    print("Approx memory by device (tensors only):",
          {d: f"{b/1e6:.2f} MB" for d, b in bytes_by_device.items()})


def all_live_modules():
    mods = []
    for obj in gc.get_objects():
        try:
            if isinstance(obj, nn.Module):
                mods.append(obj)
        except Exception:
            pass
    return mods


def list_modules():
    for m in all_live_modules():
        print(f"Module: {type(m).__name__}")
        for name, p in m.named_parameters(recurse=False):
            print(f"  param {name}: device={p.device}, shape={tuple(p.shape)}")
        for name, b in m.named_buffers(recurse=False):
            print(f"  buffer {name}: device={
                  b.device}, shape={tuple(b.shape)}")


class Structure:

    def __init__(self, noise_structure, activation_ratio=0.5):
        self.noise_structure = noise_structure
        self.activation_ratio = activation_ratio
        self.real_structure = [1] + [np.prod(noise_structure)]*2+[1]

        # [8*8*8]  # [8, 8, 8]


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
        shape = [1] * D
        shape[d] = n
        out.mul_(g1.view(shape))
    if normalize:
        s = out.sum()
        if s > 0:
            out.div_(s)
    return out


def compute_sigma(a, p, d):
    """
    Compute sigma for a d-dimensional Gaussian with peak=1,
    such that the volume of the ball where amplitude > a is p.

    Parameters:
    d (int): Number of dimensions
    a (float): Amplitude threshold (0 < a < 1)
    p (float): Desired volume

    Returns:
    float: The value of sigma
    """
    if not (0 < a < 1):
        raise ValueError("a must be between 0 and 1")
    if p <= 0:
        raise ValueError("p must be positive")
    if d <= 0:
        raise ValueError("d must be positive")

    # Volume of unit ball in d dimensions
    C_d = math.pi**(d/2) / math.gamma(d/2 + 1)

    # Radius r such that volume of ball is p
    r = (p / C_d)**(1/d)

    # From r^2 = -2 * sigma^2 * ln(a)
    sigma_squared = - (r**2) / (2 * math.log(a))

    return math.sqrt(sigma_squared)


def noise_pattern(structure, center):
    cut_at = 0.1
    sigma = compute_sigma(
        cut_at, structure.activation_ratio, len(structure.noise_structure))
    # print(f"d={len(noise_structure)} sigma={sigma}")
    scaled_center = structure.activation_ratio / \
        2+(1-structure.activation_ratio)*np.array(center)
    stdvecs = gaussian_fill(shape=structure.noise_structure,
                            center=scaled_center,
                            sigma=[sigma for n in structure.noise_structure],
                            amplitude=1,
                            normalize=False
                            ).reshape(-1)
    stdvecs = torch.where(stdvecs >= cut_at, stdvecs, 0)
    return [stdvecs]*2


def noise_pattern_table(structure, shape):
    noise_patterns = np.empty(shape, dtype=object)
    # print(f"noise pattern_table {shape=}")
    for idx in np.ndindex(shape):
        noise_patterns[idx] = noise_pattern(
            structure, fractional_index(idx, shape))
    return noise_patterns


def fractional_index(idx, shape):
    fidx = tuple(i / (n - 1) if n > 1 else 0.0 for i, n in zip(idx, shape))
    return fidx


def train_net(structure, functions):
    # print(f"Net stucture {structure}")
    model_analytic = model.SimpleNNNAnalytic(structure.real_structure)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model_analytic.parameters(),
                           lr=1E-4, weight_decay=0.0)

    yshape = functions.ys.shape[:-2]
    noise_patterns = noise_pattern_table(structure, yshape)
    epochs = functions.epochs
    # print(f"{yshape=}")
    # epochs = int(np.ceil(20000/np.prod(yshape)))
    # epochs = 10000
    # epochs = 2
    print(f"using {epochs=} device={noise_patterns.reshape(-1)[0][0].device}")
    print("intra-op threads:", torch.get_num_threads())
    print("inter-op threads:", torch.get_num_interop_threads())
    indices = list(np.ndindex(yshape))
    # list_tensors()
    # list_modules()
    for epoch in range(epochs):
        for idx in indices:
            optimizer.zero_grad()
            output = model_analytic(functions.x_tensor, noise_patterns[idx])
            loss = criterion(output, functions.ys[idx])
            loss.backward()
            optimizer.step()

    losses = []
    with torch.no_grad():
        for idx in indices:
            output = model_analytic(functions.x_tensor, noise_patterns[idx])
            loss = criterion(output, functions.ys[idx])
            losses.append(float(loss.item()))

    return model_analytic, losses


def sine_pfa(x, n_phase=1, n_freq=1, n_amp=1, shuffle=False):
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


class SineFunctions:
    def __init__(self, shape, shuffle=False, epochs=1000):
        self.shuffle = shuffle
        self.shape = tuple(shape)
        self.x = np.linspace(-2 * np.pi, 2 * np.pi, 200).reshape(-1, 1)
        self.x_tensor = torch.tensor(self.x, dtype=torch.float32)
        self.ys = sine_pfa(self.x, *shape, shuffle=shuffle)
        self.epochs = epochs
    # threads_per_run = max(1, os.cpu_count()-2)


# compare_active_count()

# plt.plot(noise_pattern([0, 0, 0])[0].detach().cpu().numpy())
# plt.figure()
# plt.show()
# test_shape([10, 10])
# test_shape([2, 2, 3], shuffle=True)

# shapes = [[i, i, i] for i in range(1, 6, 1)]
# compute_storage_comparison(shapes, "tridim")
# plot_storage_comparisons("tridim")


# shapes = [[i] for i in range(1, 101, 1)]
# compute_storage_comparison(shapes, "unidim")
# plot_storage_comparisons("unidim")


# test_shape([3, 3], shuffle=True)
