import numpy as np
import matplotlib.pyplot as plt
import torch
from multidimensional_storage import compute_sigma, gaussian_fill


def active_count(dims, sigma):
    size = int(np.ceil(np.pow(10000000, 1.0/dims)))
    # size = 256
    shape = [size]*dims
    # print(f"{shape=}")
    center = [0.5]*dims
    g = gaussian_fill(shape, center, sigma, amplitude=1.0, normalize=False)
    count = torch.count_nonzero(g > 0.1).item()
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


def test_active_count():
    for i in range(1, 4):
        sigma = compute_sigma(0.1, 0.5, i)
        count = active_count(i, sigma)
        print(f"{i=} {count=}")


test_active_count()
compare_active_count()
