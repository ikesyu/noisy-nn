import torch
from multidimensional_storage import *
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from multidimensional_storage_train import get_filename
from multidimensional_storage_generate_job import get_job_string

from matplotlib.ticker import MaxNLocator
from pathlib import Path


def set_colors(N):
    # hue: blue (2/3) -> red (0)
    h = np.linspace(2/3, 0, N)
    colors = hsv_to_rgb(np.column_stack([h, np.ones(N), np.ones(N)]))
    plt.gca().set_prop_cycle(color=colors)


def plot_noise(structure, shape):
    #
    noise_patterns = noise_pattern_table(structure, shape)

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
                      [0].detach().cpu().reshape(structure.noise_structure[-2:]))
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


def plot_ys(structure, functions):
    indices = select_indices(functions.shape)
    set_colors(len(indices))
    for idx in indices:
        plt.plot(functions.x, functions.ys[idx].detach(
        ).cpu().numpy(), label=f"{idx=}")
    plt.legend()


def plot_inference(structure, functions, m):
    indices = select_indices(functions.shape)
    set_colors(len(indices))
    for idx in indices:
        stdvecs = noise_pattern(
            structure, fractional_index(idx, functions.shape))
        output = m(functions.x_tensor, stdvecs).detach().cpu().numpy()
        plt.plot(functions.x, output, ".--", label=f"{idx=}")
    plt.legend()


def test_shape(structure, functions, shuffle=False):
    shape = tuple(functions.shape)
    plot_noise(structure, shape)

    # plt.legend()
    # print(f"testing {shape=} {ys.shape=}")
    plt.figure()
    plot_ys(structure, functions)

    print("training begin")
    m, l = train_net(structure, functions)
    print("training end")
    plt.figure()
    plot_inference(structure, functions, m)
    plt.savefig("../fig/test_shape.pdf")
    # plt.show()


def test():
    set_cuda(True)
    structure = Structure([100])
    functions = SineFunctions([10], epochs=10000)
    test_shape(structure, functions)


def retrieve(structure, functions):
    fname = get_filename(structure, functions)
    if Path(fname).is_file():
        return torch.load(fname, weights_only=False)
    else:
        print(get_job_string(structure, functions, cuda=True))
        return {"model": None, "losses": [np.nan]}


def unidim_comparison():
    structure = Structure([100])
    losses = {sl: {s: [] for s in [False, True]} for sl in [False, True]}
    nfuncs = range(1, 101)
    for shuffle_learning in [True, False]:
        for shuffle in [True, False]:
            for i in nfuncs:
                function = SineFunctions(
                    [i], epochs=20000//i, shuffle=shuffle, shuffle_learning=shuffle_learning)
                loss = retrieve(structure, function)["losses"]
                losses[shuffle_learning][shuffle].append(np.max(loss))

    for shuffle_learning in [True, False]:
        for shuffle in [True, False]:
            place = "shuffled" if shuffle else "ordered"
            learn = "random" if shuffle_learning else "ordered"
            plt.plot(nfuncs, losses[shuffle_learning][shuffle],
                     label=f"{place} location, {learn} learning")
    plt.xlabel("Number of functions")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig("../fig/unidim_comparison.pdf")
    plt.close()
    # plt.show()


def bidim_comparison():
    structure = Structure([100, 100])
    losses = {True: [], False: []}
    nfuncs = range(1, 11)
    for shuffle in [True, False]:
        for i in nfuncs:
            function = SineFunctions([i, i], epochs=20000//i, shuffle=shuffle)
            loss = retrieve(structure, function)["losses"]
            losses[shuffle].append(np.max(loss))
    plt.plot(nfuncs, losses[False], label="odered")
    plt.plot(nfuncs, losses[True], label="shuffled")
    plt.xlabel("Number of functions")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig("../fig/bidim_comparison.pdf")
    plt.close()


def tridim_comparison():
    structure = Structure([20, 20, 20])
    losses = {True: [], False: []}
    nfuncs = range(1, 11)
    for shuffle in [True, False]:
        for i in nfuncs:
            function = SineFunctions(
                [i, i, i], epochs=20000//i, shuffle=shuffle)
            loss = retrieve(structure, function)["losses"]
            losses[shuffle].append(np.max(loss))
    plt.plot(nfuncs, losses[False], label="odered")
    plt.plot(nfuncs, losses[True], label="shuffled")
    plt.xlabel("Number of functions")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig("../fig/tridim_comparison.pdf")
    plt.close()


unidim_comparison()
bidim_comparison()
tridim_comparison()
