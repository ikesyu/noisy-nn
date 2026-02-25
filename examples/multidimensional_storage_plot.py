import torch
from multidimensional_storage import *
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from multidimensional_storage_train import get_filename
from multidimensional_storage_generate_job import get_job_string

from matplotlib.ticker import MaxNLocator
from pathlib import Path
from multidimensional_storage_functions import SineFunctions, SquineFunctions, TrineFunctions


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
    structure = Structure([2**2, 2**2, 2**2])
    functions = SineFunctions([4, 4, 4], epochs=10000)
    test_shape(structure, functions)


def retrieve(structure, functions):
    fname = get_filename(structure, functions)
    if Path(fname).is_file():
        # print("# Loaded ", fname)
        return torch.load(fname, weights_only=False)
    else:
        print(get_job_string(structure, functions, cuda=True))
        return {"model": None, "losses": [np.nan]}


experiments = {"first":
               [([100], 101), ([100, 100], 11), ([20, 20, 20], 11)],
               "two":
               [([2**6], 2**6+1),
                ([2**3, 2**3], 2**5+1),
                ([2**2, 2**2, 2**2], 2**4+1)],
               "three":
               [([3**6], 3**6+1),
                ([3**3, 3**3], 3**5+1),
                ([3**2, 3**2, 3**2], 3**4+1)],
               }


def multidim_comparison(expid="first", dim=1):
    exper = experiments[expid][dim-1]
    # print("experiment ", exper)
    structure = Structure(exper[0])
    losses = {sl: {s: [] for s in [False, True]} for sl in [False, True]}
    nfuncs = list(range(1, exper[1]))
    for shuffle_learning in [True, False]:
        for shuffle in [True, False]:
            for i in nfuncs:
                function = SineFunctions(
                    [i]*dim, epochs=20000//i, shuffle=shuffle, shuffle_learning=shuffle_learning)
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
    plt.savefig(f"../fig/dim{dim}_comparison_{expid}.pdf")
    plt.close()
    # plt.show()


def unidim_comparison(expid="first"):
    multidim_comparison(expid, 1)


def bidim_comparison(expid="first"):
    exper = experiments[expid][1]
    structure = Structure(exper[0])
    losses = {True: [], False: []}
    nfuncs = range(1, exper[1])
    for shuffle in [True, False]:
        for i in nfuncs:
            function = SineFunctions([i, i], epochs=20000//i, shuffle=shuffle)
            loss = retrieve(structure, function)["losses"]
            losses[shuffle].append(np.max(loss))
    plt.plot(nfuncs, losses[False], label="ordered")
    plt.plot(nfuncs, losses[True], label="shuffled")
    plt.xlabel("Number of functions")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"../fig/bidim_comparison_{expid}.pdf")
    plt.close()


def tridim_comparison(expid="first"):
    exper = experiments[expid][2]
    structure = Structure(exper[0])
    losses = {True: [], False: []}
    nfuncs = range(1, exper[1])
    for shuffle in [True, False]:
        for i in nfuncs:
            function = SineFunctions(
                [i, i, i], epochs=20000//i, shuffle=shuffle)
            loss = retrieve(structure, function)["losses"]
            losses[shuffle].append(np.max(loss))
    plt.plot(nfuncs, losses[False], label="ordered")
    plt.plot(nfuncs, losses[True], label="shuffled")
    plt.xlabel("Number of functions")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"../fig/tridim_comparison_{expid}.pdf")
    plt.close()


def closest_factor_pair(n):
    """
    Return (a, b) with a >= b > 1 and a*b = n that minimizes (a - b).
    If no such pair exists, return None.
    """
    if n < 4:
        return None
    import math
    r = math.isqrt(n)
    for b in range(r, 1, -1):
        if n % b == 0:
            a = n // b
            return (a, b)  # a >= b by construction
    return None


def closest_factor_triple(n):
    """
    Return (a, b, c) with a >= b >= c > 1 and abc = n that makes the factors
    as close as possible by minimizing (a-c, a-b, b-c). If none exists, return None.
    """
    if n < 8:
        return None

    import math

    # safe integer cube root (floor)
    c0 = int(round(n ** (1/3)))
    while c0 ** 3 > n:
        c0 -= 1
    while (c0 + 1) ** 3 <= n:
        c0 += 1

    best = None
    best_key = None

    for c in range(c0, 1, -1):
        if n % c != 0:
            continue
        m = n // c
        r = math.isqrt(m)  # floor(sqrt(m))

        # Need b in [c, r] so that a = m//b >= b
        if r < c:
            continue

        b = None
        for t in range(r, c - 1, -1):
            if m % t == 0:
                b = t
                break
        if b is None:
            continue

        a = m // b  # guaranteed integer
        # a >= b by construction (b <= sqrt(m)), and b >= c by loop bounds

        key = (a - c, a - b, b - c)
        if best is None or key < best_key:
            best = (a, b, c)
            best_key = key
            if key == (0, 0, 0):  # perfect cube case
                break

    return best


def reshape_dims(li, dims):
    if dims < len(li):
        rem = int(np.prod(li[dims-1:]))
        return tuple(li[:dims-1])+tuple([rem])
    elif dims == len(li):
        return li
    else:
        return tuple(li)+tuple([1]*(dims-len(li)))


def function_factors(n, dim):
    if dim == 1:
        return [n]
    if dim == 2:
        return closest_factor_pair(n)
    if dim == 3:
        return closest_factor_triple(n)


def function_shape(n, dimpatt):
    dim = np.sum(dimpatt)
    ns = function_factors(n, dim)
    if ns is None:
        return None, None
    cs = []
    i = 0
    for v in dimpatt:
        if v:
            cs.append(ns[i])
            i += 1
        else:
            cs.append(1)
    return ns, cs


def number_to_binary_list(n):
    return [(n >> i) & 1 for i in range(3)]


def multidim_fit(dimbin):
    dimpatt = number_to_binary_list(dimbin)
    for dim in range(1, 4):
        structure = Structure([2**(6//dim)]*dim)
        losses = {True: [], False: []}
        nfuncs = []
        for i in range(1, 28):

            ns, cs = function_shape(i, dimpatt)
            # print(f"# {dimbin=} {dimpatt=} {ns=} {cs=}")
            if ns is None:
                continue
            nfuncs.append(i)
            for shuffle in [True, False]:
                function = TrineFunctions(construct_shape=cs,
                                          present_shape=reshape_dims(ns, dim),
                                          epochs=10000,
                                          shuffle=shuffle, shuffle_learning=True)
                loss = retrieve(structure, function)["losses"]
                losses[shuffle].append(np.max(loss))
        print(f"# {nfuncs=}")
        plt.plot(nfuncs,  losses[False], ".-", label=f"dim {dim} ordered")
        plt.plot(nfuncs,  losses[True], ".-", label=f"dim {dim} shuffled")
    plt.xlabel("Number of functions")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    fs = [l for l, v in zip(["Phase", "Amplitude", "Shape"], dimpatt) if v]
    plt.title(f"Function space {fs}")
    plt.savefig(f"../fig/multidim_fit{dimbin}.pdf")
    plt.close()


# parameter_difficulty_comparison()
for dimbin in range(1, 8):
    multidim_fit(dimbin)

# for i in range(1, 4):
#    print(pack_dims([4, 3, 2], i))

# for i in range(8**3):
#    print(f"{i=}", closest_factor_pair(i), closest_factor_triple(i))
# expid = "two"
# for i in range(1, 4):
#    multidim_comparison(expid, i)
# unidim_comparison(expid)
# bidim_comparison(expid)
# tridim_comparison(expid)

# test()


# parameter_difficulty_comparison()
