import numpy as np
import torch
import matplotlib.pyplot as plt


def sine_pfa(x, n_phase=1, n_freq=None, n_amp=None, shuffle=False):
    tosqueeze = []
    if n_freq is None:
        tosqueeze.append(1)
        n_freq = 1
    if n_amp is None:
        tosqueeze.append(2)
        n_amp = 1

    phase = np.linspace(0, 2*np.pi, n_phase, endpoint=False)
    freq = np.linspace(1, 2, n_freq)
    amp = np.linspace(1, 2, n_amp)
    # print(f"{phase=} {freq=} {amp=}")

    vals = amp[None, None, :, None, None] * np.sin(
        freq[None, :, None, None, None] * x[None, None, None, :, :] + phase[:, None, None, None, None])

    assert np.min(vals.shape) > 0, "did you pass shuffle by value?"

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
    def __init__(self, shape, shuffle=False, epochs=1000, shuffle_learning=False):
        self.shuffle = shuffle
        self.shuffle_learning = shuffle_learning
        self.shape = tuple(shape)
        self.x = np.linspace(-2 * np.pi, 2 * np.pi, 200).reshape(-1, 1)
        self.x_tensor = torch.tensor(self.x, dtype=torch.float32)
        self.ys = sine_pfa(self.x, *shape, shuffle=shuffle)
        self.epochs = epochs
    # threads_per_run = max(1, os.cpu_count()-2)


def gray_ndindex(shape):
    shape = tuple(int(s) for s in shape)
    if any(s <= 0 for s in shape):
        return  # yields nothing if any dimension is non-positive
    n = len(shape)
    idx = [0] * n
    dir = [1] * n  # direction along each axis
    total = 1
    for s in shape:
        total *= s
    for _ in range(total):
        yield tuple(idx)
        a = n - 1  # start with the last axis (fastest-changing)
        while a >= 0 and not (0 <= idx[a] + dir[a] < shape[a]):
            dir[a] = -dir[a]  # bounce and flip direction
            a -= 1  # escalate to the next slower axis
        if a >= 0:
            idx[a] += dir[a]


def square_sine(x, alpha):
    return (1-alpha)*np.sin(x)+alpha*np.sign(np.sin(x))


def triangle_sine(x, alpha):
    return (1-alpha)*np.sin(x)+alpha*2/np.pi*np.arcsin(np.sin(x))


def sine_like_functions(x, n_phase, n_amp, n_alpha, func):
    phase = np.linspace(0, 2*np.pi, n_phase, endpoint=False)
    amp = np.linspace(1, 2, n_amp)
    squareness = np.linspace(0, 1, n_alpha)

    funcs = []
    shape = (n_phase, n_amp, n_alpha)

    indices = list(gray_ndindex(shape))

    for idx in indices:
        f = amp[idx[1]]*func(x+phase[idx[0]], squareness[idx[2]])
        # f = np.hstack([[idx[0], idx[1], idx[2]],
        #               np.zeros(200-3)]).reshape(200, 1)
        funcs.append(f)

    return funcs


def plain_sine(x, n_phase, n_freq, n_amp):
    phase = np.linspace(0, 2*np.pi, n_phase, endpoint=False)
    amp = np.linspace(1, 2, n_amp, endpoint=True)
    freq = np.linspace(1, 2, n_freq, endpoint=True)

    funcs = []
    shape = (n_phase, n_freq, n_amp)

    indices = list(gray_ndindex(shape))

    for idx in indices:
        f = amp[idx[2]]*np.sin(freq[idx[1]]*x+phase[idx[0]])
        # f = np.hstack([[idx[0], idx[1], idx[2]],
        #               np.zeros(200-3)]).reshape(200, 1)
        funcs.append(f)

    return funcs


class SineLikeFunctions:
    def __init__(self, construct_shape, present_shape,
                 shuffle, epochs, shuffle_learning, function, name):
        self.shuffle = shuffle
        self.shuffle_learning = shuffle_learning
        self.construct_shape = construct_shape
        self.shape = tuple(present_shape)
        self.x = np.linspace(-2 * np.pi, 2 * np.pi, 200).reshape(-1, 1)
        self.x_tensor = torch.tensor(self.x, dtype=torch.float32)

        if function == plain_sine:
            linear_ys = np.array(plain_sine(self.x, *construct_shape))
        else:
            linear_ys = np.array(sine_like_functions(
                self.x, *construct_shape, function))
        if (shuffle):
            p = np.random.permutation(len(linear_ys))
            linear_ys = linear_ys[p, :, :]

        # print(f"{linear_ys=}")
        yshape = list(present_shape)+[len(self.x), 1]
        ys = np.empty(yshape)
        indices = list(gray_ndindex(present_shape))
        if len(indices) != len(linear_ys):
            raise Exception(f"Mismatch in sizes {
                            construct_shape = } {present_shape = }")
        for idx, y in zip(indices, linear_ys):
            # print(f"{ys.shape=} {idx=} {ys[*idx, :]=} {y=}")
            ys[*idx, :] = y
        self.ys = torch.tensor(ys, dtype=torch.float32)
        self.epochs = epochs
        self.function_type = name


class SquineFunctions(SineLikeFunctions):
    def __init__(self, construct_shape, present_shape, shuffle=False, epochs=1000, shuffle_learning=False):
        super().__init__(construct_shape, present_shape,
                         shuffle, epochs, shuffle_learning, square_sine, "squine")


class TrineFunctions(SineLikeFunctions):
    def __init__(self, construct_shape, present_shape, shuffle=False, epochs=1000, shuffle_learning=False):
        super().__init__(construct_shape, present_shape,
                         shuffle, epochs, shuffle_learning, triangle_sine, "trine")


class PlainSineFunctions(SineLikeFunctions):
    def __init__(self, construct_shape, present_shape, shuffle=False, epochs=1000, shuffle_learning=False):
        super().__init__(construct_shape, present_shape,
                         shuffle, epochs, shuffle_learning, plain_sine, "plainsine")


# present_shape = [8, 1]
# f = TrineFunctions((1, 1, 8), present_shape, shuffle=False)

# # fig, axs = plt.subplots(*present_shape)
# for i in range(present_shape[0]):
#     for j in range(present_shape[1]):
#          axs[i, j].plot(f.ys[i, j], ":", label=f"{i=} {j=}")
#          axs[j].plot(f.ys[i, j], ":", label=f"{i=} {j=}")
#         plt.plot(f.ys[i, j], ":", label=f"{i=} {j=}")
# plt.legend()
# plt.show()
