import numpy as np

T = 100000


def th(x):
    return x > 0.5


sizes = [1, 3, 3, 1]
fcs = [np.random.normal(size=(post, pre))
       for (pre, post) in zip(sizes, sizes[1:])]


def sampleNet(x):
    for i, fc in enumerate(fcs):
        x = fc.dot(x)
        if i < len(fcs)-1:
            x = th(x+np.random.uniform(size=x.shape))
    return x


def statNet(x, T):
    for i, fc in enumerate(fcs):
        x = fc.dot(x)
        if i < len(fcs)-1:
            noisyx = x.dot(np.ones((1, T)))
            noisyx = noisyx+np.random.uniform(size=noisyx.shape)
            x = np.mean(th(noisyx), axis=1, keepdims=True)
    return x


x = np.array(0.3).reshape(1, 1)

T = 10000
samples = [np.mean([sampleNet(x) for _ in range(T)]) for __ in range(4)]
stats = [np.mean([statNet(x, T)]) for __ in range(4)]


print(samples, stats)
