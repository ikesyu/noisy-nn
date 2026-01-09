from multidimensional_storage import SineFunctions,  Structure, train_net, set_cuda
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import torch
import numpy as np
import matplotlib.pyplot as plt

set_cuda(False)
structure = Structure([100])


def worker(shape, shuffle, epochs, threads_per_run):
    # Force CPU-only and cap per-process threads to avoid oversubscription
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("OMP_NUM_THREADS", str(threads_per_run))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads_per_run))
    torch.set_num_threads(threads_per_run)
    functions = SineFunctions(shape, shuffle, epochs)
    model, losses = train_net(structure, functions)
    return losses


def run_batch(shapes, shuffle):
    cpus = os.cpu_count()-1
    n_jobs = min(cpus, len(shapes))
    threads_per_run = cpus//n_jobs
    epochs = 10000
    print(f"{os.cpu_count()=} {n_jobs=} {threads_per_run=}")
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        # ex.map preserves order; repeat() passes constants to all calls
        results = list(
            ex.map(worker, shapes, repeat(shuffle), repeat(epochs), repeat(threads_per_run)))
    res = {tuple(s): r for s, r in zip(shapes, results)}
    return res


def compute_storage_comparison(shapes, label=""):
    results = {"ordered": run_batch(shapes, shuffle=False),
               "shuffled": run_batch(shapes, shuffle=True)
               }
    torch.save(results, f"../data/multidimensional_storage_{label}.pt")


def plot_storage_comparisons(label=""):
    results = torch.load(f"../data/multidimensional_storage_{label}.pt",
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


shapes = [[i] for i in range(1, 11, 1)]
compute_storage_comparison(shapes, "unidim")
plot_storage_comparisons("unidim")
