import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import sys  # noqa
sys.path.append("../")  # noqa


import nnn.noise as noise  # noqa from nnn import noise
from nnn import model  # noqa
from nnn import layer  # noqa
from nnn import activation  # noqa

device = torch.device("cuda")
torch.set_default_device(device)


# データセットの作成
x = np.linspace(-2 * np.pi, 2 * np.pi, 1000).reshape(-1, 1)
y = np.sin(x)

# PyTorchのテンソルに変換
x_tensor = torch.tensor(x, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# モデルのインスタンス化


# 損失関数と最適化アルゴリズム
criterion = nn.MSELoss()


def get_models():
    # Ts = [10, 100, 1000]  # 1 does not learn
    Ts = [100]
    models = {}
    models["analytic"] = model.SimpleNNNAnalytic()
    models["statistic"] = model.SimpleNNNStatistic()

    for t in Ts:
        models[f"sample_{t}"] = model.SimpleNNNSample(t=t, h=5E-2)
    return models


def load_weights(m, layers):
    if layers:
        for key, sd in layers.items():
            i = int(key.split(".")[1])
            m.fcs[i].load_state_dict(sd)


def snapshot_fc_layers(stages, m, indices=None, stage_name=None, compress_dtype=None):
    # indices=None => all layers in m.fcs
    if indices is None:
        indices = list(range(len(m.fcs)))
    entry = {}
    for i in indices:
        sd = m.fcs[i].state_dict()
        # clone to CPU so future training updates don't mutate the snapshot
        if compress_dtype is None:
            entry[f"fcs.{i}"] = {k: v.detach().cpu().clone()
                                 for k, v in sd.items()}
        else:
            entry[f"fcs.{i}"] = {k: v.detach().to(
                dtype=compress_dtype).cpu().clone() for k, v in sd.items()}
    stages[stage_name] = entry


def train_model(m, epochs=5000, seed=0, layers=None):
    losses = []
    torch.manual_seed(seed)
    np.random.seed(seed)
    load_weights(m, layers)
    optimizer = optim.AdamW(m.parameters(),
                            lr=3E-4, weight_decay=0.0)

    print("Training starts")
    stages = {}
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = m(x_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if (epoch + 1) % 100 == 0:
            print(f"epoch {epoch+1}: loss={loss.item()}", end="\r")
            snapshot_fc_layers(stages, m, stage_name=f"epoch_{epoch+1}")
    print("Training ends")
    return losses, stages


def full_learn(seed=0, epochs=50000):
    cache_path = f"train_{seed}.pt"
    models = get_models()
    if os.path.isfile(cache_path):
        cached = torch.load(cache_path, map_location="cpu")
        weights = cached.get("weights", {})
        losses = cached.get("losses", {})
        print(f"Loaded cached full-learn results from {cache_path}")
    else:
        weights = {}
        losses = {}

    for model_name, m in models.items():
        if model_name in weights and model_name in losses:
            print(f"Skipping cached model {model_name}")
            continue

        print(model_name)
        loss, stages = train_model(m, epochs=epochs, seed=seed)
        weights[model_name] = stages
        losses[model_name] = loss
        torch.save({"weights": weights,
                    "losses": losses}, cache_path)

    torch.save({"weights": weights,
                "losses": losses}, cache_path)
    # plt.legend()
    # plt.show()


def switch_learn(seed=0, switch_loc=15000, epochs=30000):
    def entry_cache_path(src_name, dst_name):
        return (
            f"switch_learn_{seed}_{switch_loc}_"
            f"src_{src_name}_dst_{dst_name}.pt"
        )

    weights = torch.load(f"train_{seed}.pt", map_location="cpu")["weights"]
    models = get_models()
    model_names = list(models.keys())
    tasks = []

    for dst_name in model_names:
        for src_name in model_names:
            print(f"src={src_name} dst={dst_name}")
            if src_name not in weights:
                print(
                    f"Skipping missing source model in train cache: src={src_name}")
                continue
            w = weights[src_name]
            cache_path = entry_cache_path(src_name, dst_name)
            if os.path.isfile(cache_path):
                print(f"Skipping cached src={src_name} dst={dst_name}")
                continue
            if f"epoch_{switch_loc}" not in w:
                print(f"Missing source snapshot: src={src_name} epoch={switch_loc}")
                continue
            tasks.append(
                {
                    "src_name": src_name,
                    "dst_name": dst_name,
                    "seed": seed,
                    "switch_loc": switch_loc,
                    "epochs": epochs,
                    "layers": w[f"epoch_{switch_loc}"],
                    "cache_path": cache_path,
                }
            )

            # plt.plot(loss, label=f"{src_name} {dst_name}")
        # plt.legend()

    if len(tasks) == 0:
        print(
            f"No missing switch-learning jobs for seed={seed}, switch_loc={switch_loc}")
        return

    print(f"Running {len(tasks)} missing switch-learning jobs sequentially")
    for task in tasks:
        src_name = task["src_name"]
        dst_name = task["dst_name"]
        cache_path = task["cache_path"]
        m = get_models()[dst_name]
        loss, stages = train_model(
            m, epochs=epochs - switch_loc, seed=seed,
            layers=task["layers"])
        torch.save(
            {
                "seed": seed,
                "switch_loc": switch_loc,
                "src": src_name,
                "dst": dst_name,
                "loss": loss,
            },
            cache_path,
        )
        print(f"Saved src={src_name} dst={dst_name} -> {cache_path}")
    # plt.show()


def plot_results(seed=0, switch_loc=15000):
    wl = torch.load(f"train_{seed}.pt", map_location="cpu")
    losses = wl["losses"]
    models = get_models()
    model_names = set(models.keys())

    fig_full, ax_full = plt.subplots(figsize=(8, 4.5))

    for model_name, loss in losses.items():
        if model_name not in model_names:
            continue
        epoch = range(0, len(loss))
        ax_full.semilogy(epoch, loss, "--", label=model_name)
    ax_full.grid(True)
    ax_full.legend()
    fig_full.tight_layout()
    fig_full.savefig(
        f"full_learn_{seed}.pdf",
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig_full)

    switch_losses = {}
    pattern = f"switch_learn_{seed}_{switch_loc}_src_*_dst_*.pt"
    for path in sorted(glob.glob(pattern)):
        item = torch.load(path, map_location="cpu")
        if item["src"] not in model_names or item["dst"] not in model_names:
            continue
        key = (item["src"], item["dst"])
        switch_losses[key] = item["loss"]

    # Fallback to legacy aggregate cache format.
    if len(switch_losses) == 0:
        legacy_path = f"switch_learn_{seed}_{switch_loc}.pt"
        if os.path.isfile(legacy_path):
            old_losses = torch.load(legacy_path, map_location="cpu")
            switch_losses = {
                (src, dst): loss
                for (src, dst), loss in old_losses.items()
                if src in model_names and dst in model_names
            }

    if len(switch_losses) == 0:
        print(f"No switch cache found for seed={seed}, switch_loc={switch_loc}")
        return

    # Get unique dst_names from switch_losses
    dst_names = sorted(set(dst_name for (_, dst_name) in switch_losses.keys()))
    num_subplots = len(dst_names)

    # Create subplots: one for each dst_name
    fig, axes = plt.subplots(nrows=num_subplots, ncols=1,
                             figsize=(8, 4 * num_subplots))

    # Ensure axes is iterable
    if num_subplots == 1:
        axes = [axes]

    # For each dst_name, plot relevant data
    for ax, dst_name in zip(axes, dst_names):

        # Plot switch_losses relevant to this dst_name
        for (src_name, dst_name_key), loss in switch_losses.items():
            if dst_name_key == dst_name:
                epoch = range(switch_loc, switch_loc + len(loss))
                ax.semilogy(epoch, loss, "--", label=f"{src_name} {dst_name}")

        ax.set_title(f"Results for destination: {dst_name}")
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        src = "all"
        fig.savefig(
            f"swap_{src}_{dst_name}_{switch_loc}.pdf",
            bbox_inches="tight",
            pad_inches=0.02,
        )
    plt.close(fig)


full_learn(0)
for switch_loc in [1000, 10000, 25000]:
    switch_learn(0, switch_loc)
    plot_results(0, switch_loc)
