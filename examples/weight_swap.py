import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
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
    Ts = [10, 100, 1000]
    models = {}
    models["analytic"] = model.SimpleNNNAnalytic()
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

    models = get_models()
    weights = {}
    losses = {}
    for model_name, m in models.items():
        print(model_name)
        loss, stages = train_model(m, epochs=epochs, seed=seed)
        plt.semilogy(loss, label=model_name)
        weights[model_name] = stages
        losses[model_name] = loss
    torch.save({"weights": weights,
                "losses": losses}, f"train_{seed}.pt")
    plt.legend()
    plt.show()


def switch_learn(seed=0, switch_loc=15000, epochs=30000):
    weights = torch.load(f"train_{seed}.pt", map_location="cpu")["weights"]
    models = get_models()
    losses = {}
    for dst_name, m in models.items():
        # plt.figure()
        for src_name, w in weights.items():
            print(f"src={src_name} dst={dst_name}")
            loss, stages = train_model(
                m, epochs=epochs-switch_loc, seed=seed,
                layers=w[f"epoch_{switch_loc}"])
            losses[(src_name, dst_name)] = loss
            # plt.plot(loss, label=f"{src_name} {dst_name}")
        # plt.legend()
    torch.save(losses, f"switch_learn_{seed}_{switch_loc}.pt")
    # plt.show()


def plot_results(seed=0, switch_loc=15000):
    wl = torch.load(f"train_{seed}.pt", map_location="cpu")
    losses = wl["losses"]

    for model_name, loss in losses.items():
        epoch = range(0, len(loss))
        plt.semilogy(epoch, loss, "--", label=model_name)
    plt.grid(True)
    plt.legend()

    switch_losses = torch.load(f"switch_learn_{seed}_{
                               switch_loc}.pt", map_location="cpu")

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
    plt.show()


# full_learn(0)
for switch_loc in [25000]:  # 1000, 10000,
    switch_learn(0, switch_loc)
    plot_results(0, switch_loc)
