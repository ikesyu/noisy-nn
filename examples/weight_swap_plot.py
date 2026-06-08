import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import sys  # noqa
import re
sys.path.append("../")  # noqa
import time

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
    wall_clock_times = []  # List to store the wall clock time for each epoch
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
        wall_clock_times.append(time.time())
        if (epoch + 1) % 100 == 0:
            print(f"epoch {epoch+1}: loss={loss.item()}", end="\r")
            snapshot_fc_layers(stages, m, stage_name=f"epoch_{epoch+1}")
    print("Training ends")
    return losses, stages, wall_clock_times


def full_learn(seed=0, epochs=50000):
    cache_path = f"../data/train_{seed}.pt"
    models = get_models()
    if os.path.isfile(cache_path):
        cached = torch.load(cache_path, map_location="cpu")
        weights = cached.get("weights", {})
        losses = cached.get("losses", {})
        print(f"Loaded cached full-learn results from {cache_path}")
    else:
        weights = {}
        losses = {}
        wtimes = {}

    for model_name, m in models.items():
        if model_name in weights and model_name in losses:
            print(f"Skipping cached model {model_name}")
            continue

        print(model_name)
        loss, stages, wtime = train_model(m, epochs=epochs, seed=seed)
        weights[model_name] = stages
        losses[model_name] = loss
        wtimes[model_name] = wtime
        torch.save({"weights": weights,
                    "losses": losses,
                    "wtime": wtimes
                    }, cache_path)

    torch.save({"weights": weights,
                "losses": losses,
                "wtime": wtimes
                }, cache_path)
    # plt.legend()
    # plt.show()


def switch_learn(seed=0, switch_loc=15000, epochs=30000):
    def entry_cache_path(src_name, dst_name):
        return (
            f"../data/switch_learn_{seed}_{switch_loc}_"
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
                print(
                    f"Missing source snapshot: src={src_name} epoch={switch_loc}")
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
        loss, stages, wtime = train_model(
            m, epochs=epochs - switch_loc, seed=seed,
            layers=task["layers"])
        torch.save(
            {
                "seed": seed,
                "switch_loc": switch_loc,
                "src": src_name,
                "dst": dst_name,
                "loss": loss,
                "wtime": wtime
            },
            cache_path,
        )
        print(f"Saved src={src_name} dst={dst_name} -> {cache_path}")
    # plt.show()


def plot_results(seed=0, switch_loc=15000):
    wl = torch.load(f"train_{seed}.pt", map_location="cpu")
    print(f"{wl=}")
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
        print(
            f"No switch cache found for seed={seed}, switch_loc={switch_loc}")
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


def gen_data():
    full_learn(0)
    for switch_loc in [1000, 10000, 25000]:
        switch_learn(0, switch_loc)
        # plot_results(0, switch_loc)


def simple_plot(seed=0, switch_loc=15000):
    wl = torch.load(f"../data/train_{seed}.pt", map_location="cpu")
    print(f"{wl.keys()}")
    losses = wl["losses"]
    wtime = wl.get("wtime", {})

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
        f"../weight_noswap{seed}.pdf",
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.close(fig_full)

    if len(wtime) == 0:
        print("No wall-time data found in cache. Re-run training to store 'wtime'.")
        return

    fig_wall, ax_wall = plt.subplots(figsize=(8, 4.5))
    for model_name, loss in losses.items():
        if model_name not in model_names:
            continue
        if model_name not in wtime:
            continue
        wct = wtime[model_name]
        if len(wct) == 0:
            continue
        start_time = wct[0]
        walltime = [t - start_time for t in wct]

        ax_wall.semilogy(walltime, loss, "--", label=model_name)

    ax_wall.set_xlabel("Elapsed Time (seconds)")
    ax_wall.set_ylabel("Loss")
    ax_wall.set_title("Training Loss vs. Elapsed Time")
    ax_wall.legend()
    ax_wall.grid(True)
    fig_wall.tight_layout()
    fig_wall.savefig(f"../weight_wall_noswap{seed}.pdf",
                     bbox_inches="tight",
                     pad_inches=0.02,
                     )
    plt.close(fig_wall)

    switch_targets = ["statistic", "sample_100"]

    def load_switch_series(loc):
        series = {}
        for dst_name in switch_targets:
            path = (
                f"../data/switch_learn_{seed}_{loc}_"
                f"src_analytic_dst_{dst_name}.pt"
            )
            if not os.path.isfile(path):
                continue
            item = torch.load(path, map_location="cpu")
            if item.get("src") != "analytic" or item.get("dst") != dst_name:
                continue
            wct = item.get("wtime", [])
            loss = item.get("loss", [])
            if len(wct) == 0 or len(loss) == 0:
                print(f"No wtime/loss data in switch cache: {path}")
                continue
            start_time = wct[0]
            walltime = [t - start_time for t in wct]
            n = min(len(walltime), len(loss))
            series[dst_name] = (walltime[:n], loss[:n])
        return series

    def save_switch_plot(loc, switch_series):
        fig_switch, ax_switch = plt.subplots(figsize=(8, 4.5))
        for dst_name in switch_targets:
            if dst_name not in switch_series:
                continue
            walltime, loss = switch_series[dst_name]
            ax_switch.semilogy(
                walltime,
                loss,
                "--",
                label=f"analytic -> {dst_name}",
            )

        ax_switch.set_xlabel("Elapsed Time (seconds)")
        ax_switch.set_ylabel("Loss")
        ax_switch.set_title(
            f"Switch Training (from analytic) at epoch {loc}"
        )
        ax_switch.grid(True)
        ax_switch.legend()
        fig_switch.tight_layout()
        fig_switch.savefig(
            f"../fig/weight_switch_wall_seed{
                seed}_switch{loc}_src_analytic.pdf",
            bbox_inches="tight",
            pad_inches=0.02,
        )
        plt.close(fig_switch)
        print(f"Saved switch wall-time plot for switch_loc={loc}")

    switch_series = load_switch_series(switch_loc)
    if len(switch_series) > 0:
        save_switch_plot(switch_loc, switch_series)
        return

    print(
        f"No analytic->(statistic/sample_100) switch data for "
        f"seed={seed}, switch_loc={switch_loc}."
    )
    print("Searching available switch_locs in ../data")
    available_locs = set()
    for dst_name in switch_targets:
        pattern = (
            f"../data/switch_learn_{seed}_*_"
            f"src_analytic_dst_{dst_name}.pt"
        )
        for path in glob.glob(pattern):
            match = re.search(
                rf"switch_learn_{seed}_(\d+)_src_analytic_", path)
            if match is not None:
                available_locs.add(int(match.group(1)))

    if len(available_locs) == 0:
        print("No matching switch caches found in ../data")
        return

    for loc in sorted(available_locs):
        series = load_switch_series(loc)
        if len(series) == 0:
            continue
        save_switch_plot(loc, series)


def plot_weight_swap_walltime(seed=0,switch_loc=15000):
    
    pairs = [
        ("analytic", "analytic"),
        ("statistic", "statistic"),
        ("sample_100", "sample_100"),
        ("analytic", "sample_100"),
        ("analytic", "statistic"),
    ]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    plotted = 0
    for src_name, dst_name in pairs:
        path = (
            f"../data/switch_learn_{seed}_{switch_loc}_"
            f"src_{src_name}_dst_{dst_name}.pt"
        )
        if not os.path.isfile(path):
            print(f"Missing switch cache: {path}")
            continue

        item = torch.load(path, map_location="cpu")
        loss = item.get("loss", [])
        wtime = item.get("wtime", [])
        if len(loss) == 0 or len(wtime) == 0:
            print(f"No loss/wtime data in cache: {path}")
            continue

        n = min(len(loss), len(wtime))
        start_time = wtime[0]
        walltime = [t - start_time for t in wtime[:n]]
        linestyle = "--" if src_name == dst_name else "-"
        label = f"{src_name}->{dst_name}"
        ax.semilogy(walltime, loss[:n], linestyle=linestyle, label=label)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        print("No valid switch-1000 data found to plot.")
        return

    ax.set_xlabel("Elapsed Time (seconds)")
    ax.set_ylabel("Loss")
    ax.set_title(f"Switch at {switch_loc}: Loss vs Wall Time")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    plt.xlim([0, 200])
    os.makedirs("../fig", exist_ok=True)
    fig.savefig(
        f"../fig/weight_swap_{switch_loc}.pdf",
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)
    print(f"Saved ../fig/weight_swap_{switch_loc}.pdf")


def generate_weight_swap_latex_table(seed=0):
    train_path = f"../data/train_{seed}.pt"
    if not os.path.isfile(train_path):
        print(f"Missing train cache: {train_path}")
        return

    train = torch.load(train_path, map_location="cpu")
    full_losses = train.get("losses", {})
    model_order = ["analytic", "statistic", "sample_100"]
    model_labels = {
        "analytic": "Analytical",
        "statistic": "Statistics",
        "sample_100": "Sample 100",
    }

    switch_entries = {}
    switch_locs = set()
    src_names = set()
    pattern = f"../data/switch_learn_{seed}_*_src_*_dst_*.pt"
    for path in sorted(glob.glob(pattern)):
        item = torch.load(path, map_location="cpu")
        switch_loc = item.get("switch_loc")
        src_name = item.get("src")
        dst_name = item.get("dst")
        loss = item.get("loss", [])
        if switch_loc is None or src_name is None or dst_name is None:
            continue
        if dst_name not in model_order or len(loss) == 0:
            continue

        switch_locs.add(switch_loc)
        src_names.add(src_name)
        switch_entries[(switch_loc, src_name, dst_name)] = (loss[0], loss[-1])

    src_order = [name for name in model_order if name in src_names]
    switch_loc_order = sorted(switch_locs)

    def format_loss(value):
        if value is None:
            return "--"
        return f"{value:.2e}"

    align = "l" + "c" * (2 * len(model_order))
    latex_lines = [
        r"\begin{tabular}{" + align + r"}",
        r"\hline",
    ]

    header_top = [r"Condition"]
    for model_name in model_order:
        header_top.append(
            rf"\multicolumn{{2}}{{c}}{{{model_labels[model_name]}}}"
        )
    latex_lines.append(" & ".join(header_top) + r" \\")

    header_bottom = [""]
    for _model_name in model_order:
        header_bottom.extend(["Init", "Final"])
    latex_lines.append(" & ".join(header_bottom) + r" \\")
    latex_lines.append(r"\hline")

    row_specs = [("Untrained", None, None)]
    for switch_loc in switch_loc_order:
        for src_name in src_order:
            row_specs.append((f"Switch {switch_loc} / {src_name}", switch_loc, src_name))

    for row_label, switch_loc, src_name in row_specs:
        row = [row_label]
        for dst_name in model_order:
            if switch_loc is None:
                losses = full_losses.get(dst_name, [])
                initial_loss = losses[0] if len(losses) > 0 else None
                final_loss = losses[-1] if len(losses) > 0 else None
            else:
                initial_loss, final_loss = switch_entries.get(
                    (switch_loc, src_name, dst_name),
                    (None, None),
                )
            row.append(format_loss(initial_loss))
            row.append(format_loss(final_loss))
        latex_lines.append(" & ".join(row) + r" \\")

    latex_lines.extend([
        r"\hline",
        r"\end{tabular}",
    ])

    os.makedirs("../fig", exist_ok=True)
    output_path = "../fig/weight_swap_loss_table.tex"
    with open(output_path, "w", encoding="ascii") as handle:
        handle.write("\n".join(latex_lines) + "\n")

    print(f"Saved {output_path}")


# gen_data()
#simple_plot(0, 10000)
generate_weight_swap_latex_table(0)
