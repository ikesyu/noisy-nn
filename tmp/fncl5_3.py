"""
fncl5_3.py — 論文 §5.3「勾配の不偏性の直接検証」(Fig.5)

(a) 共分散 weight mirror の回復精度:
      W_hat = Cov_T(d^{(l+1)}, z^{(l)}) / Var_T(z^{(l)})  (入力プーリング)
    を真の重み W と比較する (Pearson r, 散布図)。連続量 d と相関を取る場合と、
    二値 z と相関を取る退化ケース (対照) の両方を測る。

(b) cov_jac / cov_deriv の更新方向の忠実度:
    各層の手動勾配 (forward 統計のみ) を autograd の厳密勾配と比較し、
    層別の cosine 類似度と magnitude ratio を、未学習と部分学習後
    (backprop で --pretrain-epochs) の 2 状態で測る。
    どちらの推定も --grad-draws 回の確率的 forward で平均してから比較する。

生成物 (out/fncl5_3/):
  fig_mirror_scatter.png  -> Fig.5a (W_hat vs W 散布図 x3: d 相関 / readout / 二値対照)
  fig_grad_cosine.png     -> Fig.5b (層別 cosine, cov_jac vs cov_deriv, 2 状態)
  table_fidelity.md       -> 数表 (r / cosine / ratio)
  results.json

実行例:
  python tmp/fncl5_3.py
  python tmp/fncl5_3.py --grad-draws 128 --mirror-passes 16
  python tmp/fncl5_3.py --quick
"""
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt

from fncl_common import (add_common_args, finalize_args, make_task,
                         model_factory, config_dict, pearson, cosine,
                         norm_ratio, write_text, save_json, savefig, fncl)


# ------------------------------------------------------------
# サンプル収集と勾配推定 (train_cov の 1 epoch 分を切り出したもの)
# ------------------------------------------------------------
def collect_samples(net, x, passes: int = 1) -> dict:
    """forward フックで per-sample の d, z, y_samples を passes 回分収集する."""
    cap = fncl.Capture(net)
    z = [[] for _ in range(cap.n_hidden)]
    d = [[] for _ in range(cap.n_hidden)]
    ys, y_out = [], []
    with torch.no_grad():
        for _ in range(passes):
            y_out.append(net(x))
            ys.append(cap.y_samples)
            for l in range(cap.n_hidden):
                z[l].append(cap.z[l])
                d[l].append(cap.d[l])
    cap.remove()
    return {
        "y": torch.stack(y_out, dim=0).mean(dim=0),           # [N, 1]
        "ys": torch.cat(ys, dim=1),                           # [N, K*T, 1]
        "z": [torch.cat(zl, dim=1) for zl in z],              # [N, K*T, H]
        "d": [torch.cat(dl, dim=1) for dl in d],
    }


def measure_mirrors(net, x, passes: int):
    """(W1_hat[d 相関], Wout_hat, W1_hat_binary[二値対照]) を返す."""
    s = collect_samples(net, x, passes)
    w1_hat = fncl.cov_weight(s["d"][1], s["z"][0], pool=True)    # 正: 連続量 d と相関
    wout_hat = fncl.cov_weight(s["ys"], s["z"][1], pool=True)    # readout mirror
    w1_bin = fncl.cov_weight(s["z"][1], s["z"][0], pool=True)    # 対照: 二値 z と相関
    return w1_hat, wout_hat, w1_bin


def cov_jac_gradient(net, x, t) -> dict:
    """cov_jac の 1 回分の勾配推定 (mirror は当該 forward から直接測定, EMA なし)."""
    s = collect_samples(net, x, passes=1)
    crossings = fncl.crossing_layers(net)
    n_hidden = len(crossings)
    slope_full = [fncl.kde_slope(crossings[l], s["d"][l]) for l in range(n_hidden)]
    slope_mean = [sf.mean(dim=1) for sf in slope_full]
    w_hat = {"out": fncl.cov_weight(s["ys"], s["z"][-1], pool=True)}
    for l in range(1, n_hidden):
        w_hat[l] = fncl.cov_weight(s["d"][l], s["z"][l - 1], pool=True)
    y = s["y"]
    N, T = x.shape[0], s["z"][0].shape[1]
    a = [None] * n_hidden
    a[-1] = (2.0 * (y - t)) * w_hat["out"]                       # [N, H]
    for l in range(n_hidden - 2, -1, -1):
        a[l] = (a[l + 1] * slope_mean[l + 1]) @ w_hat[l + 1]     # [N, H_l]
    z_prev = [x.unsqueeze(1).expand(N, T, x.shape[1]), s["z"][0]]
    grads = {}
    for l in range(n_hidden):
        delta = a[l].unsqueeze(1) * slope_full[l]                # [N, T, H]
        grads[f"w{l}"] = torch.einsum("nto,nti->oi", delta, z_prev[l]) / (N * T)
    z_bar = s["z"][-1].mean(dim=1)
    grads["wout"] = torch.einsum("no,ni->oi", 2.0 * (y - t), z_bar) / N
    return grads


def cov_deriv_gradient(net, x, t, credit: str) -> dict:
    """cov_deriv (kde slope) の 1 回分の勾配推定 (対照用)."""
    s = collect_samples(net, x, passes=1)
    crossings = fncl.crossing_layers(net)
    n_hidden = len(crossings)
    L = (s["ys"].squeeze(-1) - t) ** 2                           # [N, T]
    y = s["y"]
    N, T = x.shape[0], s["z"][0].shape[1]
    z_prev = [x.unsqueeze(1).expand(N, T, x.shape[1]), s["z"][0]]
    grads = {}
    for l in range(n_hidden):
        g_bcast, _ = fncl.covariance_credit(s["z"][l], L, credit)
        dz_dd = fncl.kde_slope(crossings[l], s["d"][l])
        delta = g_bcast * dz_dd
        grads[f"w{l}"] = torch.einsum("nto,nti->oi", delta, z_prev[l]) / (N * T)
    z_bar = s["z"][-1].mean(dim=1)
    grads["wout"] = torch.einsum("no,ni->oi", 2.0 * (y - t), z_bar) / N
    return grads


def autograd_gradient(net, x, t) -> dict:
    """autograd による厳密勾配 (同じ確率的モデル上の 1 draw)."""
    for prm in net.parameters():
        prm.grad = None
    loss = ((net(x) - t) ** 2).mean()
    loss.backward()
    return {"w0": net.fcs[0].weight.grad.detach().clone(),
            "w1": net.fcs[1].weight.grad.detach().clone(),
            "wout": net.fcs[-1].weight.grad.detach().clone()}


def averaged(fn, draws: int) -> dict:
    acc = None
    for _ in range(draws):
        g = fn()
        acc = g if acc is None else {k: acc[k] + v for k, v in g.items()}
    return {k: v / draws for k, v in acc.items()}


# ------------------------------------------------------------
# 図
# ------------------------------------------------------------
def plot_mirror_scatter(pairs, path):
    """pairs: list of (title, W_hat, W_true, r)."""
    fig, axes = plt.subplots(1, len(pairs), figsize=(4.2 * len(pairs), 4.0))
    for ax, (title, w_hat, w_true, r) in zip(np.atleast_1d(axes), pairs):
        a = w_true.detach().cpu().numpy().ravel()
        b = w_hat.detach().cpu().numpy().ravel()
        lim = max(np.abs(a).max(), np.abs(b).max()) * 1.1
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, alpha=0.5)
        ax.scatter(a, b, s=8, alpha=0.6)
        ax.set_xlabel("true W")
        ax.set_ylabel("mirror W_hat")
        ax.set_title(f"{title}\nPearson r = {r:.4f}")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    savefig(fig, path)


def plot_grad_cosine(fid, path):
    """fid[state][estimator][layer] = {"cos":, "ratio":} の層別棒グラフ."""
    states = list(fid.keys())
    layers = ["w0", "w1", "wout"]
    fig, axes = plt.subplots(1, len(states), figsize=(5.2 * len(states), 4.0),
                             sharey=True)
    width = 0.35
    xpos = np.arange(len(layers))
    for ax, state in zip(np.atleast_1d(axes), states):
        for k, est in enumerate(("cov_jac", "cov_deriv")):
            vals = [fid[state][est][l]["cos"] for l in layers]
            ax.bar(xpos + (k - 0.5) * width, vals, width, label=est)
            for xp, v in zip(xpos + (k - 0.5) * width, vals):
                ax.text(xp, min(v + 0.02, 1.05), f"{v:.3f}", ha="center",
                        fontsize=7)
        ax.axhline(1.0, color="k", lw=0.6, alpha=0.5)
        ax.set_xticks(xpos)
        ax.set_xticklabels(layers)
        ax.set_ylim(0.0, 1.15)
        ax.set_title(f"cosine vs autograd gradient ({state})")
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8)
    np.atleast_1d(axes)[0].set_ylabel("cosine similarity")
    fig.tight_layout()
    savefig(fig, path)


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(
        description="§5.3 gradient fidelity: mirror recovery + update-direction "
                    "cosine vs autograd.")
    add_common_args(p, hidden_dim=32, seeds="0")
    p.add_argument("--noise", choices=("gaussian", "uniform"), default="gaussian")
    p.add_argument("--mirror-passes", type=int, default=8,
                   help="mirror 測定に使う forward パス数 (実効サンプル = passes*T)")
    p.add_argument("--grad-draws", type=int, default=32,
                   help="勾配推定の平均化 draw 数")
    p.add_argument("--pretrain-epochs", type=int, default=300,
                   help="『部分学習後』状態を作る backprop epoch 数")
    args = finalize_args(p.parse_args(), default_out="out/fncl5_3")
    if args.quick:
        args.mirror_passes, args.grad_draws, args.pretrain_epochs = 2, 4, 20

    device = torch.device(args.device)
    x_raw, target, x, t = make_task(device)
    seed = args.seed_list[0]
    torch.manual_seed(seed)
    np.random.seed(seed)
    fresh = model_factory(args.noise, args, device)

    results = {"config": config_dict(args), "mirror": {}, "gradient": {}}
    fid = {}
    for state in ("untrained", "pretrained"):
        net = fresh()
        if state == "pretrained":
            print(f"[{state}] backprop pretraining "
                  f"({args.pretrain_epochs} epochs) ...", flush=True)
            fncl.train_backprop(net, x, t, args.lr, args.pretrain_epochs)

        # (a) mirror 回復精度
        w1_hat, wout_hat, w1_bin = measure_mirrors(net, x, args.mirror_passes)
        w1 = net.fcs[1].weight
        wout = net.fcs[-1].weight
        mirror = {
            "r_hidden_d": pearson(w1_hat, w1),
            "r_readout": pearson(wout_hat, wout),
            "r_hidden_binary_z": pearson(w1_bin, w1),
        }
        results["mirror"][state] = mirror
        print(f"[{state}] mirror recovery: "
              f"r(d)={mirror['r_hidden_d']:.4f}  "
              f"r(readout)={mirror['r_readout']:.4f}  "
              f"r(binary z, 対照)={mirror['r_hidden_binary_z']:.4f}", flush=True)
        if state == "untrained":
            plot_mirror_scatter(
                [("hidden W (corr. with d)", w1_hat, w1, mirror["r_hidden_d"]),
                 ("readout W", wout_hat, wout, mirror["r_readout"]),
                 ("hidden W (corr. with binary z)", w1_bin, w1,
                  mirror["r_hidden_binary_z"])],
                args.out_dir / "fig_mirror_scatter.png")

        # (b) 更新方向の忠実度 (draw 平均後に比較)
        print(f"[{state}] averaging gradients over {args.grad_draws} draws ...",
              flush=True)
        g_auto = averaged(lambda: autograd_gradient(net, x, t), args.grad_draws)
        g_jac = averaged(lambda: cov_jac_gradient(net, x, t), args.grad_draws)
        g_der = averaged(lambda: cov_deriv_gradient(net, x, t, args.credit),
                         args.grad_draws)
        fid[state] = {}
        for est_name, g_est in (("cov_jac", g_jac), ("cov_deriv", g_der)):
            fid[state][est_name] = {
                k: {"cos": cosine(g_est[k], g_auto[k]),
                    "ratio": norm_ratio(g_est[k], g_auto[k])}
                for k in ("w0", "w1", "wout")}
            summary = "  ".join(
                f"{k}: cos={fid[state][est_name][k]['cos']:.4f} "
                f"ratio={fid[state][est_name][k]['ratio']:.2f}"
                for k in ("w0", "w1", "wout"))
            print(f"[{state}] {est_name:10s} {summary}", flush=True)
    results["gradient"] = fid

    plot_grad_cosine(fid, args.out_dir / "fig_grad_cosine.png")

    lines = ["| state | quantity | value |", "|---|---|---|"]
    for state, m in results["mirror"].items():
        for k, v in m.items():
            lines.append(f"| {state} | mirror {k} | {v:.4f} |")
    for state in fid:
        for est in fid[state]:
            for layer in fid[state][est]:
                c = fid[state][est][layer]
                lines.append(f"| {state} | {est} {layer} cosine (ratio) | "
                             f"{c['cos']:.4f} ({c['ratio']:.2f}) |")
    write_text(args.out_dir / "table_fidelity.md", "\n".join(lines) + "\n")
    save_json(args.out_dir / "results.json", results)


if __name__ == "__main__":
    main()
