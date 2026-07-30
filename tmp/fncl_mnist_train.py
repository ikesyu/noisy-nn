"""
fncl_mnist_train.py — A1 の決着: 学習時ミラー (EMA + Kolen-Pollack 追従) での忠実度と実学習
（旧名: fncl_a1c.py．出力ディレクトリ tmp/out/ は旧名のまま）

fncl_mnist_fidelity / fncl_mnist_fidelity_decomp は「backprop で事前学習したネット上でミラーを単発測定する」
§5.3 プロトコルであり、実際の cov_jac 学習が使うミラーとは異なる。train_cov の
ミラーは (i) 毎 epoch の共分散測定を EMA(0.9) で平滑化し、(ii) 適用した重み更新量を
そのままミラーに加算する Kolen-Pollack 追従を併用する。この差が深層での cosine 低下
(fncl_mnist_fidelity_decomp: ミラー由来のバイアスと判明) をどれだけ埋めるかを直接測る。

本スクリプトは MNIST (784-256-...-10, 交差エントロピー, 多出力) 上で cov_jac を
実際に走らせ、学習しながら定期的に

  - CE / accuracy (学習が成立するか = B1 の前哨),
  - 学習時ミラー W_ema と真の W の Pearson r,
  - 学習時ミラーで作った更新方向と autograd 厳密勾配の層別 cosine

を測る。backprop を同一初期重みで並走させて参照とする。
train_cov はスカラー出力・MSE 前提のため、ここに CE 多出力版を実装する
(学習則そのものは §4.3 と同一: 共分散ミラー + 再帰 + KDE slope + 局所 Adam)。

生成物 (tmp/out/fncl_a1c/): results.json, (標準出力) 進捗と最終表

実行例:
  .venv/bin/python tmp/fncl_mnist_train.py --quick
  .venv/bin/python tmp/fncl_mnist_train.py --depths 2,4 --epochs 300
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as tF

DATA_DIR = Path(__file__).resolve().parent.parent / "data_nce"
sys.path.insert(0, str(DATA_DIR))
import fncl  # noqa: E402
from fncl_common import cosine, norm_ratio, pearson, save_json  # noqa: E402
from fncl_mnist_fidelity import (autograd_gradient, averaged, build_net, ce_error,  # noqa: E402
                     collect_samples, load_mnist)


def evaluate(net, x, labels, x_te=None, y_te=None):
    """(train CE, train acc, test acc) を返す (test セットがなければ test acc は nan)."""
    with torch.no_grad():
        logits = net(x)
        ce = float(tF.cross_entropy(logits, labels))
        acc = float((logits.argmax(1) == labels).float().mean())
        te = float("nan")
        if x_te is not None:
            te = float((net(x_te).argmax(1) == y_te).float().mean())
    return ce, acc, te


def manual_grads(net, x, labels, w_mirror, cap_samples):
    """与えられたミラー w_mirror と収集済みサンプルから cov_jac の勾配を作る."""
    s = cap_samples
    crossings = fncl.crossing_layers(net)
    n_hidden = len(crossings)
    slope_full = [fncl.kde_slope(crossings[l], s["d"][l]) for l in range(n_hidden)]
    slope_mean = [sf.mean(dim=1) for sf in slope_full]
    N, T = x.shape[0], s["z"][0].shape[1]
    e = ce_error(s["y"], labels)                                   # [N, 10]
    a = [None] * n_hidden
    a[-1] = e @ w_mirror["out"]
    for l in range(n_hidden - 2, -1, -1):
        a[l] = (a[l + 1] * slope_mean[l + 1]) @ w_mirror[l + 1]
    z_prev = ([x.unsqueeze(1).expand(N, T, x.shape[1])]
              + [s["z"][l] for l in range(n_hidden - 1)])
    gW, gb = {}, {}
    for l in range(n_hidden):
        delta = a[l].unsqueeze(1) * slope_full[l]
        gW[f"w{l}"] = torch.einsum("nto,nti->oi", delta, z_prev[l]) / (N * T)
        gb[f"w{l}"] = delta.mean(dim=(0, 1))
    gW["wout"] = torch.einsum("no,ni->oi", e, s["z"][-1].mean(dim=1)) / N
    gb["wout"] = e.mean(dim=0)
    return gW, gb


def train_cov_jac_ce(net, x, labels, lr: float, epochs: int, jac_ema: float,
                     eval_every: int, grad_draws: int, log, test=(None, None)):
    """cov_jac (§4.3) の CE 多出力版。EMA + Kolen-Pollack 追従つきミラー、局所 Adam。"""
    n_hidden = len(fncl.crossing_layers(net))
    optim = fncl.ManualOpt("adam")
    W_ema = {}
    history = []
    for epoch in range(epochs):
        with torch.no_grad():
            s = collect_samples(net, x, passes=1)
            # --- 今 epoch の共分散ミラー測定 → EMA 平滑化 ---
            meas = {"out": fncl.cov_weight(s["ys"], s["z"][-1], pool=True)}
            for l in range(1, n_hidden):
                meas[l] = fncl.cov_weight(s["d"][l], s["z"][l - 1], pool=True)
            if not W_ema:
                W_ema.update(meas)
            else:
                for k, v in meas.items():
                    W_ema[k] = jac_ema * W_ema[k] + (1.0 - jac_ema) * v
            gW, gb = manual_grads(net, x, labels, W_ema, s)
            steps = {}
            for l in range(n_hidden):
                steps[l] = optim.update(f"w{l}", net.fcs[l].weight, gW[f"w{l}"], lr)
                if net.fcs[l].bias is not None:
                    optim.update(f"b{l}", net.fcs[l].bias, gb[f"w{l}"], lr)
            steps["out"] = optim.update("wout", net.fcs[-1].weight, gW["wout"], lr)
            if net.fcs[-1].bias is not None:
                optim.update("bout", net.fcs[-1].bias, gb["wout"], lr)
            # --- Kolen-Pollack 追従: 既知の更新量をミラーへ直接加算 ---
            W_ema["out"] = W_ema["out"] - steps["out"]
            for l in range(1, n_hidden):
                W_ema[l] = W_ema[l] - steps[l]

        if eval_every and (epoch % eval_every == 0 or epoch == epochs - 1):
            ce, acc, te_acc = evaluate(net, x, labels, *test)
            mirror_r = {f"w{l}": pearson(W_ema[l], net.fcs[l].weight)
                        for l in range(1, n_hidden)}
            mirror_r["wout"] = pearson(W_ema["out"], net.fcs[-1].weight)
            # 学習時ミラーで作る更新方向 vs autograd 厳密勾配
            g_auto = averaged(lambda: autograd_gradient(net, x, labels), grad_draws)

            def one_draw():
                with torch.no_grad():
                    ss = collect_samples(net, x, passes=1)
                    return manual_grads(net, x, labels, W_ema, ss)[0]

            g_est = averaged(one_draw, grad_draws)
            fid = {k: {"cos": cosine(g_est[k], g_auto[k]),
                       "ratio": norm_ratio(g_est[k], g_auto[k])} for k in g_auto}
            history.append({"epoch": epoch, "ce": ce, "acc": acc,
                            "test_acc": te_acc, "mirror_r": mirror_r,
                            "fidelity": fid})
            log(f"  [cov_jac ep {epoch:4d}] CE={ce:.4f} acc={acc:.3f} "
                f"test={te_acc:.3f} | mirror "
                + " ".join(f"{k}={v:.3f}" for k, v in mirror_r.items())
                + " | cos " + " ".join(f"{k}={v['cos']:.3f}"
                                       for k, v in fid.items()))
    return history


def train_backprop_ce(net, x, labels, lr: float, epochs: int, eval_every: int, log,
                      test=(None, None)):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    history = []
    for epoch in range(epochs):
        opt.zero_grad()
        loss = tF.cross_entropy(net(x), labels)
        loss.backward()
        opt.step()
        if eval_every and (epoch % eval_every == 0 or epoch == epochs - 1):
            ce, acc, te_acc = evaluate(net, x, labels, *test)
            history.append({"epoch": epoch, "ce": ce, "acc": acc,
                            "test_acc": te_acc})
            log(f"  [backprop ep {epoch:4d}] CE={ce:.4f} acc={acc:.3f} "
                f"test={te_acc:.3f}")
    return history


def main() -> None:
    p = argparse.ArgumentParser(description="A1c: cov_jac actually training a deep "
                                            "MNIST NNN, with training-time mirror.")
    p.add_argument("--depths", type=str, default="2,4")
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--n-inputs", type=int, default=256)
    p.add_argument("--n-test", type=int, default=0,
                   help=">0 で学習集合と素な MNIST テスト集合を作り汎化を測る")
    p.add_argument("--num-samples", type=int, default=64)
    p.add_argument("--sigma", type=float, default=0.5)
    p.add_argument("--crossing-h", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--jac-ema", type=float, default=0.9)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--grad-draws", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out", type=str, default="tmp/out/fncl_a1c")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.depths, args.width, args.n_inputs = "2", 32, 64
        args.num_samples, args.epochs = 16, 20
        args.eval_every, args.grad_draws = 10, 2
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # 学習集合とテスト集合は同一の class-balanced 抽出から前半／後半に分割する
    # (load_mnist は seed 固定なので、n_inputs + n_test 枚を取れば素な分割になる)。
    x_all, y_all = load_mnist(args.n_inputs + args.n_test)
    perm = np.random.default_rng(999).permutation(len(y_all))
    tr, te = perm[:args.n_inputs], perm[args.n_inputs:]
    x = torch.tensor(x_all[tr], device=device)
    labels = torch.tensor(y_all[tr], device=device)
    if args.n_test > 0:
        test = (torch.tensor(x_all[te], device=device),
                torch.tensor(y_all[te], device=device))
        print(f"train {x.shape[0]} / test {test[0].shape[0]}", flush=True)
    else:
        test = (None, None)
    log = lambda m: print(m, flush=True)  # noqa: E731

    results = {"config": vars(args), "runs": {}}
    for depth in [int(d) for d in args.depths.split(",")]:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        net0 = build_net(depth, args.width, args.sigma, args.crossing_h,
                         args.num_samples, device)
        init_state = {k: v.clone() for k, v in net0.state_dict().items()}
        log(f"=== depth {depth} : cov_jac ===")
        torch.manual_seed(args.seed)
        net = build_net(depth, args.width, args.sigma, args.crossing_h,
                        args.num_samples, device)
        net.load_state_dict(init_state)
        h_jac = train_cov_jac_ce(net, x, labels, args.lr, args.epochs,
                                 args.jac_ema, args.eval_every, args.grad_draws,
                                 log, test=test)
        log(f"=== depth {depth} : backprop ===")
        torch.manual_seed(args.seed)
        net_bp = build_net(depth, args.width, args.sigma, args.crossing_h,
                           args.num_samples, device)
        net_bp.load_state_dict(init_state)
        h_bp = train_backprop_ce(net_bp, x, labels, args.lr, args.epochs,
                                 args.eval_every, log, test=test)
        results["runs"][str(depth)] = {"cov_jac": h_jac, "backprop": h_bp}

    lines = ["| depth | epoch | cov_jac CE (train/test acc) | "
             "backprop CE (train/test acc) | mirror r (min) | cov_jac cosine (層別) |",
             "|---|---|---|---|---|---|"]
    for depth, run in results["runs"].items():
        bp = {h["epoch"]: h for h in run["backprop"]}
        for h in run["cov_jac"]:
            b = bp.get(h["epoch"], {})
            cos_s = " ".join(f"{k}={v['cos']:.3f}" for k, v in h["fidelity"].items())
            lines.append(
                f"| {depth} | {h['epoch']} | {h['ce']:.4f} "
                f"({h['acc']:.3f}/{h.get('test_acc', float('nan')):.3f}) | "
                f"{b.get('ce', float('nan')):.4f} "
                f"({b.get('acc', float('nan')):.3f}/"
                f"{b.get('test_acc', float('nan')):.3f}) | "
                f"{min(h['mirror_r'].values()):.4f} | {cos_s} |")
    print("\n" + "\n".join(lines) + "\n", flush=True)
    save_json(out_dir / "results.json", results)


if __name__ == "__main__":
    main()
