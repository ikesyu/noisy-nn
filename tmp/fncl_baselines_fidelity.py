"""
fncl_baselines_fidelity.py — A2 の決め手: FA / DFA の勾配忠実度を深層 MNIST 上で cov_jac と直接比較
（旧名: fncl_a2b.py．出力ディレクトリ tmp/out/ は旧名のまま）

fncl_baselines.py の結果、トイタスクの最終 MSE では FA/DFA (Adam) は cov_jac に肉薄する
(sin: dfa 0.00086 vs cov_jac 0.00056)。したがって「性能表」では本手法の位置づけを
示せない。本スクリプトは、両者の決定的な違い —— cov_jac は真の勾配を再構成するが
FA/DFA は固定ランダム行列であって真の勾配に収束しない —— を、A1 と同じ
784-256-...-10 / 交差エントロピー上で層別 cosine として直接測る。

推定量 (すべて同一 forward サンプル・同一の KDE 局所傾き・同一の読み出し誤差を使い、
隠れ層 credit の作り方だけが異なる):
  cov_jac : 共分散ミラー W_hat の再帰                (§4.3)
  fa      : 固定ランダム B の再帰                    (Lillicrap+ 2016)
  dfa     : 出力誤差を固定ランダム B_l で直接投影    (Nøkland 2016)

生成物 (tmp/out/fncl_a2b/): results.json, (標準出力) 表

実行例:
  .venv/bin/python tmp/fncl_baselines_fidelity.py --quick
  .venv/bin/python tmp/fncl_baselines_fidelity.py --depths 2,4
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as tF

DATA_DIR = Path(__file__).resolve().parent.parent / "data_nce"
sys.path.insert(0, str(DATA_DIR))
import fncl  # noqa: E402
from fncl_common import cosine, norm_ratio, save_json  # noqa: E402
from fncl_mnist_fidelity import (autograd_gradient, averaged, build_net, ce_error,  # noqa: E402
                     collect_samples, load_mnist)


def random_feedback(net, depth: int, method: str, width: int, device):
    """FA/DFA の固定ランダムフィードバック行列 (前向き重みと同じ一様スケール)."""
    B = {}
    if method == "fa":
        for l in range(1, depth):
            B[l] = ((torch.rand_like(net.fcs[l].weight) * 2 - 1)
                    / math.sqrt(net.fcs[l].weight.shape[1]))
        B["out"] = ((torch.rand_like(net.fcs[-1].weight) * 2 - 1)
                    / math.sqrt(net.fcs[-1].weight.shape[1]))
    else:  # dfa: [10, H_l] を各隠れ層に
        for l in range(depth):
            B[l] = ((torch.rand(10, width, device=device) * 2 - 1)
                    / math.sqrt(width))
    return B


def grads_from(net, x, labels, s, method: str, W=None, B=None) -> dict:
    """同一サンプル s 上で、隠れ層 credit の作り方だけを変えた勾配推定を返す."""
    crossings = fncl.crossing_layers(net)
    n_hidden = len(crossings)
    slope_full = [fncl.kde_slope(crossings[l], s["d"][l]) for l in range(n_hidden)]
    slope_mean = [sf.mean(dim=1) for sf in slope_full]
    N, T = x.shape[0], s["z"][0].shape[1]
    e = ce_error(s["y"], labels)                                   # [N, 10]
    a = [None] * n_hidden
    if method == "dfa":
        for l in range(n_hidden):
            a[l] = e @ B[l]
    else:
        fb = W if method == "cov_jac" else B
        a[-1] = e @ fb["out"]
        for l in range(n_hidden - 2, -1, -1):
            a[l] = (a[l + 1] * slope_mean[l + 1]) @ fb[l + 1]
    z_prev = ([x.unsqueeze(1).expand(N, T, x.shape[1])]
              + [s["z"][l] for l in range(n_hidden - 1)])
    grads = {}
    for l in range(n_hidden):
        delta = a[l].unsqueeze(1) * slope_full[l]
        grads[f"w{l}"] = torch.einsum("nto,nti->oi", delta, z_prev[l]) / (N * T)
    grads["wout"] = torch.einsum("no,ni->oi", e, s["z"][-1].mean(dim=1)) / N
    return grads


def main() -> None:
    p = argparse.ArgumentParser(description="A2b: gradient fidelity of FA/DFA vs "
                                            "cov_jac on the deep MNIST NNN.")
    p.add_argument("--depths", type=str, default="2,4")
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--n-inputs", type=int, default=256)
    p.add_argument("--num-samples", type=int, default=64)
    p.add_argument("--sigma", type=float, default=0.5)
    p.add_argument("--crossing-h", type=float, default=0.2)
    p.add_argument("--grad-draws", type=int, default=32)
    p.add_argument("--pretrain-epochs", type=int, default=300)
    p.add_argument("--pretrain-lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out", type=str, default="tmp/out/fncl_a2b")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.depths, args.width, args.n_inputs = "2", 32, 64
        args.num_samples, args.grad_draws, args.pretrain_epochs = 16, 4, 20
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    x_np, y_np = load_mnist(args.n_inputs)
    x = torch.tensor(x_np, device=device)
    labels = torch.tensor(y_np, device=device)

    results = {"config": vars(args), "runs": {}}
    for depth in [int(d) for d in args.depths.split(",")]:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        net0 = build_net(depth, args.width, args.sigma, args.crossing_h,
                         args.num_samples, device)
        init_state = {k: v.clone() for k, v in net0.state_dict().items()}
        run = {}
        for state in ("untrained", "pretrained"):
            net = build_net(depth, args.width, args.sigma, args.crossing_h,
                            args.num_samples, device)
            net.load_state_dict(init_state)
            if state == "pretrained":
                opt = torch.optim.Adam(net.parameters(), lr=args.pretrain_lr)
                for _ in range(args.pretrain_epochs):
                    opt.zero_grad()
                    tF.cross_entropy(net(x), labels).backward()
                    opt.step()
                with torch.no_grad():
                    lg = net(x)
                    print(f"[depth {depth}] pretrained CE="
                          f"{float(tF.cross_entropy(lg, labels)):.4f} acc="
                          f"{float((lg.argmax(1) == labels).float().mean()):.3f}",
                          flush=True)
            B_fa = random_feedback(net, depth, "fa", args.width, device)
            B_dfa = random_feedback(net, depth, "dfa", args.width, device)

            def draw():
                with torch.no_grad():
                    s = collect_samples(net, x, passes=1)
                    W = {"out": fncl.cov_weight(s["ys"], s["z"][-1], pool=True)}
                    for l in range(1, depth):
                        W[l] = fncl.cov_weight(s["d"][l], s["z"][l - 1], pool=True)
                    out = {}
                    for tag, kw in (("cov_jac", {"W": W}), ("fa", {"B": B_fa}),
                                    ("dfa", {"B": B_dfa})):
                        for k, v in grads_from(net, x, labels, s, tag, **kw).items():
                            out[f"{tag}|{k}"] = v
                    return out

            g_auto = averaged(lambda: autograd_gradient(net, x, labels),
                              args.grad_draws)
            g_est = averaged(draw, args.grad_draws)
            entry = {}
            for tag in ("cov_jac", "fa", "dfa"):
                fid = {k: {"cos": cosine(g_est[f"{tag}|{k}"], g_auto[k]),
                           "ratio": norm_ratio(g_est[f"{tag}|{k}"], g_auto[k])}
                       for k in g_auto}
                entry[tag] = fid
                print(f"[depth {depth} | {state}] {tag:8s} "
                      + "  ".join(f"{k}:{v['cos']:.3f}" for k, v in fid.items()),
                      flush=True)
            run[state] = entry
        results["runs"][str(depth)] = run

    lines = ["| depth | state | layer | cov_jac cos | fa cos | dfa cos |",
             "|---|---|---|---|---|---|"]
    for depth, run in results["runs"].items():
        for state, entry in run.items():
            for k in entry["cov_jac"]:
                lines.append(f"| {depth} | {state} | {k} | "
                             f"{entry['cov_jac'][k]['cos']:.4f} | "
                             f"{entry['fa'][k]['cos']:.4f} | "
                             f"{entry['dfa'][k]['cos']:.4f} |")
    print("\n" + "\n".join(lines) + "\n", flush=True)
    save_json(out_dir / "results.json", results)


if __name__ == "__main__":
    main()
