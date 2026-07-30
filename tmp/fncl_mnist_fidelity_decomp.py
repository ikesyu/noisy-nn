"""
fncl_mnist_fidelity_decomp.py — A1 の追試: 深層で cosine が落ちる原因の切り分け
（旧名: fncl_a1b.py．出力ディレクトリ tmp/out/ は旧名のまま）

fncl_mnist_fidelity.py は depth 4 の事前学習後で cov_jac の cosine が 0.65–0.73 に低下した
(未学習では 0.95–0.98)。低下要因は原理的に 2 つしかない:

  (1) weight mirror の誤差  … 単変量回帰の対角近似バイアス (draft_nce §7 の open problem)
  (2) 推定の分散           … 1 forward からミラーを測る fncl5_3 プロトコル固有の有限
                              サンプル誤差。実際の学習 (train_cov) はミラーを EMA(0.9)
                              ＋Kolen-Pollack 追従で平滑化するので、この分散は学習時には
                              大幅に小さい。

autograd の厳密勾配は W の転置 × 交差活性の内部 KDE 係数 (CrossingSample.backward の
coeff = mean_t(xor2-xor1)/2h) であり、cov_jac の局所傾き kde_slope と同一の推定量である。
よって「同じ forward サンプル上で W_hat だけを真の W に差し替えた」勾配を作れば、
(1) と (2) を完全に分離できる。本スクリプトは 3 変種を同一 draw 上で比較する:

  cov_jac      : ミラーを 1 forward から測る (fncl_mnist_fidelity / fncl5_3 と同一)
  cov_jac_mp<K>: ミラーを K forward から測る (学習時の EMA 平滑化に相当)
  exactW       : 再帰に真の W を使う (ミラー誤差ゼロ; 残差 = 純粋な推定分散)

さらに、事前学習 epoch を掃引して「勾配が消えた過収束状態か否か」の影響も測る
(fncl_mnist_fidelity の pretrained は 250 枚に対し acc 1.000 / CE 0.009 の過収束状態だった)。

生成物 (tmp/out/fncl_a1b/): results.json, (標準出力) 表

実行例:
  .venv/bin/python tmp/fncl_mnist_fidelity_decomp.py --quick
  .venv/bin/python tmp/fncl_mnist_fidelity_decomp.py --depths 4 --checkpoints 0,30,100,300
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
from fncl_mnist_fidelity import (build_net, ce_error, collect_samples, load_mnist,  # noqa: E402
                     autograd_gradient, averaged)


def measure_mirror(net, x, passes: int) -> dict:
    """K forward パスからミラーを測る (K が大きいほど学習時の EMA 平滑化に近い)."""
    s = collect_samples(net, x, passes)
    n_hidden = len(fncl.crossing_layers(net))
    w_hat = {"out": fncl.cov_weight(s["ys"], s["z"][-1], pool=True)}
    for l in range(1, n_hidden):
        w_hat[l] = fncl.cov_weight(s["d"][l], s["z"][l - 1], pool=True)
    return w_hat


def grads_from_samples(net, x, labels, s, w_hat) -> dict:
    """与えられた forward サンプル s と (真 or 推定) 重み w_hat で cov_jac 勾配を作る."""
    crossings = fncl.crossing_layers(net)
    n_hidden = len(crossings)
    slope_full = [fncl.kde_slope(crossings[l], s["d"][l]) for l in range(n_hidden)]
    slope_mean = [sf.mean(dim=1) for sf in slope_full]
    N, T = x.shape[0], s["z"][0].shape[1]
    e = ce_error(s["y"], labels)
    a = [None] * n_hidden
    a[-1] = e @ w_hat["out"]
    for l in range(n_hidden - 2, -1, -1):
        a[l] = (a[l + 1] * slope_mean[l + 1]) @ w_hat[l + 1]
    z_prev = ([x.unsqueeze(1).expand(N, T, x.shape[1])]
              + [s["z"][l] for l in range(n_hidden - 1)])
    grads = {}
    for l in range(n_hidden):
        delta = a[l].unsqueeze(1) * slope_full[l]
        grads[f"w{l}"] = torch.einsum("nto,nti->oi", delta, z_prev[l]) / (N * T)
    grads["wout"] = torch.einsum("no,ni->oi", e, s["z"][-1].mean(dim=1)) / N
    return grads


def true_weights(net) -> dict:
    n_hidden = len(fncl.crossing_layers(net))
    w = {"out": net.fcs[-1].weight.detach()}
    for l in range(1, n_hidden):
        w[l] = net.fcs[l].weight.detach()
    return w


def main() -> None:
    p = argparse.ArgumentParser(description="A1b: decompose the deep-net cosine drop "
                                            "into mirror error vs estimator variance.")
    p.add_argument("--depths", type=str, default="2,4")
    p.add_argument("--checkpoints", type=str, default="0,30,100,300",
                   help="backprop 事前学習 epoch のチェックポイント")
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--n-inputs", type=int, default=256)
    p.add_argument("--num-samples", type=int, default=64)
    p.add_argument("--sigma", type=float, default=0.5)
    p.add_argument("--crossing-h", type=float, default=0.2)
    p.add_argument("--mirror-passes", type=int, default=16,
                   help="cov_jac_mp<K> のミラー測定パス数")
    p.add_argument("--grad-draws", type=int, default=24)
    p.add_argument("--pretrain-lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out", type=str, default="tmp/out/fncl_a1b")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.depths, args.checkpoints, args.width = "2", "0,10", 32
        args.n_inputs, args.num_samples = 64, 16
        args.mirror_passes, args.grad_draws = 4, 4
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    x_np, y_np = load_mnist(args.n_inputs)
    x = torch.tensor(x_np, device=device)
    labels = torch.tensor(y_np, device=device)
    checkpoints = [int(c) for c in args.checkpoints.split(",")]
    mp = args.mirror_passes

    results = {"config": vars(args), "runs": {}}
    for depth in [int(d) for d in args.depths.split(",")]:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        net = build_net(depth, args.width, args.sigma, args.crossing_h,
                        args.num_samples, device)
        opt = torch.optim.Adam(net.parameters(), lr=args.pretrain_lr)
        run = {}
        done = 0
        for ck in checkpoints:
            while done < ck:                            # 逐次 backprop 事前学習
                opt.zero_grad()
                loss = tF.cross_entropy(net(x), labels)
                loss.backward()
                opt.step()
                done += 1
            with torch.no_grad():
                logits = net(x)
                ce = float(tF.cross_entropy(logits, labels))
                acc = float((logits.argmax(1) == labels).float().mean())
            print(f"[depth {depth} | ep {ck}] CE={ce:.4f} acc={acc:.3f}", flush=True)

            # ミラー: 1 パス版 (fncl5_3 プロトコル) は draw ごとに測り直す。
            # mp パス版は学習時の EMA 平滑化に相当するので 1 度だけ測って固定する。
            w_mp = measure_mirror(net, x, mp)
            w_true = true_weights(net)
            mirror_r = {f"w{l}": pearson(w_mp[l], net.fcs[l].weight)
                        for l in range(1, depth)}
            mirror_r["wout"] = pearson(w_mp["out"], net.fcs[-1].weight)

            def draw():
                s = collect_samples(net, x, passes=1)
                w1 = {"out": fncl.cov_weight(s["ys"], s["z"][-1], pool=True)}
                for l in range(1, depth):
                    w1[l] = fncl.cov_weight(s["d"][l], s["z"][l - 1], pool=True)
                out = {}
                for tag, w in (("cov_jac", w1), (f"cov_jac_mp{mp}", w_mp),
                               ("exactW", w_true)):
                    for k, v in grads_from_samples(net, x, labels, s, w).items():
                        out[f"{tag}|{k}"] = v
                return out

            g_auto = averaged(lambda: autograd_gradient(net, x, labels),
                              args.grad_draws)
            g_est = averaged(draw, args.grad_draws)
            entry = {"ce": ce, "acc": acc, "mirror_r": mirror_r, "fidelity": {}}
            for tag in ("cov_jac", f"cov_jac_mp{mp}", "exactW"):
                fid = {k: {"cos": cosine(g_est[f"{tag}|{k}"], g_auto[k]),
                           "ratio": norm_ratio(g_est[f"{tag}|{k}"], g_auto[k])}
                       for k in g_auto}
                entry["fidelity"][tag] = fid
                print(f"[depth {depth} | ep {ck}] {tag:14s} "
                      + "  ".join(f"{k}:{v['cos']:.3f}" for k, v in fid.items()),
                      flush=True)
            run[str(ck)] = entry
        results["runs"][str(depth)] = run

    lines = ["| depth | pretrain ep | CE | layer | mirror r (mp) | cov_jac | "
             f"cov_jac_mp{mp} | exactW |", "|---|---|---|---|---|---|---|---|"]
    for depth, run in results["runs"].items():
        for ck, entry in run.items():
            for k in entry["fidelity"]["exactW"]:
                mr = entry["mirror_r"].get(k)
                lines.append(
                    f"| {depth} | {ck} | {entry['ce']:.4f} | {k} | "
                    f"{'-' if mr is None else f'{mr:.4f}'} | "
                    f"{entry['fidelity']['cov_jac'][k]['cos']:.4f} | "
                    f"{entry['fidelity'][f'cov_jac_mp{mp}'][k]['cos']:.4f} | "
                    f"{entry['fidelity']['exactW'][k]['cos']:.4f} |")
    print("\n" + "\n".join(lines) + "\n", flush=True)
    save_json(out_dir / "results.json", results)


if __name__ == "__main__":
    main()
