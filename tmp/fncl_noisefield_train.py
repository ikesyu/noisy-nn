"""
fncl_noisefield_train.py — S2: 周期ノイズ場（L3）を使った本番学習（docs/idea_ca.md §5.6）
（旧名: fncl_nf2.py．出力ディレクトリ tmp/out/ は旧名のまま）

S0/S1 の結論:
  - 層内相関の対角近似バイアスは，安価な代数的近似（ブロック対角 §2.6，
    ランク 1 Woodbury §7）では**消えない**。
  - 一方，上流層のノイズを周期化して位相内で共分散を取る（L3）と，
    ミラー r は 0.963 -> 0.996 と完全に回復し，前向き CE は不変，
    コストはアンサンブル std の増加だけ（p=32 で +15%）。

本スクリプトは L3 を実学習に組み込む。1 パスで全ミラーを同時に測ることはできない
（層 l は自分のミラーには周期 1，より深いミラーには周期 p である必要があり両立しない）
ので、**ローテーション方式**を採る:

  各 epoch で対象層 t を 1 つ選び、層 0..t-1 を周期 p、層 t 以降を周期 1 とする。
  その epoch は回帰子 z[t] のミラーだけを位相内中心化で更新する。
  他のミラーは Kolen-Pollack 追従（既知の更新量をそのまま加算）で追随する。

さらに `--mirror-every K` で「ミラー測定を K epoch に 1 度だけ行い、残りの epoch は
通常のノイズ場（全層 period=1）で走る」ことができる。KP 追従が動く標的を厳密に
追うので測定頻度は落とせるはずで、前向きコストをさらに下げられる。

比較基準は A1e と同一（depth 4, train 1000 / test 1000, 400 epoch, T=64, Adam 1e-3）:
  diag 0.815 / mv_all 0.871 / backprop 0.876（いずれも test accuracy）

生成物 (tmp/out/fncl_nf2/): results.json, (標準出力) 表

実行例:
  .venv/bin/python tmp/fncl_noisefield_train.py --quick
  .venv/bin/python tmp/fncl_noisefield_train.py --arms diag,per16,per16e4
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
from fncl_common import cosine, pearson, save_json  # noqa: E402
from fncl_mnist_fidelity import autograd_gradient, averaged, build_net, load_mnist  # noqa: E402
from fncl_mnist_train import evaluate, manual_grads  # noqa: E402
from fncl_noisefield_lib import (build_net_noise, collect, cov_weight_phase,  # noqa: E402
                     install_noise_field, restore_noise_field)


def mirror_key(t: int, n_hidden: int):
    """回帰子 z[t] に対応する W_ema のキー（manual_grads の規約に合わせる）。"""
    return "out" if t == n_hidden - 1 else t + 1


def mirror_target(net, s, t: int, n_hidden: int):
    """回帰子 z[t] に対する被回帰量（最終隠れ層なら読み出しサンプル）。"""
    return s["ys"] if t == n_hidden - 1 else s["d"][t + 1]


def true_mirror_weight(net, t: int, n_hidden: int):
    return net.fcs[-1].weight if t == n_hidden - 1 else net.fcs[t + 1].weight


def train_arm(net, x, labels, test, n_hidden: int, arm: str, args, log):
    """cov_jac 学習（CE 多出力）。arm が per<p>[e<K>] なら L3 ローテーションを使う。"""
    period, mirror_every = 1, 1
    if arm.startswith("per"):
        body = arm[3:]
        if "e" in body:
            a, b = body.split("e")
            period, mirror_every = int(a), int(b)
        else:
            period = int(body)

    optim = fncl.ManualOpt("adam")
    W_ema, history = {}, []
    for epoch in range(args.epochs):
        # --- この epoch のノイズ場を決める（ローテーション） ---
        measure = (epoch % mirror_every == 0)
        target = (epoch // mirror_every) % n_hidden if measure else None
        if period > 1 and measure and target > 0:
            periods = [period] * target + [1] * (n_hidden - target)
        else:
            periods = [1] * n_hidden
        install_noise_field(net, periods=periods)

        with torch.no_grad():
            s = collect(net, x, passes=1)
            T = s["z"][0].shape[1]
            # --- ミラー測定 ---
            if not W_ema:
                # 初回は全ミラーを通常測定で初期化（以後はローテーション＋KP）
                W_ema["out"] = fncl.cov_weight(s["ys"], s["z"][-1], pool=True)
                for l in range(1, n_hidden):
                    W_ema[l] = fncl.cov_weight(s["d"][l], s["z"][l - 1], pool=True)
            elif measure:
                tgt = mirror_target(net, s, target, n_hidden)
                if period > 1 and target > 0:
                    meas = cov_weight_phase(tgt, s["z"][target], period, pass_len=T)
                else:
                    meas = fncl.cov_weight(tgt, s["z"][target], pool=True)
                k = mirror_key(target, n_hidden)
                W_ema[k] = args.jac_ema * W_ema[k] + (1.0 - args.jac_ema) * meas

            gW, gb = manual_grads(net, x, labels, W_ema, s)
            steps = {}
            for l in range(n_hidden):
                steps[l] = optim.update(f"w{l}", net.fcs[l].weight, gW[f"w{l}"],
                                        args.lr)
                if net.fcs[l].bias is not None:
                    optim.update(f"b{l}", net.fcs[l].bias, gb[f"w{l}"], args.lr)
            steps["out"] = optim.update("wout", net.fcs[-1].weight, gW["wout"],
                                        args.lr)
            if net.fcs[-1].bias is not None:
                optim.update("bout", net.fcs[-1].bias, gb["wout"], args.lr)
            # Kolen-Pollack: 既知の更新量を全ミラーへ加算（測定しなかった層も追随する）
            W_ema["out"] = W_ema["out"] - steps["out"]
            for l in range(1, n_hidden):
                W_ema[l] = W_ema[l] - steps[l]
        restore_noise_field(net)

        if epoch % args.eval_every == 0 or epoch == args.epochs - 1:
            ce, acc, te = evaluate(net, x, labels, *test)
            mr = {f"z{t + 1}": pearson(W_ema[mirror_key(t, n_hidden)],
                                       true_mirror_weight(net, t, n_hidden))
                  for t in range(n_hidden)}
            g_auto = averaged(lambda: autograd_gradient(net, x, labels),
                              args.grad_draws)

            def one():
                with torch.no_grad():
                    ss = collect(net, x, passes=1)
                    return manual_grads(net, x, labels, W_ema, ss)[0]

            g_est = averaged(one, args.grad_draws)
            cos = {k: cosine(g_est[k], g_auto[k]) for k in g_auto}
            history.append({"epoch": epoch, "ce": ce, "acc": acc, "test_acc": te,
                            "mirror_r": mr, "cos": cos})
            log(f"  [{arm} ep {epoch:4d}] CE={ce:.4f} acc={acc:.3f} test={te:.3f}"
                " | mirror " + " ".join(f"{k}={v:.3f}" for k, v in mr.items())
                + " | cos " + " ".join(f"{k}={v:.3f}" for k, v in cos.items()))
    return history


def main() -> None:
    p = argparse.ArgumentParser(description="S2: training with the periodic "
                                            "noise-field (L3) rotation.")
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--arms", type=str, default="per16,per16e4")
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--n-inputs", type=int, default=1000)
    p.add_argument("--n-test", type=int, default=1000)
    p.add_argument("--num-samples", type=int, default=64)
    p.add_argument("--noise", choices=("gaussian", "uniform"), default="gaussian")
    p.add_argument("--sigma", type=float, default=0.5,
                   help="ガウスなら std，一様なら半幅 radius（--noise uniform 時は既定 1.0）")
    p.add_argument("--crossing-h", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--jac-ema", type=float, default=0.9)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--grad-draws", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out", type=str, default="tmp/out/fncl_nf2")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.depth, args.width = 2, 32
        args.n_inputs, args.n_test = 64, 40
        args.num_samples, args.epochs = 16, 20
        args.eval_every, args.grad_draws = 10, 2
        args.arms = "diag,per8,per8e4"
    if args.noise == "uniform" and args.sigma == 0.5:
        args.sigma = 1.0                      # 一様ノイズの既定半幅（原稿 §6.2）
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    x_all, y_all = load_mnist(args.n_inputs + args.n_test)
    perm = np.random.default_rng(999).permutation(len(y_all))
    tr, te = perm[:args.n_inputs], perm[args.n_inputs:]
    x = torch.tensor(x_all[tr], device=device)
    labels = torch.tensor(y_all[tr], device=device)
    test = (torch.tensor(x_all[te], device=device),
            torch.tensor(y_all[te], device=device))
    print(f"train {x.shape[0]} / test {test[0].shape[0]}, depth {args.depth}",
          flush=True)
    log = lambda m: print(m, flush=True)  # noqa: E731

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    net0 = build_net_noise(args.noise, args.depth, args.width, args.sigma,
                           args.crossing_h, args.num_samples, device)
    init_state = {k: v.clone() for k, v in net0.state_dict().items()}

    results = {"config": vars(args), "runs": {}}
    for arm in args.arms.split(","):
        log(f"=== depth {args.depth} : {arm} ===")
        torch.manual_seed(args.seed)
        net = build_net_noise(args.noise, args.depth, args.width, args.sigma,
                              args.crossing_h, args.num_samples, device)
        net.load_state_dict(init_state)
        results["runs"][arm] = train_arm(net, x, labels, test, args.depth, arm,
                                         args, log)

    lines = ["| arm | epoch | CE | train acc | test acc | mirror r (min) | "
             "cos (最上位隠れ層) |", "|---|---|---|---|---|---|---|"]
    for arm, hist in results["runs"].items():
        for h in hist:
            top = f"w{args.depth - 1}"
            lines.append(f"| {arm} | {h['epoch']} | {h['ce']:.4f} | "
                         f"{h['acc']:.3f} | {h['test_acc']:.3f} | "
                         f"{min(h['mirror_r'].values()):.4f} | "
                         f"{h['cos'][top]:.3f} |")
    print("\n" + "\n".join(lines) + "\n", flush=True)
    save_json(out_dir / "results.json", results)


if __name__ == "__main__":
    main()
