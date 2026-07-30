"""
fncl_mnist_mirror_variants.py — A1' の答え: 多変量ミラーで深層の劣化を直せるか
（旧名: fncl_a1e.py．出力ディレクトリ tmp/out/ は旧名のまま）

実験 A1d (tmp/out/fncl_a1d, train 1000 / test 1000) で、depth 4 の cov_jac は test acc 0.813 と
backprop 0.876 に 6 点差をつけられた。層別に見ると劣化の所在は明確で、

  読み出しミラー r: 0.998 -> 0.890 (単調劣化)  →  そこから credit を受ける
  最上位隠れ層 w3 の cosine: 0.918 -> 0.243

一方、隠れ層間ミラー (w1,w2,w3) は 0.95-0.98 を保つ。すなわち劣化は「最終隠れ層の
活動の層内相関が強く、単変量回帰 Cov(d,z)/Var(z) が近傍重みを漏れ込ませる」
という draft_nce §4.3/§6.1 が予告した対角近似バイアスであり、読み出し層に集中している。

原稿 §6.1 は厳密な多変量回帰 Cov(d,z) Cov(z,z)^{-1} が O(H^3) で局所性を損なうと
して future work にしているが、**読み出し層 1 枚だけ**なら [H,H] の逆行列 1 回で済む。
本スクリプトはこれを実装して比較する:

  diag   : 全層とも単変量 (現行 cov_jac)
  mv_out : 読み出しミラーのみ多変量、隠れ層間は単変量 (提案する中間設計)
  mv_all : 全ミラーを多変量 (上限の参照値)

生成物 (tmp/out/fncl_a1e/): results.json, (標準出力) 表

実行例:
  .venv/bin/python tmp/fncl_mnist_mirror_variants.py --quick
  .venv/bin/python tmp/fncl_mnist_mirror_variants.py --depth 4 --epochs 400 --n-inputs 1000 --n-test 1000
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
from fncl_mnist_fidelity import (autograd_gradient, averaged, build_net,  # noqa: E402
                     collect_samples, load_mnist)
from fncl_mnist_train import evaluate, manual_grads  # noqa: E402


def cov_weight_mv(d_next, z_prev, ridge: float = 1e-3):
    """多変量ミラー: W_hat = [sum_n Cov_T(d,z)] [sum_n Cov_T(z,z) + lambda I]^{-1}.

    単変量版 (fncl.cov_weight) は分母に Var(z_i) しか使わないため、層内の活動相関
    Cov(z_k, z_i) を通じて近傍重み W_jk が漏れ込む (draft_nce §4.3 末尾)。
    ここでは層内共分散行列そのもので割ることでその漏れ込みを除く。
    入力ごとに中心化してから全入力で和を取る点は単変量版と同じ (pool=True 相当)。
    """
    cd = d_next - d_next.mean(dim=1, keepdim=True)              # [N, T, Ho]
    cz = z_prev - z_prev.mean(dim=1, keepdim=True)              # [N, T, Hi]
    T = d_next.shape[1]
    cov_dz = torch.einsum("nto,nti->oi", cd, cz) / T            # [Ho, Hi]
    cov_zz = torch.einsum("nti,ntj->ij", cz, cz) / T            # [Hi, Hi]
    n = d_next.shape[0]
    cov_dz, cov_zz = cov_dz / n, cov_zz / n
    lam = ridge * float(torch.diagonal(cov_zz).mean())
    A = cov_zz + lam * torch.eye(cov_zz.shape[0], device=cov_zz.device)
    return torch.linalg.solve(A, cov_dz.T).T                    # [Ho, Hi]


def cov_weight_block(d_next, z_prev, block: int, ridge: float = 1e-3):
    """ブロック対角ミラー: Cov(z,z) を b x b の対角ブロックだけ逆にする中間設計.

    draft_nce §6.1 が「対角と完全多変量の間の中間設計」として挙げたもの。
    計算量は O(H b^2) で、b=1 なら単変量、b=H なら完全多変量に一致する。
    ユニットを b 個ずつの連続ブロックに切り、各ブロック内でのみ相関を補正する。
    """
    cd = d_next - d_next.mean(dim=1, keepdim=True)
    cz = z_prev - z_prev.mean(dim=1, keepdim=True)
    T, n = d_next.shape[1], d_next.shape[0]
    cov_dz = torch.einsum("nto,nti->oi", cd, cz) / (T * n)      # [Ho, Hi]
    cov_zz = torch.einsum("nti,ntj->ij", cz, cz) / (T * n)      # [Hi, Hi]
    Hi = cov_zz.shape[0]
    lam = ridge * float(torch.diagonal(cov_zz).mean())
    W = torch.zeros_like(cov_dz)
    for b0 in range(0, Hi, block):
        b1 = min(b0 + block, Hi)
        A = cov_zz[b0:b1, b0:b1] + lam * torch.eye(b1 - b0, device=cov_zz.device)
        W[:, b0:b1] = torch.linalg.solve(A, cov_dz[:, b0:b1].T).T
    return W


def _mirror(d_next, z_prev, kind: str, block: int, ridge: float):
    if kind == "mv":
        return cov_weight_mv(d_next, z_prev, ridge)
    if kind == "blk":
        return cov_weight_block(d_next, z_prev, block, ridge)
    return fncl.cov_weight(d_next, z_prev, pool=True)


def measure(net, s, depth: int, mode: str, ridge: float) -> dict:
    """今 epoch の共分散ミラーを mode に従って測る.

    mode: diag / mv_out / mv_all / blk<b>  (blk16 なら全層ブロック対角, b=16)
    """
    block = 0
    if mode.startswith("blk"):
        block = int(mode[3:])
        out_kind = hidden_kind = "blk"
    else:
        out_kind = "mv" if mode in ("mv_out", "mv_all") else "diag"
        hidden_kind = "mv" if mode == "mv_all" else "diag"
    meas = {"out": _mirror(s["ys"], s["z"][-1], out_kind, block, ridge)}
    for l in range(1, depth):
        meas[l] = _mirror(s["d"][l], s["z"][l - 1], hidden_kind, block, ridge)
    return meas


def train(net, x, labels, test, depth, mode, args, log):
    optim = fncl.ManualOpt("adam")
    W_ema, history = {}, []
    for epoch in range(args.epochs):
        with torch.no_grad():
            s = collect_samples(net, x, passes=1)
            meas = measure(net, s, depth, mode, args.ridge)
            if not W_ema:
                W_ema.update(meas)
            else:
                for k, v in meas.items():
                    W_ema[k] = args.jac_ema * W_ema[k] + (1.0 - args.jac_ema) * v
            gW, gb = manual_grads(net, x, labels, W_ema, s)
            steps = {}
            for l in range(depth):
                steps[l] = optim.update(f"w{l}", net.fcs[l].weight, gW[f"w{l}"],
                                        args.lr)
                if net.fcs[l].bias is not None:
                    optim.update(f"b{l}", net.fcs[l].bias, gb[f"w{l}"], args.lr)
            steps["out"] = optim.update("wout", net.fcs[-1].weight, gW["wout"],
                                        args.lr)
            if net.fcs[-1].bias is not None:
                optim.update("bout", net.fcs[-1].bias, gb["wout"], args.lr)
            W_ema["out"] = W_ema["out"] - steps["out"]
            for l in range(1, depth):
                W_ema[l] = W_ema[l] - steps[l]

        if epoch % args.eval_every == 0 or epoch == args.epochs - 1:
            ce, acc, te = evaluate(net, x, labels, *test)
            mr = {f"w{l}": pearson(W_ema[l], net.fcs[l].weight)
                  for l in range(1, depth)}
            mr["wout"] = pearson(W_ema["out"], net.fcs[-1].weight)
            g_auto = averaged(lambda: autograd_gradient(net, x, labels),
                              args.grad_draws)

            def one():
                with torch.no_grad():
                    ss = collect_samples(net, x, passes=1)
                    return manual_grads(net, x, labels, W_ema, ss)[0]

            g_est = averaged(one, args.grad_draws)
            fid = {k: cosine(g_est[k], g_auto[k]) for k in g_auto}
            history.append({"epoch": epoch, "ce": ce, "acc": acc, "test_acc": te,
                            "mirror_r": mr, "cos": fid})
            log(f"  [{mode} ep {epoch:4d}] CE={ce:.4f} acc={acc:.3f} test={te:.3f}"
                f" | mirror " + " ".join(f"{k}={v:.3f}" for k, v in mr.items())
                + " | cos " + " ".join(f"{k}={v:.3f}" for k, v in fid.items()))
    return history


def main() -> None:
    p = argparse.ArgumentParser(description="A1e: multivariate mirror fixes the "
                                            "deep-net readout degradation?")
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--modes", type=str, default="diag,mv_out,mv_all")
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--n-inputs", type=int, default=1000)
    p.add_argument("--n-test", type=int, default=1000)
    p.add_argument("--num-samples", type=int, default=64)
    p.add_argument("--sigma", type=float, default=0.5)
    p.add_argument("--crossing-h", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--jac-ema", type=float, default=0.9)
    p.add_argument("--ridge", type=float, default=1e-3,
                   help="多変量ミラーの Tikhonov 正則化 (Cov(z,z) の平均対角比)")
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--grad-draws", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out", type=str, default="tmp/out/fncl_a1e")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.depth, args.width, args.n_inputs, args.n_test = 2, 32, 64, 40
        args.num_samples, args.epochs = 16, 20
        args.eval_every, args.grad_draws = 10, 2
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
    net0 = build_net(args.depth, args.width, args.sigma, args.crossing_h,
                     args.num_samples, device)
    init_state = {k: v.clone() for k, v in net0.state_dict().items()}

    results = {"config": vars(args), "runs": {}}
    for mode in args.modes.split(","):
        log(f"=== depth {args.depth} : {mode} ===")
        torch.manual_seed(args.seed)
        net = build_net(args.depth, args.width, args.sigma, args.crossing_h,
                        args.num_samples, device)
        net.load_state_dict(init_state)
        results["runs"][mode] = train(net, x, labels, test, args.depth, mode,
                                      args, log)

    lines = ["| mode | epoch | CE | train acc | test acc | mirror r (out) | "
             "cos (最上位隠れ層) |", "|---|---|---|---|---|---|---|"]
    for mode, hist in results["runs"].items():
        for h in hist:
            top = f"w{args.depth - 1}"
            lines.append(f"| {mode} | {h['epoch']} | {h['ce']:.4f} | "
                         f"{h['acc']:.3f} | {h['test_acc']:.3f} | "
                         f"{h['mirror_r']['wout']:.4f} | {h['cos'][top]:.3f} |")
    print("\n" + "\n".join(lines) + "\n", flush=True)
    save_json(out_dir / "results.json", results)


if __name__ == "__main__":
    main()
