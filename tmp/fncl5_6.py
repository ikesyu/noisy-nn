"""
fncl5_6.py — 論文 §5.6「負の結果 — 外部ノード摂動は二値交差に効かない」

(a) 手法比較: cov_deriv_kde (基準) に対し、外部摂動ゲート cov_deriv_gate と
    その antithetic / common-random-number 版 cov_deriv_gate_crn を、摂動強度
    alpha を掃引して比較する (CRN でも cov_deriv に届かないこと)。
    注: crn は 1 credit パスあたり forward を 2 回走らせる (+xi / -xi) ので、
    forward 回数で見ると gate の 2 倍の予算を使っている (それでも届かない)。

(b) 退化のデモ: CRN ペア (+xi, -xi) を RNG 状態リセットで同一ノイズの下で
    走らせたとき、per-sample 損失差 L(+xi) - L(-xi) が「厳密に 0」になる
    サンプルの割合を alpha の関数として測る。二値交差のパスワイズ応答が
    測度ゼロ集合でしか反応しない (= 外部摂動から情報がほとんど取れない)
    ことの直接の可視化。

生成物 (out/fncl5_6/):
  fig_gate_mse_vs_alpha.png -> 図 (MSE vs alpha, gate / crn / cov_deriv 基準線)
  fig_crn_degeneracy.png    -> 図 (損失差が厳密に 0 のサンプル割合 vs alpha)
  table_gate.md             -> 数表
  results.json

実行例:
  python tmp/fncl5_6.py                       # 既定: H=32, T=48, 1000 epochs
  python tmp/fncl5_6.py --alphas 0.05,0.1,0.3,1.0
  python tmp/fncl5_6.py --quick
"""
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt

from fncl_common import (add_common_args, finalize_args, make_task,
                         model_factory, run_method, config_dict,
                         write_text, save_json, savefig, fncl)


def crn_zero_fraction(args, alphas, device, block_size: int) -> list:
    """(b) alpha ごとに、CRN ペアの per-sample 損失差が厳密に 0 の割合を測る.

    摂動はブロックゲート (block_size ユニット/層) に限定する。block_size=1 は
    パスワイズ退化の最も純粋な単一ユニット版、block_size=gate_block_size は
    実際の cov_deriv_gate_crn が信号を取り出す条件そのもの。"""
    torch.manual_seed(args.seed_list[0])
    np.random.seed(args.seed_list[0])
    x_raw, target, x, t = make_task(device)
    net = model_factory(args.noise, args, device)()
    hidden_sizes = list(net.structure[1:-1])
    masks = fncl.gate_masks(hidden_sizes, 0, block_size, "cyclic", device)
    cap = fncl.Capture(net)
    pert = fncl.Perturber(net, alpha=1.0, mode="crn")
    fracs = []
    with torch.no_grad():
        for a in alphas:
            # ゲートブロックのみに固定摂動 a*m*xi を注入し, +/- を同一ノイズで評価
            pert.fixed_p = [
                a * m.view(1, 1, -1)
                * torch.randn(x.shape[0], net.t, h, device=device)
                for m, h in zip(masks, hidden_sizes)]
            snap = fncl.rng_snapshot(device)
            pert.sign = +1.0
            net(x)
            L_plus = (cap.y_samples.squeeze(-1) - t) ** 2      # [N, T]
            fncl.rng_restore(snap)                              # 同一ノイズを再現
            pert.sign = -1.0
            net(x)
            L_minus = (cap.y_samples.squeeze(-1) - t) ** 2
            frac = float((L_plus == L_minus).float().mean())    # 厳密一致の割合
            fracs.append(frac)
            print(f"  block={block_size:<3d} alpha={a:<6g} "
                  f"P[L(+xi) - L(-xi) == 0] = {frac:.4f}", flush=True)
    cap.remove()
    pert.remove()
    return fracs


def main() -> None:
    p = argparse.ArgumentParser(
        description="§5.6 negative result: external node perturbation "
                    "(incl. CRN) fails on the binary crossing.")
    # doc §8.3 のレジーム (CPU で回る予算) を既定にする
    add_common_args(p, epochs=1000, hidden_dim=32, num_samples=48, seeds="0")
    p.add_argument("--noise", choices=("gaussian", "uniform"), default="gaussian")
    p.add_argument("--alphas", type=str, default="0.05,0.1,0.3,1.0",
                   help="摂動強度 alpha の掃引 (カンマ区切り)")
    p.add_argument("--gate-block-size", type=int, default=8)
    args = finalize_args(p.parse_args(), default_out="out/fncl5_6")
    alphas = [float(a) for a in args.alphas.split(",") if a.strip() != ""]
    if args.quick:
        alphas = alphas[:2]

    device = torch.device(args.device)
    x_raw, target, x, t = make_task(device)
    seed = args.seed_list[0]
    log_every = max(1, args.epochs // 5)
    results = {"config": config_dict(args), "alphas": alphas, "mse": {}}

    # --- (a) 手法比較: 基準 cov_deriv_kde + alpha 掃引の gate / crn ---
    def run(name, spec):
        torch.manual_seed(seed)
        np.random.seed(seed)
        fresh = model_factory(args.noise, args, device)
        losses, pred, _ = run_method(spec, fresh, x, t, args.noise, args,
                                     log_every)
        mse = float(np.mean((pred - target) ** 2))
        results["mse"][name] = mse
        print(f"{name:28s} final MSE = {mse:.5f}", flush=True)
        return mse

    base = run("cov_deriv_kde", {"method": "cov_deriv", "slope": "kde"})
    gate_mse, crn_mse = [], []
    for a in alphas:
        gate_mse.append(run(f"cov_deriv_gate/alpha={a}",
                            {"method": "cov_deriv_gate", "gate_alpha": a,
                             "gate_block_size": args.gate_block_size,
                             "gate_mode": "cyclic"}))
        crn_mse.append(run(f"cov_deriv_gate_crn/alpha={a}",
                           {"method": "cov_deriv_gate_crn", "gate_alpha": a,
                            "gate_block_size": args.gate_block_size,
                            "gate_mode": "cyclic"}))

    fig = plt.figure(figsize=(7, 5))
    plt.axhline(base, color="k", ls="--", lw=1.2,
                label=f"cov_deriv_kde (baseline, {base:.3f})")
    plt.plot(alphas, gate_mse, "o-", label="cov_deriv_gate")
    plt.plot(alphas, crn_mse, "s-", label="cov_deriv_gate_crn (2x forwards)")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("perturbation strength alpha")
    plt.ylabel("final MSE (log)")
    plt.title("External node perturbation fails on the binary crossing")
    plt.grid(alpha=0.3)
    plt.legend()
    fig.tight_layout()
    savefig(fig, args.out_dir / "fig_gate_mse_vs_alpha.png")

    # --- (b) CRN 退化のデモ (単一ユニット / 手法と同じブロック) ---
    print("\n[degeneracy] fraction of exactly-zero loss differences:",
          flush=True)
    fracs_1 = crn_zero_fraction(args, alphas, device, block_size=1)
    fracs_b = crn_zero_fraction(args, alphas, device,
                                block_size=args.gate_block_size)
    results["crn_zero_fraction"] = {
        "block_1": dict(zip(map(str, alphas), fracs_1)),
        f"block_{args.gate_block_size}": dict(zip(map(str, alphas), fracs_b)),
    }
    fig = plt.figure(figsize=(6.5, 4.5))
    plt.plot(alphas, fracs_1, "o-", label="single perturbed unit per layer")
    plt.plot(alphas, fracs_b, "s-",
             label=f"block of {args.gate_block_size} (as in the method)")
    plt.xscale("log")
    plt.ylim(0.0, 1.05)
    plt.xlabel("perturbation strength alpha")
    plt.ylabel("P[ L(+xi) - L(-xi) == 0 ]  (exact)")
    plt.title("Pathwise degeneracy of the binary crossing under CRN pairs")
    plt.grid(alpha=0.3)
    plt.legend()
    fig.tight_layout()
    savefig(fig, args.out_dir / "fig_crn_degeneracy.png")

    lines = ["| run | final MSE |", "|---|---|"]
    for k, v in results["mse"].items():
        lines.append(f"| {k} | {v:.5f} |")
    lines.append("")
    lines.append("| alpha | P[dL == 0] (block=1) | "
                 f"P[dL == 0] (block={args.gate_block_size}) |")
    lines.append("|---|---|---|")
    for a, f1, fb in zip(alphas, fracs_1, fracs_b):
        lines.append(f"| {a} | {f1:.4f} | {fb:.4f} |")
    write_text(args.out_dir / "table_gate.md", "\n".join(lines) + "\n")
    save_json(args.out_dir / "results.json", results)


if __name__ == "__main__":
    main()
