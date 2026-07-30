"""
fncl_noisefield_corr.py — S0: 層内相関の機構を定量する（docs/idea_ca.md §5.6）
（旧名: fncl_nf0.py．出力ディレクトリ tmp/out/ は旧名のまま）

§5.2 の主張「ミラーの対角近似バイアスは，上流から共有される駆動が層内相関を生む
ことに由来する」を数値で確定させる。確認事項は 4 つ:

  (i)   第 1 隠れ層では相関がほぼ 0（d が T にわたり定数なので共有駆動がない）
  (ii)  層番号に対し単調増加，かつ学習が進むほど増加
  (iii) 相関のスペクトルが低ランク（= §5.3「ブロック対角が効かない」の裏づけ）
  (iv)  周期ノイズ＋位相内中心化（§5.5，L3）で相関が実際に消え，ミラーが直る

(iv) は S1 の中核でもあるので，ここに前倒しして測る。学習は行わず，
backprop で事前学習したネット上で測定するだけなので安価。

生成物 (tmp/out/fncl_nf0/): results.json, (標準出力) 表

実行例:
  .venv/bin/python tmp/fncl_noisefield_corr.py --quick
  .venv/bin/python tmp/fncl_noisefield_corr.py --depth 4 --pretrain 0,100,400
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
from fncl_common import pearson, save_json  # noqa: E402
from fncl_mnist_fidelity import build_net, load_mnist  # noqa: E402
from fncl_noisefield_lib import (collect, corr_with_null, cov_weight_phase,  # noqa: E402
                     install_noise_field, restore_noise_field)


def true_weight(net, l: int, depth: int):
    return net.fcs[-1].weight if l == depth - 1 else net.fcs[l + 1].weight


def measure_baseline(net, x, depth: int, passes: int) -> dict:
    """通常のノイズ場（全層 period=1）での層内相関とミラー精度."""
    s = collect(net, x, passes)
    out = {"corr": {}, "mirror_r": {}, "phi_prime_sign": {}}
    for l in range(depth):
        out["corr"][f"z{l + 1}"] = corr_with_null(s["z"][l], period=1)
    # ミラー: 回帰子 z^{(l)} → 対象 d^{(l+1)}（最終層は読み出し ys）
    for l in range(depth):
        tgt = s["ys"] if l == depth - 1 else s["d"][l + 1]
        w_hat = fncl.cov_weight(tgt, s["z"][l], pool=True)
        out["mirror_r"][f"z{l + 1}"] = pearson(w_hat, true_weight(net, l, depth))
    # phi' の符号分布（L2 の余地があるか）
    crossings = fncl.crossing_layers(net)
    for l in range(depth):
        sl = fncl.kde_slope(crossings[l], s["d"][l]).mean(dim=(0, 1))
        out["phi_prime_sign"][f"z{l + 1}"] = {
            "frac_positive": float((sl > 0).float().mean()),
            "mean_abs": float(sl.abs().mean())}
    return out


def check_periodicity(net, x, period: int, target: int) -> dict:
    """自己点検: 上流層の z が本当に周期 period になっているかを直接確かめる.

    z[l] を [N, T/p, p, D] に見て，q 方向の分散が 0 なら完全に周期的である．
    """
    s = collect(net, x, passes=1)
    out = {}
    for l in range(len(s["z"])):
        z = s["z"][l]
        n, t, d = z.shape
        v = z.view(n, t // period, period, d)
        within = float(v.var(dim=1).mean())        # 位相内（q 方向）の分散
        total = float(z.var(dim=1).mean())
        out[f"z{l + 1}"] = {"within_phase_var": within, "total_var": total,
                            "frozen": within < 1e-12}
    return out


def measure_periodic(net, x, depth: int, target: int, period: int,
                     passes: int) -> dict:
    """L3: 層 1..target-1 を周期 period に固定し，層 target のミラーを位相内で測る.

    位相クラス内では上流が定数なので z^{(target)} のユニットは構成上独立になり，
    単変量ミラーが不偏になるはず（§5.5）。
    """
    periods = [period] * target + [1] * (depth - target)
    install_noise_field(net, periods=periods)
    try:
        s = collect(net, x, passes)
        pl = s["z"][0].shape[1] // passes          # 1 パスあたりの T（位相はパス内で閉じる）
        R_phase = corr_with_null(s["z"][target], period=period, pass_len=pl)
        R_plain = corr_with_null(s["z"][target], period=1)
        tgt = s["ys"] if target == depth - 1 else s["d"][target + 1]
        w_phase = cov_weight_phase(tgt, s["z"][target], period, pass_len=pl)
        w_plain = fncl.cov_weight(tgt, s["z"][target], pool=True)
        wt = true_weight(net, target, depth)
        # 前向き信号への副作用（アンサンブル出力がどれだけ乱れるか）
        return {
            "corr_phase": R_phase,
            "corr_plain": R_plain,
            "mirror_r_phase": pearson(w_phase, wt),
            "mirror_r_plain": pearson(w_plain, wt),
        }
    finally:
        restore_noise_field(net)


def forward_cost(net, x, labels, depth: int, periods, draws: int = 8) -> dict:
    """ノイズ場を変えたときの前向き信号の劣化（アンサンブル CE と分散）."""
    install_noise_field(net, periods=periods)
    try:
        with torch.no_grad():
            ys = torch.stack([net(x) for _ in range(draws)], dim=0)   # [D, N, 10]
            ce = float(tF.cross_entropy(ys.mean(dim=0), labels))
            acc = float((ys.mean(dim=0).argmax(1) == labels).float().mean())
            spread = float(ys.std(dim=0).mean())    # draw 間のばらつき
        return {"ce": ce, "acc": acc, "ensemble_std": spread}
    finally:
        restore_noise_field(net)


def main() -> None:
    p = argparse.ArgumentParser(description="S0: quantify within-layer correlation "
                                            "and test the periodic-noise fix.")
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--n-train", type=int, default=1000,
                   help="事前学習に使う枚数（A1d と同一レジームにするため既定 1000）")
    p.add_argument("--n-inputs", type=int, default=256,
                   help="相関・ミラー測定に使う枚数（学習集合の部分集合）")
    p.add_argument("--num-samples", type=int, default=64)
    p.add_argument("--sigma", type=float, default=0.5)
    p.add_argument("--crossing-h", type=float, default=0.2)
    p.add_argument("--pretrain", type=str, default="0,100,400",
                   help="測定する backprop 事前学習 epoch のチェックポイント")
    p.add_argument("--pretrain-lr", type=float, default=1e-3)
    p.add_argument("--periods", type=str, default="2,4,8,16",
                   help="L3 で試すノイズ更新周期")
    p.add_argument("--passes", type=int, default=8,
                   help="測定に使う forward パス数（実効サンプル = passes*T）")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out", type=str, default="tmp/out/fncl_nf0")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.depth, args.width = 3, 32
        args.n_train, args.n_inputs = 100, 64
        args.num_samples, args.passes = 16, 2
        args.pretrain, args.periods = "0,20", "2,4"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # A1d と同一の学習集合を再現する（load_mnist(n_train + n_test) を perm 999 で分割）。
    # 測定は学習集合の先頭 n_inputs 枚で行う（メモリのため部分集合）。
    x_all, y_all = load_mnist(args.n_train * 2)
    perm = np.random.default_rng(999).permutation(len(y_all))
    tr = perm[:args.n_train]
    x_tr = torch.tensor(x_all[tr], device=device)
    y_tr = torch.tensor(y_all[tr], device=device)
    x = x_tr[:args.n_inputs]
    labels = y_tr[:args.n_inputs]
    print(f"pretrain on {x_tr.shape[0]}, measure on {x.shape[0]}", flush=True)
    checkpoints = [int(c) for c in args.pretrain.split(",")]
    periods = [int(v) for v in args.periods.split(",")]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    net = build_net(args.depth, args.width, args.sigma, args.crossing_h,
                    args.num_samples, device)
    opt = torch.optim.Adam(net.parameters(), lr=args.pretrain_lr)

    results = {"config": vars(args), "states": {}}
    done = 0
    for ck in checkpoints:
        while done < ck:
            opt.zero_grad()
            tF.cross_entropy(net(x_tr), y_tr).backward()
            opt.step()
            done += 1
        with torch.no_grad():
            lg = net(x_tr)
            ce = float(tF.cross_entropy(lg, y_tr))
            acc = float((lg.argmax(1) == y_tr).float().mean())
        print(f"\n=== pretrain {ck} epoch : CE={ce:.4f} acc={acc:.3f} ===", flush=True)

        base = measure_baseline(net, x, args.depth, args.passes)
        for l in range(args.depth):
            c = base["corr"][f"z{l + 1}"]
            sg = base["phi_prime_sign"][f"z{l + 1}"]
            print(f"  z{l + 1}: |ρ|={c['obs']['mean_abs_offdiag']:.4f} "
                  f"(null {c['null']['mean_abs_offdiag']:.4f}, "
                  f"excess {c['excess_mean_abs_offdiag']:+.4f}) "
                  f"rowsum {c['obs']['mean_abs_rowsum']:.2f} vs "
                  f"{c['null']['mean_abs_rowsum']:.2f} | "
                  f"PR {c['obs']['participation_ratio']:.1f}/"
                  f"{c['null']['participation_ratio']:.1f} "
                  f"(ratio {c['pr_ratio']:.3f}) ev1={c['obs']['ev_top1_frac']:.3f} "
                  f"| mirror r={base['mirror_r'][f'z{l + 1}']:.4f} "
                  f"| phi'>0 {sg['frac_positive']:.2f}", flush=True)

        # 自己点検: 周期ノイズが本当に上流を凍結しているか
        install_noise_field(net, periods=[4] * (args.depth - 1) + [1])
        chk = check_periodicity(net, x, 4, args.depth - 1)
        restore_noise_field(net)
        print("  [check] period=4 での位相内分散: "
              + " ".join(f"{k}={v['within_phase_var']:.2e}" for k, v in chk.items()),
              flush=True)

        # L3: 上流を周期化して対象層のミラーを位相内で測る（target>=1 のみ意味がある）
        per = {"_periodicity_check": chk}
        for target in range(1, args.depth):
            per[str(target)] = {}
            for pd in periods:
                m = measure_periodic(net, x, args.depth, target, pd, args.passes)
                per[str(target)][str(pd)] = m
                print(f"  [L3] target z{target + 1} period={pd:2d}: "
                      f"excess|ρ| {m['corr_plain']['excess_mean_abs_offdiag']:+.4f}"
                      f" -> {m['corr_phase']['excess_mean_abs_offdiag']:+.4f} | "
                      f"PRratio {m['corr_plain']['pr_ratio']:.3f} -> "
                      f"{m['corr_phase']['pr_ratio']:.3f} | "
                      f"mirror r {m['mirror_r_plain']:.4f} -> "
                      f"{m['mirror_r_phase']:.4f}", flush=True)

        cost = {}
        for pd in [1] + periods:
            pr = [pd] * (args.depth - 1) + [1]
            cost[str(pd)] = forward_cost(net, x, labels, args.depth, pr)
            print(f"  [cost] all-upstream period={pd:2d}: "
                  f"CE={cost[str(pd)]['ce']:.4f} acc={cost[str(pd)]['acc']:.3f} "
                  f"ens.std={cost[str(pd)]['ensemble_std']:.4f}", flush=True)

        results["states"][str(ck)] = {"ce": ce, "acc": acc, "baseline": base,
                                      "periodic": per, "forward_cost": cost}

    # ---- summary ----
    lines = ["| pretrain ep | layer | excess mean|ρ| | rowsum obs/null | "
             "PR obs/null | ev top1 | mirror r |",
             "|---|---|---|---|---|---|---|"]
    for ck, st in results["states"].items():
        for l in range(args.depth):
            c = st["baseline"]["corr"][f"z{l + 1}"]
            lines.append(
                f"| {ck} | z{l + 1} | {c['excess_mean_abs_offdiag']:+.4f} | "
                f"{c['obs']['mean_abs_rowsum']:.2f} / "
                f"{c['null']['mean_abs_rowsum']:.2f} | "
                f"{c['obs']['participation_ratio']:.1f} / "
                f"{c['null']['participation_ratio']:.1f} | "
                f"{c['obs']['ev_top1_frac']:.3f} | "
                f"{st['baseline']['mirror_r'][f'z{l + 1}']:.4f} |")
    print("\n" + "\n".join(lines) + "\n", flush=True)
    save_json(out_dir / "results.json", results)


if __name__ == "__main__":
    main()
