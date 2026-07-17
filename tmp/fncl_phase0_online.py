"""
fncl_phase0_online.py — FPGA 実装 Phase 0 実験②「オンライン (N=1) 学習の収束性」

現行 train_cov は全入力 (N=128) を一括処理するバッチ学習だが、ロボットへの
組み込みでは入力は 1 個ずつ流れてくる。実験① (fncl_phase0_sgd.py) で凍結した
HW 構成 (cov_jac + sgdm(momentum=0.9375) + lr=0.125 + cosine + jac_track) を
固定し、総提示回数 (budget = 128 x 1500 既定) を揃えて

    batch_ref : 現行バッチ学習 (train_cov そのまま; 実験①の基準)
    B=32, 8   : ミニバッチ緩和 (毎ステップ, グリッドから i.i.d. 抽出)
    B=1 iid   : 完全オンライン, i.i.d. 提示
    B=1 traj  : 完全オンライン, 軌道提示 (入力が端で折り返す連続掃引 =
                ロボットで実際に起きる時間相関入力; 分布は i.i.d. と同一)

を比較する。判定は「B=1 の最終 MSE が batch_ref の x2 以内か」。

追加オプション:
  --orders iid,traj : B>1 でも軌道提示 (連続 B 点の窓で累算してから更新 =
                      ロボット実機での「ストリーミング累算」条件) を検証。
  --lr-scale linear : オンライン構成の lr を lr x B/128 に下げる線形則
                      (B=1 の勾配分散はバッチの 128 倍のため) の検証。

オンライン化の要点 (そのまま RTL 仕様になる):
  - per_input credit は T サンプル内でセンタリングするので N=1 でも成立。
  - mirror は毎ステップ cov_weight(pool=True) で測り EMA。ステップ頻度が
    バッチの N/B 倍になるため、EMA 率は jac_ema^(B/N) に自動スケールして
    平滑化の実効ホライズン (提示回数換算) を揃える。
  - Kolen-Pollack 追跡は「適用したステップぶん mirror をずらす」だけなので
    ステップ粒度が変わっても厳密なまま。
  - 学習率は総ステップ数にわたる cosine (HW では ROM テーブル)。

最終 MSE (8-pass predict) に加えて mirror 回復精度 r (論文 §5.3 の指標;
W_ema と真の重みの Pearson 相関) も記録する。

生成物 (out/fncl_phase0_online/):
  table_online.md         -> 最終 MSE + mirror r の表と判定
  fig_learning_curves.png -> 提示回数を横軸にした学習曲線 (log)
  results.json

実行例:
  python tmp/fncl_phase0_online.py --quick          # 動作確認 (~1 分)
  python tmp/fncl_phase0_online.py                  # 本実験 (数時間; B=1 が支配的)
  python tmp/fncl_phase0_online.py --batch-sizes 8,1 --budget 96000
"""
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # 保存専用 (fncl_driver が pyplot を import する前に設定)
import matplotlib.pyplot as plt  # noqa: E402

import fncl_driver as fncl  # noqa: E402
from fncl_driver import (add_common_args, finalize_args, make_task,  # noqa: E402
                         model_factory, config_dict, write_text, save_json,
                         savefig)
# import の副作用で ManualOpt (sgdm) と lr_at (step) のパッチが適用される。
from fncl_phase0_sgd import ManualOptPhase0, lr_at_phase0  # noqa: E402


# ============================================================
# オンライン学習ループ (train_cov の cov_jac パスのミニバッチ / N=1 版)
# ============================================================
def train_cov_online(net, x_all, t_all, noise: str, args, batch_size: int,
                     order: str = "iid", eval_points: int = 50,
                     log_every_steps: int = 0, lr0: float = None):
    """凍結構成 (cov_jac, sgdm, jac_track) をミニバッチ / オンラインで学習する.

    毎ステップ: batch_size 個の入力で T サンプル forward -> mirror 測定 + EMA
    -> 再帰 credit -> sgdm 更新 -> KP 追跡。order == "traj" はグリッド上を端で
    折り返しながら掃引し、連続する batch_size 点を 1 窓として使う (時間相関
    入力での「累算してから更新」= ロボット実機の提示条件)。

    lr0 を指定するとその学習率を使う (線形スケール則 lr ∝ B の検証用)。
    返り値: (curve [(presentations, eval_mse), ...], mirror_r {層: r})
    """
    assert order in ("iid", "traj")
    device = x_all.device
    N_all = x_all.shape[0]
    lr0 = args.lr if lr0 is None else lr0
    steps = args.budget // batch_size
    # EMA 率をステップ頻度に合わせる: バッチ (N_all 提示/更新) の jac_ema と
    # 実効ホライズンが揃うように jac_ema^(B/N_all)。
    jac_ema_step = float(args.jac_ema ** (batch_size / N_all))
    eval_every = max(1, steps // eval_points)

    cap = fncl.Capture(net)
    n_hidden = cap.n_hidden
    optim = ManualOptPhase0("sgdm")
    W_ema = {}
    curve = []
    pos, direction = 0, 1                          # traj 用の掃引位置
    for step_i in range(steps):
        lr_t = lr_at_phase0(lr0, step_i, steps, args.lr_decay)
        if order == "iid":
            idx = torch.randint(0, N_all, (batch_size,), device=device)
        else:
            ids = []
            for _ in range(batch_size):
                ids.append(pos)
                pos += direction
                if pos in (0, N_all - 1):
                    direction = -direction
            idx = torch.tensor(ids, device=device)
        x, t = x_all[idx], t_all[idx]
        B = x.shape[0]
        with torch.no_grad():
            y = net(x)                                       # [B, 1]; フック発火
            L = (cap.y_samples.squeeze(-1) - t) ** 2         # noqa: F841 (参照用)
            z = [cap.z[l] for l in range(n_hidden)]          # [B, T, H]
            d = [cap.d[l] for l in range(n_hidden)]
            ys = cap.y_samples                               # [B, T, 1]
            T = ys.shape[1]

            # mirror 測定 + EMA (train_cov と同一; pool=True)
            slope_full = [fncl.kde_slope(cap.crossings[l], d[l])
                          for l in range(n_hidden)]
            slope_mean = [s.mean(dim=1) for s in slope_full]
            meas = {"out": fncl.cov_weight(ys, z[-1], pool=True)}
            for l in range(1, n_hidden):
                meas[l] = fncl.cov_weight(d[l], z[l - 1], pool=True)
            if not W_ema:
                W_ema.update(meas)
            else:
                for k, v in meas.items():
                    W_ema[k] = jac_ema_step * W_ema[k] + (1.0 - jac_ema_step) * v

            # 再帰 credit (出力誤差は解析的 2(y - t); cov_jac 凍結仕様)
            a_jac = [None] * n_hidden
            a_jac[-1] = 2.0 * (y - t) * W_ema["out"]         # [B, H]
            for l in range(n_hidden - 2, -1, -1):
                dd_next = a_jac[l + 1] * slope_mean[l + 1]
                a_jac[l] = dd_next @ W_ema[l + 1]

            # 勾配と sgdm 更新 + KP 追跡
            z_prev = [x.unsqueeze(1).expand(B, T, x.shape[1])] + z[:-1]
            kp_steps = {}
            for l in range(n_hidden):
                delta_hat = a_jac[l].unsqueeze(1) * slope_full[l]     # [B, T, H]
                gW = torch.einsum("nto,nti->oi", delta_hat, z_prev[l]) / (B * T)
                gb = delta_hat.mean(dim=(0, 1))
                kp_steps[l] = optim.update(f"w{l}", net.fcs[l].weight, gW, lr_t)
                if net.fcs[l].bias is not None:
                    optim.update(f"b{l}", net.fcs[l].bias, gb, lr_t)
            z_bar = z[-1].mean(dim=1)
            dL_dy = 2.0 * (y - t)
            gWout = torch.einsum("no,ni->oi", dL_dy, z_bar) / B
            kp_steps["out"] = optim.update("wout", net.fcs[-1].weight, gWout, lr_t)
            if net.fcs[-1].bias is not None:
                optim.update("bout", net.fcs[-1].bias, dL_dy.mean(dim=0), lr_t)
            W_ema["out"] = W_ema["out"] - kp_steps["out"]
            for l in range(1, n_hidden):
                W_ema[l] = W_ema[l] - kp_steps[l]

        if step_i % eval_every == 0 or step_i == steps - 1:
            mse = float(np.mean((fncl.predict(net, x_all)
                                 - t_all.squeeze(1).cpu().numpy()) ** 2))
            curve.append((step_i * batch_size, mse))
            if log_every_steps and (step_i // eval_every) % log_every_steps == 0:
                print(f"    step {step_i:7d} ({step_i * batch_size:7d} pres.) "
                      f"eval_mse={mse:.5f}", flush=True)

    mirror_r = {"out": _corr(W_ema["out"], net.fcs[-1].weight)}
    for l in range(1, n_hidden):
        mirror_r[f"W{l}"] = _corr(W_ema[l], net.fcs[l].weight)
    cap.remove()
    return curve, mirror_r


def _corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().cpu().numpy().ravel()
    b = b.detach().cpu().numpy().ravel()
    return float(np.corrcoef(a, b)[0, 1])


# ============================================================
# main
# ============================================================
def main() -> None:
    p = argparse.ArgumentParser(
        description="Phase 0-2: does frozen cov_jac survive minibatch / N=1 online?")
    add_common_args(p)
    p.set_defaults(lr=0.125)                       # 実験①で凍結した 2 の冪定数
    p.add_argument("--noise", choices=("uniform", "gaussian"), default="uniform")
    p.add_argument("--lr-decay", choices=("none", "cosine", "exp", "step"),
                   default="cosine", help="総ステップ数にわたる減衰 (HW では ROM)")
    p.add_argument("--momentum", type=float, default=0.9375, help="sgdm (1-2^-4)")
    p.add_argument("--batch-sizes", type=str, default="32,8,1",
                   help="掃引するミニバッチサイズ (カンマ区切り)")
    p.add_argument("--orders", type=str, default="iid",
                   help="提示順 (カンマ区切り; iid, traj)。traj は連続掃引窓 "
                        "(B=1 は既定で traj も追加)")
    p.add_argument("--lr-scale", choices=("none", "linear"), default="none",
                   help="linear: オンライン構成の lr を lr x B/128 に線形則で"
                        "スケール (batch_ref は --lr のまま)")
    p.add_argument("--budget", type=int, default=None,
                   help="総提示回数 (既定: 128 x --epochs = 実験①と同予算)")
    args = finalize_args(p.parse_args(), default_out="out/fncl_phase0_online")
    args.batch_list = [int(s) for s in args.batch_sizes.split(",") if s.strip()]
    args.order_list = [s for s in args.orders.split(",") if s.strip()]
    if args.quick:
        args.batch_list = args.batch_list[-2:]     # 小さい B だけ確認
    ManualOptPhase0.momentum = args.momentum

    device = torch.device(args.device)
    x_raw, target, x_all, t_all = make_task(device)
    N_all = x_all.shape[0]
    if args.budget is None:
        args.budget = N_all * args.epochs
    print(f"budget = {args.budget} presentations, frozen: cov_jac + "
          f"sgdm({args.momentum}) + lr {args.lr} + {args.lr_decay}", flush=True)

    configs = [("batch_ref", N_all, "batch")]
    for b in args.batch_list:
        orders = list(args.order_list)
        if b == 1 and "traj" not in orders:
            orders.append("traj")
        for order in orders:
            configs.append((f"B{b}_{order}", b, order))

    mse, mirrors, curves = {}, {}, {}
    for name, bsize, order in configs:
        mse[name], mirrors[name] = {}, {}
        for seed in args.seed_list:
            torch.manual_seed(seed)
            np.random.seed(seed)
            fresh = model_factory(args.noise, args, device)
            net = fresh()
            print(f"[seed {seed}] {name}", flush=True)
            if order == "batch":
                epochs = args.budget // N_all
                losses, _ = fncl.train_cov(
                    net, x_all, t_all, args.noise, args.sigma, args.radius,
                    "cov_jac", args.lr, epochs, credit=args.credit,
                    credit_passes=args.credit_passes, opt="sgdm",
                    lr_decay=args.lr_decay, jac_ema=args.jac_ema,
                    jac_track=True, log_every=0)
                curve = [(e * N_all, m) for e, m in enumerate(losses)]
                mirror_r = {}
            else:
                lr0 = (args.lr * bsize / N_all if args.lr_scale == "linear"
                       else args.lr)
                curve, mirror_r = train_cov_online(
                    net, x_all, t_all, args.noise, args, bsize, order,
                    log_every_steps=10, lr0=lr0)
            final = float(np.mean((fncl.predict(net, x_all) - target) ** 2))
            mse[name][seed] = final
            mirrors[name][seed] = mirror_r
            r_txt = (" mirror_r=" + ",".join(f"{k}:{v:.3f}"
                                             for k, v in mirror_r.items())
                     if mirror_r else "")
            print(f"[seed {seed}] {name:12s} final MSE = {final:.5f}{r_txt}",
                  flush=True)
            if seed == args.seed_list[0]:
                curves[name] = curve

    # --- 表と判定 ---------------------------------------------------------
    def mstat(name):
        vals = [mse[name][s] for s in args.seed_list]
        return float(np.mean(vals)), float(np.std(vals))

    ref_mean, _ = mstat("batch_ref")
    lines = [f"**Phase 0-2 online/minibatch: cov_jac + sgdm({args.momentum}) "
             f"+ lr {args.lr} (scale={args.lr_scale}) + {args.lr_decay}, "
             f"noise={args.noise}, "
             f"H={args.hidden_dim}, T={args.num_samples}, "
             f"budget={args.budget}, seeds={args.seed_list}**", "",
             "| config | final MSE (mean ± std) | vs batch_ref | mirror r (mean) |",
             "|---|---|---|---|"]
    for name, _, _ in configs:
        m, s = mstat(name)
        if mirrors[name][args.seed_list[0]]:
            keys = mirrors[name][args.seed_list[0]].keys()
            r_mean = {k: np.mean([mirrors[name][sd][k] for sd in args.seed_list])
                      for k in keys}
            r_txt = ", ".join(f"{k}={v:.3f}" for k, v in r_mean.items())
        else:
            r_txt = "—"
        lines.append(f"| {name} | {m:.5f} ± {s:.5f} | x{m / ref_mean:.2f} "
                     f"| {r_txt} |")
    online = [n for n, b, _ in configs if b == 1]
    if online:
        worst = max(online, key=lambda n: mstat(n)[0])
        wm, _ = mstat(worst)
        ok = wm <= 2.0 * ref_mean
        lines += ["", f"最悪の完全オンライン構成: `{worst}` "
                      f"(MSE {wm:.5f} = batch_ref の x{wm / ref_mean:.2f})",
                  ("=> **PASS**: N=1 オンラインでもバッチ水準 (x2 以内)。"
                   "Phase 1 (固定小数点化) に進める。" if ok else
                   "=> **FAIL**: x2 を超過。jac_ema スケール則・momentum・"
                   "リプレイ (直近入力の小バッファ) の検討が必要。")]
    table = "\n".join(lines) + "\n"
    print("\n" + table)
    write_text(args.out_dir / "table_online.md", table)
    save_json(args.out_dir / "results.json",
              {"config": config_dict(args), "final_mse": mse,
               "mirror_r": mirrors,
               "curves": {k: v for k, v in curves.items()}})

    fig = plt.figure(figsize=(7.0, 4.5))
    for name, curve in curves.items():
        pres, vals = zip(*curve)
        plt.plot(pres, vals, label=name, lw=1.2)
    plt.yscale("log")
    plt.xlabel("input presentations")
    plt.ylabel("eval MSE (log)")
    plt.title("Phase 0-2: batch vs minibatch vs N=1 online (equal budget)")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    fig.tight_layout()
    savefig(fig, args.out_dir / "fig_learning_curves.png")


if __name__ == "__main__":
    main()
