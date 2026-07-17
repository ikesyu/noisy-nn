"""
animation_consolidation.py — noise-annealed self-consolidation の可視化

examples/animation_2funcs_firerate_sample.py と同じパネル構成（予測曲線・
ノイズ場ストリップ・隠れ層ごとの発火率マップ）で、consolidation_poc.py の
route_B（σ アニーリング + cov_jac 学習継続）の**履歴そのもの**をアニメーション
にする。例がランタイムパラメータ α を掃引するのに対し、本スクリプトの
フレーム軸はコンソリデーションの経過 epoch である：

    当初 64 ユニット（2 層 x 32）全てにノイズが供給され発火している状態から、
    冗長なユニットの σ_k が順に 0 へアニールされ、発火率マップの行が
    1 本ずつ消えていき、最終的に 32 ユニットのみが発火する。
    その間、予測曲線 sin(x) は崩れない（無停止移行）。

パネル:
    1段目: 予測曲線 y vs x（タイトルに epoch / active 数 / MSE / イベント）
    2段目: ノイズ場 σ（行 = 隠れ層 1, 2; 列 = ユニット; ○ = アニール中）
    3-4段目: 隠れ層 1, 2 の発火率マップ（T サンプル平均; 白 = 無発火）
    5段目: 移行中の eval MSE 軌跡（許容線・スナップ位置つき）

実行例（プロジェクトルートから）:
    python tmp/animation_consolidation.py --quick          # 動作確認 (~1 分)
    python tmp/animation_consolidation.py                  # 本設定 (準備に数分)
    python tmp/animation_consolidation.py --save out/consolidation.gif
    python tmp/animation_consolidation.py --save out/consolidation.mp4  # 要 ffmpeg
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
if any(a.startswith("--save") for a in sys.argv):
    matplotlib.use("Agg")               # 保存専用実行はヘッドレスで
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.animation as animation  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parent))
import fncl_driver as fncl  # noqa: E402
import consolidation_poc as poc  # noqa: E402


def main():
    p = argparse.ArgumentParser(
        description="animate noise-annealed self-consolidation (route_B)")
    poc.add_poc_args(p)
    p.add_argument("--save", type=str, default=None,
                   help="アニメーションの保存先 (.gif / .mp4)。省略時は表示")
    p.add_argument("--interval", type=int, default=80, help="ms / frame")
    p.add_argument("--hold", type=int, default=20,
                   help="開始・終了状態で静止するフレーム数")
    p.add_argument("--pred-passes", type=int, default=4,
                   help="予測曲線の平均化 forward パス数")
    args = fncl.finalize_args(p.parse_args(),
                              default_out="out/animation_consolidation")
    if args.quick:
        args.epochs_per_step = 5
        args.remove_frac = 0.25

    device = torch.device(args.device)
    seed = args.seed_list[0]
    torch.manual_seed(seed)
    np.random.seed(seed)
    x_raw, target, x, t = fncl.make_task(device)
    H = args.hidden_dim
    quota = {0: int(H * args.remove_frac), 1: int(H * args.remove_frac)}
    n_total, n_remove = 2 * H, quota[0] + quota[1]

    # ---------------- 事前学習 (cov_jac; PoC と同一) ----------------
    net = poc.build_net(H, args.sigma, args.crossing_h, args.num_samples, device)
    trainer = poc.CovJacTrainer(net, x, t, lr=args.pre_lr, opt=args.opt,
                                jac_ema=args.jac_ema)
    log_every = max(1, args.epochs // 5)
    print(f"Pretraining cov_jac_{args.opt}  (H={H}, T={args.num_samples}, "
          f"epochs={args.epochs}) …")
    for e in range(args.epochs):
        loss = trainer.step()
        if e % log_every == 0 or e == args.epochs - 1:
            print(f"  epoch {e:5d}  mse={loss:.5f}")
    trainer.close()
    base = float(np.mean(trainer.losses[-min(100, len(trainer.losses)):]))
    tol = base * args.drift_mult + args.drift_abs
    pred0 = fncl.predict(net, x, passes=16)
    print(f"pretrained: task MSE={float(np.mean((pred0 - target) ** 2)):.5f}  "
          f"baseline={base:.5f}  tol={tol:.5f}")

    # ---------------- route_B を記録つきで実行 ----------------
    frames = []

    def record(event, net_, tr, removed, anneal):
        with torch.no_grad():
            preds = torch.stack([net_(x) for _ in range(args.pred_passes)],
                                dim=0).mean(dim=0)          # hooks が tr.cap を更新
        pred = preds.squeeze(1).cpu().numpy()
        frames.append(dict(
            epoch=len(tr.losses), event=event, removed=removed,
            anneal=(None if anneal is None else
                    (anneal[0], anneal[1],
                     float(net_.sigma_vecs[anneal[0]][anneal[1]]))),
            sigma=np.stack([s.cpu().numpy().copy() for s in net_.sigma_vecs]),
            rate=[tr.cap.z[l].mean(dim=1).cpu().numpy().T.copy()  # [H, N]
                  for l in range(2)],
            pred=pred,
            mse=float(np.mean((pred - target) ** 2)),
        ))
        if len(frames) % 20 == 0:
            print(f"  [route_B] epoch {len(tr.losses):5d}  removed {removed:2d}"
                  f"/{n_remove}  mse={frames[-1]['mse']:.5f}")

    print("Running route_B (noise-annealed self-consolidation) …")
    curve, losses, snaps, holds = poc.run_route_B(
        net, x, target, pred0, quota, tol, args, record=record)
    losses = np.asarray(losses)
    print(f"done: removed {n_remove} units in {len(losses)} epochs "
          f"(holds={holds}); final task MSE={curve[-1]['task_mse']:.5f}")

    # 開始・終了で静止するフレーム列
    order = [0] * args.hold + list(range(len(frames))) \
        + [len(frames) - 1] * args.hold
    n_epochs = len(losses)

    # ---------------- 図 ----------------
    N = len(x_raw)
    fig, axes = plt.subplots(
        5, 1, figsize=(14, 12),
        gridspec_kw={"height_ratios": [2.6, 0.8, 1.6, 1.6, 1.2]})
    ax_pred, ax_noise, ax_r1, ax_r2, ax_loss = axes
    ax_rasters = [ax_r1, ax_r2]

    # --- 予測 ---
    ax_pred.plot(x_raw, target, color="steelblue", lw=1.5, ls="--",
                 alpha=0.6, label="target sin(x)")
    pred_line, = ax_pred.plot(x_raw, frames[0]["pred"], color="black",
                              lw=2.0, label="network output")
    ax_pred.set_xlim(float(x_raw[0]), float(x_raw[-1]))
    ax_pred.set_ylim(-1.6, 1.6)
    ax_pred.set_ylabel("y")
    ax_pred.legend(loc="upper right", fontsize=9)
    ax_pred.grid(alpha=0.35)
    pred_title = ax_pred.set_title("")

    # --- ノイズ場 (行 = 層) ---
    noise_im = ax_noise.imshow(
        frames[0]["sigma"], aspect="auto", cmap="Reds",
        vmin=0.0, vmax=args.sigma, origin="lower",
        extent=[-0.5, H - 0.5, -0.5, 1.5], interpolation="nearest")
    anneal_dot, = ax_noise.plot([], [], "o", mfc="none", mec="blue", mew=1.6,
                                ms=9)
    ax_noise.set_yticks([0, 1])
    ax_noise.set_yticklabels(["layer 1", "layer 2"])
    ax_noise.set_xlabel("Neuron index")
    ax_noise.set_title("Noise field σ per neuron "
                       "(red = noise supplied, white = detached; ○ = annealing)")

    # --- 発火率マップ ---
    raster_ims = []
    for i, ax_r in enumerate(ax_rasters):
        im = ax_r.imshow(
            frames[0]["rate"][i], aspect="auto", cmap="binary",
            vmin=0.0, vmax=0.5, origin="lower",
            extent=[float(x_raw[0]), float(x_raw[-1]), -0.5, H - 0.5],
            interpolation="nearest")
        ax_r.set_ylabel("Neuron")
        ax_r.set_title(f"Hidden layer {i + 1} ({H} neurons) — firing rate "
                       f"(mean over T={args.num_samples} samples)")
        raster_ims.append(im)
    ax_rasters[-1].set_xlabel("x (input value)")
    anneal_rows = [ax_r.axhline(0, color="blue", lw=1.0, alpha=0.0)
                   for ax_r in ax_rasters]

    # --- 損失軌跡 ---
    ax_loss.axhline(tol, ls="--", c="k", lw=0.8, label="drift tolerance")
    ax_loss.axhline(base, ls=":", c="k", lw=0.8, label="pretrain baseline")
    snap_lines = [ax_loss.axvline(s, color="gray", lw=0.4, alpha=0.0)
                  for s in snaps]
    loss_line, = ax_loss.plot([], [], lw=0.8, color="tab:blue",
                              label="eval MSE (route_B)")
    loss_dot, = ax_loss.plot([], [], "o", color="tab:red", ms=4)
    ax_loss.set_xlim(0, max(1, n_epochs))
    ax_loss.set_yscale("log")
    ax_loss.set_ylim(min(losses.min(), base) * 0.5,
                     max(losses.max(), tol) * 3.0)
    ax_loss.set_xlabel("epoch (during consolidation)")
    ax_loss.set_ylabel("eval MSE")
    ax_loss.legend(loc="upper right", fontsize=8)
    ax_loss.grid(alpha=0.3, which="both")

    fig.suptitle(
        f"Noise-annealed self-consolidation: {n_total} -> "
        f"{n_total - n_remove} active units "
        f"(cov_jac_{args.opt}, sin regression, seed {seed})", fontsize=11)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))

    # ---------------- 更新関数 ----------------
    def update(idx: int):
        f = frames[order[idx]]
        pred_line.set_ydata(f["pred"])
        noise_im.set_data(f["sigma"])
        for i, im in enumerate(raster_ims):
            im.set_data(f["rate"][i])

        if f["anneal"] is not None and f["event"] != "snap":
            l, k, sig = f["anneal"]
            anneal_dot.set_data([k], [l])
            for i, ln in enumerate(anneal_rows):
                ln.set_alpha(0.55 if i == l else 0.0)
                if i == l:
                    ln.set_ydata([k, k])
            ev = {"train": f"annealing L{l + 1}#{k}: σ={sig:.3f}",
                  "hold": f"hold (drift > tol) L{l + 1}#{k}: σ={sig:.3f}"}[
                      f["event"]]
        else:
            anneal_dot.set_data([], [])
            for ln in anneal_rows:
                ln.set_alpha(0.0)
            if f["event"] == "snap":
                l, k, _ = f["anneal"]
                ev = f"snap: L{l + 1}#{k} -> σ=0"
            else:
                ev = "pretrained (all units active)" if f["removed"] == 0 \
                    else "consolidated"

        n_act = int((f["sigma"] > 0).sum())
        pred_title.set_text(
            f"epoch {f['epoch']:5d}   active {n_act}/{n_total}   "
            f"removed {f['removed']}/{n_remove}   MSE={f['mse']:.4f}   [{ev}]")

        e = f["epoch"]
        loss_line.set_data(np.arange(e), losses[:e])
        if e > 0:
            loss_dot.set_data([e - 1], [losses[e - 1]])
        for s, ln in zip(snaps, snap_lines):
            ln.set_alpha(0.5 if s <= e else 0.0)
        return ([pred_line, pred_title, noise_im, anneal_dot, loss_line,
                 loss_dot] + raster_ims + anneal_rows + snap_lines)

    ani = animation.FuncAnimation(fig, update, frames=len(order),
                                  interval=args.interval, blit=False,
                                  repeat=True)

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        fps = max(1, round(1000 / args.interval))
        if out.suffix.lower() == ".gif":
            writer = animation.PillowWriter(fps=fps)
        else:
            writer = animation.FFMpegWriter(fps=fps, bitrate=2400)
        print(f"Saving {len(order)} frames -> {out} …")
        ani.save(str(out), writer=writer, dpi=100)
        print(f"saved {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
