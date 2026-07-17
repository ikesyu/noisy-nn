"""
consolidation_stop_signal.py — アニール停止判定シグナルの検証 + min S_k 非単調性の分離実験
(docs/idea_consolidation.md §7.8 / §12.5 / §12.6)

問い 1（§12.5）: 「これ以上ノイズ場の支持領域を縮められない」ことを、タスク誤差の
劣化が起こる**前に** forward 統計だけから検知できるか。特に残存ユニットの活動相関の
上昇が停止判定に使えるか。

問い 2（§12.6.7 の分離実験）: min S_k の非単調性（上昇 -> 低下）を駆動するのは
  (b) 部分空間の共線形性の再形成（回帰残差 ||r_k||^2 の低下）か、
  (c) 出力側重みの集中（||w_k||^2 の低下 = 使われないユニットの発生）か。
S_k = ||w_k||^2 * (||r_k||^2 / ||z_k||^2) は厳密に2因子の積なので、min S_k の
ピークと谷での log 分解 dlog S = dlog w2 + dlog rr で寄与を分離できる。併せて
  - med S_k（候補全体の中央値）: 低下が基底全体の現象か 1 ユニットの現象か
  - 生存ユニット期待活性のグラム行列の有効ランク（Roy & Vetterli）:
    共線形性の再形成なら、有効ランクが生存数より深く低下するはず
を記録する。

方法: 圧縮限界を意図的に超える設定（各層 32 -> 4、計 56/64 ユニット削除）で
route_B を回し、各スナップ（ユニット離脱確定）時点で以下を記録する。

  mse         : タスク MSE（8-pass 予測）。許容 tol の持続超過点を「膝」とする。
  corr_fluct  : 残存ユニットの per-input ゆらぎ相関（T 方向中心化・入力プール、
                非対角の絶対相関の平均、層別）。
  corr_tuning : 期待活性のチューニング相関（入力方向、層別）。
  min_S/med_S : 残存候補の削除コスト S_k の最小値・中央値。
  minS_w2     : argmin ユニットの ||w_k||^2（出力側重みノルム因子）。
  minS_rr     : argmin ユニットの ||r_k||^2 / (||z_k||^2 + eps)（正規化残差因子）。
  erank1/2    : 層別の有効ランク（期待活性の共分散固有値エントロピー指数）。
  holds/drift : 閉ループの保留回数とスナップ間の最大損失。

生成物 (out/consolidation_stop_signal/):
  fig_stop_signals.png   シグナル vs 削除数（膝・検知点マーカー付き、seed 0）
  table_stop.md          seed ごとの膝・検知点・lead + min S_k 分解表
  results.json

実行例:
  python tmp/consolidation_stop_signal.py --quick
  python tmp/consolidation_stop_signal.py --seeds 0,1,2
"""
import argparse

import numpy as np
import torch
import matplotlib
if __name__ == "__main__":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import fncl_driver as fncl  # noqa: E402
from fncl_driver import save_json, savefig, write_text  # noqa: E402
import consolidation_poc as poc  # noqa: E402
from nnn.credit import EPS  # noqa: E402


# ============================================================
# シグナル計算
# ============================================================
def offdiag_corr(C: torch.Tensor) -> float:
    """共分散行列 C から非対角の絶対相関の平均."""
    d = torch.diag(C)
    R = C / torch.sqrt(torch.outer(d, d) + EPS)
    H = C.shape[0]
    if H < 2:
        return float("nan")
    off = R - torch.diag(torch.diag(R))
    return float(off.abs().sum() / (H * (H - 1)))


def corr_fluct(z: torch.Tensor, active: list) -> float:
    """per-input ゆらぎ相関: T 方向中心化 -> 入力プールした Cov(z,z)."""
    za = z[:, :, active]                                  # [N, PT, Ha]
    cz = za - za.mean(dim=1, keepdim=True)
    C = torch.einsum("nti,ntj->ij", cz, cz) / (cz.shape[0] * cz.shape[1])
    return offdiag_corr(C)


def corr_tuning(zbar: torch.Tensor, active: list) -> float:
    """チューニング相関: 期待活性の入力方向相関."""
    za = zbar[:, active]                                  # [N, Ha]
    cz = za - za.mean(dim=0, keepdim=True)
    C = cz.T @ cz / cz.shape[0]
    return offdiag_corr(C)


def effective_rank(zbar: torch.Tensor, active: list) -> float:
    """生存ユニット期待活性の有効ランク (Roy & Vetterli 2007):
    exp(固有値分布のエントロピー)。共線形性の再形成で生存数より深く低下する."""
    za = zbar[:, active]
    cz = za - za.mean(dim=0, keepdim=True)
    C = cz.T @ cz / cz.shape[0]
    ev = torch.linalg.eigvalsh(C).clamp(min=0.0)
    p = ev / (ev.sum() + EPS)
    p = p[p > 1e-12]
    return float(torch.exp(-(p * p.log()).sum()))


def score_details(net, zbar, active: dict, ridge: float):
    """候補全体の S_k から (min, median, argmin の w2 因子, argmin の rr 因子)."""
    best, all_S = None, []
    for l in (0, 1):
        if len(active[l]) < 2:
            continue
        scores = poc.score_layer(net, zbar, l, active[l], ridge)
        for k, (S, a, c, rest, resid) in scores.items():
            all_S.append(S)
            if best is None or S < best[0]:
                w2 = float((net.fcs[l + 1].weight.data[:, k] ** 2).sum())
                z2 = float((zbar[l][:, k] ** 2).sum())
                rr = float((resid ** 2).sum()) / (z2 + EPS)
                best = (S, w2, rr)
    return best[0], float(np.median(all_S)), best[1], best[2]


def detect(series: np.ndarray, n_base: int, factor: float,
           sustain: int = 2) -> int:
    """初期 n_base 点の中央値 x factor を sustain 点連続で超える最初の index."""
    base = float(np.median(series[:n_base]))
    thr = base * factor
    for i in range(len(series) - sustain + 1):
        if np.all(series[i:i + sustain] > thr):
            return i
    return -1


def decompose_fall(rem, min_S, w2, rr):
    """min S_k のピーク -> 谷の log 分解: dlogS = dlog(w2) + dlog(rr).

    ピークは前半 2/3 の argmax、谷はピーク以降の argmin。戻り値は
    (peak_removed, trough_removed, dlogS, dlog_w2, dlog_rr)。"""
    n = len(min_S)
    i_pk = int(np.argmax(min_S[: max(3, 2 * n // 3)]))
    i_tr = i_pk + int(np.argmin(min_S[i_pk:]))
    if i_tr <= i_pk:
        return None
    dS = float(np.log(min_S[i_tr] / min_S[i_pk]))
    dw = float(np.log(w2[i_tr] / w2[i_pk]))
    dr = float(np.log(rr[i_tr] / rr[i_pk]))
    return dict(peak=int(rem[i_pk]), trough=int(rem[i_tr]),
                dlogS=dS, dlog_w2=dw, dlog_rr=dr)


# ============================================================
# main
# ============================================================
def main():
    p = argparse.ArgumentParser(
        description="stop signals + min S_k decomposition for noise annealing")
    poc.add_poc_args(p)
    p.add_argument("--keep", type=int, default=4,
                   help="各層に残すユニット数（圧縮限界を超えるまで削る）")
    p.add_argument("--detect-factor", type=float, default=2.0,
                   help="検知閾値 = 初期ベースライン中央値 x この係数")
    args = fncl.finalize_args(p.parse_args(),
                              default_out="out/consolidation_stop_signal")
    if args.quick:
        args.epochs_per_step = 5

    device = torch.device(args.device)
    x_raw, target, x, t = fncl.make_task(device)
    H = args.hidden_dim
    keep = min(args.keep, H - 1)
    quota = {0: H - keep, 1: H - keep}
    n_remove = quota[0] + quota[1]
    log_every = max(1, args.epochs // 5)

    all_res = {}
    for seed in args.seed_list:
        print(f"\n===== seed {seed} =====")
        torch.manual_seed(seed)
        np.random.seed(seed)

        # --- 事前学習（PoC と同一） ---
        net = poc.build_net(H, args.sigma, args.crossing_h, args.num_samples,
                            device)
        trainer = poc.CovJacTrainer(net, x, t, lr=args.pre_lr, opt=args.opt,
                                    jac_ema=args.jac_ema)
        for e in range(args.epochs):
            loss = trainer.step()
            if e % log_every == 0 or e == args.epochs - 1:
                print(f"  [pretrain] epoch {e:5d} mse={loss:.5f}")
        trainer.close()
        base = float(np.mean(trainer.losses[-min(100, len(trainer.losses)):]))
        tol = base * args.drift_mult + args.drift_abs
        pred0 = fncl.predict(net, x, passes=16)
        print(f"  pretrained MSE={float(np.mean((pred0 - target) ** 2)):.5f} "
              f"tol={tol:.5f}; removing {n_remove}/{2 * H} units "
              f"(keep {keep}/layer)")

        # --- route_B を限界超過まで実行し、スナップごとにシグナルを記録 ---
        keys = ("removed", "mse", "e_func", "cf1", "cf2", "ct1", "ct2",
                "min_S", "med_S", "minS_w2", "minS_rr",
                "erank1", "erank2", "na1", "na2", "holds", "drift")
        sig = {k: [] for k in keys}
        seg = {"holds": 0, "loss_lo": 0}

        def measure(net_, removed):
            z, zbar = poc.collect_stats(net_, x, passes=args.stat_passes)
            active = {l: [i for i in range(H) if float(net_.sigma_vecs[l][i]) > 0]
                      for l in (0, 1)}
            m = poc.eval_net(net_, x, target, pred0)
            mn, md, w2, rr = score_details(net_, zbar, active, args.ridge)
            sig["removed"].append(removed)
            sig["mse"].append(m["task_mse"])
            sig["e_func"].append(m["e_func"])
            sig["cf1"].append(corr_fluct(z[0], active[0]))
            sig["cf2"].append(corr_fluct(z[1], active[1]))
            sig["ct1"].append(corr_tuning(zbar[0], active[0]))
            sig["ct2"].append(corr_tuning(zbar[1], active[1]))
            sig["min_S"].append(mn)
            sig["med_S"].append(md)
            sig["minS_w2"].append(w2)
            sig["minS_rr"].append(rr)
            sig["erank1"].append(effective_rank(zbar[0], active[0]))
            sig["erank2"].append(effective_rank(zbar[1], active[1]))
            sig["na1"].append(len(active[0]))
            sig["na2"].append(len(active[1]))

        def record(event, net_, tr, removed, anneal):
            if event == "start":
                measure(net_, 0)
                sig["holds"].append(0)
                sig["drift"].append(float(tr.losses[-1]) if tr.losses
                                    else float("nan"))
                seg["loss_lo"] = len(tr.losses)
            elif event == "hold":
                seg["holds"] += 1
            elif event == "snap":
                measure(net_, removed)
                lo = seg["loss_lo"]
                sig["holds"].append(seg["holds"])
                sig["drift"].append(float(np.max(tr.losses[lo:]))
                                    if len(tr.losses) > lo else float("nan"))
                seg["holds"] = 0
                seg["loss_lo"] = len(tr.losses)
                if removed % 4 == 0:
                    print(f"  [route_B] removed {removed:2d}/{n_remove} "
                          f"epoch {len(tr.losses):5d} mse={sig['mse'][-1]:.5f} "
                          f"minS={sig['min_S'][-1]:.4f} "
                          f"medS={sig['med_S'][-1]:.4f} "
                          f"er2={sig['erank2'][-1]:.1f}/{sig['na2'][-1]}")

        curve, losses, snaps, holds_total = poc.run_route_B(
            net, x, target, pred0, quota, tol, args, record=record)
        print(f"  done: {n_remove} removed in {len(losses)} epochs "
              f"(holds={holds_total})")

        # --- 分析 1: 膝と各シグナルの検知点 ---
        rem = np.asarray(sig["removed"], dtype=int)
        mse = np.asarray(sig["mse"])
        n_base = max(4, len(rem) // 8)
        knee_i = -1
        for i in range(1, len(mse)):
            if np.all(mse[i:] > tol) and mse[i] > tol:
                knee_i = i
                break
        knee = int(rem[knee_i]) if knee_i >= 0 else None

        signals = {
            "corr_fluct(max)": np.maximum(np.asarray(sig["cf1"]),
                                          np.asarray(sig["cf2"])),
            "corr_tuning(max)": np.maximum(np.asarray(sig["ct1"]),
                                           np.asarray(sig["ct2"])),
            "min_S": np.asarray(sig["min_S"]),
            "holds": np.asarray(sig["holds"], dtype=float),
        }
        det = {}
        for name, s in signals.items():
            i = detect(s, n_base, args.detect_factor)
            det[name] = {"removed": (int(rem[i]) if i >= 0 else None),
                         "lead": (None if (i < 0 or knee is None)
                                  else knee - int(rem[i]))}

        # --- 分析 2: min S_k のピーク -> 谷の log 分解（(b) vs (c)） ---
        dec = decompose_fall(rem, np.asarray(sig["min_S"]),
                             np.asarray(sig["minS_w2"]),
                             np.asarray(sig["minS_rr"]))
        all_res[seed] = {"tol": tol, "knee": knee, "detect": det,
                         "decompose": dec,
                         "signals": {k: list(map(float, v))
                                     for k, v in sig.items()}}
        print(f"  knee={knee}  " +
              "  ".join(f"{n}: det={d['removed']} lead={d['lead']}"
                        for n, d in det.items()))
        if dec:
            print(f"  minS fall {dec['peak']}->{dec['trough']}: "
                  f"dlogS={dec['dlogS']:.2f} = dlog_w2 {dec['dlog_w2']:.2f} "
                  f"+ dlog_rr {dec['dlog_rr']:.2f}")

        # --- 図（先頭 seed のみ） ---
        if seed == args.seed_list[0]:
            fig, axes = plt.subplots(6, 1, figsize=(9, 15), sharex=True)
            ax = axes[0]
            ax.plot(rem, mse, "o-", ms=3, label="task MSE at snap")
            ax.axhline(tol, ls="--", c="k", lw=0.8, label="tolerance")
            if knee is not None:
                ax.axvline(knee, color="r", lw=1.0, ls=":",
                           label=f"knee (removed={knee})")
            ax.set_yscale("log")
            ax.set_ylabel("task MSE")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3, which="both")

            ax = axes[1]
            ax.plot(rem, sig["cf1"], "o-", ms=3, label="corr_fluct layer 1")
            ax.plot(rem, sig["cf2"], "s-", ms=3, label="corr_fluct layer 2")
            d = det["corr_fluct(max)"]["removed"]
            if d is not None:
                ax.axvline(d, color="g", lw=1.0, ls=":",
                           label=f"detection (removed={d})")
            if knee is not None:
                ax.axvline(knee, color="r", lw=1.0, ls=":")
            ax.set_ylabel("per-input |corr|")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

            ax = axes[2]
            ax.plot(rem, sig["min_S"], "^-", ms=3, c="tab:brown",
                    label="min S_k")
            ax.plot(rem, sig["med_S"], "v-", ms=3, c="tab:gray",
                    label="median S_k")
            ax.set_yscale("log")
            if dec:
                ax.axvline(dec["peak"], color="purple", lw=0.8, ls="--",
                           label="minS peak/trough")
                ax.axvline(dec["trough"], color="purple", lw=0.8, ls="--")
            if knee is not None:
                ax.axvline(knee, color="r", lw=1.0, ls=":")
            ax.set_ylabel("S_k")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3, which="both")

            ax = axes[3]
            ax.plot(rem, sig["minS_w2"], "o-", ms=3,
                    label="argmin unit ||w_k||^2 (channel c)")
            ax.plot(rem, sig["minS_rr"], "s-", ms=3,
                    label="argmin unit ||r_k||^2/||z_k||^2 (channel b)")
            ax.set_yscale("log")
            if dec:
                ax.axvline(dec["peak"], color="purple", lw=0.8, ls="--")
                ax.axvline(dec["trough"], color="purple", lw=0.8, ls="--")
            if knee is not None:
                ax.axvline(knee, color="r", lw=1.0, ls=":")
            ax.set_ylabel("S_k factors")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3, which="both")

            ax = axes[4]
            ax.plot(rem, sig["erank1"], "o-", ms=3, label="effective rank L1")
            ax.plot(rem, sig["erank2"], "s-", ms=3, label="effective rank L2")
            ax.plot(rem, sig["na1"], ":", c="tab:blue", lw=1.0,
                    label="survivors L1")
            ax.plot(rem, sig["na2"], ":", c="tab:orange", lw=1.0,
                    label="survivors L2")
            if knee is not None:
                ax.axvline(knee, color="r", lw=1.0, ls=":")
            ax.set_ylabel("effective rank")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

            ax = axes[5]
            ax.bar(rem, sig["holds"], width=0.8, alpha=0.6,
                   label="holds per unit")
            ax2 = ax.twinx()
            ax2.plot(rem, sig["drift"], "r.-", ms=3, lw=0.8,
                     label="max loss in segment")
            ax2.set_yscale("log")
            if knee is not None:
                ax.axvline(knee, color="r", lw=1.0, ls=":")
            ax.set_xlabel("units removed")
            ax.set_ylabel("holds")
            ax2.set_ylabel("max eval MSE")
            ax.legend(fontsize=8, loc="upper left")
            ax2.legend(fontsize=8, loc="upper right")
            ax.grid(alpha=0.3)

            fig.suptitle(
                f"Stop signals & min S_k decomposition "
                f"(H={H}, keep {keep}/layer, seed {seed})", fontsize=11)
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.975))
            savefig(fig, args.out_dir / "fig_stop_signals.png")

    # --- 表 ---
    sigs = list(next(iter(all_res.values()))["detect"].keys())
    lines = [f"**Stop-signal verification** (H={H}, keep {keep}/layer, "
             f"remove {n_remove}/{2 * H}, detect-factor={args.detect_factor}, "
             f"seeds={args.seed_list})", "",
             "| seed | knee | " + " | ".join(f"{s} det (lead)" for s in sigs)
             + " |",
             "|---" * (2 + len(sigs)) + "|"]
    for seed, r in all_res.items():
        cells = " | ".join(
            f"{r['detect'][s]['removed']} ({r['detect'][s]['lead']})"
            for s in sigs)
        lines.append(f"| {seed} | {r['knee']} | {cells} |")
    lines += ["", "det = シグナルがベースライン x factor を持続超過した削除数、",
              "lead = knee - det（正 = 膝より先に検知）。", "",
              "**min S_k fall decomposition** "
              "(dlogS = dlog_w2 [channel c] + dlog_rr [channel b])", "",
              "| seed | peak->trough | dlogS | dlog_w2 (c) | dlog_rr (b) | "
              "share of (b) |",
              "|---|---|---|---|---|---|"]
    for seed, r in all_res.items():
        d = r["decompose"]
        if d is None:
            lines.append(f"| {seed} | — | — | — | — | — |")
        else:
            share = d["dlog_rr"] / d["dlogS"] if d["dlogS"] != 0 else float("nan")
            lines.append(f"| {seed} | {d['peak']} -> {d['trough']} "
                         f"| {d['dlogS']:.2f} | {d['dlog_w2']:.2f} "
                         f"| {d['dlog_rr']:.2f} | {share:.0%} |")
    table = "\n".join(lines) + "\n"
    print("\n" + table)
    write_text(args.out_dir / "table_stop.md", table)
    save_json(args.out_dir / "results.json",
              {"config": fncl.config_dict(args),
               "results": {str(s): all_res[s] for s in all_res}})


if __name__ == "__main__":
    main()
