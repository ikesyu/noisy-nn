"""
consolidation_floor_test.py — 圧縮フロア予測とストリーミング有効ランクの検証
(docs/idea_consolidation.md §12.6.7 の「残る検証課題」)

検証 1（フロア予測）: §12.6.7 の発見「有効ランクはタスク内在次元に張り付き、
膝は生存数 ≈ 有効ランクで起こる」から導かれる予測 —
**事前学習の時点で測った有効ランクが、その層の圧縮フロア（許容誤差内に
留まれる最小生存ユニット数）を予告する** — を直接検証する。

  予測: 層1 のフロア ≈ ceil(er1) ≈ 4（er1 ≈ 3.5–4）
        層2 のフロア ≈ ceil(er2) ≈ 2（er2 ≈ 1.9–2.0）

方法: 層ごとに独立に route_B で限界超過までアニールする（他層は無傷）。
  条件 L1: 層1 のみ 32 -> 2（層2 は 32 のまま）
  条件 L2: 層2 のみ 32 -> 1（層1 は 32 のまま）
スナップごとに (生存数, タスク MSE, 有効ランク) を記録し、
フロア = 「スナップ MSE <= tol を満たす最小の生存数」を事前学習時の
有効ランクと比較する。

検証 2（ストリーミング化）: 有効ランクをバッチ計算（統計収集専用 forward +
固有値分解）ではなく、**訓練パス自身の活性から O(H^2) の EMA で蓄積**する
（§7.8 の単一ループ統合と同じ統計基盤）。各スナップでバッチ版と比較し、
一致度（平均・最大絶対誤差）を報告する。

生成物 (out/consolidation_floor_test/):
  fig_floor.png    層別の圧縮曲線（MSE vs 生存数、有効ランク線つき）+
                   バッチ/ストリーミング有効ランクの重ね描き
  table_floor.md   seed x 層のフロア vs 事前学習有効ランク、一致度
  results.json

実行例:
  python tmp/consolidation_floor_test.py --quick
  python tmp/consolidation_floor_test.py --seeds 0,1,2
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
from consolidation_stop_signal import effective_rank  # noqa: E402
from nnn.credit import EPS  # noqa: E402


# ============================================================
# ストリーミング有効ランク付きトレーナ（検証 2）
# ============================================================
class StreamingRankTrainer(poc.CovJacTrainer):
    """CovJacTrainer + 期待活性の 1・2 次モーメントの EMA 蓄積.

    訓練 forward が既に持つ per-sample 活性 (cap.z) から毎エポック
    zbar = mean_T(z) の平均ベクトル m と 2 次モーメント行列 M を EMA 蓄積し、
    Cov = M - m m^T の固有値から有効ランクを返す。統計収集専用の forward は
    使わない（§7.8 の単一ループ統合と同じ立て付け）。追加状態は層あたり
    O(H^2)、固有値分解は評価時のみ。
    """
    rank_beta = 0.9

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.M = [None, None]
        self.m = [None, None]

    def step(self):
        loss = super().step()
        with torch.no_grad():
            for l in range(self.cap.n_hidden):
                zb = self.cap.z[l].mean(dim=1)              # [N, H]
                M_e = zb.T @ zb / zb.shape[0]
                m_e = zb.mean(dim=0)
                if self.M[l] is None:
                    self.M[l], self.m[l] = M_e, m_e
                else:
                    b = self.rank_beta
                    self.M[l] = b * self.M[l] + (1.0 - b) * M_e
                    self.m[l] = b * self.m[l] + (1.0 - b) * m_e
        return loss

    def erank(self, l: int, active: list) -> float:
        if self.M[l] is None or len(active) < 1:
            return float("nan")
        idx = torch.tensor(active, dtype=torch.long)
        C = (self.M[l][idx][:, idx]
             - torch.outer(self.m[l][idx], self.m[l][idx]))
        ev = torch.linalg.eigvalsh(C).clamp(min=0.0)
        p = ev / (ev.sum() + EPS)
        p = p[p > 1e-12]
        return float(torch.exp(-(p * p.log()).sum()))


# ============================================================
# 1 条件（1 層のみ限界超過アニール）
# ============================================================
def run_condition(net, x, target, pred0, layer: int, keep: int, tol, args):
    """層 layer のみ H -> keep までアニールし、スナップ列を記録して返す."""
    H = args.hidden_dim
    quota = {layer: H - keep, 1 - layer: 0}
    rec = {"na": [], "mse": [], "er_batch": [], "er_stream": []}

    def record(event, net_, tr, removed, anneal):
        if event not in ("start", "snap"):
            return
        active = [i for i in range(H) if float(net_.sigma_vecs[layer][i]) > 0]
        _, zbar = poc.collect_stats(net_, x, passes=args.stat_passes)
        m = poc.eval_net(net_, x, target, pred0)
        rec["na"].append(len(active))
        rec["mse"].append(m["task_mse"])
        rec["er_batch"].append(effective_rank(zbar[layer], active))
        rec["er_stream"].append(tr.erank(layer, active))
        if removed % 4 == 0 or event == "start":
            print(f"    [L{layer + 1}] na={len(active):2d} "
                  f"mse={m['task_mse']:.5f} "
                  f"er_b={rec['er_batch'][-1]:.2f} "
                  f"er_s={rec['er_stream'][-1]:.2f}")

    curve, losses, snaps, holds = poc.run_route_B(
        net, x, target, pred0, quota, tol, args, record=record)
    return rec, len(losses), holds


def floor_of(rec, tol) -> int:
    """フロア = スナップ MSE <= tol を満たす最小の生存数."""
    ok = [na for na, m in zip(rec["na"], rec["mse"]) if m <= tol]
    return int(min(ok)) if ok else int(max(rec["na"]))


# ============================================================
# main
# ============================================================
def main():
    p = argparse.ArgumentParser(
        description="compression-floor prediction & streaming effective rank")
    poc.add_poc_args(p)
    p.add_argument("--keep-l1", type=int, default=2,
                   help="条件 L1 で層1に残す最小ユニット数")
    p.add_argument("--keep-l2", type=int, default=1,
                   help="条件 L2 で層2に残す最小ユニット数")
    p.add_argument("--rank-beta", type=float, default=0.9,
                   help="ストリーミング有効ランクの EMA 係数")
    args = fncl.finalize_args(p.parse_args(),
                              default_out="out/consolidation_floor_test")
    if args.quick:
        args.epochs_per_step = 5
    StreamingRankTrainer.rank_beta = args.rank_beta

    device = torch.device(args.device)
    x_raw, target, x, t = fncl.make_task(device)
    H = args.hidden_dim
    log_every = max(1, args.epochs // 5)

    # run_route_B が内部生成するトレーナをストリーミング版へ差し替える
    poc.CovJacTrainer = StreamingRankTrainer

    all_res = {}
    for seed in args.seed_list:
        print(f"\n===== seed {seed} =====")
        torch.manual_seed(seed)
        np.random.seed(seed)

        net = poc.build_net(H, args.sigma, args.crossing_h, args.num_samples,
                            device)
        trainer = StreamingRankTrainer(net, x, t, lr=args.pre_lr, opt=args.opt,
                                       jac_ema=args.jac_ema)
        for e in range(args.epochs):
            loss = trainer.step()
            if e % log_every == 0 or e == args.epochs - 1:
                print(f"  [pretrain] epoch {e:5d} mse={loss:.5f}")
        trainer.close()
        base = float(np.mean(trainer.losses[-min(100, len(trainer.losses)):]))
        tol = base * args.drift_mult + args.drift_abs
        ckpt = poc.checkpoint(net)
        pred0 = fncl.predict(net, x, passes=16)

        # 事前学習時の有効ランク（バッチ / ストリーミング）
        _, zbar0 = poc.collect_stats(net, x, passes=args.stat_passes)
        er0 = {l: effective_rank(zbar0[l], list(range(H))) for l in (0, 1)}
        er0_s = {l: trainer.erank(l, list(range(H))) for l in (0, 1)}
        print(f"  pretrained: tol={tol:.5f}  er_batch=(L1 {er0[0]:.2f}, "
              f"L2 {er0[1]:.2f})  er_stream=(L1 {er0_s[0]:.2f}, "
              f"L2 {er0_s[1]:.2f})")

        res = {"tol": tol, "er0_batch": {str(l): er0[l] for l in (0, 1)},
               "er0_stream": {str(l): er0_s[l] for l in (0, 1)}}
        for layer, keep in ((1, args.keep_l2), (0, args.keep_l1)):
            print(f"  --- condition L{layer + 1}: {H} -> {keep} "
                  f"(other layer untouched) ---")
            poc.restore(net, ckpt)
            rec, epochs, holds = run_condition(
                net, x, target, pred0, layer, keep, tol, args)
            fl = floor_of(rec, tol)
            # 開始スナップは route_B のトレーナが未学習で EMA 未初期化 (nan)
            agree = np.abs(np.asarray(rec["er_batch"])
                           - np.asarray(rec["er_stream"]))
            mae = float(np.nanmean(agree)) if np.any(~np.isnan(agree)) else float("nan")
            mx = float(np.nanmax(agree)) if np.any(~np.isnan(agree)) else float("nan")
            res[f"L{layer + 1}"] = {
                "keep": keep, "floor": fl, "epochs": epochs, "holds": holds,
                "er_pre": er0[layer],
                "stream_mae": mae,
                "stream_max": mx,
                "rec": {k: list(map(float, v)) for k, v in rec.items()}}
            print(f"    floor={fl} (er_pre={er0[layer]:.2f})  "
                  f"stream |err|: mean={mae:.3f} max={mx:.3f}")
        all_res[seed] = res

        # --- 図（先頭 seed のみ） ---
        if seed == args.seed_list[0]:
            fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex="col")
            for col, layer in enumerate((0, 1)):
                r = res[f"L{layer + 1}"]["rec"]
                na = np.asarray(r["na"])
                ax = axes[0][col]
                ax.plot(na, r["mse"], "o-", ms=3, label="task MSE at snap")
                ax.axhline(tol, ls="--", c="k", lw=0.8, label="tolerance")
                ax.axvline(er0[layer], color="purple", lw=1.2, ls="--",
                           label=f"pretrain eff. rank = {er0[layer]:.2f}")
                ax.axvline(res[f"L{layer + 1}"]["floor"], color="r", lw=1.0,
                           ls=":",
                           label=f"floor = {res[f'L{layer + 1}']['floor']}")
                ax.set_yscale("log")
                ax.set_title(f"anneal layer {layer + 1} only "
                             f"({H} -> {res[f'L{layer + 1}']['keep']})")
                ax.set_ylabel("task MSE")
                ax.legend(fontsize=7)
                ax.grid(alpha=0.3, which="both")
                ax.invert_xaxis()

                ax = axes[1][col]
                ax.plot(na, r["er_batch"], "o-", ms=3, label="eff. rank (batch)")
                ax.plot(na, r["er_stream"], "s--", ms=3,
                        label="eff. rank (streaming EMA)")
                ax.plot(na, na, ":", c="gray", lw=1.0, label="survivors")
                ax.axvline(er0[layer], color="purple", lw=1.2, ls="--")
                ax.set_xlabel(f"survivors in layer {layer + 1}")
                ax.set_ylabel("effective rank")
                ax.legend(fontsize=7)
                ax.grid(alpha=0.3)
                ax.invert_xaxis()
            fig.suptitle(
                f"Compression floor vs pretrain effective rank "
                f"(H={H}, seed {seed})", fontsize=11)
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
            savefig(fig, args.out_dir / "fig_floor.png")

    # --- 表 ---
    lines = [f"**Compression-floor prediction & streaming effective rank** "
             f"(H={H}, seeds={args.seed_list}, rank-beta={args.rank_beta})", "",
             "| seed | layer | er (pretrain) | predicted floor ceil(er) "
             "| observed floor | holds | stream MAE / max |",
             "|---|---|---|---|---|---|---|"]
    for seed, r in all_res.items():
        for lname in ("L1", "L2"):
            d = r[lname]
            lines.append(
                f"| {seed} | {lname} | {d['er_pre']:.2f} "
                f"| {int(np.ceil(d['er_pre']))} | {d['floor']} "
                f"| {d['holds']} | {d['stream_mae']:.3f} / "
                f"{d['stream_max']:.3f} |")
    lines += ["", "observed floor = スナップ MSE <= tol を満たす最小の生存数。",
              "stream MAE/max = ストリーミング有効ランクとバッチ版の絶対誤差。"]
    table = "\n".join(lines) + "\n"
    print("\n" + table)
    write_text(args.out_dir / "table_floor.md", table)
    save_json(args.out_dir / "results.json",
              {"config": fncl.config_dict(args),
               "results": {str(s): all_res[s] for s in all_res}})


if __name__ == "__main__":
    main()
