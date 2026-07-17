"""
consolidation_rank_correction.py — 有効ランクのノイズフロア補正の検証
(docs/idea_consolidation.md §12.6.8(4) の「残る改良点」)

問題: 有効ランクはチューニング共分散 C = Cov_n(zbar) の固有値から計算するが、
有限 T の zbar = mean_T(z) にはサンプリングノイズ eps が乗り、

    Cov_n(zbar) = C_true + SamplingCov(eps)

の右辺第2項がスペクトルを底上げして有効ランクを**過大評価**する（§12.6.8 で
観測した +0.39 の正バイアス）。この項は同じサンプルから計算できる：

    SamplingCov ~= (G0 + G1 + G1^T) / T
    G0 = mean_n Cov_t(z_t)          （lag-0 ゆらぎ共分散）
    G1 = mean_n Cov_t(z_t, z_{t+1}) （lag-1; 交差活性 z_t は隣接サンプルを
                                      共有する xor 差分なので lag-1 相関を持つ。
                                      cyclic xor なので周回ペアも含める）

検証内容（各 seed）:
  A. バッチ推定量の比較（事前学習ネット、生存部分集合 na in {32,16,8,4}）:
       naive         : 補正なし（P=1, T=64）
       diag(G0)      : 対角・lag-0 のみ（二項分散の素朴版）
       diag(G0+2G1)  : 対角・lag-1 込み
       full          : 全行列 (G0+G1+G1^T)/T
     参照 er_ref は P=64（実効 4096 サンプル/入力）+ full 補正。
     部分集合とサンプルを R 回引き直してバイアス±SD を推定。
  B. ストリーミング推定量: 訓練パスの活性から M, m に加えて G0, G1 も EMA 蓄積
     し（層あたり O(H^2)）、300 epoch の継続学習後に naive / corrected を
     er_ref と比較する。

生成物 (out/consolidation_rank_correction/):
  fig_correction.png   バイアス vs 生存数（推定量別）
  table_correction.md  バッチ/ストリーミングのバイアス表
  results.json

実行例:
  python tmp/consolidation_rank_correction.py --quick
  python tmp/consolidation_rank_correction.py --seeds 0,1,2
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
from consolidation_floor_test import StreamingRankTrainer  # noqa: E402
from nnn.stats import Capture  # noqa: E402
from nnn.credit import EPS  # noqa: E402


# ============================================================
# サンプル収集（パス構造を保持: [N, P, T, H]）
# ============================================================
def collect_passes(net, x, passes: int):
    cap = Capture(net)
    zs = [[] for _ in range(cap.n_hidden)]
    with torch.no_grad():
        for _ in range(passes):
            net(x)
            for l in range(cap.n_hidden):
                zs[l].append(cap.z[l])                     # [N, T, H]
    cap.remove()
    return [torch.stack(zs[l], dim=1) for l in range(cap.n_hidden)]  # [N,P,T,H]


def fluct_covs(z4: torch.Tensor):
    """lag-0 / lag-1（cyclic）のゆらぎ共分散 G0, G1（入力・パスでプール）.

    z4: [N, P, T, H]。z_t は cyclic xor 由来なので lag-1 も周回込みで取る。
    """
    cz = z4 - z4.mean(dim=2, keepdim=True)                 # T 方向中心化
    N, P, T, H = cz.shape
    G0 = torch.einsum("npti,nptj->ij", cz, cz) / (N * P * T)
    cz1 = torch.roll(cz, shifts=-1, dims=2)                # z_{t+1}（cyclic）
    G1 = torch.einsum("npti,nptj->ij", cz, cz1) / (N * P * T)
    return G0, G1


def erank_of_cov(C: torch.Tensor, idx) -> float:
    Ci = C[idx][:, idx]
    ev = torch.linalg.eigvalsh(Ci).clamp(min=0.0)
    p = ev / (ev.sum() + EPS)
    p = p[p > 1e-12]
    return float(torch.exp(-(p * p.log()).sum()))


def tuning_cov(z4: torch.Tensor):
    """zbar のチューニング共分散（入力方向、中心化）と実効サンプル数 T_eff."""
    zbar = z4.mean(dim=(1, 2))                             # [N, H]
    cz = zbar - zbar.mean(dim=0, keepdim=True)
    C = cz.T @ cz / cz.shape[0]
    return C, z4.shape[1] * z4.shape[2]


def estimators(z4: torch.Tensor):
    """4 種のチューニング共分散推定量 {name: C_hat} を返す."""
    C_raw, T_eff = tuning_cov(z4)
    G0, G1 = fluct_covs(z4)
    corr_full = (G0 + G1 + G1.T) / T_eff
    return {
        "naive": C_raw,
        "diag_G0": C_raw - torch.diag(torch.diag(G0)) / T_eff,
        "diag_G0_2G1": C_raw - torch.diag(torch.diag(G0) + 2 * torch.diag(G1)) / T_eff,
        "full": C_raw - corr_full,
    }


# ============================================================
# ストリーミング（G0, G1 も EMA 蓄積）
# ============================================================
class CorrectedStreamingTrainer(StreamingRankTrainer):
    """StreamingRankTrainer + lag-0/lag-1 ゆらぎ共分散の EMA（層あたり O(H^2)）."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.G0 = [None, None]
        self.G1 = [None, None]
        self.T = None

    def step(self):
        loss = super().step()
        with torch.no_grad():
            for l in range(self.cap.n_hidden):
                z = self.cap.z[l]                          # [N, T, H]
                self.T = z.shape[1]
                cz = z - z.mean(dim=1, keepdim=True)
                G0_e = torch.einsum("nti,ntj->ij", cz, cz) / (z.shape[0] * z.shape[1])
                cz1 = torch.roll(cz, shifts=-1, dims=1)
                G1_e = torch.einsum("nti,ntj->ij", cz, cz1) / (z.shape[0] * z.shape[1])
                if self.G0[l] is None:
                    self.G0[l], self.G1[l] = G0_e, G1_e
                else:
                    b = self.rank_beta
                    self.G0[l] = b * self.G0[l] + (1 - b) * G0_e
                    self.G1[l] = b * self.G1[l] + (1 - b) * G1_e
        return loss

    def erank_corrected(self, l: int, active: list) -> float:
        idx = torch.tensor(active, dtype=torch.long)
        C = (self.M[l][idx][:, idx]
             - torch.outer(self.m[l][idx], self.m[l][idx]))
        # M は「ノイズ入り zbar の Gram」の EMA なので、各エポックのサンプリング
        # 項 (G0+G1+G1^T)/T は平均してもバイアスとして 1/T のまま残る（EMA が
        # 縮めるのは分散だけ）。したがって補正の分母は T。
        G = (self.G0[l] + self.G1[l] + self.G1[l].T)[idx][:, idx]
        C = C - G / self.T
        ev = torch.linalg.eigvalsh(C).clamp(min=0.0)
        p = ev / (ev.sum() + EPS)
        p = p[p > 1e-12]
        return float(torch.exp(-(p * p.log()).sum()))


# ============================================================
# main
# ============================================================
def main():
    p = argparse.ArgumentParser(
        description="noise-floor correction for the effective-rank gauge")
    poc.add_poc_args(p)
    p.add_argument("--repeats", type=int, default=10,
                   help="バッチ比較のサンプル引き直し回数")
    p.add_argument("--ref-passes", type=int, default=64,
                   help="参照 er_ref のパス数")
    p.add_argument("--stream-epochs", type=int, default=300,
                   help="ストリーミング評価用の継続学習 epoch 数")
    p.add_argument("--rank-beta", type=float, default=0.9)
    args = fncl.finalize_args(p.parse_args(),
                              default_out="out/consolidation_rank_correction")
    if args.quick:
        args.repeats = 3
        args.ref_passes = 16
        args.stream_epochs = 50
    CorrectedStreamingTrainer.rank_beta = args.rank_beta

    device = torch.device(args.device)
    x_raw, target, x, t = fncl.make_task(device)
    H = args.hidden_dim
    log_every = max(1, args.epochs // 5)
    subset_sizes = [s for s in (H, H // 2, H // 4, H // 8) if s >= 2]
    est_names = ["naive", "diag_G0", "diag_G0_2G1", "full"]

    all_res = {}
    for seed in args.seed_list:
        print(f"\n===== seed {seed} =====")
        torch.manual_seed(seed)
        np.random.seed(seed)
        rng = np.random.default_rng(1000 + seed)

        net = poc.build_net(H, args.sigma, args.crossing_h, args.num_samples,
                            device)
        trainer = CorrectedStreamingTrainer(net, x, t, lr=args.pre_lr,
                                            opt=args.opt, jac_ema=args.jac_ema)
        for e in range(args.epochs):
            loss = trainer.step()
            if e % log_every == 0 or e == args.epochs - 1:
                print(f"  [pretrain] epoch {e:5d} mse={loss:.5f}")

        # --- A. バッチ推定量の比較 --------------------------------------
        z4_ref = collect_passes(net, x, args.ref_passes)
        bias = {l: {n: {sz: [] for sz in subset_sizes} for n in est_names}
                for l in (0, 1)}
        for r in range(args.repeats):
            z4 = collect_passes(net, x, 1)                # P=1: T=num_samples
            for l in (0, 1):
                Cs = estimators(z4[l])
                C_ref = estimators(z4_ref[l])["full"]
                for sz in subset_sizes:
                    idx = (list(range(H)) if sz == H else
                           sorted(rng.choice(H, size=sz, replace=False)))
                    idx_t = torch.tensor(idx, dtype=torch.long)
                    ref = erank_of_cov(C_ref, idx_t)
                    for n in est_names:
                        bias[l][n][sz].append(erank_of_cov(Cs[n], idx_t) - ref)
        for l in (0, 1):
            for n in est_names:
                msg = "  ".join(
                    f"na={sz}: {np.mean(bias[l][n][sz]):+.3f}±"
                    f"{np.std(bias[l][n][sz]):.3f}" for sz in subset_sizes)
                print(f"  [batch L{l + 1}] {n:12s} {msg}")

        # --- B. ストリーミング推定量 ------------------------------------
        trainer.run(args.stream_epochs)                    # 定常状態で EMA を貯める
        stream = {}
        for l in (0, 1):
            act = list(range(H))
            C_ref = estimators(z4_ref[l])["full"]
            ref = erank_of_cov(C_ref, torch.tensor(act))
            naive = trainer.erank(l, act)
            corr = trainer.erank_corrected(l, act)
            stream[l] = {"ref": ref, "naive": naive - ref, "corrected": corr - ref}
            print(f"  [stream L{l + 1}] ref={ref:.2f}  naive bias={naive - ref:+.3f}"
                  f"  corrected bias={corr - ref:+.3f}")
        trainer.close()

        all_res[seed] = {
            "bias": {f"L{l + 1}": {n: {str(sz): [float(v) for v in bias[l][n][sz]]
                                       for sz in subset_sizes}
                                   for n in est_names} for l in (0, 1)},
            "stream": {f"L{l + 1}": {k: float(v) for k, v in stream[l].items()}
                       for l in (0, 1)}}

        # --- 図（先頭 seed のみ） ----------------------------------------
        if seed == args.seed_list[0]:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
            for col, l in enumerate((0, 1)):
                ax = axes[col]
                for n in est_names:
                    means = [np.mean(bias[l][n][sz]) for sz in subset_sizes]
                    sds = [np.std(bias[l][n][sz]) for sz in subset_sizes]
                    ax.errorbar(subset_sizes, means, yerr=sds, marker="o",
                                ms=4, capsize=3, label=n)
                ax.axhline(0.0, c="k", lw=0.8)
                ax.scatter([H], [stream[l]["naive"]], marker="*", s=110,
                           c="tab:red", zorder=5, label="stream naive")
                ax.scatter([H], [stream[l]["corrected"]], marker="*", s=110,
                           c="tab:green", zorder=5, label="stream corrected")
                ax.set_xlabel("survivors (subset size)")
                ax.set_title(f"layer {l + 1}")
                if col == 0:
                    ax.set_ylabel("effective-rank bias vs reference")
                ax.legend(fontsize=7)
                ax.grid(alpha=0.3)
            fig.suptitle(f"Noise-floor correction of the effective rank "
                         f"(H={H}, T={args.num_samples}, seed {seed})",
                         fontsize=11)
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
            savefig(fig, args.out_dir / "fig_correction.png")

    # --- 表 -------------------------------------------------------------
    lines = [f"**Noise-floor correction** (H={H}, T={args.num_samples}, "
             f"repeats={args.repeats}, seeds={args.seed_list})", "",
             "batch bias (mean over repeats & seeds, na=H):", "",
             "| layer | " + " | ".join(est_names) + " | stream naive | "
             "stream corrected |",
             "|---" * (len(est_names) + 3) + "|"]
    for l in (0, 1):
        cells = []
        for n in est_names:
            v = [np.mean(all_res[s]["bias"][f"L{l + 1}"][n][str(H)])
                 for s in all_res]
            cells.append(f"{np.mean(v):+.3f}")
        sn = np.mean([all_res[s]["stream"][f"L{l + 1}"]["naive"]
                      for s in all_res])
        sc = np.mean([all_res[s]["stream"][f"L{l + 1}"]["corrected"]
                      for s in all_res])
        lines.append(f"| L{l + 1} | " + " | ".join(cells)
                     + f" | {sn:+.3f} | {sc:+.3f} |")
    table = "\n".join(lines) + "\n"
    print("\n" + table)
    write_text(args.out_dir / "table_correction.md", table)
    save_json(args.out_dir / "results.json",
              {"config": fncl.config_dict(args),
               "results": {str(s): all_res[s] for s in all_res}})


if __name__ == "__main__":
    main()
