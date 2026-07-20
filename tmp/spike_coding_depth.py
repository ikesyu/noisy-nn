"""
spike_coding_depth.py — 項目(10): 深さが符号化様式を決めるか

仮説
----
多層 NNN では、上流ユニットの二値活動のゆらぎが下流へ**輸送**される
（コンソリデーション稿 §4.5 の「sigma=0 リーク」: sigma_k=0 の層2ユニットが
上流ゆらぎだけで発火し続けた）。したがって層 l の実効ノイズは

    sigma_eff,l^2 = sigma_l^2 + Var(輸送された上流ゆらぎ)

となり、**注入ノイズを増やさなくても深い層ほど分母が大きくなる**。
Gamma = |決定論的駆動の時間微分| / sigma_eff がこれに従って深さとともに
減るなら、予測は:

    **符号化様式は深さとともにタイミング側からレート側へ移る。**
    極端には sigma_2 = 0 でも、層2は輸送ゆらぎだけでレート符号として動く。

生物学的には「精密なタイミングは末梢の性質、レートは深部の性質」という
よく知られた対比に対応する（聴神経・網膜の精密なスパイク時刻 vs 皮質の
レート的応答）。本構想では、それが別々のニューロンモデルではなく
**同一機構でのノイズ輸送の帰結**として出てくる。

測り方（層ごとに同じ指標を出す）
--------------------------------
決定論的駆動と実効ノイズを、**試行集団から直接**測る（層に依らない定義）:

    d_l(t) を R 試行ぶん集め、
      決定論的駆動 = 試行平均 mean_r d_l(r,t)
      実効ノイズ   = 試行間 std_r d_l(r,t)      <- 輸送ゆらぎを自動的に含む
      Gamma_eff,l  = median_t,k |d/dt 試行平均| / median 試行間 std

この定義は層1では従来の Gamma = |w dx/dt|/sigma に一致し、深い層では
輸送ぶんを自動的に繰り入れる。符号化指標（CV・試行間再現性・timing frac・
決定論的交差の割合）は spike_coding_regime.py のものをそのまま再利用する。

アーム
------
  uniform : 全層に同じ sigma を注入（標準的な設定）
  input   : sigma_1 = sigma, sigma_{l>1} = 0（**本命**: 下流は輸送ゆらぎのみ）
  deep    : sigma_1 = 0, sigma_{l>1} = sigma（逆向きの対照）

使い方
------
    python tmp/spike_coding_depth.py --quick
    python tmp/spike_coding_depth.py --arms uniform,input,deep --layers 3
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
if __name__ == "__main__":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from nnn.activation import Crossing  # noqa: E402
import fncl_driver as fncl  # noqa: E402
from fncl_driver import save_json, savefig, write_text  # noqa: E402
from spike_coding_regime import (  # noqa: E402
    autocorr_time, causal_ma, info_decomposition, isi_stats, make_signal,
    reliability, timescale_params, _ridge_fit_eval)


class DeepTemporalNNN(nn.Module):
    """L 層の時間軸つき NNN。各層で ±h 交差、最終層の窓平均を線形読み出し。

    層ごとに独立な sigma を持てる（sigmas リスト）。層1のみ入力が
    決定論的で、層 l>=2 は上流の二値活動（ゆらぐ）を受ける。
    """

    def __init__(self, hidden: int, layers: int, sigmas, h: float,
                 window: int, device, shared: float = 0.0):
        super().__init__()
        self.H, self.L, self.h, self.window = hidden, layers, h, window
        self.sigmas = list(sigmas)
        # ユニット間で共有されるノイズの分散比 (0 = 独立, 1 = 完全共有)。
        # ファンインのプーリングは独立成分を sqrt(N) でしか増やさないが、
        # 共有成分は N で増える（平均で消せない）。深さの効果がどちらに
        # 転ぶかを決める鍵なので、これを対照として振る。
        self.shared = float(shared)
        # 層1: 刺激レンジ [-1,1] を覆うタイリング
        centers = torch.linspace(-1.0, 1.0, hidden, device=device)
        mag = 1.0 + 0.5 * torch.rand(hidden, device=device)
        sign = torch.where(torch.rand(hidden, device=device) < 0.5, -1.0, 1.0)
        w1 = mag * sign
        self.w1 = nn.Parameter(w1)
        self.b1 = nn.Parameter(-w1 * centers)
        # 層2以降: 全結合。交差率 ~0.1-0.3 の入力に対し帯 ±h を跨げるスケール
        self.Ws = nn.ParameterList([
            nn.Parameter(torch.randn(hidden, hidden, device=device)
                         * (2.0 / np.sqrt(hidden)))
            for _ in range(layers - 1)])
        self.bs = nn.ParameterList([
            nn.Parameter(torch.zeros(hidden, device=device))
            for _ in range(layers - 1)])
        self.a = nn.Parameter(torch.zeros(hidden, device=device))
        self.b_out = nn.Parameter(torch.zeros(1, device=device))

    def _noise(self, d: torch.Tensor) -> torch.Tensor:
        """周辺分散を 1 に保ったまま、ユニット間の共有成分比を self.shared に。"""
        eps = torch.randn_like(d)
        if self.shared <= 0.0:
            return eps
        sh = torch.randn(d.shape[0], d.shape[1], 1, device=d.device)
        return (np.sqrt(1.0 - self.shared) * eps
                + np.sqrt(self.shared) * sh.expand_as(d))

    def forward_all(self, x: torch.Tensor, sigmas=None):
        """x [N,T] -> ([d_l 注入前], [z_l], [dn_l 注入後])  各要素 [N,T,H]

        決定論的駆動は**注入前** d の試行平均（層1では厳密に決定論的、
        層 l>=2 では輸送ゆらぎを平均で落とした成分）、実効ノイズは
        **注入後** dn の試行間 std（注入 + 輸送の合計）から測る。
        層1の d を注入前で取ると試行間 std が 0 になってしまうので、
        ノイズ側は必ず dn を使う。
        """
        sig = self.sigmas if sigmas is None else sigmas
        ds, zs, dns = [], [], []
        d = x.unsqueeze(-1) * self.w1 + self.b1
        for l in range(self.L):
            if l > 0:
                d = zs[-1] @ self.Ws[l - 1].T + self.bs[l - 1]
            ds.append(d)
            dn = d + sig[l] * self._noise(d) if sig[l] > 0 else d
            dns.append(dn)
            zs.append(Crossing.apply(dn, self.h))
        return ds, zs, dns

    def forward(self, x: torch.Tensor, sigmas=None) -> torch.Tensor:
        _, zs, _ = self.forward_all(x, sigmas)
        return causal_ma(zs[-1], self.window) @ self.a + self.b_out


def train(net, x, y, trials, epochs, lr, log_every=0):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    xb = x.unsqueeze(0).repeat(trials, 1)
    yb = y.unsqueeze(0).repeat(trials, 1)
    losses = []
    for e in range(epochs):
        opt.zero_grad()
        loss = F.mse_loss(net(xb), yb)
        loss.backward()
        opt.step()
        losses.append(float(loss))
        if log_every and e % log_every == 0:
            print(f"      epoch {e:4d} mse={float(loss):.5f}")
    return losses


def drive_and_noise(net, x, trials, sigmas, device):
    """層ごとの決定論的駆動と実効ノイズを試行集団から測る（本稿の中核）。

    実効ノイズ = 同一刺激に対する試行間 std。層1では注入ノイズそのもの、
    深い層では **輸送された上流ゆらぎ + 自層の注入ノイズ** が自動的に入る。
    """
    with torch.no_grad():
        xb = x.unsqueeze(0).repeat(trials, 1)
        ds, zs, dns = net.forward_all(xb, sigmas)
        out = []
        for l, (d, dn) in enumerate(zip(ds, dns)):
            mu = d.mean(dim=0)                       # [T,H] 決定論的駆動
            sd = dn.std(dim=0)                       # [T,H] 実効ノイズ(注入+輸送)
            drift = (torch.diff(mu, dim=0).abs().median())
            sig_eff = sd.median()
            out.append({
                "layer": l + 1,
                "sigma_inj": float(sigmas[l]),
                "sigma_eff": float(sig_eff),
                "drive_step": float(drift),
                "gamma_eff": float(drift / (sig_eff + 1e-12)),
                # 注入ぶんを除いた輸送ゆらぎの寄与（分散で差し引く）
                "transport_std": float(torch.sqrt(torch.clamp(
                    sd.median() ** 2 - sigmas[l] ** 2, min=0.0))),
                "rate": float((zs[l] > 0).float().mean()),
            })
        return out, [(z > 0).float().cpu().numpy() for z in zs]


def analyze_layers(net, x, y, args, sigmas, tp, seed, device) -> list:
    stats, spikes = drive_and_noise(net, x, args.trials, sigmas, device)
    stim = y.cpu().numpy()
    rows = []
    for st, spk in zip(stats, spikes):
        rng = np.random.RandomState(seed)
        rel = reliability(spk, tp["rel_widths"])
        info = info_decomposition(spk, stim, args.n_sub, tp["sub_len"],
                                  tp["stride"], args.n_lvl, rng=rng)
        isi = isi_stats(spk)
        # 決定論的交差の割合: 全層 sigma=0 の反実仮想（完全に決定論的な網）
        with torch.no_grad():
            xb = x.unsqueeze(0).repeat(args.trials, 1)
            _, z0, _ = net.forward_all(xb, [0.0] * net.L)
            nu_sig = float((z0[st["layer"] - 1] > 0).float().mean())
        r = dict(st)
        r.update(info)
        r.update({"rel_fine": rel[float(tp["rel_widths"][0])],
                  "rel_coarse": rel[float(tp["rel_widths"][-1])],
                  "cv": isi["cv"], "mean_isi": isi["mean_isi"],
                  "nu_signal_only": nu_sig,
                  "signal_share": nu_sig / (st["rate"] + 1e-12),
                  "seed": seed})
        rows.append(r)
    return rows


def run_arm(arm: str, sigma: float, args, device, seed: int) -> list:
    torch.manual_seed(seed)
    np.random.seed(seed)
    x, y = make_signal(args.steps, args.speed, device, kind="sine")
    tp = timescale_params(autocorr_time(x.cpu().numpy()), args)
    L = args.layers
    if arm == "uniform":
        sigmas = [sigma] * L
    elif arm == "input":
        sigmas = [sigma] + [0.0] * (L - 1)
    elif arm == "deep":
        sigmas = [0.0] + [sigma] * (L - 1)
    else:
        raise ValueError(arm)
    net = DeepTemporalNNN(args.hidden_dim, L, sigmas, args.crossing_h,
                          tp["window"], device, shared=args.shared)
    losses = train(net, x, y, args.trials, args.epochs, args.lr,
                   log_every=args.epochs // 2 if args.verbose else 0)
    with torch.no_grad():
        xb = x.unsqueeze(0).repeat(args.trials, 1)
        r = causal_ma(net.forward_all(xb)[1][-1], net.window)
        X = r.reshape(-1, net.H).cpu().numpy()
        t = y.unsqueeze(0).repeat(args.trials, 1).reshape(-1).cpu().numpy()
    task = _ridge_fit_eval(X, t) / (float(np.var(t)) + 1e-12)
    print(f"    [{arm} sigma={sigma:g}] loss={np.mean(losses[-20:]):.4f} "
          f"refitMSE={task:.3f} sigmas={sigmas}")
    rows = analyze_layers(net, x, y, args, sigmas, tp, seed, device)
    for r in rows:
        r.update({"arm": arm, "sigma": sigma, "task_mse_refit": task,
                  "tau": tp["tau"], "shared": args.shared})
        print(f"      層{r['layer']}: sigma_inj={r['sigma_inj']:.3g} "
              f"sigma_eff={r['sigma_eff']:.3f} "
              f"(輸送 {r['transport_std']:.3f}) Gamma={r['gamma_eff']:.3f} "
              f"rate={r['rate']:.3f} CV={r['cv']:.2f} "
              f"rel={r['rel_fine']:.3f} timfr={r['timing_frac']:.3f} "
              f"sig_share={r['signal_share']:.2f}")
    return rows


def make_figures(rows, args):
    arms = sorted({r["arm"] for r in rows})
    sigmas = sorted({r["sigma"] for r in rows})
    layers = sorted({r["layer"] for r in rows})

    def agg(arm, sig, layer, key):
        v = [r[key] for r in rows if r["arm"] == arm and r["sigma"] == sig
             and r["layer"] == layer and np.isfinite(r[key])]
        return float(np.mean(v)) if v else np.nan

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    panels = [("gamma_eff", r"$\Gamma_{\rm eff}$ (log)", True),
              ("sigma_eff", "effective noise (log)", True),
              ("cv", "ISI CV", False),
              ("rel_fine", "trial reliability (fine)", False),
              ("timing_frac", "timing info fraction", False),
              ("signal_share", "signal-driven share", False)]
    for ax, (key, ttl, logy) in zip(axes.ravel(), panels):
        for arm in arms:
            for sig in sigmas:
                ys = [agg(arm, sig, l, key) for l in layers]
                ax.plot(layers, ys, "-o",
                        label=f"{arm} σ={sig:g}", alpha=.85)
        ax.set_xlabel("layer (depth)")
        ax.set_xticks(layers)
        ax.set_title(ttl)
        if logy:
            ax.set_yscale("log")
        ax.grid(alpha=.3)
    axes[0, 0].axhline(1.0, color="k", ls=":", lw=1)
    axes.ravel()[0].legend(fontsize=6, ncol=2)
    fig.suptitle("Coding regime shifts with depth (noise transport)")
    fig.tight_layout()
    savefig(fig, args.out_dir / "fig_depth_regime.png")


def main():
    p = argparse.ArgumentParser(description="depth-dependent coding regime")
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--arms", type=str, default="uniform,input,deep")
    p.add_argument("--shared", type=float, default=0.0,
                   help="ユニット間で共有されるノイズの分散比 (0=独立, 1=完全共有)")
    p.add_argument("--sigmas", type=str, default="0.1,0.4")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=1536)
    p.add_argument("--trials", type=int, default=24)
    p.add_argument("--n-sub", type=int, default=6)
    p.add_argument("--n-lvl", type=int, default=6)
    p.add_argument("--window-mode", choices=("tau", "abs"), default="tau")
    p.add_argument("--window-frac", type=float, default=0.5)
    p.add_argument("--span-frac", type=float, default=1.0)
    p.add_argument("--info-frac", type=float, default=0.25)
    p.add_argument("--lag-taps", type=int, default=12)
    p.add_argument("--jitter-fracs", type=str, default="0.05,0.2,0.8")
    p.add_argument("--rel-fracs", type=str, default="0.05,0.1,0.2,0.35,0.6")
    p.add_argument("--signal", type=str, default="sine")
    p.add_argument("--verbose", action="store_true")
    fncl.add_common_args(p, epochs=1200, hidden_dim=16, seeds="0,1,2")
    args = fncl.finalize_args(p.parse_args(),
                              default_out="out/coding_depth")
    if args.quick:
        args.steps, args.trials, args.epochs = 512, 8, 150
        args.sigmas, args.layers = "0.4", 2
    args.jitter_fracs = [float(v) for v in args.jitter_fracs.split(",")]
    args.rel_fracs = [float(v) for v in args.rel_fracs.split(",")]
    args.window, args.n_lag, args.sub_len, args.stride = 16, 16, 3, 6
    args.jitter_list = [1, 2, 4]
    device = torch.device(args.device)

    rows = []
    for arm in [a.strip() for a in args.arms.split(",")]:
        for sigma in [float(v) for v in args.sigmas.split(",")]:
            for seed in args.seed_list:
                print(f"\n  ===== arm={arm} sigma={sigma:g} seed={seed} "
                      f"(L={args.layers}) =====")
                rows += run_arm(arm, sigma, args, device, seed)

    # ---- 表 ----
    lines = [f"**深さと符号化様式** (L={args.layers}, H={args.hidden_dim}, "
             f"T={args.steps}, R={args.trials}, h={args.crossing_h}, "
             f"speed={args.speed}, epochs={args.epochs}, "
             f"{len(args.seed_list)} seeds)", "",
             f"shared={args.shared}", "",
             "| arm | sigma | layer | sigma_inj | sigma_eff | 輸送 std | "
             "Gamma_eff | rate | CV | rel(fine) | timing frac | sig share | "
             "task MSE |", "|---" * 13 + "|"]
    seen = {}
    for r in rows:
        seen.setdefault((r["arm"], r["sigma"], r["layer"]), []).append(r)
    for (arm, sig, l), v in sorted(seen.items()):
        m = lambda k: float(np.mean([q[k] for q in v            # noqa: E731
                                     if np.isfinite(q[k])]) or 0)
        lines.append(
            f"| {arm} | {sig:g} | {l} | {m('sigma_inj'):.3g} "
            f"| {m('sigma_eff'):.3f} | {m('transport_std'):.3f} "
            f"| {m('gamma_eff'):.3f} | {m('rate'):.3f} | {m('cv'):.2f} "
            f"| {m('rel_fine'):.3f} | {m('timing_frac'):.3f} "
            f"| {m('signal_share'):.2f} | {m('task_mse_refit'):.3f} |")
    table = "\n".join(lines) + "\n"
    print("\n" + table)
    write_text(args.out_dir / "table_depth.md", table)
    save_json(args.out_dir / "results.json",
              {"config": fncl.config_dict(args), "rows": rows})
    make_figures(rows, args)


if __name__ == "__main__":
    main()
