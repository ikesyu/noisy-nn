"""Verify the channel-width prediction for the distribution axis (9.4).

Claim under test: the residual of distribution-addressing is an amplification
cost -- variance-matched uniform vs Gaussian differ only in higher moments, so
the readout amplifies a weak differential signal together with sampling noise.
If so, a MORE contrasting pair at the same variance should widen the channel
and pull the error toward the region-mode level.

The contrasting pair is the biological up/down-state contrast: BIMODAL
(two Gaussian modes at +-mu with width s, mu^2 + s^2 = sigma0^2) versus
UNIMODAL Gaussian N(0, sigma0^2).  Same variance by construction, strongly
different shape (sub-Gaussian, two peaks).

Protocol identical to the shape mode of neuromod_shape_axis.py: dual sin/cos,
sample level, 3 seeds; interpolation via the per-element mixture family
(Gaussian with probability w, else bimodal; variance constant in w); plus the
bias/variance decomposition of the residual.

Run from the repository root:
    .venv/bin/python tmp/neuromod_shape_bimodal.py
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parent))

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "shape_axis", Path(__file__).parent / "neuromod_shape_axis.py")
SA = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SA)

SIGMA0 = SA.SIGMA0
S_MODE = 0.3 * SIGMA0                      # width of each mode
MU = math.sqrt(SIGMA0 ** 2 - S_MODE ** 2)  # mode position: variance matched


class BimodalAxisNNN(SA.AxisNNN):
    """cond = ("bimodal", w): each element Gaussian N(0, sigma0^2) with
    probability w, else bimodal (+-MU + S_MODE * N(0,1))."""

    def _noise(self, shape, cond, device):
        kind, arg = cond
        if kind != "bimodal":
            return super()._noise(shape, cond, device)
        w = float(arg)
        gauss = SIGMA0 * torch.randn(shape, device=device)
        sign = torch.where(torch.rand(shape, device=device) < 0.5, -1.0, 1.0)
        bimod = sign * MU + S_MODE * torch.randn(shape, device=device)
        pick = (torch.rand(shape, device=device) < w).float()
        return pick * gauss + (1.0 - pick) * bimod


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out-dir", default="tmp/out/sr_standard")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.linspace(-math.pi, math.pi, 256, device=device).unsqueeze(1)
    ys, yc = torch.sin(x), torch.cos(x)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"bimodal modes at +-{MU:.3f}, width {S_MODE:.3f} "
          f"(variance matched to sigma0={SIGMA0})")

    errs, lam_curves, pred0, decomp = [], [], None, []
    for seed in SA.SEEDS:
        t0 = time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        net = BimodalAxisNNN().to(device)
        ca, cb = ("bimodal", 0.0), ("bimodal", 1.0)
        opt = torch.optim.Adam(net.parameters(), lr=args.lr)
        for _ in range(args.epochs):
            loss = (((net(x, ca) - ys) ** 2).mean()
                    + ((net(x, cb) - yc) ** 2).mean())
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            ea = float(((net(x, ca) - ys) ** 2).mean())
            eb = float(((net(x, cb) - yc) ** 2).mean())
            ba = float(((torch.stack([net(x, ca) for _ in range(64)]).mean(0)
                         - ys) ** 2).mean())
            bb = float(((torch.stack([net(x, cb) for _ in range(64)]).mean(0)
                         - yc) ** 2).mean())
            lams, preds = [], {}
            for w in SA.WS:
                y = torch.stack([net(x, ("bimodal", w))
                                 for _ in range(32)]).mean(0)
                lams.append(SA.implied_lambda(y, ys, yc))
                if w in (0.0, 0.2, 0.5, 0.8, 1.0):
                    preds[w] = y.cpu().numpy().ravel()
        errs.append((ea, eb))
        lam_curves.append(lams)
        decomp.append((ba, bb))
        if pred0 is None:
            pred0 = preds
        print(f"[seed {seed}] ({time.time()-t0:.0f}s) err_sin={ea:.4f} "
              f"err_cos={eb:.4f} bias_sin={ba:.4f} bias_cos={bb:.4f} "
              f"lambda: {' '.join(f'{v:.2f}' for v in lams)}")

    errs, lam_curves, decomp = map(np.array, (errs, lam_curves, decomp))
    print("\ncomparison (3-seed mean single-forward err, sin/cos):")
    print(f"  region (where)        : 0.0017 / 0.0018   [shape_axis.csv]")
    print(f"  uniform vs gauss (how): 0.0204 / 0.0385   [shape_axis.csv]")
    print(f"  bimodal vs gauss (how): {errs[:,0].mean():.4f} / "
          f"{errs[:,1].mean():.4f}   bias {decomp[:,0].mean():.4f} / "
          f"{decomp[:,1].mean():.4f}")

    # figure row
    plt.rcParams.update({"font.size": 11})
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
    xn = x.cpu().numpy().ravel()
    ax = axes[0]
    ax.plot(xn, np.sin(xn), "k--", lw=1, label="targets")
    ax.plot(xn, np.cos(xn), "k--", lw=1)
    ax.plot(xn, pred0[0.0], color="tab:blue", lw=2,
            label=f"bimodal -> sin (err {errs[:,0].mean():.3f})")
    ax.plot(xn, pred0[1.0], color="tab:red", lw=2,
            label=f"gaussian -> cos (err {errs[:,1].mean():.3f})")
    ax.set_title("addressing by DISTRIBUTION: bimodal vs unimodal")
    ax.legend(fontsize=8)
    ax = axes[1]
    cmap = plt.cm.coolwarm(np.linspace(0, 1, 5))
    for wv, col in zip((0.0, 0.2, 0.5, 0.8, 1.0), cmap):
        ax.plot(xn, pred0[wv], color=col, lw=1.8, label=f"w={wv:g}")
    ax.set_title("interpolated outputs")
    ax.legend(fontsize=8)
    ax = axes[2]
    m, sd = lam_curves.mean(0), lam_curves.std(0)
    ax.plot(SA.WS, m, color="tab:green", lw=2, marker="o", ms=4)
    ax.fill_between(SA.WS, m - sd, m + sd, color="tab:green", alpha=0.2)
    ax.plot([0, 1], [0, 1], color="0.5", ls="--", lw=1)
    ax.set_xlabel("w")
    ax.set_ylabel("implied lambda-hat")
    ax.set_title("lawful interpolation")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle("Up/down-state-like contrast widens the distribution channel",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(out / "fig_shape_bimodal.png", dpi=150, bbox_inches="tight")
    print(f"saved {out}/fig_shape_bimodal.png")


if __name__ == "__main__":
    main()
