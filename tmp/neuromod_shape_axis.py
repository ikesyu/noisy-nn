"""Two addressing axes for one network: WHERE the noise is vs HOW it is shaped.

sin/cos toy comparison of two multiplexing mechanisms, both at the sample level
(real noise injection, T stochastic forwards, crossing activation):

    region  the standard mechanism of this research line: Gaussian noise
            everywhere, but the SPATIAL support of the field differs.
            sin is trained under mask A (units 0..39), cos under mask B
            (units 24..63); 16 units are shared.  Interpolation mixes the
            two fields: sigma(w) = (1-w) A + w B.

    shape   the axis from the unpublished prior result
            (examples/regression_two_functions_noisetype.py, there with
            analytic layers): every unit gets the SAME intensity, but the
            DISTRIBUTION differs -- variance-matched uniform for sin,
            Gaussian for cos.  Interpolation draws each noise element from
            the Gaussian with probability w, else from the uniform
            (a mixture family with constant variance for every w).

Claims to demonstrate: both axes (i) learn the two functions in one weight
set, and (ii) interpolate lawfully (implied mixing coefficient lambda-hat
tracks w).  The shape axis needs no spatial structure at all, and -- unlike
intensity -- a distribution's SHAPE cannot be absorbed by weight rescaling
(docs/idea_neuromod.md 9.4).

Run from the repository root:
    .venv/bin/python tmp/neuromod_shape_axis.py
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from nnn import activation

HID = 64
T_SAMPLES = 64
H_CROSS = 0.2
SIGMA0 = 0.8
SEEDS = [7, 11, 23]
WS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SQRT3 = math.sqrt(3.0)


class AxisNNN(nn.Module):
    """[1, HID, HID, 1] sample-level NNN with controllable noise per condition.

    cond = ("region", sigma_vec)          Gaussian noise, per-unit std vector
    cond = ("shape", w)                   std SIGMA0 everywhere; each element
                                          drawn from Gaussian with prob w,
                                          else variance-matched uniform
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, HID)
        self.fc2 = nn.Linear(HID, HID)
        self.fc3 = nn.Linear(HID, 1, bias=False)

    def _noise(self, shape, cond, device):
        kind, arg = cond
        if kind == "region":
            return torch.randn(shape, device=device) * arg      # arg: [HID] stds
        w = float(arg)
        gauss = torch.randn(shape, device=device)
        unif = (torch.rand(shape, device=device) * 2.0 - 1.0) * SQRT3
        pick = (torch.rand(shape, device=device) < w).float()
        return SIGMA0 * (pick * gauss + (1.0 - pick) * unif)

    def forward(self, x, cond):
        dev = x.device
        a1 = self.fc1(x).unsqueeze(1).repeat(1, T_SAMPLES, 1)
        z1 = activation.CrossingSample.apply(
            a1 + self._noise(a1.shape, cond, dev), H_CROSS)
        a2 = self.fc2(z1)
        z2 = activation.CrossingSample.apply(
            a2 + self._noise(a2.shape, cond, dev), H_CROSS)
        return self.fc3(z2).mean(dim=1)


def region_fields(device):
    a = torch.zeros(HID, device=device)
    b = torch.zeros(HID, device=device)
    a[:40] = SIGMA0        # units 0..39
    b[24:] = SIGMA0        # units 24..63; overlap 24..39 (16 units)
    return a, b


def implied_lambda(y, ta, tb):
    d = (tb - ta).flatten()
    return float(((y - ta).flatten() @ d) / (d @ d))


def run_mode(mode, seed, x, ys, yc, device, epochs, lr):
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = AxisNNN().to(device)
    fa, fb = region_fields(device)
    cond_a = ("region", fa) if mode == "region" else ("shape", 0.0)
    cond_b = ("region", fb) if mode == "region" else ("shape", 1.0)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(epochs):
        loss = (((net(x, cond_a) - ys) ** 2).mean()
                + ((net(x, cond_b) - yc) ** 2).mean())
        opt.zero_grad()
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        err_a = float(((net(x, cond_a) - ys) ** 2).mean())
        err_b = float(((net(x, cond_b) - yc) ** 2).mean())
        lams, preds = [], {}
        for w in WS:
            cond = (("region", (1 - w) * fa + w * fb) if mode == "region"
                    else ("shape", w))
            y = torch.stack([net(x, cond) for _ in range(32)]).mean(0)
            lams.append(implied_lambda(y, ys, yc))
            if w in (0.0, 0.25, 0.5, 0.75, 1.0) or w in (0.2, 0.8):
                preds[w] = y.cpu().numpy().ravel()
    return err_a, err_b, lams, preds, net


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=6000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--points", type=int, default=256)
    p.add_argument("--out-dir", default="tmp/out/sr_standard")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.linspace(-math.pi, math.pi, args.points,
                       device=device).unsqueeze(1)
    ys, yc = torch.sin(x), torch.cos(x)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    results = {}
    for mode in ("region", "shape"):
        errs, lam_curves, pred0 = [], [], None
        for seed in SEEDS:
            t0 = time.time()
            ea, eb, lams, preds, _ = run_mode(mode, seed, x, ys, yc, device,
                                              args.epochs, args.lr)
            errs.append((ea, eb))
            lam_curves.append(lams)
            if pred0 is None:
                pred0 = preds
            print(f"[{mode} seed {seed}] ({time.time()-t0:.0f}s) "
                  f"err_sin={ea:.4f} err_cos={eb:.4f} "
                  f"lambda: {' '.join(f'{v:.2f}' for v in lams)}")
        results[mode] = (np.array(errs), np.array(lam_curves), pred0)

    # ---------------- figure ----------------
    plt.rcParams.update({"font.size": 11})
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    xn = x.cpu().numpy().ravel()
    titles = {"region": "addressing by REGION (where the noise is)",
              "shape": "addressing by DISTRIBUTION (how the noise is shaped)"}
    for row, mode in enumerate(("region", "shape")):
        errs, lam_curves, preds = results[mode]
        ax = axes[row, 0]
        ax.plot(xn, np.sin(xn), "k--", lw=1, label="targets")
        ax.plot(xn, np.cos(xn), "k--", lw=1)
        ax.plot(xn, preds[0.0], color="tab:blue", lw=2,
                label=f"cond A -> sin (err {errs[:,0].mean():.3f})")
        ax.plot(xn, preds[1.0], color="tab:red", lw=2,
                label=f"cond B -> cos (err {errs[:,1].mean():.3f})")
        ax.set_title(titles[mode])
        ax.legend(fontsize=8)
        ax = axes[row, 1]
        cmap = plt.cm.coolwarm(np.linspace(0, 1, 5))
        for wv, col in zip((0.0, 0.25, 0.5, 0.75, 1.0), cmap):
            if wv in preds:
                ax.plot(xn, preds[wv], color=col, lw=1.8, label=f"w={wv:g}")
        ax.set_title("interpolated outputs")
        ax.legend(fontsize=8)
        ax = axes[row, 2]
        m, sd = lam_curves.mean(0), lam_curves.std(0)
        ax.plot(WS, m, color="tab:green", lw=2, marker="o", ms=4)
        ax.fill_between(WS, m - sd, m + sd, color="tab:green", alpha=0.2)
        ax.plot([0, 1], [0, 1], color="0.5", ls="--", lw=1)
        ax.set_xlabel("w")
        ax.set_ylabel("implied lambda-hat")
        ax.set_title("lawful interpolation")
        for ax in axes[row]:
            ax.grid(alpha=0.3)
    fig.suptitle("One weight set, two functions: two independent addressing axes",
                 y=1.0)
    fig.tight_layout()
    fig.savefig(out / "fig_shape_axis.png", dpi=150, bbox_inches="tight")
    print(f"saved {out}/fig_shape_axis.png")

    with open(out / "shape_axis.csv", "w") as f:
        f.write("# two addressing axes, sin/cos (sample level, variance-matched)\n")
        f.write("# columns: mode(0=region,1=shape),seed,err_sin,err_cos,"
                + ",".join(f"lam_w{w:g}" for w in WS) + "\n")
        for mi, mode in enumerate(("region", "shape")):
            errs, lam_curves, _ = results[mode]
            for si, seed in enumerate(SEEDS):
                f.write(f"{mi},{seed},{errs[si,0]:.5f},{errs[si,1]:.5f},"
                        + ",".join(f"{v:.4f}" for v in lam_curves[si]) + "\n")
    print(f"saved {out}/shape_axis.csv")


if __name__ == "__main__":
    main()
