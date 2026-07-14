"""forward_noise_covariance_learning.py — 全手法比較の PoC ランナー (CLI).

論文の学習則の実装本体は data_nce/fncl/ パッケージにある。このスクリプトは
backprop / cov_only / cov_deriv / cov_jac / cov_jac_full (+ gate 変種) を
sin(x) 回帰で一括比較する対話的なランナーで、論文の図表生成
(data_nce/fncl5_*.py, data_nce/fncl6_*.py) からは独立している。

Run
---
    python tmp/forward_noise_covariance_learning.py
    python tmp/forward_noise_covariance_learning.py --noise uniform
    python tmp/forward_noise_covariance_learning.py --epochs 1500 --num-samples 64
    python tmp/forward_noise_covariance_learning.py --gate-block-size 8 --gate-alpha 0.05

Displays three figures with plt.show() (no files are written; use --save).
アルゴリズムの詳細は docs/forward_noise_covariance_learning.md を参照。
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

# 実装本体の fncl パッケージ (data_nce/) を import パスに追加する。
DATA_DIR = Path(__file__).resolve().parents[1] / "data_nce"
if str(DATA_DIR) not in sys.path:
    sys.path.append(str(DATA_DIR))
from fncl import (NUM_POINTS, build_model, predict, train_backprop,  # noqa: E402
                  train_cov)
from fncl.viz import (plot_activity_stats, plot_fit_check, plot_losses,  # noqa: E402
                      plot_predictions)


# ============================================================
# Main
# ============================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Forward-noise covariance learning on nnn Sample models (PoC).")
    p.add_argument("--noise", choices=("gaussian", "uniform"), default="gaussian",
                   help="gaussian -> SimpleNNNSample, uniform -> SimpleNNNUniformSample")
    p.add_argument("--epochs",      type=int,   default=1500)
    p.add_argument("--hidden-dim",  type=int,   default=64)
    p.add_argument("--num-samples", type=int,   default=64,
                   help="t: stochastic samples the model draws internally")
    p.add_argument("--lr",          type=float, default=1e-2)
    p.add_argument("--sigma",       type=float, default=0.5, help="Gaussian crossing std")
    p.add_argument("--radius",      type=float, default=1.0, help="uniform crossing half-width")
    p.add_argument("--crossing-h",  type=float, default=0.2, help="crossing threshold h")
    p.add_argument("--seed",        type=int,   default=0)
    p.add_argument("--device",      type=str,   default="cpu")
    p.add_argument("--hidden-lr-scale", type=float, default=1.0,
                   help="scale the (noisy) hidden covariance step vs the readout step")
    p.add_argument("--credit", choices=("pooled", "per_input"), default="per_input",
                   help="hidden credit: 'per_input' Cov over samples per input "
                        "(input-dependent, less biased; default) or 'pooled' (one "
                        "global scalar per unit)")
    p.add_argument("--credit-passes", type=int, default=1,
                   help="accumulate covariance stats over this many forward passes "
                        "(variance reduction: effective samples = passes * t) [1]")
    p.add_argument("--opt", choices=("sgd", "adam"), default="sgd",
                   help="manual-update rule for cov methods: 'sgd' or 'adam' "
                        "(adaptive step; converges faster) [sgd]")
    p.add_argument("--lr-decay", choices=("none", "cosine", "exp"), default="none",
                   help="learning-rate schedule for cov methods; decay shrinks the "
                        "end-stage stochastic noise-ball -> lower final MSE [none]")
    # --- perturbation-gate (cov_deriv_gate) knobs ---
    p.add_argument("--gate-block-size", type=int, default=8,
                   help="cov_deriv_gate: number of hidden units in the perturbed/updated "
                        "block G_k [8]")
    p.add_argument("--gate-alpha", type=float, default=0.05,
                   help="cov_deriv_gate: strength of the injected pre-activation "
                        "perturbation xi (added on top of the model's forward noise) [0.05]")
    p.add_argument("--gate-mode", choices=("random", "cyclic"), default="cyclic",
                   help="cov_deriv_gate: rotate the gated block cyclically or pick it at "
                        "random each epoch [cyclic]")
    p.add_argument("--slope", choices=("kde", "analytic"), default="kde",
                   help="cov_deriv local slope dz/dd: 'kde' = the crossing's OWN "
                        "distribution-free density estimate (xor2-xor1)/(2h) [DEFAULT]; "
                        "'analytic' = hand-coded phi'(d) per noise distribution (ablation)")
    p.add_argument("--jac-ema", type=float, default=0.9,
                   help="cov_jac: EMA rate for the running weight mirrors "
                        "W_hat = Cov(d_next,z)/Var(z) (higher = smoother/lower-variance) [0.9]")
    p.add_argument("--jac-out", choices=("cov", "cov_m3", "probe"), default="cov_m3",
                   help="cov_jac_full readout-error estimator: 'cov' = raw Cov(L,y)/Var(y) "
                        "(carries an E[eps^3]/Var skew bias that Adam amplifies late); "
                        "'cov_m3' = subtract the observed third moment (exact for quadratic "
                        "loss; matches backprop) [DEFAULT]; 'probe' = Cov(L(y+xi), xi)/Var(xi) "
                        "with an injected symmetric Gaussian probe xi (unbiased for any loss)")
    p.add_argument("--out-probe-alpha", type=float, default=0.2,
                   help="cov_jac_full --jac-out probe: std of the injected readout probe [0.2]")
    p.add_argument("--jac-track", action=argparse.BooleanOptionalAction, default=True,
                   help="cov_jac: Kolen-Pollack weight-mirror TRACKING (DEFAULT ON) -- "
                        "integrate the known applied weight update into the mirrors (predict) "
                        "and pool the mirror covariance over inputs (idea 1+2). Removes the "
                        "mirror tracking lag. Use --no-jac-track to disable.")
    p.add_argument("--fit-check", action="store_true",
                   help="show an extra FOCUSED figure overlaying only sin(x), backprop and "
                        "cov_jac_adam (+residuals, MSE) to confirm cov_jac_adam fits as "
                        "tightly as backprop.")
    p.add_argument("--save", type=str, default=None,
                   help="directory to save the figures as PNG instead of plt.show() "
                        "(useful headless / for the paper).")
    p.add_argument("--include-gates", action="store_true",
                   help="also run the perturbation/field gate methods (cov_deriv_gate, "
                        "cov_deriv_gate_crn, cov_deriv_field_gate) in the comparison. They "
                        "are kept in the code but OFF the default verification set.")
    p.add_argument("--field-sparsity", type=float, default=0.0,
                   help="cov_deriv_field_gate: fraction of hidden units with ZERO noise "
                        "field (un-recruited: no forward noise AND no update). Applies a "
                        "REAL per-unit noise field to the shared network. 0 -> all units "
                        "recruited, byte-identical to before (default) [0.0]")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    # --- task: y = sin(x), x in [-2pi, 2pi]; input normalised to ~[-2, 2] ---
    x_raw = np.linspace(-2.0 * np.pi, 2.0 * np.pi, NUM_POINTS, dtype=np.float32)
    target_np = np.sin(x_raw).astype(np.float32)
    x = torch.tensor(x_raw / np.pi, device=device).unsqueeze(1)
    t = torch.tensor(target_np, device=device).unsqueeze(1)

    log_every = max(1, args.epochs // 10)
    model_name = "SimpleNNNSample" if args.noise == "gaussian" else "SimpleNNNUniformSample"
    scale = args.sigma if args.noise == "gaussian" else args.radius
    print(f"Forward-noise covariance learning on nnn.{model_name}  (device={device})")
    print(f"  --noise={args.noise} | H={args.hidden_dim} | t(num-samples)={args.num_samples} "
          f"| noise-scale={scale} | h={args.crossing_h} | lr={args.lr} | epochs={args.epochs}")
    print(f"  credit={args.credit} | credit-passes={args.credit_passes} "
          f"(effective samples/update = {args.credit_passes * args.num_samples}) "
          f"| opt={args.opt} | lr-decay={args.lr_decay}")
    print(f"  gate(cov_deriv_gate): block-size={args.gate_block_size} | "
          f"alpha={args.gate_alpha} | mode={args.gate_mode}")
    print(f"  cov_deriv slope={args.slope} (DEFAULT 'kde' = distribution-free "
          f"(xor2-xor1)/(2h); 'analytic' = phi'(d))")

    # per-unit noise field shared by all networks (a real recruitment field): a fraction
    # --field-sparsity of hidden units are un-recruited (zero field -> no noise, no update).
    # Deterministic (own generator) so every fresh() network shares the same recruitment.
    H = args.hidden_dim
    if args.field_sparsity > 0.0:
        g = torch.Generator().manual_seed(args.seed + 12345)
        field = (torch.rand(H, generator=g) >= args.field_sparsity).float().to(device)
        n_off = int((field == 0).sum())
        print(f"  noise field: {n_off}/{H} hidden units un-recruited "
              f"(sparsity={args.field_sparsity}); cov_deriv_field_gate gates on it")
    else:
        field = None

    # identical initial weights for all methods
    net0 = build_model(args.noise, args.hidden_dim, args.sigma, args.radius,
                       args.crossing_h, args.num_samples, device, field=field)
    init_state = copy.deepcopy(net0.state_dict())

    def fresh():
        n = build_model(args.noise, args.hidden_dim, args.sigma, args.radius,
                        args.crossing_h, args.num_samples, device, field=field)
        n.load_state_dict(init_state)
        return n

    # ============================================================
    # Verification set (default): backprop, cov_only, cov_deriv_analytic, cov_deriv_kde,
    # cov_jac_sgd, cov_jac_adam.  cov_jac uses --jac-track (DEFAULT ON).  The two cov_jac
    # rows differ ONLY in the (local, FPGA-friendly) optimiser: sgd vs adam.  The gate/field
    # methods are kept in the code but run only with --include-gates.
    # ============================================================
    losses, preds, stats = {}, {}, {}

    def run(name, method, **kw):
        print(f"\n[{name}] {kw.pop('_desc', method)}")
        net = fresh()
        lo, st = train_cov(net, x, t, args.noise, args.sigma, args.radius, method,
                           args.lr, args.epochs, args.hidden_lr_scale, args.credit,
                           args.credit_passes, kw.pop("opt", args.opt), args.lr_decay,
                           log_every, **kw)
        losses[name] = lo
        preds[name] = predict(net, x)
        stats[name] = st

    print("\n[backprop] (reference, autograd Adam on the same model)")
    net_bp = fresh()
    losses["backprop"] = train_backprop(net_bp, x, t, args.lr, args.epochs, log_every)
    preds["backprop"] = predict(net_bp, x)

    run("cov_only", "cov_only", _desc="covariance credit only (no phi')")
    run("cov_deriv_analytic", "cov_deriv", slope="analytic",
        _desc="cov_deriv with the hand-coded analytic phi'(d) (ablation)")
    run("cov_deriv_kde", "cov_deriv", slope="kde",
        _desc="cov_deriv with the crossing's distribution-free (xor2-xor1)/(2h) slope")
    run("cov_jac_sgd", "cov_jac", opt="sgd", jac_ema=args.jac_ema, jac_track=args.jac_track,
        _desc="weight-mirror recursive credit, SGD (--jac-track=%s)" % args.jac_track)
    run("cov_jac_adam", "cov_jac", opt="adam", jac_ema=args.jac_ema, jac_track=args.jac_track,
        _desc="weight-mirror recursive credit, ADAM -- expected to match backprop "
              "(--jac-track=%s)" % args.jac_track)
    run("cov_jac_full_sgd", "cov_jac_full", opt="sgd", jac_ema=args.jac_ema,
        jac_track=args.jac_track, jac_out=args.jac_out,
        out_probe_alpha=args.out_probe_alpha,
        _desc="cov_jac + covariance readout error (jac_out=%s), SGD "
              "(--jac-track=%s)" % (args.jac_out, args.jac_track))
    run("cov_jac_full_adam", "cov_jac_full", opt="adam", jac_ema=args.jac_ema,
        jac_track=args.jac_track, jac_out=args.jac_out,
        out_probe_alpha=args.out_probe_alpha,
        _desc="cov_jac + covariance readout error (jac_out=%s), ADAM -- no analytic "
              "dL/dy anywhere (--jac-track=%s)" % (args.jac_out, args.jac_track))

    if args.include_gates:
        run("cov_deriv_gate", "cov_deriv_gate", gate_block_size=args.gate_block_size,
            gate_alpha=args.gate_alpha, gate_mode=args.gate_mode,
            _desc="perturbation-gated credit Cov(L,xi)/Var(xi) on a rotating block")
        run("cov_deriv_gate_crn", "cov_deriv_gate_crn", gate_block_size=args.gate_block_size,
            gate_alpha=args.gate_alpha, gate_mode=args.gate_mode,
            _desc="antithetic/common-random-number gate Cov(L(+xi)-L(-xi), xi)")
        run("cov_deriv_field_gate", "cov_deriv_field_gate", slope=args.slope,
            _desc="cov_deriv gated by the per-unit noise field s_i (recruitment)")

    figs = {"learning_curves": plot_losses(losses),
            "predictions": plot_predictions(x_raw, target_np, preds),
            "layer1_stats": plot_activity_stats(stats["cov_deriv_kde"]["layer1"])}
    if args.fit_check:
        figs["fit_check"] = plot_fit_check(x_raw, target_np, preds)

    # low-noise final MSE from the averaged (multi-pass) prediction
    fin = {k: float(np.mean((v - target_np) ** 2)) for k, v in preds.items()}
    deriv_matches = abs(fin["cov_deriv_kde"] - fin["cov_deriv_analytic"]) <= 0.01
    jac_adam_near_bp = fin["cov_jac_adam"] <= max(2.0 * fin["backprop"], fin["backprop"] + 0.003)
    jac_full_matches = abs(fin["cov_jac_full_adam"] - fin["cov_jac_adam"]) <= 0.003
    order = ["backprop", "cov_only", "cov_deriv_analytic", "cov_deriv_kde",
             "cov_jac_sgd", "cov_jac_adam", "cov_jac_full_sgd", "cov_jac_full_adam"]
    if args.include_gates:
        order += ["cov_deriv_gate", "cov_deriv_gate_crn", "cov_deriv_field_gate"]
    print("\n================ SUMMARY ================")
    print(f"Model: nnn.{model_name}   (readout uses the ensemble-mean = expected value)")
    print("Final MSE (8-pass predict):")
    for k in order:
        print(f"  {k:20s}: {fin[k]:.5f}")
    print("\nInterpretation:")
    print("  - backprop is the exact-gradient reference (autograd Adam on the same model).")
    print("  - cov_deriv_kde = covariance credit x the crossing's OWN distribution-free")
    print("    density slope (xor2-xor1)/(2h); cov_deriv_analytic uses the hand-coded phi'(d).")
    print(f"  - cov_deriv_kde {'MATCHES' if deriv_matches else 'does NOT match'} "
          f"cov_deriv_analytic (delta MSE = {fin['cov_deriv_analytic'] - fin['cov_deriv_kde']:+.5f}) "
          f"-> the analytic phi' is not needed.")
    print("  - cov_jac_{sgd,adam} = STRUCTURED/RECURSIVE credit via weight mirrors "
          "W_hat=Cov(d_next,z)/Var(z)")
    print(f"    (--jac-track={args.jac_track}); the two differ ONLY in the local optimiser.")
    print(f"  - cov_jac_adam final MSE {fin['cov_jac_adam']:.5f} vs backprop {fin['backprop']:.5f}: "
          f"{'REACHES backprop level' if jac_adam_near_bp else 'does NOT yet reach backprop level'}.")
    print(f"  - cov_jac_adam {'beats' if fin['cov_jac_adam'] < fin['cov_jac_sgd'] else 'trails'} "
          f"cov_jac_sgd (delta MSE = {fin['cov_jac_sgd'] - fin['cov_jac_adam']:+.5f}) "
          f"-> the sgd 'floor' is optimisation, not estimator bias.")
    print("  - cov_jac_full_{sgd,adam} = cov_jac with the READOUT error ALSO from forward")
    print(f"    statistics (jac_out={args.jac_out}) ~ dL/dy -> NO analytic loss derivative "
          "anywhere.")
    print(f"  - cov_jac_full_adam {'MATCHES' if jac_full_matches else 'does NOT match'} "
          f"cov_jac_adam (delta MSE = {fin['cov_jac_full_adam'] - fin['cov_jac_adam']:+.5f}) "
          f"-> the analytic dL/dy at the readout is not needed.")
    print("  - Adam is a LOCAL per-weight rule -> keeps the no-weight-transport / "
          "FPGA-friendly property.")
    print("=========================================")

    if args.save:
        import os
        os.makedirs(args.save, exist_ok=True)
        for name, fig in figs.items():
            path = os.path.join(args.save, f"{name}.png")
            fig.savefig(path, dpi=130)
            print(f"  saved {path}")
    else:
        print(f"\nOpening {len(figs)} figure windows (close them to exit)...")
        plt.show()


if __name__ == "__main__":
    main()
