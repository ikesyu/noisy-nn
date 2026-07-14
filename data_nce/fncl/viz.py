"""fncl.viz — 学習結果の確認用プロット (learning curves / predictions / fit check)."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Plots (displayed with plt.show(), not saved)
# ============================================================
def plot_losses(losses: dict):
    fig = plt.figure(figsize=(7, 5))
    for name, curve in losses.items():
        plt.semilogy(curve, label=name)
    plt.xlabel("epoch")
    plt.ylabel("MSE (eval, log scale)")
    plt.title("Learning curves: backprop vs forward-noise covariance learning")
    plt.legend()
    plt.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_predictions(x_raw, target, preds: dict):
    order = np.argsort(x_raw)
    fig = plt.figure(figsize=(8, 5))
    plt.plot(x_raw[order], target[order], "k-", lw=2.5, label="target sin(x)")
    styles = {"backprop": "--", "cov_only": "-.", "cov_deriv": "-",
              "cov_deriv_kde": "-", "cov_deriv_analytic": (0, (3, 1, 1, 1)),
              "cov_jac": (0, (1, 1)), "cov_jac_sgd": (0, (1, 1)),
              "cov_jac_adam": (0, (4, 1, 1, 1)),
              "cov_jac_full_sgd": (0, (2, 2)), "cov_jac_full_adam": (0, (6, 1, 1, 1)),
              "cov_deriv_gate": ":", "cov_deriv_gate_crn": (0, (5, 1)),
              "cov_deriv_field_gate": (0, (3, 1, 1, 1, 1, 1))}
    for name, y in preds.items():
        plt.plot(x_raw[order], y[order], linestyle=styles.get(name, "-"),
                 lw=1.6, label=name)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Predictions on y = sin(x)")
    plt.legend()
    plt.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_activity_stats(stats_layer: dict):
    idx = np.arange(stats_layer["mean_activity"].numel())
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    axes[0].bar(idx, stats_layer["mean_activity"].numpy(), color="tab:blue")
    axes[0].set_ylabel("mean activity\n<z>")
    axes[0].set_title("cov_deriv, hidden layer 1 (final epoch)")
    axes[1].bar(idx, stats_layer["g_z"].numpy(), color="tab:green")
    axes[1].set_ylabel("covariance credit\ng_z = Cov(L,z)/Var(z)")
    axes[2].bar(idx, stats_layer["phi_prime"].numpy(), color="tab:red")
    axes[2].set_ylabel("local derivative\nmean phi'(d)")
    axes[2].set_xlabel("hidden unit index")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_fit_check(x_raw, target, preds: dict,
                   focus=("backprop", "cov_jac_adam", "cov_jac_full_adam")):
    """Focused confirmation figure: does cov_jac_adam approximate sin(x) as TIGHTLY as
    backprop?  Overlays only the target and the `focus` methods, plus a residual panel and
    the MSE in the legend, so the fit quality is unmistakable (rather than buried in the
    all-methods plot)."""
    order = np.argsort(x_raw)
    colors = {"backprop": "tab:blue", "cov_jac_adam": "tab:red",
              "cov_jac_sgd": "tab:green", "cov_jac_full_adam": "tab:purple"}
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    axes[0].plot(x_raw[order], target[order], "k-", lw=3.0, alpha=0.35, label="target sin(x)")
    for name in focus:
        if name not in preds:
            continue
        mse = float(np.mean((preds[name] - target) ** 2))
        axes[0].plot(x_raw[order], preds[name][order], color=colors.get(name), lw=1.8,
                     label=f"{name}  (MSE={mse:.2e})")
        axes[1].plot(x_raw[order], (preds[name] - target)[order], color=colors.get(name),
                     lw=1.2, label=name)
    axes[0].set_ylabel("y")
    axes[0].set_title("Fit check: cov_jac_adam vs backprop on y = sin(x)")
    axes[0].legend()
    axes[1].axhline(0.0, color="k", lw=0.6)
    axes[1].set_ylabel("residual\n(pred - sin)")
    axes[1].set_xlabel("x")
    axes[1].legend(loc="upper right", fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
