"""
consolidation_multiscale.py — 案C: マルチスケール L1 基底（バンプ幅の階層化）
(docs/idea_consolidation.md §12.9.12)

容量研究の残る学習可能性限界への対処: バンプ幅は σ/|w1| で決まり、初期化
|w1| ~ 1 では幅 ~0.4–0.5（正規化入力）となって freq-3 の半ローブ幅 0.33 を
上回る。cov_jac 学習は |w1| を十分速く育てられず、sin(3x) は単独学習でも
限界的（MSE 0.041 @600ep）、sin(4x) は学習不能（0.47）だった。

案C: L1 の初期化を複数スケールの階層にする（poc.build_net の scales）:
広いバンプ（|w|~1, 半数）+ 中間（|w|~2, 1/4）+ 細かいバンプ（|w|~4, 1/4）。
細スケール群は幅 ~0.125 で freq-3/4 の構造を最初から解像できる。

検証:
  (1) 学習可能性対照: sin(kx) k=2..4 の単独学習（全ネットワーク・600 epochs）
      を既定初期化 vs マルチスケール初期化で比較。
  (2) 容量パイプライン: softA（案A 二層共有）+ compress（§12.9.10 の絶対基準
      受理つき）+ retry を、マルチスケール初期化で再実行（§12.9.9 と同一
      プロトコル）。freq-3 ブロックが格納できるようになるかが焦点。

生成物 (out/consolidation_multiscale/):
  fig_multiscale.png / table_multiscale.md / results.json

実行例:
  python tmp/consolidation_multiscale.py --quick
  python tmp/consolidation_multiscale.py --seeds 0
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
import consolidation_recruit as crc  # noqa: E402
import consolidation_capacity as ccap  # noqa: E402

SCALES = [(0.5, 1.0), (0.25, 2.0), (0.25, 4.0)]


def learnability(x_raw, x, args, device, scales, ks=(2, 3, 4)):
    out = {}
    for k in ks:
        torch.manual_seed(0)
        np.random.seed(0)
        net = poc.build_net(args.hidden_dim, args.sigma, args.crossing_h,
                            args.num_samples, device, scales=scales)
        t = torch.tensor(np.sin(k * x_raw).astype(np.float32),
                         device=device).unsqueeze(1)
        tr = poc.CovJacTrainer(net, x, t, lr=args.pre_lr, opt=args.opt,
                               jac_ema=args.jac_ema)
        tr.run(args.epochs_task)
        tr.close()
        pred = fncl.predict(net, x, passes=16)
        out[k] = float(np.mean((pred - t.squeeze(1).cpu().numpy()) ** 2))
    return out


def main():
    p = argparse.ArgumentParser(description="case C: multi-scale L1 bump "
                                            "basis")
    poc.add_poc_args(p)
    p.add_argument("--epochs-task", type=int, default=600)
    p.add_argument("--stop-recovery", type=int, default=8)
    p.add_argument("--rho-init", type=float, default=1.0)
    p.add_argument("--recruit-thresh", type=float, default=0.5)
    p.add_argument("--settle-epochs", type=int, default=100)
    p.add_argument("--layer-fails", type=int, default=3)
    p.add_argument("--use-eps", type=float, default=0.02)
    p.add_argument("--mse-star", type=float, default=0.05)
    p.add_argument("--n-phases", type=int, default=4)
    args = fncl.finalize_args(p.parse_args(),
                              default_out="out/consolidation_multiscale")
    args.cleanup = True
    if args.quick:
        args.epochs_task = 120
        args.epochs_per_step = 5
        args.stop_recovery = 3
        args.settle_epochs = 20
        args.n_phases = 2

    device = torch.device(args.device)
    N = 128
    x_raw = np.linspace(-2.0 * np.pi, 2.0 * np.pi, N, dtype=np.float32)
    x = torch.tensor(x_raw / np.pi, device=device).unsqueeze(1)
    freqs = (1, 2) if args.quick else (1, 2, 3)
    tasks = ccap.make_tasks(x_raw, freqs, args.n_phases)
    K = len(tasks)
    H = args.hidden_dim
    thr = args.mse_star

    # ---------------- (1) 学習可能性対照 ----------------
    print("===== learnability control (single task, full net) =====")
    learn_def = learnability(x_raw, x, args, device, scales=None)
    learn_ms = learnability(x_raw, x, args, device, scales=SCALES)
    for k in learn_def:
        print(f"  sin({k}x): default {learn_def[k]:.4f} -> multiscale "
              f"{learn_ms[k]:.4f}")

    # ---------------- (2) 容量パイプライン（マルチスケール初期化） ----------
    all_results = {"learnability": {"default": learn_def,
                                    "multiscale": learn_ms}}
    for seed in args.seed_list:
        print(f"\n===== seed {seed}  ({K} tasks, multiscale softA pipeline) "
              f"=====")
        args.l1_scales = SCALES
        out_a = crc.run_sequence_recruit(seed, args, device, tasks, x,
                                         cleanup_thr=thr, share_l1=True)
        args.l1_scales = None
        res = {"softA_ms": {
            "mses": [t["mse_final"] for t in out_a["tasks"]],
            "own": [t["n_own"]["0"] + t["n_own"]["1"] for t in out_a["tasks"]],
            "drift": [t["drift"] for t in out_a["tasks"]]}}
        res["softA_ms"]["stored"] = ccap.stored_flags(res["softA_ms"]["mses"],
                                                      thr)
        net = out_a["net"]
        registry = out_a["registry"]

        stored_idx = [i for i, r in enumerate(registry)
                      if r["mse_at_consolidation"] <= thr]
        extra_tasks = [tasks[i] for i in range(K) if i not in stored_idx]
        ctxs, U, n_before, n_after = ccap.compress(net, x, registry,
                                                   stored_idx, args, device,
                                                   accept_thr=thr)
        ext = (ccap.extend(net, x, U, extra_tasks, args, device,
                           share_l1=True) if extra_tasks else [])
        mses_c2 = ccap.eval_shared(net, x, ctxs, U, args.sigma,
                                   args.crossing_h)
        res["compress"] = {
            "union_before": n_before, "union_after": n_after,
            "mses_after_retry": mses_c2,
            "stored_after": ccap.stored_flags(mses_c2, thr)}
        res["retry"] = {
            "tasks": [{k: v for k, v in e.items() if k != "descriptor"}
                      for e in ext],
            "stored": ccap.stored_flags([e["mse"] for e in ext], thr)}

        cap_a = sum(res["softA_ms"]["stored"])
        cap_total = (sum(res["compress"]["stored_after"])
                     + sum(res["retry"]["stored"]))
        max_drift = max(res["softA_ms"]["drift"])
        res["summary"] = {"capacity_softA_ms": cap_a,
                          "capacity_total": cap_total,
                          "max_drift": max_drift}
        print(f"\n  ===== multiscale: softA_ms {cap_a}/{K} -> "
              f"+compress+retry {cap_total}/{K} "
              f"(max drift {max_drift:.2e}) =====")
        all_results[str(seed)] = res

        # ---------------- 図（先頭 seed） ----------------
        if seed == args.seed_list[0]:
            fig = plt.figure(figsize=(13, 4.6))
            gs = fig.add_gridspec(1, 3)
            ax = fig.add_subplot(gs[0, 0])
            ks = sorted(learn_def)
            wd = 0.35
            ax.bar(np.arange(len(ks)) - wd / 2, [learn_def[k] for k in ks],
                   wd, color="tab:gray", label="default init")
            ax.bar(np.arange(len(ks)) + wd / 2, [learn_ms[k] for k in ks],
                   wd, color="tab:purple", label="multiscale init")
            ax.axhline(thr, ls="--", c="k", lw=0.9)
            ax.set_yscale("log")
            ax.set_xticks(range(len(ks)))
            ax.set_xticklabels([f"sin({k}x)" for k in ks])
            ax.set_ylabel(f"standalone MSE ({args.epochs_task} ep)")
            ax.set_title("Learnability control", fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3, axis="y", which="both")

            ax = fig.add_subplot(gs[0, 1])
            idx = np.arange(K)
            colors = ["tab:green" if ok else "tab:red"
                      for ok in res["softA_ms"]["stored"]]
            ax.bar(idx, res["softA_ms"]["mses"], color=colors)
            ax.axhline(thr, ls="--", c="k", lw=0.9)
            ax.set_yscale("log")
            ax.set_xticks(idx)
            ax.set_xticklabels([t[0] for t in tasks], rotation=60, fontsize=6)
            ax.set_ylabel("final MSE (softA_ms)")
            ax.set_title(f"Sequential storage: {cap_a}/{K}", fontsize=10)
            ax.grid(alpha=0.3, axis="y", which="both")

            ax = fig.add_subplot(gs[0, 2])
            bars = {"soft": 4, "softA": 6, "softA_ms": cap_a,
                    "+comp+retry": cap_total}
            ax.bar(range(len(bars)), list(bars.values()),
                   color=["tab:blue", "tab:red", "tab:purple", "tab:green"])
            ax.set_xticks(range(len(bars)))
            ax.set_xticklabels(list(bars.keys()), fontsize=8)
            ax.set_ylabel(f"capacity (/{K})")
            ax.set_title("Capacity progression (seed 0)", fontsize=10)
            ax.grid(alpha=0.3, axis="y")
            fig.suptitle(f"Case C: multi-scale L1 basis (H={H}, seed {seed})",
                         fontsize=12)
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
            savefig(fig, args.out_dir / "fig_multiscale.png")

    # ---------------- 表 ----------------
    seed0 = str(args.seed_list[0])
    r0 = all_results[seed0]
    lines = [f"**Case C: multi-scale L1 basis** (H={H}, T={args.num_samples}, "
             f"scales={SCALES}, thr={thr}, seeds={args.seeds})", "",
             "learnability (standalone, full net, "
             f"{args.epochs_task} ep): "
             + ", ".join(f"sin({k}x) {learn_def[k]:.3f}->{learn_ms[k]:.3f}"
                         for k in sorted(learn_def)), "",
             "| task | softA_ms MSE | own | drift |",
             "|---" * 4 + "|"]
    for i, (nm, _) in enumerate(tasks):
        st = "**" if r0["softA_ms"]["stored"][i] else ""
        lines.append(f"| {nm} | {st}{r0['softA_ms']['mses'][i]:.4f}{st} "
                     f"| {r0['softA_ms']['own'][i]} "
                     f"| {r0['softA_ms']['drift'][i]:.2e} |")
    for e in r0["retry"]["tasks"]:
        lines.append(f"| {e['name']} (retry) | {e['mse']:.4f} "
                     f"| {e['n_own']['0'] + e['n_own']['1']} | — |")
    s = r0["summary"]
    lines += ["",
              f"- capacity: softA_ms **{s['capacity_softA_ms']}/{K}** -> "
              f"+compress+retry **{s['capacity_total']}/{K}** "
              f"(union {r0['compress']['union_before']} -> "
              f"{r0['compress']['union_after']})",
              f"- max drift: {s['max_drift']:.2e}",
              f"- 比較 (seed 0): soft 4/12, softA 6/12", ""]
    table = "\n".join(lines) + "\n"
    print("\n" + table)
    write_text(args.out_dir / "table_multiscale.md", table)

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items() if k != "descriptor"}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        return o

    save_json(args.out_dir / "results.json",
              {"config": fncl.config_dict(args),
               "results": clean(all_results)})


if __name__ == "__main__":
    main()
