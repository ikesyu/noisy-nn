"""
consolidation_midcompress.py — 課題9: 途中圧縮 (§12.9.12 知見3 -> §12.9.16)

凍結系パイプラインの残差（softA + 事後圧縮で 11/16、案B の 12 に 1 差）の
候補原因は「圧縮のタイミング」だった: 圧縮を全タスク提示後に 1 回だけ
行うと、逐次フェーズの後半は空きが尽きた状態で走り、初回提示の格納が
失敗する（失敗タスクは再提示でしか救えない）。stop 則の緩和はこの残差を
埋めないことが §12.9.14 で確定したため、残る仮説は**途中圧縮**
（空きが尽きる前に畳む）である。

設計:
  - 獲得は softA（二層共有・share_l1・掃除つき; ccap.extend と同じ規約）。
  - 各タスクの格納後に空きを点検し、min(空きL1, 空きL2) < free_thr なら
    **それまでに格納した全タスク**（過去の圧縮産物を含む）を union
    アニールで圧縮する（§12.9.10 の絶対格納基準つき）。圧縮後は
    U = 生存 union が凍結語彙になり、解放されたユニットは空きに戻る。
    圧縮のたびに全格納タスクの記述子（wout / support）を更新するので、
    再圧縮は一様に扱える。
  - 全タスク提示後は基準パイプラインと同じく最終圧縮 + 失敗タスクの
    再提示を行う（比較条件を揃えるため）。

アーム:
  end : 基準（§12.9.13 grand の softA アーム = 逐次 -> 圧縮 -> 再提示）。
  mid : 上記の途中圧縮つき逐次 -> （最終圧縮）-> 再提示。

実行例:
  python tmp/consolidation_midcompress.py --quick
  python tmp/consolidation_midcompress.py --seeds 0 --free-thr 10
"""
import argparse

import numpy as np
import torch
import matplotlib
if __name__ == "__main__":
    matplotlib.use("Agg")

import fncl_driver as fncl  # noqa: E402
from fncl_driver import save_json, write_text  # noqa: E402
import consolidation_poc as poc  # noqa: E402
import consolidation_soft as csoft  # noqa: E402
import consolidation_recruit as crc  # noqa: E402
import consolidation_capacity as ccap  # noqa: E402
import consolidation_joint as cjoint  # noqa: E402
from consolidation_lib import (  # noqa: E402
    TaskCtx, alive, eval_with_descriptor, freeze_masks, joint_round,
    predict, set_field, zero_cross_columns)
from consolidation_multiscale import SCALES  # noqa: E402


def acquire_one(net, x, name, target, past, free, args, device):
    """softA 規約で 1 タスク獲得（ccap.extend の 1 反復と同じ手順）。"""
    H = args.hidden_dim
    print(f"\n  ===== [acquire] {name}  (free L1 {len(free[0])} / "
          f"L2 {len(free[1])}, vocab {len(past[0])}+{len(past[1])}) =====")
    net.fcs[2].weight.data.zero_()
    net.fcs[2].bias.data.zero_()
    zero_cross_columns(net, past)
    masks = freeze_masks(H, past, device, share_l1=True)
    mobil = {l: sorted(set(free[l]) | set(past[l])) for l in (0, 1)}
    set_field(net, mobil, args.sigma, args.crossing_h)
    t = torch.tensor(target, device=device).unsqueeze(1)
    trainer = poc.CovJacTrainer(net, x, t, lr=args.pre_lr, opt=args.opt,
                                jac_ema=args.jac_ema)
    trainer.grad_masks = masks
    for e in range(args.epochs_task):
        loss = trainer.step()
        if e % max(1, args.epochs_task // 2) == 0:
            print(f"    [learn] epoch {e:5d} mse={loss:.5f}")
    trainer.close()
    base = float(np.mean(trainer.losses[-min(100, len(trainer.losses)):]))
    tol = base * args.drift_mult + args.drift_abs
    _, holds = csoft.anneal_until_stop(net, x, target, tol, args,
                                       eligible=free, grad_masks=masks,
                                       vocab=past, share_l1=True)
    region = {l: [k for k in free[l] if float(net.sigma_vecs[l][k]) > 0]
              for l in (0, 1)}
    r = {"name": name, "target": target, "region": region,
         "support": {l: sorted(region[l] + past[l]) for l in (0, 1)},
         "sigma0": float(args.sigma), "h0": float(args.crossing_h),
         "wout": net.fcs[2].weight.data.clone(),
         "b_out": net.fcs[2].bias.data.clone(), "tol": tol}
    mse = float(np.mean((eval_with_descriptor(net, x, r) - target) ** 2))
    print(f"    own L1 {len(region[0])} + L2 {len(region[1])}; "
          f"MSE={mse:.5f}")
    return r, mse, region


def reseed_free(net, free, args):
    """解放ユニットの入力側を新品の初期化に戻す。

    解放ユニットはどの格納記述子でも沈黙（sigma=0, h 番兵）しており、
    その W1 行 / W2 行は don't-care なので、再初期化は格納タスクに厳密に
    無影響。狙いは「解放は空きを返すが初期化は返さない」の解消:
    解放ユニットの W1 は前のタスクの構造に再訓練済みで、細スケールの
    バンプ・タイリングが摩耗している（§12.9.16 知見）。L1 はマルチスケール
    タイリングの流儀（群をランダムに選び倍率つきバンプを再配置）、
    L2 行は一様小乱数で再播種する。
    """
    H = net.sigma_vecs[0].shape[0]
    scales = getattr(args, "l1_scales", None) or [(1.0, 1.0)]
    fracs = torch.tensor([f for f, _ in scales], dtype=torch.float32)
    with torch.no_grad():
        for k in free[0]:
            gi = int(torch.multinomial(fracs, 1))
            magv = scales[gi][1]
            c = float(torch.rand(1)) * 4.0 - 2.0
            w = magv * (0.8 + 0.4 * float(torch.rand(1)))
            if float(torch.rand(1)) < 0.5:
                w = -w
            net.fcs[0].weight.data[k, 0] = w
            net.fcs[0].bias.data[k] = float(-w * c)
        bound = 1.0 / float(np.sqrt(H))
        for k in free[1]:
            net.fcs[1].weight.data[k].uniform_(-bound, bound)
            net.fcs[1].bias.data[k] = 0.0


def compress_stored(net, x, stored, args, device, accept_thr):
    """格納済み全タスク（過去の圧縮産物を含む）を union アニールで圧縮する。

    圧縮後は各タスクの記述子（wout / b_out / support）を共有場 U 上の
    ものに更新するので、以後の再圧縮・評価は一様に扱える。
    """
    H = args.hidden_dim
    union = {l: sorted({k for r in stored for k in r["support"][l]})
             for l in (0, 1)}
    sigma0, h0 = float(args.sigma), float(args.crossing_h)
    for l in (0, 1):
        sig = torch.zeros(H)
        hv = torch.full((H,), poc.H_DEAD)
        for k in union[l]:
            sig[k] = sigma0
            hv[k] = h0
        net.sigma_vecs[l] = sig.to(device)
        net.h_vecs[l] = hv.to(device)
    ctxs = [TaskCtx.from_registry(net, x, r, args) for r in stored]
    for _ in range(args.settle_epochs):
        joint_round(net, ctxs)
    for ctx in ctxs:
        eq = float(np.mean(ctx.trainer.losses[-50:]))
        ctx.tol = max(ctx.tol, eq * args.drift_mult + args.drift_abs)

    def accept():
        for ctx in ctxs:
            ctx.activate(net)
            pred = predict(net, x, passes=16)
            if float(np.mean((pred - ctx.target) ** 2)) > accept_thr:
                return False
        return True

    _, holds, removed = cjoint.union_anneal(net, x, ctxs, args,
                                            accept_fn=accept)
    U = {l: alive(net, l) for l in (0, 1)}
    n_before = len(union[0]) + len(union[1])
    n_after = len(U[0]) + len(U[1])
    print(f"  [compress] union {len(union[0])}+{len(union[1])} = {n_before}"
          f" -> {len(U[0])}+{len(U[1])} = {n_after} "
          f"(removed L1 {removed[0]} / L2 {removed[1]}, holds={holds})")
    for r, ctx in zip(stored, ctxs):
        r["wout"] = ctx.wout.clone()
        r["b_out"] = ctx.b_out.clone()
        r["support"] = {l: sorted(U[l]) for l in (0, 1)}
        r["tol"] = float(ctx.tol)
        ctx.trainer.close()
    return U, n_before, n_after


def run_mid(seed, args, device, tasks, x, thr, free_thr=None, reseed=False):
    """途中圧縮つき softA 逐次獲得 + 最終圧縮 + 再提示。

    free_thr=0 で途中圧縮なし（= end 基準を同一獲得機構で実行）。
    reseed=True で解放時に入力側を再播種する（reseed_free）。
    """
    if free_thr is None:
        free_thr = args.free_thr
    torch.manual_seed(seed)
    np.random.seed(seed)
    H = args.hidden_dim
    net = poc.build_net(H, args.sigma, args.crossing_h, args.num_samples,
                        device, scales=getattr(args, "l1_scales", None))
    past = {0: [], 1: []}
    free = {0: list(range(H)), 1: list(range(H))}
    stored, failed, events = [], [], []
    first_pass = {}
    for name, target in tasks:
        r, mse, region = acquire_one(net, x, name, target, past, free,
                                     args, device)
        first_pass[name] = mse
        if mse <= thr:
            stored.append(r)
            past = {l: sorted(past[l] + region[l]) for l in (0, 1)}
            free = {l: [k for k in free[l] if k not in region[l]]
                    for l in (0, 1)}
        else:
            failed.append((name, target))
            for l in (0, 1):
                for k in region[l]:
                    poc.kill_unit(net, l, k)
            print(f"    [cleanup] 格納失敗 (MSE > {thr}) -> 回収")
            if reseed:
                reseed_free(net, region, args)
        if (min(len(free[0]), len(free[1])) < free_thr
                and len(stored) >= 2):
            U, nb, na = compress_stored(net, x, stored, args, device, thr)
            events.append({"after": name, "n_stored": len(stored),
                           "union_before": nb, "union_after": na})
            past = {l: sorted(U[l]) for l in (0, 1)}
            free = {l: [k for k in range(H) if k not in U[l]]
                    for l in (0, 1)}
            if reseed:
                reseed_free(net, free, args)
    # ---- 最終圧縮 + 再提示（基準パイプラインと同じ仕上げ） ----
    just_compressed = bool(events) and events[-1]["after"] == tasks[-1][0]
    if len(stored) >= 2 and not just_compressed:
        U, nb, na = compress_stored(net, x, stored, args, device, thr)
        events.append({"after": "(final)", "n_stored": len(stored),
                       "union_before": nb, "union_after": na})
        past = {l: sorted(U[l]) for l in (0, 1)}
        free = {l: [k for k in range(H) if k not in U[l]] for l in (0, 1)}
        if reseed:
            reseed_free(net, free, args)
    retry = []
    for name, target in list(failed):
        r, mse, region = acquire_one(net, x, name, target, past, free,
                                     args, device)
        retry.append({"name": name, "mse": mse})
        if mse <= thr:
            stored.append(r)
            past = {l: sorted(past[l] + region[l]) for l in (0, 1)}
            free = {l: [k for k in free[l] if k not in region[l]]
                    for l in (0, 1)}
        else:
            for l in (0, 1):
                for k in region[l]:
                    poc.kill_unit(net, l, k)
    # ---- 最終評価 ----
    final = {}
    for r in stored:
        pred = eval_with_descriptor(net, x, r)
        final[r["name"]] = float(np.mean((pred - r["target"]) ** 2))
    return {"first_pass": first_pass, "final": final, "events": events,
            "retry": retry,
            "n_units": len(past[0]) + len(past[1]),
            "free_end": (len(free[0]), len(free[1]))}


def run_end(seed, args, device, tasks, x, x_raw, thr):
    """基準: §12.9.13 grand の softA アーム（逐次 -> 圧縮 -> 再提示）。"""
    K = len(tasks)
    out = crc.run_sequence_recruit(seed, args, device, tasks, x,
                                   cleanup_thr=thr, share_l1=True)
    mses = [t["mse_final"] for t in out["tasks"]]
    net, registry = out["net"], out["registry"]
    first_pass = {t["name"]: t["mse_final"] for t in out["tasks"]}
    stored_idx = [i for i, r in enumerate(registry)
                  if r["mse_at_consolidation"] <= thr]
    extras = [tasks[i] for i in range(K) if i not in stored_idx]
    ctxs, U, nb, na = ccap.compress(net, x, registry, stored_idx, args,
                                    device, accept_thr=thr)
    ext = (ccap.extend(net, x, U, extras, args, device, share_l1=True)
           if extras else [])
    mses_c2 = (ccap.eval_shared(net, x, ctxs, U, args.sigma,
                                args.crossing_h) if ctxs else [])
    final = {}
    for i, m in zip(stored_idx, mses_c2):
        final[registry[i]["name"]] = m
    for e in ext:
        if e["mse"] <= thr:
            final[e["name"]] = e["mse"]
    return {"first_pass": first_pass, "final": final,
            "events": [{"after": "(final)", "n_stored": len(stored_idx),
                        "union_before": nb, "union_after": na}],
            "retry": [{"name": e["name"], "mse": e["mse"]} for e in ext],
            "n_units": None, "free_end": None,
            "seq_mses": mses}


def main():
    p = argparse.ArgumentParser(description="mid-sequence compression A/B")
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
    p.add_argument("--freqs", type=str, default="1,2,3,4")
    p.add_argument("--free-thr", type=int, default=10,
                   help="min(空きL1, 空きL2) がこれ未満で途中圧縮を起動")
    p.add_argument("--arms", type=str, default="end,end_rs,mid,mid_rs",
                   help="end_probe(crc基準)/end/mid/end_rs/mid_rs")
    args = fncl.finalize_args(p.parse_args(),
                              default_out="out/consolidation_midcompress")
    args.cleanup = True
    args.l1_scales = SCALES
    if args.quick:
        args.epochs_task = 120
        args.epochs_per_step = 5
        args.stop_recovery = 3
        args.settle_epochs = 20
        args.n_phases = 2
        args.freqs = "1,2"

    device = torch.device(args.device)
    N = 128
    x_raw = np.linspace(-2.0 * np.pi, 2.0 * np.pi, N, dtype=np.float32)
    x = torch.tensor(x_raw / np.pi, device=device).unsqueeze(1)
    freqs = tuple(int(v) for v in args.freqs.split(","))
    tasks = ccap.make_tasks(x_raw, freqs, args.n_phases)
    K = len(tasks)
    thr = args.mse_star
    arms = [a.strip() for a in args.arms.split(",")]

    all_results = {}
    for seed in args.seed_list:
        res = {}
        for arm in arms:
            print(f"\n########## seed {seed} / arm {arm} ({K} tasks) "
                  f"##########")
            if arm == "end_probe":
                res[arm] = run_end(seed, args, device, tasks, x, x_raw, thr)
            else:
                ft = 0 if arm.startswith("end") else args.free_thr
                rs = arm.endswith("_rs")
                res[arm] = run_mid(seed, args, device, tasks, x, thr,
                                   free_thr=ft, reseed=rs)
            r = res[arm]
            cap_first = sum(1 for nm, _ in tasks
                            if r["first_pass"].get(nm, 1e9) <= thr)
            cap_final = sum(1 for nm, _ in tasks
                            if r["final"].get(nm, 1e9) <= thr)
            r["capacity_first"] = cap_first
            r["capacity_final"] = cap_final
            print(f"\n  ===== [{arm}] capacity: first {cap_first}/{K} -> "
                  f"final {cap_final}/{K} "
                  f"(compressions: {len(r['events'])}) =====")
        all_results[seed] = res

    # ---------------- 表 ----------------
    seed0 = args.seed_list[0]
    r0 = all_results[seed0]
    lines = [f"**途中圧縮 A/B** (softA, H={args.hidden_dim}, {K} tasks "
             f"sin(kx+phi) k={args.freqs}, multiscale, cleanup on, "
             f"thr={thr}, free-thr={args.free_thr}, seed {seed0})", "",
             "| arm | 初回提示 | 最終 | 圧縮回数 | union 遷移 |",
             "|---" * 5 + "|"]
    for arm in arms:
        r = r0[arm]
        tr = " -> ".join(f"{e['union_before']}->{e['union_after']}"
                         for e in r["events"])
        lines.append(f"| {arm} | {r['capacity_first']}/{K} "
                     f"| **{r['capacity_final']}/{K}** "
                     f"| {len(r['events'])} | {tr} |")
    lines += ["", "per-task (初回提示 MSE / 最終 MSE; 太字 = 格納):", "",
              "| task | " + " | ".join(f"{a} 初回 | {a} 最終" for a in arms)
              + " |", "|---" * (1 + 2 * len(arms)) + "|"]
    for nm, _ in tasks:
        cells = []
        for arm in arms:
            r = r0[arm]
            fp = r["first_pass"].get(nm)
            fn = r["final"].get(nm)
            b1 = "**" if fp is not None and fp <= thr else ""
            b2 = "**" if fn is not None and fn <= thr else ""
            cells.append(f"{b1}{fp:.4f}{b1}" if fp is not None else "—")
            cells.append(f"{b2}{fn:.4f}{b2}" if fn is not None else "—")
        lines.append(f"| {nm} | " + " | ".join(cells) + " |")
    table = "\n".join(lines) + "\n"
    print("\n" + table)
    tag = "_".join(arms)
    write_text(args.out_dir / f"table_mid_{tag}.md", table)

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, torch.Tensor):
            return None
        return o

    save_json(args.out_dir / f"results_{tag}.json",
              {"config": fncl.config_dict(args),
               "results": {str(sd): {a: {k: clean(v)
                                         for k, v in all_results[sd][a]
                                         .items()
                                         if k not in ("net",)}
                           for a in all_results[sd]}
                           for sd in all_results}})


if __name__ == "__main__":
    main()
