"""
fncl_noisefield_screen.py — S1: ノイズ場・推定量の介入をオフラインで選別する（docs/idea_ca.md §5.6）
（旧名: fncl_nf1.py．出力ディレクトリ tmp/out/ は旧名のまま）

学習し直さずに，backprop で 400 epoch 事前学習した depth 4 のネット上で
「ミラーの測り方」だけを差し替え，回復した r と前向き信号への副作用を比較する。
S0 で以下が分かっているので候補を絞ってある:

  - 層内相関は**ランク 1 の一様な共通モード**（行和が |rho| の H 倍）。
    → ユニット平均を引くだけ（O(H)）で消えるはず。
  - 周期ノイズ＋位相内中心化（L3）はミラーを 0.998 に完全回復する。
    ただし上流の実効ドロー数が減り，アンサンブルの分散が sqrt(T/p) 倍になる。
  - phi' の符号は既に約半々なので L2（符号の多様化）に伸びしろはない → 除外。

比較する推定量:
  diag      : 現行の単変量ミラー（基準）
  cm        : 共通モード除去（ユニット平均を引く，O(H)，前向きコストなし）
  rank1     : 先頭主成分の射影除去（O(H)，一様でない負荷にも対応）
  mv        : 完全多変量（O(H^3)，上限の参照値）
  per<p>    : 周期ノイズ p ＋位相内中心化（L3，ノイズ場の設計）
  sig<s>    : sigma を s 倍（L1，対照．前向き信号を壊すことの確認）

生成物 (tmp/out/fncl_nf1/): results.json, (標準出力) 表

実行例:
  .venv/bin/python tmp/fncl_noisefield_screen.py --quick
  .venv/bin/python tmp/fncl_noisefield_screen.py --depth 4 --pretrain 400
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as tF

DATA_DIR = Path(__file__).resolve().parent.parent / "data_nce"
sys.path.insert(0, str(DATA_DIR))
import fncl  # noqa: E402
from fncl_common import pearson, save_json  # noqa: E402
from fncl_mnist_fidelity import build_net, load_mnist  # noqa: E402
from fncl_mnist_mirror_variants import cov_weight_mv  # noqa: E402
from fncl_noisefield_lib import (build_net_noise, collect, cov_weight_decorr,  # noqa: E402
                     cov_weight_phase, cov_weight_rank1, cov_weight_woodbury,
                     install_noise_field, restore_noise_field)


def true_weight(net, l: int, depth: int):
    return net.fcs[-1].weight if l == depth - 1 else net.fcs[l + 1].weight


def mirrors_under(net, x, depth: int, passes: int, kind: str, param=None) -> dict:
    """指定した推定量で全層のミラーを測り，真の重みとの Pearson r を返す."""
    periods = None
    scales = None
    if kind == "per":
        # 対象層ごとに上流を周期化するので，ここでは層ごとに測り直す
        out = {}
        for l in range(depth):
            if l == 0:
                # 第 1 隠れ層は元々相関がない（上流が定数）ので通常測定
                install_noise_field(net, periods=[1] * depth)
                s = collect(net, x, passes)
                tgt = s["ys"] if l == depth - 1 else s["d"][l + 1]
                w = fncl.cov_weight(tgt, s["z"][l], pool=True)
            else:
                install_noise_field(net, periods=[param] * l + [1] * (depth - l))
                s = collect(net, x, passes)
                pl = s["z"][0].shape[1] // passes
                tgt = s["ys"] if l == depth - 1 else s["d"][l + 1]
                w = cov_weight_phase(tgt, s["z"][l], param, pass_len=pl)
            out[f"z{l + 1}"] = pearson(w, true_weight(net, l, depth))
        restore_noise_field(net)
        return out

    if kind == "sig":
        scales = [param] * depth
    install_noise_field(net, periods=periods or [1] * depth,
                        std_scale=scales or [1.0] * depth)
    try:
        s = collect(net, x, passes)
        out = {}
        for l in range(depth):
            tgt = s["ys"] if l == depth - 1 else s["d"][l + 1]
            z = s["z"][l]
            if kind in ("diag", "sig"):
                w = fncl.cov_weight(tgt, z, pool=True)
            elif kind == "cm":
                w = cov_weight_decorr(tgt, z, mode="cm")
            elif kind == "rank1":
                w = cov_weight_decorr(tgt, z, mode="rank1")
            elif kind == "wood":
                w = cov_weight_woodbury(tgt, z)
            elif kind == "unif":
                w = cov_weight_rank1(tgt, z, uniform=True)
            elif kind == "r1lo":
                w = cov_weight_rank1(tgt, z, uniform=False)
            elif kind == "mv":
                w = cov_weight_mv(tgt, z)
            else:
                raise ValueError(kind)
            out[f"z{l + 1}"] = pearson(w, true_weight(net, l, depth))
        return out
    finally:
        restore_noise_field(net)


def forward_quality(net, x, labels, depth: int, kind: str, param=None,
                    draws: int = 8) -> dict:
    """その介入のもとでの前向き信号（推論）の質．推定量だけの変更なら不変."""
    if kind == "per":
        # 最も負荷の高いケース（読み出しミラー測定時 = 上流すべて周期化）で評価
        install_noise_field(net, periods=[param] * (depth - 1) + [1])
    elif kind == "sig":
        install_noise_field(net, periods=[1] * depth, std_scale=[param] * depth)
    else:
        install_noise_field(net, periods=[1] * depth)
    try:
        with torch.no_grad():
            ys = torch.stack([net(x) for _ in range(draws)], dim=0)
            return {"ce": float(tF.cross_entropy(ys.mean(dim=0), labels)),
                    "acc": float((ys.mean(dim=0).argmax(1) == labels).float().mean()),
                    "ensemble_std": float(ys.std(dim=0).mean())}
    finally:
        restore_noise_field(net)


def main() -> None:
    p = argparse.ArgumentParser(description="S1: offline screening of noise-field / "
                                            "estimator interventions.")
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--n-train", type=int, default=1000)
    p.add_argument("--n-inputs", type=int, default=256)
    p.add_argument("--num-samples", type=int, default=64)
    p.add_argument("--noise", choices=("gaussian", "uniform"), default="gaussian")
    p.add_argument("--sigma", type=float, default=0.5,
                   help="ガウスなら std，一様なら半幅（uniform 時は既定 1.0）")
    p.add_argument("--crossing-h", type=float, default=0.2)
    p.add_argument("--pretrain", type=int, default=400)
    p.add_argument("--pretrain-lr", type=float, default=1e-3)
    p.add_argument("--passes", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out", type=str, default="tmp/out/fncl_nf1")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.depth, args.width = 3, 32
        args.n_train, args.n_inputs = 100, 64
        args.num_samples, args.passes, args.pretrain = 16, 2, 30
    if args.noise == "uniform" and args.sigma == 0.5:
        args.sigma = 1.0
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    x_all, y_all = load_mnist(args.n_train * 2)
    perm = np.random.default_rng(999).permutation(len(y_all))
    tr = perm[:args.n_train]
    x_tr = torch.tensor(x_all[tr], device=device)
    y_tr = torch.tensor(y_all[tr], device=device)
    x, labels = x_tr[:args.n_inputs], y_tr[:args.n_inputs]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    net = build_net_noise(args.noise, args.depth, args.width, args.sigma,
                          args.crossing_h, args.num_samples, device)
    opt = torch.optim.Adam(net.parameters(), lr=args.pretrain_lr)
    for _ in range(args.pretrain):
        opt.zero_grad()
        tF.cross_entropy(net(x_tr), y_tr).backward()
        opt.step()
    with torch.no_grad():
        lg = net(x_tr)
        print(f"pretrained {args.pretrain} ep: CE={float(tF.cross_entropy(lg, y_tr)):.4f} "
              f"acc={float((lg.argmax(1) == y_tr).float().mean()):.3f}", flush=True)

    arms = [("diag", "diag", None), ("unif", "unif", None), ("r1lo", "r1lo", None),
            ("wood", "wood", None), ("cm", "cm", None), ("rank1", "rank1", None),
            ("mv", "mv", None)]
    # 位相クラスあたり 2 サンプル以上必要なので period <= T/2 に限る
    arms += [(f"per{q}", "per", q) for q in (4, 8, 16, 32)
             if q <= args.num_samples // 2]
    arms += [(f"sig{s}", "sig", s) for s in (1.5, 2.0)]

    results = {"config": vars(args), "arms": {}}
    for name, kind, param in arms:
        mr = mirrors_under(net, x, args.depth, args.passes, kind, param)
        fq = forward_quality(net, x, labels, args.depth, kind, param)
        results["arms"][name] = {"mirror_r": mr, "forward": fq}
        print(f"  {name:8s} mirror r: "
              + " ".join(f"{k}={v:.4f}" for k, v in mr.items())
              + f" | fwd CE={fq['ce']:.4f} acc={fq['acc']:.3f} "
              f"ens.std={fq['ensemble_std']:.4f}", flush=True)

    layers = [f"z{l + 1}" for l in range(args.depth)]
    lines = ["| arm | " + " | ".join(layers)
             + " | min r | fwd CE | ens.std |", "|---" * (len(layers) + 4) + "|"]
    for name, res in results["arms"].items():
        mr = res["mirror_r"]
        lines.append(f"| {name} | "
                     + " | ".join(f"{mr[k]:.4f}" for k in layers)
                     + f" | {min(mr.values()):.4f} | {res['forward']['ce']:.4f} "
                     f"| {res['forward']['ensemble_std']:.4f} |")
    print("\n" + "\n".join(lines) + "\n", flush=True)
    save_json(out_dir / "results.json", results)


if __name__ == "__main__":
    main()
