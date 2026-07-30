"""
fncl_mnist_fidelity.py — 査読対応 A1「深い/広いネットでの勾配忠実度の実測」(docs/draft_nce.md 追記 §A1)
（旧名: fncl_a1.py．出力ディレクトリ tmp/out/ は旧名のまま）

MNIST (784 次元入力・10 クラス・交差エントロピー損失) 上の NNN
784-256-...-256-10 (隠れ層数 --depths で掃引) で、§5.3 と同じプロトコルにより

  (a) 共分散 weight mirror の回復精度 (Pearson r, 層別),
  (b) cov_jac の更新方向と autograd 厳密勾配の層別 cosine 類似度・ノルム比
      (未学習 / backprop --pretrain-epochs 事前学習後の 2 状態, --grad-draws 平均)

を測る。単変量ミラーの対角近似誤差が再帰で層を跨いで蓄積するか (draft_nce §7 の
open problem) の直接測定。完了条件: cosine > 0.9 なら green (改訂稿 Fig 拡張へ)。

読み出し誤差は解析形 (softmax - onehot) を用いる cov_jac 構成 (cov_jac_full ではない)。
対照として cov_deriv (スカラー共分散 credit) の cosine も併record する。

生成物 (tmp/out/fncl_a1/): results.json, (標準出力) 層別 cosine 表

実行例:
  .venv/bin/python tmp/fncl_mnist_fidelity.py --quick
  .venv/bin/python tmp/fncl_mnist_fidelity.py --device cuda            # 本番 (depth 2,3,4)
  .venv/bin/python tmp/fncl_mnist_fidelity.py --device cuda --depths 2
"""
from __future__ import annotations

import argparse
import gzip
import struct
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as tF

DATA_DIR = Path(__file__).resolve().parent.parent / "data_nce"
sys.path.insert(0, str(DATA_DIR))
import fncl  # noqa: E402
from fncl_common import cosine, norm_ratio, pearson, save_json  # noqa: E402
from nnn import model as nnn_model  # noqa: E402

MNIST_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
MNIST_DIR = Path(__file__).resolve().parent / "data" / "mnist"


# ============================================================
# MNIST (raw idx, 依存ライブラリなし)
# ============================================================
def _fetch(name: str) -> Path:
    MNIST_DIR.mkdir(parents=True, exist_ok=True)
    path = MNIST_DIR / name
    if not path.exists():
        print(f"downloading {name} ...", flush=True)
        urllib.request.urlretrieve(MNIST_URL + name, path)
    return path


def load_mnist(n: int, seed: int = 12345):
    """先頭から class-balanced に n 枚を選ぶ。x in [-1,1]^784, y in {0..9}."""
    with gzip.open(_fetch("train-images-idx3-ubyte.gz")) as f:
        _, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)
    with gzip.open(_fetch("train-labels-idx1-ubyte.gz")) as f:
        _ = f.read(8)
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    rng = np.random.default_rng(seed)
    idx = []
    per_class = n // 10
    for c in range(10):
        cand = np.where(labels == c)[0]
        idx.append(rng.choice(cand, per_class, replace=False))
    idx = rng.permutation(np.concatenate(idx))
    x = images[idx].astype(np.float32) / 255.0 * 2.0 - 1.0
    y = labels[idx].astype(np.int64)
    return x, y


# ============================================================
# モデル・サンプル収集 (fncl5_3 の任意深さ/多クラス CE 一般化)
# ============================================================
def build_net(depth: int, width: int, sigma: float, h: float, t: int,
              device: torch.device):
    structure = [784] + [width] * depth + [10]
    net = nnn_model.SimpleNNNSample(structure=structure, std=sigma, h=h, t=t,
                                    output_bias=True)
    return net.to(device)


def collect_samples(net, x, passes: int = 1) -> dict:
    cap = fncl.Capture(net)
    z = [[] for _ in range(cap.n_hidden)]
    d = [[] for _ in range(cap.n_hidden)]
    ys, y_out = [], []
    with torch.no_grad():
        for _ in range(passes):
            y_out.append(net(x))
            ys.append(cap.y_samples)
            for l in range(cap.n_hidden):
                z[l].append(cap.z[l])
                d[l].append(cap.d[l])
    cap.remove()
    return {"y": torch.stack(y_out, dim=0).mean(dim=0),      # [N, 10]
            "ys": torch.cat(ys, dim=1),                      # [N, K*T, 10]
            "z": [torch.cat(zl, dim=1) for zl in z],
            "d": [torch.cat(dl, dim=1) for dl in d]}


def ce_error(y_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """dL/dy of mean cross-entropy on the ensemble-mean logits: softmax - onehot."""
    return torch.softmax(y_logits, dim=1) - tF.one_hot(labels, 10).float()


def cov_jac_gradient(net, x, labels) -> dict:
    """cov_jac の 1 draw 勾配推定 (mirror は当該 forward から直接測定, EMA なし)."""
    s = collect_samples(net, x, passes=1)
    crossings = fncl.crossing_layers(net)
    n_hidden = len(crossings)
    slope_full = [fncl.kde_slope(crossings[l], s["d"][l]) for l in range(n_hidden)]
    slope_mean = [sf.mean(dim=1) for sf in slope_full]
    w_hat = {"out": fncl.cov_weight(s["ys"], s["z"][-1], pool=True)}   # [10, H]
    for l in range(1, n_hidden):
        w_hat[l] = fncl.cov_weight(s["d"][l], s["z"][l - 1], pool=True)
    N, T = x.shape[0], s["z"][0].shape[1]
    e = ce_error(s["y"], labels)                                       # [N, 10]
    a = [None] * n_hidden
    a[-1] = e @ w_hat["out"]                                           # [N, H]
    for l in range(n_hidden - 2, -1, -1):
        a[l] = (a[l + 1] * slope_mean[l + 1]) @ w_hat[l + 1]
    z_prev = ([x.unsqueeze(1).expand(N, T, x.shape[1])]
              + [s["z"][l] for l in range(n_hidden - 1)])
    grads = {}
    for l in range(n_hidden):
        delta = a[l].unsqueeze(1) * slope_full[l]                      # [N, T, H]
        grads[f"w{l}"] = torch.einsum("nto,nti->oi", delta, z_prev[l]) / (N * T)
    z_bar = s["z"][-1].mean(dim=1)
    grads["wout"] = torch.einsum("no,ni->oi", e, z_bar) / N
    return grads


def cov_deriv_gradient(net, x, labels) -> dict:
    """cov_deriv (per-input スカラー共分散 credit, kde slope) の 1 draw (対照用)."""
    s = collect_samples(net, x, passes=1)
    crossings = fncl.crossing_layers(net)
    n_hidden = len(crossings)
    # per-sample CE loss on the per-sample logits
    N, T = x.shape[0], s["z"][0].shape[1]
    ys = s["ys"]                                                       # [N, T, 10]
    L = tF.cross_entropy(ys.reshape(N * T, 10),
                         labels.unsqueeze(1).expand(N, T).reshape(N * T),
                         reduction="none").view(N, T)
    z_prev = ([x.unsqueeze(1).expand(N, T, x.shape[1])]
              + [s["z"][l] for l in range(n_hidden - 1)])
    grads = {}
    for l in range(n_hidden):
        g_bcast, _ = fncl.covariance_credit(s["z"][l], L, "per_input")
        dz_dd = fncl.kde_slope(crossings[l], s["d"][l])
        delta = g_bcast * dz_dd
        grads[f"w{l}"] = torch.einsum("nto,nti->oi", delta, z_prev[l]) / (N * T)
    e = ce_error(s["y"], labels)
    z_bar = s["z"][-1].mean(dim=1)
    grads["wout"] = torch.einsum("no,ni->oi", e, z_bar) / N
    return grads


def autograd_gradient(net, x, labels) -> dict:
    for prm in net.parameters():
        prm.grad = None
    loss = tF.cross_entropy(net(x), labels)
    loss.backward()
    grads = {f"w{l}": net.fcs[l].weight.grad.detach().clone()
             for l in range(len(net.fcs) - 1)}
    grads["wout"] = net.fcs[-1].weight.grad.detach().clone()
    return grads


def averaged(fn, draws: int) -> dict:
    acc = None
    for _ in range(draws):
        g = fn()
        acc = g if acc is None else {k: acc[k] + v for k, v in g.items()}
    return {k: v / draws for k, v in acc.items()}


def pretrain_backprop(net, x, labels, lr: float, epochs: int, log_every: int):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for epoch in range(epochs):
        opt.zero_grad()
        logits = net(x)
        loss = tF.cross_entropy(logits, labels)
        loss.backward()
        opt.step()
        if log_every and (epoch % log_every == 0 or epoch == epochs - 1):
            acc = float((logits.argmax(1) == labels).float().mean())
            print(f"  [pretrain] epoch {epoch:4d}  CE={float(loss):.4f}  "
                  f"acc={acc:.3f}", flush=True)


# ============================================================
# main
# ============================================================
def main() -> None:
    p = argparse.ArgumentParser(description="A1: cov_jac gradient fidelity on "
                                            "MNIST 784-256-...-10 with CE loss.")
    p.add_argument("--depths", type=str, default="2,3,4",
                   help="カンマ区切り: 隠れ層数の掃引")
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--n-inputs", type=int, default=256, help="MNIST サブセット枚数")
    p.add_argument("--num-samples", type=int, default=64, help="T")
    p.add_argument("--sigma", type=float, default=0.5)
    p.add_argument("--crossing-h", type=float, default=0.2)
    p.add_argument("--mirror-passes", type=int, default=8)
    p.add_argument("--grad-draws", type=int, default=32)
    p.add_argument("--pretrain-epochs", type=int, default=300)
    p.add_argument("--pretrain-lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=str, default="tmp/out/fncl_a1")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.depths, args.width, args.n_inputs = "2", 32, 64
        args.num_samples, args.mirror_passes, args.grad_draws = 16, 2, 4
        args.pretrain_epochs = 20
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    x_np, y_np = load_mnist(args.n_inputs)
    x = torch.tensor(x_np, device=device)
    labels = torch.tensor(y_np, device=device)
    print(f"MNIST subset: {x.shape}, device={device}", flush=True)

    results = {"config": vars(args), "runs": {}}
    for depth in [int(d) for d in args.depths.split(",")]:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        net0 = build_net(depth, args.width, args.sigma, args.crossing_h,
                         args.num_samples, device)
        init_state = {k: v.clone() for k, v in net0.state_dict().items()}
        run = {}
        for state in ("untrained", "pretrained"):
            net = build_net(depth, args.width, args.sigma, args.crossing_h,
                            args.num_samples, device)
            net.load_state_dict(init_state)
            if state == "pretrained":
                print(f"[depth {depth} | {state}] backprop pretraining "
                      f"({args.pretrain_epochs} epochs) ...", flush=True)
                pretrain_backprop(net, x, labels, args.pretrain_lr,
                                  args.pretrain_epochs,
                                  max(1, args.pretrain_epochs // 5))

            # (a) mirror 回復精度 (mirror_passes 分のサンプルで測定)
            s = collect_samples(net, x, args.mirror_passes)
            mirror = {}
            for l in range(1, depth):
                w_hat = fncl.cov_weight(s["d"][l], s["z"][l - 1], pool=True)
                mirror[f"w{l}"] = pearson(w_hat, net.fcs[l].weight)
            wout_hat = fncl.cov_weight(s["ys"], s["z"][-1], pool=True)
            mirror["wout"] = pearson(wout_hat, net.fcs[-1].weight)
            run.setdefault(state, {})["mirror_r"] = mirror
            print(f"[depth {depth} | {state}] mirror r: "
                  + "  ".join(f"{k}={v:.4f}" for k, v in mirror.items()),
                  flush=True)

            # (b) cov_jac / cov_deriv vs autograd (draw 平均後に比較)
            print(f"[depth {depth} | {state}] averaging gradients over "
                  f"{args.grad_draws} draws ...", flush=True)
            g_auto = averaged(lambda: autograd_gradient(net, x, labels),
                              args.grad_draws)
            g_jac = averaged(lambda: cov_jac_gradient(net, x, labels),
                             args.grad_draws)
            g_der = averaged(lambda: cov_deriv_gradient(net, x, labels),
                             args.grad_draws)
            for est_name, g_est in (("cov_jac", g_jac), ("cov_deriv", g_der)):
                fid = {k: {"cos": cosine(g_est[k], g_auto[k]),
                           "ratio": norm_ratio(g_est[k], g_auto[k])}
                       for k in g_auto}
                run[state][est_name] = fid
                print(f"[depth {depth} | {state}] {est_name:9s} "
                      + "  ".join(f"{k}: cos={v['cos']:.4f} r={v['ratio']:.2f}"
                                  for k, v in fid.items()), flush=True)
        results["runs"][str(depth)] = run

    # ---- summary table ----
    lines = ["| depth | state | layer | mirror r | cov_jac cos (ratio) | "
             "cov_deriv cos |", "|---|---|---|---|---|---|"]
    for depth, run in results["runs"].items():
        for state in ("untrained", "pretrained"):
            layers = list(run[state]["cov_jac"].keys())
            for k in layers:
                mr = run[state]["mirror_r"].get(k)
                cj = run[state]["cov_jac"][k]
                cd = run[state]["cov_deriv"][k]
                lines.append(
                    f"| {depth} | {state} | {k} | "
                    f"{'-' if mr is None else f'{mr:.4f}'} | "
                    f"{cj['cos']:.4f} ({cj['ratio']:.2f}) | {cd['cos']:.4f} |")
    print("\n" + "\n".join(lines) + "\n", flush=True)
    save_json(out_dir / "results.json", results)


if __name__ == "__main__":
    main()
