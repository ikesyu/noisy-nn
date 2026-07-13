"""
fncl6_1.py — 論文 §6「ハードウェア指向解析」の定量支援 (Tab.2)

1 回の重み更新あたりの演算量・ワーキングメモリを、backprop / cov_deriv /
cov_jac について N (入力数), T (確率サンプル数), H (隠れ幅), L (隠れ層数) の
関数として MAC レベルで概算し、代表値で数表 (markdown) を出力する。
併せて、ガウスノイズと一様ノイズの HW プリミティブ比較表を出力する。

これは回路実装ではない。論文の Tab.2 (定性比較) を裏づける桁の見積もりであり、
「FPGA での優位性の証明」を主張するためのものではない (docs §11 の注意を参照)。

注意 (係数の解釈):
  - 隠れ activation z は二値 {0,1} なので、z を掛ける積和 (外積更新・共分散統計の
    大半) は乗算器ではなく AND ゲート + 加算器 (gated add) に退化する。表では
    これを "MAC(z-gated)" として通常の MAC と区別する。
  - backprop 参照実装も同じ確率的モデル (T サンプル) を流すため、forward 量は
    全手法共通としてカウントから除外し、「forward 以外に追加で必要な量」を比べる。

生成物 (out/fncl6_1/):
  table_ops.md        -> 更新 1 回あたりの追加演算・メモリの概算表
  table_primitives.md -> ガウス vs 一様の HW プリミティブ比較表

実行例:
  python tmp/fncl6_1.py
  python tmp/fncl6_1.py --N 128 --T 64 --H 64 --layers 2
"""
import argparse
from pathlib import Path


def fmt(n: float) -> str:
    if n == 0:
        return "0"
    for unit, s in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
        if abs(n) >= unit:
            return f"{n / unit:.2f}{s}"
    return f"{n:.0f}"


def op_counts(N: int, T: int, H: int, L: int, d_in: int = 1, d_out: int = 1):
    """更新 1 回あたり、forward 以外に追加で必要な演算量の概算 (式は列 'formula')."""
    # 重み行列サイズの合計 (バイアス除く)
    w_sizes = d_in * H + (L - 1) * H * H + H * d_out

    rows = []

    # ---------- backprop ----------
    bwd_matmul = N * T * ((L - 1) * H * H + H * d_out)   # W^T delta の逆方向行列積
    wgrad = N * T * w_sizes                              # delta * z_prev^T の外積
    rows.append(dict(
        method="backprop",
        item="backward transposed matmul (W^T delta)",
        formula="N*T*((L-1)*H^2 + H*d_out)", count=bwd_matmul, kind="MAC",
        note="転置重みへの第 2 データパス + forward/backward 同期が必要"))
    rows.append(dict(
        method="backprop", item="weight-gradient outer products",
        formula="N*T*sum(W sizes)", count=wgrad, kind="MAC(z-gated)",
        note="z_prev は二値"))
    rows.append(dict(
        method="backprop", item="working memory (delta buffers)",
        formula="N*T*H*L", count=N * T * H * L, kind="words",
        note="+ 転置アクセス可能な重み複製 (sum(W sizes) words)"))

    # ---------- cov_deriv ----------
    stats = 2 * N * T * H * L      # (L*z, z^2) の累算; z 二値なので実質 gated add
    slope = N * T * H * L          # KDE 傾き: xor カウンタの差分 (加算のみ)
    credit_div = N * H * L         # per-input credit の除算
    outer = N * T * (d_in * H + (L - 1) * H * H)  # 隠れ層の外積更新
    readout = N * H * d_out        # ensemble-mean readout の外積
    rows.append(dict(
        method="cov_deriv", item="covariance statistics (L*z, z^2)",
        formula="2*N*T*H*L", count=stats, kind="MAC(z-gated)",
        note="ユニットあたり 4 アキュムレータ"))
    rows.append(dict(
        method="cov_deriv", item="KDE slope (xor counter difference)",
        formula="N*T*H*L", count=slope, kind="add",
        note="forward の交差カウンタを再利用"))
    rows.append(dict(
        method="cov_deriv", item="credit divisions",
        formula="N*H*L", count=credit_div, kind="div",
        note="Cov/Var; 近似除算で可"))
    rows.append(dict(
        method="cov_deriv", item="weight-update outer products",
        formula="N*T*(d_in*H + (L-1)*H^2) + N*H*d_out",
        count=outer + readout, kind="MAC(z-gated)", note=""))
    rows.append(dict(
        method="cov_deriv", item="working memory (per-unit stats)",
        formula="4*H*L", count=4 * H * L, kind="words",
        note="転置重みパス・delta バッファ不要"))

    # ---------- cov_jac ----------
    mirror = N * T * ((L - 1) * H * H + H * d_out + H * L)  # Cov(d_next, z) + Var(z)
    mirror_div = (L - 1) * H * H + H * d_out                # プーリング後に 1 回
    recursion = N * ((L - 1) * H * H + H * d_out)           # delta の再帰 (入力毎)
    rows.append(dict(
        method="cov_jac", item="mirror statistics Cov(d_next, z), Var(z)",
        formula="N*T*((L-1)*H^2 + H*d_out + H*L)", count=mirror,
        kind="MAC(z-gated)", note="d_next は連続量 (真の乗算はここだけ)"))
    rows.append(dict(
        method="cov_jac", item="mirror divisions (after pooling)",
        formula="(L-1)*H^2 + H*d_out", count=mirror_div, kind="div",
        note="入力プーリングで更新 1 回につき 1 回"))
    rows.append(dict(
        method="cov_jac", item="credit recursion (W_hat^T delta)",
        formula="N*((L-1)*H^2 + H*d_out)", count=recursion, kind="MAC",
        note="T に依らない (入力毎に 1 回)"))
    rows.append(dict(
        method="cov_jac", item="weight-update outer products",
        formula="same as cov_deriv", count=outer + readout,
        kind="MAC(z-gated)", note=""))
    rows.append(dict(
        method="cov_jac", item="working memory (mirrors + EMA)",
        formula="2*sum(W sizes)", count=2 * w_sizes, kind="words",
        note="mirror は forward 統計から測定 (転置読み出しではない)"))

    return rows


PRIMITIVES_TABLE = """\
| primitive | gaussian noise | uniform noise |
|---|---|---|
| 乱数生成 | Box-Muller / CLT 近似 (追加回路が必要) | LFSR そのまま (シフト + XOR) |
| 発火確率のサポート | 全域で正 (裾が無限) | abs(d-c) >= r で厳密に 0 (コンパクトサポート) |
| 交差 activation | 比較器 x2 + XOR | 同一 |
| 分布フリー局所微分 phi_T'(d) | xor カウンタ差分 / 2h (分布知識不要) | 同一 |
| 学習統計 (Cov, Var) | カウンタ + 積和 + 除算 | 同一 (z 二値のため大半は gated add) |
| (解析式を使う場合) phi_bar / phi_bar' | 2F(1-F) / 2(1-2F)p(d), erf 系 (LUT) | 2 次多項式 / 一次式 -(d-c)/r^2 |

- 分布フリーの phi_T' を使う限り、ノイズ分布の解析式は推論にも学習にも一切
  現れない (kde = analytic は §5.4/§6.2 で確認)。ネットワークの動作は分布の
  取り替えに対して不変。
- 一様ノイズを選ぶ本質的な理由は 2 点: (1) 乱数生成が LFSR で閉じる、
  (2) 期待応答がコンパクトサポート (しきい値から遠いユニットはスパイクも
  学習統計も厳密に 0 -> イベント駆動実装での活動・電力の疎性)。
  解析式の単純さは KDE slope を使う構成では使われず、副次的。
"""


def main() -> None:
    p = argparse.ArgumentParser(
        description="§6 hardware-oriented analysis: rough per-update operation "
                    "counts (Tab.2 support).")
    p.add_argument("--N", type=int, default=128, help="入力点数")
    p.add_argument("--T", type=int, default=64, help="確率サンプル数")
    p.add_argument("--H", type=int, default=64, help="隠れ幅")
    p.add_argument("--layers", type=int, default=2, help="隠れ層数 L")
    p.add_argument("--out", type=str, default="out/fncl6_1")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = op_counts(args.N, args.T, args.H, args.layers)

    header = (f"1 回の重み更新あたりの追加演算 (forward は全手法共通なので除外). "
              f"N={args.N}, T={args.T}, H={args.H}, L={args.layers}.\n")
    lines = [header,
             "| method | item | formula | count | kind | note |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['method']} | {r['item']} | `{r['formula']}` | "
                     f"{fmt(r['count'])} | {r['kind']} | {r['note']} |")
    lines.append("")
    totals = {}
    for r in rows:
        if r["kind"].startswith("MAC") or r["kind"] == "add":
            totals[r["method"]] = totals.get(r["method"], 0) + r["count"]
    lines.append("| method | total ops (MAC + add) |")
    lines.append("|---|---|")
    for m, v in totals.items():
        lines.append(f"| {m} | {fmt(v)} |")
    table = "\n".join(lines) + "\n"
    print(table)
    (out / "table_ops.md").write_text(table, encoding="utf-8")
    print(f"  saved {out / 'table_ops.md'}")

    print(PRIMITIVES_TABLE)
    (out / "table_primitives.md").write_text(PRIMITIVES_TABLE, encoding="utf-8")
    print(f"  saved {out / 'table_primitives.md'}")


if __name__ == "__main__":
    main()
