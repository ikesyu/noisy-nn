"""
fncl6_1.py — 論文 §6.1 Table 4 (ハードウェア指向解析) の数値を算出する.

backprop / cov_deriv / cov_jac の 1 回の重み更新あたりの追加コスト
(T 回の forward サンプリングは全手法共通なので除外) を、
N (入力数), T (確率サンプル数), H (隠れ幅), L (隠れ層数) から見積もり、
標準出力に表として表示する。既定値 N=128, T=64, H=64, L=2 が論文の Table 4。

  Full-Precision MACs : 通常の乗算器を要する積和
  Binary-Gated MACs   : 隠れ activation z が二値 {0,1} のため AND ゲート +
                        加算器で済む積和 (KDE slope のカウンタ加算も含む)
  Divisions           : Cov/Var などの除算
  Memory (words)      : ワーキングメモリ

実行例:
  python data_nce/fncl6_1.py
  python data_nce/fncl6_1.py --N 128 --T 64 --H 64 --layers 2
"""
import argparse


def fmt(n: float) -> str:
    """概数表示 (Table 4 の表記に対応する 3 有効数字の k/M 表示)."""
    if n == 0:
        return "0"
    for unit, s in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
        if abs(n) >= unit:
            return f"{n / unit:.3g}{s}"
    return f"{n:.0f}"


def table4(N: int, T: int, H: int, L: int, d_in: int = 1, d_out: int = 1) -> dict:
    """Table 4 の 4 列 (method ごとの dict) を返す.

    memory は成分のタプル (backprop は delta バッファ + 転置重み複製の 2 成分)。
    """
    w_sizes = d_in * H + (L - 1) * H * H + H * d_out   # 全重み行列の要素数
    inter = (L - 1) * H * H + H * d_out                # 層間 (隠れ間 + readout) の重み数
    outer = N * T * (d_in * H + (L - 1) * H * H)       # 隠れ層の外積更新
    readout = N * H * d_out                            # ensemble-mean readout の外積
    return {
        "backprop": {
            # W^T delta の逆方向行列積 (delta は連続量 -> 真の乗算)
            "full_mac": N * T * inter,
            # delta z_prev^T の外積 (z 二値)
            "gated_mac": N * T * w_sizes,
            "div": 0,
            # delta バッファ + 転置アクセス可能な重み複製 (W^T)
            "memory": (N * T * H * L, w_sizes),
        },
        "cov_deriv": {
            "full_mac": 0,
            # 共分散統計 (L*z, z^2) + KDE slope カウンタ差分 + 外積更新
            "gated_mac": 2 * N * T * H * L + N * T * H * L + outer + readout,
            # per-input credit の除算 Cov/Var
            "div": N * H * L,
            # ユニットあたり 4 アキュムレータ
            "memory": (4 * H * L,),
        },
        "cov_jac": {
            # credit 再帰 W_hat^T delta (T に依らず入力毎に 1 回)
            "full_mac": N * inter,
            # mirror 統計 Cov(d_next, z), Var(z) + 外積更新
            "gated_mac": N * T * (inter + H * L) + outer + readout,
            # mirror の除算 (入力プーリング後に 1 回)
            "div": inter,
            # mirror + その EMA の 2 セット
            "memory": (2 * w_sizes,),
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="§6.1 Table 4: rough per-update operation counts.")
    p.add_argument("--N", type=int, default=128, help="入力点数")
    p.add_argument("--T", type=int, default=64, help="確率サンプル数")
    p.add_argument("--H", type=int, default=64, help="隠れ幅")
    p.add_argument("--layers", type=int, default=2, help="隠れ層数 L")
    args = p.parse_args()

    rows = table4(args.N, args.T, args.H, args.layers)

    print(f"Table 4 — 1 回の重み更新あたりの追加コスト "
          f"(T 回の forward は全手法共通なので除外)")
    print(f"N={args.N}, T={args.T}, H={args.H}, L={args.layers}\n")
    header = ["method", "Full-Precision MACs", "Binary-Gated MACs",
              "Divisions", "Memory (words)"]
    print("| " + " | ".join(header) + " |")
    print("|---" * len(header) + "|")
    for method, r in rows.items():
        cells = [f"{r['full_mac']:,} ({fmt(r['full_mac'])})",
                 f"{r['gated_mac']:,} ({fmt(r['gated_mac'])})",
                 f"{r['div']:,} ({fmt(r['div'])})"]
        mem = " + ".join(f"{m:,}" for m in r["memory"])
        mem += " (" + " + ".join(fmt(m) for m in r["memory"]) + ")"
        cells.append(mem)
        print(f"| {method} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
