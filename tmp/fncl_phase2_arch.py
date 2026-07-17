"""
fncl_phase2_arch.py — FPGA 実装 Phase 2「アーキテクチャ設計」のサイクル・資源計算

docs/idea_fpga.md §5 のマイクロアーキテクチャ (ROW/COL の 2 本の 1-D PE アレイ +
ブロードキャストバス) を前提に、レーン数 L を振って

  - 提示 1 回あたりのサイクル数 (フェーズ別)
  - 更新レート / 総学習時間 (budget 192k) / 推論レート
  - メモリ量 (配列別) と EBR / DSP の概算
  - デバイス適合 (ECP5-25/45/85, CertusPro-NX, iCE40 UP5K)

を機械的に出す。数式がそのまま設計根拠なので、仕様変更時はここを更新して
表を作り直す。実行: python fncl_phase2_arch.py  (out/fncl_phase2_arch/ に保存)
"""
import math
from pathlib import Path

H, T = 64, 64                 # 凍結構成
BUDGET = 192000               # Phase 0/1 と同じ総提示回数
DIV_LAT = 26                  # パイプライン非回復除算器のレイテンシ (24bit 商)
N_DIV = 1                     # 除算器本数 (1 結果/サイクルのパイプライン型)

# 語長契約 (fncl_phase1_fxp.SAT_WIDTHS + 統計量)
W_BITS = {"W1": 18, "v_W1 (ROW)": 18, "M1 (COL)": 20, "vM 複製 (COL)": 18,
          "Σdz (COL)": 24}
VEC_WORDS = 14                # H 長ベクトル群 (W0,b0,b1,Wout,Mout,v,統計...) の本数
VEC_BITS = 20                 # 代表語長


def cycles(L: int) -> dict:
    """フェーズ別サイクル数。r = 1 レーンが受け持つ行/列数."""
    r = math.ceil(H / L)
    sample = T * (H * r + 4)              # ブロードキャスト H スロット x r + 交差/流し込み
    mirror = 2 * H * r + (2 * H) // N_DIV + DIV_LAT   # Num+EMA rmw / 除算 2H 回
    credit = (H + 4) * r                  # a0 = M^T dd1 (dd1_j を H 回ブロードキャスト)
    update = 2 * H * r + 8 * r + 32       # ROW(i 順) と COL(j 順, KP 複製) + ベクトル
    misc = 64                             # FSM 遷移・ROM 引き・L0 ベクトル演算
    total = sample + mirror + credit + update + misc
    return {"sample": sample, "mirror": mirror, "credit": credit,
            "update": update, "misc": misc, "total": total,
            "infer": sample + misc}


def memory_bits() -> dict:
    m = {k: H * H * b for k, b in W_BITS.items()}
    m["H 長ベクトル群"] = VEC_WORDS * H * VEC_BITS
    return m


def dsp(L: int) -> int:
    """ROW 1 個/レーン + COL 2 個/レーン (24x18 は 2 DSP 相当) + 共有 4."""
    return 3 * L + 4


def ebr(L: int) -> int:
    """レーンローカル分割の EBR 数: ROW 側 1 本/レーン (W1+v をパック),
    COL 側 2 本/レーン (M1+vM / Σdz)。L が小さいほど深く詰められる."""
    return 3 * L


DEVICES = [
    # (名称, LUT, EBR(18kb), DSP18x18) — 概数。採用前にデータシートで要確認
    ("ECP5-25 (LFE5U-25)", 24000, 56, 28),
    ("ECP5-45 (LFE5U-45)", 44000, 108, 72),
    ("ECP5-85 (LFE5U-85)", 84000, 208, 156),
    ("CertusPro-NX-40 (概数)", 39000, 140, 56),
    ("CertusPro-NX-100 (概数)", 96000, 224, 156),
]


def main():
    out_dir = Path("out/fncl_phase2_arch")
    out_dir.mkdir(parents=True, exist_ok=True)
    fclk = 75e6
    lines = [f"**Phase 2 アーキテクチャ見積り** (H={H}, T={T}, 語長契約 = "
             "fncl_phase1_fxp.SAT_WIDTHS, fclk = 75 MHz 想定)", ""]

    lines += ["| レーン数 L | sample | mirror | credit | update | 合計 cyc/提示 "
              "| 更新レート | 192k 学習 | 推論 (T=64) | DSP | EBR |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for L in (64, 32, 16, 8):
        c = cycles(L)
        upd = fclk / c["total"]
        inf = fclk / c["infer"]
        lines.append(
            f"| {L} | {c['sample']} | {c['mirror']} | {c['credit']} "
            f"| {c['update']} | **{c['total']}** | {upd/1e3:.1f} kHz "
            f"| {BUDGET / upd:.0f} s | {inf/1e3:.1f} kHz "
            f"| {dsp(L)} | {ebr(L)} |")

    lines += ["", "**メモリ内訳** (学習コア全体):", "",
              "| 配列 | 語長 x 深さ | kb |", "|---|---|---|"]
    total = 0
    for name, bits in memory_bits().items():
        total += bits
        depth = bits // ({**W_BITS, "H 長ベクトル群": VEC_BITS}[name])
        lines.append(f"| {name} | {({**W_BITS, 'H 長ベクトル群': VEC_BITS})[name]}b x {depth} "
                     f"| {bits / 1024:.0f} |")
    lines.append(f"| **合計** |  | **{total / 1024:.0f}** |")
    lines += ["", f"(推論専用に縮退させると W1/W0/Wout + LFSR のみ -> "
              f"{(H * H * 18 + VEC_WORDS // 3 * H * VEC_BITS) / 1024:.0f} kb 程度)"]

    lines += ["", "**デバイス適合** (概数; 採用前にデータシート確認):", "",
              "| デバイス | LUT | EBR | DSP | 適合するレーン数 |",
              "|---|---|---|---|---|"]
    for name, lut, ebr_n, dsp_n in DEVICES:
        fits = [str(L) for L in (64, 32, 16, 8)
                if dsp(L) <= dsp_n and ebr(L) <= ebr_n]
        lines.append(f"| {name} | {lut // 1000}k | {ebr_n} | {dsp_n} "
                     f"| L ≤ {fits[0] if fits else '-'} |")
    lines += ["", "iCE40 UP5K (5.3k LUT, 8 DSP, 120kb EBR + 1Mb SPRAM) は学習コアは"
              "不可、推論専用 (L=8, ~1.4 kHz @48MHz) なら W1 を SPRAM に置いて可。"]

    table = "\n".join(lines) + "\n"
    print(table)
    (out_dir / "table_arch.md").write_text(table, encoding="utf-8")
    print(f"saved {out_dir / 'table_arch.md'}")


if __name__ == "__main__":
    main()
