"""丸めシフトと飽和 (組み合わせ式ヘルパ)。

ゴールデンモデルの rshift / sat と bit-exact:
    rshift(x, s) = (x + 2^(s-1)) >>> s   (s <= 0 は左シフト)
    sat(x, b)    = clamp(x, -2^(b-1), 2^(b-1) - 1)
"""
from amaranth.hdl import Mux, Value


def rshift_round(x: Value, s: int) -> Value:
    """丸め付き算術右シフト。x は signed Value、s は定数。"""
    if s <= 0:
        return x << (-s)
    return (x + (1 << (s - 1))) >> s


def saturate(x: Value, bits: int) -> Value:
    """符号込み bits ビットへの飽和。"""
    lo = -(1 << (bits - 1))
    hi = (1 << (bits - 1)) - 1
    return Mux(x < lo, lo, Mux(x > hi, hi, x))
