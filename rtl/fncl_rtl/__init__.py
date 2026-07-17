"""fncl_rtl — cov_jac 学習コアの Amaranth RTL (FPGA 実装 Phase 3).

ゴールデンモデルは tmp/fncl_phase1_fxp.py (train_stream_fxp)。各モジュールは
そのビット演算 (丸めシフト rshift / 丸め除算 rdiv / 飽和 sat / LFSR /
crossing_code) と bit-exact に一致するように書かれ、rtl/tests/ で突き合わせる。
アーキテクチャは docs/idea_fpga.md §5 (ROW/COL の 2 本の 1-D PE アレイ)。
"""
from .lfsr import GaloisLfsr           # noqa: F401
from .rounding import rshift_round, saturate  # noqa: F401
from .crossing import CrossingUnit     # noqa: F401
from .divider import RoundDiv          # noqa: F401
