"""tmp/rl.constants -- shared numeric constants and repo-path bootstrap.

Importing this module puts the repository root on sys.path so that both `nnn` and
`data_nce.fncl` resolve from any script under tmp/ (same approach as
`data_nce/fncl/network.py`).
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]   # tmp/rl/ -> repo root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

EPS = 1e-6

# CartPole-v1 defaults (idea_rl.md §20.5)
GAMMA = 0.99
DEF_HIDDEN = 64
DEF_STD = 0.6
DEF_H = 0.15
DEF_T = 64
