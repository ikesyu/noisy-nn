"""neuromod — the standard Foraging / Avoidance / Sheltering benchmark for NNN
noise fields.

**Scope rule.** This package holds only what belongs to *this problem*: the
problem setup, the data generation, the benchmark's train/score protocol, and all
drawing and file output.  It deliberately contains no general NNN machinery.
Learning rules live in `nnn/`, and the RL, consolidation, and reservoir lines
have their own packages; the neuromodulation work barely touches them, so nothing
from those directions should migrate here.

    world     the problem: scene, sensing, behaviour targets, and the closed-loop
              drives that gate which field is recruited
    fields    this benchmark's neuromodulator-like noise fields, plus the
              gauge-invariant participation readout (crossing rate nu)
    protocol  how the benchmark is trained and scored: one shared weight set under
              three fields, per-state error, field separation
    viz       every matplotlib panel, animation, and file save

Individual challenges are scripts directly under `tmp/` (`neuromod_*.py`) that
import this package and vary only their own question.  `tmp/neuromod_behavior_modes.py`
is the reference driver.

Canon: symbols follow `docs/idea_core.md`; research context is `docs/idea_neuromod.md`.
"""
