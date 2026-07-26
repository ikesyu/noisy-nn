"""consolidation — noise-field self-consolidation library (docs/idea_consolidation.md).

The shared primitives used by the consolidation experiments (and by the
multivalued experiments): ConsolidableNNN (the rho=(sigma,h) mobilisation dial),
the vanishing-path ops (anneal/snap/kill), the persistent-state cov_jac trainer,
redundancy scoring, per-task mobilisation fields + descriptors, and multi-task
helpers. See consolidation.core for the full API.

Split out of the flat tmp/consolidation_lib.py (kept as a compat shim) into this
package, mirroring tmp/rl/ and tmp/reservoir/. Experiment drivers stay as
tmp/consolidation_*.py; the canonical results live in docs/idea_consolidation.md
(settled values: §12.9.13 + §15).
"""
from .core import *  # noqa: F401,F403
from . import core  # noqa: F401
