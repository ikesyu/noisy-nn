"""Compatibility shim — the consolidation library now lives in the
tmp/consolidation/ package (see consolidation/core.py).

Existing `from consolidation_lib import X` keeps working unchanged; new code
should prefer `from consolidation import X`. This shim re-binds every public and
module-level name from consolidation.core so any prior import continues to
resolve. Remove it once all importers (consolidation_*.py and multivalued_*.py)
have migrated to `import consolidation`.
"""
from consolidation import core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})
del _core
