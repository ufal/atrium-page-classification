"""
tests/conftest.py – session-level setup shared by all test modules.

Responsibilities:
  1. Force the non-interactive Agg backend BEFORE any matplotlib import
     (utils.py does `from matplotlib import pyplot as plt` at module level).
  2. Insert the project root onto sys.path so bare `import utils` etc. work
     regardless of the working directory pytest is invoked from.
"""
import sys
import matplotlib

# ── 1. Non-interactive backend ──────────────────────────────────────────────
# Must be set before the first `import matplotlib.pyplot` anywhere in the
# process.  conftest.py is the earliest hook pytest executes, so this
# reliably takes effect before test modules are collected.
matplotlib.use("Agg")

# ── 2. Project-root on sys.path ──────────────────────────────────────────────
from pathlib import Path  # noqa: E402  (import after matplotlib setup is intentional)

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))