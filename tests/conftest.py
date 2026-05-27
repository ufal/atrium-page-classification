"""
tests/conftest.py – session-level setup shared by all test modules.

Responsibilities:
  1. Force the non-interactive Agg backend BEFORE any matplotlib import
     (utils.py does `from matplotlib import pyplot as plt` at module level).
  2. Insert the project root onto sys.path so bare `import utils` etc. work
     regardless of the working directory pytest is invoked from.
  3. Mock optional heavy dependencies (clip, seaborn, …) that may be absent
     from the lightweight test venv, so classifier.py and minor_classes.py
     can be imported without them.
"""
import sys
import matplotlib

# ── 1. Non-interactive backend ──────────────────────────────────────────────
matplotlib.use("Agg")

# ── 2. Mock absent optional dependencies ────────────────────────────────────
from unittest.mock import MagicMock  # noqa: E402

# clip (OpenAI CLIP) – installed via a git URL, may be missing
if "clip" not in sys.modules:
    _clip_mock = MagicMock()
    _preprocess_mock = MagicMock()
    _preprocess_mock.transforms = [
        MagicMock(size=224),                               # [0] Resize
        MagicMock(),                                       # [1]
        MagicMock(),                                       # [2]
        MagicMock(),                                       # [3]
        MagicMock(mean=(0.48156, 0.45777, 0.40785),        # [4] Normalize
                  std=(0.26862, 0.26130, 0.27577)),
    ]
    _clip_mock.load.return_value = (MagicMock(), _preprocess_mock)
    _clip_mock.tokenize.return_value = MagicMock()
    sys.modules["clip"]       = _clip_mock
    sys.modules["clip.model"] = MagicMock()   # referenced by convert_weights()

# seaborn – imported at module level by minor_classes.py
# Any other package that is pip-optional can be added to this list.
for _mod in ("seaborn",):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

# ── 3. Project-root on sys.path ──────────────────────────────────────────────
from pathlib import Path  # noqa: E402

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))