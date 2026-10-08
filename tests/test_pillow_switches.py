"""
tests/test_pillow_switches.py
=============================
Importing `utils` must not change how Pillow decodes (atrium-page-classification, found in the
atrium-project#53 audit).

`utils.py` used to set `ImageFile.LOAD_TRUNCATED_IMAGES = True` and raise `Image.MAX_IMAGE_PIXELS`
at import. The service imports `utils` lazily, on its first `/predict_image`
(`service/document_json.doc_id_for_image`), so the first request of a process refused a truncated
upload with a 422 (`service/api.py::_decode_image`) and every later one classified its grey fill.
The switches are now `utils.tolerate_scan_quirks()`, which the batch tools call.

Pillow's switches are process-wide, so each case runs in a fresh interpreter. No torch is needed:
`utils` pulls matplotlib and scikit-learn, which is why the service imports it lazily.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("PIL")
pytest.importorskip("matplotlib")
pytest.importorskip("sklearn")

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run(body: str) -> dict:
    """Run `body` in a fresh interpreter with the repo importable; it prints one JSON object."""
    code = f"import sys; sys.path.insert(0, {str(REPO_ROOT)!r})\n{body}"
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=240, cwd=REPO_ROOT)
    assert done.returncode == 0, done.stderr[-2000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


_SWITCHES = """
from PIL import Image, ImageFile
def switches():
    return {"truncated": ImageFile.LOAD_TRUNCATED_IMAGES, "pixels": Image.MAX_IMAGE_PIXELS}
before = switches()
"""

#: A JPEG with its tail cut off: it opens (the header is intact) and fails when decoded.
_TRUNCATED = """
import io
buf = io.BytesIO()
Image.effect_noise((200, 200), 80).convert("RGB").save(buf, "JPEG")
cut = buf.getvalue()[: len(buf.getvalue()) // 2]
def decodes():
    try:
        Image.open(io.BytesIO(cut)).convert("RGB")
        return True
    except (OSError, SyntaxError):
        return False
"""


def test_importing_utils_leaves_pillows_switches_alone():
    out = _run(_SWITCHES + "import utils\nprint(__import__('json').dumps({'before': before, 'after': switches()}))")
    assert out["before"] == out["after"]
    assert out["after"]["truncated"] is False


def test_the_services_lazy_doc_id_derivation_leaves_them_alone():
    out = _run(
        _SWITCHES
        + "from service.document_json import doc_id_for_image\n"
        + "doc_id_for_image('CTX01_0007.png')\n"
        + "print(__import__('json').dumps({'before': before, 'after': switches()}))"
    )
    assert out["before"] == out["after"]


def test_a_truncated_image_is_still_refused_once_utils_has_been_imported():
    out = _run(
        _SWITCHES
        + _TRUNCATED
        + "import utils\n"
        + "utils.doc_id_and_page('CTX01_0007.png')\n"
        + "print(__import__('json').dumps({'decodes': decodes()}))"
    )
    assert out == {"decodes": False}


def test_tolerate_scan_quirks_is_what_the_batch_tools_call():
    out = _run(
        _SWITCHES
        + _TRUNCATED
        + "import utils\n"
        + "refused = not decodes()\n"
        + "utils.tolerate_scan_quirks()\n"
        + "print(__import__('json').dumps({'refused_before': refused, 'decodes_after': decodes(), 'after': switches()}))"
    )
    assert out["refused_before"] is True and out["decodes_after"] is True
    assert out["after"] == {"truncated": True, "pixels": 4221790634}


def test_the_batch_entry_points_make_the_call():
    """run.py and parallel_best.py import utils for the batch path, and must opt in."""
    assert "tolerate_scan_quirks()" in (REPO_ROOT / "run.py").read_text(encoding="utf-8")
    assert (REPO_ROOT / "parallel_best.py").read_text(encoding="utf-8").count("tolerate_scan_quirks()") == 2
