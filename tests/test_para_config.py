"""
tests/test_para_config.py
=========================
Repo-specific checks on setup/para_config.txt, the component list the paradata
licence resolver reads.

Kept out of tests/test_paradata.py on purpose: that file is identical across the
ATRIUM repositories (only PROGRAM_NAME differs), and these assertions are about
this repository's components only.
"""

from pathlib import Path

from atrium_paradata import _load_para_config

SETUP_DIR = Path(__file__).resolve().parent.parent / "setup"


def _components() -> dict:
    return {c["name"]: c for c in _load_para_config(str(SETUP_DIR))["components"]}


def test_ultralytics_is_declared_as_a_conditional_agpl_component():
    """run.py logs `ultralytics` on every --yolo run; without an entry it resolved to UNKNOWN."""
    comp = _components()["ultralytics"]
    assert comp["license"] == "AGPL-3.0"
    assert comp["loaded"] == "conditional"


def test_deepdoctection_is_not_declared():
    """No code in this repository imports DeepDoctection, so it is not a component of any run."""
    assert "deepdoctection" not in _components()


def test_lindat_dataset_is_cc_by_nc():
    assert _components()["lindat_dataset"]["license"] == "CC BY-NC 4.0"


def test_every_logged_component_is_declared():
    """Each name run.py passes to log_component() must have a para_config.txt entry."""
    import re

    run_py = (SETUP_DIR.parent / "run.py").read_text(encoding="utf-8")
    logged = set(re.findall(r'log_component\(\s*"([^"]+)"', run_py))
    assert logged, "expected run.py to log at least one component"
    assert logged <= set(_components())


def test_the_pdf_engine_is_declared_as_a_conditional_permissive_component():
    """atrium-project#72 D.1: /predict_document renders PDFs with pypdfium2, so it is a component
    of every record made from a PDF; PyMuPDF (AGPL-3.0) rendered them before, declared nowhere."""
    comp = _components()["pypdfium2"]
    assert comp["license"] == "Apache-2.0"
    assert comp["loaded"] == "conditional"
    assert "pymupdf" not in {name.lower() for name in _components()}


def test_no_requirements_file_installs_pymupdf():
    """The #6 D / #72 D.1 *done when*: the service no longer installs the undeclared AGPL engine.
    Every requirements file in the repository counts, since the images install two of them and
    the lanes the rest."""
    import re

    offenders = []
    for path in sorted(SETUP_DIR.parent.rglob("requirements*.txt")):
        if {".git", ".venv", "venv", "env", "site-packages"} & set(path.parts):
            continue  # clones and virtual environments are not this repository's declarations
        for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = raw.split("#", 1)[0].strip()
            name = re.split(r"[<>=!~;\[ ]", line, maxsplit=1)[0].lower().replace("_", "-")
            if name in {"pymupdf", "fitz", "pymupdfb", "pymupdf-layout"}:
                offenders.append(f"{path.relative_to(SETUP_DIR.parent)}:{number}")
    assert not offenders, f"PyMuPDF (AGPL-3.0) is installed again: {offenders}"
