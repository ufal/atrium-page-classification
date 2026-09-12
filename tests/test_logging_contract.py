"""
tests/test_logging_contract.py
===============================
Static regression guard for issue atrium-project#61 (logs as an event stream).

Byte-identical across the hub (docs/templates/shared/) and all five tool repos,
enforced by para-drift.reusable.yml exactly like service/atrium_service.py and
service/healthcheck.py. Edit the hub copy, never a vendored one;
scripts/revendor_shared.sh does the copy.

Deliberately STATIC (ast/re over source, no imports of the modules under test).
The obvious alternative — import service.inference and assert the root logger
gained no handler — would have SKIPPED in exactly the repo that regressed:
page-classification's service/inference.py needs torch, tests/test_service_
entrypoint.py's equivalent runtime check is `@pytest.mark.slow`, and the fast
lane (`pytest -m "not slow"`, run on every PR by the hub's docker-tool.
reusable.yml) deliberately has no torch. A runtime assertion here would inherit
that skip and silently stop guarding the one repo #61 was filed against. Static
analysis has no such gap: it reads every file's source whether or not its
third-party imports are installed.

Four checks, one per #61 acceptance criterion / found defect:

1. No `logging.basicConfig(` outside the entrypoint's own `if __name__ ==
   "__main__":` block — page-classification's defect (D1): an import-time call
   in a LIBRARY module silently made the real entrypoint's later basicConfig()
   a no-op (basicConfig is documented to do nothing once the root logger
   already has handlers), so LOG_LEVEL was read and discarded.
2. No `print(` in service library modules — translator's and alto's defect
   (D2): lifecycle and warning records went to print() instead of a logger, so
   they could not be leveled, filtered, or silenced by LOG_LEVEL. Three files
   are allowlisted because their stdout genuinely IS their output contract,
   not service logging: `healthcheck.py` (the Docker HEALTHCHECK probe body,
   byte-identical across all five and covered by its own para-drift check),
   `api_client.py` (page-classification's CLI client), and `test_api.py`
   (nlp-enrich's CLI smoke test). Batch-pipeline entrypoints at the repo root
   (run.py, run_pipeline.py, main.py, llm_run.py, entrypoint.py) are out of
   scope by construction: this file only looks under service/, because those
   are one-shot processes whose stdout already is the event stream — the
   distinction issue #61 draws between "logs" and "process output".
3. The entrypoint's `__main__` block reads LOG_LEVEL from the environment —
   the config half of #61, mirroring the existing
   test_main_block_reads_the_deployment_environment idiom.
4. At least one `logger.<level>(...)` call exists under service/ — the "not a
   placebo" check. Checks 1 and 3 alone would have passed for both nlp-enrich
   and translator before their fixes: LOG_LEVEL was wired correctly and
   controlled zero call sites. This is the check that would have caught them.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import List, Optional

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SERVICE_DIR = REPO_ROOT / "service"

# Every OTHER canonical test under docs/templates/shared/ is self-contained —
# test_para_licenses.py imports para_licenses.py from beside it, and so on. This
# one is not: it inspects the CONSUMING repo's service/ tree, which is exactly
# what the hub itself does not have. hub-self-check.yml runs `pytest .` with
# working-directory: docs/templates/shared, where REPO_ROOT above resolves to
# docs/templates/ — no service/ directory — so _service_py_files() below would
# return [], and checks 1-2 would pass having examined NOTHING while checks 3-4
# correctly fail on an empty set. An honest skip beats a vacuous pass. This
# never fires in a tool repo, where service/ always exists.
if not SERVICE_DIR.is_dir():
    pytest.skip(
        "no service/ directory here — this file inspects a TOOL REPO's service "
        "layer, and the hub's docs/templates/shared/ is not one (issue #61)",
        allow_module_level=True,
    )

#: Files whose stdout is their own output contract, not service logging (see
#: check 2's docstring paragraph above). Matched by filename, not path, so this
#: is identical text in every repo regardless of what else lives under service/.
PRINT_ALLOWLIST = {"healthcheck.py", "api_client.py", "test_api.py"}

_LOGGER_LEVELS = {"debug", "info", "warning", "error", "critical", "exception"}


def _service_py_files() -> List[Path]:
    if not SERVICE_DIR.is_dir():
        return []
    return sorted(SERVICE_DIR.glob("*.py"))


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _find_main_if(tree: ast.Module) -> Optional[ast.If]:
    """The top-level ``if __name__ == "__main__":`` node, or None.

    Same detection idiom as tests/test_service_entrypoint.py's
    _main_block_source(): a substring match on ast.dump(node.test) rather than
    a literal comparison, so this does not care whether the source wrote
    `"__main__" == __name__` or the other way around.
    """
    for node in tree.body:
        if isinstance(node, ast.If) and "__main__" in ast.dump(node.test):
            return node
    return None


def _node_ids(node: Optional[ast.AST]) -> set:
    if node is None:
        return set()
    return {id(n) for n in ast.walk(node)}


def _is_call_to(node: ast.AST, name: str) -> bool:
    """True if *node* is a Call to a bare name or an attribute both named *name*
    (covers `basicConfig(...)` and `logging.basicConfig(...)` alike)."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Name):
        return func.id == name
    if isinstance(func, ast.Attribute):
        return func.attr == name
    return False


def _is_uvicorn_run(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return False
    value = node.func.value
    return node.func.attr == "run" and isinstance(value, ast.Name) and value.id == "uvicorn"


def _find_entrypoint() -> Optional[Path]:
    """The service/*.py file whose __main__ block calls uvicorn.run(...).

    Distinguishes the real server entrypoint (service/api.py in four repos,
    service/text_api.py in alto-postprocess) from CLI scripts that also carry
    a __main__ block (api_client.py, test_api.py, healthcheck.py) and from
    page-classification's service/inference.py, which mentions "__main__" only
    in a comment and has no such block at all.
    """
    for path in _service_py_files():
        main_if = _find_main_if(_parse(path))
        if main_if is not None and any(_is_uvicorn_run(n) for n in ast.walk(main_if)):
            return path
    return None


def test_basicconfig_only_in_the_main_block():
    """(issue #61, D1) logging.basicConfig() must never run at import time.

    A library module configuring the ROOT logger as a side effect of being
    imported silently overrides whatever the real entrypoint chooses later —
    and because basicConfig() is a documented no-op once the root logger
    already has handlers, the entrypoint's own call becomes dead code that
    still LOOKS like it configures LOG_LEVEL. That is exactly what
    page-classification/service/inference.py did until #61.
    """
    violations = []
    for path in _service_py_files():
        tree = _parse(path)
        main_if = _find_main_if(tree)
        allowed_ids = _node_ids(main_if)
        for node in ast.walk(tree):
            if _is_call_to(node, "basicConfig") and id(node) not in allowed_ids:
                violations.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
    assert not violations, (
        "logging.basicConfig() called outside the entrypoint's __main__ block "
        f"(issue #61): {violations}. Library modules must only "
        "logging.getLogger(__name__); only the real entrypoint's __main__ "
        "block may configure the root logger."
    )


def test_no_print_in_service_library_modules():
    """(issue #61, D2) service/ library modules must log, not print.

    print() cannot be leveled, filtered by LOG_LEVEL, or routed anywhere but
    stdout — the exact gap #61 closed in translator's and alto-postprocess's
    lifecycle/warning records. See PRINT_ALLOWLIST's docstring paragraph above
    for the three files this deliberately excludes and why; everything else
    under service/ is service logging and must use `logger`.
    """
    violations = []
    for path in _service_py_files():
        if path.name in PRINT_ALLOWLIST:
            continue
        tree = _parse(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == "print":
                    violations.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
    assert not violations, (
        f"print() found in a service library module (issue #61): {violations}. "
        "Use `logger` (see PRINT_ALLOWLIST for the deliberate exceptions: "
        "healthcheck.py, api_client.py, test_api.py — CLI scripts whose stdout "
        "is their own output contract, not service logging)."
    )


def test_entrypoint_reads_log_level():
    """(issue #61) The one place allowed to configure logging must read LOG_LEVEL.

    Mirrors tests/test_service_entrypoint.py's
    test_main_block_reads_the_deployment_environment for PORT/HOST — same
    reasoning, applied to the variable this issue makes real.
    """
    entrypoint = _find_entrypoint()
    assert entrypoint is not None, (
        "no service/*.py file has a __main__ block calling uvicorn.run(...); "
        "cannot verify it reads LOG_LEVEL"
    )
    tree = _parse(entrypoint)
    main_if = _find_main_if(tree)
    source = ast.get_source_segment(entrypoint.read_text(encoding="utf-8"), main_if) or ""
    assert '"LOG_LEVEL"' in source, (
        f"{entrypoint.relative_to(REPO_ROOT)}'s __main__ block does not read LOG_LEVEL "
        "(issue #61) — the k8s manifest and every README document this variable; a "
        "hardcoded level makes that documentation false."
    )


def test_log_level_controls_at_least_one_call_site():
    """(issue #61) LOG_LEVEL must not be a placebo.

    Checks 1 and 3 alone pass even when the entrypoint's root-logger config is
    perfectly correct and zero code under service/ ever calls it — exactly the
    state translator's and nlp-enrich's service/*.py were in before #61: a
    correctly-wired LOG_LEVEL that controlled nothing, because every lifecycle
    record was still print(). This check requires at least one real emitter.
    """
    count = 0
    for path in _service_py_files():
        tree = _parse(path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "logger"
                and func.attr in _LOGGER_LEVELS
            ):
                count += 1
    assert count > 0, (
        "no logger.<level>(...) call found anywhere under service/ (issue #61) — "
        "LOG_LEVEL can be perfectly configured and still control nothing if no "
        "code ever logs through it."
    )
