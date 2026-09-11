"""
tests/test_service_entrypoint.py
================================
The `api` image must START, and it must start on the port it was told to.

Two regressions this file exists to catch. Both reached four repos' Dockerfiles before
anything noticed, because building an image never invokes its ENTRYPOINT and the
in-process API tests (`tests/test_api_contract.py`) need no server at all.

1. **The entrypoint does not import.** atrium-project#58 originally prescribed
   ``ENTRYPOINT ["python", "service/api.py"]``. A *script* launch sets ``sys.path[0]`` to
   the script's own directory (``/app/service``) and leaves ``__package__`` empty, so
   ``from .atrium_service import ...`` raises "attempted relative import with no known
   parent package" and a repo-root import such as ``from main import ...`` raises
   ``ModuleNotFoundError`` — before the ``app`` object is ever built. The container then
   exits immediately. ``python -m service.api`` is the fix: it puts ``sys.path[0]`` at the
   repo root *and* gives the module its package context, which is byte for byte the
   environment the old ``uvicorn service.api:app`` entrypoint already ran in.

   (alto-postprocess is the exception that proves this: its ``python service/text_api.py``
   entrypoint works only because ``text_api.py`` carries a 25-line ``sys.path`` bootstrap
   above its first first-party import. The entrypoint line does not carry that bootstrap,
   so copying the line without it copies the half that does nothing.)

2. **The entrypoint hardcodes the port again.** The exec-form
   ``["uvicorn", "service.api:app", "--host", "0.0.0.0", "--port", "8000", ...]`` cannot
   expand ``$PORT`` — exec form runs no shell — while ``service/healthcheck.py`` reads it.
   Setting ``PORT`` therefore moved the health PROBE and not the listener, and the
   container reported **unhealthy forever** rather than merely ignoring the knob. The
   container-level proof is the hub's ``docker-build-smoke`` probe starting an image at a
   non-default port; this file is its fast-lane counterpart, catching the same regression
   in milliseconds with no Docker and no network.

Test 1 needs the service's third-party dependencies and skips cleanly without them — but
only after distinguishing a missing third-party dep from the two structural failures
above, which are never a skip. Tests 2 and 3 are static and always run.
"""

from __future__ import annotations

import ast
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
API_MODULE = "service.api"
API_SOURCE = REPO_ROOT / "service" / "api.py"

#: Environment variables the __main__ block must read for the k8s manifest's `env:` block
#: to mean anything. PORT is the one with teeth (healthcheck.py probes it); HOST and
#: GRACEFUL_SHUTDOWN_S are in the same contract (atrium-project#58, #55).
REQUIRED_ENV_READS = ("PORT", "HOST", "GRACEFUL_SHUTDOWN_S")


def _api_entrypoint() -> list:
    """argv of the `api` stage's ENTRYPOINT, parsed out of the Dockerfile."""
    stage, found = None, None
    for raw in (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        match = re.match(r"^FROM\s+\S+\s+AS\s+(\S+)", line, re.IGNORECASE)
        if match:
            stage = match.group(1)
            continue
        if stage != "api" or not line.upper().startswith("ENTRYPOINT"):
            continue
        payload = line[len("ENTRYPOINT") :].strip()
        try:
            found = json.loads(payload) if payload.startswith("[") else shlex.split(payload)
        except ValueError:
            found = None
    return found or []


def _main_block_source() -> str:
    """Source of service/api.py's ``if __name__ == "__main__":`` block, or ""."""
    source = API_SOURCE.read_text(encoding="utf-8")
    for node in ast.parse(source).body:
        if not isinstance(node, ast.If):
            continue
        if "__main__" in ast.dump(node.test):
            return ast.get_source_segment(source, node) or ""
    return ""


def _is_structural(stderr: str) -> bool:
    """True when the import failed for the reason this test exists to catch.

    A missing third-party wheel in the fast lane is a skip. A relative import with no
    package context, or a missing FIRST-PARTY module, is the defect itself — those are
    what a script launch breaks, and they must never be skipped past.
    """
    if "attempted relative import with no known parent package" in stderr:
        return True
    for name in re.findall(r"No module named '([^']+)'", stderr):
        head = name.split(".")[0]
        if (REPO_ROOT / f"{head}.py").exists() or (REPO_ROOT / head).is_dir():
            return True
    return False


@pytest.mark.slow
def test_api_module_imports_the_way_the_entrypoint_launches_it(tmp_path):
    """Import service.api with the sys.path and package context `-m` really gives it."""
    # run_name is deliberately NOT "__main__": the module body must execute (that is what
    # is under test) while its own __main__ block stays inert, so this never starts uvicorn.
    code = f"import runpy;runpy.run_module({API_MODULE!r}, run_name='_entrypoint_probe')"
    env = dict(os.environ)
    # A stray PYTHONPATH pointing at the repo root would mask the very failure this test
    # exists to catch.
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        [sys.executable, "-c", code],
        # The repo root, because that is WORKDIR /app in the image and `-m` takes
        # sys.path[0] from the working directory.
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0 and not _is_structural(result.stderr):
        pytest.skip(
            "service dependencies are not installed in this environment "
            f"(last line: {result.stderr.strip().splitlines()[-1] if result.stderr.strip() else '?'})"
        )
    assert result.returncode == 0, (
        f"`python -m {API_MODULE}` does not import — this is exactly how the Dockerfile "
        f"`api` stage starts the service, so the container exits at startup.\n\n"
        f"stderr:\n{result.stderr}"
    )


def test_api_entrypoint_does_not_hardcode_the_port():
    """The `api` stage must leave the port to $PORT (atrium-project#58).

    Fast-lane counterpart of the hub's non-default-port container probe. An exec-form
    ENTRYPOINT array runs no shell, so a `--port` baked in here cannot be overridden by
    the manifest's `env: PORT` — and because service/healthcheck.py DOES read PORT, the
    result is a container that reports unhealthy forever rather than one that merely
    ignores the setting.
    """
    argv = _api_entrypoint()
    assert argv, "the Dockerfile declares no ENTRYPOINT for the `api` stage"

    assert "--port" not in argv, (
        f"the `api` stage hardcodes the port in its ENTRYPOINT ({argv}). Exec form runs "
        f"no shell, so $PORT cannot expand — while service/healthcheck.py:49 reads it, "
        f"making any non-default PORT a permanently-unhealthy container. Read the port "
        f"in service/api.py's __main__ block instead (atrium-project#58)."
    )
    assert argv[:2] == ["python", "-m"], (
        f"expected the `api` stage to launch the module ({['python', '-m', API_MODULE]}), "
        f"got {argv}. A bare `python service/api.py` sets sys.path[0] to service/ with no "
        f"package context and fails at import; see this module's docstring."
    )
    assert argv[2] == API_MODULE, f"expected `-m {API_MODULE}`, got `-m {argv[2]}`"


def test_main_block_reads_the_deployment_environment():
    """PORT/HOST/GRACEFUL_SHUTDOWN_S must come from the environment, not be literals.

    The manifest the partner deploys from (atrium-project
    docs/templates/k8s/atrium-service.deployment.yaml) declares `env: PORT`. If this block
    stops reading it, that declaration silently means nothing again — which is the whole
    of atrium-project#58.
    """
    block = _main_block_source()
    assert block, (
        'service/api.py has no `if __name__ == "__main__":` block, so '
        "`python -m service.api` — the Dockerfile `api` stage ENTRYPOINT — starts nothing"
    )
    missing = [name for name in REQUIRED_ENV_READS if f'"{name}"' not in block]
    assert not missing, (
        f"service/api.py's __main__ block does not read {missing} from the environment. "
        f"The k8s manifest declares these; a hardcoded value makes that declaration inert."
    )
    assert "uvicorn.run" in block, "the __main__ block never calls uvicorn.run"
