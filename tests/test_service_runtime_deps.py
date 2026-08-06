"""
tests/test_service_runtime_deps.py
==================================
The gate for atrium-project#10 (G3): `service/requirements.txt` carried six pytest/contract
deps and **no ASGI server at all**, while `docker-compose.yml`'s `api` service ran
`uvicorn service.api:app` and `setup/setup_api_service.sh` told the user to do the same.
`grep -rn uvicorn --include="*.txt" .` matched nothing repo-wide, so
`docker compose --profile api up api` failed at container start for 12 days.

**The gate matters more than the fix here**, because the reason it survived is structural, not
careless:

* `tests/test_service_api.py` and `tests/test_api_contract.py` drive the app **in-process**
  through `TestClient`. That needs no server, so a missing `uvicorn` is invisible to them.
* the hub's `docker-build-smoke` job only **builds** the image; nothing runs it.

So both existing gates are blind to this whole class by construction. This file closes it with
a dependency-only assertion that runs in the fast lane, in seconds, with no Docker:

1. every console entrypoint the deployment actually invokes — **parsed out of
   `docker-compose*.yml` and `setup/setup_api_service.sh`, not hardcoded** — is declared in a
   requirements file the image installs;
2. every third-party module anything under `service/` imports is declared too. That half found
   `fitz`/PyMuPDF (used by `/predict_document`, declared nowhere) and `requests` (used by the
   client script the README documents) while G3 was being fixed — the same defect, one import
   further in.

A real container smoke test (`docker run` the api target, wait for the port, curl `/health`)
is still the belt to this braces and belongs in CI; see the plan's Phase 3.2 note.
"""

import ast
import configparser
import re
import shlex
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).parent.parent

#: Every requirements file the Docker image installs (Dockerfile:40-42) and
#: setup/setup_api_service.sh installs. The union is what "declared" means: it does not matter
#: WHICH of them carries a dep, only that something the image builds from does.
REQUIREMENTS_FILES = [
    REPO_ROOT / "setup" / "requirements.txt",
    REPO_ROOT / "setup" / "requirements-test.txt",
    REPO_ROOT / "service" / "requirements.txt",
]

#: Interpreters and shell built-ins are provided by the base image, not by pip, so an
#: entrypoint starting with one of these is making no dependency claim.
_NOT_PIP_PROVIDED = {"sh", "bash", "python", "python3", "exec", "echo", "cd", "source", "pip", "pip3"}

#: Import name → distribution name, for the handful where they differ. Deliberately tiny and
#: explicit rather than derived from the installed environment: the point is to check the
#: DECLARATIONS, which must hold even for a dep that is not installed in this venv (PyMuPDF is
#: exactly that case in the fast lane).
_IMPORT_TO_DISTRIBUTION = {
    "PIL": "pillow",
    "fitz": "pymupdf",
    "sklearn": "scikit-learn",
    "yaml": "pyyaml",
    "cv2": "opencv-python",
}


def _normalise(name: str) -> str:
    """PEP 503 name normalisation, so `PyMuPDF`, `pymupdf` and `py_mupdf` compare equal."""
    return re.sub(r"[-_.]+", "-", name).strip().lower()


def _declared_distributions() -> set:
    declared = set()
    for path in REQUIREMENTS_FILES:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line or line.startswith("-"):  # blank, comment, or an -r/-e directive
                continue
            # Strip extras and any version specifier: `uvicorn[standard]>=0.40.0` → `uvicorn`.
            declared.add(_normalise(re.split(r"[<>=!~;\[]", line, maxsplit=1)[0]))
    return declared


def _compose_entrypoints() -> dict:
    """{(compose file, service, key): argv} for every entrypoint/command in the compose files."""
    found = {}
    for compose_path in sorted(REPO_ROOT.glob("docker-compose*.yml")):
        data = yaml.safe_load(compose_path.read_text(encoding="utf-8")) or {}
        for service, spec in (data.get("services") or {}).items():
            if not isinstance(spec, dict):
                continue
            for key in ("entrypoint", "command"):
                value = spec.get(key)
                if not value:
                    continue
                argv = value if isinstance(value, list) else shlex.split(str(value))
                if argv:
                    found[(compose_path.name, service, key)] = argv
    return found


def _service_imports() -> dict:
    """{module name: [files that import it]} for every import under service/."""
    imports: dict = {}
    for path in sorted((REPO_ROOT / "service").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                # level > 0 is a relative import, i.e. a sibling in this same package.
                names = [node.module] if node.level == 0 and node.module else []
            else:
                continue
            for name in names:
                imports.setdefault(name.split(".")[0], []).append(path.name)
    return imports


def _is_first_party(module: str) -> bool:
    """A module that ships in this repo, so no requirements entry can or should cover it."""
    for parent in (REPO_ROOT, REPO_ROOT / "service"):
        if (parent / f"{module}.py").exists() or (parent / module / "__init__.py").exists():
            return True
    return False


class TestDeploymentEntrypointsAreInstallable:
    def test_compose_entrypoints_are_declared(self):
        """G3 itself. Reads the compose files rather than naming `uvicorn`, so switching to
        hypercorn/granian without declaring it fails here too."""
        declared = _declared_distributions()
        entrypoints = _compose_entrypoints()
        assert entrypoints, "no entrypoint/command found in any docker-compose*.yml — parser broke"

        missing = {
            ":".join(where): argv[0]
            for where, argv in entrypoints.items()
            if argv[0] not in _NOT_PIP_PROVIDED and _normalise(argv[0]) not in declared
        }
        assert not missing, (
            f"console entrypoint(s) invoked by the deployment but declared in none of "
            f"{[p.name for p in REQUIREMENTS_FILES]}: {missing}. The container will exit "
            f"immediately with 'executable file not found'."
        )

    def test_the_api_service_really_does_invoke_a_server(self):
        """Guards the guard: if the `api` service ever loses its entrypoint, the assertion
        above would pass by having nothing left to check."""
        entrypoints = _compose_entrypoints()
        api_entrypoints = [
            argv for (_file, service, key), argv in entrypoints.items() if service == "api" and key == "entrypoint"
        ]
        assert api_entrypoints, "docker-compose.yml declares no entrypoint for the `api` service"
        assert any(argv[0] not in _NOT_PIP_PROVIDED for argv in api_entrypoints), (
            "the `api` service no longer starts a pip-installed server, so "
            "test_compose_entrypoints_are_declared has nothing left to assert"
        )

    def test_setup_script_start_command_is_declared(self):
        """setup/setup_api_service.sh's closing instructions are the other documented start
        path, and they were pointing at the same absent binary."""
        declared = _declared_distributions()
        script = (REPO_ROOT / "setup" / "setup_api_service.sh").read_text(encoding="utf-8")

        # The script echoes the command for the user to copy; pick the tokens that look like a
        # console script invocation rather than trying to interpret shell.
        invoked = {
            match.group(1)
            for match in re.finditer(r"^\s*echo\s+\"?\s*([a-z0-9_-]+)\s+service\.api:app", script, re.MULTILINE)
        }
        assert invoked, "setup_api_service.sh no longer documents how to start the server"
        assert not [name for name in invoked if _normalise(name) not in declared]


class TestServiceImportsAreDeclared:
    def test_every_third_party_service_import_is_declared(self):
        """The generalisation of G3: a serving dependency that is imported but undeclared is
        the same bug whether it is the server itself or PyMuPDF."""
        declared = _declared_distributions()
        undeclared = {}
        for module, files in _service_imports().items():
            if module in sys.stdlib_module_names or _is_first_party(module):
                continue
            distribution = _IMPORT_TO_DISTRIBUTION.get(module, module)
            if _normalise(distribution) not in declared:
                undeclared[module] = sorted(set(files))
        assert not undeclared, (
            f"service module(s) import packages declared in none of "
            f"{[p.name for p in REQUIREMENTS_FILES]}: {undeclared}"
        )

    def test_the_import_scanner_sees_the_real_service_modules(self):
        """Guards the guard again: an ast/glob regression that returned nothing would make the
        assertion above vacuously true."""
        imports = _service_imports()
        assert "fastapi" in imports
        assert "fitz" in imports, "the /predict_document PDF path disappeared, or the scan did"


class TestVersionPinsStayConsistent:
    def test_service_requirements_do_not_contradict_the_base_install(self):
        """The image installs setup/requirements.txt and service/requirements.txt into one
        environment, so a dep named in both must not carry incompatible floors — pip would
        resolve one of them away silently. Pillow is in both by design (the service imports it
        directly); this pins that they agree."""
        floors = {}
        for path in (REPO_ROOT / "setup" / "requirements.txt", REPO_ROOT / "service" / "requirements.txt"):
            for raw in path.read_text(encoding="utf-8").splitlines():
                line = raw.split("#", 1)[0].strip()
                if not line or line.startswith("-"):
                    continue
                name = _normalise(re.split(r"[<>=!~;\[]", line, maxsplit=1)[0])
                floors.setdefault(name, set()).add(line.replace(" ", ""))

        conflicting = {name: sorted(specs) for name, specs in floors.items() if len(specs) > 1}
        assert not conflicting, f"same dependency declared with different specifiers: {conflicting}"

    def test_numpy_cap_and_dependabot_ignore_are_still_both_in_place(self):
        """atrium-project#10 (G2): the `numpy >= 2.5.0` dependabot ignore on the `/setup`
        manifest was commented out five minutes after being added, and PR #38 reproduced the
        break within the hour. The `<2.5` cap in setup/requirements.txt and that ignore are two
        halves of one policy — lifting either alone puts the bump back next Monday, so assert
        both, here, where the fast lane will see it.

        Lift this test together with both halves when the images move to Python 3.12.
        """
        requirements = (REPO_ROOT / "setup" / "requirements.txt").read_text(encoding="utf-8")
        assert re.search(r"^numpy>=[\d.]+,<2\.5\s*$", requirements, re.MULTILINE), (
            "setup/requirements.txt lost its numpy<2.5 cap; numpy 2.5.0 declares "
            "Requires-Python >=3.12 and every image here is python:3.11-slim"
        )

        dependabot = yaml.safe_load((REPO_ROOT / ".github" / "dependabot.yml").read_text(encoding="utf-8"))
        guarded = set()
        for update in dependabot.get("updates", []):
            for ignore in update.get("ignore", []) or []:
                if ignore.get("dependency-name") == "numpy" and ">= 2.5.0" in (ignore.get("versions") or []):
                    guarded.add(update.get("directory"))
        assert guarded == {"/setup", "/service"}, (
            f"the numpy>=2.5.0 dependabot ignore is live only for {sorted(guarded)}; it must "
            f"cover every pip manifest, or dependabot reopens the break for the uncovered one"
        )

    def test_python_version_still_justifies_the_numpy_cap(self):
        """The cap's stated lift condition is "when the images move to 3.12". If the Dockerfile
        has moved, this fails and sends the reader to the cap rather than leaving a stale pin
        nobody dares touch."""
        dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
        assert "FROM python:3.11-slim" in dockerfile, (
            "the base image moved off 3.11 — re-read the numpy<2.5 cap in "
            "setup/requirements.txt and the matching dependabot ignore; both can now be lifted"
        )


def test_para_config_version_is_readable_by_the_service():
    """`/info` and `app.version` come from setup/para_config.txt via read_tool_version(). A
    service that cannot read its own version answers `/info` with 0.0.0, which
    security.reusable.yml's version check would then compare against the release tag."""
    config = configparser.ConfigParser()
    config.read(REPO_ROOT / "setup" / "para_config.txt", encoding="utf-8")
    assert config.get("tool", "version", fallback=None)
