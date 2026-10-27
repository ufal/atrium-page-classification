"""
tests/test_release_bundle.py — the release bundle is self-contained, checked on every push.

The release workflow packs the service into a zip (its "Package … artifact" step) and only then,
at tag time, checks that every first-party module the bundled files import is in the bundle
("Verify bundle is self-contained"). A module added to the code without its `cp` line therefore
failed the release after the tag was cut: page-classification v1.10.0-v1.10.3-beta (page_inputs.py)
and ocr-postprocess v1.13.0-beta (record_scoring.py) published their images but no GitHub release
(atrium-project#72, #81).

This test reads the packaging step's `cp` commands from release.yml and applies the same rule to
the files they name, so the omission fails on the push that introduces it. It is static, like the
workflow's check: nothing is imported, and torch is not needed.
"""

from __future__ import annotations

import ast
import shlex
from pathlib import Path
from typing import Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / ".github" / "workflows" / "release.yml"


def _packaging_commands() -> List[List[str]]:
    """The shell commands of the step that builds the bundle, one token list per command.

    The step is the first one whose name starts with "Package"; comment lines are dropped and
    backslash continuations joined, as the shell reads them.
    """
    lines = RELEASE.read_text(encoding="utf-8").splitlines()
    start = next(
        (i for i, line in enumerate(lines) if line.strip().lower().startswith("- name: package")),
        None,
    )
    assert start is not None, f"{RELEASE.name} has no step named 'Package …'"
    body: List[str] = []
    for line in lines[start + 1 :]:
        if line.strip().startswith("- name:"):
            break
        body.append(line)
    script, pending = [], ""
    for line in body:
        text = line.strip()
        if not text or text.startswith("#") or text in ("run: |", "run: >"):
            continue
        if text.endswith("\\"):
            pending += text[:-1] + " "
            continue
        script.append(pending + text)
        pending = ""
    commands = []
    for line in script:
        tokens = []
        for token in shlex.split(line):
            if token in ("||", "&&", ";") or token.startswith("2>") or token.startswith(">"):
                break
            tokens.append(token)
        if tokens:
            commands.append(tokens)
    return commands


def _bundle_sources() -> Tuple[str, List[str]]:
    """(the bundle directory, every source path its `cp` commands copy into it)."""
    commands = _packaging_commands()
    bundle = next((cmd[-1] for cmd in commands if cmd[:2] == ["mkdir", "-p"]), None)
    assert bundle, "the packaging step does not create its bundle directory with `mkdir -p`"
    bundle = bundle.rstrip("/")
    sources: List[str] = []
    for cmd in commands:
        if cmd[0] != "cp":
            continue
        args = [token for token in cmd[1:] if not token.startswith("-")]
        if args and args[-1].rstrip("/") == bundle:
            sources.extend(args[:-1])
    assert sources, f"no `cp … {bundle}/` command found in the packaging step"
    return bundle, sources


def _shipped_files(sources: List[str]) -> List[Path]:
    files: List[Path] = []
    for source in sources:
        path = ROOT / source
        if path.is_dir():
            files.extend(p for p in path.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
        elif path.is_file():
            files.append(path)
    return files


def _imported(path: Path) -> Set[str]:
    """Top-level module names a file imports, absolute imports only (the workflow's rule)."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return set()
    names: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
        elif isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
    return names


def test_every_bundle_source_exists():
    """A `cp` source that is not in the repository makes the packaging step fail, or (behind
    `|| true`) silently ships less than the list says."""
    _, sources = _bundle_sources()
    missing = [source for source in sources if not (ROOT / source).exists()]
    assert not missing, f"release.yml copies files that do not exist: {missing}"


def test_bundle_is_self_contained():
    """Every first-party module a bundled file imports is in the bundle, as the release checks."""
    _, sources = _bundle_sources()
    first_party = {p.stem for p in ROOT.glob("*.py")}
    files = [p for p in _shipped_files(sources) if p.suffix == ".py"]
    shipped = {p.stem for p in files}
    needed: Dict[str, Set[str]] = {}
    for path in files:
        for module in _imported(path) & first_party:
            needed.setdefault(module, set()).add(str(path.relative_to(ROOT)))
    missing = {module: sorted(users) for module, users in needed.items() if module not in shipped}
    assert not missing, (
        "the release bundle would not be self-contained; add these modules to the `cp` list of "
        f"release.yml's packaging step: {missing}"
    )
