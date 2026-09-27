"""tests/test_limits_contract.py — every limit is a documented setting (atrium-project#53).

David's *done when* for #53 factor III: "every limit in the five services is a documented
environment setting, and ``/info`` reports its current value." This file checks the first
half, and the part of the second half that does not need the service's heavy dependencies,
identically in all five repos (canonical: hub ``docs/templates/shared/``, vendored to
``tests/``, para-drift):

* every limit in the repo's ``tool_limits.py`` ``LIMITS`` is set by an environment variable
  the source scanner can see (a literal ``limit("NAME", …)``), so ``tests/test_env_contract.py``
  holds it to ``.env.example`` like every other variable;
* ``.env.example`` declares each one, tagged ``[limit]``;
* ``service/README.md``'s ``## Limits`` table lists exactly the same keys, variables and
  defaults;
* setting each variable to another valid value changes the value ``LIMITS.values()`` reports
  — which is what ``/info`` ``limits`` is built from (each repo's ``tests/test_api_contract.py``
  asserts that ``/info`` equals ``LIMITS.values()``).

It needs the repo-local ``tool_limits.py`` and skips cleanly where there is none (the hub's
own ``docs/templates/shared/``).
"""

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOL_LIMITS = REPO_ROOT / "tool_limits.py"
ENV_EXAMPLE = REPO_ROOT / ".env.example"
SERVICE_README = REPO_ROOT / "service" / "README.md"

if not TOOL_LIMITS.is_file():
    pytest.skip(
        "no tool_limits.py here — this file checks a TOOL REPO's limits declaration, and the "
        "hub's docs/templates/shared/ is not one (atrium-project#53)",
        allow_module_level=True,
    )

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import atrium_limits  # noqa: E402

#: The upload limit is declared through atrium_limits.upload_limit(), whose variable the
#: scanner already finds in the shared modules (test_env_contract.py's _SHARED_CORE).
_DECLARED_BY_SHARED_HELPER = {"MAX_UPLOAD_MB"}

_SKIP_PARTS = {"tests", ".git", "node_modules", "site-packages", "dist-packages", "__pycache__"}


def _limits() -> atrium_limits.LimitSet:
    module = importlib.import_module("tool_limits")
    limits = getattr(module, "LIMITS", None)
    assert isinstance(limits, atrium_limits.LimitSet), "tool_limits.py must define LIMITS = LimitSet(...)"
    return limits


def _shipped_source() -> str:
    texts = []
    for py in sorted(REPO_ROOT.rglob("*.py")):
        rel = py.relative_to(REPO_ROOT)
        if _SKIP_PARTS & set(rel.parts):
            continue
        texts.append(py.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(texts)


def _norm(value) -> str:
    text = str(value).strip().strip("`")
    try:
        number = float(text.replace(" ", "").replace(" ", "").replace(",", ""))
    except ValueError:
        return text
    return str(int(number)) if number.is_integer() else str(number)


def test_tool_limits_declares_a_limit_set():
    limits = _limits()
    assert "max_upload_mb" in limits, "every service has the §4.5 upload limit (atrium_limits.upload_limit)"


def test_every_limit_is_visible_to_the_env_scanner():
    source = _shipped_source()
    invisible = sorted(
        spec.env
        for spec in _limits()
        if spec.env not in _DECLARED_BY_SHARED_HELPER
        and not re.search(r'(?<![\w])(?:resolve_)?limit\(\s*["\']' + re.escape(spec.env) + r'["\']', source)
    )
    assert not invisible, (
        "declare these with a literal `limit(\"NAME\", default, ...)` so tests/test_env_contract.py "
        f"holds them to .env.example: {invisible}"
    )


def _env_blocks() -> dict:
    """``NAME`` -> the comment block directly above its ``NAME=`` / ``# NAME=`` line."""
    lines = ENV_EXAMPLE.read_text(encoding="utf-8").splitlines()
    blocks = {}
    for i, line in enumerate(lines):
        m = re.match(r"^(?:#\s*)?([A-Z_][A-Z0-9_]*)=", line)
        if not m:
            continue
        comment = []
        j = i - 1
        while j >= 0 and lines[j].startswith("#") and not re.match(r"^#\s*[A-Z_][A-Z0-9_]*=", lines[j]):
            comment.append(lines[j])
            j -= 1
        blocks[m.group(1)] = "\n".join(reversed(comment))
    return blocks


def test_env_example_declares_every_limit_tagged_limit():
    blocks = _env_blocks()
    missing = sorted(spec.env for spec in _limits() if spec.env not in blocks)
    assert not missing, f".env.example does not declare these limits: {missing}"
    untagged = sorted(spec.env for spec in _limits() if "[limit]" not in blocks[spec.env])
    assert not untagged, (
        "tag these `[limit]` in .env.example (docs/templates/env.example.template), never "
        f"[algorithmic]: {untagged}"
    )


def _readme_limits_rows() -> dict:
    if not SERVICE_README.is_file():
        pytest.fail("service/README.md is missing; it must carry the `## Limits` table")
    rows = {}
    in_section = False
    for line in SERVICE_README.read_text(encoding="utf-8").splitlines():
        if re.match(r"^## Limits\b", line):
            in_section = True
            continue
        if in_section and line.startswith("## "):
            break
        if not in_section:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 4 or not re.fullmatch(r"`[a-z][a-z0-9_]*`", cells[0]):
            continue
        env = re.fullmatch(r"`([A-Z_][A-Z0-9_]*)`", cells[1])
        rows[cells[0].strip("`")] = {"env": env.group(1) if env else None, "default": cells[2]}
    assert in_section, "service/README.md has no `## Limits` section"
    return rows


def test_readme_limits_table_matches_the_declaration():
    limits = _limits()
    rows = _readme_limits_rows()
    meta = limits.meta()
    assert set(rows) == set(meta), (
        f"service/README.md `## Limits` keys {sorted(rows)} differ from tool_limits.LIMITS {sorted(meta)}"
    )
    wrong = []
    for key, entry in meta.items():
        if rows[key]["env"] != entry["env"]:
            wrong.append(f"{key}: README says {rows[key]['env']!r}, tool_limits says {entry['env']!r}")
        elif entry["env"] is not None and _norm(rows[key]["default"]) != _norm(entry["default"]):
            wrong.append(f"{key}: README default {rows[key]['default']!r}, code default {entry['default']!r}")
    assert not wrong, "\n".join(wrong)


def _other_value(spec: atrium_limits.LimitSpec):
    step = 1 if spec.kind is int else 0.5
    candidate = spec.default + step
    if spec.maximum is not None and candidate > spec.maximum:
        candidate = spec.default - step
    if spec.minimum is not None and candidate < spec.minimum:
        pytest.skip(f"{spec.env} has no other valid value")
    return candidate


def test_setting_each_variable_changes_the_reported_value(monkeypatch):
    limits = _limits()
    for spec in limits:
        monkeypatch.delenv(spec.env, raising=False)
        if spec.legacy_env:
            monkeypatch.delenv(spec.legacy_env, raising=False)
        other = _other_value(spec)
        monkeypatch.setenv(spec.env, str(other))
        values = limits.values()
        assert values[spec.key] == other, f"setting {spec.env}={other} left /info at {values[spec.key]!r}"
        assert limits.meta()[spec.key]["source"] == "env"
        monkeypatch.delenv(spec.env)


def test_meta_names_a_variable_or_a_derivation_for_every_key():
    for key, entry in _limits().meta().items():
        assert entry["env"] or entry.get("derived_from"), f"{key} is neither a setting nor derived"
