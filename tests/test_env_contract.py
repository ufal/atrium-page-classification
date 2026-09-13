"""tests/test_env_contract.py — .env.example is the published contract; keep it true.

atrium-project#60 asks for exactly this and says why: *"Each of the five repos has an
.env.example whose variable names are a superset of the service-layer os.getenv/
os.environ reads — check this with a script and keep it, or the tables rot within a
release."* That rot is not hypothetical: it happened twice before this file existed.
Three `.env.example` headers shipped a paragraph claiming `LOG_LEVEL` was "withheld"
while the same file declared `LOG_LEVEL=INFO` two sections later, and one repo's
`service/README.md` published a `MAX_UPLOAD_MB` default that disagreed with its own
`service/text_api.py`. Both were written by hand, both looked plausible, and nothing
caught either until a human happened to read both halves of the same file at once.

WHY THIS FILE IS SELF-CONTAINED. Every other canonical test under
docs/templates/shared/ is self-contained (test_para_licenses.py imports the module
beside it, and so on) — this one is not fully: it needs REPO-LOCAL data (which names a
repo deliberately does not publish, and why) that cannot be the same in all five repos.
That data lives in tests/env_contract_data.py, hand-written per repo, never vendored,
and imported below. Do not import atrium_test_support or any other repo-local helper —
this file must collect and pass in five repos whose only shared ancestor is the hub.

WHY THE ESCAPE HATCHES ARE NAME-KEYED, NOT DIRECTORY-KEYED. A directory-scoped
exclusion ("skip everything under api_util/") is untestable for staleness: nothing
notices when the code inside that directory changes. A name-keyed one
(NOT_PUBLISHED = {"UDPIPE_URL": "batch-only, see api_util/summarize_nt_udp.py"}) can be
asserted against directly — test_declared_omissions_are_still_read below fails the
moment a declared name is no longer read anywhere, which is the stale-exemption case a
directory exclusion can never catch.

WHY NAMES *AND* DEFAULTS. A names-only check would have passed all four of
nlp-enrich's wrong literals (MAX_UPLOAD_MB, MAX_WORDS, API_JOB_TIMEOUT,
MAX_RESCALE_DIM all present, all wrong) and alto-postprocess's README `MAX_UPLOAD_MB=10`
against a `resolve_max_upload_mb(25)` call site. Defaults are compared where they can be
resolved with a single hop against a literal in the source; where they cannot (a value
computed from another config file, a class instance's default parameter with no
`ast.Constant` in sight), this file makes no claim rather than guessing.

THE SHARED-CORE FLOOR (test_the_shared_core_is_in_the_scan). Three modules are
para-drift byte-identical across all five repos: atrium_paradata.py,
service/atrium_service.py and service/healthcheck.py. Scanning only those plus the
entrypoint's __main__ block yields the SAME thirteen names in every repo. That set is
asserted literally, byte-identically, in every vendored copy of this file — it is the
check that catches a broken scanner (a bad _SKIP_DIRS entry, a regex that stopped
matching, a repo that lost service/) rather than a genuinely clean repo, because a
scanner that has stopped scanning cannot satisfy it.

Run: pytest tests/test_env_contract.py
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_EXAMPLE = REPO_ROOT / ".env.example"
GITIGNORE = REPO_ROOT / ".gitignore"
ENV_CONTRACT_DATA = REPO_ROOT / "tests" / "env_contract_data.py"

# Every OTHER canonical test under docs/templates/shared/ that inspects a tool repo's
# own tree (test_logging_contract.py) skips cleanly when that tree is absent, because
# hub-self-check.yml's "Run canonical shared-module tests" step runs this file with
# working-directory: docs/templates/shared — where REPO_ROOT above resolves to
# docs/templates/, which is not a tool repo. An honest skip beats a vacuous pass.
if not ENV_CONTRACT_DATA.is_file():
    pytest.skip(
        "no tests/env_contract_data.py here — this file inspects a TOOL REPO's "
        "environment-variable declarations, and the hub's docs/templates/shared/ is "
        "not one (atrium-project#60)",
        allow_module_level=True,
    )

from tests.env_contract_data import (  # noqa: E402  (see the skip above)
    CONSUMED_ELSEWHERE,
    NOT_PUBLISHED,
    PROSE_DEFAULTS,
)

# Read from the process environment but supplied by the platform, not by an operator
# editing .env — documenting them as knobs would be misleading. Canonical: identical
# in every repo, unlike NOT_PUBLISHED/CONSUMED_ELSEWHERE which are repo-local.
_NOT_OPERATOR_KNOBS = {"HOME", "PATH", "PYTHONPATH"}

# The names read identically in all five repos via the three para-drift-enforced
# modules (atrium_paradata.py, service/atrium_service.py, service/healthcheck.py) and
# every entrypoint's __main__ block. See "THE SHARED-CORE FLOOR" above. Canonical: do
# not add a repo-specific name here, and do not remove one because a repo's entrypoint
# happens not to reach it today — file that as NOT_PUBLISHED with a reason instead.
_SHARED_CORE = {
    "ATRIUM_RUNNER_IMAGE",
    "ATRIUM_RUNNER_REPO",
    "ATRIUM_RUNNER_REF",
    "ATRIUM_REQUEST_ID",
    "ALLOWED_ORIGINS",
    "MAX_UPLOAD_MB",
    "MAX_UPLOAD_BYTES",
    "HEALTHCHECK_PATH",
    "PORT",
    "HOST",
    "GRACEFUL_SHUTDOWN_S",
    "RELOAD",
    "LOG_LEVEL",
}

_ENV_READ = re.compile(r'(?:os\.)?(?:environ\.get|getenv|environ\[)\(?\s*["\']([A-Z_][A-Z0-9_]*)["\']')
_ENV_HELPER = re.compile(r'_env_(?:float|int|str)\(\s*((?:["\'][A-Z_][A-Z0-9_]*["\']\s*,?\s*)+)')
_SKIP_DIRS = {"tests", "eval", "data_samples", "agent_dev_logs", ".git"}

# Directory NAMES with no repo-specific meaning, wherever they sit under the repo
# root: build caches and node_modules, plus "site-packages"/"dist-packages" as a
# defensive second layer alongside the pyvenv.cfg detection below (a venv's OWN root
# name is arbitrary -- .venv, venv, env, venv-trans -- but every virtualenv and every
# system Python names its third-party package directory exactly one of these two).
_SKIP_MARKERS = {"site-packages", "dist-packages", "node_modules", "__pycache__"}


def _venv_roots() -> set:
    """Every directory under the repo root that IS a Python virtual environment.

    `pyvenv.cfg` is the one file `python -m venv` / `virtualenv` always writes at a
    venv's root, regardless of what the venv directory itself is named -- unlike
    _SKIP_MARKERS above, this needs no name guess at all. Found the hard way: a real
    checkout with a `venv-trans/` and a `.venv/` sitting inside the repo (never
    committed, but very much present on disk) turned this file's ~4 real findings
    into ~250 platform/tooling variables (ANDROID_DATA, COVERAGE_FILE,
    PIP_CONFIG_FILE, ...) pulled from every installed library's own os.getenv calls
    -- including `.venv/bin/activate_this.py`, which sits OUTSIDE site-packages
    entirely, so that marker alone does not catch it.
    """
    return {p.parent for p in REPO_ROOT.rglob("pyvenv.cfg")}


def _indirect_env_reads(source: str) -> set[str]:
    """Resolve two shapes of `os.environ.get(<name>, ...)` where `<name>` is not an
    inline string literal, so a plain regex over the call site misses it entirely.

    1. A MODULE-LEVEL STRING CONSTANT: `_ENV_RUNNER_IMAGE = "ATRIUM_RUNNER_IMAGE"`,
       then `os.environ.get(_ENV_RUNNER_IMAGE)`. atrium_paradata.py reads
       ATRIUM_RUNNER_IMAGE / _REPO / _REF this way for all three.

    2. A LOOP OVER A NAME->KEY DICT: nlp-enrich's backing-service override reads
       `for env_key, fact_key in _ENV_OVERRIDABLE.items(): os.environ.get(env_key, ...)`
       where `_ENV_OVERRIDABLE = {"UDPIPE_URL": "udpipe_url", "NAMETAG_URL": ...}`.
       Every key in such a dict is genuinely read once per loop execution — this is
       not a guess, it is what the loop body does — so all of them resolve, not just
       one. Scoped narrowly: the dict must be a module-level literal with only
       string-constant keys, the loop target must be a 2-tuple of plain names, and
       the get/getenv call inside the loop body must use the loop's KEY variable.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:  # pragma: no cover - every file here parses
        return set()

    constants = {
        target.id: node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
        for target in node.targets
        if isinstance(target, ast.Name)
    }

    env_name_dicts: dict[str, set[str]] = {}
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict)):
            continue
        keys = node.value.keys
        if not keys or not all(isinstance(k, ast.Constant) and isinstance(k.value, str) for k in keys):
            continue
        literal_keys = {k.value for k in keys if re.fullmatch(r"[A-Z_][A-Z0-9_]*", k.value)}
        if literal_keys:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    env_name_dicts[target.id] = literal_keys

    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        func = node.func
        attr = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if attr not in {"get", "getenv"}:
            continue
        first = node.args[0]
        if isinstance(first, ast.Name) and first.id in constants:
            value = constants[first.id]
            if re.fullmatch(r"[A-Z_][A-Z0-9_]*", value):
                names.add(value)

    for node in ast.walk(tree):
        if not isinstance(node, ast.For):
            continue
        target, it = node.target, node.iter
        if not (
            isinstance(target, ast.Tuple)
            and len(target.elts) == 2
            and all(isinstance(e, ast.Name) for e in target.elts)
            and isinstance(it, ast.Call)
            and isinstance(it.func, ast.Attribute)
            and it.func.attr == "items"
            and isinstance(it.func.value, ast.Name)
            and it.func.value.id in env_name_dicts
        ):
            continue
        key_var = target.elts[0].id
        for inner in ast.walk(node):
            if not isinstance(inner, ast.Call) or not inner.args:
                continue
            inner_func = inner.func
            inner_attr = inner_func.attr if isinstance(inner_func, ast.Attribute) else getattr(inner_func, "id", "")
            if inner_attr not in {"get", "getenv"}:
                continue
            first = inner.args[0]
            if isinstance(first, ast.Name) and first.id == key_var:
                names |= env_name_dicts[it.func.value.id]
                break
    return names


def _literal_default(node: ast.expr) -> str | None:
    """A single-hop literal default: a plain constant, or unresolvable (None)."""
    if isinstance(node, ast.Constant) and not isinstance(node.value, bool):
        return str(node.value)
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return "true" if node.value else "false"
    return None


def _add_parents(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child._env_contract_parent = node  # type: ignore[attr-defined]


def _code_defaults(source: str) -> dict[str, str]:
    """`os.getenv(NAME, <literal>)` / `resolve_max_upload_mb(<literal>)` defaults.

    Deliberately conservative: a default that is a name, an f-string, or a function
    call is left unresolved rather than guessed at. See the module docstring's
    "WHY NAMES *AND* DEFAULTS" paragraph — never guess, fail only on contradiction.

    One more shape is deliberately excluded: `os.environ.get(NAME, "") or FALLBACK`.
    `""` there is a plumbing detail (something falsy for the `or` to fall through),
    not the effective default — translator's `processors/lemmatizer.py` reads
    UDPIPE_URL exactly this way, with the real default (a LINDAT URL) in FALLBACK,
    two hops away. Recording `""` as "the code default" would have this file
    contradict service/README.md's correct prose cell for that name. A call whose
    immediate parent is `... or ...` is left unresolved rather than guessed at.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:  # pragma: no cover
        return {}
    _add_parents(tree)

    defaults: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        attr = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if attr in {"getenv", "get"} and node.args:
            parent = getattr(node, "_env_contract_parent", None)
            if isinstance(parent, ast.BoolOp) and isinstance(parent.op, ast.Or):
                continue
            name_arg = node.args[0]
            if isinstance(name_arg, ast.Constant) and isinstance(name_arg.value, str):
                name = name_arg.value
                if len(node.args) >= 2:
                    resolved = _literal_default(node.args[1])
                    if resolved is not None:
                        defaults.setdefault(name, resolved)
        elif attr == "resolve_max_upload_mb" and node.args:
            resolved = _literal_default(node.args[0])
            if resolved is not None:
                defaults["MAX_UPLOAD_MB"] = str(float(resolved))
    return defaults


def _iter_scanned_files():
    venv_roots = _venv_roots()
    for py in sorted(REPO_ROOT.rglob("*.py")):
        parts = py.relative_to(REPO_ROOT).parts
        if any(part in _SKIP_DIRS for part in parts):
            continue
        if any(part in _SKIP_MARKERS for part in parts):
            continue
        if any(root in py.parents for root in venv_roots):
            continue
        yield py


def _runtime_env_reads() -> dict[str, set[str]]:
    """Every environment variable the shipped code reads, and where."""
    found: dict[str, set[str]] = {}
    for py in _iter_scanned_files():
        source = py.read_text(encoding="utf-8")
        names = set(_ENV_READ.findall(source))
        for group in _ENV_HELPER.findall(source):
            names.update(re.findall(r'["\']([A-Z_][A-Z0-9_]*)["\']', group))
        names |= _indirect_env_reads(source)
        for name in names - _NOT_OPERATOR_KNOBS:
            found.setdefault(name, set()).add(str(py.relative_to(REPO_ROOT)))
    return found


def _runtime_code_defaults() -> dict[str, str]:
    defaults: dict[str, str] = {}
    for py in _iter_scanned_files():
        defaults.update({k: v for k, v in _code_defaults(py.read_text(encoding="utf-8")).items() if k not in defaults})
    return defaults


_ENV_LINE = re.compile(r"^(?P<hash>#\s*)?([A-Z_][A-Z0-9_]*)=(.*)$", re.MULTILINE)


def _all_declared() -> dict[str, tuple[bool, str]]:
    """Every `NAME=value` in .env.example -> (is_active, raw_value)."""
    declared: dict[str, tuple[bool, str]] = {}
    for m in _ENV_LINE.finditer(ENV_EXAMPLE.read_text(encoding="utf-8")):
        declared[m.group(2)] = (m.group("hash") is None, m.group(3))
    return declared


def _documented() -> set[str]:
    """Names declared as `KEY=` in .env.example, commented-out entries included."""
    return set(_all_declared())


def _active_documented() -> dict[str, str]:
    """Uncommented `KEY=value` entries only — these are what Rule 1 pins to the code default."""
    return {name: value for name, (active, value) in _all_declared().items() if active}


def _normalize(value: str) -> str:
    value = value.strip().strip('"').strip("'")
    low = value.lower()
    if low in {"true", "1", "yes", "on"}:
        return "true"
    if low in {"false", "0", "no", "off"}:
        return "false"
    try:
        return str(float(value))
    except ValueError:
        return value


def test_env_example_documents_every_variable_the_code_reads():
    reads = _runtime_env_reads()
    undocumented = sorted(set(reads) - _documented() - set(NOT_PUBLISHED))
    detail = "\n".join(f"  {name}  (read in {', '.join(sorted(reads[name]))})" for name in undocumented)
    assert not undocumented, (
        "These variables are read by shipped code but are not in .env.example and not "
        f"declared in tests/env_contract_data.py's NOT_PUBLISHED, so a partner can only "
        f"discover them by reading the source:\n{detail}"
    )


def test_env_example_documents_nothing_imaginary():
    """A contract that lists knobs the code does not read is its own kind of lie."""
    reads = set(_runtime_env_reads())
    phantom = sorted(_documented() - reads - set(CONSUMED_ELSEWHERE))
    assert not phantom, (
        f".env.example documents variables nothing reads: {phantom}. Either the code "
        "stopped reading them, or they belong in tests/env_contract_data.py's "
        "CONSUMED_ELSEWHERE with a reason."
    )


def test_declared_omissions_are_still_read():
    """A NOT_PUBLISHED entry the code no longer reads is a stale exemption."""
    reads = set(_runtime_env_reads())
    stale = sorted(set(NOT_PUBLISHED) - reads)
    assert not stale, (
        f"tests/env_contract_data.py's NOT_PUBLISHED declares these as deliberately "
        f"withheld from .env.example, but nothing reads them any more — remove the "
        f"exemption (it is now free to widen silently): {stale}"
    )


def test_declared_omissions_are_not_also_published():
    """The exact shape of the LOG_LEVEL defect this issue found: withheld AND published."""
    contradictions = sorted(set(NOT_PUBLISHED) & _documented())
    assert not contradictions, (
        ".env.example both DECLARES and PUBLISHES these variables — a file cannot say "
        f"a name is withheld while also assigning it a value: {contradictions}"
    )


def test_consumed_elsewhere_entries_are_published():
    missing = sorted(set(CONSUMED_ELSEWHERE) - _documented())
    assert not missing, (
        f"tests/env_contract_data.py's CONSUMED_ELSEWHERE names these as published but "
        f"they are not in .env.example at all: {missing}"
    )


def test_consumed_elsewhere_entries_are_not_read():
    reads = set(_runtime_env_reads())
    also_read = sorted(set(CONSUMED_ELSEWHERE) & reads)
    assert not also_read, (
        f"tests/env_contract_data.py's CONSUMED_ELSEWHERE claims these are read by "
        f"nothing in this repo's Python, but the scanner found a read for them — if "
        f"that is now wrong, drop the CONSUMED_ELSEWHERE entry instead: {also_read}"
    )


def test_every_declaration_carries_a_reason():
    thin = []
    for source_name, mapping in (
        ("NOT_PUBLISHED", NOT_PUBLISHED),
        ("CONSUMED_ELSEWHERE", CONSUMED_ELSEWHERE),
        ("PROSE_DEFAULTS", PROSE_DEFAULTS),
    ):
        for name, reason in mapping.items():
            if len(reason.strip()) < 30:
                thin.append(f"{source_name}[{name!r}] = {reason!r}")
    assert not thin, (
        "a one-word reason is how an escape hatch becomes a dumping ground — each "
        "declaration needs >= 30 characters of explanation:\n" + "\n".join(thin)
    )


def test_the_shared_core_is_in_the_scan():
    """The 13-name floor that a scanner cannot satisfy by having stopped scanning.

    Byte-identical in every vendored copy of this file, because its subject —
    atrium_paradata.py, service/atrium_service.py, service/healthcheck.py, and the
    entrypoint's __main__ block — is itself para-drift canonical.
    """
    reads = set(_runtime_env_reads())
    missing = sorted(_SHARED_CORE - reads)
    assert not missing, (
        "these names come from the three para-drift-enforced shared modules and every "
        "entrypoint's __main__ block, so they must be found in EVERY repo — their "
        "absence means the scanner stopped scanning (a bad _SKIP_DIRS entry, a moved "
        f"service/ directory, a broken regex), not that the repo is clean: {missing}"
    )


def test_env_example_defaults_match_the_code():
    """Rule 1 (docs/templates/env.example.template): every active value is the code
    default, verbatim. Catches nlp-enrich's four-defaults drift and alto-postprocess's
    README MAX_UPLOAD_MB=10-vs-25 in the form this file actually enforces.
    """
    code_defaults = _runtime_code_defaults()
    active = _active_documented()
    mismatches = []
    for name, doc_value in active.items():
        if name in PROSE_DEFAULTS or name not in code_defaults:
            continue
        if _normalize(doc_value) != _normalize(code_defaults[name]):
            mismatches.append(f"{name}: .env.example says {doc_value!r}, code default is {code_defaults[name]!r}")
    assert not mismatches, "\n".join(mismatches)


def test_dot_env_is_gitignored():
    """docker-tool.reusable.yml's `cp .env.example .env` materialises this file in
    every CI checkout; it must never be a candidate for `git add`.
    """
    if not GITIGNORE.is_file():
        pytest.fail(".gitignore does not exist")
    lines = {line.strip() for line in GITIGNORE.read_text(encoding="utf-8").splitlines()}
    assert ".env" in lines, (
        ".gitignore has no exact `.env` line. This cannot instead assert `.env` is "
        "ABSENT from the working tree — docker-tool.reusable.yml's CI lane creates it "
        "before this test ever runs."
    )


_README_HEADING = re.compile(r"^## Configuration \(environment\)")
_README_ROW = re.compile(r"^\|\s*`([A-Z][A-Z0-9_]*)`\s*\|\s*([^|]*?)\s*\|")


def _service_readme_rows() -> dict[str, str] | None:
    readme = REPO_ROOT / "service" / "README.md"
    if not readme.is_file():
        return None
    lines = readme.read_text(encoding="utf-8").splitlines()
    rows: dict[str, str] = {}
    in_section = False
    for line in lines:
        if _README_HEADING.match(line):
            in_section = True
            continue
        if in_section:
            if line.startswith("## "):
                break
            m = _README_ROW.match(line)
            if m:
                rows[m.group(1)] = m.group(2).strip("`")
    return rows


def test_service_readme_is_a_subset_of_env_example():
    rows = _service_readme_rows()
    if rows is None:
        pytest.skip("no service/README.md here")
    documented = _documented()
    missing = sorted(set(rows) - documented)
    assert not missing, (
        f"service/README.md's env table names variables .env.example does not: "
        f"{missing} — the README is meant to be a curated SUBSET of the ledger"
    )


def test_service_readme_defaults_match_the_code():
    rows = _service_readme_rows()
    if rows is None:
        pytest.skip("no service/README.md here")
    code_defaults = _runtime_code_defaults()
    mismatches = []
    for name, doc_value in rows.items():
        if name in PROSE_DEFAULTS or name not in code_defaults:
            continue
        if _normalize(doc_value) != _normalize(code_defaults[name]):
            mismatches.append(f"{name}: service/README.md says {doc_value!r}, code default is {code_defaults[name]!r}")
    assert not mismatches, "\n".join(mismatches)
