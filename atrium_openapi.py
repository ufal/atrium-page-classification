#!/usr/bin/env python3
"""atrium_openapi.py — the OpenAPI document as a release artefact (atrium-project#32 item 3).

Canonical copy: the hub's ``docs/templates/shared/atrium_openapi.py``, vendored byte-identically
to the root of every tool repo (``docs/templates/shared/MANIFEST.json``, held by para-drift). It
imports only the standard library, so a release job can run it with the runner's own Python;
``service/atrium_service.py`` imports it lazily for :func:`digest`.

Why it exists. The AMČR pipeline generates its clients from the ``openapi.json`` attached to the
release it trusts, and reads a running service's ``/openapi.json`` only to check that the image
matches that release (motyc, atrium-project#32, 2026-09-27). So each tool:

* commits its spec as ``service/openapi.json`` — ``export`` writes it and ``check`` fails when the
  code no longer produces it (``info.version`` aside: it is stamped at release time);
* attaches it to every release — ``stamp`` writes ``openapi.json`` with the release version, in
  the canonical form of :func:`canonical_bytes`, and ``openapi.json.sha256``, a ``sha256sum`` line
  for it. One value throughout: ``sha256sum -c openapi.json.sha256`` verifies the asset, GitHub
  shows the same digest for it, and ``/info`` reports it as ``openapi_sha256``;
* compares it with the previous release before publishing — ``baseline`` fetches the previous
  release's ``openapi.json`` and ``compare`` fails on a breaking change unless the major version
  went up (0.x included: a 0.x tool needs 1.0 to break), and always on a removed reason code.

A renamed service (atrium-project#72, the ocr-postprocess move of atrium-alto-postprocess#56)
declares its old id: ``info.x-atrium-service-previous`` (written by ``atrium_service``'s
``attach_openapi_contract(app, service, previous=...)``). ``compare`` accepts a changed
``info.x-atrium-service`` only when that declaration equals the baseline's id; an undeclared
change, or one declaring another id, stays fatal whatever the version. Nothing else about the
comparison changes: the renamed release is still held to every other rule.

Subcommands (run from the repo root)::

    python atrium_openapi.py export   --app service.api:app [--out service/openapi.json] [--prepare MOD:FUNC]
    python atrium_openapi.py check    --app service.api:app [--spec service/openapi.json] [--prepare MOD:FUNC]
    python atrium_openapi.py stamp    [--spec service/openapi.json] --para-config para_config.txt --out-dir dist
    python atrium_openapi.py baseline --repo ufal/atrium-<tool> [--tag vX.Y.Z] --out previous.json [--token-stdin]
    python atrium_openapi.py compare  --base previous.json --rev dist/openapi.json --para-config para_config.txt
                                      [--oasdiff oasdiff] [--require-oasdiff] [--summary "$GITHUB_STEP_SUMMARY"]
    python atrium_openapi.py digest   dist/openapi.json
    python atrium_openapi.py --selftest

``--prepare MOD:FUNC`` runs before the app is imported: page-classification's service needs
torch, and its ``tests/openapi_contract_data.py`` provides the stub its light test lane already
uses. ``baseline`` takes a GitHub token only on stdin (``--token-stdin``): an environment read
here would be one more variable ``tests/test_env_contract.py`` holds each repo's ledger to.

The oasdiff rules. ``compare`` runs ``oasdiff breaking --fail-on ERR`` (pinned: :data:`OASDIFF_VERSION`)
with three rules raised from their INFO default to ERR — an optional response property removed,
an operationId removed (generated clients name methods after it) and a declared error status
removed — and after two normalisations of both documents: a two-branch ``anyOf`` with a ``null``
branch (pydantic's ``Optional``) becomes ``oneOf``, which oasdiff reads as "nullable" (otherwise it
reports every property under the reference as removed), and the ``AtriumDocument*`` components
become ``{type: object}``, because the record schema is guarded by the schema freeze
(``tests/test_schema_freeze.py``) and compared here by its ``x-atrium-record-schema`` major and
sha. Measured against oasdiff v1.32.1 on 2026-09-28 (``agent_dev_logs/plans/32.plan.md``). Without
oasdiff and without ``--require-oasdiff``, a small built-in rule set stands in and says so.
"""

from __future__ import annotations

import argparse
import configparser
import copy
import hashlib
import importlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

#: The committed spec of a repo's ``api`` service, and the release asset it becomes.
SPEC_PATH = "service/openapi.json"
ASSET = "openapi.json"

#: The component the record schema is published under (``atrium_service.RECORD_COMPONENT``).
RECORD_COMPONENT = "AtriumDocument"

#: ``info`` keys of the service id and of a declared rename (``atrium_service`` writes both).
SERVICE_KEY = "x-atrium-service"
PREVIOUS_SERVICE_KEY = "x-atrium-service-previous"

#: The oasdiff release ``compare`` is verified against; CI installs exactly this one.
OASDIFF_VERSION = "v1.32.1"

#: oasdiff rules whose default level (INFO) would let a real break through ``--fail-on ERR``.
RAISED_RULES = (
    "response-optional-property-removed",
    "api-operation-id-removed",
    "response-non-success-status-removed",
)

_VERSION_RE = re.compile(r"^v?(\d+)(?:\.(\d+))?(?:\.(\d+))?(?:[-+.](.+))?$")


class OpenAPIError(RuntimeError):
    """A failure that must stop the job (exit 2), as opposed to a finding (exit 1)."""


# ── the document ───────────────────────────────────────────────────────────────────────────


def canonical_bytes(document: Any) -> bytes:
    """The canonical form the digests are taken of: sorted keys, no whitespace, UTF-8."""
    return json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def digest(document: Any) -> str:
    """sha256 (hex) of :func:`canonical_bytes` — ``/info`` ``openapi_sha256`` and the ``.sha256`` asset.

    The release asset is written in that form (``stamp``), so this is also the plain sha256 of its
    bytes. Written with :func:`dumps` instead, the asset would not match its own ``.sha256`` line.
    """
    return hashlib.sha256(canonical_bytes(document)).hexdigest()


def dumps(document: Any) -> str:
    """The committed form: FastAPI's own key order, two-space indent, a final newline."""
    return json.dumps(document, indent=2, ensure_ascii=False) + "\n"


def load(path: Path | str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def without_version(spec: Dict[str, Any]) -> Dict[str, Any]:
    """``spec`` with ``info.version`` blanked: the committed spec is stamped only at release time."""
    out = copy.deepcopy(spec)
    out.setdefault("info", {})["version"] = ""
    return out


def read_version(para_config: Path | str) -> str:
    """``[tool] version`` of ``para_config.txt``, without a leading ``v`` (as ``/info`` reports it)."""
    config = configparser.ConfigParser()
    if not config.read(para_config, encoding="utf-8"):
        raise OpenAPIError(f"cannot read {para_config}")
    version = config.get("tool", "version", fallback="").strip()
    if not version:
        raise OpenAPIError(f"{para_config} has no [tool] version")
    return version[1:] if version.lower().startswith("v") else version


def version_key(version: str) -> Tuple[int, int, int, int, str]:
    """Order release versions: ``1.8.0-beta`` < ``1.8.0`` < ``1.8.1-beta``; a leading ``v`` is ignored."""
    match = _VERSION_RE.match(version.strip())
    if not match:
        raise OpenAPIError(f"{version!r} is not a version")
    major, minor, patch, pre = match.groups()
    return int(major), int(minor or 0), int(patch or 0), 0 if pre else 1, pre or ""


def major(version: str) -> int:
    return version_key(version)[0]


def differences(a: Any, b: Any, path: str = "", limit: int = 20) -> List[str]:
    """JSON pointers (at most ``limit``) where ``a`` and ``b`` differ — for readable failures."""
    found: List[str] = []

    def walk(x: Any, y: Any, where: str) -> None:
        if len(found) >= limit:
            return
        if isinstance(x, dict) and isinstance(y, dict):
            for key in sorted(set(x) | set(y), key=str):
                child = f"{where}/{str(key).replace('~', '~0').replace('/', '~1')}"
                if key not in x:
                    found.append(f"{child}: added")
                elif key not in y:
                    found.append(f"{child}: removed")
                else:
                    walk(x[key], y[key], child)
        elif isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
            for index, (xi, yi) in enumerate(zip(x, y, strict=True)):
                walk(xi, yi, f"{where}/{index}")
        elif x != y:
            found.append(f"{where or '/'}: changed")

    walk(a, b, path)
    return found


# ── the live spec ──────────────────────────────────────────────────────────────────────────


def _call(ref: str) -> Any:
    module, _, attr = ref.partition(":")
    if not module or not attr:
        raise OpenAPIError(f"{ref!r} is not MODULE:ATTRIBUTE")
    return getattr(importlib.import_module(module), attr)


def live_spec(app_ref: str, root: Path | str = ".", prepare: Optional[str] = None) -> Dict[str, Any]:
    """The spec the app at ``app_ref`` (``service.api:app``) generates, imported from ``root``.

    A duplicate operationId is an error here, not FastAPI's warning: two routes with one
    operationId would give a generated client one method for both.
    """
    root = str(Path(root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    if prepare:
        _call(prepare)()
    app = _call(app_ref)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*Duplicate Operation ID.*")
        return json.loads(json.dumps(app.openapi()))


def check(committed: Dict[str, Any], live: Dict[str, Any]) -> List[str]:
    """Where the live spec differs from the committed one, ``info.version`` aside (empty: current).

    Read each pointer as committed → generated: ``removed`` is in the committed file but no
    longer generated, ``added`` is generated but not committed yet.
    """
    return differences(without_version(committed), without_version(live))


def stamp(spec: Dict[str, Any], version: str) -> Dict[str, Any]:
    """The release asset: the committed spec with ``info.version`` set to the release version."""
    out = copy.deepcopy(spec)
    out.setdefault("info", {})["version"] = version
    return out


# ── the baseline ───────────────────────────────────────────────────────────────────────────

Opener = Callable[[urllib.request.Request], Any]


def _get(url: str, token: Optional[str], opener: Opener, accept: str = "application/vnd.github+json") -> bytes:
    request = urllib.request.Request(url, headers={"Accept": accept, "User-Agent": "atrium-openapi"})
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with opener(request) as response:
            return response.read()
    except (urllib.error.URLError, OSError) as exc:
        raise OpenAPIError(f"GET {url} failed: {exc}") from exc


def select_baseline(
    releases: Iterable[Dict[str, Any]], tag: Optional[str], asset: str = ASSET
) -> Optional[Dict[str, Any]]:
    """The release a new spec is compared with: the highest version below ``tag`` carrying ``asset``.

    Drafts and prereleases never count. Without ``tag`` (a pull request or a branch push) the
    highest release carrying ``asset`` counts. ``None``: no such release, the bootstrap case.
    """
    below = version_key(tag) if tag else None
    candidates = []
    for release in releases:
        name = str(release.get("tag_name") or "")
        if release.get("draft") or release.get("prerelease") or not _VERSION_RE.match(name):
            continue
        if not any(a.get("name") == asset for a in release.get("assets") or []):
            continue
        key = version_key(name)
        if below is not None and key >= below:
            continue
        candidates.append((key, release))
    return max(candidates, key=lambda pair: pair[0])[1] if candidates else None


def baseline(
    repo: str,
    tag: Optional[str],
    token: Optional[str] = None,
    asset: str = ASSET,
    api_url: str = "https://api.github.com",
    opener: Opener = urllib.request.urlopen,
    max_pages: int = 10,
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """``(tag, spec)`` of the baseline release, or ``(None, None)`` when there is none.

    Any API or download error raises :class:`OpenAPIError`: only a successful listing that
    finds no eligible release is the bootstrap pass. The asset is fetched from its public
    ``browser_download_url`` without the token (the redirect target refuses one).
    """
    releases: List[Dict[str, Any]] = []
    for page in range(1, max_pages + 1):
        batch = json.loads(_get(f"{api_url}/repos/{repo}/releases?per_page=100&page={page}", token, opener))
        if not isinstance(batch, list):
            raise OpenAPIError(f"unexpected release listing for {repo}: {str(batch)[:200]}")
        releases.extend(batch)
        if len(batch) < 100:
            break
    chosen = select_baseline(releases, tag, asset)
    if chosen is None:
        return None, None
    url = next(a["browser_download_url"] for a in chosen["assets"] if a.get("name") == asset)
    return chosen["tag_name"], json.loads(_get(url, None, opener, accept="application/octet-stream"))


# ── the comparison ─────────────────────────────────────────────────────────────────────────


def normalise(spec: Dict[str, Any]) -> Dict[str, Any]:
    """The form both sides are diffed in: ``anyOf[X, null]`` as ``oneOf``, the record schema stubbed."""

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            node = {key: walk(value) for key, value in node.items()}
            branches = node.get("anyOf")
            if isinstance(branches, list) and len(branches) == 2 and {"type": "null"} in branches:
                node["oneOf"] = node.pop("anyOf")
            return node
        if isinstance(node, list):
            return [walk(item) for item in node]
        return node

    out = walk(spec)
    schemas = out.get("components", {}).get("schemas", {})
    for name in list(schemas):
        if name == RECORD_COMPONENT or name.startswith(RECORD_COMPONENT + "_"):
            schemas[name] = {"type": "object"}
    return out


def _operations(spec: Dict[str, Any]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    methods = {"get", "put", "post", "delete", "patch", "options", "head", "trace"}
    return {
        (path, method): operation
        for path, item in (spec.get("paths") or {}).items()
        for method, operation in (item or {}).items()
        if method in methods and isinstance(operation, dict)
    }


def _resolve(spec: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    ref = schema.get("$ref", "")
    if ref.startswith("#/components/schemas/"):
        return (spec.get("components", {}).get("schemas", {})).get(ref.rsplit("/", 1)[-1], {})
    return schema


def builtin_breaking(base: Dict[str, Any], rev: Dict[str, Any]) -> List[str]:
    """The stand-in for oasdiff: the breaking changes a generated client notices first.

    Removed operations and operationIds, removed response statuses, removed properties of a
    JSON response body, and request fields that became required. Coarser than oasdiff; used
    only where oasdiff is not installed and not required.
    """
    found: List[str] = []
    base_ops, rev_ops = _operations(base), _operations(rev)
    for key, operation in base_ops.items():
        where = f"{key[1].upper()} {key[0]}"
        if key not in rev_ops:
            found.append(f"{where}: operation removed")
            continue
        other = rev_ops[key]
        if operation.get("operationId") != other.get("operationId"):
            found.append(f"{where}: operationId {operation.get('operationId')!r} → {other.get('operationId')!r}")
        for status, response in (operation.get("responses") or {}).items():
            if status not in (other.get("responses") or {}):
                found.append(f"{where}: response {status} removed")
                continue
            for media, content in (response.get("content") or {}).items():
                new = ((other["responses"][status].get("content") or {}).get(media) or {}).get("schema")
                if new is None:
                    found.append(f"{where}: response {status} {media} removed")
                    continue
                old_props = _resolve(base, content.get("schema") or {}).get("properties") or {}
                new_props = _resolve(rev, new).get("properties") or {}
                for prop in sorted(set(old_props) - set(new_props)):
                    found.append(f"{where}: response {status} property {prop!r} removed")
        for media, content in ((operation.get("requestBody") or {}).get("content") or {}).items():
            new = (((other.get("requestBody") or {}).get("content") or {}).get(media) or {}).get("schema")
            if new is None:
                continue
            old_required = set(_resolve(base, content.get("schema") or {}).get("required") or [])
            new_required = set(_resolve(rev, new).get("required") or [])
            for prop in sorted(new_required - old_required):
                found.append(f"{where}: request property {prop!r} became required")
    return found


def run_oasdiff(base: Dict[str, Any], rev: Dict[str, Any], binary: str) -> Tuple[bool, str]:
    """``(breaking, report)`` from ``oasdiff breaking --fail-on ERR`` on the normalised documents."""
    with tempfile.TemporaryDirectory() as tmp:
        base_path, rev_path, levels = Path(tmp, "base.json"), Path(tmp, "rev.json"), Path(tmp, "levels.txt")
        base_path.write_text(json.dumps(normalise(base)), encoding="utf-8")
        rev_path.write_text(json.dumps(normalise(rev)), encoding="utf-8")
        levels.write_text("".join(f"{rule} err\n" for rule in RAISED_RULES), encoding="utf-8")
        proc = subprocess.run(
            [binary, "breaking", str(base_path), str(rev_path), "--fail-on", "ERR"]
            + ["--severity-levels", str(levels), "--format", "text"],
            capture_output=True,
            text=True,
            check=False,
        )
    report = (proc.stdout + proc.stderr).strip()
    if proc.returncode not in (0, 1):
        raise OpenAPIError(f"oasdiff exited {proc.returncode}: {report[:2000]}")
    return proc.returncode == 1, report


def compare(
    base: Optional[Dict[str, Any]],
    rev: Dict[str, Any],
    rev_version: str,
    oasdiff: Optional[str] = None,
    require_oasdiff: bool = False,
    base_label: str = "the previous release",
) -> Tuple[int, List[str]]:
    """``(exit code, report lines)``: 0 compatible, 1 a finding that fails the build.

    Always fatal: a changed service id the new spec does not declare as a rename (its
    ``info.x-atrium-service-previous`` equal to the baseline's id, atrium-project#72); a
    removed reason code, or a status removed from one (published codes are never renamed or
    removed, §4.4). Fatal unless the major version went up: every breaking change oasdiff (or
    the built-in rules) reports, a removed operationId, and a record-schema major change.
    """
    lines: List[str] = []
    if base is None:
        return 0, [f"No baseline: no earlier release carries `{ASSET}`. This release is the bootstrap."]
    base_version = str(base.get("info", {}).get("version") or "0")
    bumped = major(rev_version) > major(base_version)
    lines.append(f"Comparing `{rev_version}` with {base_label} (`{base_version}`); major version raised: {bumped}.")
    fatal: List[str] = []
    breaking: List[str] = []

    old_service = base.get("info", {}).get(SERVICE_KEY)
    new_service = rev.get("info", {}).get(SERVICE_KEY)
    declared = rev.get("info", {}).get(PREVIOUS_SERVICE_KEY)
    if old_service and old_service != new_service:
        if declared == old_service:
            lines.append(
                f"Declared rename: `info.{SERVICE_KEY}` {old_service!r} → {new_service!r} "
                f"(`info.{PREVIOUS_SERVICE_KEY}`); every other rule still applies."
            )
        elif declared:
            fatal.append(
                f"`info.{SERVICE_KEY}` changed: {old_service!r} → {new_service!r}, but "
                f"`info.{PREVIOUS_SERVICE_KEY}` declares {declared!r}, not the baseline's id"
            )
        else:
            fatal.append(
                f"`info.{SERVICE_KEY}` changed: {old_service!r} → {new_service!r} (a rename must declare "
                f"`info.{PREVIOUS_SERVICE_KEY}: {old_service!r}`: attach_openapi_contract(..., previous=...))"
            )

    old_codes = base.get("x-atrium-reason-codes") or {}
    new_codes = rev.get("x-atrium-reason-codes") or {}
    for code in sorted(old_codes):
        if code not in new_codes:
            fatal.append(f"reason code `{code}` removed (a published code is never renamed or removed)")
            continue
        lost = set(old_codes[code].get("statuses") or []) - set(new_codes[code].get("statuses") or [])
        if lost:
            fatal.append(f"reason code `{code}` no longer sent with HTTP {sorted(lost)}")

    old_ids = {op.get("operationId") for op in _operations(base).values()} - {None}
    new_ids = {op.get("operationId") for op in _operations(rev).values()} - {None}
    breaking.extend(f"operationId `{op_id}` removed" for op_id in sorted(old_ids - new_ids))

    old_record = base.get("x-atrium-record-schema") or {}
    new_record = rev.get("x-atrium-record-schema") or {}
    if old_record and new_record:
        if major(str(old_record.get("schema_version"))) != major(str(new_record.get("schema_version"))):
            breaking.append(
                f"record schema major changed: {old_record.get('schema_version')} → {new_record.get('schema_version')}"
            )
        elif old_record.get("sha256") != new_record.get("sha256"):
            lines.append("The record schema changed within its major (additive under the schema freeze).")

    binary = shutil.which(oasdiff) if oasdiff else None
    if binary:
        found, report = run_oasdiff(base, rev, binary)
        if found:
            breaking.append("oasdiff reports breaking changes:\n\n```\n" + report + "\n```")
        else:
            lines.append("oasdiff: no breaking changes.")
    elif require_oasdiff:
        fatal.append(f"oasdiff `{oasdiff}` is not installed (install {OASDIFF_VERSION})")
    else:
        extra = builtin_breaking(base, rev)
        lines.append("oasdiff not installed: the built-in rules stood in.")
        breaking.extend(extra)

    for item in fatal:
        lines.append(f"- ❌ {item}")
    for item in breaking:
        lines.append(f"- {'⚠️' if bumped else '❌'} {item}")
    if breaking and bumped:
        lines.append("Breaking changes are allowed: the major version went up.")
    failed = bool(fatal) or (bool(breaking) and not bumped)
    lines.append("Result: " + ("FAIL" if failed else "compatible"))
    return (1 if failed else 0), lines


# ── response validation for the contract tests ─────────────────────────────────────────────


def validate_response(
    spec: Dict[str, Any], path: str, method: str, status: int, body: Any, media_type: str = "application/json"
) -> None:
    """Validate a response body against the schema ``spec`` declares for it (needs ``jsonschema``).

    Raises ``jsonschema.ValidationError`` when the body does not conform, ``KeyError`` when the
    spec declares no such response. Used by each repo's contract tests, so the published
    schema — not only the pydantic model it came from — is what real responses are held to.
    """
    import jsonschema  # optional: only the tests need it

    operation = spec["paths"][path][method.lower()]
    schema = operation["responses"][str(status)]["content"][media_type]["schema"]
    root = dict(schema)
    root["components"] = spec.get("components", {})
    jsonschema.Draft202012Validator(root).validate(body)


# ── command line ───────────────────────────────────────────────────────────────────────────


def _write_summary(path: Optional[str], title: str, lines: List[str]) -> None:
    if not path:
        return
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(f"### {title}\n\n" + "\n".join(lines) + "\n\n")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--selftest", action="store_true", help="run the built-in self-test and exit")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("export", help="write the app's spec (the committed service/openapi.json)")
    p.add_argument("--app", required=True)
    p.add_argument("--out", default=SPEC_PATH, help="'-' for stdout")
    p.add_argument("--prepare")
    p.add_argument("--root", default=".")

    p = sub.add_parser("check", help="fail when the committed spec is not what the app generates")
    p.add_argument("--app", required=True)
    p.add_argument("--spec", default=SPEC_PATH)
    p.add_argument("--prepare")
    p.add_argument("--root", default=".")

    p = sub.add_parser("stamp", help="write the release assets openapi.json (canonical form) and openapi.json.sha256")
    p.add_argument("--spec", default=SPEC_PATH)
    p.add_argument("--para-config", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--asset", default=ASSET)

    p = sub.add_parser("baseline", help="fetch the previous release's openapi.json")
    p.add_argument("--repo", required=True)
    p.add_argument("--tag", help="the release being made; the baseline is the highest version below it")
    p.add_argument("--out", required=True)
    p.add_argument("--asset", default=ASSET)
    p.add_argument("--token-stdin", action="store_true", help="read a GitHub token from stdin")
    p.add_argument("--api-url", default="https://api.github.com")

    p = sub.add_parser("compare", help="fail on a breaking change against the baseline")
    p.add_argument("--base", required=True, help="the baseline spec; a missing file is the bootstrap")
    p.add_argument("--rev", required=True)
    p.add_argument("--para-config", required=True)
    p.add_argument("--oasdiff", default="oasdiff")
    p.add_argument("--require-oasdiff", action="store_true")
    p.add_argument("--summary", help="append a markdown report here (e.g. $GITHUB_STEP_SUMMARY)")

    p = sub.add_parser("digest", help="print the canonical sha256 of a spec")
    p.add_argument("spec")

    args = parser.parse_args(argv)
    if args.selftest:
        return _selftest()
    try:
        if args.command == "export":
            text = dumps(live_spec(args.app, args.root, args.prepare))
            if args.out == "-":
                sys.stdout.write(text)
            else:
                Path(args.out).write_text(text, encoding="utf-8")
                print(f"wrote {args.out}")
            return 0
        if args.command == "check":
            problems = check(load(args.spec), live_spec(args.app, args.root, args.prepare))
            if problems:
                print(f"{args.spec} → what {args.app} generates:", *problems, sep="\n  ", file=sys.stderr)
                print(
                    f"Regenerate it: python atrium_openapi.py export --app {args.app} --out {args.spec}",
                    file=sys.stderr,
                )
                return 1
            print(f"{args.spec} is current")
            return 0
        if args.command == "stamp":
            asset = stamp(load(args.spec), read_version(args.para_config))
            out = Path(args.out_dir)
            out.mkdir(parents=True, exist_ok=True)
            # The canonical form, not dumps(): the asset's sha256 is then the digest /info reports.
            (out / args.asset).write_bytes(canonical_bytes(asset))
            (out / f"{args.asset}.sha256").write_text(f"{digest(asset)}  {args.asset}\n", encoding="utf-8")
            print(f"wrote {out / args.asset} (version {asset['info']['version']}, sha256 {digest(asset)})")
            return 0
        if args.command == "baseline":
            token = sys.stdin.read().strip() if args.token_stdin else None
            tag, spec = baseline(args.repo, args.tag, token=token or None, asset=args.asset, api_url=args.api_url)
            if spec is None:
                print(f"no earlier release of {args.repo} carries {args.asset}: bootstrap")
                return 0
            Path(args.out).write_text(dumps(spec), encoding="utf-8")
            print(f"baseline: {tag} → {args.out}")
            return 0
        if args.command == "compare":
            base = load(args.base) if Path(args.base).is_file() else None
            code, lines = compare(
                base,
                load(args.rev),
                read_version(args.para_config),
                oasdiff=args.oasdiff,
                require_oasdiff=args.require_oasdiff,
            )
            print("\n".join(lines))
            _write_summary(args.summary, "OpenAPI compatibility (atrium-project#32)", lines)
            return code
        if args.command == "digest":
            print(digest(load(args.spec)))
            return 0
    except OpenAPIError as exc:
        print(f"atrium_openapi: {exc}", file=sys.stderr)
        return 2
    parser.print_help()
    return 2


# ── self-test (para-drift runs `python atrium_openapi.py --selftest` in every repo) ────────


def _selftest() -> int:
    def spec(
        version: str, props: Dict[str, Any], codes: Iterable[str] = ("busy",), op_id: str = "info"
    ) -> Dict[str, Any]:
        return {
            "openapi": "3.1.0",
            "info": {"title": "t", "version": version, "x-atrium-service": "atrium-test"},
            "paths": {
                "/info": {
                    "get": {
                        "operationId": op_id,
                        "responses": {
                            "200": {"content": {"application/json": {"schema": {"$ref": "#/components/schemas/I"}}}}
                        },
                    }
                }
            },
            "components": {"schemas": {"I": {"type": "object", "properties": props, "required": sorted(props)}}},
            "x-atrium-reason-codes": {code: {"description": code, "statuses": [429]} for code in codes},
        }

    base = spec("1.2.0", {"a": {"type": "string"}, "b": {"type": "string"}})
    assert digest(base) == digest(json.loads(json.dumps(base, indent=4))), "digest must ignore formatting"
    assert version_key("1.8.0-beta") < version_key("1.8.0") < version_key("v1.8.1-beta") < version_key("2.0.0")
    assert major("0.22.0") == 0 and major("v1.2.2-beta") == 1
    assert check(base, stamp(base, "9.9.9")) == [], "check must ignore info.version"
    assert "/components/schemas/I/properties/b: removed" in check(base, spec("1.2.0", {"a": {"type": "string"}}))

    same, _ = compare(base, stamp(base, "1.3.0"), "1.3.0")
    assert same == 0, "an identical spec is compatible"
    removed = spec("1.3.0", {"a": {"type": "string"}})
    code, lines = compare(base, removed, "1.3.0")
    assert code == 1 and any("property 'b' removed" in line for line in lines), lines
    assert compare(base, spec("2.0.0", {"a": {"type": "string"}}), "2.0.0")[0] == 0, "a major bump allows a break"
    lost_code = spec("2.0.0", {"a": {"type": "string"}, "b": {"type": "string"}}, codes=())
    assert compare(base, lost_code, "2.0.0")[0] == 1, "a removed reason code fails even with a major bump"
    renamed = spec("1.3.0", {"a": {"type": "string"}, "b": {"type": "string"}}, op_id="get_info")
    assert compare(base, renamed, "1.3.0")[0] == 1, "a removed operationId is breaking"
    assert compare(None, base, "1.2.0") == (
        0,
        [f"No baseline: no earlier release carries `{ASSET}`. This release is the bootstrap."],
    )

    moved = stamp(base, "1.3.0")
    moved["info"][SERVICE_KEY] = "atrium-renamed"
    assert compare(base, moved, "2.0.0")[0] == 1, "an undeclared rename fails even with a major bump"
    moved["info"][PREVIOUS_SERVICE_KEY] = "atrium-other"
    assert compare(base, moved, "2.0.0")[0] == 1, "a rename declaring another id fails"
    moved["info"][PREVIOUS_SERVICE_KEY] = "atrium-test"
    code, lines = compare(base, moved, "1.3.0")
    assert code == 0 and any("Declared rename" in line for line in lines), lines

    nullable = normalise(
        {"anyOf": [{"$ref": "#/x"}, {"type": "null"}], "components": {"schemas": {"AtriumDocument": {}}}}
    )
    assert nullable["oneOf"] == [{"$ref": "#/x"}, {"type": "null"}] and "anyOf" not in nullable
    assert nullable["components"]["schemas"]["AtriumDocument"] == {"type": "object"}

    def release(tag: str, asset: bool = True, **flags: Any) -> Dict[str, Any]:
        assets = [{"name": ASSET, "browser_download_url": f"https://example.invalid/{tag}"}] if asset else []
        return {"tag_name": tag, "draft": False, "prerelease": False, "assets": assets, **flags}

    listing = [
        release("v1.8.0-beta"),
        release("v1.9.0"),
        release("v1.9.1", asset=False),
        release("v1.10.0", draft=True),
        release("v2.0.0", prerelease=True),
        release("v1.7.5-beta"),
    ]
    assert select_baseline(listing, "v1.9.1")["tag_name"] == "v1.9.0"
    assert select_baseline(listing, "v1.9.0")["tag_name"] == "v1.8.0-beta"
    assert select_baseline(listing, None)["tag_name"] == "v1.9.0"
    assert select_baseline(listing, "v1.7.5-beta") is None

    print("atrium_openapi selftest: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
