"""The schema freeze, checked in every repository: this copy of the schema against `doc-schema-v1`.

WHY THIS EXISTS. atrium-project#54 froze `atrium_document.schema.json` at 1.0 and cut the tag
`doc-schema-v1` for it, separate from the moving `v1` tag the tool repos vendor from. A tag on its
own checks nothing. The canonical file kept changing after it, which is allowed for an additive
change (versioning rule 1; the first one was llm-enrich#18's `lines[].style.region`), but nothing
said which changes those were, and no repository could show that the schema it carries still
honours the frozen one. "Frozen" was a claim in a changelog.

This file is vendored with the schema (hub: docs/templates/shared/; tool repos: tests/, with the
schema and its frozen copy at the repo root), so every repository checks its OWN copy:

  * the frozen copy `atrium_document.schema.doc-schema-v1.json` is byte-for-byte the file the tag
    names -- pinned by git blob id, the id `git rev-parse doc-schema-v1:<path>` prints, so it
    needs no git and no tags;
  * nothing declared at the freeze was removed or renamed (versioning rule 2 -- that is a MAJOR);
  * every constraining difference since the freeze is in `POST_FREEZE_CHANGES`, pinned to the node
    it adds. Descriptions, `$comment`s and examples are free.

Read it as the answer to "may I change the schema?". An additive change -- a new optional field or
block, or a narrowing that still admits every shape in the hub's tests/test_document_required.py --
takes no new tag and no `SCHEMA_VERSION` bump: register it here, write its
`## Changelog — <date>` in the hub's docs/document_schema.md, add its producer shapes, then
re-vendor (scripts/revendor_shared.sh) and move `v1`. Removing or renaming a declared property is a
`2.0`, with its own tag `doc-schema-v2`, frozen copy and `FREEZES[2]` entry. The procedure is in
docs/document_schema.md, "Freeze & conformance".

What needs the hub -- the tag itself, the changelog sections, and validating every producer's
record shape under both schemas (jsonschema) -- is the hub's tests/test_schema_freeze_records.py.

The imports below are plain top-level imports, as in test_document_originators.py: pytest puts
this file's directory on sys.path in the hub, and pytest.ini's `pythonpath = .` does it in a tool
repo.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, Optional

import pytest

import atrium_document
from atrium_document import SCHEMA_VERSION, load_schema

#: One entry per schema MAJOR. A freeze tag is cut once per MAJOR and is never moved or deleted.
#: `blob` is `git rev-parse <tag>:docs/templates/shared/atrium_document.schema.json`; `snapshot` is
#: the frozen copy's filename, vendored beside atrium_document.py exactly like the live schema.
FREEZES: Dict[int, Dict[str, str]] = {
    1: {
        "tag": "doc-schema-v1",
        "commit": "544298bb5817fe57d8521579700332184b86d778",
        "blob": "1945c43ea5a4b6e370022fba75fe1d835b32b34c",
        "snapshot": "atrium_document.schema.doc-schema-v1.json",
    },
}

#: Every structural change to the schema since the freeze of the current MAJOR, keyed by the JSON
#: pointer it lives under. `schema` is the node at that pointer NOW, annotations left out -- so the
#: register reads as the post-freeze contract, and a later edit under a registered pointer (a
#: widened enum, say) has to edit its entry too. `changelog` is the date of its
#: `## Changelog — <date>` section in the hub's docs/document_schema.md. An entry that matches no
#: difference fails, so the register is exactly as long as the diff.
POST_FREEZE_CHANGES: Dict[str, Dict[str, Any]] = {
    "/properties/lines/items/properties/style/properties/region": {
        "kind": "added: a closed enum; absent means body text",
        "schema": {"type": "string", "enum": ["page_header", "page_footer", "footnote"]},
        "issue": "ufal/atrium-llm-enrich#18",
        "changelog": "2026-09-25",
    },
}

#: Keywords that annotate and do not constrain. Changing them is not a schema change.
_ANNOTATIONS = frozenset({"description", "$comment", "examples", "title", "default"})

#: Keywords compared whole, so an added enum value is one changed pointer rather than every later
#: index shifting. The comparison stays exact: reordering an enum is still a difference.
_ATOMIC = frozenset({"enum", "required", "type"})

#: Keywords whose object keys are NAMES (of properties, of definitions), not keywords -- so a
#: property called `title` or `description` is still a property.
_NAMED = frozenset({"properties", "$defs", "patternProperties", "dependentSchemas"})


def current_freeze() -> Dict[str, str]:
    major = int(SCHEMA_VERSION.split(".")[0])
    if major not in FREEZES:
        pytest.fail(f"no freeze recorded for schema MAJOR {major} -- see test_the_current_major_has_a_freeze")
    return FREEZES[major]


def snapshot_path(freeze: Dict[str, str]) -> str:
    """The frozen copy sits beside whichever atrium_document.py is imported, like the live schema
    (atrium_document.schema_path()): docs/templates/shared/ in the hub, the repo root elsewhere."""
    return os.path.join(os.path.dirname(os.path.abspath(atrium_document.__file__)), freeze["snapshot"])


def load_snapshot(freeze: Dict[str, str]) -> Dict[str, Any]:
    path = snapshot_path(freeze)
    if not os.path.exists(path):
        pytest.fail(
            f"{freeze['snapshot']} not found beside atrium_document.py -- re-vendor (scripts/revendor_shared.sh)"
        )
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def git_blob_id(data: bytes) -> str:
    """The id `git hash-object` gives these bytes -- computed, so no git is needed."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _esc(key: str) -> str:
    return key.replace("~", "~0").replace("/", "~1")


def flatten(node: Any, ptr: str = "", out: Optional[Dict[str, str]] = None, names: bool = False) -> Dict[str, str]:
    """`{json_pointer: canonical json}` for every constraining leaf of a schema.

    Annotations are dropped, `_ATOMIC` keywords are kept whole, and everything else is walked down
    to its scalars, so one edited `minimum` is one pointer.
    """
    if out is None:
        out = {}
    if isinstance(node, dict):
        if not node:
            out[ptr] = "{}"
        for key, value in node.items():
            if not names and key in _ANNOTATIONS:
                continue
            p = f"{ptr}/{_esc(key)}"
            if (not names and key in _ATOMIC) or not isinstance(value, (dict, list)):
                out[p] = json.dumps(value, sort_keys=True, ensure_ascii=False)
            else:
                flatten(value, p, out, names=not names and key in _NAMED)
    elif isinstance(node, list):
        if not node:
            out[ptr] = "[]"
        for i, value in enumerate(node):
            if isinstance(value, (dict, list)):
                flatten(value, f"{ptr}/{i}", out)
            else:
                out[f"{ptr}/{i}"] = json.dumps(value, sort_keys=True, ensure_ascii=False)
    return out


def declared(node: Any, ptr: str = "", out: Optional[set] = None) -> set:
    """Pointers of every declared property and definition, e.g. `/properties/lines/items/properties/style`."""
    if out is None:
        out = set()
    if isinstance(node, dict):
        for key, value in node.items():
            p = f"{ptr}/{_esc(key)}"
            if key in _NAMED and isinstance(value, dict):
                for name, sub in value.items():
                    out.add(f"{p}/{_esc(name)}")
                    declared(sub, f"{p}/{_esc(name)}", out)
            else:
                declared(value, p, out)
    elif isinstance(node, list):
        for i, value in enumerate(node):
            declared(value, f"{ptr}/{i}", out)
    return out


def differences(frozen: Dict[str, Any], current: Dict[str, Any]) -> set:
    a, b = flatten(frozen), flatten(current)
    return {p for p in a.keys() | b.keys() if a.get(p) != b.get(p)}


def resolve(schema: Any, pointer: str) -> Any:
    node = schema
    for token in pointer.split("/")[1:]:
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(node, list):
            node = node[int(token)] if token.isdigit() and int(token) < len(node) else None
        elif isinstance(node, dict):
            node = node.get(token)
        else:
            node = None
        if node is None:
            return None
    return node


def _under(pointer: str, prefix: str) -> bool:
    return pointer == prefix or pointer.startswith(prefix + "/")


# ── the frozen copy is the tag ────────────────────────────────────────────────


def test_the_current_major_has_a_freeze():
    major = int(SCHEMA_VERSION.split(".")[0])
    assert major in FREEZES, (
        f"SCHEMA_VERSION is {SCHEMA_VERSION} and no freeze is recorded for MAJOR {major}. A MAJOR "
        f"bump ends with: re-vendor, move v1, cut doc-schema-v{major} on the hub commit that "
        f"carries the bump, add atrium_document.schema.doc-schema-v{major}.json to the shared "
        f"files, and FREEZES[{major}] here (docs/document_schema.md, 'Freeze & conformance')"
    )


def test_the_frozen_copy_is_the_tagged_blob():
    """The vendored frozen copy is byte-for-byte the file the freeze tag names."""
    freeze = current_freeze()
    load_snapshot(freeze)  # a clear failure when it was not vendored
    with open(snapshot_path(freeze), "rb") as fh:
        got = git_blob_id(fh.read())
    assert got == freeze["blob"], (
        f"{freeze['snapshot']} hashes to {got}, not the blob {freeze['tag']} names "
        f"({freeze['blob']}). A frozen copy is never edited -- restore it from the hub"
    )


# ── what may change after the freeze ──────────────────────────────────────────


def test_nothing_declared_at_the_freeze_was_removed_or_renamed():
    """Versioning rule 2: removing or renaming a field is a MAJOR bump. There is no register
    escape hatch for this one -- a record that carries the old name must stay describable."""
    freeze = current_freeze()
    missing = sorted(declared(load_snapshot(freeze)) - declared(load_schema()))
    assert not missing, (
        f"declared at {freeze['tag']} and gone now: {missing}. That is a breaking change "
        f"(versioning rule 2) -- restore it, or make it the MAJOR bump it is"
    )


def test_every_change_since_the_freeze_is_registered():
    """Every constraining difference falls under a POST_FREEZE_CHANGES entry, and every entry
    still describes one."""
    freeze = current_freeze()
    diff = differences(load_snapshot(freeze), load_schema())
    unregistered = sorted(p for p in diff if not any(_under(p, prefix) for prefix in POST_FREEZE_CHANGES))
    assert not unregistered, (
        f"the schema differs from {freeze['tag']} at {unregistered}, and no POST_FREEZE_CHANGES "
        f"entry covers it. If the change is additive, register it with its issue and changelog date"
    )
    stale = sorted(prefix for prefix in POST_FREEZE_CHANGES if not any(_under(p, prefix) for p in diff))
    assert not stale, f"POST_FREEZE_CHANGES entries that match no difference any more: {stale}"


@pytest.mark.parametrize("prefix", sorted(POST_FREEZE_CHANGES))
def test_each_registered_change_is_exactly_what_the_register_says(prefix):
    """The register pins the node, not just the pointer: widening `region`'s enum later is a new
    post-freeze change, and has to show up here and in a changelog, not slip in under the entry
    that declared it."""
    node = resolve(load_schema(), prefix)
    assert node is not None, f"{prefix} is registered but absent from the schema"
    assert flatten(node) == flatten(POST_FREEZE_CHANGES[prefix]["schema"]), (
        f"the schema at {prefix} is no longer what POST_FREEZE_CHANGES records -- update the "
        f"entry, and its changelog date, if the new change is additive"
    )
