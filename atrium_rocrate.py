"""
atrium_rocrate.py — the RO-Crate view of an ATRIUM document record, and the run's CreateAction.

WHY THIS MODULE EXISTS
======================
``atrium_document.py`` produces a per-document record that ``docs/document_schema.md`` calls
"one FAIR, versioned JSON for search and catalogue export". The DMP's WP3 standards column names
**RO-Crate** (ufal/atrium-project#54), and the AMČR pilot stores every archived record as an
RO-Crate with one Process Run Crate ``CreateAction`` per run (atrium-project#71; grounding report
of 2026-09-30, items 1-3). Everything such a crate needs, ATRIUM already records:

    source            -> the original input, by doc_id + sha256/sha512 (archive-managed, never a path)
    derived_from      -> the persistent step outputs = the crate's data entities
    regenerable       -> recipes for files that DO NOT EXIST = never data entities
    provenance        -> the accreted licence union + one entry per contributing run
    assembled.blocks  -> per-block stamps = the granularity a CreateAction chain needs
    atrium_paradata   -> tool_version / repository / docker_image / run_uuid / times per run
    atrium_vocab      -> resolvable URIs for every controlled term the record carries

So this module maps rather than invents. Where a field does not exist it is **omitted**, not
guessed: a crate that quietly asserts a checksum nobody computed is worse than one that admits the
gap.

WHAT IT PRODUCES
================
* ``document_crate(record)`` — the crate of one record: RO-Crate 1.2, Process Run Crate 0.5.
  With ``fragment=True`` the same entities WITHOUT the metadata descriptor and the root, for
  embedding in a crate someone else owns (AMČR's record crate, aiscr-webamcr#4316).
* ``run_crate(records)`` — one pipeline run over several records, referencing their crates.
* ``create_action(paradata)`` — one run as a nested JSON-LD ``CreateAction``: what every service
  returns as ``paradata`` and what AMČR stores as the run's paradata file.
* ``wrap_fragment(fragment)`` — a fragment under a stub root, which is how a fragment is validated.

WHAT THIS MODULE IS NOT
=======================
* **Not a publisher.** It builds a metadata graph and can write it next to a directory of
  outputs. Nothing is uploaded, registered, or minted.
* **Not a second provenance model.** ``atrium_paradata.py`` stays the record of how a run
  behaved; this is a *view* over what is already recorded, built after the fact.
* **Not a writer of document records.** It never mutates one.

DEPENDENCIES
============
Standard library only. An RDF library is deliberately **not** introduced, for the same reason
``atrium_vocab.py`` refuses one: the serialisation here is small, closed and fully determined by
its input, and the output is drift-gated by byte comparison. The conformance check is the
RO-Crate validator, run in the hub's CI (``tools/ci/rocrate_check.py``), never a dependency here.

``atrium_vocab`` is imported softly — with it, every controlled term in the record becomes a
``DefinedTerm`` with a resolvable ``https://w3id.org/atrium/`` URI; without it the crate is still
valid, just less semantic.

DETERMINISM
===========
Two calls on the same record produce byte-identical JSON. ``@graph`` is sorted by ``@id`` (with
the descriptor and the root pinned first) and ``json.dumps`` runs with ``sort_keys=True``.
``datePublished`` is derived from the record's own newest block stamp, never from the clock, and a
run's identifier is the ``run_uuid`` the record carries — nothing is minted here.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

try:
    import atrium_vocab
except ImportError:  # pragma: no cover - degraded mode, see module docstring
    atrium_vocab = None  # type: ignore

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

#: RO-Crate specification this module targets. The three strings move together, and moving them
#: is a deliberate act: `conformsTo` is what a consumer reads to decide how to interpret the
#: graph, so a bump is a compatibility statement about the crates already written, exactly like
#: SCHEMA_VERSION in atrium_document.py. 1.2 is the version agreed with AMČR (atrium-project#71):
#: the newest one the RO-Crate validator checks together with Process Run Crate 0.5.
ROCRATE_VERSION = "1.2"
ROCRATE_CONTEXT = "https://w3id.org/ro/crate/1.2/context"
ROCRATE_CONFORMS_TO = "https://w3id.org/ro/crate/1.2"

#: The Process Run Crate profile both crate kinds declare on their ROOT, as RO-Crate 1.2 places
#: profiles: each describes tool runs (ATRIUM's stages are separate containers chained by an
#: orchestrator, a *process* run, not a CWL/WDL workflow). Pinned to the version roc-validator
#: 0.12.0 ships (`process-run-crate`, IRI below); move it with ROCRATE_VERSION, to 0.6 with 1.3,
#: when the validator checks that pair.
PROCESS_RUN_PROFILE = "https://w3id.org/ro/wfrun/process/0.5"
PROCESS_RUN_PROFILE_NAME = "Process Run Crate"
PROCESS_RUN_PROFILE_VERSION = "0.5"

#: The Workflow Run terms namespace (`ContainerImage`, `containerImage`, `registry`, `tag`).
WFRUN_BASE = "https://w3id.org/ro/terms/workflow-run#"
DOCKER_IMAGE_TYPE = WFRUN_BASE + "DockerImage"

#: `actionStatus` values: literal schema.org IRIs, as the Process Run Crate profile checks them.
COMPLETED_ACTION_STATUS = "http://schema.org/CompletedActionStatus"
FAILED_ACTION_STATUS = "http://schema.org/FailedActionStatus"

#: The file every RO-Crate is identified by. Not configurable — it is the spec.
METADATA_FILENAME = "ro-crate-metadata.json"

#: The root data entity's `@id` is `./` by RO-Crate rule, never a name of our choosing. A fragment
#: has no root of its own: the crate that embeds it keeps its own `./`.
ROOT_ID = "./"

#: The original input's entity in a crate built from a record, unless the host names its own.
SOURCE_ID = "#source"

#: The document record as an input of a service call (`create_action`).
RECORD_ID = "#record"

#: ATRIUM's own namespace, so extension terms carry a resolvable prefix rather than being
#: dropped by a consumer that compacts strictly against the RO-Crate context. Kept in step
#: with atrium_vocab.SKOS_BASE, which is the one place an ATRIUM URI is rooted.
ATRIUM_BASE = getattr(atrium_vocab, "SKOS_BASE", "https://w3id.org/atrium/")

#: The ATRIUM tool authors, identical in every tool repo's CITATION.cff. Since atrium-project#71
#: they are each tool's `creator` — the people who made the software — and no longer the crate
#: root's `author`: the record is the archive's, the tools are theirs. ORCIDs are the `@id`s, the
#: RO-Crate convention, so no ATRIUM identifier is minted for a person.
AUTHORS: Tuple[Dict[str, str], ...] = (
    {"orcid": "https://orcid.org/0009-0002-4773-2797", "name": "Kateryna Lutsai"},
    {"orcid": "https://orcid.org/0000-0002-6895-8536", "name": "Pavel Straňák"},
    {"orcid": "https://orcid.org/0009-0005-8722-0245", "name": "David Novák"},
    {"orcid": "https://orcid.org/0000-0001-5718-9447", "name": "Dana Křivánková"},
)

#: program -> repository, mirroring atrium_paradata._REPO_URLS. `digital-convert` is a ROLE
#: that lives in atrium-llm-enrich (see the BLOCK_OWNERS note in atrium_document.py), so it
#: maps to that repo rather than one of its own. A service's logger names its program
#: `<program>-api`; `_repository()` strips that suffix before looking it up here.
REPO_URLS: Dict[str, str] = {
    "alto-postprocess": "https://github.com/ufal/atrium-alto-postprocess",
    "page-classification": "https://github.com/ufal/atrium-page-classification",
    "translator": "https://github.com/ufal/atrium-translator",
    "nlp-enrich": "https://github.com/ufal/atrium-nlp-enrich",
    "llm-enrich": "https://github.com/ufal/atrium-llm-enrich",
    "digital-convert": "https://github.com/ufal/atrium-llm-enrich",
}

#: Where each controlled term in the record lives in the atrium_vocab registry: the record
#: path that carries it, and the scheme that governs it. Kept as an explicit table rather
#: than derived from the schema's `x-atrium-scheme` annotations, because this module must
#: work when only the .py files were vendored and the schema JSON was not.
CONTROLLED_TERMS: Tuple[Tuple[str, str], ...] = (
    ("page_categories.*", "page-category"),
    ("pages[].category", "page-category"),
    ("pages[].quality_band", "quality-band"),
    ("lines[].categ", "line-category"),
    ("entities[].type_teitok", "entity-type"),
    ("entities[].type_cnec", "cnec"),
    ("enrichment.items[].teater_category", "theme"),
)

#: Terms this module emits that the RO-Crate 1.2 context does not define, declared in the
#: crate's own `@context` so the graph stays interpretable without a side agreement. The
#: Workflow Run terms are declared here rather than through their remote context, so a crate
#: depends on exactly one remote document. `paradataRecord` is a JSON literal (JSON-LD 1.1): the
#: run's whole paradata record, kept as it is rather than flattened into undeclared properties.
_LOCAL_CONTEXT: Dict[str, Any] = {
    "@version": 1.1,
    "atrium": ATRIUM_BASE,
    "regenerableFrom": ATRIUM_BASE + "term/regenerableFrom",
    "regenerableWith": ATRIUM_BASE + "term/regenerableWith",
    "recordBlock": ATRIUM_BASE + "term/recordBlock",
    "sha512": ATRIUM_BASE + "term/sha512",
    "paradataRecord": {"@id": ATRIUM_BASE + "term/paradataRecord", "@type": "@json"},
    "ContainerImage": WFRUN_BASE + "ContainerImage",
    "containerImage": WFRUN_BASE + "containerImage",
    "registry": WFRUN_BASE + "registry",
    "tag": WFRUN_BASE + "tag",
}

#: What the Process Run Crate profile accepts as `startTime`/`endTime` (seconds, at most
#: milliseconds, and a numeric offset). `_timestamp()` writes every time in this shape.
_ACTION_TIME = re.compile(r"^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}:\d{2}(\.\d{3})?\+\d{2}:\d{2})?$")

#: A tool's `version` when no paradata says which release ran. RO-Crate 1.2 makes `version` a
#: MUST on every SoftwareApplication, so the gap is stated rather than left out; a service's own
#: action never carries it (`action_problems()`), since a service always knows its release.
UNRECORDED_VERSION = "unrecorded"

#: What a run's stable id looks like (atrium_paradata `run_uuid`).
_RUN_UUID = re.compile(r"^urn:uuid:[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")

#: Process Run Crate's types for an action's `object` and `result`. `File` is deliberately not
#: one of them here: in RO-Crate 1.2 a `File` is a data entity the root must list in `hasPart`,
#: and the inputs and outputs of a service call are not files of the crate that embeds them.
_OBJECT_TYPES = ("CreativeWork", "Dataset", "MediaObject", "Collection", "PropertyValue", "File")

Paradata = Union[Mapping[str, Mapping[str, Any]], Sequence[Mapping[str, Any]], None]


# ──────────────────────────────────────────────────────────────────────────────
# Small helpers
# ──────────────────────────────────────────────────────────────────────────────


def _ref(entity_id: str) -> Dict[str, str]:
    """A JSON-LD reference. RO-Crate requires `{"@id": …}`, never a bare string."""
    return {"@id": entity_id}


def _refs(entity_ids: Iterable[str]) -> List[Dict[str, str]]:
    return [_ref(i) for i in entity_ids]


def _prune(entity: Dict[str, Any]) -> Dict[str, Any]:
    """Drop empty values.

    A crate must not assert what the record does not say. An absent `docker_image` becomes a
    missing property, not `""` — the two are different claims, and only one of them is true.
    """
    return {k: v for k, v in entity.items() if v not in (None, "", [], {})}


def _context() -> List[Any]:
    return [ROCRATE_CONTEXT, dict(sorted(_LOCAL_CONTEXT.items()))]


def _version(value: Any) -> str:
    """A release version as the tag names it, without the `v`; "" when unknown."""
    text = str(value or "").strip()
    if text.lower() in ("", "unknown", "none"):
        return ""
    return text[1:] if text[:1] in ("v", "V") else text


def _timestamp(value: Any) -> Optional[str]:
    """An ISO 8601 time as the Process Run Crate profile accepts it: UTC, whole seconds.

    Paradata and stamps carry `datetime.isoformat()`, with microseconds, which the profile's
    pattern refuses. A value that does not parse is omitted rather than passed through: a time
    the validator rejects says no more than a missing one.
    """
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def _date_published(record: Dict[str, Any]) -> str:
    """The record's own newest timestamp, as a date — never `datetime.now()`.

    RO-Crate requires `datePublished` on the root. Reading the clock would make the crate
    non-reproducible: exporting an archived record twice would produce two different crates,
    and the byte-comparison gate this module is meant to pass would fail on every run. The
    newest block stamp is the honest answer to "when was this record finished".
    """
    stamps = [
        str(b.get("updated_at") or "")
        for b in ((record.get("assembled") or {}).get("blocks") or {}).values()
        if isinstance(b, dict)
    ]
    stamps += [str(c.get("at") or "") for c in ((record.get("provenance") or {}).get("contributors") or [])]
    newest = max((s for s in stamps if s), default="")
    return newest[:10] if len(newest) >= 10 else "1970-01-01"


def _license_entity(provenance: Dict[str, Any]) -> Tuple[Optional[Dict[str, str]], List[Dict[str, Any]]]:
    """The root's `license`, plus the contextual entity it points at.

    `para_licenses.merge_effective_licenses()` has already done the hard part — the
    most-restrictive union over every contributing tool — so this only has to render it. A
    licence with a URL becomes a referenced entity (dereferenceable, which is the point); one
    without stays a literal, because inventing a URL for it would be a claim.
    """
    name = str(provenance.get("license") or "")
    url = str(provenance.get("license_url") or "")
    if url:
        entity = _prune(
            {
                "@id": url,
                "@type": "CreativeWork",
                "name": name,
                "identifier": name,
                "description": "The effective licence: the most restrictive of every contributing tool's.",
            }
        )
        return _ref(url), [entity]
    return ({"@value": name} if name else None), []


def _controlled_values(record: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Every (scheme, value) pair the record actually carries, deduplicated and sorted.

    Deliberately tolerant: a value the registry has never heard of is still emitted as a
    DefinedTerm under its scheme. `atrium_vocab.validate_labels()` reports rather than raises,
    and `atrium_document.schema.json` adds no `enum` anywhere precisely so a naming slip is an
    advisory rather than a stalled pipeline — a crate exporter must not be stricter than the
    contract it exports.
    """
    found = set()

    def add(scheme: str, value: Any) -> None:
        if isinstance(value, str) and value:
            found.add((scheme, value))

    for value in (record.get("page_categories") or {}).values():
        add("page-category", value)
    for page in record.get("pages") or []:
        if isinstance(page, dict):
            add("page-category", page.get("category"))
            add("quality-band", page.get("quality_band"))
    for line in record.get("lines") or []:
        if isinstance(line, dict):
            add("line-category", line.get("categ"))
    for ent in record.get("entities") or []:
        if isinstance(ent, dict):
            add("entity-type", ent.get("type_teitok"))
            add("cnec", ent.get("type_cnec"))
    for item in (record.get("enrichment") or {}).get("items") or []:
        if isinstance(item, dict):
            add("theme", item.get("teater_category"))
    return sorted(found)


def _concept_uri(scheme: str, value: str) -> str:
    """A term's URI, via atrium_vocab when it was vendored alongside this module.

    The fallback reproduces `concept_uri()`'s shape rather than omitting the term, so a crate
    built in degraded mode is still joinable with one built with the registry present. It does
    NOT reproduce `_slug()`, so a value containing a space differs between the two — which is
    why the registry is the supported path and this is only a fallback.
    """
    if atrium_vocab is not None:
        return atrium_vocab.concept_uri(scheme, value)
    return "{}{}/{}".format(ATRIUM_BASE, scheme, value)


def _scheme_uri(scheme: str) -> str:
    if atrium_vocab is not None:
        return atrium_vocab.scheme_uri(scheme)
    return "{}scheme/{}".format(ATRIUM_BASE, scheme)


def _legacy_run_id(program: str, run_id: str) -> str:
    """The run id of a record written before `run_uuid` existed (atrium-project#71).

    `run_id` has one-second resolution, so two parallel runs of one program can share it; the
    `run_uuid` every run now carries cannot collide. This form stays only so older records keep
    exporting to the same crate they always did.
    """
    return "#run-{}-{}".format(program, run_id or "unknown")


def _run_action_id(program: str, run_id: str, run_uuid: str) -> str:
    return run_uuid if _RUN_UUID.match(run_uuid or "") else _legacy_run_id(program, run_id)


def _block_id(name: str) -> str:
    return "#block-" + name


def _identifier_entity(entity_id: str, name: str, value: str) -> Dict[str, Any]:
    """A root identifier as a `PropertyValue`, the form RO-Crate 1.2 recommends for it."""
    return {"@id": entity_id, "@type": "PropertyValue", "name": name, "propertyID": name, "value": value}


def _profile_entity() -> Dict[str, Any]:
    """The profile a root's `conformsTo` names: RO-Crate 1.2 requires it to be a `Profile` entity."""
    return {
        "@id": PROCESS_RUN_PROFILE,
        "@type": ["CreativeWork", "Profile"],
        "name": PROCESS_RUN_PROFILE_NAME,
        "version": PROCESS_RUN_PROFILE_VERSION,
    }


def _person_entities() -> List[Dict[str, Any]]:
    return [{"@id": a["orcid"], "@type": "Person", "name": a["name"]} for a in AUTHORS]


def ni_uri(sha256_hex: str) -> str:
    """The RFC 6920 named-information URI of a SHA-256 digest: `ni:///sha-256;<base64url>`.

    A content-addressed id: the same bytes get the same id in every action, whichever service
    read or wrote them, so the output of one stage and the input of the next join by themselves.
    """
    digest = bytes.fromhex(sha256_hex)
    return "ni:///sha-256;" + base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


# ──────────────────────────────────────────────────────────────────────────────
# Tools, images, agents
# ──────────────────────────────────────────────────────────────────────────────


def _repository(program: str, paradata: Mapping[str, Any]) -> str:
    """The tool's repository: the paradata's own (ATRIUM_RUNNER_REPO or para_config), else the
    table's. The paradata's bare organisation fallback (`https://github.com/ufal`) names no tool,
    so it loses to the table."""
    recorded = str(paradata.get("repository") or "").rstrip("/")
    if recorded.startswith("http") and recorded.count("/") > 3:
        return recorded
    base = program[: -len("-api")] if program.endswith("-api") else program
    return REPO_URLS.get(program) or REPO_URLS.get(base) or ""


def _tool_id(program: str, repository: str, version: str) -> str:
    """A tool's stable, absolute id: its release page when the version is known, else its
    repository. A program no table knows keeps a local id rather than an invented URL."""
    if repository and version:
        return "{}/releases/tag/v{}".format(repository, version)
    return repository or "#tool-" + program


def _tool_entity(program: str, paradata: Mapping[str, Any]) -> Dict[str, Any]:
    """The `SoftwareApplication` of one run: `version`, never `softwareVersion` (atrium-project#71),
    and the ATRIUM authors as its `creator`. Without paradata the version is `UNRECORDED_VERSION`
    and the id is the repository, not a release."""
    repository = _repository(program, paradata)
    version = _version(paradata.get("tool_version"))
    python = str(paradata.get("python_version") or "").split(" ")[0]
    return _prune(
        {
            "@id": _tool_id(program, repository, version),
            "@type": "SoftwareApplication",
            "name": repository.rsplit("/", 1)[-1] if repository else program,
            "alternateName": program,
            "url": repository,
            "version": version or UNRECORDED_VERSION,
            "creator": _refs(a["orcid"] for a in AUTHORS),
            "runtimePlatform": "Python " + python if python else None,
        }
    )


def _container_image(reference: str) -> Optional[Dict[str, Any]]:
    """A Workflow Run `ContainerImage` from an image reference such as
    `ghcr.io/ufal/atrium-translator-api:1.3.0-beta` (ATRIUM_RUNNER_IMAGE, paradata `docker_image`).

    The reference is recorded as the service gave it: the tag the registry carries
    (atrium-project#69 B3), and a digest when the reference names one.
    """
    reference = str(reference or "").strip()
    if not reference:
        return None
    name, _, digest = reference.partition("@")
    tag = ""
    if ":" in name.rsplit("/", 1)[-1]:
        name, tag = name.rsplit(":", 1)
    first, _, rest = name.partition("/")
    if rest and ("." in first or ":" in first or first == "localhost"):
        registry, path = first, rest
    else:
        registry, path = "docker.io", name
    sha256 = digest.split(":", 1)[1] if digest.startswith("sha256:") else ""
    return _prune(
        {
            "@id": "#image:" + reference,
            # The schema.org type beside the Workflow Run one is what RO-Crate 1.2 asks every
            # entity to carry; CreativeWork, not SoftwareApplication, so the image is not taken
            # for a second tool.
            "@type": ["ContainerImage", "CreativeWork"],
            "additionalType": _ref(DOCKER_IMAGE_TYPE),
            "name": path,
            "registry": registry,
            "tag": tag,
            "sha256": sha256,
        }
    )


def _agent_entity(agent: str) -> Optional[Dict[str, Any]]:
    """The operating organisation (ATRIUM_RUN_AGENT, an IRI), never invented and never the
    processing account that only reads and calls back (grounding report §8)."""
    agent = str(agent or "").strip()
    if not agent:
        return None
    return {"@id": agent, "@type": "Organization", "name": agent, "url": agent}


def _paradata_records(paradata: Paradata) -> List[Dict[str, Any]]:
    """Paradata as a list of records, from either accepted shape.

    A list of records is the current form. The `{program: record}` map is the form before
    atrium-project#71 and stays accepted: each record gains its key as `program` when it has none.
    """
    if not paradata:
        return []
    if isinstance(paradata, Mapping):
        return [dict(rec, program=rec.get("program") or prog) for prog, rec in paradata.items() if rec]
    return [dict(rec) for rec in paradata if rec]


def _match_paradata(records: Sequence[Dict[str, Any]], program: str, run_id: str, run_uuid: str) -> Dict[str, Any]:
    """The paradata of one run: by `run_uuid`, else by program and `run_id`, else by program."""
    if run_uuid:
        for rec in records:
            if rec.get("run_uuid") == run_uuid:
                return rec
    for rec in records:
        if rec.get("program") == program and run_id and rec.get("run_id") == run_id:
            return rec
    for rec in records:
        if rec.get("program") == program:
            return rec
    return {}


# ──────────────────────────────────────────────────────────────────────────────
# One run, as a CreateAction (atrium-project#71 R2)
# ──────────────────────────────────────────────────────────────────────────────


def file_entity(
    name: str,
    data: Optional[bytes] = None,
    *,
    media_type: Optional[str] = None,
    sha256: Optional[str] = None,
    size: Optional[int] = None,
) -> Dict[str, Any]:
    """An input or output of a service call, for `create_action(inputs=…, outputs=…)`.

    With the bytes (or their sha256) the id is the content-addressed `ni:` URI, so the result of
    one stage and the object of the next are one entity. A `CreativeWork`, not a `File`: the
    call's files are not data entities of whichever crate embeds the action.
    """
    if data is not None:
        sha256 = hashlib.sha256(data).hexdigest()
        size = len(data)
    entity_id = ni_uri(sha256) if sha256 else "#file-" + re.sub(r"[^A-Za-z0-9._-]", "_", name or "unnamed")
    return _prune(
        {
            "@id": entity_id,
            "@type": "CreativeWork",
            "name": name,
            "encodingFormat": media_type,
            "contentSize": str(size) if size is not None else None,
            "sha256": sha256,
        }
    )


def record_entity(doc_id: str) -> Dict[str, Any]:
    """The document record a call was handed (`#record`): the seed, or the previous stage's record."""
    return {
        "@id": RECORD_ID,
        "@type": "CreativeWork",
        "name": "{}.document.json".format(doc_id),
        "identifier": doc_id,
        "encodingFormat": "application/json",
    }


def block_entities(names: Iterable[str]) -> List[Dict[str, Any]]:
    """The record blocks a call wrote, as the same `#block-<name>` entities `document_crate` uses."""
    return [
        {"@id": _block_id(n), "@type": "CreativeWork", "name": n, "recordBlock": n} for n in sorted(set(names))
    ]


def blocks_written(record: Optional[Dict[str, Any]], run_uuid: str = "", run_id: str = "", program: str = "") -> List[str]:
    """The blocks of `record` whose stamp names this run: by `run_uuid`, else by `run_id` + `program`."""
    stamps = ((record or {}).get("assembled") or {}).get("blocks") or {}
    out = []
    for name, stamp in stamps.items():
        if not isinstance(stamp, dict):
            continue
        if run_uuid and stamp.get("run_uuid") == run_uuid:
            out.append(name)
        elif not run_uuid and run_id and stamp.get("run_id") == run_id and stamp.get("program") == program:
            out.append(name)
    return sorted(out)


def create_action(
    paradata: Mapping[str, Any],
    *,
    inputs: Sequence[Mapping[str, Any]] = (),
    outputs: Sequence[Mapping[str, Any]] = (),
    status: str = "completed",
    error: Optional[str] = None,
    action_id: Optional[str] = None,
) -> Dict[str, Any]:
    """One run as a nested JSON-LD Process Run Crate `CreateAction` (atrium-project#71 R2).

    `paradata` is a finished `ParadataLogger.record` (or a merged pipeline-run record). The
    action is what every service returns as `paradata`, and what AMČR stores as the run's
    paradata file; `action_fragment()` flattens it for a crate. It carries:

    * `@id` — the run's `run_uuid`, the same value stamped into the record it wrote
      (`action_id` overrides it where the record's stamp comes from another logger);
    * `instrument` — the tool (`version`, release-page id, its authors as `creator`) and
      `containerImage` — the image the service runs as (ATRIUM_RUNNER_IMAGE);
    * `object` / `result` — `inputs` / `outputs` (`file_entity()`, `record_entity()`,
      `block_entities()`);
    * `agent` — ATRIUM_RUN_AGENT, only when it is set;
    * `startTime` / `endTime`, `actionStatus` (`completed` or `failed`, with `error`);
    * `paradataRecord` — the paradata record itself, so nothing it says is lost.
    """
    if status not in ("completed", "failed"):
        raise ValueError("status must be 'completed' or 'failed', not {!r}".format(status))
    pd = dict(paradata or {})
    program = str(pd.get("program") or pd.get("pipeline") or "unknown")
    tool = _tool_entity(program, pd)
    tool["creator"] = _person_entities()
    run_id = str(pd.get("run_id") or "")
    failed = status == "failed"
    version = tool.get("version") if tool.get("version") != UNRECORDED_VERSION else ""
    action = {
        "@context": _context(),
        "@id": action_id or _run_action_id(program, run_id, str(pd.get("run_uuid") or "")),
        "@type": "CreateAction",
        "name": "{} run {}".format(tool["name"], run_id or "(unidentified)"),
        "description": "One call of {}{}: what it read (object), what it wrote (result), and its paradata.".format(
            tool["name"], " " + version if version else ""
        ),
        "instrument": tool,
        "containerImage": _container_image(str(pd.get("docker_image") or "")),
        "object": [dict(i) for i in inputs],
        "result": [dict(o) for o in outputs],
        "agent": _agent_entity(str(pd.get("run_agent") or "")),
        "startTime": _timestamp(pd.get("start_time")),
        "endTime": _timestamp(pd.get("end_time")),
        "actionStatus": FAILED_ACTION_STATUS if failed else COMPLETED_ACTION_STATUS,
        "error": (error or "The run failed.") if failed else None,
        "paradataRecord": pd,
    }
    return _prune(action)


def action_problems(action: Any) -> List[str]:
    """What is wrong with a `CreateAction` a service returned, as one line per problem; [] when nothing.

    The shared contract check: every tool's contract test and the E2E assert it, so the rule
    lives once. It checks the Process Run Crate MUST (an `instrument` that is a
    `SoftwareApplication`) and what ATRIUM guarantees on top: a `run_uuid` id, an absolute tool
    id with `version` and `url`, the profile's time format, a known `actionStatus` with `error`
    only on failure, typed `object`/`result` entries, and the paradata record.
    """
    if not isinstance(action, dict):
        return ["the action is a {}, not an object".format(type(action).__name__)]
    problems: List[str] = []
    if action.get("@type") != "CreateAction":
        problems.append("@type is {!r}, not 'CreateAction'".format(action.get("@type")))
    if not _RUN_UUID.match(str(action.get("@id") or "")):
        problems.append("@id {!r} is not the run's urn:uuid".format(action.get("@id")))
    if not action.get("name"):
        problems.append("no name")
    tool = action.get("instrument")
    if not isinstance(tool, dict) or tool.get("@type") != "SoftwareApplication":
        problems.append("instrument is not a SoftwareApplication")
    else:
        if not str(tool.get("@id") or "").startswith("https://"):
            problems.append("instrument @id {!r} is not an absolute URL".format(tool.get("@id")))
        for prop in ("name", "url", "version"):
            if not tool.get(prop):
                problems.append("instrument has no {}".format(prop))
        if tool.get("version") == UNRECORDED_VERSION:
            problems.append("instrument version is not recorded (para_config.txt [tool] version)")
        if "softwareVersion" in tool:
            problems.append("instrument carries softwareVersion; use version")
    for prop in ("startTime", "endTime"):
        value = action.get(prop)
        if prop == "endTime" and not value:
            problems.append("no endTime")
        elif value and not _ACTION_TIME.match(str(value)):
            problems.append("{} {!r} is not YYYY-MM-DDTHH:MM:SS+HH:MM".format(prop, value))
    status = action.get("actionStatus")
    if status not in (COMPLETED_ACTION_STATUS, FAILED_ACTION_STATUS):
        problems.append("actionStatus {!r} is not a schema.org action status".format(status))
    if action.get("error") and status != FAILED_ACTION_STATUS:
        problems.append("error is set but the action did not fail")
    for prop in ("object", "result"):
        for item in action.get(prop) or []:
            if not isinstance(item, dict) or not item.get("@id") or item.get("@type") not in _OBJECT_TYPES:
                problems.append("{} entry {!r} has no @id or an unexpected @type".format(prop, item))
    if status == COMPLETED_ACTION_STATUS and not action.get("result"):
        problems.append("a completed action has no result")
    agent = action.get("agent")
    if agent is not None and (not isinstance(agent, dict) or agent.get("@type") not in ("Person", "Organization")):
        problems.append("agent is not a Person or Organization")
    if not isinstance(action.get("paradataRecord"), dict):
        problems.append("no paradataRecord")
    return problems


def _flatten(node: Any, out: Dict[str, Dict[str, Any]]) -> Any:
    """Replace every nested entity under `node` by a reference, collecting the entities in `out`."""
    if isinstance(node, list):
        return [_flatten(item, out) for item in node]
    if isinstance(node, dict) and "@id" in node and set(node) != {"@id"}:
        entity = {k: (v if k == "paradataRecord" else _flatten(v, out)) for k, v in node.items() if k != "@context"}
        out.setdefault(node["@id"], {}).update(entity)
        return _ref(node["@id"])
    return node


def action_fragment(action: Mapping[str, Any]) -> Dict[str, Any]:
    """A nested `create_action()` result as a fragment: flat entities, no descriptor, no root.

    This is how a service's `paradata` joins a crate someone else owns: its entities go into the
    host's `@graph` and the action into the root's `mentions` (`wrap_fragment()` does exactly that).
    """
    entities: Dict[str, Dict[str, Any]] = {}
    _flatten(dict(action), entities)
    for entity in entities.values():
        # RO-Crate requires a FLAT graph, and a validator reads a nested JSON object as a nested
        # entity whatever the term's @json type says. In a crate the paradata record therefore
        # travels as its JSON text; in the nested action, which is not a crate, as the object.
        if isinstance(entity.get("paradataRecord"), dict):
            entity["paradataRecord"] = json.dumps(entity["paradataRecord"], ensure_ascii=False, sort_keys=True)
    return {"@context": _context(), "@graph": [entities[k] for k in sorted(entities)]}


# ──────────────────────────────────────────────────────────────────────────────
# The per-document crate
# ──────────────────────────────────────────────────────────────────────────────


def _content_size(path: str, data_dir: Optional[str]) -> Optional[str]:
    """A data entity's size in bytes, when its file is present under `data_dir`; else None."""
    if data_dir is None:
        return None
    full = path if os.path.isabs(path) else os.path.join(data_dir, path)
    return str(os.path.getsize(full)) if os.path.isfile(full) else None


def document_crate(
    record: Dict[str, Any],
    *,
    paradata: Paradata = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    fragment: bool = False,
    source_id: Optional[str] = None,
    data_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """The RO-Crate metadata graph for ONE document record.

    `record` is a parsed `<doc_id>.document.json`. `paradata` is the paradata records of the runs
    that wrote it (a list; the older `{program: record}` map is still accepted). With them each
    run's `CreateAction` gains its times, image and agent, and each tool its version and release
    id; without them the crate is complete, just less specific about what ran.

    **The reference-discipline rule, expressed in RO-Crate terms.** `derived_from` values are
    data entities and go in `hasPart`. `regenerable` entries are recipes for files that do not
    exist, so they are contextual entities and MUST NOT appear in `hasPart` — a crate that
    lists a file it does not contain is invalid, and this is the exact failure the record's
    "transient artifacts are never referenced" rule exists to prevent, one layer up.

    **Fragment mode** (`fragment=True`): the same entities with no metadata descriptor, no root
    and no `derived_from` files, for a crate someone else owns (AMČR's record crate). The host
    keeps its own root `./`, lists the actions in its `mentions`, the terms in its `about`, and
    declares the Process Run Crate profile; `source_id` names the host's entity for the original,
    which then replaces `#source`. `wrap_fragment()` shows the whole of it.

    `data_dir` is where the crate's files are: each data entity found there gains its
    `contentSize`.
    """
    doc_id = str(record.get("doc_id") or "")
    provenance = record.get("provenance") or {}
    blocks = (record.get("assembled") or {}).get("blocks") or {}
    runs = _paradata_records(paradata)

    graph: List[Dict[str, Any]] = []
    root: Dict[str, Any] = {
        "@id": ROOT_ID,
        "@type": "Dataset",
        "identifier": _ref("#doc_id") if doc_id else None,
        "name": name or "ATRIUM document record {}".format(doc_id),
        "description": description
        or (
            "Per-document aggregate record accreted across the ATRIUM pipeline, exported as "
            "RO-Crate. Blocks reflect contributed steps only; a block is absent until its tool "
            "has run."
        ),
        "datePublished": _date_published(record),
        "schemaVersion": str(record.get("schema_version") or ""),
        "conformsTo": _ref(PROCESS_RUN_PROFILE),
    }
    root = _prune(root)

    # ── licence ───────────────────────────────────────────────────────────────
    license_ref, license_entities = _license_entity(provenance)
    if license_ref is not None:
        root["license"] = license_ref
    graph.extend(license_entities)

    # ── the original input ────────────────────────────────────────────────────
    # NOT a data entity: `source` carries no path by design ("originals are archive-managed,
    # not pipeline-local"), so the file is not in the crate and cannot be in `hasPart`. It is
    # what the crate is BASED ON, which is a different and truthful claim. A host crate that
    # holds the original itself names it with `source_id`, and no `#source` is written.
    source = record.get("source") or {}
    original = source_id or (SOURCE_ID if source else None)
    if source and not source_id:
        graph.append(
            _prune(
                {
                    "@id": SOURCE_ID,
                    "@type": "CreativeWork",
                    "name": source.get("filename") or doc_id,
                    "identifier": doc_id,
                    "encodingFormat": source.get("media_type"),
                    "sha256": source.get("sha256"),
                    "sha512": source.get("sha512"),
                    "description": (
                        "The ORIGINAL input this record was built from, identified by doc_id and its "
                        "digest. Archive-managed and not part of this crate. Acquired as: {}.".format(
                            source.get("origin") or "unrecorded"
                        )
                    ),
                }
            )
        )
    if original:
        root["isBasedOn"] = _ref(original)

    # ── data entities: persistent step outputs only ───────────────────────────
    has_part: List[str] = []
    for key in sorted(record.get("derived_from") or {}):
        path = str((record.get("derived_from") or {})[key])
        if not path or fragment:
            continue
        has_part.append(path)
        graph.append(
            _prune(
                {
                    "@id": path,
                    "@type": "File",
                    "name": key,
                    "description": "Persistent step output recorded in derived_from[{!r}].".format(key),
                    "encodingFormat": _encoding_format(path),
                    "contentSize": _content_size(path, data_dir),
                }
            )
        )
    if has_part:
        root["hasPart"] = _refs(sorted(has_part))

    # ── regenerable recipes: contextual, never hasPart ────────────────────────
    mentions: List[str] = []
    for key in sorted(record.get("regenerable") or {}):
        recipe = (record.get("regenerable") or {})[key]
        if not isinstance(recipe, dict):
            continue
        rid = "#regenerable-" + key
        mentions.append(rid)
        graph.append(
            _prune(
                {
                    "@id": rid,
                    "@type": "CreativeWork",
                    "name": key,
                    # A reference when the file it is regenerated from is in this crate, as
                    # RO-Crate asks of any value naming an entity; a literal path otherwise.
                    "regenerableFrom": (
                        _ref(recipe["from"]) if recipe.get("from") in has_part else recipe.get("from")
                    ),
                    "regenerableWith": recipe.get("converter"),
                    "description": (
                        "DISPOSABLE derivation, recorded as a reproducible recipe rather than a "
                        "stored path (detail: {}). Deliberately absent from hasPart: the file is "
                        "not in this crate and is not meant to be.".format(recipe.get("detail") or "unspecified")
                    ),
                }
            )
        )

    # ── one contextual entity per stamped block ───────────────────────────────
    # This is where the record's granularity survives into the crate. `assembled.blocks` is
    # "the source of granularity" — re-running one tool rewrites exactly one entry — and a
    # crate that flattened it to a single "produced by the pipeline" claim would throw away
    # the one thing the accretion design bought. Who wrote a block is the run whose action
    # lists it as `result`; `creator` is for people (the tools' authors), not for tools.
    for name_ in sorted(blocks):
        stamp = blocks[name_] if isinstance(blocks[name_], dict) else {}
        graph.append(
            _prune(
                {
                    "@id": _block_id(name_),
                    "@type": "CreativeWork",
                    "name": name_,
                    "recordBlock": name_,
                    "dateModified": stamp.get("updated_at"),
                    "description": "Block {!r} of the document record, as stamped by assembled.blocks.".format(name_),
                }
            )
        )

    # ── one CreateAction per contributing run, one SoftwareApplication per tool ──
    # A stamp whose run has no provenance.contributors entry (a record assembled from a
    # hand-built baseline) still gets an action, reconstructed from the stamp, so no block is
    # left without the run that wrote it.
    contributors = [c for c in provenance.get("contributors") or [] if isinstance(c, dict)]
    seen_runs = {(str(c.get("program") or ""), str(c.get("run_id") or "")) for c in contributors}
    for name_ in sorted(blocks):
        stamp = blocks[name_] if isinstance(blocks[name_], dict) else {}
        key = (str(stamp.get("program") or ""), str(stamp.get("run_id") or ""))
        if key[0] and key not in seen_runs:
            seen_runs.add(key)
            contributors.append(
                {
                    "program": key[0],
                    "run_id": key[1],
                    "run_uuid": stamp.get("run_uuid", ""),
                    "paradata_ref": stamp.get("paradata_ref", ""),
                    "blocks": ",".join(n for n in sorted(blocks) if (blocks[n] or {}).get("run_id") == key[1]),
                    "at": stamp.get("updated_at", ""),
                    "_reconstructed": True,
                }
            )

    tools: Dict[str, Dict[str, Any]] = {}
    extras: Dict[str, Dict[str, Any]] = {}
    for contributor in contributors:
        program = str(contributor.get("program") or "unknown")
        run_id = str(contributor.get("run_id") or "")
        run_uuid = str(contributor.get("run_uuid") or "")
        pd = _match_paradata(runs, program, run_id, run_uuid)
        run_uuid = run_uuid or str(pd.get("run_uuid") or "")
        tool = _tool_entity(program, pd)
        tools.setdefault(tool["@id"], tool)
        image = _container_image(str(pd.get("docker_image") or ""))
        agent = _agent_entity(str(pd.get("run_agent") or ""))
        for extra in (image, agent):
            if extra:
                extras.setdefault(extra["@id"], extra)
        wrote = [b.strip() for b in str(contributor.get("blocks") or "").split(",") if b.strip()]
        action_id = _run_action_id(program, run_id, run_uuid)
        mentions.append(action_id)
        how = (
            "Reconstructed from assembled.blocks: the record lists no contributor entry for this run."
            if contributor.get("_reconstructed")
            else "Contributed block(s) {}.".format(", ".join(wrote) or "none recorded")
        )
        graph.append(
            _prune(
                {
                    "@id": action_id,
                    "@type": "CreateAction",
                    "name": "{} run {}".format(program, run_id or "unknown"),
                    "description": "{} Run paradata: {}.".format(how, contributor.get("paradata_ref") or "not recorded"),
                    "startTime": _timestamp(pd.get("start_time")),
                    "endTime": _timestamp(pd.get("end_time") or contributor.get("at")),
                    "actionStatus": COMPLETED_ACTION_STATUS,
                    "instrument": _ref(tool["@id"]),
                    "containerImage": _ref(image["@id"]) if image else None,
                    "agent": _ref(agent["@id"]) if agent else None,
                    "object": _ref(original) if original else None,
                    "result": _refs(_block_id(b) for b in wrote if b in blocks) or None,
                }
            )
        )
    graph.extend(tools.values())
    graph.extend(extras.values())
    if tools:
        graph.extend(_person_entities())

    # ── controlled terms ──────────────────────────────────────────────────────
    schemes = set()
    about: List[str] = []
    for scheme, value in _controlled_values(record):
        uri = _concept_uri(scheme, value)
        schemes.add(scheme)
        about.append(uri)
        graph.append(
            {
                "@id": uri,
                "@type": "DefinedTerm",
                "name": value,
                "termCode": value,
                "inDefinedTermSet": _ref(_scheme_uri(scheme)),
            }
        )
    for scheme in sorted(schemes):
        graph.append({"@id": _scheme_uri(scheme), "@type": "DefinedTermSet", "name": scheme})
    if about:
        root["about"] = _refs(sorted(set(about)))
    if mentions:
        root["mentions"] = _refs(sorted(set(mentions)))

    if fragment:
        return _fragment(graph)
    if doc_id:
        graph.append(_identifier_entity("#doc_id", "doc_id", doc_id))
    graph.append(_profile_entity())
    graph.append(root)
    return _assemble(graph)


# ──────────────────────────────────────────────────────────────────────────────
# The run-level crate
# ──────────────────────────────────────────────────────────────────────────────


def run_crate(
    records: Sequence[Dict[str, Any]],
    *,
    run_paradata: Optional[Dict[str, Any]] = None,
    document_crate_dir: str = "{doc_id}",
    paradata_refs: Sequence[str] = (),
    name: Optional[str] = None,
    description: Optional[str] = None,
    data_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """The RO-Crate metadata graph for ONE PIPELINE RUN over several documents.

    The per-document crate is the primitive; this one references those crates rather than
    re-describing their contents, so a run crate stays the same size whether the run covered
    five documents or five thousand — which matters for a corpus the DMP sizes at 1.2 million
    pages.

    `run_paradata` is `merge_run_paradata()`'s output. Its `pipeline_stages[]` become the
    run-level `CreateAction` chain, and `start_time` / `end_time` are recorded on the root.
    Everything is optional: a run crate built from records alone is still valid, it simply
    says less about the run than about its results.
    """
    graph: List[Dict[str, Any]] = []
    run = run_paradata or {}
    doc_ids = [str(r.get("doc_id") or "") for r in records]
    run_id = str(run.get("run_id") or "")

    root: Dict[str, Any] = {
        "@id": ROOT_ID,
        "@type": "Dataset",
        "identifier": _ref("#run_id"),
        "name": name or "ATRIUM pipeline run {}".format(run_id or "(unidentified)"),
        "description": description
        or (
            "One ATRIUM pipeline run, packaged as RO-Crate: the per-document crates it "
            "produced, the paradata that records how it behaved, and the tool chain that ran."
        ),
        "datePublished": _run_date_published(records, run_paradata),
        "conformsTo": _ref(PROCESS_RUN_PROFILE),
    }

    provenance = (records[0].get("provenance") if records else {}) or {}
    license_ref, license_entities = _license_entity(run if run_paradata else provenance)
    if license_ref is None:
        license_ref, license_entities = _license_entity(provenance)
    if license_ref is not None:
        root["license"] = license_ref
    graph.extend(license_entities)

    # ── the per-document crates are the parts ─────────────────────────────────
    parts: List[str] = []
    documents: List[str] = []
    for doc_id in sorted(set(d for d in doc_ids if d)):
        part_id = document_crate_dir.format(doc_id=doc_id).rstrip("/") + "/"
        parts.append(part_id)
        documents.append(part_id)
        graph.append(
            {
                "@id": part_id,
                "@type": "Dataset",
                "identifier": doc_id,
                "name": "ATRIUM document record {}".format(doc_id),
                # RO-Crate 1.2's form for a referenced crate: the version-less base profile, and
                # the crate's own metadata file as `subjectOf`.
                "conformsTo": _refs(["https://w3id.org/ro/crate", ROCRATE_CONFORMS_TO]),
                "subjectOf": _ref(part_id + METADATA_FILENAME),
                "description": "Nested per-document RO-Crate; see its own {}.".format(METADATA_FILENAME),
            }
        )
        graph.append(
            {
                "@id": part_id + METADATA_FILENAME,
                "@type": "CreativeWork",
                "name": "RO-Crate metadata of {}".format(doc_id),
                "encodingFormat": "application/ld+json",
            }
        )
    for ref in sorted(set(paradata_refs)):
        parts.append(ref)
        graph.append(
            _prune(
                {
                    "@id": ref,
                    "@type": "File",
                    "name": os.path.basename(ref),
                    "encodingFormat": "application/json",
                    "contentSize": _content_size(ref, data_dir),
                    "description": "atrium_paradata run record — how this run behaved.",
                }
            )
        )
    if parts:
        root["hasPart"] = _refs(sorted(set(parts)))

    # ── the stage chain ───────────────────────────────────────────────────────
    mentions: List[str] = []
    tools: Dict[str, Dict[str, Any]] = {}
    extras: Dict[str, Dict[str, Any]] = {}
    agent = _agent_entity(str(run.get("run_agent") or ""))
    if agent:
        extras[agent["@id"]] = agent
    stages = [s for s in run.get("pipeline_stages") or [] if isinstance(s, dict)]
    for stage in stages:
        program = str(stage.get("program") or "unknown")
        # A merged record names one repository and version, its first stage's; they describe a
        # stage only when it is the only one. Each stage carries its own since atrium-project#71.
        single = len(stages) == 1
        stage_info = {
            "repository": stage.get("repository") or (run.get("repository") if single else None),
            "tool_version": stage.get("tool_version") or (run.get("tool_version") if single else None),
        }
        tool = _tool_entity(program, stage_info)
        tools.setdefault(tool["@id"], tool)
        image = _container_image(str(stage.get("docker_image") or ""))
        if image:
            extras.setdefault(image["@id"], image)
        stage_run_id = str(stage.get("run_id") or run_id)
        action_id = _run_action_id(program, stage_run_id, str(stage.get("run_uuid") or ""))
        mentions.append(action_id)
        graph.append(
            _prune(
                {
                    "@id": action_id,
                    "@type": "CreateAction",
                    "name": "stage {}: {}".format(stage.get("order", "?"), program),
                    "instrument": _ref(tool["@id"]),
                    "containerImage": _ref(image["@id"]) if image else None,
                    "agent": _ref(agent["@id"]) if agent else None,
                    "position": stage.get("order"),
                    "startTime": _timestamp(stage.get("start_time")),
                    "endTime": _timestamp(stage.get("end_time")),
                    "actionStatus": COMPLETED_ACTION_STATUS,
                    "result": _refs(documents) or None,
                    "description": (
                        "script={} method={} inputs={} processed={} skipped={} duration_s={}".format(
                            stage.get("script") or "?",
                            stage.get("method") or "?",
                            stage.get("input_files_total", "?"),
                            stage.get("successfully_processed", "?"),
                            stage.get("skipped_files", "?"),
                            stage.get("duration_seconds", "?"),
                        )
                    ),
                }
            )
        )
    graph.extend(tools.values())
    graph.extend(extras.values())
    if tools:
        graph.extend(_person_entities())
    if mentions:
        root["mentions"] = _refs(sorted(set(mentions)))

    for key, prop in (("start_time", "startTime"), ("end_time", "endTime")):
        if _timestamp(run.get(key)):
            root[prop] = _timestamp(run.get(key))

    graph.append(_identifier_entity("#run_id", "run_uuid" if run.get("run_uuid") else "run_id",
                                    str(run.get("run_uuid") or "") or run_id or "atrium-run"))
    graph.append(_profile_entity())
    graph.append(root)
    return _assemble(graph)


def _run_date_published(records: Sequence[Dict[str, Any]], run_paradata: Optional[Dict[str, Any]]) -> str:
    explicit = str((run_paradata or {}).get("end_time") or "")
    if len(explicit) >= 10:
        return explicit[:10]
    dates = [_date_published(r) for r in records]
    return max(dates) if dates else "1970-01-01"


# ──────────────────────────────────────────────────────────────────────────────
# Fragments: entities for a crate someone else owns
# ──────────────────────────────────────────────────────────────────────────────


def _merged(graph: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Entities by `@id`, merging repeats.

    Same subject described twice (a tool that both stamped a block and contributed a run, say).
    Merge rather than let the last one win: dropping half a description silently is the failure
    mode `merge_document_records()` was fixed for.
    """
    merged: Dict[str, Dict[str, Any]] = {}
    for entity in graph:
        eid = entity["@id"]
        if eid in merged:
            merged[eid].update({k: v for k, v in entity.items() if v not in (None, "", [], {})})
        else:
            merged[eid] = dict(entity)
    return merged


def _fragment(graph: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    merged = _merged(graph)
    return {"@context": _context(), "@graph": [merged[k] for k in sorted(merged)]}


def wrap_fragment(
    fragment: Mapping[str, Any],
    *,
    name: str,
    description: str,
    license: Union[str, Mapping[str, Any]],
    date_published: str,
) -> Dict[str, Any]:
    """A fragment under a stub root: the smallest complete crate that embeds it.

    It is what a host does with a fragment, done once: its entities go into the graph, the root
    lists every `CreateAction` in `mentions`, every `DefinedTerm` in `about` and `#source` (when
    present) in `isBasedOn`, and declares the Process Run Crate profile. A fragment has no root,
    so this is also the only way to validate one (`tools/ci/rocrate_check.py`).
    """
    graph = [dict(e) for e in fragment.get("@graph") or []]
    by_type: Dict[str, List[str]] = {}
    for entity in graph:
        types = entity.get("@type")
        for t in types if isinstance(types, list) else [types]:
            by_type.setdefault(str(t), []).append(entity["@id"])
    root = _prune(
        {
            "@id": ROOT_ID,
            "@type": "Dataset",
            "name": name,
            "description": description,
            "datePublished": date_published,
            "license": _license_value(license),
            "conformsTo": _ref(PROCESS_RUN_PROFILE),
            "mentions": _refs(
                sorted(by_type.get("CreateAction", []) + [e["@id"] for e in graph if "regenerableFrom" in e])
            ),
            "about": _refs(sorted(by_type.get("DefinedTerm", []))),
            "isBasedOn": _ref(SOURCE_ID) if SOURCE_ID in {e["@id"] for e in graph} else None,
        }
    )
    if isinstance(license, str) and license.startswith("http") and license not in {e["@id"] for e in graph}:
        graph.append(
            {"@id": license, "@type": "CreativeWork", "name": license, "description": "The licence of this crate."}
        )
    return _assemble(graph + [_profile_entity(), root])


def _license_value(license: Union[str, Mapping[str, Any]]) -> Any:
    """A root `license`: a reference for a URL, a literal for a bare name, as `_license_entity()` does."""
    if not isinstance(license, str):
        return dict(license)
    return _ref(license) if license.startswith("http") else {"@value": license}


# ──────────────────────────────────────────────────────────────────────────────
# Assembly, serialisation, output
# ──────────────────────────────────────────────────────────────────────────────


def _assemble(graph: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Add the metadata descriptor, deduplicate, and order the graph deterministically.

    The descriptor and the root entity are pinned to the front — the spec does not require an
    order, but every RO-Crate in the wild puts them there and a reader opening the file should
    not have to search for what the crate is about. Everything else sorts by `@id`. RO-Crate 1.2
    puts the specification on the descriptor and the profiles on the root.
    """
    descriptor = {
        "@id": METADATA_FILENAME,
        "@type": "CreativeWork",
        "conformsTo": _ref(ROCRATE_CONFORMS_TO),
        "about": _ref(ROOT_ID),
    }
    merged = _merged(graph)
    merged[METADATA_FILENAME] = descriptor

    pinned = [METADATA_FILENAME, ROOT_ID]
    ordered = [merged[p] for p in pinned if p in merged]
    ordered += [merged[k] for k in sorted(merged) if k not in pinned]

    return {"@context": _context(), "@graph": ordered}


def to_json(crate: Dict[str, Any]) -> str:
    """The crate as the bytes that go on disk. `sort_keys=True` — byte-stable."""
    return json.dumps(crate, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def write_crate(crate: Dict[str, Any], out_dir: str) -> str:
    """Write `ro-crate-metadata.json` into `out_dir`, atomically.

    Write-then-rename for the same reason `DocumentRecord.finalize()` does it: a crash
    mid-write must not leave a half-written metadata file that a consumer reads as a corrupt
    crate rather than as a missing one.
    """
    os.makedirs(out_dir or ".", exist_ok=True)
    path = os.path.join(out_dir or ".", METADATA_FILENAME)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(to_json(crate))
    os.replace(tmp, path)
    return path


def _encoding_format(path: str) -> Optional[str]:
    """Media type from the pipeline's own filename conventions, not from `mimetypes`.

    `mimetypes` answers `text/xml` for `.teitok.xml` and nothing at all for `.conllu`, and the
    suffixes here are a closed set fixed by `KNOWN_PIPELINE_SUFFIXES` in atrium_document.py.
    Unknown suffix returns None, which `_prune()` then omits — see its docstring.
    """
    lower = path.lower()
    for suffix, media in (
        (".teitok.xml", "application/tei+xml"),
        (".alto.xml", "application/alto+xml"),
        (".conllu", "text/x-conllu"),
        (".document.json", "application/json"),
        (".json", "application/json"),
        (".xml", "application/xml"),
        (".csv", "text/csv"),
        (".txt", "text/plain"),
        (".md", "text/markdown"),
        (".pdf", "application/pdf"),
    ):
        if lower.endswith(suffix):
            return media
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────────────────────────────────────

_SAMPLE_RUN_UUIDS = {
    "alto-postprocess": "urn:uuid:5a1e0c1e-1b2d-4c3e-8f40-0a1b2c3d4e5f",
    "nlp-enrich": "urn:uuid:9c8b7a6d-5e4f-4321-9abc-def012345678",
}


def _sample_record() -> Dict[str, Any]:
    """A record exercising every mapping branch, including the ones that must NOT map."""
    return {
        "schema_version": "1.0",
        "record_type": "atrium-document",
        "doc_id": "CTX000000001",
        "source": {
            "sha256": "a" * 64,
            "sha512": "b" * 128,
            "filename": "CTX000000001.alto.xml",
            "media_type": "application/alto+xml",
            "page_count": 1,
            "origin": "ABBYY-ALTO",
        },
        "derived_from": {
            "teitok": "TEITOK/CTX000000001.teitok.xml",
            "paradata": "paradata/260724-101112_pipeline-run.json",
        },
        "regenerable": {
            "markdown": {"from": "TEITOK/CTX000000001.teitok.xml", "converter": "xml_to_md@0.3.0", "detail": "full"}
        },
        "provenance": {
            "license": "CC BY-NC-SA 4.0",
            "license_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
            "contributors": [
                {
                    "program": "alto-postprocess",
                    "run_id": "260724-101112",
                    "run_uuid": _SAMPLE_RUN_UUIDS["alto-postprocess"],
                    "paradata_ref": "paradata/260724-101112_alto-postprocess.json",
                    "blocks": "pages,content",
                    "at": "2026-07-24T10:11:12.345678+00:00",
                },
                {
                    "program": "nlp-enrich",
                    "run_id": "260724-101500",
                    "run_uuid": _SAMPLE_RUN_UUIDS["nlp-enrich"],
                    "paradata_ref": "paradata/260724-101500_nlp-enrich.json",
                    "blocks": "entities,derived_from",
                    "at": "2026-07-24T10:15:00+00:00",
                },
            ],
        },
        "assembled": {
            "blocks": {
                "pages": {
                    "program": "alto-postprocess",
                    "run_id": "260724-101112",
                    "run_uuid": _SAMPLE_RUN_UUIDS["alto-postprocess"],
                    "updated_at": "2026-07-24T10:11:12+00:00",
                },
                "content": {
                    "program": "alto-postprocess",
                    "run_id": "260724-101112",
                    "run_uuid": _SAMPLE_RUN_UUIDS["alto-postprocess"],
                    "updated_at": "2026-07-24T10:11:12+00:00",
                },
                "entities": {
                    "program": "nlp-enrich",
                    "run_id": "260724-101500",
                    "run_uuid": _SAMPLE_RUN_UUIDS["nlp-enrich"],
                    "updated_at": "2026-07-24T10:15:00+00:00",
                },
                "derived_from": {
                    "program": "nlp-enrich",
                    "run_id": "260724-101500",
                    "run_uuid": _SAMPLE_RUN_UUIDS["nlp-enrich"],
                    "updated_at": "2026-07-24T10:15:00+00:00",
                },
            },
            "had_baseline": True,
            "note": "Blocks reflect CONTRIBUTED steps only; a block is absent until its tool has run.",
        },
        "page_categories": {"1": "TEXT"},
        "pages": [{"page": "1", "page_index": 1, "quality_band": "Clear", "category": "TEXT"}],
        "content": {"text": "Náčrt sondy.", "reading_order": "ltr-columns"},
        "entities": [{"surface": "Praha", "type_teitok": "LOC", "type_cnec": "gu", "page": "1", "line": 0}],
    }


def _sample_paradata() -> List[Dict[str, Any]]:
    """The paradata records of the sample record's two runs, as `ParadataLogger.record` has them."""
    return [
        {
            "schema_version": "2.0",
            "program": "alto-postprocess",
            "tool_version": "v1.6.0-beta",
            "repository": "https://github.com/ufal/atrium-alto-postprocess",
            "docker_image": "ghcr.io/ufal/atrium-alto-postprocess-api:1.6.0-beta",
            "python_version": "3.11.15 (main) [GCC]",
            "run_id": "260724-101112",
            "run_uuid": _SAMPLE_RUN_UUIDS["alto-postprocess"],
            "run_agent": "https://ror.org/00x0x0x00",
            "start_time": "2026-07-24T10:11:12.000001+00:00",
            "end_time": "2026-07-24T10:11:40.250000+00:00",
        },
        {
            "schema_version": "2.0",
            "program": "nlp-enrich",
            "tool_version": "0.22.0",
            "repository": "https://github.com/ufal",
            "docker_image": "ghcr.io/ufal/atrium-nlp-enrich-api:0.22.0",
            "python_version": "3.11.15",
            "run_id": "260724-101500",
            "run_uuid": _SAMPLE_RUN_UUIDS["nlp-enrich"],
            "run_agent": "https://ror.org/00x0x0x00",
            "start_time": "2026-07-24T10:15:00+00:00",
            "end_time": "2026-07-24T10:16:02+00:00",
        },
    ]


def _walk_refs(node: Any) -> Iterable[str]:
    if isinstance(node, dict):
        if set(node) == {"@id"}:
            yield node["@id"]
        else:
            for k, v in node.items():
                if k != "paradataRecord":
                    yield from _walk_refs(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk_refs(v)


def _selftest(stream: Any = None) -> int:
    """Structural checks. Exit 1 on any problem — the same contract as atrium_vocab --selftest."""
    out = stream or sys.stdout
    problems: List[str] = []
    record = _sample_record()
    crate = document_crate(record, paradata=_sample_paradata())
    graph = crate["@graph"]
    by_id = {e["@id"]: e for e in graph}

    # 1. RO-Crate 1.2's own mandatory shape: the spec on the descriptor, the profile on the root.
    if graph[0]["@id"] != METADATA_FILENAME:
        problems.append("first graph entity is not the metadata descriptor")
    if graph[0].get("about") != {"@id": ROOT_ID}:
        problems.append("descriptor does not point `about` at the root")
    if graph[0].get("conformsTo") != {"@id": ROCRATE_CONFORMS_TO}:
        problems.append("descriptor does not conform to RO-Crate {}".format(ROCRATE_VERSION))
    root = by_id.get(ROOT_ID, {})
    for prop in ("@type", "name", "description", "datePublished", "license"):
        if not root.get(prop):
            problems.append("root is missing required property {!r}".format(prop))
    if root.get("@type") != "Dataset":
        problems.append("root @type is not Dataset")
    if root.get("conformsTo") != {"@id": PROCESS_RUN_PROFILE} or PROCESS_RUN_PROFILE not in by_id:
        problems.append("root does not declare the Process Run Crate profile as an entity")
    if "author" in root:
        problems.append("root carries author: the ATRIUM authors are each tool's creator")

    # 2. No dangling references.
    for ref in sorted(set(_walk_refs(graph))):
        if ref in by_id or ref.startswith(("http://", "https://")):
            continue
        problems.append("dangling reference {!r} — no entity and not an absolute URI".format(ref))

    # 3. The reference-discipline rule: regenerable is never a part.
    parts = {r["@id"] for r in root.get("hasPart", [])}
    if any(p.startswith("#regenerable-") for p in parts):
        problems.append("a regenerable recipe appears in hasPart")
    if "TEITOK/CTX000000001.teitok.xml" not in parts:
        problems.append("a derived_from output is missing from hasPart")
    if SOURCE_ID in parts:
        problems.append("the archive-managed original appears in hasPart")

    # 4. Determinism — the property the byte-comparison gate depends on.
    if to_json(document_crate(record)) != to_json(document_crate(record)):
        problems.append("document_crate() is not deterministic")
    if root.get("datePublished") != "2026-07-24":
        problems.append("datePublished was not derived from the record's own newest stamp")

    # 5. Runs, tools, people: stable ids, `version`, authors as creator, profile time format.
    for program, run_uuid in _SAMPLE_RUN_UUIDS.items():
        action = by_id.get(run_uuid, {})
        if action.get("@type") != "CreateAction":
            problems.append("the {} run lost its CreateAction (or its run_uuid id)".format(program))
        for prop in ("startTime", "endTime"):
            if action.get(prop) and not _ACTION_TIME.match(action[prop]):
                problems.append("{} {} is not in the profile's format".format(program, prop))
    tools = [e for e in graph if e.get("@type") == "SoftwareApplication"]
    for tool in tools:
        if not str(tool["@id"]).startswith("https://") or tool.get("version") in (None, "", UNRECORDED_VERSION):
            problems.append("tool {} lacks an absolute id or a recorded version".format(tool["@id"]))
        if "softwareVersion" in tool:
            problems.append("tool {} carries softwareVersion".format(tool["@id"]))
        if {c["@id"] for c in tool.get("creator", [])} != {a["orcid"] for a in AUTHORS}:
            problems.append("tool {} does not name the ATRIUM authors as creator".format(tool["@id"]))
    if "https://github.com/ufal/atrium-nlp-enrich/releases/tag/v0.22.0" not in by_id:
        problems.append("a release id was not built from the table's repository")

    # 6. Controlled terms became DefinedTerms.
    if len([e for e in graph if e.get("@type") == "DefinedTerm"]) != 4:  # TEXT, Clear, LOC, gu
        problems.append("expected 4 DefinedTerms")

    # 7. The run crate.
    rc = run_crate(
        [record],
        run_paradata={
            "run_id": "260724-101112",
            "end_time": "2026-07-24T11:00:00+00:00",
            "pipeline_stages": [{"order": 1, "program": "alto-postprocess", "run_id": "260724-101112"}],
        },
        paradata_refs=["paradata/260724-101112_pipeline-run.json"],
    )
    rroot = {e["@id"]: e for e in rc["@graph"]}[ROOT_ID]
    if {"@id": "CTX000000001/"} not in rroot.get("hasPart", []):
        problems.append("run crate does not reference the per-document crate")
    if rroot.get("conformsTo") != {"@id": PROCESS_RUN_PROFILE}:
        problems.append("run crate does not declare the process-run profile")

    # 8. The fragment: no descriptor, no root, no data entity; wrapped, it is a crate again.
    frag = document_crate(record, paradata=_sample_paradata(), fragment=True)
    frag_ids = {e["@id"] for e in frag["@graph"]}
    if {METADATA_FILENAME, ROOT_ID} & frag_ids or any(e.get("@type") == "File" for e in frag["@graph"]):
        problems.append("the fragment carries a descriptor, a root or a data entity")
    wrapped = wrap_fragment(
        frag,
        name="stub",
        description="stub",
        license="https://creativecommons.org/licenses/by-nc-sa/4.0/",
        date_published="2026-07-24",
    )
    wroot = {e["@id"]: e for e in wrapped["@graph"]}[ROOT_ID]
    if not set(_SAMPLE_RUN_UUIDS.values()) <= {r["@id"] for r in wroot.get("mentions", [])}:
        problems.append("wrap_fragment() does not mention every action")

    # 9. A service's CreateAction meets the shared contract.
    action = create_action(
        _sample_paradata()[0],
        inputs=[file_entity("x.alto.xml", b"<alto/>", media_type="application/alto+xml"), record_entity("C-1")],
        outputs=block_entities(["pages", "lines"]),
    )
    for problem in action_problems(action):
        problems.append("create_action(): " + problem)

    # 10. JSON round-trip.
    for built in (crate, rc, wrapped, action):
        json.loads(json.dumps(built))

    for problem in problems:
        print("PROBLEM: " + problem, file=out)
    print(
        "atrium_rocrate selftest: {} problem(s); {} entities in the document crate".format(len(problems), len(graph)),
        file=out,
    )
    return 1 if problems else 0


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def _load(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _cli(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="python atrium_rocrate.py")
    p.add_argument("--selftest", action="store_true", help="structural checks; exit 1 on any")
    p.add_argument("--document", metavar="RECORD", help="path to one <doc_id>.document.json")
    p.add_argument("--run", nargs="+", metavar="RECORD", help="several document records -> one run crate")
    p.add_argument(
        "--paradata",
        metavar="JSON",
        action="append",
        default=[],
        help="a paradata record: with --document, each run's (repeat it); with --run, the merged run record",
    )
    p.add_argument("--fragment", action="store_true", help="with --document: entities only, no descriptor or root")
    p.add_argument("--wrap", action="store_true", help="with --fragment: put the fragment under a stub root")
    p.add_argument("--source-id", metavar="ID", default=None, help="with --fragment: the host's id of the original")
    p.add_argument("--data-dir", metavar="DIR", default=None, help="where the crate's files are (contentSize)")
    p.add_argument("--out-dir", metavar="DIR", default=None, help="write ro-crate-metadata.json here")
    args = p.parse_args(argv)

    if args.selftest:
        return _selftest()

    if args.document:
        record = _load(args.document)
        runs = [_load(path) for path in args.paradata]
        crate = document_crate(
            record, paradata=runs, fragment=args.fragment, source_id=args.source_id, data_dir=args.data_dir
        )
        if args.fragment and args.wrap:
            provenance = record.get("provenance") or {}
            crate = wrap_fragment(
                crate,
                name="ATRIUM record fragment {} (stub root)".format(record.get("doc_id") or ""),
                description="The fragment of one ATRIUM record under a stub root, as a host crate embeds it.",
                license=str(provenance.get("license_url") or provenance.get("license") or "unspecified"),
                date_published=_date_published(record),
            )
    elif args.run:
        crate = run_crate(
            [_load(r) for r in args.run],
            run_paradata=_load(args.paradata[0]) if args.paradata else None,
            data_dir=args.data_dir,
        )
    else:
        p.error("one of --selftest, --document or --run is required")
        return 2

    if args.out_dir:
        print(write_crate(crate, args.out_dir))
    else:
        sys.stdout.write(to_json(crate))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
