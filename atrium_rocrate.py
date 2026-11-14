"""
atrium_rocrate.py — the RO-Crate view of an ATRIUM document record, and the run's CreateAction.

WHY THIS MODULE EXISTS
======================
The AMČR pilot stores every distribution with its paradata: one Process Run Crate
``CreateAction`` per step, RO-Crate 1.2 / Process Run Crate 0.5 (atrium-project#71, #84). Since
record schema 2.0 the document record holds content only (no stamps, no contributors, no
licence), so the runs are described by their actions alone:

    paradata 3.0 inputs   -> the action's `object`: distributions by name and SHA-512
    paradata 3.0 outputs  -> the action's `result`; the record's names the blocks AND fields written
    stored: false         -> an input or output that is not stored, still named by SHA-512 (D23)
    tool_version / repository / docker_image / run_uuid / times -> instrument, image, @id, times
    components            -> facts in the paradata record; nothing is combined into a licence (D2)
    atrium_vocab          -> resolvable URIs for every controlled term the record carries

So this module maps rather than invents. Where a field does not exist it is **omitted**, not
guessed: a crate that quietly asserts a checksum nobody computed is worse than one that admits the
gap.

WHAT IT PRODUCES
================
* ``create_action(paradata)`` / ``action_from_paradata(paradata)`` — one run as a nested JSON-LD
  ``CreateAction``: what every service returns as ``paradata`` and what AMČR stores as the run's
  paradata file.
* ``document_crate(record, paradata=…)`` — ÚFAL's crate of one record and the runs that wrote it.
  It is NOT AMČR's archival crate (whose root is the whole AMČR record and states the document's
  licence); with ``fragment=True`` it gives the entities AMČR reuses: the actions, the software,
  the controlled terms — no descriptor, no root, no block entities.
* ``run_crate(records)`` — one pipeline run over several records, referencing their crates.
* ``wrap_fragment(fragment)`` — a fragment under a stub root, which is how a fragment is validated.

A crate root names a licence only when its caller asserts one (``license=``): the rights in a
document are the archive's to state (contract §8, layer 1), and the tools assert none (D2). Without
one the root's ``license`` names ``#no-licence-asserted``, an entity that asserts nothing: RO-Crate
1.2 requires a root licence, and this is not a calculated one (atrium-project#6, confirmed 2026-11-12).

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

``atrium_vocab`` is imported softly — with it, every ATRIUM-controlled term in the record becomes a
``DefinedTerm`` with a resolvable ``https://w3id.org/atrium/`` URI; without it the crate is still
valid, just less semantic. An AMČR or TEATER concept the record names by id
(``enrichment.items[].teater_category_ids``) keeps its own source URI in either mode.

DETERMINISM
===========
Two calls on the same input produce byte-identical JSON. ``@graph`` is sorted by ``@id`` (with
the descriptor and the root pinned first) and ``json.dumps`` runs with ``sort_keys=True``.
``datePublished`` is derived from the newest run's end time, never from the clock, and a run's
identifier is the ``run_uuid`` its paradata carry — nothing is minted here.
"""

from __future__ import annotations

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

#: The record's AMČR distribution name: an input or output with this name is the document record.
RECORD_DISTRIBUTION = "atrium/document-json"

#: What a crate root's `license` names when its caller asserts none. RO-Crate 1.2 requires a root
#: licence (rule 13.3); the tools assert none (D2), so the root names an entity that asserts nothing.
#: Not a calculated licence: motyc's option 1 on atrium-project#6 (2026-10-29), confirmed 2026-11-12.
NO_LICENCE_ID = "#no-licence-asserted"

#: ATRIUM's own namespace, so extension terms carry a resolvable prefix rather than being
#: dropped by a consumer that compacts strictly against the RO-Crate context. Kept in step
#: with atrium_vocab.SKOS_BASE, which is the one place an ATRIUM URI is rooted.
ATRIUM_BASE = getattr(atrium_vocab, "SKOS_BASE", "https://w3id.org/atrium/")

#: The people who wrote the ATRIUM tools. Since atrium-project#71 they are each tool's `creator`
#: — the people who made the software — and no longer the crate root's `author`: the record is
#: the archive's, the tools are theirs. ORCIDs are the `@id`s, the RO-Crate convention, so no
#: ATRIUM identifier is minted for a person. Who wrote which tool is `PROGRAM_AUTHORS`.
AUTHORS: Tuple[Dict[str, str], ...] = (
    {"orcid": "https://orcid.org/0009-0002-4773-2797", "name": "Kateryna Lutsai"},
    {"orcid": "https://orcid.org/0000-0002-6895-8536", "name": "Pavel Straňák"},
    {"orcid": "https://orcid.org/0000-0001-6592-6291", "name": "David Novák"},
    {"orcid": "https://orcid.org/0000-0001-5718-9447", "name": "Dana Křivánková"},
    {"orcid": "https://orcid.org/0009-0002-8856-2501", "name": "David Spáčil"},
)

#: The authors every tool has (Lutsai, Novák, Straňák): the `creator` of a program the table
#: below does not list, such as a predecessor whose CITATION.cff this module does not know.
CORE_AUTHORS: Tuple[str, ...] = (
    "https://orcid.org/0009-0002-4773-2797",
    "https://orcid.org/0000-0001-6592-6291",
    "https://orcid.org/0000-0002-6895-8536",
)

#: program -> its authors' ORCIDs, in the order of that tool's CITATION.cff (atrium-project#13).
#: A tool's `creator` is its own authors, not everyone's (until 2026-10-28 every tool named the
#: same four). `alto-postprocess` is ocr-postprocess under its old name. A service's `-api`
#: suffix is stripped before the lookup, as in `_repository()`.
PROGRAM_AUTHORS: Dict[str, Tuple[str, ...]] = {
    "page-classification": (
        "https://orcid.org/0009-0002-4773-2797",
        "https://orcid.org/0000-0001-5718-9447",
        "https://orcid.org/0000-0001-6592-6291",
        "https://orcid.org/0000-0002-6895-8536",
    ),
    "ocr-postprocess": (
        "https://orcid.org/0009-0002-4773-2797",
        "https://orcid.org/0000-0001-5718-9447",
        "https://orcid.org/0009-0002-8856-2501",
        "https://orcid.org/0000-0001-6592-6291",
        "https://orcid.org/0000-0002-6895-8536",
    ),
    "keyword-extract": (
        "https://orcid.org/0009-0002-4773-2797",
        "https://orcid.org/0009-0002-8856-2501",
        "https://orcid.org/0000-0001-6592-6291",
        "https://orcid.org/0000-0002-6895-8536",
    ),
    "digital-convert": CORE_AUTHORS,
    "nlp-enrich": CORE_AUTHORS,
    "translator": CORE_AUTHORS,
}
PROGRAM_AUTHORS["alto-postprocess"] = PROGRAM_AUTHORS["ocr-postprocess"]


def authors_of(program: str) -> Tuple[Dict[str, str], ...]:
    """The authors of one program (`PROGRAM_AUTHORS`), `CORE_AUTHORS` for a program it does not list."""
    base = program[: -len("-api")] if program.endswith("-api") else program
    orcids = PROGRAM_AUTHORS.get(program) or PROGRAM_AUTHORS.get(base) or CORE_AUTHORS
    people = {a["orcid"]: a for a in AUTHORS}
    return tuple(people[orcid] for orcid in orcids)


#: program -> repository, mirroring atrium_paradata._REPO_URLS. Since the repository moves of
#: 2026-10-01 every current program has its own repository: `ocr-postprocess` (a rename of
#: atrium-alto-postprocess), `keyword-extract` (the keyword stage of atrium-nlp-enrich and
#: atrium-llm-enrich) and `digital-convert` (atrium-llm-enrich's converter). The predecessors
#: stay in the table, pointing at their archived repositories, because paradata written before
#: the move still name them; the crate then links the release that really produced the run. A
#: service's logger names its program
#: `<program>-api`; `_repository()` strips that suffix before looking it up here.
REPO_URLS: Dict[str, str] = {
    "ocr-postprocess": "https://github.com/ufal/atrium-ocr-postprocess",
    "alto-postprocess": "https://github.com/ufal/atrium-alto-postprocess",
    "page-classification": "https://github.com/ufal/atrium-page-classification",
    "translator": "https://github.com/ufal/atrium-translator",
    "nlp-enrich": "https://github.com/ufal/atrium-nlp-enrich",
    "keyword-extract": "https://github.com/ufal/atrium-keyword-extract",
    "llm-enrich": "https://github.com/ufal/atrium-llm-enrich",
    "digital-convert": "https://github.com/ufal/atrium-digital-convert",
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
)

#: The source thesauri a record names concepts of, by id, in `enrichment.items[].teater_category_ids`
#: (keyword-extract writes them): source -> (concept namespace, the ATRIUM harvest scheme the
#: released vocabulary asset files them under). The concepts keep their own URIs, so the crate
#: references them and mints nothing. `teater_category` itself is a source label, not an ATRIUM
#: theme (V-5 in docs/skos_strategy.md), so it never becomes an ATRIUM URI either.
SOURCE_CONCEPTS: Dict[str, Tuple[str, str]] = {
    "amcr": (getattr(atrium_vocab, "NS", {}).get("amcr", "https://api.aiscr.cz/id/"), "amcr-harvest"),
    "teater": (getattr(atrium_vocab, "NS", {}).get("teater", "https://teater.aiscr.cz/id/"), "teater-harvest"),
}

#: Terms this module emits that the RO-Crate 1.2 context does not define, declared in the
#: crate's own `@context` so the graph stays interpretable without a side agreement. The
#: Workflow Run terms are declared here rather than through their remote context, so a crate
#: depends on exactly one remote document. `paradataRecord` and `recordFields` are JSON literals
#: (JSON-LD 1.1): the run's whole paradata record, and the record blocks and fields a run wrote
#: (#84 §4). `stored` is false on an input or output that is not stored (D23).
_LOCAL_CONTEXT: Dict[str, Any] = {
    "@version": 1.1,
    "atrium": ATRIUM_BASE,
    "sha512": ATRIUM_BASE + "term/sha512",
    "stored": ATRIUM_BASE + "term/stored",
    "recordFields": {"@id": ATRIUM_BASE + "term/recordFields", "@type": "@json"},
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

#: A SHA-512 digest as the paradata carry it: 128 lower-case hex characters.
_SHA512 = re.compile(r"^[0-9a-f]{128}$")

#: The JSON-literal properties: objects in a nested action, JSON text in a (flat) crate.
_JSON_LITERALS = ("paradataRecord", "recordFields")

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


def _date_published(actions: Iterable[Mapping[str, Any]]) -> str:
    """The newest run's end date — never `datetime.now()`.

    RO-Crate requires `datePublished` on the root. Reading the clock would make the crate
    non-reproducible: exporting an archived record twice would produce two different crates.
    Since record schema 2.0 the record carries no stamps, so the runs that wrote it say when it was
    finished; without any, the date is the epoch, stated rather than invented.
    """
    ends = [str(a.get("endTime") or "") for a in actions if isinstance(a, Mapping)]
    newest = max((e for e in ends if e), default="")
    return newest[:10] if len(newest) >= 10 else "1970-01-01"


def _controlled_values(record: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Every (scheme, value) pair the record actually carries, deduplicated and sorted.

    Deliberately tolerant: a value the registry has never heard of is still emitted as a
    DefinedTerm under its scheme. `atrium_vocab.validate_labels()` reports rather than raises,
    and outside `lines[].categ` (the five categories, D18 rule 2) the record schema adds no `enum`
    for a controlled term, so a naming slip is an advisory rather than a stalled pipeline — a crate
    exporter must not be stricter than the contract it exports.
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
    return sorted(found)


def _source_concepts(record: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    """Every (source, id, label) the record's enrichment items name, deduplicated and sorted.

    The ids come from `teater_category_ids`; an item without them contributes nothing, since its
    `teater_category` label alone cannot be resolved without guessing. When two items name the
    same concept under different labels, the first label in sort order is kept, so the output
    stays deterministic.
    """
    labels: Dict[Tuple[str, str], str] = {}
    for item in (record.get("enrichment") or {}).get("items") or []:
        if not isinstance(item, dict):
            continue
        label = item.get("teater_category")
        for ref in item.get("teater_category_ids") or []:
            if not isinstance(ref, dict):
                continue
            source, concept_id = ref.get("source"), ref.get("id")
            if source not in SOURCE_CONCEPTS or not isinstance(concept_id, str) or not concept_id:
                continue
            name = label if isinstance(label, str) and label else concept_id
            key = (source, concept_id)
            labels[key] = min(labels.get(key, name), name)
    return sorted((source, concept_id, name) for (source, concept_id), name in labels.items())


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


def _person_entities(programs: Iterable[str]) -> List[Dict[str, Any]]:
    """The `Person` entities of the given programs' authors, each once, in CITATION.cff order."""
    people: Dict[str, Dict[str, Any]] = {}
    for program in programs:
        for a in authors_of(program):
            people.setdefault(a["orcid"], {"@id": a["orcid"], "@type": "Person", "name": a["name"]})
    return list(people.values())


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
    and its own authors (`authors_of`) as its `creator`. Without paradata the version is `UNRECORDED_VERSION`
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
            "creator": _refs(a["orcid"] for a in authors_of(program)),
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


# ──────────────────────────────────────────────────────────────────────────────
# One run, as a CreateAction (atrium-project#71 R2)
# ──────────────────────────────────────────────────────────────────────────────


def file_entity(
    name: str,
    data: Optional[bytes] = None,
    *,
    media_type: Optional[str] = None,
    sha512: Optional[str] = None,
    size: Optional[int] = None,
    stored: bool = True,
) -> Dict[str, Any]:
    """An input or output of a run, for `create_action(inputs=…, outputs=…)` (#84 §4, §6).

    `name` is the AMČR distribution name (`atrium/document-json`, `atr/docproc-json`,
    `atr/teitok-xml`, `metadata`, `metadata-en`). The file is named by its SHA-512 — computed from
    `data`, or given — never by a local path or a SHA-256 `ni:` URI. The `@id` joins name and
    digest, so the output of one step and the input of the next are one entity. A file that is not
    stored (D23) carries `stored: false` and is still named by its SHA-512. A `CreativeWork`, not a
    `File`: a run's inputs and outputs are not data entities of whichever crate embeds the action.
    """
    if data is not None:
        sha512 = hashlib.sha512(data).hexdigest()
        size = len(data)
    sha512 = str(sha512 or "").lower()
    if not _SHA512.match(sha512):
        raise ValueError("{}: an input or output is named by its SHA-512 (128 hex characters)".format(name))
    if not name:
        raise ValueError("an input or output is named by its AMČR distribution name")
    return _prune(
        {
            "@id": "#{}@{}".format(name, sha512[:16]),
            "@type": "CreativeWork",
            "name": name,
            "encodingFormat": media_type,
            "contentSize": str(size) if size is not None else None,
            "sha512": sha512,
            "stored": None if stored else False,
        }
    )


def record_entity(doc_id: str, data: Optional[bytes] = None, *, sha512: Optional[str] = None) -> Dict[str, Any]:
    """The record version a run read (the seed, or the previous step's record), named by its
    SHA-512 like any input, with the document's id."""
    entity = file_entity(RECORD_DISTRIBUTION, data, media_type="application/json", sha512=sha512)
    entity["identifier"] = doc_id
    return entity


def record_result(
    record_fields: Mapping[str, Iterable[str]],
    data: Optional[bytes] = None,
    *,
    sha512: Optional[str] = None,
    stored: bool = True,
) -> Dict[str, Any]:
    """The record a run wrote, as its `result`: the distribution by SHA-512 and the blocks AND
    fields the run wrote (`DocumentRecord.written()`, #84 §4). Replaces the per-block stamps."""
    entity = file_entity(RECORD_DISTRIBUTION, data, media_type="application/json", sha512=sha512, stored=stored)
    entity["recordFields"] = {str(b): [str(f) for f in fields] for b, fields in sorted(record_fields.items())}
    return entity


def _entity_from_paradata(item: Mapping[str, Any]) -> Dict[str, Any]:
    """One paradata 3.0 `inputs`/`outputs` entry as an action's object or result."""
    if item.get("record_fields") is not None:
        return record_result(item["record_fields"], sha512=item.get("sha512"), stored=bool(item.get("stored", True)))
    return file_entity(
        str(item.get("name") or ""),
        media_type=item.get("media_type"),
        sha512=item.get("sha512"),
        stored=bool(item.get("stored", True)),
    )


def action_from_paradata(paradata: Mapping[str, Any]) -> Dict[str, Any]:
    """A paradata 3.0 record (`ParadataLogger.record`) as its `CreateAction`: `inputs` become the
    `object`, `outputs` the `result`, `action_status`/`error` the status. The usual way to build one."""
    pd = dict(paradata or {})
    return create_action(
        pd,
        inputs=[_entity_from_paradata(i) for i in pd.get("inputs") or [] if isinstance(i, Mapping)],
        outputs=[_entity_from_paradata(o) for o in pd.get("outputs") or [] if isinstance(o, Mapping)],
        status="failed" if pd.get("action_status") == "failed" else "completed",
        error=pd.get("error") or None,
    )


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

    * `@id` — the run's `run_uuid` (`action_id` overrides it);
    * `instrument` — the tool (`version`, release-page id, its authors as `creator`) and
      `containerImage` — the image the service runs as (ATRIUM_RUNNER_IMAGE);
    * `object` / `result` — `inputs` / `outputs` (`file_entity()`, `record_entity()`,
      `record_result()`), by distribution name and SHA-512;
    * `agent` — ATRIUM_RUN_AGENT, only when it is set;
    * `startTime` / `endTime`, `actionStatus` (`completed` or `failed`, with `error`);
    * `paradataRecord` — the paradata record itself, so nothing it says is lost.
    """
    if status not in ("completed", "failed"):
        raise ValueError("status must be 'completed' or 'failed', not {!r}".format(status))
    pd = dict(paradata or {})
    program = str(pd.get("program") or pd.get("pipeline") or "unknown")
    tool = _tool_entity(program, pd)
    tool["creator"] = _person_entities([program])
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
    only on failure, typed `object`/`result` entries named by SHA-512 (never an `ni:` id, #84 §4),
    the record's result naming its fields, and a paradata record with no calculated licence (D2).
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
                continue
            if str(item["@id"]).startswith("ni:"):
                problems.append("{} entry {!r} is named by a SHA-256 ni: URI".format(prop, item["@id"]))
            if not _SHA512.match(str(item.get("sha512") or "")):
                problems.append("{} entry {!r} is not named by its SHA-512".format(prop, item["@id"]))
            if not item.get("name"):
                problems.append("{} entry {!r} has no distribution name".format(prop, item["@id"]))
            if prop == "result" and item.get("name") == RECORD_DISTRIBUTION and not item.get("recordFields"):
                problems.append("the record result {!r} does not name the blocks and fields written".format(item["@id"]))
    # No result is not a problem: `result` names stored outputs only (#84 §4), and a call that
    # answered without writing a stored distribution (a classification with no record asked for)
    # has none.
    agent = action.get("agent")
    if agent is not None and (not isinstance(agent, dict) or agent.get("@type") not in ("Person", "Organization")):
        problems.append("agent is not a Person or Organization")
    record = action.get("paradataRecord")
    if not isinstance(record, dict):
        problems.append("no paradataRecord")
    elif {"license", "license_url", "license_detail"} & set(record):
        problems.append("the paradata record carries a calculated licence (D2)")
    return problems


def _flatten(node: Any, out: Dict[str, Dict[str, Any]]) -> Any:
    """Replace every nested entity under `node` by a reference, collecting the entities in `out`."""
    if isinstance(node, list):
        return [_flatten(item, out) for item in node]
    if isinstance(node, dict) and "@id" in node and set(node) != {"@id"}:
        entity = {k: (v if k in _JSON_LITERALS else _flatten(v, out)) for k, v in node.items() if k != "@context"}
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
        # travels as its JSON text; in the nested action, which is not a crate, as the object. The
        # record fields a run wrote (`recordFields`) the same.
        for key in _JSON_LITERALS:
            if isinstance(entity.get(key), dict):
                entity[key] = json.dumps(entity[key], ensure_ascii=False, sort_keys=True)
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
    actions: Sequence[Mapping[str, Any]] = (),
    name: Optional[str] = None,
    description: Optional[str] = None,
    fragment: bool = False,
    source_id: Optional[str] = None,
    license: Optional[Union[str, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """ÚFAL's RO-Crate metadata graph for ONE document record and the runs that wrote it.

    `record` is a parsed `<doc_id>.document.json` (record schema 2.0: content only). The runs
    come from `paradata` (paradata 3.0 records, as `ParadataLogger.record` has them; each becomes
    its action through `action_from_paradata()`) and/or `actions` (CreateActions as the services
    returned them). Each action names what it read and wrote by distribution and SHA-512, and the
    record's result names the blocks and fields written — the record itself carries no stamps.
    Without runs the crate is still complete, just silent about how the record was made.

    **Not AMČR's archival crate.** AMČR builds that one; its root is the whole AMČR record and
    states the document's licence. What AMČR reuses is the **fragment** (`fragment=True`): the
    actions, the software, the controlled terms — no metadata descriptor, no root, no block
    entities. The host lists the actions in its `mentions`, the terms in its `about`, and declares
    the Process Run Crate profile; `source_id` names the host's entity for the original, which then
    replaces `#source`. `wrap_fragment()` shows the whole of it.

    The root names a licence only when the caller asserts one (`license`); the tools assert none,
    so without one it names `#no-licence-asserted`.
    """
    doc_id = str(record.get("doc_id") or "")
    run_actions = [dict(a) for a in actions] + [action_from_paradata(pd) for pd in _paradata_records(paradata)]

    graph: List[Dict[str, Any]] = []
    root: Dict[str, Any] = _prune(
        {
            "@id": ROOT_ID,
            "@type": "Dataset",
            "identifier": _ref("#doc_id") if doc_id else None,
            "name": name or "ATRIUM document record {}".format(doc_id),
            "description": description
            or (
                "One ATRIUM document record (content only, record schema {}) and the runs that wrote "
                "it, each described by its Process Run Crate CreateAction. The rights in the "
                "document are stated by the archive, not by this crate.".format(record.get("schema_version") or "")
            ),
            "datePublished": _date_published(run_actions),
            "schemaVersion": str(record.get("schema_version") or ""),
            "conformsTo": _ref(PROCESS_RUN_PROFILE),
        }
    )

    # ── the original input ────────────────────────────────────────────────────
    # NOT a data entity: the original is archive-managed and not in the crate. It is what the
    # crate is BASED ON. A host crate that holds the original names it with `source_id`.
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
                    "sha512": source.get("sha512"),
                    "description": (
                        "The ORIGINAL input this record was built from, identified by doc_id and its "
                        "SHA-512. Archive-managed and not part of this crate. Text obtained as: {}.".format(
                            source.get("origin") or "unrecorded"
                        )
                    ),
                }
            )
        )
    if original:
        root["isBasedOn"] = _ref(original)

    # ── one CreateAction per run, flattened: tools, images, agents, files ─────
    mentions: List[str] = []
    entities: Dict[str, Dict[str, Any]] = {}
    for action in run_actions:
        flat = action_fragment(action)
        for entity in flat["@graph"]:
            entities.setdefault(entity["@id"], {}).update(entity)
            if entity.get("@type") == "CreateAction":
                mentions.append(entity["@id"])
    graph.extend(entities.values())
    programs = [e["alternateName"] for e in entities.values() if e.get("@type") == "SoftwareApplication" and e.get("alternateName")]
    if programs:
        graph.extend(_person_entities(programs))

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
    for source_name, concept_id, label in _source_concepts(record):
        namespace, harvest = SOURCE_CONCEPTS[source_name]
        schemes.add(harvest)
        about.append(namespace + concept_id)
        graph.append(
            {
                "@id": namespace + concept_id,
                "@type": "DefinedTerm",
                "name": label,
                "termCode": concept_id,
                "inDefinedTermSet": _ref(_scheme_uri(harvest)),
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
    root["license"] = _root_license(license, graph, "The licence the caller asserted.")
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
        "datePublished": _run_date_published(run_paradata),
        "conformsTo": _ref(PROCESS_RUN_PROFILE),
    }

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
                    "actionStatus": FAILED_ACTION_STATUS if stage.get("action_status") == "failed" else COMPLETED_ACTION_STATUS,
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
        graph.extend(_person_entities(t["alternateName"] for t in tools.values()))
    if mentions:
        root["mentions"] = _refs(sorted(set(mentions)))

    for key, prop in (("start_time", "startTime"), ("end_time", "endTime")):
        if _timestamp(run.get(key)):
            root[prop] = _timestamp(run.get(key))

    graph.append(_identifier_entity("#run_id", "run_uuid" if run.get("run_uuid") else "run_id",
                                    str(run.get("run_uuid") or "") or run_id or "atrium-run"))
    root["license"] = _root_license(None, graph, "")
    graph.append(_profile_entity())
    graph.append(root)
    return _assemble(graph)


def _run_date_published(run_paradata: Optional[Dict[str, Any]]) -> str:
    """The run's end date from its merged paradata; the epoch when it has none (never the clock)."""
    explicit = str((run_paradata or {}).get("end_time") or "")
    return explicit[:10] if len(explicit) >= 10 else "1970-01-01"


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
    license: Optional[Union[str, Mapping[str, Any]]] = None,
    date_published: str,
) -> Dict[str, Any]:
    """A fragment under a stub root: the smallest complete crate that embeds it.

    It is what a host does with a fragment, done once: its entities go into the graph, the root
    lists every `CreateAction` in `mentions`, every `DefinedTerm` in `about` and `#source` (when
    present) in `isBasedOn`, and declares the Process Run Crate profile. A fragment has no root,
    so this is also the only way to validate one (`tools/ci/rocrate_check.py`).

    `license` is the host's own assertion (AMČR states the document's licence on its record
    crate). Without one the stub root, like every root here, names `#no-licence-asserted` (D2).
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
            "license": _root_license(license, graph, "The licence of this crate."),
            "conformsTo": _ref(PROCESS_RUN_PROFILE),
            "mentions": _refs(sorted(by_type.get("CreateAction", []))),
            "about": _refs(sorted(by_type.get("DefinedTerm", []))),
            "isBasedOn": _ref(SOURCE_ID) if SOURCE_ID in {e["@id"] for e in graph} else None,
        }
    )
    return _assemble(graph + [_profile_entity(), root])


def _license_value(license: Union[str, Mapping[str, Any]]) -> Any:
    """A root `license` a host asserts: a reference for a URL, a literal for a bare name."""
    if not isinstance(license, str):
        return dict(license)
    return _ref(license) if license.startswith("http") else {"@value": license}


def _no_licence_entity() -> Dict[str, str]:
    """The entity `NO_LICENCE_ID` names: no licence asserted, and nothing computed in its place."""
    return {
        "@id": NO_LICENCE_ID,
        "@type": "CreativeWork",
        "name": "No licence asserted",
        "description": (
            "No licence is asserted for this crate. The rights in its contents depend on the source "
            "and on AMČR's permission to process it. Each run's paradata lists the components that "
            "ran, each with its own terms."
        ),
    }


def _root_license(
    license: Optional[Union[str, Mapping[str, Any]]], graph: List[Dict[str, Any]], description: str
) -> Any:
    """A root's `license`: the caller's assertion, or `#no-licence-asserted` without one. The entity
    it names (the placeholder, or an asserted URL) is added to `graph` unless already there."""
    ids = {e.get("@id") for e in graph}
    if not license:
        if NO_LICENCE_ID not in ids:
            graph.append(_no_licence_entity())
        return _ref(NO_LICENCE_ID)
    if isinstance(license, str) and license.startswith("http") and license not in ids:
        graph.append({"@id": license, "@type": "CreativeWork", "name": license, "description": description})
    return _license_value(license)


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


# ──────────────────────────────────────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────────────────────────────────────

_SAMPLE_RUN_UUIDS = {
    "ocr-postprocess": "urn:uuid:5a1e0c1e-1b2d-4c3e-8f40-0a1b2c3d4e5f",
    "nlp-enrich": "urn:uuid:9c8b7a6d-5e4f-4321-9abc-def012345678",
}

#: Sample digests: the original, the stored docproc, and two record versions.
_SAMPLE_SHA512 = {
    "original": "b" * 128,
    "docproc": "c" * 128,
    "record-1": "d" * 128,
    "record-2": "e" * 128,
    "teitok": "f" * 128,
}


def _sample_record() -> Dict[str, Any]:
    """A record-2.0 sample exercising every mapping branch the crate has."""
    return {
        "schema_version": "2.0",
        "record_type": "atrium-document",
        "doc_id": "CTX000000001",
        "source": {
            "sha512": _SAMPLE_SHA512["original"],
            "filename": "CTX000000001.pdf",
            "media_type": "application/pdf",
            "page_count": 1,
            "origin": "atr",
        },
        "page_categories": {"1": "TEXT"},
        "pages": [{"page": "1", "page_index": 1, "content_status": "text", "quality_band": "Clear", "category": "TEXT"}],
        "content": {"text": "Náčrt sondy."},
        "lines": [{"page": "1", "line": 1, "text": "Náčrt sondy.", "block": "p1-b0", "text_source": "atr", "bbox_precision": "absent"}],
        "entities": [{"surface": "Praha", "type_teitok": "LOC", "type_cnec": "gu", "page": "1", "line": 1}],
    }


def _sample_paradata() -> List[Dict[str, Any]]:
    """The paradata 3.0 records of the sample record's two runs, as `ParadataLogger.record` has them."""
    return [
        {
            "schema_version": "3.0",
            "program": "ocr-postprocess",
            "tool_version": "v1.16.0-beta",
            "repository": "https://github.com/ufal/atrium-ocr-postprocess",
            "docker_image": "ghcr.io/ufal/atrium-ocr-postprocess-api:1.16.0-beta",
            "python_version": "3.11.15 (main) [GCC]",
            "run_id": "260724-101112",
            "run_uuid": _SAMPLE_RUN_UUIDS["ocr-postprocess"],
            "run_agent": "https://ror.org/00x0x0x00",
            "action_status": "completed",
            "start_time": "2026-07-24T10:11:12.000001+00:00",
            "end_time": "2026-07-24T10:11:40.250000+00:00",
            "parameters": {"rules": "frozen-2026-11"},
            "inputs": [
                {"name": "atr/docproc-json", "sha512": _SAMPLE_SHA512["docproc"], "media_type": "application/json", "stored": True},
                {"name": "atrium/document-json", "sha512": _SAMPLE_SHA512["record-1"], "stored": True},
            ],
            "outputs": [
                {
                    "name": "atrium/document-json",
                    "sha512": _SAMPLE_SHA512["record-2"],
                    "stored": True,
                    "record_fields": {"lines": ["page", "line", "text", "block", "text_source", "bbox_precision"], "pages": ["page", "page_index", "content_status"]},
                }
            ],
            "components": [{"name": "fasttext", "license": "CC BY-NC 4.0"}],
            "limits": [],
        },
        {
            "schema_version": "3.0",
            "program": "nlp-enrich",
            "tool_version": "1.4.0",
            "repository": "https://github.com/ufal",
            "docker_image": "ghcr.io/ufal/atrium-nlp-enrich-api:1.4.0",
            "python_version": "3.11.15",
            "run_id": "260724-101500",
            "run_uuid": _SAMPLE_RUN_UUIDS["nlp-enrich"],
            "run_agent": "https://ror.org/00x0x0x00",
            "action_status": "completed",
            "start_time": "2026-07-24T10:15:00+00:00",
            "end_time": "2026-07-24T10:16:02+00:00",
            "inputs": [{"name": "atrium/document-json", "sha512": _SAMPLE_SHA512["record-2"], "stored": True}],
            "outputs": [
                {"name": "atrium/document-json", "sha512": "a" * 128, "stored": True, "record_fields": {"entities": ["surface", "type_teitok", "type_cnec", "page", "line"]}},
                {"name": "atr/teitok-xml", "sha512": _SAMPLE_SHA512["teitok"], "media_type": "application/tei+xml", "stored": True},
            ],
            "components": [{"name": "UDPipe 2", "license": "CC BY-NC-SA 4.0"}],
            "limits": [],
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
    for prop in ("@type", "name", "description", "datePublished"):
        if not root.get(prop):
            problems.append("root is missing required property {!r}".format(prop))
    if root.get("@type") != "Dataset":
        problems.append("root @type is not Dataset")
    if root.get("conformsTo") != {"@id": PROCESS_RUN_PROFILE} or PROCESS_RUN_PROFILE not in by_id:
        problems.append("root does not declare the Process Run Crate profile as an entity")
    if "author" in root:
        problems.append("root carries author: the ATRIUM authors are each tool's creator")
    if root.get("license") != {"@id": NO_LICENCE_ID} or by_id.get(NO_LICENCE_ID) != _no_licence_entity():
        problems.append("the root does not name #no-licence-asserted, its entity (RO-Crate 1.2 rule 13.3; D2)")

    # 2. No dangling references.
    for ref in sorted(set(_walk_refs(graph))):
        if ref in by_id or ref.startswith(("http://", "https://")):
            continue
        problems.append("dangling reference {!r} — no entity and not an absolute URI".format(ref))

    # 3. Nothing of record schema 1.x or of a calculated licence leaks into a crate.
    text = to_json(crate)
    for banned in ("ni:///", "#block-", "effective_license", "most restrictive", "license_detail"):
        if banned in text:
            problems.append("the crate carries {!r}".format(banned))
    if SOURCE_ID in {r["@id"] for r in root.get("hasPart", [])}:
        problems.append("the archive-managed original appears in hasPart")

    # 4. Determinism — the property the byte-comparison gate depends on.
    if to_json(document_crate(record, paradata=_sample_paradata())) != text:
        problems.append("document_crate() is not deterministic")
    if root.get("datePublished") != "2026-07-24":
        problems.append("datePublished was not derived from the newest run's end time")

    # 5. Runs, tools, people: stable ids, `version`, authors as creator, profile time format, and
    #    the record's results naming their fields.
    for program, run_uuid in _SAMPLE_RUN_UUIDS.items():
        action = by_id.get(run_uuid, {})
        if action.get("@type") != "CreateAction":
            problems.append("the {} run lost its CreateAction (or its run_uuid id)".format(program))
        for prop in ("startTime", "endTime"):
            if action.get(prop) and not _ACTION_TIME.match(action[prop]):
                problems.append("{} {} is not in the profile's format".format(program, prop))
        for res in action.get("result") or []:
            entity = by_id.get(res["@id"], {})
            if entity.get("name") == RECORD_DISTRIBUTION and not isinstance(entity.get("recordFields"), str):
                problems.append("{}'s record result does not carry its fields as JSON text".format(program))
    tools = [e for e in graph if e.get("@type") == "SoftwareApplication"]
    for tool in tools:
        if not str(tool["@id"]).startswith("https://") or tool.get("version") in (None, "", UNRECORDED_VERSION):
            problems.append("tool {} lacks an absolute id or a recorded version".format(tool["@id"]))
        if "softwareVersion" in tool:
            problems.append("tool {} carries softwareVersion".format(tool["@id"]))
        if {c["@id"] for c in tool.get("creator", [])} != {a["orcid"] for a in authors_of(tool["alternateName"])}:
            problems.append("tool {} does not name its own authors as creator".format(tool["@id"]))
        if not all(isinstance(by_id.get(c["@id"]), dict) for c in tool.get("creator", [])):
            problems.append("tool {} names a creator the graph does not describe".format(tool["@id"]))
    if "https://github.com/ufal/atrium-nlp-enrich/releases/tag/v1.4.0" not in by_id:
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
            "pipeline_stages": [{"order": 1, "program": "ocr-postprocess", "run_id": "260724-101112"}],
        },
        paradata_refs=["paradata/260724-101112_pipeline-run.json"],
    )
    rroot = {e["@id"]: e for e in rc["@graph"]}[ROOT_ID]
    if {"@id": "CTX000000001/"} not in rroot.get("hasPart", []):
        problems.append("run crate does not reference the per-document crate")
    if rroot.get("conformsTo") != {"@id": PROCESS_RUN_PROFILE}:
        problems.append("run crate does not declare the process-run profile")
    if rroot.get("license") != {"@id": NO_LICENCE_ID} or NO_LICENCE_ID not in {e["@id"] for e in rc["@graph"]}:
        problems.append("run crate root does not name #no-licence-asserted")

    # 8. The fragment: no descriptor, no root, no data entity; wrapped, it is a crate again.
    frag = document_crate(record, paradata=_sample_paradata(), fragment=True)
    frag_ids = {e["@id"] for e in frag["@graph"]}
    if {METADATA_FILENAME, ROOT_ID} & frag_ids or any(e.get("@type") == "File" for e in frag["@graph"]):
        problems.append("the fragment carries a descriptor, a root or a data entity")
    wrapped = wrap_fragment(
        frag,
        name="stub",
        description="stub",
        license="https://creativecommons.org/licenses/by-nc/4.0/",
        date_published="2026-07-24",
    )
    wroot = {e["@id"]: e for e in wrapped["@graph"]}[ROOT_ID]
    if not set(_SAMPLE_RUN_UUIDS.values()) <= {r["@id"] for r in wroot.get("mentions", [])}:
        problems.append("wrap_fragment() does not mention every action")
    if wroot.get("license") != {"@id": "https://creativecommons.org/licenses/by-nc/4.0/"}:
        problems.append("wrap_fragment() drops the licence its host asserted")
    if NO_LICENCE_ID in frag_ids:
        problems.append("the fragment carries a licence entity: the host's root decides its own")

    # 9. A service's CreateAction meets the shared contract; a 1.x-shaped one does not.
    action = create_action(
        _sample_paradata()[0],
        inputs=[
            file_entity("atr/docproc-json", b"{}", media_type="application/json"),
            record_entity("C-1", b"{}"),
            file_entity("atr/docproc-json", sha512="1" * 128, stored=False),
        ],
        outputs=[record_result({"lines": ["page", "line", "text"]}, b"{}")],
    )
    for problem in action_problems(action):
        problems.append("create_action(): " + problem)
    for problem in action_problems(action_from_paradata(_sample_paradata()[1])):
        problems.append("action_from_paradata(): " + problem)
    old = dict(action, result=[{"@id": "ni:///sha-256;x", "@type": "CreativeWork", "name": "pages"}])
    old["paradataRecord"] = dict(action["paradataRecord"], license="CC BY-NC 4.0")
    if len(action_problems(old)) < 2:
        problems.append("action_problems() accepts an ni: result or a calculated licence")

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
        help="a paradata 3.0 record: with --document, each run's (repeat it); with --run, the merged run record",
    )
    p.add_argument("--fragment", action="store_true", help="with --document: entities only, no descriptor or root")
    p.add_argument("--wrap", action="store_true", help="with --fragment: put the fragment under a stub root")
    p.add_argument("--source-id", metavar="ID", default=None, help="with --fragment: the host's id of the original")
    p.add_argument("--data-dir", metavar="DIR", default=None, help="with --run: where the paradata files are (contentSize)")
    p.add_argument("--out-dir", metavar="DIR", default=None, help="write ro-crate-metadata.json here")
    args = p.parse_args(argv)

    if args.selftest:
        return _selftest()

    if args.document:
        record = _load(args.document)
        runs = [_load(path) for path in args.paradata]
        crate = document_crate(record, paradata=runs, fragment=args.fragment, source_id=args.source_id)
        if args.fragment and args.wrap:
            crate = wrap_fragment(
                crate,
                name="ATRIUM record fragment {} (stub root)".format(record.get("doc_id") or ""),
                description="The fragment of one ATRIUM record under a stub root, as a host crate embeds it.",
                date_published=_date_published(action_from_paradata(pd) for pd in runs),
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
