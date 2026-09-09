"""
atrium_rocrate.py — the RO-Crate view of an ATRIUM document record.

WHY THIS MODULE EXISTS
======================
``atrium_document.py`` produces a per-document record that ``docs/document_schema.md`` calls
"one FAIR, versioned JSON for search and catalogue export". It has never actually been
exported to anything. The DMP's WP3 standards column names **RO-Crate**, and until this
module there were zero occurrences of the string in any of the six repos
(ufal/atrium-project#54) — a new commitment, not continuity.

An RO-Crate is a directory with an ``ro-crate-metadata.json`` at its root: JSON-LD naming
the dataset, the files in it, who made them and with what. Everything such a file needs,
ATRIUM already records:

    source            -> the original input, by sha256 (archive-managed, never a path)
    derived_from      -> the persistent step outputs = the crate's data entities
    regenerable       -> recipes for files that DO NOT EXIST = never data entities
    provenance        -> the accreted licence union + one entry per contributing run
    assembled.blocks  -> per-block stamps = the granularity a CreateAction chain needs
    atrium_paradata   -> tool_version / repository / docker_image / runner_ref per run
    atrium_vocab      -> resolvable URIs for every controlled term the record carries

So this module maps rather than invents. Where a field does not exist it is **omitted**, not
guessed: a crate that quietly asserts a checksum nobody computed is worse than one that
admits the gap.

WHAT THIS MODULE IS NOT
=======================
* **Not a publisher.** It builds a metadata graph and can write it next to a directory of
  outputs. Nothing is uploaded, registered, or minted. Compare ``atrium_vocab.py``, which is
  explicitly "not a publication pipeline".
* **Not a second provenance model.** ``atrium_paradata.py`` stays the record of how a run
  behaved; this is a *view* over what is already recorded. CIDOC-CRM and PROV-O were
  deliberately rejected for the raw-extraction layer on 1.2-million-page throughput grounds,
  and nothing here reverses that — the crate is built once, at export time, from records that
  are already on disk.
* **Not a writer of document records.** It never mutates one. ``document_crate(record)``
  takes a record and returns a dict.

DEPENDENCIES
============
Standard library only. An RDF library is deliberately **not** introduced, for the same reason
``atrium_vocab.py`` refuses one: the serialisation here is small, closed and fully determined
by the record, and the output is drift-gated by byte comparison. A general-purpose serialiser
that emits blank-node identifiers or unordered graphs would fail that gate on every run.

``atrium_vocab`` is imported softly — with it, every controlled term in the record becomes a
``DefinedTerm`` with a resolvable ``https://w3id.org/atrium/`` URI; without it the crate is
still valid, just less semantic.

DETERMINISM
===========
Two calls on the same record produce byte-identical JSON. ``@graph`` is sorted by ``@id``
(with the two RO-Crate-mandated entities pinned first) and ``json.dumps`` runs with
``sort_keys=True``. ``datePublished`` is derived from the record's own newest block stamp, not
from the clock, so re-exporting an archived record reproduces the crate it shipped with.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import atrium_vocab
except ImportError:  # pragma: no cover - degraded mode, see module docstring
    atrium_vocab = None  # type: ignore

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

#: RO-Crate specification this module targets. Both strings move together, and moving them
#: is a deliberate act: `conformsTo` is what a consumer reads to decide how to interpret the
#: graph, so a bump is a compatibility statement about the crates already written, exactly
#: like SCHEMA_VERSION in atrium_document.py.
ROCRATE_VERSION = "1.1"
ROCRATE_CONTEXT = "https://w3id.org/ro/crate/1.1/context"
ROCRATE_CONFORMS_TO = "https://w3id.org/ro/crate/1.1"

#: Profile URI for the run-level crate, which is a *process* run rather than a workflow run
#: (ATRIUM's five stages are five containers driven by a shell/CI pipeline, not a CWL/WDL
#: workflow). VERIFY THIS PIN before the first published crate: the Workflow-Run RO-Crate
#: profiles are versioned independently of the RO-Crate spec itself.
PROCESS_RUN_PROFILE = "https://w3id.org/ro/wfrun/process/0.5"

#: The file every RO-Crate is identified by. Not configurable — it is the spec.
METADATA_FILENAME = "ro-crate-metadata.json"

#: The root data entity's `@id` is `./` by RO-Crate rule, never a name of our choosing.
ROOT_ID = "./"

#: ATRIUM's own namespace, so extension terms carry a resolvable prefix rather than being
#: dropped by a consumer that compacts strictly against the RO-Crate context. Kept in step
#: with atrium_vocab.SKOS_BASE, which is the one place an ATRIUM URI is rooted.
ATRIUM_BASE = getattr(atrium_vocab, "SKOS_BASE", "https://w3id.org/atrium/")

#: Authors, identical in every tool repo's CITATION.cff. ORCIDs are the `@id`s, which is the
#: RO-Crate convention and the reason no ATRIUM identifier has to be minted for a person.
AUTHORS: Tuple[Dict[str, str], ...] = (
    {"orcid": "https://orcid.org/0009-0002-4773-2797", "name": "Kateryna Lutsai"},
    {"orcid": "https://orcid.org/0000-0002-6895-8536", "name": "Pavel Straňák"},
    {"orcid": "https://orcid.org/0009-0005-8722-0245", "name": "David Novák"},
    {"orcid": "https://orcid.org/0000-0001-5718-9447", "name": "Dana Křivánková"},
)

#: program -> repository, mirroring atrium_paradata._REPO_URLS. `digital-convert` is a ROLE
#: that lives in atrium-llm-enrich (see the BLOCK_OWNERS note in atrium_document.py), so it
#: maps to that repo rather than one of its own.
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

#: Extension terms this module emits that the RO-Crate 1.1 context does not define. Declared
#: in the crate's own `@context` so the graph stays interpretable without a side agreement.
_LOCAL_CONTEXT: Dict[str, str] = {
    "atrium": ATRIUM_BASE,
    "regenerableFrom": ATRIUM_BASE + "term/regenerableFrom",
    "recordBlock": ATRIUM_BASE + "term/recordBlock",
    "schemaVersion": ATRIUM_BASE + "term/schemaVersion",
}


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
        entity = _prune({"@id": url, "@type": "CreativeWork", "name": name, "identifier": name})
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


def _tool_id(program: str) -> str:
    return "#tool-" + program


def _run_id_for(program: str, run_id: str) -> str:
    return "#run-{}-{}".format(program, run_id or "unknown")


def _block_id(name: str) -> str:
    return "#block-" + name


# ──────────────────────────────────────────────────────────────────────────────
# The per-document crate
# ──────────────────────────────────────────────────────────────────────────────


def document_crate(
    record: Dict[str, Any],
    *,
    paradata: Optional[Dict[str, Dict[str, Any]]] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Dict[str, Any]:
    """The RO-Crate metadata graph for ONE document record.

    `record` is a parsed `<doc_id>.document.json`. `paradata` is an optional
    `{program: paradata_record}` map — when supplied, each `SoftwareApplication` gains the
    `tool_version` / `repository` / `docker_image` / `runner_ref` that only paradata carries.
    The crate is complete without it; it is simply less specific about what ran.

    **The reference-discipline rule, expressed in RO-Crate terms.** `derived_from` values are
    data entities and go in `hasPart`. `regenerable` entries are recipes for files that do not
    exist, so they are contextual entities and MUST NOT appear in `hasPart` — a crate that
    lists a file it does not contain is invalid, and this is the exact failure the record's
    "transient artifacts are never referenced" rule exists to prevent, one layer up.
    """
    doc_id = str(record.get("doc_id") or "")
    provenance = record.get("provenance") or {}
    blocks = (record.get("assembled") or {}).get("blocks") or {}

    graph: List[Dict[str, Any]] = []
    root: Dict[str, Any] = {
        "@id": ROOT_ID,
        "@type": "Dataset",
        "identifier": doc_id,
        "name": name or "ATRIUM document record {}".format(doc_id),
        "description": description
        or (
            "Per-document aggregate record accreted across the ATRIUM pipeline, exported as "
            "RO-Crate. Blocks reflect contributed steps only; a block is absent until its tool "
            "has run."
        ),
        "datePublished": _date_published(record),
        "schemaVersion": str(record.get("schema_version") or ""),
        "author": _refs(a["orcid"] for a in AUTHORS),
    }

    # ── authors ───────────────────────────────────────────────────────────────
    for author in AUTHORS:
        graph.append({"@id": author["orcid"], "@type": "Person", "name": author["name"]})

    # ── licence ───────────────────────────────────────────────────────────────
    license_ref, license_entities = _license_entity(provenance)
    if license_ref is not None:
        root["license"] = license_ref
    graph.extend(license_entities)

    # ── the original input ────────────────────────────────────────────────────
    # NOT a data entity: `source` carries no path by design ("originals are archive-managed,
    # not pipeline-local"), so the file is not in the crate and cannot be in `hasPart`. It is
    # what the crate is BASED ON, which is a different and truthful claim.
    source = record.get("source") or {}
    if source:
        graph.append(
            _prune(
                {
                    "@id": "#source",
                    "@type": "CreativeWork",
                    "name": source.get("filename") or doc_id,
                    "identifier": doc_id,
                    "encodingFormat": source.get("media_type"),
                    "sha256": source.get("sha256"),
                    "description": (
                        "The ORIGINAL input this record was built from, identified by doc_id + "
                        "sha256. Archive-managed and not part of this crate. Acquired as: {}.".format(
                            source.get("origin") or "unrecorded"
                        )
                    ),
                }
            )
        )
        root["isBasedOn"] = _ref("#source")

    # ── data entities: persistent step outputs only ───────────────────────────
    has_part: List[str] = []
    for key in sorted(record.get("derived_from") or {}):
        path = str((record.get("derived_from") or {})[key])
        if not path:
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
                    "regenerableFrom": recipe.get("from"),
                    "softwareVersion": recipe.get("converter"),
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
    # the one thing the accretion design bought.
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
                    "creator": _ref(_tool_id(str(stamp.get("program") or "unknown"))),
                    "description": "Block {!r} of the document record, as stamped by assembled.blocks.".format(name_),
                }
            )
        )

    # ── one CreateAction per contributing run, one SoftwareApplication per tool ──
    programs = set()
    for contributor in provenance.get("contributors") or []:
        if not isinstance(contributor, dict):
            continue
        program = str(contributor.get("program") or "unknown")
        run_id = str(contributor.get("run_id") or "")
        programs.add(program)
        wrote = [b.strip() for b in str(contributor.get("blocks") or "").split(",") if b.strip()]
        action_id = _run_id_for(program, run_id)
        mentions.append(action_id)
        graph.append(
            _prune(
                {
                    "@id": action_id,
                    "@type": "CreateAction",
                    "name": "{} run {}".format(program, run_id or "unknown"),
                    "startTime": contributor.get("at"),
                    "endTime": contributor.get("at"),
                    "instrument": _ref(_tool_id(program)),
                    "object": _ref("#source") if source else None,
                    "result": _refs(_block_id(b) for b in wrote) or None,
                    "description": (
                        "Contributed block(s) {}. Run paradata: {}.".format(
                            ", ".join(wrote) or "none recorded",
                            contributor.get("paradata_ref") or "not recorded",
                        )
                    ),
                }
            )
        )

    # Programs named by a block stamp but absent from contributors[] still need a tool entity,
    # or `creator` above dangles. That happens on a record assembled from a baseline whose
    # contributor list a hand-built upload did not carry.
    for stamp in blocks.values():
        if isinstance(stamp, dict) and stamp.get("program"):
            programs.add(str(stamp["program"]))

    for program in sorted(programs):
        pd = (paradata or {}).get(program) or {}
        graph.append(
            _prune(
                {
                    "@id": _tool_id(program),
                    "@type": "SoftwareApplication",
                    "name": program,
                    "url": pd.get("repository") or REPO_URLS.get(program),
                    "softwareVersion": pd.get("tool_version"),
                    "identifier": pd.get("docker_image"),
                    "runtimePlatform": pd.get("python_version"),
                    "releaseNotes": pd.get("runner_ref"),
                }
            )
        )

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

    graph.append(root)
    return _assemble(graph, conforms_to=[ROCRATE_CONFORMS_TO])


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
) -> Dict[str, Any]:
    """The RO-Crate metadata graph for ONE PIPELINE RUN over several documents.

    The per-document crate is the primitive; this one references those crates rather than
    re-describing their contents, so a run crate stays the same size whether the run covered
    five documents or five thousand — which matters for a corpus the DMP sizes at 1.2 million
    pages.

    `run_paradata` is `merge_run_paradata()`'s output. Its `pipeline_stages[]` become the
    run-level `CreateAction` chain, and `intermediate_formats` / `statistics` are recorded on
    the root. Everything is optional: a run crate built from records alone is still valid, it
    simply says less about the run than about its results.
    """
    graph: List[Dict[str, Any]] = []
    doc_ids = [str(r.get("doc_id") or "") for r in records]
    run_id = str((run_paradata or {}).get("run_id") or "")

    root: Dict[str, Any] = {
        "@id": ROOT_ID,
        "@type": "Dataset",
        "identifier": run_id or "atrium-run",
        "name": name or "ATRIUM pipeline run {}".format(run_id or "(unidentified)"),
        "description": description
        or (
            "One ATRIUM pipeline run, packaged as RO-Crate: the per-document crates it "
            "produced, the paradata that records how it behaved, and the tool chain that ran."
        ),
        "datePublished": _run_date_published(records, run_paradata),
        "author": _refs(a["orcid"] for a in AUTHORS),
    }
    for author in AUTHORS:
        graph.append({"@id": author["orcid"], "@type": "Person", "name": author["name"]})

    provenance = (records[0].get("provenance") if records else {}) or {}
    license_ref, license_entities = _license_entity((run_paradata or {}) if run_paradata else provenance)
    if license_ref is None:
        license_ref, license_entities = _license_entity(provenance)
    if license_ref is not None:
        root["license"] = license_ref
    graph.extend(license_entities)

    # ── the per-document crates are the parts ─────────────────────────────────
    parts: List[str] = []
    for doc_id in sorted(set(d for d in doc_ids if d)):
        part_id = document_crate_dir.format(doc_id=doc_id).rstrip("/") + "/"
        parts.append(part_id)
        graph.append(
            {
                "@id": part_id,
                "@type": "Dataset",
                "identifier": doc_id,
                "name": "ATRIUM document record {}".format(doc_id),
                "conformsTo": _ref(ROCRATE_CONFORMS_TO),
                "description": "Nested per-document RO-Crate; see its own {}.".format(METADATA_FILENAME),
            }
        )
    for ref in sorted(set(paradata_refs)):
        parts.append(ref)
        graph.append(
            {
                "@id": ref,
                "@type": "File",
                "name": os.path.basename(ref),
                "encodingFormat": "application/json",
                "description": "atrium_paradata run record — how this run behaved.",
            }
        )
    if parts:
        root["hasPart"] = _refs(sorted(set(parts)))

    # ── the stage chain ───────────────────────────────────────────────────────
    mentions: List[str] = []
    programs = set()
    for stage in (run_paradata or {}).get("pipeline_stages") or []:
        if not isinstance(stage, dict):
            continue
        program = str(stage.get("program") or "unknown")
        programs.add(program)
        action_id = _run_id_for(program, str(stage.get("run_id") or run_id))
        mentions.append(action_id)
        graph.append(
            _prune(
                {
                    "@id": action_id,
                    "@type": "CreateAction",
                    "name": "stage {}: {}".format(stage.get("order", "?"), program),
                    "instrument": _ref(_tool_id(program)),
                    "position": stage.get("order"),
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
    for program in sorted(programs):
        graph.append(
            _prune(
                {
                    "@id": _tool_id(program),
                    "@type": "SoftwareApplication",
                    "name": program,
                    "url": REPO_URLS.get(program),
                    "softwareVersion": (run_paradata or {}).get("tool_version") if len(programs) == 1 else None,
                }
            )
        )
    if mentions:
        root["mentions"] = _refs(sorted(set(mentions)))

    for key, prop in (("start_time", "startTime"), ("end_time", "endTime")):
        if (run_paradata or {}).get(key):
            root[prop] = (run_paradata or {})[key]

    graph.append(root)
    # A run crate makes a process-provenance claim the plain RO-Crate profile does not cover,
    # so it declares the extra profile. See PROCESS_RUN_PROFILE's note about pinning.
    return _assemble(graph, conforms_to=[ROCRATE_CONFORMS_TO, PROCESS_RUN_PROFILE])


def _run_date_published(records: Sequence[Dict[str, Any]], run_paradata: Optional[Dict[str, Any]]) -> str:
    explicit = str((run_paradata or {}).get("end_time") or "")
    if len(explicit) >= 10:
        return explicit[:10]
    dates = [_date_published(r) for r in records]
    return max(dates) if dates else "1970-01-01"


# ──────────────────────────────────────────────────────────────────────────────
# Assembly, serialisation, output
# ──────────────────────────────────────────────────────────────────────────────


def _assemble(graph: List[Dict[str, Any]], *, conforms_to: Sequence[str]) -> Dict[str, Any]:
    """Add the metadata descriptor, deduplicate, and order the graph deterministically.

    The descriptor and the root entity are pinned to the front — the spec does not require an
    order, but every RO-Crate in the wild puts them there and a reader opening the file should
    not have to search for what the crate is about. Everything else sorts by `@id`.
    """
    descriptor = {
        "@id": METADATA_FILENAME,
        "@type": "CreativeWork",
        "conformsTo": _refs(conforms_to) if len(conforms_to) > 1 else _ref(conforms_to[0]),
        "about": _ref(ROOT_ID),
    }

    merged: Dict[str, Dict[str, Any]] = {}
    for entity in graph:
        eid = entity["@id"]
        if eid in merged:
            # Same subject described twice (a tool that both stamped a block and contributed a
            # run, say). Merge rather than let the last one win: dropping half a description
            # silently is the failure mode `merge_document_records()` was fixed for.
            merged[eid].update({k: v for k, v in entity.items() if v not in (None, "", [], {})})
        else:
            merged[eid] = dict(entity)
    merged[METADATA_FILENAME] = descriptor

    pinned = [METADATA_FILENAME, ROOT_ID]
    ordered = [merged[p] for p in pinned if p in merged]
    ordered += [merged[k] for k in sorted(merged) if k not in pinned]

    return {"@context": [ROCRATE_CONTEXT, dict(sorted(_LOCAL_CONTEXT.items()))], "@graph": ordered}


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


def _sample_record() -> Dict[str, Any]:
    """A record exercising every mapping branch, including the ones that must NOT map."""
    return {
        "schema_version": "1.0",
        "record_type": "atrium-document",
        "doc_id": "CTX000000001",
        "source": {
            "sha256": "a" * 64,
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
                    "paradata_ref": "paradata/260724-101112_alto-postprocess.json",
                    "blocks": "pages,content",
                    "at": "2026-07-24T10:11:12+00:00",
                },
                {
                    "program": "nlp-enrich",
                    "run_id": "260724-101500",
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
                    "updated_at": "2026-07-24T10:11:12+00:00",
                },
                "content": {
                    "program": "alto-postprocess",
                    "run_id": "260724-101112",
                    "updated_at": "2026-07-24T10:11:12+00:00",
                },
                "entities": {
                    "program": "nlp-enrich",
                    "run_id": "260724-101500",
                    "updated_at": "2026-07-24T10:15:00+00:00",
                },
                "derived_from": {
                    "program": "nlp-enrich",
                    "run_id": "260724-101500",
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


def _selftest(stream: Any = None) -> int:
    """Structural checks. Exit 1 on any problem — the same contract as atrium_vocab --selftest."""
    out = stream or sys.stdout
    problems: List[str] = []
    record = _sample_record()
    crate = document_crate(record)
    graph = crate["@graph"]
    by_id = {e["@id"]: e for e in graph}

    # 1. RO-Crate's own mandatory shape.
    if graph[0]["@id"] != METADATA_FILENAME:
        problems.append("first graph entity is not the metadata descriptor")
    if graph[0].get("about") != {"@id": ROOT_ID}:
        problems.append("descriptor does not point `about` at the root")
    if ROOT_ID not in by_id:
        problems.append("no root data entity")
    root = by_id.get(ROOT_ID, {})
    for prop in ("@type", "name", "description", "datePublished", "license"):
        if not root.get(prop):
            problems.append("root is missing required property {!r}".format(prop))
    if root.get("@type") != "Dataset":
        problems.append("root @type is not Dataset")

    # 2. No dangling references. Every {"@id": …} either names an entity in this graph or is
    #    an absolute URI (a person's ORCID, a licence, a vocabulary term).
    def walk(node: Any) -> Iterable[str]:
        if isinstance(node, dict):
            if set(node) == {"@id"}:
                yield node["@id"]
            else:
                for v in node.values():
                    for r in walk(v):
                        yield r
        elif isinstance(node, list):
            for v in node:
                for r in walk(v):
                    yield r

    for ref in sorted(set(walk(graph))):
        if ref in by_id or ref.startswith(("http://", "https://")):
            continue
        problems.append("dangling reference {!r} — no entity and not an absolute URI".format(ref))

    # 3. The reference-discipline rule: regenerable is never a part.
    parts = {r["@id"] for r in root.get("hasPart", [])}
    if any(p.startswith("#regenerable-") for p in parts):
        problems.append("a regenerable recipe appears in hasPart")
    if "TEITOK/CTX000000001.teitok.xml" not in parts:
        problems.append("a derived_from output is missing from hasPart")
    if "#source" in parts:
        problems.append("the archive-managed original appears in hasPart")

    # 4. Determinism — the property the byte-comparison gate depends on.
    if to_json(document_crate(record)) != to_json(document_crate(record)):
        problems.append("document_crate() is not deterministic")
    if "T" in str(root.get("datePublished")) or len(str(root.get("datePublished"))) != 10:
        problems.append("datePublished is not a bare YYYY-MM-DD date")
    if root.get("datePublished") != "2026-07-24":
        problems.append("datePublished was not derived from the record's own newest stamp")

    # 5. Provenance granularity survived.
    for name in ("pages", "content", "entities", "derived_from"):
        if _block_id(name) not in by_id:
            problems.append("block {!r} lost its contextual entity".format(name))
    if _run_id_for("alto-postprocess", "260724-101112") not in by_id:
        problems.append("a contributing run lost its CreateAction")
    if _tool_id("nlp-enrich") not in by_id:
        problems.append("a contributing tool lost its SoftwareApplication")

    # 6. Controlled terms became DefinedTerms.
    terms = [e for e in graph if e.get("@type") == "DefinedTerm"]
    if len(terms) != 4:  # TEXT (twice, deduplicated), Clear, LOC, gu
        problems.append("expected 4 DefinedTerms, got {}".format(len(terms)))
    if not any(e.get("@type") == "DefinedTermSet" for e in graph):
        problems.append("no DefinedTermSet emitted")

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
    if len(rc["@graph"][0].get("conformsTo", [])) != 2:
        problems.append("run crate does not declare the process-run profile")
    if to_json(rc) != to_json(rc):
        problems.append("run_crate() is not deterministic")

    # 8. JSON round-trip.
    json.loads(to_json(crate))
    json.loads(to_json(rc))

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
    p.add_argument("--paradata", metavar="JSON", default=None, help="paradata run record for --run")
    p.add_argument("--out-dir", metavar="DIR", default=None, help="write ro-crate-metadata.json here")
    args = p.parse_args(argv)

    if args.selftest:
        return _selftest()

    if args.document:
        crate = document_crate(_load(args.document))
    elif args.run:
        crate = run_crate(
            [_load(r) for r in args.run],
            run_paradata=_load(args.paradata) if args.paradata else None,
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
