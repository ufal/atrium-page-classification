"""
atrium_document.py  –  Per-document aggregate record for ATRIUM pipelines (record schema 2.0).

Where a step's paradata answer *how a run behaved*, this answers *what we know about one
document*: the identity of the file, its text in a form the enrichment steps work on, and what
they add. It holds CONTENT ONLY (doc-schema-v2, atrium-project#84): no run data, no licence, no
copy of the layout.

The tools are separate containers and never see each other's outputs, so the record is built
by **accretion**: every tool takes the previous version of the JSON (if it is given one) and
returns it with **only its own block(s)** updated. Nothing else is touched.

    seed ──► [record builder] ──► doc.json ──► [nlp-enrich] ──► doc.json ──► …
              pages, lines, content            its fields only

Contract (see docs/document_schema.md):
  1. Optional baseline in, updated record out. `doc_id` travels WITH the baseline: the seed (or,
     outside the pilot, the first tool) names the document, every stage after it inherits.
  2. A tool writes its own block(s) and fields only; every other block is passed through untouched.
  3. No baseline given → the tool emits just its own part (standalone-safe).
  4. The record carries no run data: which run wrote which blocks and fields is named by that
     run's paradata `result` (`DocumentRecord.written()` gives it), never by a stamp in the record.
  5. No licence is computed or written anywhere (D2): components and their terms are paradata.
  6. Unknown or newer blocks of the same major are preserved; a newer major is refused; an older
     major is migrated on load (`migrate_document()`).

The positional blocks — `pages`, `content`, `lines` — have ONE writer, the record builder
(ocr-postprocess), which builds them from the document's stored `atr/docproc-json` whatever
produced it (digital-convert, the ATR, the PCSS converter). The layout itself (blocks, tables,
reading-order roles, page verdicts) stays in docproc; each line names its docproc block
(`lines[].block`), its text source and how precise its box is (D16, D20, D25, D26).

A record may start as an AMČR **seed** (atrium-project#71): `doc_id`, and a `source` with the
archive's `sha512`, file name and media type, nothing else. A seed is valid INPUT, checked by
`validate_seed()` (and by `validate_baseline()`, which every tool's inherited-baseline gate calls);
a record a tool EMITS is always checked against the full schema.

Two things the JSON Schema deliberately cannot check, and where they live instead:

  * a field-ownership drop — `lines[]` requires only `page`+`line`, so a row stripped of its
    `text` by `merge_block()` is a *valid* row. Use `assert_fields_survived()` (or
    `dropped_fields()`) right after a merge; `warn_dropped_fields=True` turns it on globally.
  * the schema itself being reachable — `schema_path()` / `load_schema()` /
    `validate_document()` resolve it next to whichever copy of this module was vendored.
"""

from __future__ import annotations

import copy
import json
import os
import re
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

# ──────────────────────────────────────────────────────────────────────────────
# Constants & Schema version
# ──────────────────────────────────────────────────────────────────────────────

#: 2.0: doc-schema-v2 (atrium-project#84). Released as the `doc-schema-v2` tag only after the
#: validator, the E2E checks and a run through the assembled chain (D12); `doc-schema-v1` stays.
SCHEMA_VERSION = "2.0"

RECORD_TYPE = "atrium-document"
RECORD_TYPE_MERGED = "atrium-document-merged"

#: Filename suffix for the record of one document.
FILE_SUFFIX = ".document.json"

#: The keys an AMČR seed may carry (atrium-project#71): the record's identity and nothing a tool
#: writes. `is_seed()` is "carries no other key"; `seed_schema()` says what each must hold.
SEED_KEYS: Tuple[str, ...] = ("schema_version", "record_type", "doc_id", "source")

#: What a seed's `source` must state: the archive's digest of the original, its file name and its
#: media type. `origin` is refused in a seed: it describes what the record builder found.
SEED_SOURCE_REQUIRED: Tuple[str, ...] = ("sha512", "filename", "media_type")

#: `source` fields that IDENTIFY the original. The seed writes them (intake, H1); outside the pilot
#: the first tool may. First writer wins.
SOURCE_IDENTITY_FIELDS: Tuple[str, ...] = ("sha512", "filename", "media_type")

#: `source` fields that DESCRIBE the original, written by the record builder (§5 of the contract).
SOURCE_DESCRIBED_FIELDS: Tuple[str, ...] = ("language", "page_count", "origin")

#: `source` fields that may disagree between readers without being a fault: a tool reading a
#: derived or uploaded file sees another name than the archive's. The first writer's value is
#: kept with a NOTE, never a WARNING or a strict refusal.
SOURCE_DESCRIPTIVE_FIELDS = frozenset({"filename", "media_type"})

#: `source.origin` (descriptive since 2.0, #84 §1): how the record's text was obtained. It no longer
#: selects a writer: the positional blocks have one.
SOURCE_ORIGINS: Tuple[str, ...] = ("digital", "atr", "legacy")

#: `lines[].text_source` (D25): the text layer of a born-digital file, the ATR, or converted PCSS ALTO.
TEXT_SOURCES: Tuple[str, ...] = ("text-layer", "atr", "legacy")

#: `lines[].bbox_precision` (D16, D25): a box the producer measured for the line, one derived from
#: the docproc block that contains it, or none.
BBOX_PRECISIONS: Tuple[str, ...] = ("measured", "derived", "absent")

#: `lines[].unit` (D18 rule 5): `block` marks a row holding a whole docproc block scored as one unit.
#: Absent means a visual line. Such a row keeps its score but stays out of the calibrated statistics.
LINE_UNITS: Tuple[str, ...] = ("block",)

#: `pages[].content_status` (D29): a page with text, or an empty page (no lines, no page score).
CONTENT_STATUSES: Tuple[str, ...] = ("text", "empty")

#: A run's stable identifier as `atrium_paradata` mints it: `urn:uuid:` and a lower-case UUID. Kept
#: for callers that still pass `run_uuid`; it names the run's paradata, never a record field.
RUN_UUID_PATTERN = re.compile(r"^urn:uuid:[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")

#: Top-level keys schema 2.0 removed (D1, D2, D19, D26). The schema refuses them; a tool cannot
#: write them; `_migrate_1_to_2()` drops them from a 1.x record.
REMOVED_BLOCKS: Tuple[str, ...] = ("provenance", "assembled", "derived_from", "regenerable", "forms", "tables")

#: Fields schema 2.0 removed inside blocks, by block (`source` included).
REMOVED_FIELDS: Dict[str, Tuple[str, ...]] = {
    "source": ("sha256",),
    "content": ("reading_order",),
    "pages": ("text_layer", "needs_ocr", "needs_ocr_reason", "ocr"),
    "lines": ("group_id", "style"),
}

#: Structural keys the module itself maintains, and the removed blocks — never a tool's own block.
RESERVED_KEYS = frozenset({"schema_version", "record_type", "doc_id", "source"}) | frozenset(REMOVED_BLOCKS)

#: Repository moves of 2026-10-01 (atrium-project#72): a program id keeps working after its
#: repository is succeeded. `alto-postprocess` became `ocr-postprocess` and the LLM keyword stage
#: of `llm-enrich` became `keyword-extract`. Grants are keyed by the current id; a predecessor's
#: id resolves to it through `canonical_program()`.
PROGRAM_SUCCESSORS: Dict[str, str] = {
    "alto-postprocess": "ocr-postprocess",
    "llm-enrich": "keyword-extract",
}


def canonical_program(name: Optional[str]) -> Optional[str]:
    """The current program id for `name`: a predecessor maps to its successor, anything else to itself."""
    if not name:
        return name
    return PROGRAM_SUCCESSORS.get(name, name)


def same_program(a: Optional[str], b: Optional[str]) -> bool:
    """True when two program ids name the same writer, ignoring a repository succession."""
    return canonical_program(a) == canonical_program(b)


#: The record builder (D20): the one writer of the positional blocks, of `source`'s described
#: fields, and — as the line-quality model's owner (D18) — of every line's and page's score.
RECORD_BUILDER = "ocr-postprocess"

#: Which tool owns which top-level block. One owner per block; blocks other tools add fields to
#: are split by FIELD (BLOCK_FIELD_OWNERS), so nothing is co-mutated. digital-convert owns no
#: block since 2.0: it writes docproc (D20).
BLOCK_OWNERS: Dict[str, Union[str, Tuple[str, ...]]] = {
    "pages": RECORD_BUILDER,
    "content": RECORD_BUILDER,
    "lines": RECORD_BUILDER,
    "quality_summary": RECORD_BUILDER,
    "page_categories": "page-classification",
    "entities": "nlp-enrich",
    "enrichment": "keyword-extract",
    "keywords": "keyword-extract",
    #: Reserved, not in the pilot (contract §5): declared, written by no tool.
    "translations": "translator",
}


def _owner_candidates(name: str) -> Tuple[str, ...]:
    """BLOCK_OWNERS[name] normalised to a tuple — one entry for single-owner blocks."""
    owners = BLOCK_OWNERS.get(name)
    if not owners:
        return ()
    return (owners,) if isinstance(owners, str) else tuple(owners)


#: Field-level ownership inside list blocks that more than one tool contributes to (contract §5).
#: A tool may only write the fields listed for it (plus the block's key fields).
BLOCK_FIELD_OWNERS: Dict[str, Dict[str, List[str]]] = {
    "pages": {
        RECORD_BUILDER: ["page", "canvas", "content_status", "quality_score", "quality_band"],
        "page-classification": ["category", "category_confidence"],
        "nlp-enrich": ["teitok_surface"],
    },
    "lines": {
        RECORD_BUILDER: [
            "text",
            "bbox",
            "block",
            "text_source",
            "bbox_precision",
            "unit",
            "categ",
            "quality_score",
            "lang",
        ],
        "nlp-enrich": ["lemma", "upos", "feats", "teitok_ref"],
    },
    "entities": {
        "nlp-enrich": [
            "surface",
            "lemma",
            "type_onto",
            "type_cnec",
            "type_teitok",
            "char_span",
            "bbox",
            "teitok_ref",
        ],
        #: Reserved, written by no tool in the pilot (atrium-project#70).
        "translator": ["translation_en"],
        "keyword-extract": ["pid"],
    },
}

#: Who may write which DESCRIBED `source` field. The identity fields are first-writer-wins.
SOURCE_FIELD_OWNERS: Dict[str, List[str]] = {RECORD_BUILDER: list(SOURCE_DESCRIBED_FIELDS)}

#: Natural key fields per list block, used to align records when merging by field. A page row is
#: keyed by its physical page number, the join key with docproc `pageNumber` (H2); the printed
#: label (`page`) is a field the record builder writes, and `lines[]`/`entities[]` name it.
BLOCK_KEY_FIELDS: Dict[str, List[str]] = {
    "pages": ["page_index"],
    "lines": ["page", "line"],
    "entities": ["page", "line", "char_span"],
}

#: Multi-dot pipeline suffixes to strip before falling back to a plain
#: ``split(".")[0]``. Longest/most-specific first, so ``.teitok.xml`` is
#: recognised before the generic ``.xml`` would otherwise short-circuit it.
#: Keep this list in sync across every tool that derives a doc_id from a
#: filename — see canonical_doc_id().
#:
#: The single-dot input formats after ``.txt`` are those alto-postprocess's
#: ``--method text-lines`` reads (its text_formats.READERS; atrium-alto-postprocess#31).
#: Without them a dotted name such as ``scan.2019.pdf`` fell back to the first dot
#: (``scan``), so ``report.v1.docx`` and ``report.v2.docx`` were one document to every
#: tool. Compression wrappers (``.gz``, ``.bz2``, ``.xz``) are deliberately NOT here:
#: ``x.txt.gz`` would become ``x.txt``; the first-dot fallback already answers ``x``.
KNOWN_PIPELINE_SUFFIXES: List[str] = [
    ".document.json",
    ".categories.json",
    ".teitok.xml",
    ".alto.xml",
    ".udpipe.conllu",
    ".conllu",
    ".xml",
    ".json",
    ".md",
    ".csv",
    ".txt",
    ".pdf",
    ".docx",
    ".docm",
    ".dotx",
    ".odt",
    ".ods",
    ".odp",
    ".xlsx",
    ".xlsm",
    ".pptx",
    ".pptm",
    ".epub",
    ".rtf",
    ".html",
    ".htm",
    ".xhtml",
    ".hocr",
    ".tei",
    ".jsonl",
    ".ndjson",
    ".tsv",
    ".tab",
    ".markdown",
    ".mdown",
    ".text",
    ".log",
    ".srt",
    ".vtt",
    ".eml",
    ".mbox",
    ".mbx",
    ".zip",
]


def canonical_doc_id(path_or_record: Any) -> str:
    """
    The one doc_id derivation every tool should use (issue #13 cross-cutting
    finding: four different derivations — ``Path.stem``, ``name.split(".")[0]``,
    a bespoke TEITOK/CoNLL-U stripper, a CSV column — silently forked the same
    document into different records on any multi-dot filename).

    If passed a dict (the JSON record), it returns the authoritative doc_id.
    If passed a path/string, it strips the longest matching known pipeline suffix
    from the basename; falls back to everything before the first dot. ``CTX000000001.alto.xml``
    and ``CTX000000001.udpipe.conllu`` and ``CTX000000001.document.json`` all
    resolve to ``CTX000000001``.
    """
    if isinstance(path_or_record, dict):
        return path_or_record.get("doc_id", path_or_record.get("id", ""))

    name = os.path.basename(str(path_or_record))
    lower = name.lower()
    for suffix in KNOWN_PIPELINE_SUFFIXES:
        if lower.endswith(suffix):
            return name[: -len(suffix)]
    return name.split(".")[0]


def _sanitise(obj: Any, _depth: int = 0) -> Any:
    if _depth > 10:
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _sanitise(v, _depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitise(v, _depth + 1) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _key_value(value: Any) -> str:
    """
    One canonical string per key value, so merge_block() aligns rows on IDENTITY rather
    than on Python type.

    This used to be a bare ``json.dumps``, and the type sensitivity was a silent
    row-forking bug. ``lines[].page`` is a STRING in the schema ("so 'iv' or 'A-1'
    survive") while ``lines[].line`` is an integer, and nothing stops a writer from
    passing ``1`` where the schema asks for ``"1"`` — ``additionalProperties: true`` plus
    ``required: [page, line]`` means such a row still validates. ``json.dumps(1)`` and
    ``json.dumps("1")`` differ, so the originator's row and a later contributor's row for
    the SAME line became two rows: one carrying the text, one carrying the morphology,
    neither complete. Downstream that reads as a document with twice the lines and half
    the text on each.

    Scalars therefore compare as text; containers keep their JSON shape, because
    ``entities[].char_span`` is a two-element list and must stay structurally compared.
    Integral floats collapse to ints so ``1.0`` and ``1`` are one row too.
    """
    if value is None or isinstance(value, bool):
        return json.dumps(value)
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, (int, float, str)):
        return str(value)
    return json.dumps(value, sort_keys=True, default=str)


def _record_key(record: Dict[str, Any], key_fields: Iterable[str]) -> tuple:
    return tuple(_key_value(record.get(k)) for k in key_fields)


class InvalidRecordError(ValueError):
    """A record this module cannot add to: a block a tool merges rows into is not a list of objects.

    ``reason`` is the services' registered code: ``atrium_service.attach_error_handlers()`` answers
    it with HTTP 422 ``invalid_record``. A ``ValueError``, so a caller that already catches one keeps
    working.
    """

    reason = "invalid_record"


def _json_type(value: Any) -> str:
    """The JSON name of `value`'s type, for messages about a record's shape."""
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    return "null" if value is None else type(value).__name__


def _list_block(data: Dict[str, Any], name: str, rows: bool = True) -> List[Any]:
    """The rows of list block `name` in `data` (a copy of the list), or [] when it has none.

    A merge used to take ``list(data.get(name) or [])`` on trust: a `pages` sent as a string or an
    object became its characters or keys, and the merge then failed on ``.get`` with an
    AttributeError, a 500 from a service and a traceback from a CLI (atrium-project#72). Such a
    record does not validate, but the merge runs before validation, so its shape is checked here.
    An empty or missing block is still no rows, as before. With `rows`, every row must be an
    object too, since the merge matches rows by their key fields.
    """
    value = data.get(name)
    if not value:
        return []
    if not isinstance(value, list):
        raise InvalidRecordError(
            f"the record's {name!r} block is a JSON {_json_type(value)}, not a list, so it cannot be added to"
        )
    if rows:
        for i, row in enumerate(value):
            if not isinstance(row, dict):
                raise InvalidRecordError(
                    f"row {i} of the record's {name!r} block is a JSON {_json_type(row)}, not an object, "
                    f"so the block cannot be added to"
                )
    return list(value)


class RemovedInSchema2(RuntimeError):
    """A call that wrote something schema 2.0 removed from the record (D1, D2).

    Raised by the methods a 1.x caller used for run data, so a tool that re-vendors this module
    fails at the call with the reason, not later at validation with a schema path.
    """


# ──────────────────────────────────────────────────────────────────────────────
# The record
# ──────────────────────────────────────────────────────────────────────────────


class DocumentRecord:
    """
    One document's aggregate record, opened by one tool for one contribution.

    Typical use, alongside the tool's ParadataLogger::

        with DocumentRecord.open(doc_id, "keyword-extract", baseline=args.document_json) as doc:
            doc.set_block("enrichment", {"items": items})
        logger.add_output("atrium/document-json", path=doc_path, record_fields=doc.written())
        # → writes <out_dir>/<doc_id>.document.json; the paradata name what this run wrote

    `run_id`, `run_uuid` and `paradata_ref` are still accepted, so a caller need not change at
    once; since 2.0 none of them is written into the record (D1).
    """

    def __init__(
        self,
        doc_id: str,
        program: str,
        baseline: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
        paradata_ref: Optional[str] = None,
        out_dir: str = ".",
        strict: bool = False,
        warn_dropped_fields: bool = False,
        run_uuid: Optional[str] = None,
    ) -> None:
        if not doc_id:
            raise ValueError("doc_id is required — the record is keyed on it.")
        if run_uuid and not RUN_UUID_PATTERN.match(run_uuid):
            raise ValueError(f"run_uuid {run_uuid!r} is not 'urn:uuid:' and a lower-case UUID.")

        self.program = program
        self.run_id = run_id or ""
        self.run_uuid = run_uuid or ""
        self.paradata_ref = paradata_ref or ""
        self.out_dir = out_dir
        self.strict = strict
        #: Opt-in: complain when merge_block() filters out a field the caller supplied.
        self.warn_dropped_fields = warn_dropped_fields

        # Rule 2/6: the baseline is deep-copied and never rewritten except where this tool writes.
        # A 1.x baseline handed over as a dict is migrated here, as load_document() migrates a file.
        data = copy.deepcopy(baseline) if baseline else {}
        self._data: Dict[str, Any] = migrate_document(data) if data else {}
        self._data["schema_version"] = SCHEMA_VERSION
        self._data.setdefault("record_type", RECORD_TYPE)

        #: What the CALLER derived for the file it happens to be reading. Kept because a tool
        #: names its OWN outputs after that file even when the record is keyed differently.
        self.derived_doc_id = doc_id
        self.doc_id = self._inherit_doc_id(doc_id)
        self._data["doc_id"] = self.doc_id

        self._had_baseline = bool(baseline)
        #: block -> fields this run wrote, in write order (`written()`).
        self._written: Dict[str, List[str]] = {}
        self._dropped_fields: Dict[str, List[str]] = {}
        self._finalised = False

    def _inherit_doc_id(self, derived: str) -> str:
        """The key this record is accreted under: the BASELINE's doc_id whenever there is one.

        WHY THE BASELINE WINS (atrium-project#10, D1's second half). `__init__` used to do a
        bare ``self._data["doc_id"] = doc_id`` — the caller's derivation overwrote the key the
        deep-copied baseline had arrived with, without so much as comparing them. That is a
        FORK: every block already in ``self._data`` was written under the baseline's id, so
        re-stamping the record with a different one hands the next stage a document whose
        contents belong to a document it has never heard of. The E2E gate is where it finally
        surfaced (hub run 31076188660): stage 3 emitted ``CTX000000003-1`` into a chain whose
        other four stages all said ``CTX000000003``.

        The caller's value is a GUESS FROM A LOCAL FILENAME, and the pipeline routinely feeds a
        stage something other than the original: the translator's real input is
        ``PAGE_ALTO/<doc>/<doc>-1.alto.xml``, a page alto-postprocess split out, so
        ``canonical_doc_id()`` correctly answers ``<doc>-1`` — correct for that FILE, wrong for
        the RECORD. No filename heuristic can close that gap in general (a document may legally
        be named ``sbn.2019-1``); the baseline can, because it carries the answer explicitly.
        So the rule is not "derive it more cleverly" but "do not re-derive what you were told":
        the originator names the document, every later stage inherits that name, and
        ``canonical_doc_id()`` decides only for a run that has no baseline at all (rule 3).

        The divergence is still worth saying out loud — it usually means a tool's own naming
        disagrees with the pipeline's — but it is not an error and never fatal: the record that
        comes out is the correct one, and raising here (even under ``strict``) would stall a
        pipeline over an id this method has just repaired. Compare
        ``merge_document_records()``, which DOES raise on differing doc_ids: there the ids
        arrive from two independent records and disagreement means two documents, while here
        there is one record and one of the two names for it is authoritative.
        """
        inherited = str(self._data.get("doc_id") or "")
        if not inherited or inherited == derived:
            return derived
        self._note(
            f"{self.program!r} derived doc_id {derived!r} from its own input, but the baseline it "
            f"was handed is keyed {inherited!r} — keeping {inherited!r}, which is the key every "
            f"block already in this record was written under. A derived input file (a page split "
            f"out of a multi-page original, say) is the usual reason the two differ; a genuinely "
            f"WRONG baseline is the other, and only the caller can tell those apart."
        )
        return inherited

    # ── constructors ────────────────────────────────────────────────────────

    @classmethod
    def open(
        cls,
        doc_id: str,
        program: str,
        baseline: Optional[str] = None,
        **kwargs: Any,
    ) -> "DocumentRecord":
        """
        Open a record for one tool's contribution.

        `baseline` is a path to the previous version of the JSON, or None. A missing or empty
        path is not an error (rule 3): the tool simply emits its own part.
        """
        data: Optional[Dict[str, Any]] = None
        if baseline:
            if os.path.exists(baseline):
                data = load_document(baseline)
            else:
                print(
                    f"[document] baseline {baseline} not found — emitting own part only",
                    file=sys.stderr,
                )
        return cls(doc_id, program, baseline=data, **kwargs)

    # ── writers ─────────────────────────────────────────────────────────────

    def set_source(self, sha256: Any = None, **fields: Any) -> "DocumentRecord":
        """
        Describe the ORIGINAL input (contract §5, H1).

        The identity fields (`sha512`, `filename`, `media_type`) come from the seed and are
        first-writer-wins: a later call may fill one in, never change it; a different
        `filename`/`media_type` is a NOTE (a tool reads a derivative), a different `sha512` a
        complaint. The described fields (`language`, `page_count`, `origin`) are the record
        builder's (`SOURCE_FIELD_OWNERS`); `origin` is one of `SOURCE_ORIGINS`.

        `sha256` was removed in 2.0: the original is named by its SHA-512 only. Passing it is a
        complaint (an error under `strict`) and the value is not written; so is any other field.
        """
        if sha256:
            self._complain("source.sha256 was removed in schema 2.0; the original is named by its sha512 only")
        incoming = {k: v for k, v in fields.items() if v is not None}
        unknown = sorted(k for k in incoming if k not in SOURCE_IDENTITY_FIELDS + SOURCE_DESCRIBED_FIELDS)
        if unknown:
            self._complain(f"source has no field(s) {unknown} in schema 2.0 — not written")
        own = set(SOURCE_FIELD_OWNERS.get(canonical_program(self.program) or "", []))
        refused = sorted(k for k in incoming if k in SOURCE_DESCRIBED_FIELDS and k not in own)
        if refused:
            self._complain(
                f"source fields {refused} are written by the record builder ({RECORD_BUILDER}), not {self.program!r}"
            )
        origin = incoming.get("origin")
        if origin is not None and origin not in SOURCE_ORIGINS:
            self._complain(f"source.origin {origin!r} is not one of {list(SOURCE_ORIGINS)} — not written")
            refused.append("origin")
        clean = _sanitise({k: v for k, v in incoming.items() if k not in unknown and k not in refused})

        existing = self._data.setdefault("source", {})
        differing = sorted(k for k, v in clean.items() if k in existing and existing[k] != v)
        conflicts = [f"{k}: {existing[k]!r} kept, {clean[k]!r} discarded" for k in differing if k not in SOURCE_DESCRIPTIVE_FIELDS]
        if conflicts:
            self._complain(
                f"source is immutable after the first writer, but {self.program!r} passed conflicting values — "
                + "; ".join(conflicts)
            )
        described = [k for k in differing if k in SOURCE_DESCRIPTIVE_FIELDS]
        if described:
            self._note(
                f"{self.program!r} read the original as "
                + ", ".join(f"{k} {clean[k]!r}" for k in described)
                + "; the record keeps the first writer's "
                + ", ".join(f"{k} {existing[k]!r}" for k in described)
                + " (source is first-writer-wins; a seed names the archive's file)."
            )
        added = {k: v for k, v in clean.items() if k not in existing}
        existing.update(added)
        if added:
            self._mark_written("source", added)
        return self

    def set_block(self, name: str, payload: Any) -> "DocumentRecord":
        """Replace this tool's OWN block wholesale (rule 2)."""
        self._assert_owner(name)
        clean = _sanitise(payload)
        self._data[name] = clean
        if isinstance(clean, list):
            fields: List[str] = []
            for row in clean:
                if isinstance(row, dict):
                    fields.extend(f for f in row if f not in fields)
            self._mark_written(name, fields)
        elif isinstance(clean, dict):
            self._mark_written(name, clean)
        else:
            self._mark_written(name, [])
        return self

    def merge_block(
        self,
        name: str,
        records: List[Dict[str, Any]],
        key_fields: Optional[List[str]] = None,
        own_fields: Optional[List[str]] = None,
    ) -> "DocumentRecord":
        """
        Field-level merge into a list block shared by several tools.

        Existing records are matched on `key_fields` and only `own_fields` (default: the
        program's grant in BLOCK_FIELD_OWNERS) are written, so a co-contributor's fields on the
        same row survive untouched. New rows are appended.
        """
        if name in RESERVED_KEYS:
            raise ValueError(f"{name!r} is maintained by the module or removed in schema 2.0, not a tool block.")
        keys = key_fields or BLOCK_KEY_FIELDS.get(name)
        if not keys:
            raise ValueError(f"no key fields known for block {name!r} — pass key_fields=[...]")
        # Before anything is noted or checked: a block that is not a list of objects is refused
        # with the record left as it was (InvalidRecordError, the services' 422 invalid_record).
        existing: List[Dict[str, Any]] = _list_block(self._data, name)

        program = canonical_program(self.program) or ""
        declared = BLOCK_FIELD_OWNERS.get(name, {})
        allowed = own_fields if own_fields is not None else declared.get(program)
        if allowed is None:
            self._complain(f"{self.program!r} has no declared field ownership in block {name!r}")
            allowed = []
        elif own_fields is not None and program not in declared and program not in _owner_candidates(name):
            # own_fields narrows or extends a declared writer's own set; it never confers writership.
            self._complain(
                f"{self.program!r} is neither an owner nor a declared field contributor of "
                f"block {name!r} — own_fields narrows an existing grant, it does not create one"
            )
        removed = set(REMOVED_FIELDS.get(name, ()))
        writable = (set(allowed) | set(keys)) - removed

        dropped = sorted({f for r in records for f in r} - writable)
        if dropped:
            self._dropped_fields[name] = dropped
            if self.warn_dropped_fields or removed & set(dropped):
                self._complain(
                    f"{self.program!r} may not write {dropped} in block {name!r} — dropped. "
                    f"Declare them in BLOCK_FIELD_OWNERS or pass own_fields=[...]"
                    + (f" ({sorted(removed & set(dropped))} were removed in schema 2.0)" if removed & set(dropped) else "")
                )

        index = {_record_key(r, keys): r for r in existing}
        wrote: List[str] = []
        for incoming in records:
            k = _record_key(incoming, keys)
            patch = {f: v for f, v in incoming.items() if f in writable}
            wrote.extend(f for f in patch if f not in wrote)
            target = index.get(k)
            if target is None:
                existing.append(_sanitise(patch))
                index[k] = existing[-1]
            else:
                target.update(_sanitise(patch))
        self._data[name] = existing
        self._mark_written(name, wrote)
        return self

    def get_block(self, name: str, default: Any = None) -> Any:
        """
        Read-only access to any block of the record as it stands (baseline plus this
        contribution's own writes so far) — a deep copy, so callers can inspect a
        block (e.g. a co-owned block another tool wrote) without reaching into
        ``_data`` directly or risking a caller mutating the record in place.
        """
        value = self._data.get(name, default)
        return copy.deepcopy(value)

    def dropped_fields(self, name: Optional[str] = None) -> Dict[str, List[str]]:
        """
        Fields this run handed to merge_block() that its grant did not authorise, so they
        were filtered out. Empty when nothing was lost. Pass `name` for one block.
        """
        if name is not None:
            return {name: list(self._dropped_fields.get(name, []))} if name in self._dropped_fields else {}
        return {k: list(v) for k, v in self._dropped_fields.items()}

    def assert_fields_survived(
        self,
        name: str,
        records: List[Dict[str, Any]],
        fields: Optional[Iterable[str]] = None,
    ) -> None:
        """
        Round-trip assertion for a block this run just merged (Issue #18 §1b, the "local"
        mitigation, and the check plan §2's Layer D is specified to run before validate()).

        `jsonschema.validate()` cannot catch a field-ownership drop: `lines[]` requires only
        `page`+`line`, so a row stripped of its `text` is a valid row. This compares what was
        handed in against what is actually in the record, keyed the same way merge_block()
        keyed it, and raises naming the block, the row and the missing fields.

        `fields` defaults to every field present on the incoming rows. Values are not
        compared — only presence — because a co-contributor legitimately refines a value.
        """
        keys = BLOCK_KEY_FIELDS.get(name)
        if not keys:
            raise ValueError(f"no key fields known for block {name!r} — cannot round-trip it")

        written = {_record_key(r, keys): r for r in (self._data.get(name) or [])}
        for row in records:
            wanted = list(fields) if fields is not None else list(row)
            got = written.get(_record_key(row, keys), {})
            lost = [f for f in wanted if f in row and f not in got]
            if lost:
                where = ", ".join(f"{k}={row.get(k)!r}" for k in keys)
                raise RuntimeError(
                    f"block {name!r} row ({where}): {lost} dropped by merge_block — "
                    f"{self.program!r} is not declared for them in "
                    f"BLOCK_FIELD_OWNERS[{name!r}] (declared: "
                    f"{sorted(BLOCK_FIELD_OWNERS.get(name, {}).get(self.program) or [])})"
                )

    def written(self) -> Dict[str, List[str]]:
        """The blocks this run wrote and, for each, the fields: what its paradata `result` names
        for the record (#84 §4; `atrium_rocrate.record_result()`). Key fields of list blocks are
        included. Empty when the run wrote nothing."""
        return {k: list(v) for k, v in self._written.items()}

    def add_derived_from(self, key: str, ref: str) -> "DocumentRecord":
        """Removed in 2.0 (D1): an input is named in the run's paradata `object`, by distribution and SHA-512."""
        raise RemovedInSchema2(
            "derived_from was removed in record schema 2.0 (D1): name the input in the run's paradata "
            "(ParadataLogger.add_input), not in the record"
        )

    def add_regenerable(self, key: str, recipe: Dict[str, Any]) -> "DocumentRecord":
        """Removed in 2.0 (D1): a regeneration recipe is run data, kept in the paradata if anywhere."""
        raise RemovedInSchema2(
            "regenerable was removed in record schema 2.0 (D1): a regeneration recipe is run data and "
            "does not belong in the record"
        )

    def add_license_detail(self, license_detail: Dict[str, Any]) -> "DocumentRecord":
        """Removed in 2.0 (D2): no licence is computed or written; components and their terms are paradata."""
        raise RemovedInSchema2(
            "the record carries no licence since schema 2.0 (D2): list components with their terms in "
            "the paradata (ParadataLogger.log_component)"
        )

    # ── internals ───────────────────────────────────────────────────────────

    def _mark_written(self, block: str, fields: Iterable[str]) -> None:
        held = self._written.setdefault(block, [])
        held.extend(f for f in fields if f not in held)

    def _assert_owner(self, name: str) -> None:
        if name in RESERVED_KEYS:
            raise ValueError(f"{name!r} is maintained by the module or removed in schema 2.0, not a tool block.")
        owners = _owner_candidates(name)
        program = canonical_program(self.program) or ""
        # A wholesale set_block() on a field-split block erases every co-contributor's fields.
        co_contributors = sorted(set(BLOCK_FIELD_OWNERS.get(name, {})) - {program} - set(owners))
        if co_contributors:
            self._complain(
                f"block {name!r} is field-split with {co_contributors} — "
                f"use merge_block(), not set_block(), or a co-contributor's fields will be lost"
            )
        if owners and program not in owners:
            self._complain(f"block {name!r} is owned by {' or '.join(owners)}, not {self.program!r}")

    def _complain(self, message: str) -> None:
        if self.strict:
            raise ValueError(message)
        print(f"[document] WARNING – {message}", file=sys.stderr)

    def _note(self, message: str) -> None:
        """Visible but never fatal — for things a reader should know that are not errors."""
        print(f"[document] NOTE – {message}", file=sys.stderr)

    # ── output ──────────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """The record as it would be written — baseline passed through, own blocks applied."""
        out = copy.deepcopy(self._data)
        if not out.get("source"):
            out.pop("source", None)
        # Stable, predictable key order for diff-friendly output.
        order = [
            "schema_version",
            "record_type",
            "doc_id",
            "source",
            "page_categories",
            "pages",
            "content",
            "lines",
            "quality_summary",
            "entities",
            "translations",
            "enrichment",
            "keywords",
        ]
        ordered = {k: out[k] for k in order if k in out}
        for k in out:  # any unknown/newer block of this major is preserved (rule 6)
            if k not in ordered:
                ordered[k] = out[k]
        return ordered

    def finalize(self, out_path: Optional[str] = None) -> str:
        if self._finalised:
            # No `from None`: that only means anything inside an except block, and here it
            # would hide a real cause if finalize() is ever retried from a handler.
            raise RuntimeError("finalize() has already been called.")
        if not self._written:
            print(
                f"[document] WARNING – {self.program} contributed no block to {self.doc_id}",
                file=sys.stderr,
            )

        path = out_path or os.path.join(self.out_dir, f"{self.doc_id}{FILE_SUFFIX}")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # Write-then-rename: a crash mid-write must never leave a corrupt record for
        # the next tool's load_document() to trip over.
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)

        self._finalised = True
        print(f"[document] Record written → {path}", flush=True)
        return path

    def __enter__(self) -> "DocumentRecord":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is None and not self._finalised:
            try:
                self.finalize()
            except Exception as e:  # pragma: no cover - defensive, mirrors paradata
                print(f"[document] WARNING – could not write record: {e}", file=sys.stderr)
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Reader & Migration
# ──────────────────────────────────────────────────────────────────────────────


def _major(version: Any) -> int:
    """The MAJOR of a `schema_version` string; a missing or unreadable one counts as current."""
    try:
        return int(str(version).split(".")[0])
    except (TypeError, ValueError):
        return int(SCHEMA_VERSION.split(".")[0])


#: How a 1.x `source.origin` maps onto 2.0's descriptive value (prefix match, case-insensitive):
#: the born-digital converter's spellings, the PCSS (ABBYY) ALTO route, and OCR/VLM engines.
_ORIGIN_1_TO_2: Tuple[Tuple[str, str], ...] = (
    ("digital", "digital"),
    ("docx", "digital"),
    ("pdf", "digital"),
    ("abbyy", "legacy"),
    ("alto", "legacy"),
    ("ocr:", "atr"),
    ("vlm:", "atr"),
    ("atr", "atr"),
    ("legacy", "legacy"),
)

_TEXT_SOURCE_FOR_ORIGIN = {"digital": "text-layer", "atr": "atr", "legacy": "legacy"}

#: 1.x line categories that were digital-convert's decode verdicts, not text quality (D18 rule 2).
_DECODE_VERDICTS_1X = frozenset({"Garbage", "Inverted"})


def _migrate_1_to_2(record: Dict[str, Any]) -> Dict[str, Any]:
    """Rewrite a 1.x record as 2.0 (doc-schema-v2, atrium-project#84). Content survives; run data,
    licence and layout do not.

    * removed: `provenance`, `assembled`, `derived_from`, `regenerable`, `forms`, `tables`,
      `content.reading_order`, `source.sha256`, `pages[].text_layer|needs_ocr|needs_ocr_reason|ocr`,
      `lines[].group_id|style`;
    * `source.origin` → `digital`/`atr`/`legacy` (dropped when it maps to none), and each line's
      `text_source` follows from it;
    * a page whose lines were numbered from 0 (1.x digital-convert) is renumbered from 1, and the
      entities on it follow;
    * a `Garbage`/`Inverted` `categ` (a decode verdict) is dropped with its `quality_score`;
    * a missing `pages[].page_index` is the numeric label, else the row's position; a missing
      `content_status` follows from whether the page has lines (when the record has `lines`), and
      an empty page loses its page score (D18 rule 7);
      a canvas with no unit gets 1.x's default, `px`; a line with a box gets `bbox_precision:
      measured`;
    * a `quality_summary` is recomputed (`quality_summary()`): it reads the rows rewritten above.
    """
    out = {k: v for k, v in record.items() if k not in REMOVED_BLOCKS}

    source = out.get("source")
    origin_2: Optional[str] = None
    if isinstance(source, dict):
        source = {k: v for k, v in source.items() if k in SOURCE_IDENTITY_FIELDS + SOURCE_DESCRIBED_FIELDS}
        origin = source.pop("origin", None)
        if isinstance(origin, str):
            folded = origin.casefold()
            origin_2 = next((new for prefix, new in _ORIGIN_1_TO_2 if folded.startswith(prefix)), None)
        if origin_2:
            source["origin"] = origin_2
        out["source"] = source

    content = out.get("content")
    if isinstance(content, dict):
        out["content"] = {k: v for k, v in content.items() if k not in REMOVED_FIELDS["content"]}

    lines = out.get("lines")
    offsets: Dict[str, int] = {}
    if isinstance(lines, list):
        first: Dict[str, int] = {}
        for row in lines:
            if isinstance(row, dict) and isinstance(row.get("line"), int) and not isinstance(row.get("line"), bool):
                page = _key_value(row.get("page"))
                first[page] = min(first.get(page, row["line"]), row["line"])
        offsets = {page: 1 for page, low in first.items() if low == 0}
        migrated: List[Any] = []
        for row in lines:
            if not isinstance(row, dict):
                migrated.append(row)
                continue
            row = {k: v for k, v in row.items() if k not in REMOVED_FIELDS["lines"]}
            if _key_value(row.get("page")) in offsets and isinstance(row.get("line"), int):
                row["line"] = row["line"] + 1
            if row.get("categ") in _DECODE_VERDICTS_1X:
                row.pop("categ", None)
                row.pop("quality_score", None)
            if origin_2 and "text_source" not in row:
                row["text_source"] = _TEXT_SOURCE_FOR_ORIGIN[origin_2]
            if "bbox" in row and "bbox_precision" not in row:
                row["bbox_precision"] = "measured"
            migrated.append(row)
        out["lines"] = migrated

    entities = out.get("entities")
    if offsets and isinstance(entities, list):
        for row in entities:
            if isinstance(row, dict) and _key_value(row.get("page")) in offsets and isinstance(row.get("line"), int):
                row["line"] = row["line"] + 1

    pages = out.get("pages")
    if isinstance(pages, list):
        with_lines = {_key_value(r.get("page")) for r in (lines or []) if isinstance(r, dict)} if isinstance(lines, list) else None
        migrated_pages: List[Any] = []
        for position, row in enumerate(pages, start=1):
            if not isinstance(row, dict):
                migrated_pages.append(row)
                continue
            row = {k: v for k, v in row.items() if k not in REMOVED_FIELDS["pages"]}
            if not isinstance(row.get("page_index"), int) or isinstance(row.get("page_index"), bool):
                label = str(row.get("page", ""))
                row["page_index"] = int(label) if label.isdigit() and int(label) > 0 else position
            if with_lines is not None and "content_status" not in row:
                row["content_status"] = "text" if _key_value(row.get("page")) in with_lines else "empty"
            if row.get("content_status") == "empty":  # an empty page carries no page score (D18 rule 7)
                row.pop("quality_score", None)
                row.pop("quality_band", None)
            canvas = row.get("canvas")
            if isinstance(canvas, dict) and "unit" not in canvas:
                row["canvas"] = {**canvas, "unit": "px"}
            migrated_pages.append(row)
        out["pages"] = migrated_pages

    if isinstance(out.get("quality_summary"), dict):  # a read-out of the rows just rewritten
        out["quality_summary"] = quality_summary(out)

    out["schema_version"] = "2.0"
    return out


def migrate_document(record: Dict[str, Any]) -> Dict[str, Any]:
    """Apply schema migrations up to the current SCHEMA_VERSION (rule 6).

    A record of an older MAJOR is rewritten step by step (`_migrate_1_to_2()`); a record of the
    current major is returned as it is; a newer major is refused. A seed is identity only and is
    the same in every major, so only its `schema_version` moves.
    """
    major = _major(record.get("schema_version", SCHEMA_VERSION))
    current = _major(SCHEMA_VERSION)
    if major > current:
        raise ValueError(
            f"Schema version {record.get('schema_version')} is newer than supported {SCHEMA_VERSION}. Please update tools."
        )
    if major == current:
        return record
    if is_seed(record):
        return {**record, "schema_version": SCHEMA_VERSION}
    if major == 1:
        record = _migrate_1_to_2(record)
    return record


#: Canonical filename of the JSON Schema that validates these records.
SCHEMA_FILENAME = "atrium_document.schema.json"


def schema_path() -> Optional[str]:
    """
    Locate `atrium_document.schema.json` next to whichever copy of this module is imported.

    Plan §2's Layer D makes schema validation the gatekeeper ("if jsonschema.validate()
    fails, no doc.json is emitted"), but there was no way for shared code to FIND the
    schema: the hub keeps it in `docs/templates/shared/` while every tool repo keeps it at
    the repo root, and the only locator anywhere was a relative `parent.parent` walk inside
    one test. Resolving it relative to `__file__` is correct in both layouts, because
    para-drift enforces that the module and the schema travel together.

    Returns None rather than raising, so a tool can degrade to "validated: no" instead of
    dying when only the module was vendored.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    for candidate in (
        os.path.join(here, SCHEMA_FILENAME),
        os.path.join(os.path.dirname(here), SCHEMA_FILENAME),
    ):
        if os.path.exists(candidate):
            return candidate
    return None


def load_schema() -> Dict[str, Any]:
    """The parsed JSON Schema. Raises FileNotFoundError when it was not vendored."""
    path = schema_path()
    if not path:
        raise FileNotFoundError(
            f"{SCHEMA_FILENAME} not found next to {__file__} — re-vendor it "
            f"(scripts/revendor_shared.sh); para-drift expects the pair to travel together."
        )
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def validate_document(record: Dict[str, Any]) -> None:
    """
    Validate a record against the canonical schema (the current major). Raises on failure,
    returns None on pass.

    A record of an older major fails here: read it with `load_document()`, or pass it through
    `migrate_document()` first. `jsonschema` is an optional import: without it this raises
    RuntimeError rather than passing silently — a validation gate that quietly becomes a no-op is
    worse than no gate.
    """
    try:
        import jsonschema
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise RuntimeError(
            "jsonschema is not installed, so the record cannot be validated. Install it "
            "(requirements-test.txt / requirements_digital.txt) rather than skipping the gate."
        ) from exc
    jsonschema.validate(record, load_schema())


# ──────────────────────────────────────────────────────────────────────────────
# The AMČR seed (atrium-project#71)
# ──────────────────────────────────────────────────────────────────────────────


def is_seed(record: Any) -> bool:
    """True when `record` carries nothing beyond the seed's keys (`SEED_KEYS`).

    That is the record before any tool has written to it. Whether it is a GOOD seed is
    `validate_seed()`'s question, not this one's.
    """
    return isinstance(record, dict) and bool(record) and set(record) <= set(SEED_KEYS)


def seed_schema() -> Dict[str, Any]:
    """The JSON Schema of an AMČR seed, built from the record schema so the two cannot drift.

    A seed is `doc_id` and a `source` stating the archive's `sha512`, the file name and the media
    type (`SEED_SOURCE_REQUIRED`), optionally `schema_version` and `record_type`, and nothing
    else. Each property is the record schema's own subschema; the seed only adds what it
    requires, refuses `source.origin` (the record builder's) and closes the top level. A seed is
    identity only, the same in every major, so its `schema_version` may name any. `python
    atrium_document.py seed-schema` prints it for a caller with no Python.
    """
    full = load_schema()
    props = full["properties"]
    source = copy.deepcopy(props["source"])
    source["required"] = list(SEED_SOURCE_REQUIRED)
    source["not"] = {"required": ["origin"]}
    record_type = copy.deepcopy(props["record_type"])
    record_type["enum"] = [RECORD_TYPE]
    return {
        "$schema": full.get("$schema", "https://json-schema.org/draft/2020-12/schema"),
        "title": "ATRIUM document seed",
        "description": (
            "The record an archive writes before the first ATRIUM stage (atrium-project#71): the "
            "document's id and the original's digest, file name and media type. Valid input to "
            "every tool; a record a tool emits is validated against the full record schema. "
            f"Derived from atrium_document.schema.json at schema_version {SCHEMA_VERSION}."
        ),
        "type": "object",
        "required": ["doc_id", "source"],
        "additionalProperties": False,
        "properties": {
            "schema_version": {"type": "string", "pattern": r"^[0-9]+\.[0-9]+$"},
            "record_type": record_type,
            "doc_id": copy.deepcopy(props["doc_id"]),
            "source": source,
        },
    }


def validate_seed(record: Dict[str, Any]) -> None:
    """Validate an AMČR seed against `seed_schema()`. Raises on failure, like `validate_document()`."""
    try:
        import jsonschema
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise RuntimeError(
            "jsonschema is not installed, so the seed cannot be validated. Install it rather "
            "than skipping the gate."
        ) from exc
    jsonschema.validate(record, seed_schema())


def validate_baseline(record: Dict[str, Any]) -> None:
    """Validate a record a tool was HANDED: a seed against the seed profile, anything else in full.

    A 1.x baseline is checked as it will be used, after `migrate_document()` (on a copy). A record
    a tool emits is still checked with `validate_document()`.
    """
    if is_seed(record):
        validate_seed(record)
    else:
        validate_document(migrate_document(copy.deepcopy(record)))


def load_document(path: str) -> Dict[str, Any]:
    """Read a document record, migrating an older major transparently (rule 6).

    ``utf-8-sig`` reads a record with or without a UTF-8 byte-order mark (a seed saved by a
    Windows editor has one). A newer major is refused.
    """
    with open(path, "r", encoding="utf-8-sig") as fh:
        data = json.load(fh)
    return migrate_document(data)


# ──────────────────────────────────────────────────────────────────────────────
# Derived read-outs (atrium-project#73)
# ──────────────────────────────────────────────────────────────────────────────


def _score(value: Any) -> Optional[float]:
    """A stored quality score as a float, or None (absent, a bool, not a number)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def quality_summary(record: Dict[str, Any]) -> Dict[str, Any]:
    """The `quality_summary` block of `record`: numbers read off its stored quality fields.

    * `pages`: `total` rows; `empty`, those with `content_status: empty` (counted apart, D18 rule
      7); `scored`, those with a `quality_score`; and `mean`, `median`, `min`, `max` of those
      scores (absent when none is scored);
    * `lines`: `total` rows; `by_categ`, visual lines per `categ` value, in key order; and, when
      there are any, `block_units`, the rows marked `unit: block` (a whole block scored as one unit,
      D18 rule 5), which stay out of `by_categ`.

    Pure and deterministic; statistics are rounded to four decimals. It sets no threshold and
    makes no band, so nothing in it can gate or route a document. The record builder writes it
    whenever it writes or scores pages and lines.
    """
    rows = record.get("pages")
    pages = [row for row in (rows if isinstance(rows, list) else []) if isinstance(row, dict)]
    scores = sorted(s for s in (_score(row.get("quality_score")) for row in pages) if s is not None)
    empty = sum(1 for row in pages if row.get("content_status") == "empty")
    page_part: Dict[str, Any] = {"total": len(pages), "empty": empty, "scored": len(scores)}
    if scores:
        middle = len(scores) // 2
        median = scores[middle] if len(scores) % 2 else (scores[middle - 1] + scores[middle]) / 2
        page_part.update(
            mean=round(sum(scores) / len(scores), 4),
            median=round(median, 4),
            min=round(scores[0], 4),
            max=round(scores[-1], 4),
        )
    lines = [row for row in record.get("lines") or [] if isinstance(row, dict)]
    by_categ: Dict[str, int] = {}
    block_units = 0
    for row in lines:
        if row.get("unit") == "block":
            block_units += 1
            continue
        categ = row.get("categ")
        if isinstance(categ, str) and categ:
            by_categ[categ] = by_categ.get(categ, 0) + 1
    line_part: Dict[str, Any] = {"total": len(lines), "by_categ": dict(sorted(by_categ.items()))}
    if block_units:
        line_part["block_units"] = block_units
    return {"pages": page_part, "lines": line_part}


# ──────────────────────────────────────────────────────────────────────────────
# Merging Logic (only for parallel branches re-joining)
# ──────────────────────────────────────────────────────────────────────────────


def merge_document_records(json_paths: List[str], out_path: str) -> str:
    """
    Fold several partial records for the SAME document into one.

    The pipeline is linear, so the normal path needs no merge — each tool hands its output to
    the next. This exists for fan-out/fan-in. Since 2.0 the record has no stamps saying which run
    wrote a block, so blocks are taken in ARGUMENT order: pass the records in chain order, and a
    later record's block replaces an earlier one's. `source` is first-writer-wins per field, and a
    contradiction between two inputs is an error, not resolved by order. Records of an older
    major are migrated as they are read.
    """
    if not json_paths:
        raise ValueError("no record paths given")

    merged: Dict[str, Any] = {}
    doc_ids: List[str] = []
    source: Dict[str, Any] = {}
    source_conflicts: List[str] = []

    for p in json_paths:
        data = load_document(p)
        doc_id = data.get("doc_id", "")
        if doc_id and doc_id not in doc_ids:
            doc_ids.append(doc_id)
        for key, value in (data.get("source") or {}).items():
            if key not in source:
                source[key] = value
            elif source[key] != value and key not in SOURCE_DESCRIPTIVE_FIELDS:
                source_conflicts.append(f"{key}: {source[key]!r} kept, {value!r} from {p} discarded")
        for key, value in data.items():
            if key in ("source", "schema_version", "record_type"):
                continue
            merged[key] = value

    if len(doc_ids) > 1:
        raise ValueError(f"records belong to different documents: {doc_ids}")
    if source_conflicts:
        raise ValueError(
            "records disagree about the immutable `source` of the same document — "
            + "; ".join(sorted(source_conflicts))
        )

    out: Dict[str, Any] = {"schema_version": SCHEMA_VERSION, "record_type": RECORD_TYPE_MERGED}
    if "doc_id" in merged:
        out["doc_id"] = merged.pop("doc_id")
    if source:
        out["source"] = source
    out.update(merged)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"[document] Merged record → {out_path}", flush=True)
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# CLI (mirrors atrium_paradata.py's shim so shell stages can use it too)
# ──────────────────────────────────────────────────────────────────────────────


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(prog="python atrium_document.py")
    sub = p.add_subparsers(dest="cmd", required=True)

    st = sub.add_parser("set-block", help="write one block from a JSON file/stdin")
    st.add_argument("--doc-id", required=True)
    st.add_argument("--program", required=True)
    st.add_argument("--block", required=True)
    st.add_argument("--payload", required=True, help="path to a JSON file, or '-' for stdin")
    st.add_argument("--baseline", default=None, help="previous version of the record")
    st.add_argument("--out", default=None)
    st.add_argument("--written", default=None, help="also write the blocks and fields written, as JSON, here")
    st.add_argument("--strict", action="store_true")

    me = sub.add_parser("merge", help="fold parallel partial records for one document (chain order)")
    me.add_argument("--paths", nargs="+", required=True)
    me.add_argument("--out", required=True)

    mi = sub.add_parser("migrate", help="rewrite a record at the current schema version")
    mi.add_argument("--path", required=True)

    sub.add_parser("seed-schema", help="print the JSON Schema of an AMČR seed (atrium-project#71)")

    args = p.parse_args()

    if args.cmd == "set-block":
        raw = sys.stdin.read() if args.payload == "-" else open(args.payload, encoding="utf-8").read()
        with DocumentRecord.open(args.doc_id, args.program, baseline=args.baseline, strict=args.strict) as doc:
            doc.set_block(args.block, json.loads(raw))
            if args.out:
                doc.finalize(args.out)
            if args.written:
                with open(args.written, "w", encoding="utf-8") as fh:
                    json.dump(doc.written(), fh, ensure_ascii=False, indent=2)

    elif args.cmd == "merge":
        merge_document_records(args.paths, args.out)

    elif args.cmd == "migrate":
        data = load_document(args.path)
        with open(args.path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        print(f"[document] Migrated {args.path} to {SCHEMA_VERSION}", flush=True)

    elif args.cmd == "seed-schema":
        print(json.dumps(seed_schema(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()
