"""
atrium_document.py  –  Per-document aggregate record ("paradata pair") for ATRIUM pipelines.

Sibling of `atrium_paradata.py`. Where paradata answers *how a run behaved*, this answers
*what we know about one document*: its text, pages, tables, forms, entities and enrichment,
gathered across the pipeline into one FAIR, versioned JSON.

The tools are separate containers and never see each other's outputs, so the record is built
by **accretion**: every tool takes the previous version of the JSON (if it is given one) and
returns it with **only its own block** updated. Nothing else is touched.

    doc.json ──► [tool A] ──► doc.json ──► [tool B] ──► doc.json ──► …
                  writes A's block only     writes B's block only

Contract (see docs/document_schema.md):
  1. Optional baseline in, updated record out.
  2. A tool writes its own block(s) only; every other block is passed through untouched.
  3. No baseline given → the tool emits just its own part (standalone-safe).
  4. Each block is stamped with the writing tool's program / run_id / paradata_ref.
  5. Licenses accrete through `para_licenses.merge_effective_licenses` (most restrictive wins).
  6. Unknown or newer blocks are preserved; a newer major schema is refused.

Only ever reference **persistent** artifacts: the original input, or a previous step's stored
output. Transient derivatives (page images/thumbnails, the annotated Markdown) belong in
`regenerable` as a recipe, never as a stored path.

Four blocks — `pages`, `content`, `lines`, `tables` — are the document's *positional plane*,
and since Issue #18 they have two possible **originators**: `alto-postprocess` for an OCR/ALTO
document, `digital-convert` for a digital-born PDF/DOCX. Which one applies is fixed per record
by `source.origin` (see `ORIGIN_ORIGINATORS` / `resolve_originator()`). Calling `set_source()`
first is the natural order, but it is **no longer required**: a block written before the origin
is known is re-checked as soon as one arrives, and again in `to_dict()`.

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
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

try:
    from para_licenses import merge_effective_licenses
except ImportError:
    merge_effective_licenses = None  # type: ignore

# ──────────────────────────────────────────────────────────────────────────────
# Constants & Schema version
# ──────────────────────────────────────────────────────────────────────────────

SCHEMA_VERSION = "1.0"
LICENSE_NAME = "CC BY-NC 4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by-nc/4.0/"

RECORD_TYPE = "atrium-document"
RECORD_TYPE_MERGED = "atrium-document-merged"

#: Filename suffix for the record of one document.
FILE_SUFFIX = ".document.json"

#: Structural keys the module itself maintains — never a tool's "own block".
RESERVED_KEYS = frozenset(
    {"schema_version", "record_type", "doc_id", "source", "provenance", "assembled"}
)

#: Which tool owns which top-level block. One owner per block; blocks shared between
#: tools are split by FIELD instead (see BLOCK_FIELD_OWNERS) so nothing is co-mutated.
#:
#: RESOLVED (Issue #18 §1a, 2026-08-03): a TUPLE value means the block has more than
#: one possible ORIGINATOR and the choice is fixed per DOCUMENT, not per ecosystem.
#: An ALTO/OCR document's positional plane (pages/content/lines/tables) comes from
#: alto-postprocess; a digital-born PDF/DOCX's comes from digital-convert; and no
#: document is ever both. Which one applies is decided by `source.origin` — already
#: first-writer-wins in set_source() — via ORIGIN_ORIGINATORS below, and checked by
#: _assert_origin_consistent().
#:
#: The two rejected alternatives, recorded so this is not relitigated:
#:   1. Give alto-postprocess a "no-op passthrough" mode so it stays owner of record.
#:      Rejected: it stamps assembled.blocks[<block>].program = "alto-postprocess"
#:      and appends a provenance.contributors[] entry with a paradata_ref, for a run
#:      that did nothing, in a record whose purpose is FAIR catalogue export. Rule 4
#:      exists for attribution granularity; this would falsify it. It also puts the
#:      alto container back on the digital-born critical path, which the whole
#:      four-layer converter design exists to avoid.
#:   2. Simply add a second name here with no further guard. Rejected: `content` and
#:      `tables` are NOT field-split, so either originator could set_block() straight
#:      over the other's. The mutual exclusivity that makes two originators safe would
#:      be an unwritten assumption instead of a check.
#:
#: This table authorises WRITES only. The read-time answer to "who wrote this block in
#: THIS record" is, as it always was, assembled.blocks[<block>].program. Any consumer
#: hardcoding a program name off this table was already reading the wrong contract —
#: see the ownership section of docs/document_schema.md.
#:
#: `forms` has no such conflict — it is always llm-enrich (VLM/LLM-driven field
#: extraction), regardless of whether the document is scanned or digital-born, so it
#: is a plain single-owner block from day one.
BLOCK_OWNERS: Dict[str, Union[str, Tuple[str, ...]]] = {
    "pages": ("alto-postprocess", "digital-convert"),
    "content": ("alto-postprocess", "digital-convert"),
    "lines": ("alto-postprocess", "digital-convert"),
    "tables": ("alto-postprocess", "digital-convert"),
    "page_categories": "page-classification",
    "translations": "translator",
    "entities": "nlp-enrich",
    "enrichment": "llm-enrich",
    "forms": "llm-enrich",
}

#: Which originator a document's `source.origin` authorises (Issue #18 §1a). Prefix
#: match, so "ocr:pero"/"ocr:tesseract-ces" and "digital-born-pdf" both resolve without
#: enumerating every engine. Order matters only in that the first matching prefix wins.
#:
#: Checked rather than assumed because the failure it catches is silent: a record
#: carrying half an OCR positional plane and half a digital-born one means the routing
#: that picks between them ran twice and disagreed, and nothing downstream would notice
#: — the schema requires only page+line on a lines[] row, so a half-built plane
#: validates clean.
#:
#: An origin not listed here is not an error: the check simply abstains, so a new
#: origin string can land before this table is taught about it (rule 6's spirit).
#: Matching is CASE-INSENSITIVE, and the bare `pdf` spelling is listed alongside `docx`.
#: Both were gaps rather than choices: the schema blesses bare `docx` but had no bare `pdf`,
#: so the symmetric spelling a converter author reaches for matched nothing, and `DOCX` or
#: `Digital-Born-PDF` or `abbyy-alto` matched nothing either. Since a non-match makes the
#: check abstain, each of those spellings silently switched §1a off for that document — the
#: opposite of the intended "refuse a mixed plane" behaviour, and invisible.
ORIGIN_ORIGINATORS: Tuple[Tuple[str, str], ...] = (
    ("digital-born", "digital-convert"),
    ("docx", "digital-convert"),
    ("pdf", "digital-convert"),
    ("ABBYY-ALTO", "alto-postprocess"),
    ("ocr:", "alto-postprocess"),
    ("vlm:", "alto-postprocess"),
)


def resolve_originator(origin: Optional[str]) -> Optional[str]:
    """
    The program authorised to originate a document's positional plane, from `source.origin`.

    Returns None when `origin` is empty or matches no known prefix — callers must treat that
    as "abstain", never as "refuse" (rule 6's spirit: a new acquisition method may land
    before this table is taught about it). Public because the converter's routing and
    merge_document_records() both need the same answer as the write-time check.
    """
    if not origin:
        return None
    folded = str(origin).casefold()
    for prefix, originator in ORIGIN_ORIGINATORS:
        if folded.startswith(prefix.casefold()):
            return originator
    return None


def _owner_candidates(name: str) -> Tuple[str, ...]:
    """BLOCK_OWNERS[name] normalised to a tuple — one entry for single-owner blocks."""
    owners = BLOCK_OWNERS.get(name)
    if not owners:
        return ()
    return (owners,) if isinstance(owners, str) else tuple(owners)


#: Field-level ownership inside list blocks that more than one tool contributes to.
#: A tool may only write the fields listed for it (plus the block's key fields).
BLOCK_FIELD_OWNERS: Dict[str, Dict[str, List[str]]] = {
    "pages": {
        # `page_index` and `needs_ocr_reason` are granted to BOTH originators.
        #   * page_index — it is the only ordering key that works when `page` is 'iv' or
        #     'A-1', and roman-numeral front matter is at least as common in scanned volumes
        #     as in digital-born ones. Granting it to the digital path alone left ALTO
        #     records with no ordering fallback at all, and the schema promises the label
        #     survives.
        #   * needs_ocr_reason — needs_ocr means OPPOSITE things on the two paths ("no text
        #     layer" vs "a text layer that decodes to garbage"), the renderer emits it as a
        #     cue the model reads, and with no field to carry the distinction every
        #     digital-born page rendered as "no extractable text layer", which is false.
        "alto-postprocess": [
            "page_index",
            "quality_score",
            "quality_band",
            "needs_ocr",
            "needs_ocr_reason",
            "ocr",
            "canvas",
        ],
        # Issue #18: the digital-born originator fills the same positional ROLE as
        # alto-postprocess (see BLOCK_OWNERS), so it needs substantially the same field
        # set — minus `ocr` (no OCR engine ran; leaving it unowned is what keeps
        # "was this OCR'd" answerable from the record) and plus `page_index`.
        #
        # `needs_ocr` IS granted, deliberately. Issue #10's research pass found
        # digital-born PDFs with non-embedded WinAnsi Helvetica and no /ToUnicode decode
        # to garbage across EVERY text parser (sondě -> sondI, hřeby -> hIeby) — a
        # systematic, not random, failure. This converter is the only component
        # positioned to detect it, and needs_ocr=True is how §3's "route per-page before
        # deferring to OCR" is expressed in the record. The converter REPORTS; routing
        # POLICY stays outside both tools.
        "digital-convert": [
            "page_index",
            "canvas",
            "quality_score",
            "quality_band",
            "needs_ocr",
            "needs_ocr_reason",
        ],
        "page-classification": ["category", "category_confidence"],
        "nlp-enrich": ["teitok_surface"],
    },
    "lines": {
        "alto-postprocess": ["categ", "quality_score", "lang", "text"],
        "nlp-enrich": ["lemma", "upos", "feats", "teitok_ref", "bbox"],
        # Issue #18: the digital-born originator. This MUST include `text` — the
        # earlier draft granted only ["group_id"], which merge_block() silently
        # honours: text and bbox were filtered out with no warning, and the result
        # still validated because lines[] only *requires* page+line. That is the
        # exact class of failure the round-trip assertion in the converter's
        # Layer D and tests/test_document_originators.py now pin.
        #
        # `bbox` is granted here as well as to nlp-enrich, deliberately: on the ALTO
        # path nlp-enrich derives it while aligning to TEITOK; on the digital-born
        # path there is no TEITOK to align to, and the PDF adapter's native
        # coordinates are the only bbox the record will ever have. The two never meet
        # on one document — enforced by _assert_origin_consistent(), not left to
        # convention.
        #
        # `style` closes the mapping table's last open row (plan §1, "Font family / size —
        # ⚠️ still homeless", to be decided in PR 1). Decided as the plan's own recommended
        # option (c): map SEMANTIC style only — bold / italic / heading_level — and drop
        # typeface and point size. A downstream reader can act on "this was a heading"; it
        # cannot act on "this was Helvetica 12pt", and heading-ness partly duplicates `categ`
        # already. python-docx (w:b, w:i, outlineLvl) and pdfplumber (fontname, size) both
        # supply what is needed. ALTO has no style signal, so the field is simply absent
        # there — the same additive shape as group_id.
        "digital-convert": [
            "text",
            "bbox",
            "group_id",
            "style",
            "lang",
            "quality_score",
            "categ",
        ],
    },
    #: `tables` is field-split for the same reason `pages` is: two candidate originators
    #: plus, on the ALTO path, no cell text of its own. Without this entry merge_block()
    #: fell through to `allowed = []` and emptied every row down to its key — the §1b
    #: silent-drop failure again, on a block the Definition of Done requires the converter
    #: to originate.
    "tables": {
        "alto-postprocess": ["page", "caption", "n_rows", "n_cols", "group_id", "cells"],
        "digital-convert": ["page", "caption", "n_rows", "n_cols", "group_id", "cells"],
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
        "translator": ["translation_en"],
        "llm-enrich": ["pid"],
    },
}

#: Natural key fields per list block, used to align records when merging by field.
#:
#: `tables` is here for the same reason `pages` is: BLOCK_OWNERS gives it two candidate
#: originators, so it is a real list block a tool may contribute to, and without an entry
#: merge_block("tables", …) raised `no key fields known` — leaving set_block() as the only
#: way in, which is wholesale replacement. `table_id` is the grid's identity; a table's
#: page is a property of it, not part of its key.
BLOCK_KEY_FIELDS: Dict[str, List[str]] = {
    "pages": ["page"],
    "lines": ["page", "line"],
    "tables": ["table_id"],
    "entities": ["page", "line", "char_span"],
}

#: Multi-dot pipeline suffixes to strip before falling back to a plain
#: ``split(".")[0]``. Longest/most-specific first, so ``.teitok.xml`` is
#: recognised before the generic ``.xml`` would otherwise short-circuit it.
#: Keep this list in sync across every tool that derives a doc_id from a
#: filename — see canonical_doc_id().
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


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


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


# ──────────────────────────────────────────────────────────────────────────────
# The record
# ──────────────────────────────────────────────────────────────────────────────


class DocumentRecord:
    """
    One document's aggregate record, opened by one tool for one contribution.

    Typical use, alongside the tool's existing ParadataLogger::

        with DocumentRecord.open(doc_id, "llm-enrich", baseline=args.document_json,
                                 run_id=logger.run_id) as doc:
            doc.set_block("enrichment", {"items": items})
            doc.add_regenerable("markdown", {"from": teitok_path,
                                             "converter": "xml_to_md@0.3.0",
                                             "detail": "full"})
        # → writes <out_dir>/<doc_id>.document.json
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
    ) -> None:
        if not doc_id:
            raise ValueError("doc_id is required — the record is keyed on it.")

        self.doc_id = doc_id
        self.program = program
        self.run_id = run_id or datetime.now(tz=timezone.utc).strftime("%y%m%d-%H%M%S")
        self.paradata_ref = paradata_ref or ""
        self.out_dir = out_dir
        self.strict = strict
        #: Opt-in: complain when merge_block() filters out a field the caller supplied.
        #: OFF by default deliberately. Some existing call sites pass context fields they
        #: do not own for readability, so switching this on globally is a tightening pass
        #: with its own call-site cleanup (Issue #18 §1b, "ecosystem" mitigation) rather
        #: than something to change under other work. A tool that wants the guarantee
        #: today asks for it here, or calls assert_fields_survived() after the merge.
        self.warn_dropped_fields = warn_dropped_fields

        # Rule 2/6: the baseline is deep-copied and never rewritten except where this tool writes.
        self._data: Dict[str, Any] = copy.deepcopy(baseline) if baseline else {}
        self._data.setdefault("schema_version", SCHEMA_VERSION)
        self._data.setdefault("record_type", RECORD_TYPE)
        self._data["doc_id"] = doc_id

        self._had_baseline = bool(baseline)
        self._touched: List[str] = []
        self._license_blocks: List[Dict[str, Any]] = []
        self._dropped_fields: Dict[str, List[str]] = {}
        #: Blocks written before `source.origin` was known, re-checked once it is.
        self._origin_deferred: List[str] = []
        #: Origins already reported as unrecognised, so the note is emitted once each.
        self._origin_unmatched: List[str] = []
        self._finalised = False

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

    def set_source(self, sha256: str = "", **fields: Any) -> "DocumentRecord":
        """
        Describe the ORIGINAL input. First writer wins — later tools must not overwrite it.

        The durable key is `doc_id` + `sha256`; `filename`/`media_type`/`origin`/`page_count`/
        `language` are metadata. Never a pipeline-local path to a derived artifact.

        A second call is still a no-op, but a second call carrying a DIFFERENT value now
        complains instead of discarding it in silence. Since Issue #18 §1a, `source.origin`
        is what authorises the positional blocks, so two stages disagreeing about it is not
        a cosmetic duplicate — it means the routing that picks between the OCR and the
        digital-born plane ran twice and reached two answers. Re-asserting the same values
        stays silent, which is the common harmless case.
        """
        existing = self._data.get("source")
        incoming = {k: v for k, v in fields.items() if v is not None}
        if sha256:
            incoming["sha256"] = sha256
        if existing:
            conflicts = sorted(
                f"{k}: {existing[k]!r} kept, {v!r} discarded"
                for k, v in _sanitise(incoming).items()
                if k in existing and existing[k] != v
            )
            if conflicts:
                self._complain(
                    "source is immutable after the first writer, but "
                    f"{self.program!r} passed conflicting values — {'; '.join(conflicts)}"
                )
            return self
        self._data["source"] = _sanitise(incoming)
        self._resolve_deferred_origin_checks()
        return self

    def set_block(self, name: str, payload: Any) -> "DocumentRecord":
        """Replace this tool's OWN block wholesale (rule 2)."""
        self._assert_owner(name)
        self._data[name] = _sanitise(payload)
        self._stamp(name)
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

        Existing records are matched on `key_fields` and only `own_fields` are written, so a
        co-contributor's fields on the same row survive untouched. New rows are appended.
        """
        if name in RESERVED_KEYS:
            raise ValueError(f"{name!r} is maintained by the module, not a tool block.")

        keys = key_fields or BLOCK_KEY_FIELDS.get(name)
        if not keys:
            raise ValueError(f"no key fields known for block {name!r} — pass key_fields=[...]")

        # `is not None`, not `or`: an explicit own_fields=[] means "write the key fields and
        # nothing else", and `or` used to treat that as "not supplied" and hand back the
        # program's full declared grant instead — writing more than the caller asked for. The
        # `allowed is None` test below shows None was always the intended sentinel.
        declared = BLOCK_FIELD_OWNERS.get(name, {})
        allowed = own_fields if own_fields is not None else declared.get(self.program)
        if allowed is None:
            self._complain(f"{self.program!r} has no declared field ownership in block {name!r}")
            allowed = []
        elif (
            own_fields is not None
            and self.program not in declared
            and self.program not in _owner_candidates(name)
        ):
            # own_fields is for a declared writer to NARROW or extend its own field set. It
            # must not confer writership: merge_block() never calls _assert_owner(), and
            # _assert_origin_consistent() abstains for non-candidates, so passing own_fields
            # was the one way an undeclared program could write any block and get stamped as
            # its author. The plan's Definition of Done asks for a mismatch to be refused on
            # "both set_block() and merge_block()"; without this it held only for programs
            # that happened to be declared.
            self._complain(
                f"{self.program!r} is neither an owner nor a declared field contributor of "
                f"block {name!r} — own_fields narrows an existing grant, it does not create one"
            )
        self._assert_origin_consistent(name)  # Issue #18 §1a

        writable = set(allowed) | set(keys)

        # Issue #18 §1b: record what this merge is about to throw away. The filtering below
        # is silent by design (a co-contributor passing context fields it does not own is
        # normal), and that silence is exactly how a grant of only ["group_id"] produced
        # records with no text in them that still validated. Tracking it costs nothing and
        # makes the loss inspectable — via dropped_fields() or assert_fields_survived() —
        # without changing behaviour for callers who have not asked.
        dropped = sorted({f for r in records for f in r} - writable)
        if dropped:
            self._dropped_fields[name] = dropped
            if self.warn_dropped_fields:
                self._complain(
                    f"{self.program!r} may not write {dropped} in block {name!r} — dropped. "
                    f"Declare them in BLOCK_FIELD_OWNERS or pass own_fields=[...]"
                )

        existing: List[Dict[str, Any]] = list(self._data.get(name) or [])
        index = {_record_key(r, keys): r for r in existing}

        for incoming in records:
            k = _record_key(incoming, keys)
            patch = {f: v for f, v in incoming.items() if f in writable}
            target = index.get(k)
            if target is None:
                existing.append(_sanitise(patch))
                index[k] = existing[-1]
            else:
                target.update(_sanitise(patch))

        self._data[name] = existing
        self._stamp(name)
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
            return (
                {name: list(self._dropped_fields.get(name, []))}
                if name in self._dropped_fields
                else {}
            )
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

    def add_derived_from(self, key: str, ref: str) -> "DocumentRecord":
        """Record a PERSISTENT step output this contribution was derived from."""
        block = self._data.setdefault("derived_from", {})
        block[key] = str(ref)
        self._stamp("derived_from")
        return self

    def add_regenerable(self, key: str, recipe: Dict[str, Any]) -> "DocumentRecord":
        """
        Record a DISPOSABLE derivation as a reproducible recipe, never a stored path.

        e.g. add_regenerable("markdown", {"from": "TEITOK/x.teitok.xml",
                                          "converter": "xml_to_md@0.3.0", "detail": "full"})
        """
        block = self._data.setdefault("regenerable", {})
        block[key] = _sanitise(recipe)
        self._stamp("regenerable")
        return self

    def add_license_detail(self, license_detail: Dict[str, Any]) -> "DocumentRecord":
        """Contribute this tool's paradata `license_detail` to the accreting union (rule 5)."""
        if license_detail:
            self._license_blocks.append(license_detail)
        return self

    # ── internals ───────────────────────────────────────────────────────────

    def _assert_owner(self, name: str) -> None:
        if name in RESERVED_KEYS:
            raise ValueError(f"{name!r} is maintained by the module, not a tool block.")
        owners = _owner_candidates(name)
        # The warning exists because a wholesale set_block() on a field-split block erases
        # every CO-CONTRIBUTOR's fields. Alternative ORIGINATORS are not co-contributors —
        # they are mutually exclusive per document (§1a), so they can never both have fields
        # on one record to erase. `tables` is declared for both originators and nobody else,
        # so set_block() is the correct call for it; `pages`, `lines` and `entities` each
        # still have a genuine co-contributor and still warn exactly as before.
        co_contributors = sorted(
            set(BLOCK_FIELD_OWNERS.get(name, {})) - {self.program} - set(owners)
        )
        if co_contributors:
            self._complain(
                f"block {name!r} is field-split with {co_contributors} — "
                f"use merge_block(), not set_block(), or a co-contributor's fields will be lost"
            )
        if owners and self.program not in owners:
            self._complain(
                f"block {name!r} is owned by {' or '.join(owners)}, not {self.program!r}"
            )
        self._assert_origin_consistent(name)

    def _assert_origin_consistent(self, name: str) -> None:
        """
        Issue #18 §1a: for a block with several possible originators, the document's
        `source.origin` decides which one may write it.

        Self-guarding, so calling it unconditionally from both set_block() and
        merge_block() is a no-op for every pre-#18 caller. It returns early for:
          * single-owner blocks (len(owners) < 2);
          * programs that are not originator candidates at all — nlp-enrich merging
            morphology into lines[] is a field contribution, not an origination claim;
          * records with no `source` yet — but only PROVISIONALLY: the block is remembered
            and re-checked as soon as an origin is known (see below);
          * origins this table has not been taught (abstain rather than block, rule 6's
            spirit — a note goes to stderr so the silence is at least visible);
          * a document whose own record already asks for OCR re-acquisition (see
            `_ocr_handoff_requested`).

        The deferral matters. This check reads `source.origin`, so a run that wrote its
        blocks BEFORE calling set_source() used to escape it *permanently* — and since
        set_source() is first-writer-wins, the wrong origin was then frozen in. Plan §2's
        Layer C says set_source() must come first "since that is what authorizes them", but
        that ordering was documented in the plan only, enforced nowhere, and produced
        exactly the half-OCR/half-digital positional plane §1a exists to refuse, silently,
        with strict=True. Deferring instead of abstaining removes the ordering requirement
        from the caller altogether.

        Deliberately reads `source.origin` rather than assembled.blocks[name].program:
        rule 4 says the stamp on a field-split block names the MOST RECENT writer, so
        once nlp-enrich merges into lines[] the originator signal is gone. `source` is
        immutable after first write and `origin` is the field already designed to record
        how the text was obtained.
        """
        owners = _owner_candidates(name)
        if len(owners) < 2 or self.program not in owners:
            return
        origin = (self._data.get("source") or {}).get("origin")
        if not origin:
            if name not in self._origin_deferred:
                self._origin_deferred.append(name)
            return

        originator = resolve_originator(origin)
        if originator is None:
            if origin not in self._origin_unmatched:
                self._origin_unmatched.append(origin)
                self._note(
                    f"source.origin {origin!r} matches no ORIGIN_ORIGINATORS prefix — the "
                    f"§1a originator check is ABSTAINING for {self.program!r} on block "
                    f"{name!r}. Teach ORIGIN_ORIGINATORS this origin to have it enforced."
                )
            return
        if originator == self.program:
            return
        if self.program == "alto-postprocess" and self._ocr_handoff_requested():
            # The documented digital-born -> OCR hand-off, not a mixed plane by accident.
            # digital-convert is granted pages[].needs_ocr precisely so it can say "this
            # page's embedded text layer does not decode; re-acquire it by OCR" (Issue #10's
            # WinAnsi/no-/ToUnicode corruption), and the schema says routing policy then acts
            # on it. Without this branch that hand-off was unreachable: origin is frozen at
            # `digital-born-*`, so every pages/lines write alto-postprocess made afterwards
            # was refused, and §3's "route per page before deferring to OCR" contradicted
            # §1a's "no document is ever both".
            #
            # It stays truthful. `source.origin` describes how the ORIGINAL INPUT was
            # acquired, and that really was a digital-born PDF; the OCR ran over its rendered
            # pages. Who wrote the plane is answered where rule 4 says it is —
            # assembled.blocks[<block>].program — and `pages[].ocr` (never granted to
            # digital-convert) records that an engine ran. The authorisation is the
            # converter's own recorded request, so it is auditable from the record itself.
            self._note(
                f"block {name!r}: honouring the pages[].needs_ocr hand-off — "
                f"{self.program!r} re-originating a {origin!r} document"
            )
            return
        self._complain(
            f"block {name!r}: source.origin {origin!r} is originated by "
            f"{originator!r}, not {self.program!r}"
        )

    def _ocr_handoff_requested(self) -> bool:
        """True when this record's own `pages[]` asks for OCR re-acquisition."""
        for page in self._data.get("pages") or []:
            if isinstance(page, dict) and page.get("needs_ocr") is True:
                return True
        return False

    def _resolve_deferred_origin_checks(self) -> None:
        """Re-run the origin check for blocks written before `source.origin` was known."""
        pending, self._origin_deferred = list(self._origin_deferred), []
        for name in pending:
            self._assert_origin_consistent(name)

    def _complain(self, message: str) -> None:
        if self.strict:
            raise ValueError(message)
        print(f"[document] WARNING – {message}", file=sys.stderr)

    def _note(self, message: str) -> None:
        """Visible but never fatal — for things a reader should know that are not errors.

        Distinct from _complain() on purpose: an unrecognised `source.origin` must keep
        abstaining even under strict=True (rule 6's spirit — a new acquisition method may
        land before this module is taught about it), but abstaining in complete silence is
        how §1a quietly stops applying to a whole class of documents.
        """
        print(f"[document] NOTE – {message}", file=sys.stderr)

    def _stamp(self, block: str) -> None:
        """Rule 4: per-block provenance — this is where granularity comes from."""
        if block not in self._touched:
            self._touched.append(block)
        blocks = self._data.setdefault("assembled", {}).setdefault("blocks", {})
        blocks[block] = {
            "program": self.program,
            "run_id": self.run_id,
            "paradata_ref": self.paradata_ref,
            "updated_at": _utc_now_iso(),
        }

    def _provenance(self) -> Dict[str, Any]:
        prov: Dict[str, Any] = dict(self._data.get("provenance") or {})
        prior = prov.get("license_detail")
        blocks = ([prior] if prior else []) + self._license_blocks

        if merge_effective_licenses is not None and blocks:
            merged = merge_effective_licenses(blocks)
            prov["license"] = merged.get("effective_license", LICENSE_NAME)
            prov["license_url"] = merged.get("effective_license_url", LICENSE_URL)
            prov["license_detail"] = merged
        elif not prov.get("license"):
            prov["license"] = LICENSE_NAME
            prov["license_url"] = LICENSE_URL
            prov["license_note"] = (
                "License helper unavailable or no components recorded; defaulted conservatively to CC BY-NC 4.0."
            )

        contributors: List[Dict[str, str]] = list(prov.get("contributors") or [])
        if self._touched and not any(
            c.get("program") == self.program and c.get("run_id") == self.run_id
            for c in contributors
        ):
            contributors.append(
                {
                    "program": self.program,
                    "run_id": self.run_id,
                    "paradata_ref": self.paradata_ref,
                    "blocks": ",".join(self._touched),
                    "at": _utc_now_iso(),
                }
            )
        prov["contributors"] = contributors
        return prov

    # ── output ──────────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """The record as it would be written — baseline passed through, own blocks applied."""
        # Last chance for the §1a check on blocks written before set_source(). A run that
        # never calls set_source() at all is still fine (rule 3); this only fires when an
        # origin arrived late and contradicts what was already written.
        self._resolve_deferred_origin_checks()
        out = copy.deepcopy(self._data)
        out["provenance"] = self._provenance()
        assembled = out.setdefault("assembled", {})
        assembled["had_baseline"] = self._had_baseline
        assembled["note"] = (
            "Blocks reflect CONTRIBUTED steps only; a block is absent until its tool has run."
        )
        # Stable, predictable key order for diff-friendly output.
        order = [
            "schema_version",
            "record_type",
            "doc_id",
            "source",
            "derived_from",
            "regenerable",
            "provenance",
            "assembled",
            "page_categories",
            "pages",
            "content",
            "lines",
            "tables",
            "entities",
            "translations",
            "enrichment",
            "forms",
        ]
        ordered = {k: out[k] for k in order if k in out}
        for k in out:  # any unknown/newer block is preserved (rule 6)
            if k not in ordered:
                ordered[k] = out[k]
        return ordered

    def finalize(self, out_path: Optional[str] = None) -> str:
        if self._finalised:
            # No `from None`: that only means anything inside an except block, and here it
            # would hide a real cause if finalize() is ever retried from a handler.
            raise RuntimeError("finalize() has already been called.")
        if not self._touched:
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


def migrate_document(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply schema migrations up to the current SCHEMA_VERSION.

    `1.0` is the first published version, so there is nothing to migrate yet. When a breaking
    bump lands, add `_migrate_1_0_to_2_0()` and branch here — the same sequential pattern
    `atrium_paradata.migrate_paradata()` uses.
    """
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
    Validate a record against the canonical schema. Raises on failure, returns None on pass.

    The one call plan §2's Layer D needs, in the module that owns the contract, so the rule
    "no doc.json is emitted if validation fails" does not have to be re-implemented per tool.
    `jsonschema` is an optional import: without it this raises RuntimeError rather than
    passing silently, because a validation gate that quietly becomes a no-op is worse than
    no gate — you cannot tell the two apart from the output.
    """
    try:
        import jsonschema
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise RuntimeError(
            "jsonschema is not installed, so the record cannot be validated. Install it "
            "(requirements-test.txt / requirements_digital.txt) rather than skipping the gate."
        ) from exc
    jsonschema.validate(record, load_schema())


def load_document(path: str) -> Dict[str, Any]:
    """Read a document record, migrating older schemas transparently (rule 6)."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    v = str(data.get("schema_version", SCHEMA_VERSION))
    major = int(v.split(".")[0])
    current_major = int(SCHEMA_VERSION.split(".")[0])

    if major > current_major:
        raise ValueError(
            f"Schema version {v} is newer than supported {SCHEMA_VERSION}. Please update tools."
        )
    elif major < current_major:
        data = migrate_document(data)

    return data


# ──────────────────────────────────────────────────────────────────────────────
# Merging Logic (only for parallel branches re-joining)
# ──────────────────────────────────────────────────────────────────────────────


def merge_document_records(json_paths: List[str], out_path: str) -> str:
    """
    Fold several partial records for the SAME document into one.

    The pipeline is linear, so the normal path needs no merge — each tool hands its output to
    the next. This exists for fan-out/fan-in (e.g. two tools run in parallel on one document)
    and resolves per block using `assembled.blocks[*].updated_at`: newest contribution wins.

    Three keys are NOT resolved that way, because "newest wins" is wrong for them:

    * **`source`** is first-writer-wins for the life of the record (and since Issue #18 §1a
      it is what authorises the positional plane). It carries no `assembled.blocks` stamp, so
      both sides of the `updated_at` comparison were the empty string, `"" >= ""` was true,
      and every input file overwrote the previous one — last-path-wins, silently, dropping
      the first record's `sha256` and able to swap the origin out from under an
      already-written plane. Now the first non-empty value wins per sub-key, and a
      contradiction between two inputs is reported rather than resolved by argument order.
    * **`derived_from`** and **`regenerable`** are append-only maps whose whole purpose is to
      accumulate one entry per stage. Replacing them wholesale threw away every entry the
      losing branch had added — real provenance loss on exactly the fan-in this function
      exists for. They are merged key-wise instead, first writer winning per key.

    A fan-in is also a third write path that bypassed the §1a check entirely, so the merged
    positional plane is verified against the merged `source.origin` at the end.
    """
    if not json_paths:
        raise ValueError("no record paths given")

    merged: Dict[str, Any] = {}
    stamps: Dict[str, Dict[str, Any]] = {}
    license_blocks: List[Dict[str, Any]] = []
    contributors: List[Dict[str, str]] = []
    doc_ids: List[str] = []
    source: Dict[str, Any] = {}
    source_conflicts: List[str] = []
    accumulating: Dict[str, Dict[str, Any]] = {"derived_from": {}, "regenerable": {}}

    for p in json_paths:
        data = load_document(p)
        doc_id = data.get("doc_id", "")
        if doc_id and doc_id not in doc_ids:
            doc_ids.append(doc_id)

        blocks = (data.get("assembled") or {}).get("blocks") or {}
        prov = data.get("provenance") or {}
        if prov.get("license_detail"):
            license_blocks.append(prov["license_detail"])
        for c in prov.get("contributors") or []:
            if c not in contributors:
                contributors.append(c)

        for key, value in (data.get("source") or {}).items():
            if key not in source:
                source[key] = value
            elif source[key] != value:
                source_conflicts.append(
                    f"{key}: {source[key]!r} kept, {value!r} from {p} discarded"
                )

        for key, bucket in accumulating.items():
            for sub, value in (data.get(key) or {}).items():
                bucket.setdefault(sub, value)

        for key, value in data.items():
            if key in ("assembled", "provenance", "source") or key in accumulating:
                continue
            incoming = blocks.get(key, {}).get("updated_at", "")
            held = stamps.get(key, {}).get("updated_at", "")
            # `>` not `>=`: on a tie (or on two unstamped keys) the FIRST record read wins,
            # so the result does not depend on the order json_paths happens to be in.
            if key not in merged or incoming > held:
                merged[key] = value
                if key in blocks:
                    stamps[key] = blocks[key]

    if len(doc_ids) > 1:
        raise ValueError(f"records belong to different documents: {doc_ids}")
    if source_conflicts:
        raise ValueError(
            "records disagree about the immutable `source` of the same document — "
            + "; ".join(sorted(source_conflicts))
        )

    if source:
        merged["source"] = source
    for key, bucket in accumulating.items():
        if bucket:
            merged[key] = bucket

    # Issue #18 §1a on the fan-in path: a merged record must not carry half an OCR
    # positional plane and half a digital-born one. Each block's stamp names who wrote it,
    # and `source.origin` says who was authorised to.
    authorised = resolve_originator(source.get("origin"))
    if authorised:
        for block, stamp in stamps.items():
            if len(_owner_candidates(block)) < 2:
                continue
            wrote = stamp.get("program")
            if wrote and wrote in _owner_candidates(block) and wrote != authorised:
                raise ValueError(
                    f"merged record mixes positional originators: block {block!r} was written "
                    f"by {wrote!r} but source.origin {source['origin']!r} authorises "
                    f"{authorised!r}. One of the inputs was produced outside the §1a contract."
                )

    if merge_effective_licenses is not None and license_blocks:
        lic = merge_effective_licenses(license_blocks)
    else:
        lic = {
            "effective_license": LICENSE_NAME,
            "effective_license_url": LICENSE_URL,
            "notes": "License helper unavailable; defaulted to CC BY-NC 4.0.",
        }

    merged["schema_version"] = SCHEMA_VERSION
    merged["record_type"] = RECORD_TYPE_MERGED
    merged["provenance"] = {
        "license": lic["effective_license"],
        "license_url": lic["effective_license_url"],
        "license_detail": lic,
        "contributors": contributors,
    }
    merged["assembled"] = {
        "blocks": stamps,
        "merged_from": len(json_paths),
        "merged_at": _utc_now_iso(),
        "note": "Blocks reflect CONTRIBUTED steps only; newest contribution per block wins.",
    }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(merged, fh, ensure_ascii=False, indent=2)
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
    st.add_argument("--run-id", default=None)
    st.add_argument("--paradata-ref", default="")
    st.add_argument("--strict", action="store_true")

    me = sub.add_parser("merge", help="fold parallel partial records for one document")
    me.add_argument("--paths", nargs="+", required=True)
    me.add_argument("--out", required=True)

    mi = sub.add_parser("migrate", help="rewrite a record at the current schema version")
    mi.add_argument("--path", required=True)

    args = p.parse_args()

    if args.cmd == "set-block":
        raw = (
            sys.stdin.read() if args.payload == "-" else open(args.payload, encoding="utf-8").read()
        )
        with DocumentRecord.open(
            args.doc_id,
            args.program,
            baseline=args.baseline,
            run_id=args.run_id,
            paradata_ref=args.paradata_ref,
            strict=args.strict,
        ) as doc:
            doc.set_block(args.block, json.loads(raw))
            if args.out:
                doc.finalize(args.out)

    elif args.cmd == "merge":
        merge_document_records(args.paths, args.out)

    elif args.cmd == "migrate":
        data = load_document(args.path)
        with open(args.path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        print(f"[document] Migrated {args.path} to {SCHEMA_VERSION}", flush=True)


if __name__ == "__main__":
    _cli()
