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
RESERVED_KEYS = frozenset({"schema_version", "record_type", "doc_id", "source", "provenance", "assembled"})

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
ORIGIN_ORIGINATORS: Tuple[Tuple[str, str], ...] = (
    ("digital-born", "digital-convert"),
    ("docx", "digital-convert"),
    ("ABBYY-ALTO", "alto-postprocess"),
    ("ocr:", "alto-postprocess"),
    ("vlm:", "alto-postprocess"),
)


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
        "alto-postprocess": ["quality_score", "quality_band", "needs_ocr", "ocr", "canvas"],
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
        "digital-convert": ["page_index", "canvas", "quality_score", "quality_band", "needs_ocr"],
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
        "digital-convert": ["text", "bbox", "group_id", "lang", "quality_score", "categ"],
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
BLOCK_KEY_FIELDS: Dict[str, List[str]] = {
    "pages": ["page"],
    "lines": ["page", "line"],
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


def _record_key(record: Dict[str, Any], key_fields: Iterable[str]) -> tuple:
    return tuple(json.dumps(record.get(k), sort_keys=True, default=str) for k in key_fields)


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
    ) -> None:
        if not doc_id:
            raise ValueError("doc_id is required — the record is keyed on it.")

        self.doc_id = doc_id
        self.program = program
        self.run_id = run_id or datetime.now(tz=timezone.utc).strftime("%y%m%d-%H%M%S")
        self.paradata_ref = paradata_ref or ""
        self.out_dir = out_dir
        self.strict = strict

        # Rule 2/6: the baseline is deep-copied and never rewritten except where this tool writes.
        self._data: Dict[str, Any] = copy.deepcopy(baseline) if baseline else {}
        self._data.setdefault("schema_version", SCHEMA_VERSION)
        self._data.setdefault("record_type", RECORD_TYPE)
        self._data["doc_id"] = doc_id

        self._had_baseline = bool(baseline)
        self._touched: List[str] = []
        self._license_blocks: List[Dict[str, Any]] = []
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
        """
        if self._data.get("source"):
            return self
        src = {k: v for k, v in fields.items() if v is not None}
        if sha256:
            src["sha256"] = sha256
        self._data["source"] = _sanitise(src)
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

        allowed = own_fields or BLOCK_FIELD_OWNERS.get(name, {}).get(self.program)
        if allowed is None:
            self._complain(f"{self.program!r} has no declared field ownership in block {name!r}")
            allowed = []
        self._assert_origin_consistent(name)  # Issue #18 §1a

        writable = set(allowed) | set(keys)
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
        if name in BLOCK_FIELD_OWNERS:
            self._complain(
                f"block {name!r} is field-split across {sorted(BLOCK_FIELD_OWNERS[name])} — "
                f"use merge_block(), not set_block(), or a co-contributor's fields will be lost"
            )
        owners = _owner_candidates(name)
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
          * records with no `source` yet (rule 3: standalone runs emit their own part);
          * origins this table has not been taught (abstain rather than block).

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
            return
        for prefix, originator in ORIGIN_ORIGINATORS:
            if str(origin).startswith(prefix):
                if originator != self.program:
                    self._complain(
                        f"block {name!r}: source.origin {origin!r} is originated by "
                        f"{originator!r}, not {self.program!r}"
                    )
                return

    def _complain(self, message: str) -> None:
        if self.strict:
            raise ValueError(message)
        print(f"[document] WARNING – {message}", file=sys.stderr)

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
            c.get("program") == self.program and c.get("run_id") == self.run_id for c in contributors
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
        out = copy.deepcopy(self._data)
        out["provenance"] = self._provenance()
        assembled = out.setdefault("assembled", {})
        assembled["had_baseline"] = self._had_baseline
        assembled["note"] = "Blocks reflect CONTRIBUTED steps only; a block is absent until its tool has run."
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
            raise RuntimeError("finalize() has already been called.") from None
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


def load_document(path: str) -> Dict[str, Any]:
    """Read a document record, migrating older schemas transparently (rule 6)."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    v = str(data.get("schema_version", SCHEMA_VERSION))
    major = int(v.split(".")[0])
    current_major = int(SCHEMA_VERSION.split(".")[0])

    if major > current_major:
        raise ValueError(f"Schema version {v} is newer than supported {SCHEMA_VERSION}. Please update tools.")
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
    """
    if not json_paths:
        raise ValueError("no record paths given")

    merged: Dict[str, Any] = {}
    stamps: Dict[str, Dict[str, Any]] = {}
    license_blocks: List[Dict[str, Any]] = []
    contributors: List[Dict[str, str]] = []
    doc_ids: List[str] = []

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

        for key, value in data.items():
            if key in ("assembled", "provenance"):
                continue
            incoming = blocks.get(key, {}).get("updated_at", "")
            held = stamps.get(key, {}).get("updated_at", "")
            if key not in merged or incoming >= held:
                merged[key] = value
                if key in blocks:
                    stamps[key] = blocks[key]

    if len(doc_ids) > 1:
        raise ValueError(f"records belong to different documents: {doc_ids}")

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
        raw = sys.stdin.read() if args.payload == "-" else open(args.payload, encoding="utf-8").read()
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
