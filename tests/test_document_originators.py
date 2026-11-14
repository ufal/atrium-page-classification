"""
tests/test_document_originators.py

Record schema 2.0 (doc-schema-v2, atrium-project#84): who writes what in the document record.

Until 2.0 the positional plane (`pages`, `content`, `lines`, `tables`) had two originators, chosen
per record by `source.origin`, plus a scoring co-contribution and a per-page OCR hand-off, and every
block carried a stamp of the run that wrote it. The contract of 12 November replaced all of it:

  1. ONE writer of the positional blocks, the record builder (ocr-postprocess), which builds them
     from the stored `atr/docproc-json` whatever produced it (D20). digital-convert writes docproc
     and owns no block; `source.origin` is descriptive only.
  2. No run data in the record (D1): no `provenance`, `assembled`, `derived_from`, `regenerable`;
     what a run wrote is its paradata `result` (`DocumentRecord.written()`).
  3. No layout in the record (D26): no `tables`, `content.reading_order`, `lines[].group_id|style`,
     no page verdicts; each line names its docproc block, text source and box precision (D25).
  4. A 1.x record is migrated on load (`_migrate_1_to_2()`).

These tests pin each of those, and the field grants every tool's merge depends on.

Canonical copy
--------------
This file is **hub-canonical**: `docs/templates/shared/test_document_originators.py` in
atrium-project, vendored into every tool repo at `tests/test_document_originators.py` beside the
canonical `atrium_document.py` + `atrium_document.schema.json`. `para-drift.reusable.yml` `diff -u`s
all copies, so edit the hub copy and re-vendor with `scripts/revendor_shared.sh`.

The imports below are plain top-level imports and must stay that way: pytest puts this file's
directory on `sys.path` in the hub, and `pytest.ini`'s `pythonpath = .` does it in a tool repo.
"""

import copy
import json

import pytest

from atrium_document import (
    BLOCK_FIELD_OWNERS,
    BLOCK_KEY_FIELDS,
    BLOCK_OWNERS,
    PROGRAM_SUCCESSORS,
    RECORD_BUILDER,
    REMOVED_BLOCKS,
    SCHEMA_VERSION,
    DocumentRecord,
    RemovedInSchema2,
    canonical_program,
    load_document,
    merge_document_records,
    migrate_document,
    quality_summary,
    same_program,
    validate_baseline,
    validate_document,
)
from atrium_paradata import ParadataLogger

BUILDER = "ocr-postprocess"
DIGITAL = "digital-convert"
SHA512 = "a" * 128


def _seed():
    return {
        "schema_version": "2.0",
        "record_type": "atrium-document",
        "doc_id": "CTX000000001",
        "source": {"sha512": SHA512, "filename": "CTX000000001.pdf", "media_type": "application/pdf"},
    }


def _open(tmp_path, program, baseline=None, strict=True, **kwargs):
    """A record opened as `program`, written under tmp_path (never the repo root)."""
    return DocumentRecord(
        doc_id="CTX000000001",
        program=program,
        baseline=baseline if baseline is not None else _seed(),
        out_dir=str(tmp_path),
        strict=strict,
        **kwargs,
    )


def _schema_error():
    """What `validate_document()` raises for a record the schema refuses: jsonschema's
    ValidationError. Without jsonschema the test is skipped, so a missing validator (a
    RuntimeError) can never pass for a refusal."""
    return pytest.importorskip("jsonschema").ValidationError


def _line(page="1", line=1, **extra):
    row = {
        "page": page,
        "line": line,
        "text": "Náčrt sondy.",
        "bbox": [72.0, 100.0, 300.0, 114.0],
        "block": "p1-b0",
        "text_source": "atr",
        "bbox_precision": "derived",
    }
    row.update(extra)
    return row


def _built(tmp_path):
    """The record the builder writes from docproc: two pages, one empty, one line, scored."""
    doc = _open(tmp_path, BUILDER)
    doc.set_source(language=["cs"], page_count=2, origin="atr")
    doc.merge_block(
        "pages",
        [
            {"page": "1", "page_index": 1, "canvas": {"width": 595, "height": 842, "unit": "pt"}, "content_status": "text"},
            {"page": "2", "page_index": 2, "content_status": "empty"},
        ],
    )
    doc.merge_block("lines", [_line(categ="Clear", quality_score=0.93, lang="cs")])
    doc.merge_block("pages", [{"page_index": 1, "quality_score": 0.93, "quality_band": "Clear"}])
    doc.set_block("content", {"text": "Náčrt sondy."})
    doc.set_block("quality_summary", quality_summary(doc.to_dict()))
    return doc


# ── the grants ───────────────────────────────────────────────────────────────


def test_the_positional_blocks_have_one_writer_the_record_builder():
    """D20: ocr-postprocess builds pages, content and lines from docproc; digital-convert owns none."""
    assert RECORD_BUILDER == BUILDER
    for block in ("pages", "content", "lines", "quality_summary"):
        assert BLOCK_OWNERS[block] == BUILDER, block
    assert DIGITAL not in json.dumps(BLOCK_OWNERS) and DIGITAL not in json.dumps(BLOCK_FIELD_OWNERS)
    for block in REMOVED_BLOCKS:
        assert block not in BLOCK_OWNERS and block not in BLOCK_FIELD_OWNERS


def test_the_field_grants_are_the_contracts_table():
    """Contract §5: who writes which field. A change here is a change of the contract."""
    assert BLOCK_FIELD_OWNERS["pages"] == {
        BUILDER: ["page", "canvas", "content_status", "quality_score", "quality_band"],
        "page-classification": ["category", "category_confidence"],
        "nlp-enrich": ["teitok_surface"],
    }
    assert BLOCK_FIELD_OWNERS["lines"] == {
        BUILDER: ["text", "bbox", "block", "text_source", "bbox_precision", "unit", "categ", "quality_score", "lang"],
        "nlp-enrich": ["lemma", "upos", "feats", "teitok_ref"],
    }
    assert set(BLOCK_FIELD_OWNERS["entities"]) == {"nlp-enrich", "translator", "keyword-extract"}
    assert BLOCK_FIELD_OWNERS["entities"]["keyword-extract"] == ["pid"]
    # A page row is keyed by its physical page number, the join key with docproc (H2).
    assert BLOCK_KEY_FIELDS == {"pages": ["page_index"], "lines": ["page", "line"], "entities": ["page", "line", "char_span"]}
    assert (BLOCK_OWNERS["enrichment"], BLOCK_OWNERS["keywords"]) == ("keyword-extract", "keyword-extract")


def test_a_predecessor_program_id_writes_as_its_successor(tmp_path):
    """The repository moves of 2026-10-01: alto-postprocess is ocr-postprocess."""
    assert PROGRAM_SUCCESSORS["alto-postprocess"] == BUILDER
    assert canonical_program("alto-postprocess") == BUILDER and same_program("llm-enrich", "keyword-extract")
    doc = _open(tmp_path, "alto-postprocess")
    doc.merge_block("lines", [_line()])
    assert doc.get_block("lines")[0]["text"] == "Náčrt sondy."


# ── the record builder ──────────────────────────────────────────────────────


def test_the_builder_writes_a_valid_record(tmp_path):
    out = _built(tmp_path).to_dict()
    validate_document(out)
    assert out["schema_version"] == SCHEMA_VERSION == "2.0"
    assert out["source"] == {
        "sha512": SHA512,
        "filename": "CTX000000001.pdf",
        "media_type": "application/pdf",
        "language": ["cs"],
        "page_count": 2,
        "origin": "atr",
    }


def test_a_line_survives_with_its_box_block_source_and_precision(tmp_path):
    """The §1b class of failure (a granted field filtered out in silence) must not come back."""
    doc = _built(tmp_path)
    row = _line(categ="Clear", quality_score=0.93, lang="cs")
    doc.assert_fields_survived("lines", [row])
    assert doc.dropped_fields() == {}


def test_written_names_the_blocks_and_fields_of_this_run(tmp_path):
    """What the paradata `result` names for the record (#84 §4), in place of stamps."""
    written = _built(tmp_path).written()
    assert written["source"] == ["language", "page_count", "origin"]
    assert set(written["lines"]) >= {"page", "line", "text", "bbox", "block", "text_source", "bbox_precision", "categ"}
    assert written["pages"][:2] == ["page", "page_index"] and "quality_band" in written["pages"]
    assert written["content"] == ["text"] and written["quality_summary"] == ["pages", "lines"]


def test_quality_summary_counts_empty_pages_apart(tmp_path):
    """D18 rule 7: an empty page has no lines and no score, and is counted apart."""
    summary = _built(tmp_path).to_dict()["quality_summary"]
    assert summary["pages"] == {"total": 2, "empty": 1, "scored": 1, "mean": 0.93, "median": 0.93, "min": 0.93, "max": 0.93}
    assert summary["lines"] == {"total": 1, "by_categ": {"Clear": 1}}


def test_a_block_scored_as_one_unit_stays_out_of_the_calibrated_counts(tmp_path):
    """D18 rule 5: a row marked `unit: block` keeps its score but is counted apart from `by_categ`."""
    doc = _built(tmp_path)
    doc.merge_block("lines", [_line(line=2, text="Celý odstavec bez zalomení", unit="block", categ="Clear", quality_score=0.9)])
    doc.set_block("quality_summary", quality_summary(doc.to_dict()))
    out = doc.to_dict()
    assert out["quality_summary"]["lines"] == {"total": 2, "by_categ": {"Clear": 1}, "block_units": 1}
    assert out["lines"][1]["unit"] == "block" and out["lines"][1]["quality_score"] == 0.9
    validate_document(out)
    out["lines"][1]["unit"] = "line"  # only `block` is written; a visual line has no `unit`
    with pytest.raises(_schema_error()):
        validate_document(out)


def test_digital_convert_cannot_write_the_record(tmp_path):
    """It writes docproc since 2.0 (D20); a write into the record is refused."""
    doc = _open(tmp_path, DIGITAL)
    with pytest.raises(ValueError):
        doc.merge_block("lines", [_line()])
    with pytest.raises(ValueError):
        doc.set_block("content", {"text": "x"})
    with pytest.raises(ValueError):
        doc.set_source(origin="digital")


# ── the other writers ───────────────────────────────────────────────────────


def test_nlp_enrich_adds_its_fields_and_nothing_positional(tmp_path):
    base = _built(tmp_path).to_dict()
    doc = _open(tmp_path, "nlp-enrich", baseline=base, strict=False)
    doc.merge_block("lines", [{"page": "1", "line": 1, "lemma": "náčrt", "upos": "NOUN", "bbox": [0, 0, 1, 1], "text": "x"}])
    row = doc.get_block("lines")[0]
    assert (row["lemma"], row["text"], row["bbox"]) == ("náčrt", "Náčrt sondy.", [72.0, 100.0, 300.0, 114.0])
    assert doc.dropped_fields("lines") == {"lines": ["bbox", "text"]}
    doc.merge_block("pages", [{"page_index": 1, "teitok_surface": "facs-1"}])
    doc.set_block("entities", [{"surface": "Praha", "page": "1", "line": 1, "char_span": [0, 5], "bbox": [1, 2, 3, 4]}])
    validate_document(doc.to_dict())
    assert doc.written() == {
        "lines": ["page", "line", "lemma", "upos"],
        "pages": ["page_index", "teitok_surface"],
        "entities": ["surface", "page", "line", "char_span", "bbox"],
    }


def test_page_classification_and_keyword_extract_add_their_fields(tmp_path):
    base = _built(tmp_path).to_dict()
    pc = _open(tmp_path, "page-classification", baseline=base)
    pc.set_block("page_categories", {"1": "TEXT"})
    pc.merge_block("pages", [{"page_index": 1, "category": "TEXT", "category_confidence": 0.97}])
    kw = _open(tmp_path, "keyword-extract", baseline=pc.to_dict())
    kw.set_block("enrichment", {"items": [], "summary": None, "topics": []})
    kw.set_block("keywords", {"document": [], "pages": []})
    out = kw.to_dict()
    validate_document(out)
    assert out["pages"][0]["category"] == "TEXT" and out["pages"][0]["quality_band"] == "Clear"


def test_a_standalone_page_classification_run_is_valid(tmp_path):
    """Rule 3: no baseline, no seed — its own part only, keyed by the physical page (H2)."""
    doc = DocumentRecord("CTX000000001", "page-classification", out_dir=str(tmp_path), strict=True)
    doc.set_block("page_categories", {"1": "TEXT"})
    doc.merge_block("pages", [{"page_index": 1, "category": "TEXT", "category_confidence": 0.97}])
    validate_document(doc.to_dict())


def test_a_record_with_nothing_written_is_refused(tmp_path):
    with pytest.raises(_schema_error()):
        validate_document(DocumentRecord("CTX000000001", "nlp-enrich", out_dir=str(tmp_path)).to_dict())


def test_set_block_on_a_field_split_block_warns_of_lost_co_contributor_fields(tmp_path):
    doc = _open(tmp_path, BUILDER)
    with pytest.raises(ValueError, match="field-split"):
        doc.set_block("lines", [_line()])


def test_own_fields_narrows_a_grant_and_never_creates_one(tmp_path):
    doc = _open(tmp_path, "translator")
    with pytest.raises(ValueError, match="neither an owner"):
        doc.merge_block("lines", [{"page": "1", "line": 1, "text": "x"}], own_fields=["text"])


# ── what 2.0 removed ────────────────────────────────────────────────────────


@pytest.mark.parametrize("block", REMOVED_BLOCKS)
def test_a_removed_block_cannot_be_written(tmp_path, block):
    doc = _open(tmp_path, BUILDER)
    with pytest.raises(ValueError, match="removed in schema 2.0|maintained by the module"):
        doc.set_block(block, {})


def test_a_removed_field_is_dropped_with_a_complaint(tmp_path):
    doc = _open(tmp_path, BUILDER)
    with pytest.raises(ValueError, match="removed in schema 2.0"):
        doc.merge_block("lines", [_line(style={"bold": True})], own_fields=["text", "style"])
    lax = _open(tmp_path, BUILDER, strict=False)
    lax.merge_block("pages", [{"page": "1", "page_index": 1, "needs_ocr": True}], own_fields=["page_index", "needs_ocr"])
    assert "needs_ocr" not in lax.get_block("pages")[0]


@pytest.mark.parametrize("call", ["add_derived_from", "add_regenerable", "add_license_detail"])
def test_the_run_data_writers_say_why_they_are_gone(tmp_path, call):
    doc = _open(tmp_path, "nlp-enrich")
    args = {"add_derived_from": ("teitok", "x"), "add_regenerable": ("markdown", {}), "add_license_detail": ({"x": 1},)}[call]
    with pytest.raises(RemovedInSchema2):
        getattr(doc, call)(*args)


def test_the_record_carries_no_run_data(tmp_path):
    """D1: no provenance, stamps or contributors — whatever run_id/run_uuid the caller passes."""
    doc = _open(tmp_path, BUILDER, run_id="260724-101112", run_uuid="urn:uuid:5a1e0c1e-1b2d-4c3e-8f40-0a1b2c3d4e5f", paradata_ref="p.json")
    doc.merge_block("lines", [_line()])
    text = json.dumps(doc.to_dict())
    for word in ("provenance", "assembled", "run_uuid", "run_id", "paradata_ref", "license"):
        assert word not in text, word


def test_the_source_takes_no_sha256_and_no_unknown_field(tmp_path):
    doc = _open(tmp_path, BUILDER)
    with pytest.raises(ValueError, match="sha256"):
        doc.set_source(sha256="b" * 64)
    with pytest.raises(ValueError, match="no field"):
        doc.set_source(path="/tmp/x.pdf")
    with pytest.raises(ValueError, match="origin"):
        doc.set_source(origin="ABBYY-ALTO")


def test_the_source_identity_is_first_writer_wins(tmp_path):
    doc = _open(tmp_path, BUILDER)
    doc.set_source(filename="x.alto.xml")  # a derived file's name: a NOTE, the seed's name stays
    assert doc.to_dict()["source"]["filename"] == "CTX000000001.pdf"
    with pytest.raises(ValueError, match="immutable"):
        doc.set_source(sha512="c" * 128)


@pytest.mark.parametrize(
    "path, value",
    [
        (("pages", 0, "text_layer"), "digital"),
        (("lines", 0, "group_id"), "g1"),
        (("lines", 0, "categ"), "Garbage"),
        (("content", "reading_order"), "ltr-columns"),
        (("source", "sha256"), "b" * 64),
    ],
)
def test_the_schema_refuses_what_2_0_removed(tmp_path, path, value):
    out = _built(tmp_path).to_dict()
    node = out
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    with pytest.raises(_schema_error()):
        validate_document(out)


def test_the_schema_holds_the_line_and_page_rules(tmp_path):
    out = _built(tmp_path).to_dict()
    bad = copy.deepcopy(out)
    del bad["lines"][0]["bbox_precision"]  # a box says how it was obtained
    with pytest.raises(_schema_error()):
        validate_document(bad)
    bad = copy.deepcopy(out)
    bad["lines"][0]["bbox_precision"] = "absent"  # no box then
    with pytest.raises(_schema_error()):
        validate_document(bad)
    bad = copy.deepcopy(out)
    bad["pages"][1]["quality_score"] = 0.1  # an empty page has no score (D18 rule 7)
    with pytest.raises(_schema_error()):
        validate_document(bad)
    bad = copy.deepcopy(out)
    del bad["pages"][0]["page_index"]  # every page row has its physical page number (H2)
    with pytest.raises(_schema_error()):
        validate_document(bad)


def test_an_unknown_block_of_a_newer_minor_passes_through(tmp_path):
    """Rule 6: a 2.x addition survives a tool that vendored an older 2.x schema."""
    base = _built(tmp_path).to_dict()
    base["schema_version"] = "2.1"
    base["newer_block"] = {"x": 1}
    doc = _open(tmp_path, "nlp-enrich", baseline=base)
    doc.merge_block("lines", [{"page": "1", "line": 1, "lemma": "náčrt"}])
    out = doc.to_dict()
    assert out["newer_block"] == {"x": 1}
    validate_document(out)


# ── migration from 1.x ──────────────────────────────────────────────────────


def _v1_digital():
    """A 1.x born-digital record, as digital-convert and ocr-postprocess's scoring wrote it."""
    return {
        "schema_version": "1.0",
        "record_type": "atrium-document",
        "doc_id": "CTX000000001",
        "source": {"sha512": SHA512, "sha256": "b" * 64, "filename": "x.pdf", "media_type": "application/pdf", "origin": "digital-born-pdf"},
        "derived_from": {"teitok": "TEITOK/x.teitok.xml"},
        "regenerable": {"markdown": {"from": "x"}},
        "provenance": {"license": "CC BY-NC 4.0", "license_basis": "calculated", "contributors": [{"program": DIGITAL}]},
        "assembled": {"blocks": {"lines": {"program": DIGITAL, "run_id": "r"}}},
        "pages": [
            {"page": "i", "text_layer": "digital", "needs_ocr": False, "canvas": {"width": 612, "height": 792}},
            {"page": "2", "page_index": 2, "text_layer": "garbled", "needs_ocr": True, "needs_ocr_reason": "x", "ocr": {"engine": "e"}},
        ],
        "content": {"text": "a b", "reading_order": "ltr-columns"},
        "lines": [
            {"page": "i", "line": 0, "text": "a", "bbox": [1, 2, 3, 4], "group_id": "g", "style": {"bold": True}, "categ": "Clear"},
            {"page": "2", "line": 0, "text": "b", "categ": "Garbage", "quality_score": 0.1},
        ],
        "entities": [{"surface": "a", "page": "i", "line": 0, "char_span": [0, 1]}],
        "tables": [{"table_id": "t1"}],
        "forms": {},
    }


def test_a_1x_record_migrates_to_a_valid_2_0_one():
    migrated = migrate_document(_v1_digital())
    validate_document(migrated)
    assert migrated["schema_version"] == "2.0"
    for gone in REMOVED_BLOCKS:
        assert gone not in migrated
    assert migrated["source"] == {"sha512": SHA512, "filename": "x.pdf", "media_type": "application/pdf", "origin": "digital"}
    assert migrated["content"] == {"text": "a b"}


def test_migration_renumbers_lines_from_1_and_the_entities_follow():
    migrated = migrate_document(_v1_digital())
    assert [(r["page"], r["line"]) for r in migrated["lines"]] == [("i", 1), ("2", 1)]
    assert migrated["entities"][0]["line"] == 1


def test_migration_drops_decode_verdicts_and_layout_but_keeps_text():
    first, second = migrate_document(_v1_digital())["lines"]
    assert first == {"page": "i", "line": 1, "text": "a", "bbox": [1, 2, 3, 4], "categ": "Clear", "text_source": "text-layer", "bbox_precision": "measured"}
    assert second == {"page": "2", "line": 1, "text": "b", "text_source": "text-layer"}


def test_migration_fills_the_page_index_status_and_canvas_unit():
    pages = migrate_document(_v1_digital())["pages"]
    assert pages[0] == {"page": "i", "canvas": {"width": 612, "height": 792, "unit": "px"}, "page_index": 1, "content_status": "text"}
    assert pages[1] == {"page": "2", "page_index": 2, "content_status": "text"}


@pytest.mark.parametrize(
    "origin, expected",
    [("ABBYY-ALTO", "legacy"), ("ocr:pero", "atr"), ("vlm:glm-4v", "atr"), ("docx", "digital"), ("something-new", None)],
)
def test_migration_makes_the_origin_descriptive(origin, expected):
    record = _v1_digital()
    record["source"]["origin"] = origin
    assert migrate_document(record)["source"].get("origin") == expected


def test_load_document_and_a_dict_baseline_both_migrate(tmp_path):
    path = tmp_path / "x.document.json"
    path.write_text(json.dumps(_v1_digital()), encoding="utf-8")
    assert load_document(str(path))["schema_version"] == "2.0"
    doc = _open(tmp_path, "nlp-enrich", baseline=_v1_digital())
    assert "provenance" not in doc.to_dict() and doc.to_dict()["lines"][0]["line"] == 1
    validate_baseline(_v1_digital())  # a 1.x baseline is checked as it will be used


def test_a_newer_major_is_refused(tmp_path):
    record = dict(_seed(), schema_version="3.0")
    with pytest.raises(ValueError, match="newer"):
        migrate_document(record)
    with pytest.raises(ValueError, match="newer"):
        _open(tmp_path, BUILDER, baseline=record)


def test_a_1x_seed_is_still_a_seed():
    seed = dict(_seed(), schema_version="1.0")
    validate_baseline(seed)
    assert migrate_document(seed) == _seed()


# ── fan-in ──────────────────────────────────────────────────────────────────


def test_merge_takes_blocks_in_chain_order_and_keeps_the_source(tmp_path):
    first = _built(tmp_path).to_dict()
    second = copy.deepcopy(first)
    second["content"] = {"text": "rebuilt"}
    paths = []
    for i, record in enumerate((first, second)):
        path = tmp_path / f"{i}.json"
        path.write_text(json.dumps(record), encoding="utf-8")
        paths.append(str(path))
    merged = json.loads(open(merge_document_records(paths, str(tmp_path / "m.json")), encoding="utf-8").read())
    assert merged["content"] == {"text": "rebuilt"} and merged["record_type"] == "atrium-document-merged"
    validate_document(merged)
    second["source"]["sha512"] = "c" * 128
    (tmp_path / "1.json").write_text(json.dumps(second), encoding="utf-8")
    with pytest.raises(ValueError, match="immutable"):
        merge_document_records(paths, str(tmp_path / "m2.json"))


# ── the record and its paradata ─────────────────────────────────────────────


def test_the_paradata_name_what_the_record_does_not(tmp_path):
    """The hand-off that replaces the stamps: the record's result names its blocks and fields,
    every input and output is named by distribution and SHA-512, and no licence is computed."""
    doc = _built(tmp_path)
    path = doc.finalize(str(tmp_path / "out.document.json"))
    logger = ParadataLogger(BUILDER, {}, paradata_dir=None)
    logger.add_input("atr/docproc-json", b"{}")
    logger.add_input("atr/docproc-json", sha512="1" * 128, stored=False)
    logger.add_output("atrium/document-json", path=path, record_fields=doc.written())
    logger.log_component("fasttext", "cc-by-nc-4.0", "0.9.3")
    logger.finalize()
    record = logger.record
    assert record["schema_version"] == "3.0" and record["action_status"] == "completed"
    assert {"license", "license_url", "license_detail", "limits_applied"}.isdisjoint(record)
    assert record["inputs"][1] == {"name": "atr/docproc-json", "sha512": "1" * 128, "stored": False}
    assert record["outputs"][0]["record_fields"]["content"] == ["text"]
    assert record["components"][-1] == {
        "name": "fasttext",
        "version": "0.9.3",
        "license": "CC BY-NC 4.0",
        "license_url": "https://creativecommons.org/licenses/by-nc/4.0/",
    }
    assert str(tmp_path) not in json.dumps(record["outputs"])  # a path is hashed, never recorded


def test_a_run_that_raises_is_recorded_as_failed(tmp_path):
    logger = ParadataLogger(BUILDER, {}, paradata_dir=None)
    with pytest.raises(RuntimeError):
        with logger:
            raise RuntimeError("the ATR failed")
    assert logger.record["action_status"] == "failed" and "the ATR failed" in logger.record["error"]
