"""
tests/test_document_originators.py

Issue #18 §1a — block ownership for digital-born documents.

Four blocks (`pages`, `content`, `lines`, `tables`) describe a document's POSITIONAL
PLANE, and there are two ways to acquire one: OCR/ALTO, or direct extraction from a
digital-born PDF/DOCX. They are mutually exclusive per document, so `BLOCK_OWNERS` lists
both possible originators and `source.origin` picks between them per record.

These tests pin three things that were each a live defect before this landed:

  1. `digital-convert` can actually originate the positional blocks (it could not — every
     path either raised or silently ate the payload).
  2. The two originators cannot be mixed on one document, and `merge_block()` is checked
     as well as `set_block()` (it bypassed `_assert_owner()` entirely, so `pages` and
     `lines` were never ownership-checked at all).
  3. `lines[]` rows survive the round trip with their `text` and `bbox` intact. The
     earlier `BLOCK_FIELD_OWNERS` draft granted the converter only `["group_id"]`, which
     `merge_block()` honours SILENTLY — `text` was filtered out with no warning and the
     record still validated, because `lines[]` requires only `page`+`line`.

Style follows tests/test_document_record.py (one rule per test, named in the docstring),
but uses the public `logger.run_id` rather than `logger._run_id` — the #13 hardening pass
added it precisely so tests stop reaching into privates.

(That reference used to read `tests/test_atrium_document.py`, which does not exist in this
repo. Worth more than a pointer fix: plan §1b deferred the `merge_block()` dropped-field
warning on the grounds that "at least one existing call site passes context fields it
doesn't own for readability (tests/test_atrium_document.py has the translator merging
entities with surface)". No such call site exists here — the only translator usage is
`set_block("translations", ...)` — so the stated cost of that fix was unverified. It is now
available opt-in via `warn_dropped_fields=True` and `assert_fields_survived()`, which needs
no call-site cleanup at all.)

Canonical copy (issue #10, finding D9)
--------------------------------------
This file is **hub-canonical**. It lives in atrium-project as
`docs/templates/shared/test_document_originators.py`, beside the canonical
`atrium_document.py` + `atrium_document.schema.json` it exercises, and is vendored into
every tool repo at `tests/test_document_originators.py` — exactly the arrangement
`test_para_licenses.py` uses. `para-drift.reusable.yml` `diff -u`s all six copies, so edit
the hub copy and re-vendor with `scripts/revendor_shared.sh`; never edit a vendored one.

It was promoted out of atrium-llm-enrich, which held the only copy. The hub cited this
path as what "pins" the §1a contract (`docs/document_schema.md`, `atrium_document.py`'s
`BLOCK_FIELD_OWNERS` comment) while containing no such file, and the other four repos had
no origin coverage at all — so a regression in `_assert_origin_consistent()` passed Hub
Self-Check cleanly and propagated to five repos the moment `v1` moved.

The imports below are deliberately plain top-level imports, and must stay that way: pytest
puts the test file's own directory on `sys.path`, which resolves
`atrium_document`/`atrium_paradata` both from here in `docs/templates/shared/` and once
vendored into a tool repo (where the repo root is importable via `pytest.ini`'s
`pythonpath = .` or `tests/__init__.py`). Any path arithmetic added here would resolve in
one layout and not the other.
"""

import json

import pytest

from atrium_document import (
    BLOCK_FIELD_OWNERS,
    BLOCK_KEY_FIELDS,
    BLOCK_OWNERS,
    ORIGIN_ORIGINATORS,
    DocumentRecord,
    merge_document_records,
    resolve_originator,
    validate_document,
)
from atrium_paradata import ParadataLogger

DIGITAL = "digital-convert"
ALTO = "alto-postprocess"


@pytest.fixture
def mock_paradata(tmp_path):
    """Generates a mock paradata record and returns the run_id and ref."""
    para_dir = tmp_path / "paradata"
    para_dir.mkdir()
    logger = ParadataLogger(DIGITAL, {}, paradata_dir=str(para_dir))
    logger.log_component("docling")
    paradata_ref = logger.finalize()
    return logger.run_id, paradata_ref


def _open(tmp_path, mock_paradata, program, origin=None, baseline=None, strict=True):
    """A record opened as `program`, with `source.origin` seeded when given.

    `out_dir` is pinned to tmp_path. Without it `out_dir` defaults to ".", so every test
    using this as a context manager without an explicit `finalize(path)` had __exit__ write
    CTX000000001.document.json into the REPO ROOT — an untracked file appearing after a test
    run, and a dirty tree in CI.
    """
    run_id, paradata_ref = mock_paradata
    doc = DocumentRecord(
        doc_id="CTX000000001",
        program=program,
        baseline=baseline,
        run_id=run_id,
        paradata_ref=paradata_ref,
        out_dir=str(tmp_path),
        strict=strict,
    )
    if origin is not None:
        doc.set_source(origin=origin, filename="CTX000000001.pdf")
    return doc


# ── the table itself ─────────────────────────────────────────────────────────


def test_positional_blocks_declare_two_originators():
    """The positional plane has two possible originators; everything else has one."""
    for block in ("pages", "content", "lines", "tables"):
        assert BLOCK_OWNERS[block] == (ALTO, DIGITAL), block
    for block in ("page_categories", "translations", "entities", "enrichment", "forms"):
        assert isinstance(BLOCK_OWNERS[block], str), block


def test_digital_convert_grant_includes_text_and_bbox():
    """The regression that made the first §1 landing silently lossy.

    A grant of only ["group_id"] passes every check and still produces a record with no
    text in it, because merge_block() filters unowned fields without complaining and the
    schema requires only page+line on a lines[] row.
    """
    grant = BLOCK_FIELD_OWNERS["lines"][DIGITAL]
    assert "text" in grant, "the converter must be able to originate line text"
    assert "bbox" in grant, "the PDF adapter's native coordinates have nowhere else to go"
    assert "group_id" in grant


def test_no_program_named_llm_enrich_digital_survives():
    """Renamed to `digital-convert`: these identities are roles, not repo names, and they
    are permanent in provenance.contributors[] once exported."""
    for block, owners in BLOCK_FIELD_OWNERS.items():
        assert "llm-enrich-digital" not in owners, block


def test_every_originator_is_reachable_from_some_origin():
    """A candidate in BLOCK_OWNERS with no ORIGIN_ORIGINATORS prefix could never write."""
    reachable = {originator for _prefix, originator in ORIGIN_ORIGINATORS}
    for block, owners in BLOCK_OWNERS.items():
        if isinstance(owners, tuple):
            assert set(owners) <= reachable, block


# ── set_block: content (not field-split, so the owner check is the whole story) ──


def test_digital_convert_may_originate_content_for_a_digital_born_doc(tmp_path, mock_paradata):
    out = tmp_path / "digital.document.json"
    with _open(tmp_path, mock_paradata, DIGITAL, origin="digital-born-pdf") as doc:
        doc.set_block("content", {"text": "Digital-born body text.", "reading_order": "ltr"})
        doc.finalize(str(out))

    record = json.loads(out.read_text(encoding="utf-8"))
    assert record["content"]["text"] == "Digital-born body text."
    # Rule 4: the read-time contract names the tool that actually wrote it.
    assert record["assembled"]["blocks"]["content"]["program"] == DIGITAL


def test_alto_postprocess_still_owns_content_on_the_ocr_path(tmp_path, mock_paradata):
    """Regression guard: widening authorisation must not change the OCR path at all."""
    out = tmp_path / "alto.document.json"
    with _open(tmp_path, mock_paradata, ALTO, origin="ocr:tesseract-ces") as doc:
        doc.set_block("content", {"text": "OCR'd body text."})
        doc.finalize(str(out))

    record = json.loads(out.read_text(encoding="utf-8"))
    assert record["assembled"]["blocks"]["content"]["program"] == ALTO


def test_origin_mismatch_is_refused(tmp_path, mock_paradata):
    """A digital converter must not claim the positional plane of an OCR'd document."""
    doc = _open(tmp_path, mock_paradata, DIGITAL, origin="ocr:pero")
    with pytest.raises(ValueError, match="originated by 'alto-postprocess'"):
        doc.set_block("content", {"text": "wrong originator"})


def test_origin_mismatch_the_other_way_is_also_refused(tmp_path, mock_paradata):
    doc = _open(tmp_path, mock_paradata, ALTO, origin="digital-born-pdf")
    with pytest.raises(ValueError, match="originated by 'digital-convert'"):
        doc.set_block("content", {"text": "wrong originator"})


def test_a_third_program_is_still_refused_outright(tmp_path, mock_paradata):
    """Widening to two candidates must not widen to anyone."""
    doc = _open(tmp_path, mock_paradata, "translator", origin="digital-born-pdf")
    with pytest.raises(ValueError, match="is owned by"):
        doc.set_block("content", {"text": "not yours"})


def test_unknown_origin_abstains_rather_than_blocking(tmp_path, mock_paradata):
    """Rule 6's spirit: a new origin string may land before this table learns it."""
    with _open(tmp_path, mock_paradata, DIGITAL, origin="some-future-acquisition") as doc:
        doc.set_block("content", {"text": "allowed"})
        assert doc.get_block("content")["text"] == "allowed"


def test_no_source_yet_abstains(tmp_path, mock_paradata):
    """Rule 3: a standalone run with no baseline and no source still emits its own part."""
    with _open(tmp_path, mock_paradata, DIGITAL, origin=None) as doc:
        doc.set_block("content", {"text": "standalone"})
        assert doc.get_block("content")["text"] == "standalone"


# ── merge_block: pages / lines (field-split, and previously unchecked) ────────


def test_digital_convert_lines_round_trip_keeps_text_and_bbox(tmp_path, mock_paradata):
    """The silent-drop regression, end to end.

    Every field handed in must come back. Asserting on the record rather than on the
    grant is the point: the grant is what broke, and a test that reads the grant back
    would have passed while the data was being eaten.
    """
    rows = [
        {
            "page": "1",
            "line": 0,
            "text": "První odstavec, řádek jedna.",
            "bbox": [72.0, 700.0, 300.0, 712.0],
            "group_id": "p1",
            "lang": "cs",
        },
        {
            "page": "1",
            "line": 1,
            "text": "První odstavec, řádek dvě.",
            "bbox": [72.0, 686.0, 290.0, 698.0],
            "group_id": "p1",
            "lang": "cs",
        },
        {
            "page": "1",
            "line": 2,
            "text": "Druhý odstavec.",
            "bbox": [72.0, 660.0, 250.0, 672.0],
            "group_id": "p2",
            "lang": "cs",
        },
    ]
    out = tmp_path / "lines.document.json"
    with _open(tmp_path, mock_paradata, DIGITAL, origin="digital-born-pdf") as doc:
        doc.merge_block("lines", rows)
        doc.finalize(str(out))

    written = json.loads(out.read_text(encoding="utf-8"))["lines"]
    assert len(written) == len(rows)
    for handed_in, came_back in zip(rows, written, strict=False):
        for field, value in handed_in.items():
            assert came_back.get(field) == value, f"{field} was dropped by merge_block"


def test_digital_convert_may_write_page_canvas_and_needs_ocr(tmp_path, mock_paradata):
    """`canvas` carries PyMuPDF/pdfplumber page geometry; `needs_ocr` is the digital->OCR
    handoff for Issue #10's undecodable-text-layer case."""
    out = tmp_path / "pages.document.json"
    with _open(tmp_path, mock_paradata, DIGITAL, origin="digital-born-pdf") as doc:
        doc.merge_block("lines", [{"page": "1", "line": 0, "text": "sondI"}])
        doc.merge_block(
            "pages",
            [
                {
                    "page": "1",
                    "page_index": 1,
                    "canvas": {"width": 612, "height": 792, "unit": "pt"},
                    "quality_score": 0.21,
                    "quality_band": "Trash",
                    "needs_ocr": True,
                }
            ],
        )
        doc.finalize(str(out))

    page = json.loads(out.read_text(encoding="utf-8"))["pages"][0]
    assert page["canvas"]["width"] == 612
    assert page["needs_ocr"] is True
    assert page["quality_band"] == "Trash"
    # `ocr` stays unowned by the converter: "was this OCR'd" must remain answerable.
    assert "ocr" not in page


def test_merge_block_enforces_origin_too(tmp_path, mock_paradata):
    """merge_block() never called _assert_owner(), so pages/lines skipped the check
    entirely — the half of §1a that the original write-up missed."""
    doc = _open(tmp_path, mock_paradata, DIGITAL, origin="ABBYY-ALTO")
    with pytest.raises(ValueError, match="originated by 'alto-postprocess'"):
        doc.merge_block("lines", [{"page": "1", "line": 0, "text": "x"}])


def test_nlp_enrich_merging_into_lines_is_unaffected(tmp_path, mock_paradata):
    """A field contributor is not an origination claim: nlp-enrich is not a candidate
    originator, so the origin check must abstain for it on either path."""
    baseline = tmp_path / "CTX000000001.document.json"
    baseline.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "doc_id": "CTX000000001",
                "source": {"origin": "digital-born-pdf"},
                "lines": [{"page": "1", "line": 0, "text": "Původní řádek.", "group_id": "p1"}],
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out.document.json"
    run_id, paradata_ref = mock_paradata
    with DocumentRecord.open(
        doc_id="CTX000000001",
        program="nlp-enrich",
        baseline=str(baseline),
        run_id=run_id,
        paradata_ref=paradata_ref,
        strict=True,
    ) as doc:
        doc.merge_block("lines", [{"page": "1", "line": 0, "lemma": "původní", "upos": "ADJ"}])
        doc.finalize(str(out))

    line = json.loads(out.read_text(encoding="utf-8"))["lines"][0]
    assert line["lemma"] == "původní"
    assert line["text"] == "Původní řádek."  # rule 2: co-contributor's field survives
    assert line["group_id"] == "p1"  # and so does the originator's


# ── the read-time contract ───────────────────────────────────────────────────


def test_read_time_contract_is_the_stamp_not_the_table(tmp_path, mock_paradata):
    """BLOCK_OWNERS authorises writes. Who actually wrote a block in a GIVEN record is
    assembled.blocks[<block>].program — the claim made in docs/document_schema.md, pinned
    here so it is enforced rather than merely asserted."""
    out = tmp_path / "stamped.document.json"
    with _open(tmp_path, mock_paradata, DIGITAL, origin="docx") as doc:
        doc.set_block("content", {"text": "From a DOCX."})
        doc.merge_block("lines", [{"page": "1", "line": 0, "text": "From a DOCX."}])
        doc.finalize(str(out))

    blocks = json.loads(out.read_text(encoding="utf-8"))["assembled"]["blocks"]
    assert blocks["content"]["program"] == DIGITAL
    assert blocks["lines"]["program"] == DIGITAL
    assert BLOCK_OWNERS["lines"] != DIGITAL  # the table alone would have said otherwise


def test_contributors_records_the_originating_run(tmp_path, mock_paradata):
    """Rule 4: the full picture lives in provenance.contributors[], which is what makes
    rejecting the 'no-op alto passthrough' option matter — that option would have put
    alto-postprocess in here for a run that did nothing."""
    run_id, _ = mock_paradata
    out = tmp_path / "prov.document.json"
    with _open(tmp_path, mock_paradata, DIGITAL, origin="digital-born-pdf") as doc:
        doc.set_block("content", {"text": "x"})
        doc.finalize(str(out))

    contributors = json.loads(out.read_text(encoding="utf-8"))["provenance"]["contributors"]
    mine = [c for c in contributors if c["program"] == DIGITAL]
    assert len(mine) == 1
    assert mine[0]["run_id"] == run_id
    assert "content" in mine[0]["blocks"]
    assert not any(c["program"] == ALTO for c in contributors)


# ── non-strict behaviour ─────────────────────────────────────────────────────


def test_non_strict_warns_instead_of_raising(tmp_path, mock_paradata, capsys):
    """`strict=False` stays a warning, matching every other _complain() path."""
    doc = _open(tmp_path, mock_paradata, DIGITAL, origin="ocr:pero", strict=False)
    doc.set_block("content", {"text": "written anyway"})
    err = capsys.readouterr().err
    assert "WARNING" in err
    assert "originated by 'alto-postprocess'" in err
    assert doc.get_block("content")["text"] == "written anyway"


# ── the write-order hole: blocks written before set_source() ──────────────────


def test_origin_check_is_deferred_not_skipped_when_source_comes_last(tmp_path, mock_paradata):
    """The check reads `source.origin`, so writing first used to escape it PERMANENTLY.

    Plan §2's Layer C says set_source() must be called "before any block write, since that
    is what authorizes them", but nothing enforced the order: a run that wrote its positional
    plane and only then set an OCR origin produced exactly the half-digital/half-OCR record
    §1a exists to refuse — under strict=True, with no error. set_source() being
    first-writer-wins then froze the wrong origin in.
    """
    doc = _open(tmp_path, mock_paradata, DIGITAL, origin=None)
    doc.merge_block("lines", [{"page": "1", "line": 0, "text": "digital text"}])
    with pytest.raises(ValueError, match="originated by 'alto-postprocess'"):
        doc.set_source(origin="ocr:pero")


def test_deferred_check_also_fires_at_write_time(tmp_path, mock_paradata):
    """Same hole reached the other way: an origin arriving from a baseline, not set_source()."""
    doc = _open(tmp_path, mock_paradata, DIGITAL, origin=None)
    doc.set_block("content", {"text": "digital body"})
    doc._data["source"] = {"origin": "ABBYY-ALTO"}  # e.g. a merge or a hand-edited baseline
    with pytest.raises(ValueError, match="originated by 'alto-postprocess'"):
        doc.to_dict()


def test_a_matching_late_origin_is_accepted(tmp_path, mock_paradata):
    """Deferring must not turn a legitimate write order into a failure."""
    doc = _open(tmp_path, mock_paradata, DIGITAL, origin=None)
    doc.merge_block("lines", [{"page": "1", "line": 0, "text": "ok"}])
    doc.set_source(origin="digital-born-pdf")
    assert doc.to_dict()["lines"][0]["text"] == "ok"


def test_source_conflict_is_reported_rather_than_discarded(tmp_path, mock_paradata):
    """`source` stays immutable, but a second writer disagreeing is no longer silent —
    since §1a it means the routing that picks the positional plane ran twice and disagreed."""
    doc = _open(tmp_path, mock_paradata, DIGITAL, origin="digital-born-pdf")
    with pytest.raises(ValueError, match="source is immutable"):
        doc.set_source(origin="ocr:pero")
    assert doc.to_dict()["source"]["origin"] == "digital-born-pdf"

    doc2 = _open(tmp_path, mock_paradata, DIGITAL, origin="digital-born-pdf")
    doc2.set_source(origin="digital-born-pdf")  # re-asserting the same value stays silent


# ── origin spelling ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "origin,expected",
    [
        ("digital-born-pdf", DIGITAL),
        ("digital-born-docx", DIGITAL),
        ("docx", DIGITAL),
        ("pdf", DIGITAL),  # the symmetric bare spelling; had no entry at all
        ("DOCX", DIGITAL),  # matching used to be case-sensitive
        ("Digital-Born-PDF", DIGITAL),
        ("ABBYY-ALTO", ALTO),
        ("abbyy-alto", ALTO),
        ("ocr:pero", ALTO),
        ("OCR:tesseract-ces", ALTO),
        ("vlm:glm-4v", ALTO),
        ("some-future-acquisition", None),  # abstain, per rule 6's spirit
        (None, None),
        ("", None),
    ],
)
def test_origin_spellings_resolve(origin, expected):
    """A non-match makes the check ABSTAIN, so every unmatched spelling silently switched
    §1a off for that document — the opposite of the intended behaviour, and invisible."""
    assert resolve_originator(origin) == expected


def test_unmatched_origin_is_noted_on_stderr(tmp_path, mock_paradata, capsys):
    """Abstaining is correct; abstaining in total silence is how §1a stops applying."""
    with _open(tmp_path, mock_paradata, DIGITAL, origin="some-future-acquisition") as doc:
        doc.set_block("content", {"text": "allowed"})
    err = capsys.readouterr().err
    assert "NOTE" in err and "matches no ORIGIN_ORIGINATORS prefix" in err


# ── the needs_ocr hand-off ───────────────────────────────────────────────────


def test_needs_ocr_handoff_lets_alto_re_originate(tmp_path, mock_paradata):
    """§3's "route per page before deferring to OCR" was unreachable.

    `needs_ocr` is granted to digital-convert precisely so it can say "this page's embedded
    text layer does not decode — re-acquire it by OCR". But `source.origin` is frozen at
    `digital-born-pdf`, so every pages/lines write alto-postprocess made afterwards was
    refused, and §3 contradicted §1a's "no document is ever both". The converter's own
    recorded request is the authorisation, and it is auditable from the record.
    """
    with _open(tmp_path, mock_paradata, DIGITAL, origin="digital-born-pdf") as first:
        first.merge_block(
            "pages",
            [
                {
                    "page": "1",
                    "page_index": 1,
                    "needs_ocr": True,
                    "needs_ocr_reason": "text layer decodes to corrupt diacritics (no /ToUnicode)",
                }
            ],
        )
        first.merge_block("lines", [{"page": "1", "line": 0, "text": "sondI", "categ": "Garbage"}])
        baseline = first.to_dict()

    second = DocumentRecord("CTX000000001", ALTO, baseline=baseline, out_dir=str(tmp_path), strict=True)
    second.merge_block("lines", [{"page": "1", "line": 0, "text": "sondě", "categ": "Text"}])
    assert second.get_block("lines")[0]["text"] == "sondě"
    # Rule 4: the read-time truth is the stamp, and `ocr` stays alto's alone.
    assert second.to_dict()["assembled"]["blocks"]["lines"]["program"] == ALTO
    assert second.to_dict()["source"]["origin"] == "digital-born-pdf"


def test_without_needs_ocr_alto_is_still_refused(tmp_path, mock_paradata):
    """The hand-off must be the converter's explicit request, not a general escape hatch."""
    with _open(tmp_path, mock_paradata, DIGITAL, origin="digital-born-pdf") as first:
        first.merge_block("pages", [{"page": "1", "page_index": 1, "needs_ocr": False}])
        baseline = first.to_dict()

    second = DocumentRecord("CTX000000001", ALTO, baseline=baseline, out_dir=str(tmp_path), strict=True)
    with pytest.raises(ValueError, match="originated by 'digital-convert'"):
        second.merge_block("lines", [{"page": "1", "line": 0, "text": "x"}])


# ── merge_block field discipline ─────────────────────────────────────────────


def test_own_fields_empty_writes_only_the_keys(tmp_path, mock_paradata):
    """`allowed = own_fields or ...` treated an explicit [] as "not supplied" and handed back
    the program's full grant, writing more than the caller asked for."""
    with _open(tmp_path, mock_paradata, DIGITAL, origin="digital-born-pdf") as doc:
        doc.merge_block("lines", [{"page": "1", "line": 0, "text": "not mine to write"}], own_fields=[])
        assert doc.get_block("lines") == [{"page": "1", "line": 0}]


def test_own_fields_does_not_confer_writership(tmp_path, mock_paradata):
    """merge_block() never called _assert_owner(), and the origin check abstains for
    non-candidates, so own_fields was the one way an undeclared program could write any
    block and be stamped as its author. The merge_block counterpart of
    test_a_third_program_is_still_refused_outright."""
    doc = _open(tmp_path, mock_paradata, "some-random-tool", origin="digital-born-pdf")
    with pytest.raises(ValueError, match="neither an owner nor a declared field contributor"):
        doc.merge_block("lines", [{"page": "1", "line": 0, "text": "x"}], own_fields=["text"])


def test_int_and_str_page_labels_are_one_row(tmp_path, mock_paradata):
    """Rows keyed on `json.dumps(value)` forked on TYPE: the schema types `page` as a string
    but nothing coerces it, so an originator passing "1" and a contributor passing 1 built
    two rows for one line — one with the text, one with the morphology, neither complete,
    and the record still validated."""
    with _open(tmp_path, mock_paradata, DIGITAL, origin="digital-born-pdf") as first:
        first.merge_block("lines", [{"page": 1, "line": 0, "text": "int page"}])
        baseline = first.to_dict()

    with DocumentRecord("CTX000000001", "nlp-enrich", baseline=baseline, out_dir=str(tmp_path), strict=True) as doc:
        doc.merge_block("lines", [{"page": "1", "line": 0, "lemma": "x"}])
        rows = doc.get_block("lines")

    assert len(rows) == 1, "the same physical line must not become two rows"
    assert rows[0]["text"] == "int page" and rows[0]["lemma"] == "x"


def test_assert_fields_survived_catches_the_1b_drop(tmp_path, mock_paradata):
    """The §1b round-trip assertion plan §2's Layer D is specified to run before validate().

    jsonschema cannot catch this: `lines[]` requires only page+line, so a row stripped of its
    text is a valid row.
    """
    rows = [{"page": "1", "line": 0, "text": "nlp-enrich may not originate text"}]
    doc = _open(tmp_path, mock_paradata, "nlp-enrich", origin="digital-born-pdf", strict=False)
    doc.merge_block("lines", rows)
    assert doc.dropped_fields() == {"lines": ["text"]}
    with pytest.raises(RuntimeError, match="dropped by merge_block"):
        doc.assert_fields_survived("lines", rows)


def test_assert_fields_survived_passes_for_the_declared_originator(tmp_path, mock_paradata):
    rows = [
        {
            "page": "1",
            "line": 0,
            "text": "a",
            "bbox": [1, 2, 3, 4],
            "group_id": "p1",
            "style": {"bold": True, "heading_level": 1},
            "lang": "cs",
        },
    ]
    with _open(tmp_path, mock_paradata, DIGITAL, origin="digital-born-pdf") as doc:
        doc.merge_block("lines", rows)
        doc.assert_fields_survived("lines", rows)
        assert doc.dropped_fields() == {}


def test_warn_dropped_fields_is_opt_in(tmp_path, mock_paradata, capsys):
    """Ecosystem-wide it is a tightening pass with call-site cleanup (§1b), so it is a flag."""
    run_id, ref = mock_paradata
    quiet = DocumentRecord("D", "nlp-enrich", run_id=run_id, paradata_ref=ref, out_dir=str(tmp_path))
    quiet.merge_block("lines", [{"page": "1", "line": 0, "text": "eaten"}])
    assert "may not write" not in capsys.readouterr().err

    loud = DocumentRecord(
        "D",
        "nlp-enrich",
        run_id=run_id,
        paradata_ref=ref,
        out_dir=str(tmp_path),
        warn_dropped_fields=True,
    )
    loud.merge_block("lines", [{"page": "1", "line": 0, "text": "eaten"}])
    assert "may not write ['text']" in capsys.readouterr().err


# ── tables ───────────────────────────────────────────────────────────────────


def test_tables_is_mergeable_and_keeps_its_fields(tmp_path, mock_paradata):
    """`tables` had two declared originators, no BLOCK_KEY_FIELDS entry and no
    BLOCK_FIELD_OWNERS entry: merge_block() raised "no key fields known", and with
    key_fields supplied it emptied every row down to its key — the §1b silent drop again,
    on a block the Definition of Done requires the converter to originate."""
    assert BLOCK_KEY_FIELDS["tables"] == ["table_id"]
    rows = [
        {
            "table_id": "t1",
            "page": "1",
            "n_rows": 2,
            "n_cols": 2,
            "group_id": "t1",
            "cells": [
                {"row": 0, "col": 0, "is_header": True, "group_id": "t1.r0c0"},
                {"row": 0, "col": 1, "is_header": True, "group_id": "t1.r0c1"},
                {"row": 1, "col": 0, "group_id": "t1.r1c0"},
                {"row": 1, "col": 1, "group_id": "t1.r1c1"},
            ],
        }
    ]
    with _open(tmp_path, mock_paradata, DIGITAL, origin="docx") as doc:
        doc.merge_block("tables", rows)
        doc.assert_fields_survived("tables", rows)
        written = doc.get_block("tables")

    assert len(written[0]["cells"]) == 4
    # The join the schema promises: every cell addresses the lines carrying its text.
    assert all(c["group_id"] for c in written[0]["cells"])


def test_set_block_on_tables_does_not_warn_about_co_contributors(tmp_path, mock_paradata, capsys):
    """The field-split warning exists because set_block() erases a CO-CONTRIBUTOR's fields.
    Alternative originators are mutually exclusive per document, so they are not
    co-contributors — `tables` is declared for both and nobody else."""
    with _open(tmp_path, mock_paradata, DIGITAL, origin="docx") as doc:
        doc.set_block("tables", [{"table_id": "t1", "page": "1", "n_rows": 1, "n_cols": 1}])
    err = capsys.readouterr().err
    assert "field-split" not in err


def test_lines_still_warns_on_set_block(tmp_path, mock_paradata):
    """Regression guard for the same change: `lines` has a genuine co-contributor."""
    doc = _open(tmp_path, mock_paradata, DIGITAL, origin="docx")
    with pytest.raises(ValueError, match="field-split with \\['nlp-enrich'\\]"):
        doc.set_block("lines", [{"page": "1", "line": 0, "text": "erases morphology"}])


# ── fan-in ───────────────────────────────────────────────────────────────────


def test_merge_keeps_the_first_source_and_accumulates_derived_from(tmp_path, mock_paradata):
    """`source` carries no assembled.blocks stamp, so both sides of the `updated_at`
    comparison were "" and every input overwrote the previous — last-path-wins, dropping the
    first record's sha256 and able to swap the §1a origin out from under a written plane.
    `derived_from`/`regenerable` are append-only maps and were replaced wholesale."""
    run_id, ref = mock_paradata
    first = DocumentRecord("CTX1", DIGITAL, run_id=run_id, paradata_ref=ref, out_dir=str(tmp_path))
    first.set_source(sha256="a" * 64, origin="digital-born-pdf", filename="CTX1.pdf")
    first.add_derived_from("pdf", "IN/CTX1.pdf")
    first.set_block("content", {"text": "body"})
    p1 = first.finalize(str(tmp_path / "one.json"))

    second = DocumentRecord("CTX1", "nlp-enrich", run_id=run_id, paradata_ref=ref, out_dir=str(tmp_path))
    second.set_source(origin="digital-born-pdf")
    second.add_derived_from("teitok", "TEITOK/CTX1.teitok.xml")
    second.merge_block("entities", [{"page": "1", "line": 0, "char_span": [0, 4], "surface": "body"}])
    p2 = second.finalize(str(tmp_path / "two.json"))

    merged = json.loads(open(merge_document_records([p1, p2], str(tmp_path / "m.json")), encoding="utf-8").read())
    assert merged["source"]["sha256"] == "a" * 64
    assert merged["source"]["origin"] == "digital-born-pdf"
    assert merged["derived_from"] == {"pdf": "IN/CTX1.pdf", "teitok": "TEITOK/CTX1.teitok.xml"}


def test_merge_refuses_records_that_disagree_about_source(tmp_path, mock_paradata):
    run_id, ref = mock_paradata
    a = DocumentRecord("CTX1", DIGITAL, run_id=run_id, paradata_ref=ref, out_dir=str(tmp_path))
    a.set_source(origin="digital-born-pdf")
    a.set_block("content", {"text": "x"})
    pa = a.finalize(str(tmp_path / "a.json"))

    b = DocumentRecord("CTX1", "page-classification", run_id=run_id, paradata_ref=ref, out_dir=str(tmp_path))
    b.set_source(origin="ocr:pero")
    b.set_block("page_categories", {"1": "Text"})
    pb = b.finalize(str(tmp_path / "b.json"))

    with pytest.raises(ValueError, match="disagree about the immutable `source`"):
        merge_document_records([pa, pb], str(tmp_path / "m.json"))


# ── the acceptance criterion: schema validation ───────────────────────────────


def test_a_full_digital_born_record_validates_against_the_canonical_schema(tmp_path, mock_paradata):
    """Issue #18's acceptance criterion is "the output JSON strictly passes validation
    against the canonical atrium_document.schema.json". The only test that validated
    anything validated an enrichment-only record, so `tables`, `forms`, `lines[].group_id`
    and `lines[].style` — everything #18 added — had no validation coverage at all."""
    with _open(tmp_path, mock_paradata, DIGITAL, origin="digital-born-pdf") as doc:
        doc.merge_block(
            "pages",
            [
                {
                    "page": "iv",
                    "page_index": 1,
                    "canvas": {"width": 612, "height": 792, "unit": "pt"},
                    "quality_score": 0.21,
                    "quality_band": "Trash",
                    "needs_ocr": True,
                    "needs_ocr_reason": "text layer decodes to corrupt diacritics (no /ToUnicode)",
                }
            ],
        )
        doc.merge_block(
            "lines",
            [
                {
                    "page": "iv",
                    "line": 0,
                    "text": "Zpráva o sondě",
                    "bbox": [72.0, 60.0, 300.0, 74.0],
                    "group_id": "p1",
                    "style": {"bold": True, "heading_level": 1},
                    "lang": "cs",
                    "quality_score": 0.9,
                    "categ": "Heading",
                }
            ],
        )
        doc.merge_block(
            "tables",
            [
                {
                    "table_id": "t1",
                    "page": "iv",
                    "n_rows": 1,
                    "n_cols": 1,
                    "group_id": "t1",
                    "cells": [
                        {
                            "row": 0,
                            "col": 0,
                            "is_header": True,
                            "group_id": "t1.r0c0",
                            "bbox": [72.0, 100.0, 200.0, 114.0],
                        }
                    ],
                }
            ],
        )
        doc.set_block("content", {"text": "Zpráva o sondě", "reading_order": "ltr-columns"})
        record = doc.to_dict()

    validate_document(record)  # raises on failure; no doc.json would be emitted


def test_schema_is_locatable_next_to_the_module():
    """Plan §2 makes validation the Layer D gate, but shared code had no way to FIND the
    schema: the hub keeps it in docs/templates/shared/ and tool repos at the root, and the
    only locator anywhere was a relative walk inside one test."""
    from atrium_document import SCHEMA_FILENAME, load_schema, schema_path

    assert schema_path() is not None and schema_path().endswith(SCHEMA_FILENAME)
    assert load_schema()["title"] == "ATRIUM document record"
