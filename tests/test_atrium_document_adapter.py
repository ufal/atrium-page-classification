"""
tests/test_atrium_document_adapter.py
======================================
Real (non-mocked) unit tests for atrium_document_adapter.py.

Four other test files in this repo used to mock `atrium_document` out of sys.modules
before import (test_run.py, test_ensamble.py, test_parallel_best.py, test_inference.py).
That masking is what let a mismatch — importing a class that was never actually defined in
atrium_document.py — ship to `test` HEAD with a fully green suite, because nothing
exercised the adapter against the real DocumentRecord API. This file imports the real
module on purpose.

The masks are gone as of issue atrium-project#10. They were vestigial: `atrium_document.py`
is vendored at the repo root and conftest.py puts that root on sys.path, so the real module
imports cleanly and the suite is byte-for-byte identical without them (362 passed, 12
skipped, 2 xfailed either way). Worse, `sys.modules` is process-global and none of the four
ever removed its stub, so whether THIS file tested the real module or a MagicMock — against
which every assertion passes vacuously — was decided by pytest's alphabetical collection
order. It happened to sort second, ahead of all four. A file named `test_a*.py`, a `-k`
filter, or xdist would have silently flipped it.

`test_the_real_module_is_under_test` below is the guard that keeps this true.

No ML models, no network, no GPU required — pure DataFrame + JSON I/O.
"""

import json

import pandas as pd
import pytest


def test_the_real_module_is_under_test():
    """The masks this file's docstring describes are gone; this is what keeps them gone.

    A MagicMock in sys.modules satisfies every assertion in this file without executing a
    line of the adapter, so the failure mode is a green suite that tests nothing. Assert on
    a real attribute rather than the module object: an import-time stub would still be a
    module, but it would not carry DocumentRecord with a real __module__.
    """
    import atrium_document

    assert type(atrium_document.DocumentRecord) is type, (
        "atrium_document is stubbed — a test module masked it in sys.modules and never "
        "restored it, so everything here passes vacuously"
    )
    assert atrium_document.DocumentRecord.__module__ == "atrium_document"

import atrium_document_adapter as adapter
from atrium_document import DocumentRecord, load_document, validate_document
from atrium_document_adapter import write_document_record, write_document_records_dir


def _rdf(rows):
    return pd.DataFrame(rows)


def _baseline(tmp_path, name, payload):
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class TestWriteDocumentRecord:
    """Single-document run: --document-json / --document-json-out are FILE paths."""

    def test_no_baseline_emits_own_part_only(self, tmp_path):
        rdf = _rdf([{"FILE": "CTX01", "PAGE": "1", "CLASS-1": "Text", "SCORE-1": 0.87}])
        out_path = tmp_path / "CTX01.document.json"

        result = write_document_record(rdf=rdf, document_json_out=str(out_path))

        assert result == str(out_path)
        data = load_document(str(out_path))
        assert data["page_categories"] == {"1": "Text"}
        assert data["pages"] == [{"page": "1", "category": "Text", "category_confidence": 0.87}]
        assert data["assembled"]["had_baseline"] is False

    def test_baseline_fields_survive_alongside_own_fields(self, tmp_path):
        baseline_path = tmp_path / "CTX01.in.json"
        baseline_path.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "doc_id": "CTX01",
                    "pages": [{"page": "1", "quality_score": 0.9, "quality_band": "Clear"}],
                }
            )
        )
        rdf = _rdf([{"FILE": "CTX01", "PAGE": "1", "CLASS-1": "Text", "SCORE-1": 0.87}])
        out_path = tmp_path / "CTX01.out.json"

        write_document_record(rdf=rdf, document_json=str(baseline_path), document_json_out=str(out_path))

        data = load_document(str(out_path))
        page = data["pages"][0]
        assert page["quality_score"] == 0.9  # alto-postprocess field preserved
        assert page["quality_band"] == "Clear"
        assert page["category"] == "Text"  # our field added
        assert page["category_confidence"] == 0.87

    def test_no_out_path_is_a_noop(self):
        rdf = _rdf([{"FILE": "CTX01", "PAGE": "1", "CLASS-1": "Text", "SCORE-1": 0.87}])
        assert write_document_record(rdf=rdf, document_json_out=None) is None

    def test_missing_score_column_omits_confidence(self, tmp_path):
        # top_N == 1 historically dropped SCORE-1; guard against that shape too.
        rdf = _rdf([{"FILE": "CTX01", "PAGE": "1", "CLASS-1": "Text"}])
        out_path = tmp_path / "CTX01.document.json"

        write_document_record(rdf=rdf, document_json_out=str(out_path))

        page = load_document(str(out_path))["pages"][0]
        assert page["category"] == "Text"
        assert "category_confidence" not in page


class TestWriteDocumentRecordsDir:
    """Batch run: --document-json-dir / --document-json-out-dir are DIRECTORIES."""

    def test_groups_by_file_into_separate_records(self, tmp_path):
        out_dir = tmp_path / "out"
        rdf = _rdf(
            [
                {"FILE": "CTX01", "PAGE": "1", "CLASS-1": "Text", "SCORE-1": 0.9},
                {"FILE": "CTX02", "PAGE": "1", "CLASS-1": "Plate", "SCORE-1": 0.7},
            ]
        )

        written = write_document_records_dir(rdf=rdf, document_json_out_dir=str(out_dir))

        assert sorted(written) == sorted([str(out_dir / "CTX01.document.json"), str(out_dir / "CTX02.document.json")])
        assert load_document(str(out_dir / "CTX01.document.json"))["page_categories"] == {"1": "Text"}
        assert load_document(str(out_dir / "CTX02.document.json"))["page_categories"] == {"1": "Plate"}

    def test_chunk_straddling_document_does_not_clobber_earlier_chunk(self, tmp_path):
        """A document whose pages land in two different chunks of the same run must end up
        with BOTH chunks' pages/categories, not just the later chunk's."""
        in_dir = tmp_path / "in"
        out_dir = tmp_path / "out"
        in_dir.mkdir()
        (in_dir / "CTX01.document.json").write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "doc_id": "CTX01",
                    "pages": [
                        {"page": "1", "quality_band": "Clear"},
                        {"page": "2", "quality_band": "Noisy"},
                    ],
                }
            )
        )

        chunk1 = _rdf([{"FILE": "CTX01", "PAGE": "1", "CLASS-1": "Text", "SCORE-1": 0.87}])
        chunk2 = _rdf([{"FILE": "CTX01", "PAGE": "2", "CLASS-1": "Plate", "SCORE-1": 0.61}])

        write_document_records_dir(rdf=chunk1, document_json_dir=str(in_dir), document_json_out_dir=str(out_dir))
        write_document_records_dir(rdf=chunk2, document_json_dir=str(in_dir), document_json_out_dir=str(out_dir))

        data = load_document(str(out_dir / "CTX01.document.json"))
        assert data["page_categories"] == {"1": "Text", "2": "Plate"}
        pages = {p["page"]: p for p in data["pages"]}
        assert pages["1"]["category"] == "Text" and pages["1"]["quality_band"] == "Clear"
        assert pages["2"]["category"] == "Plate" and pages["2"]["quality_band"] == "Noisy"

    def test_no_out_dir_is_a_noop(self):
        rdf = _rdf([{"FILE": "CTX01", "PAGE": "1", "CLASS-1": "Text", "SCORE-1": 0.9}])
        assert write_document_records_dir(rdf=rdf, document_json_out_dir=None) == []


class TestOwnershipGuard:
    """Sanity check that the adapter only ever writes through the declared block/field
    ownership — a regression here would mean it silently started co-mutating a block owned
    by another tool."""

    def test_strict_mode_rejects_writing_an_unowned_block(self, tmp_path):
        doc = DocumentRecord.open("CTX01", "page-classification", baseline=None, strict=True)
        with pytest.raises(ValueError):
            doc.set_block("enrichment", {"items": []})  # owned by llm-enrich, not us


class TestLayerDValidationGate:
    """atrium-project#10 (D4): `validate_document()` had zero production call sites in any of
    the five repos, so the "no doc.json is emitted if validation fails" gate that
    docs/document_schema.md calls normative was protecting nothing.

    These tests assert the ecosystem-wide POLICY, not merely that a call exists:
    hard-fail on this tool's own output, warn on an inherited baseline.
    """

    @pytest.fixture(autouse=True)
    def _reset_gate_warning(self, monkeypatch):
        """`_schema_gate_disabled_warned` is a module global that latches after the first
        degraded-gate warning, so leaving it set would make a later test's assertion depend on
        execution order."""
        monkeypatch.setattr(adapter, "_schema_gate_disabled_warned", False)

    def test_valid_output_passes_the_real_schema(self, tmp_path):
        """The gate is only meaningful if normal output actually clears it — otherwise every
        run would be taking the warn-or-raise branch."""
        rdf = _rdf([{"FILE": "CTX01", "PAGE": "1", "CLASS-1": "Text", "SCORE-1": 0.87}])
        out_path = tmp_path / "CTX01.document.json"

        write_document_record(rdf=rdf, document_json_out=str(out_path))

        validate_document(load_document(str(out_path)))  # raises if the record is invalid

    def test_own_invalid_output_raises_and_emits_nothing(self, tmp_path):
        """A confidence above 1.0 — `pages[].category_confidence` has `maximum: 1` — is the
        shape a bad averaging change would produce. Layer D says never EMIT that: the raise
        has to happen before the write, or the next tool loads a record we knew was broken."""
        rdf = _rdf([{"FILE": "CTX01", "PAGE": "1", "CLASS-1": "Text", "SCORE-1": 1.5}])
        out_path = tmp_path / "CTX01.document.json"

        with pytest.raises(RuntimeError, match="refusing to emit it"):
            write_document_record(rdf=rdf, document_json_out=str(out_path))

        assert not out_path.exists()
        # And no half-written temp file left for the next run's baseline resolution to find.
        assert list(tmp_path.iterdir()) == []

    def test_invalid_baseline_warns_and_still_produces_a_record(self, tmp_path, capsys):
        """An upstream tool wrote a `lines[]` row with no `line` (the block requires
        `page`+`line`). Refusing to run would turn one bad record into a stalled pipeline, and
        rule 6 already commits to passing unknown content through — so this warns and continues.

        It also demotes the second half: `lines` is alto-postprocess's block, nothing here
        touches it, so the defect passes straight into this run's output. Failing on somebody
        else's record would contradict the decision just taken, so the own-output check warns
        too. Both warnings are asserted because the pair IS the policy.
        """
        baseline = _baseline(
            tmp_path,
            "CTX01.in.json",
            {
                "schema_version": "1.0",
                "doc_id": "CTX01",
                "pages": [{"page": "1", "quality_band": "Clear"}],
                "lines": [{"page": "1"}],  # missing the required `line`
            },
        )
        rdf = _rdf([{"FILE": "CTX01", "PAGE": "1", "CLASS-1": "Text", "SCORE-1": 0.87}])
        out_path = tmp_path / "CTX01.out.json"

        write_document_record(rdf=rdf, document_json=str(baseline), document_json_out=str(out_path))

        assert out_path.exists()
        data = load_document(str(out_path))
        assert data["pages"][0]["category"] == "Text"  # our contribution landed
        assert data["lines"] == [{"page": "1"}]  # the invalid block passed through untouched

        err = capsys.readouterr().err
        assert "inherited baseline CTX01.in.json does not validate" in err
        assert "emitting it anyway because the baseline was already invalid" in err

    def test_own_merge_repairs_a_baseline_defect_in_its_own_field_scope(self, tmp_path, capsys):
        """Companion to the above, and the reason it uses `lines` rather than `pages`: a
        baseline whose `pages[].page` is an int is invalid, but this tool's merge rewrites that
        very key as a string, so the emitted record is valid again. The baseline warning still
        fires (the upstream record WAS invalid, and that is worth saying) while the own-output
        check finds nothing to complain about."""
        baseline = _baseline(
            tmp_path,
            "CTX01.in.json",
            {"schema_version": "1.0", "doc_id": "CTX01", "pages": [{"page": 1}]},
        )
        rdf = _rdf([{"FILE": "CTX01", "PAGE": "1", "CLASS-1": "Text", "SCORE-1": 0.87}])
        out_path = tmp_path / "CTX01.out.json"

        write_document_record(rdf=rdf, document_json=str(baseline), document_json_out=str(out_path))

        validate_document(load_document(str(out_path)))
        err = capsys.readouterr().err
        assert "inherited baseline CTX01.in.json does not validate" in err
        assert "emitting it anyway" not in err

    def test_valid_baseline_does_not_demote_the_own_output_check(self, tmp_path):
        """The demotion must be conditional. With a VALID baseline, an invalid own output is
        entirely ours and still has to raise — otherwise the first half of the policy would
        quietly disable the second."""
        baseline = _baseline(
            tmp_path,
            "CTX01.in.json",
            {"schema_version": "1.0", "doc_id": "CTX01", "pages": [{"page": "1", "quality_band": "Clear"}]},
        )
        rdf = _rdf([{"FILE": "CTX01", "PAGE": "1", "CLASS-1": "Text", "SCORE-1": 1.5}])
        out_path = tmp_path / "CTX01.out.json"

        with pytest.raises(RuntimeError, match="refusing to emit it"):
            write_document_record(rdf=rdf, document_json=str(baseline), document_json_out=str(out_path))
        assert not out_path.exists()

    def test_missing_jsonschema_degrades_loudly_exactly_once(self, tmp_path, monkeypatch, capsys):
        """`validate_document()` raises RuntimeError when `jsonschema` is absent. That means
        the GATE is missing, not that the record is bad, so it must not be mistaken for a
        validation failure — but it must also not pass in silence, because a no-op gate is
        indistinguishable in the output from a gate that passed. One loud line, then carry on.

        `jsonschema` is declared in setup/requirements.txt and setup/requirements-test.txt, so
        this path should never be taken in a supported install; it is here because D4's own
        description of the risk is that the call "would raise its own fail-loud RuntimeError".
        """

        def _no_jsonschema(record):
            raise RuntimeError("jsonschema is not installed, so the record cannot be validated.")

        monkeypatch.setattr(adapter, "validate_document", _no_jsonschema)

        # An output that WOULD be rejected, to prove the degraded gate lets it through rather
        # than silently swallowing a real validation error.
        rdf = _rdf(
            [
                {"FILE": "CTX01", "PAGE": "1", "CLASS-1": "Text", "SCORE-1": 1.5},
                {"FILE": "CTX02", "PAGE": "1", "CLASS-1": "Plate", "SCORE-1": 1.5},
            ]
        )
        out_dir = tmp_path / "out"

        written = write_document_records_dir(rdf=rdf, document_json_out_dir=str(out_dir))

        assert len(written) == 2
        err = capsys.readouterr().err
        assert "schema validation is DISABLED" in err
        assert "jsonschema is not installed" in err
        # Two documents, one warning — a 400-page batch must not bury the line under 400 copies.
        assert err.count("schema validation is DISABLED") == 1


class TestFieldSurvivalAssertion:
    """atrium-project#10 (D8): `assert_fields_survived()` — the §1b round-trip inspection added
    on 08-03 precisely so a silent field drop becomes visible — was called from no repo.

    `jsonschema` cannot substitute for it: `pages[]` requires only `page`, so a row stripped of
    its `category` is a *valid* row, and Layer D would wave it through.
    """

    def test_a_field_outside_the_grant_raises_instead_of_vanishing(self, tmp_path, monkeypatch):
        """Simulates the realistic regression: `_page_patches()` starts emitting a field the
        grant does not cover (here `quality_band`, which belongs to alto-postprocess). Without
        the assertion, merge_block() drops it in silence and the record still validates."""
        real_page_patches = adapter._page_patches

        def _leaky_page_patches(group):
            page_categories, patches = real_page_patches(group)
            for patch in patches:
                patch["quality_band"] = "Clear"  # alto-postprocess's field, not ours
            return page_categories, patches

        monkeypatch.setattr(adapter, "_page_patches", _leaky_page_patches)

        rdf = _rdf([{"FILE": "CTX01", "PAGE": "1", "CLASS-1": "Text", "SCORE-1": 0.87}])
        out_path = tmp_path / "CTX01.document.json"

        with pytest.raises(RuntimeError, match="quality_band"):
            write_document_record(rdf=rdf, document_json_out=str(out_path))

        assert not out_path.exists()

    def test_the_declared_grant_is_a_subset_of_the_hub_table(self):
        """The other direction: OWN_PAGE_FIELDS is passed as `own_fields`, which OVERRIDES
        BLOCK_FIELD_OWNERS in merge_block() — so nothing at runtime would notice if the two
        drifted apart. Pin it here instead."""
        from atrium_document import BLOCK_FIELD_OWNERS

        declared = set(BLOCK_FIELD_OWNERS["pages"]["page-classification"])
        assert set(adapter.OWN_PAGE_FIELDS) <= declared

    def test_owned_fields_survive_a_merge_onto_a_co_contributors_row(self, tmp_path):
        """The assertion's happy path, exercised through the real merge: a co-contributor
        already holds this page row, and both of our fields still land on it."""
        baseline = _baseline(
            tmp_path,
            "CTX01.in.json",
            {
                "schema_version": "1.0",
                "doc_id": "CTX01",
                "pages": [{"page": "1", "quality_score": 0.9, "quality_band": "Clear"}],
            },
        )
        rdf = _rdf([{"FILE": "CTX01", "PAGE": "1", "CLASS-1": "Text", "SCORE-1": 0.87}])
        out_path = tmp_path / "CTX01.out.json"

        write_document_record(rdf=rdf, document_json=str(baseline), document_json_out=str(out_path))

        page = load_document(str(out_path))["pages"][0]
        for field in adapter.OWN_PAGE_FIELDS:
            assert field in page
