"""
tests/test_atrium_document_adapter.py
======================================
Real (non-mocked) unit tests for atrium_document_adapter.py.

Every other test file in this repo that touches page-classification's modules mocks
`atrium_document` out of sys.modules before import (test_run.py, test_ensamble.py,
test_parallel_best.py, test_inference.py). That's the right call for *those* files, since
they're testing unrelated logic — but it meant nothing ever exercised the adapter against
the real atrium_document.DocumentRecord API, and the mismatch (importing a class that was
never actually defined in atrium_document.py) shipped to `test` HEAD with a fully green
suite. This file imports the real module on purpose.

No ML models, no network, no GPU required — pure DataFrame + JSON I/O.
"""

import json

import pandas as pd
import pytest

from atrium_document import DocumentRecord, load_document
from atrium_document_adapter import write_document_record, write_document_records_dir


def _rdf(rows):
    return pd.DataFrame(rows)


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
