"""
tests/test_service_document_json.py
===================================
The service's half of accretion contract rule 1 — atrium-project#10 (J2).

`service/api.py`'s two endpoints neither accepted nor returned a `document_json` part; there
were zero hits for the string anywhere under `service/`, while `run.py` implemented the
contract fully. This file is the gate for the fix.

**Why most of it does not import `service.api`.** `service/api.py` imports
`service.inference`, which imports torch, so every existing service test starts with
`pytest.importorskip("torch")` — and torch is in no test requirements file, so those tests
have never run in the fast lane. A gate that skips is exactly the shape of gate the review
found blind to J2 and G3 in the first place. The accretion therefore lives in
`service/document_json.py`, which imports nothing heavier than pandas, and is tested here for
real. The thin HTTP layer keeps its skip, at the end of the file.

No ML models, no network, no GPU.
"""

import json
import sys
import types

import pytest

from atrium_document import canonical_doc_id, load_document, validate_document
from service.document_json import build_document_record, doc_id_for_document, doc_id_for_image

# What manager.predict() hands back: a top-N list of {label, score}, best first.
PREDS = [{"label": "TEXT", "score": 0.91}, {"label": "DRAW", "score": 0.06}]


class _MockManager:
    """Stands in for service.inference.manager — no weights, no torch."""

    device = "cpu"
    available_versions = ["v4.3"]

    def get_model_details(self, version):
        return "mocked_model"

    def predict(self, image, version, topn):
        return PREDS

    def warmup(self, versions=None):
        pass


# ── making the HTTP layer testable without torch ────────────────────────────────────────────
# service/api.py does `from .inference import manager`, and service/inference.py imports torch
# at module level. Every pre-existing service test therefore opens with
# importorskip("torch") — and torch is in no requirements-test.txt in the ecosystem, so those
# tests have never once run in the fast lane. That is not incidental to J2/G3: it is why they
# survived. Stub the ONE symbol api.py actually needs (the same technique tests/test_run.py
# already uses for atrium_document) so the endpoint contract is checked here for real.
#
# Inserted only when torch is genuinely absent, so a full-stack environment still exercises the
# real module, and left in sys.modules rather than undone per-test: service.api caches the
# binding at import time, so removing the stub afterwards would only leave a half-real module.
try:  # pragma: no cover - environment-dependent
    import torch  # noqa: F401
except ImportError:  # pragma: no cover - environment-dependent
    _stub = types.ModuleType("service.inference")
    _stub.manager = _MockManager()
    sys.modules.setdefault("service.inference", _stub)


# `source.sha256` is pattern-constrained to 64 lowercase hex chars by the schema, so a
# placeholder like "abc123" makes the baseline itself invalid and every assertion below reads
# as a Layer D warning instead of what it is testing.
SHA256 = "9" * 64


def _upstream_baseline(doc_id="CTX01"):
    """A record as alto-postprocess/nlp-enrich would hand it over: blocks pc does not own."""
    return {
        "schema_version": "1.0",
        "doc_id": doc_id,
        "source": {"sha256": SHA256, "filename": f"{doc_id}.alto.xml", "media_type": "application/xml"},
        "pages": [
            {"page": "1", "quality_score": 0.9, "quality_band": "Clear"},
            {"page": "2", "quality_score": 0.4, "quality_band": "Noisy"},
        ],
        "lines": [{"page": "1", "line": 1, "text": "Pohřebiště"}],
    }


class TestDocIdDerivation:
    """The service must land on the same record identity the CLI would — a fork here writes a
    record under a key no other stage reads (D3, the cause J2's absence was hiding)."""

    def test_image_upload_splits_the_page_off_a_multi_dot_name(self):
        assert doc_id_for_image("CTX01.scan_0007.png") == ("CTX01", "7")

    def test_image_upload_agrees_with_the_rest_of_the_pipeline(self):
        doc_id, _page = doc_id_for_image("CTX01.scan_0007.png")
        assert doc_id == canonical_doc_id("CTX01.alto.xml")

    def test_image_upload_without_a_page_label_is_page_one(self):
        assert doc_id_for_image("coverpage.png") == ("coverpage", "1")

    def test_pdf_upload_keeps_the_pdf_pages_and_does_not_split_the_filename(self):
        """A PDF's pages are its own 1..N, so a trailing number in the FILENAME is part of the
        document's name, not a page label."""
        assert doc_id_for_document("CTX01.scan.pdf") == "CTX01"
        assert doc_id_for_document("survey_2021.pdf") == "survey_2021"

    def test_missing_filename_degrades_instead_of_raising(self):
        """`DocumentRecord` refuses an empty doc_id; a nameless upload must not 500."""
        assert doc_id_for_image(None)[0]
        assert doc_id_for_document("")


class TestBuildDocumentRecord:
    def test_originates_a_record_with_no_baseline(self):
        """page-classification is stage 1 of the pipeline: the E2E passes it
        `--document-json-out` alone, so the service has to be able to START a record, not
        only accrete onto one."""
        record, schema_err = build_document_record("CTX01", [("1", PREDS)], baseline_bytes=None)

        assert schema_err is None
        assert record["doc_id"] == "CTX01"
        assert record["page_categories"] == {"1": "TEXT"}
        assert record["pages"] == [{"page": "1", "category": "TEXT", "category_confidence": 0.91}]
        assert record["assembled"]["had_baseline"] is False
        validate_document(record)

    def test_upstream_blocks_survive_the_accretion(self):
        """Rule 2: write only what you own. Everything alto-postprocess and nlp-enrich put in
        the baseline has to come back untouched — that is the entire point of the part."""
        baseline = _upstream_baseline()
        record, schema_err = build_document_record(
            "CTX01",
            [("1", PREDS), ("2", [{"label": "DRAW", "score": 0.55}])],
            baseline_bytes=json.dumps(baseline).encode("utf-8"),
        )

        assert schema_err is None
        assert record["doc_id"] == baseline["doc_id"]  # identity unchanged (D2's failure mode)
        assert record["source"] == baseline["source"]  # first writer wins
        assert record["lines"] == baseline["lines"]  # a block we do not own
        assert record["assembled"]["had_baseline"] is True

        pages = {p["page"]: p for p in record["pages"]}
        assert len(pages) == len(baseline["pages"])  # no forked rows
        assert pages["1"]["quality_band"] == "Clear" and pages["1"]["category"] == "TEXT"
        assert pages["2"]["quality_score"] == 0.4 and pages["2"]["category"] == "DRAW"
        validate_document(record)

    def test_page_count_follows_the_document_not_a_hardcoded_one(self):
        """The stub half of alto's J1 was a hardcoded single-page block. Assert the real shape:
        one row per classified page."""
        pages = [(str(n), PREDS) for n in range(1, 6)]
        record, _ = build_document_record("CTX01", pages, baseline_bytes=None)
        assert len(record["pages"]) == 5
        assert sorted(record["page_categories"]) == ["1", "2", "3", "4", "5"]

    def test_failed_prediction_page_is_dropped_rather_than_written_empty(self):
        """`manager.predict()` returns `{"error": ...}` when every model failed. A `pages[]`
        row needs only `page` to satisfy the schema, so writing one anyway would hand the next
        tool a page it believes was classified."""
        record, _ = build_document_record(
            "CTX01", [("1", PREDS), ("2", {"error": "All models failed."})], baseline_bytes=None
        )
        assert record["page_categories"] == {"1": "TEXT"}
        assert [p["page"] for p in record["pages"]] == ["1"]

    def test_no_usable_prediction_contributes_nothing(self):
        """Rule 3's spirit: nothing to contribute means emit nothing, not an empty block."""
        record, schema_err = build_document_record("CTX01", [("1", {"error": "boom"})], baseline_bytes=None)
        assert record is None and schema_err is None

    def test_invalid_baseline_is_accepted_and_reported_in_the_response(self):
        """Layer D's inherited-defect case (D4): the caller's baseline does not validate, so
        the adapter warns and emits rather than refusing. The service surfaces that as a field
        an automated caller can test instead of a line it would have to grep the log for."""
        baseline = _upstream_baseline()
        baseline["lines"] = [{"page": "1"}]  # missing the required `line`

        record, schema_err = build_document_record(
            "CTX01", [("1", PREDS)], baseline_bytes=json.dumps(baseline).encode("utf-8")
        )

        assert record is not None  # not refused
        assert record["pages"][0]["category"] == "TEXT"  # our contribution still landed
        assert schema_err and "line" in schema_err

    def test_own_invalid_output_raises_for_api_py_to_map_to_a_500(self):
        """`pages[].category_confidence` has `maximum: 1`. Layer D says never EMIT that, and
        the service must not turn the refusal into a 200 with a broken record in it."""
        with pytest.raises(RuntimeError, match="refusing to emit it"):
            build_document_record("CTX01", [("1", [{"label": "TEXT", "score": 1.5}])], baseline_bytes=None)

    def test_a_baseline_named_like_the_output_does_not_collide(self, tmp_path):
        """A client is free to upload the record under its own `<doc_id>.document.json` name;
        baseline and output must not resolve to the same temp path."""
        baseline_path = tmp_path / "CTX01.document.json"
        baseline_path.write_text(json.dumps(_upstream_baseline()), encoding="utf-8")

        record, _ = build_document_record("CTX01", [("1", PREDS)], baseline_bytes=baseline_path.read_bytes())

        assert record["lines"] == _upstream_baseline()["lines"]
        # The upload itself is untouched — nothing wrote back over the caller's file.
        assert load_document(str(baseline_path)) == _upstream_baseline()

    def test_missing_score_omits_confidence_rather_than_guessing(self):
        record, _ = build_document_record("CTX01", [("1", [{"label": "TEXT"}])], baseline_bytes=None)
        assert record["pages"][0]["category"] == "TEXT"
        assert "category_confidence" not in record["pages"][0]


# ════════════════════════════════════════════════════════════════════════════════════════════
# The HTTP layer — runs in the fast lane thanks to the service.inference stub above.
# ════════════════════════════════════════════════════════════════════════════════════════════
pytest.importorskip("fastapi")


@pytest.fixture
def client(monkeypatch):
    from fastapi.testclient import TestClient

    from service import api

    # api.py binds `manager` at import time, so the stub module alone is not enough when torch
    # IS installed and the real ModelManager got imported.
    monkeypatch.setattr(api, "manager", _MockManager())
    return TestClient(api.app)


def _png_bytes():
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color="white").save(buf, format="PNG")
    return buf.getvalue()


class TestPredictImageDocumentJson:
    def test_baseline_in_updated_record_out(self, client):
        baseline = _upstream_baseline()
        response = client.post(
            "/predict_image",
            data={"version": "v4.3", "topn": 3},
            files={
                "file": ("CTX01_0001.png", _png_bytes(), "image/png"),
                "document_json": ("CTX01.document.json", json.dumps(baseline).encode("utf-8"), "application/json"),
            },
        )
        assert response.status_code == 200
        body = response.json()
        record = body["document_json"]
        assert record["doc_id"] == "CTX01"
        assert record["lines"] == baseline["lines"]
        assert record["pages"][0]["category"] == "TEXT"
        assert body["document_json_schema_error"] is None

    def test_document_json_out_alone_originates_a_record(self, client):
        response = client.post(
            "/predict_image",
            data={"version": "v4.3", "topn": 3, "document_json_out": "true"},
            files={"file": ("CTX01_0007.png", _png_bytes(), "image/png")},
        )
        assert response.status_code == 200
        record = response.json()["document_json"]
        assert record["doc_id"] == "CTX01"
        assert record["page_categories"] == {"7": "TEXT"}

    def test_unparseable_baseline_is_422_not_a_classifier_500(self, client):
        """§4.4: unusable input is the caller's problem. The blanket
        `500 Error processing image.` would send them to debug the classifier instead."""
        response = client.post(
            "/predict_image",
            data={"version": "v4.3", "topn": 3},
            files={
                "file": ("CTX01_0001.png", _png_bytes(), "image/png"),
                "document_json": ("CTX01.document.json", b"{not json", "application/json"),
            },
        )
        assert response.status_code == 422
        assert "document_json" in response.json()["detail"]

    def test_empty_baseline_part_means_no_baseline(self, client):
        """A client that sends the field with an empty body means "none", not "zero bytes of
        JSON" — taken literally that reaches load_document() and dies."""
        response = client.post(
            "/predict_image",
            data={"version": "v4.3", "topn": 3},
            files={
                "file": ("CTX01_0001.png", _png_bytes(), "image/png"),
                "document_json": ("empty.json", b"", "application/json"),
            },
        )
        assert response.status_code == 200
        assert response.json()["document_json"]["assembled"]["had_baseline"] is False

    def test_absent_part_leaves_the_old_response_shape(self, client):
        """The part is opt-in, so this is additive on the wire — an existing client that sends
        neither field sees exactly what it saw before."""
        response = client.post(
            "/predict_image",
            data={"version": "v4.3", "topn": 3},
            files={"file": ("CTX01_0001.png", _png_bytes(), "image/png")},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["type"] == "image" and body["predictions"]
        assert body["document_json"] is None


class _FakePixmap:
    width, height = 4, 4
    samples = b"\xff" * (4 * 4 * 3)


class _FakePdfPage:
    def get_pixmap(self):
        return _FakePixmap()


class _FakePdf:
    """Minimum surface predict_document uses: len(), load_page()."""

    def __init__(self, page_count):
        self._page_count = page_count

    def __len__(self):
        return self._page_count

    def load_page(self, index):
        return _FakePdfPage()


@pytest.fixture
def fake_fitz(monkeypatch):
    """Stub PyMuPDF the same way service.inference is stubbed above.

    PyMuPDF is a real runtime dependency (it was the OTHER undeclared one this round found —
    see service/requirements.txt), but it is a ~20 MB wheel and the fast lane should not carry
    it to check the accretion wiring of `/predict_document`. Rasterising is not what is under
    test here; the per-page record is.
    """
    module = types.ModuleType("fitz")
    module.open = lambda stream=None, filetype=None: _FakePdf(3)
    monkeypatch.setitem(sys.modules, "fitz", module)
    return module


class TestPredictDocumentDocumentJson:
    def test_every_pdf_page_lands_in_the_record(self, client, fake_fitz):
        """alto's J1 wrote a hardcoded single-page block whatever the document held. Assert the
        page count follows the PDF, and that our fields land on each page."""
        response = client.post(
            "/predict_document",
            data={"version": "v4.3", "topn": 3, "document_json_out": "true"},
            files={"file": ("CTX01.scan.pdf", b"%PDF-1.4 fake", "application/pdf")},
        )
        assert response.status_code == 200
        body = response.json()
        assert len(body["pages"]) == 3

        record = body["document_json"]
        assert record["doc_id"] == "CTX01"  # no filename page-split for a whole PDF
        assert record["page_categories"] == {"1": "TEXT", "2": "TEXT", "3": "TEXT"}
        assert [p["page"] for p in record["pages"]] == ["1", "2", "3"]

    def test_upstream_blocks_survive_a_pdf_run(self, client, fake_fitz):
        baseline = _upstream_baseline()
        response = client.post(
            "/predict_document",
            data={"version": "v4.3", "topn": 3},
            files={
                "file": ("CTX01.pdf", b"%PDF-1.4 fake", "application/pdf"),
                "document_json": ("CTX01.document.json", json.dumps(baseline).encode("utf-8"), "application/json"),
            },
        )
        assert response.status_code == 200
        record = response.json()["document_json"]
        assert record["lines"] == baseline["lines"]
        assert record["source"] == baseline["source"]

    def test_absent_part_leaves_the_old_response_shape(self, client, fake_fitz):
        response = client.post(
            "/predict_document",
            data={"version": "v4.3", "topn": 3},
            files={"file": ("CTX01.pdf", b"%PDF-1.4 fake", "application/pdf")},
        )
        assert response.status_code == 200
        assert set(response.json()) == {"type", "pages"}


class TestOpenApiAdvertisesTheContract:
    """A part that is implemented but undocumented is J2 one layer down: nothing generating a
    client from the spec would ever send it."""

    @pytest.mark.parametrize("path", ["/predict_image", "/predict_document"])
    def test_both_endpoints_declare_the_parts(self, path):
        from service.api import app

        body = app.openapi()["paths"][path]["post"]["requestBody"]
        schema_ref = body["content"]["multipart/form-data"]["schema"]["$ref"].rsplit("/", 1)[-1]
        properties = app.openapi()["components"]["schemas"][schema_ref]["properties"]
        assert "document_json" in properties
        assert "document_json_out" in properties
