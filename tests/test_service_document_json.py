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

import io
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

#: (atrium-project#68) An AMČR seed is keyed by the AMČR file id, never by the upload's name.
SEED = "C-202000543A-DT-27"


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
        document's name, not a page label.

        `.pdf` is a KNOWN_PIPELINE_SUFFIXES entry since atrium-alto-postprocess#31 (the text-lines
        inputs), so a dotted PDF name keeps its inner dots — `CTX01.scan.pdf` is `CTX01.scan`, as
        every other tool keys it — where the first-dot fallback used to answer `CTX01`."""
        assert doc_id_for_document("CTX01.scan.pdf") == "CTX01.scan"
        assert doc_id_for_document("CTX01.pdf") == "CTX01"
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

    def test_a_seed_keyed_unlike_the_upload_comes_back_with_our_block(self, client):
        """(atrium-project#68) The record keeps the seed's doc_id and carries this tool's block.
        alto-postprocess and llm-enrich returned the untouched seed here; this service writes to
        an explicit path, and the test keeps it that way."""
        baseline = _upstream_baseline(SEED)
        response = client.post(
            "/predict_image",
            data={"version": "v4.3", "topn": 3},
            files={
                "file": ("scan_0001.png", _png_bytes(), "image/png"),
                "document_json": ("seed.document.json", json.dumps(baseline).encode("utf-8"), "application/json"),
            },
        )
        assert response.status_code == 200
        record = response.json()["document_json"]
        assert record["doc_id"] == SEED
        assert record["page_categories"] == {"1": "TEXT"}
        assert record["pages"][0]["category"] == "TEXT"
        assert record["pages"][0]["quality_score"] == 0.9  # the seed's own field on the same row
        assert record["lines"] == baseline["lines"]

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


class _FakeBitmap:
    """What `page.render()` returns: `to_pil()` and `close()`."""

    def to_pil(self):
        from PIL import Image

        return Image.new("RGB", (4, 4), color="white")

    def close(self):
        _FakePdfPage.events.append("bitmap.close")


class _FakePdfPage:
    #: Every `scale=` the service rendered at, across all fake pages.
    scale_requests: list = []
    #: Every page, bitmap and document closed, in order — PDFium memory the service must free.
    events: list = []
    #: An A4 page in PDF points (1/72 inch), as PDFium's `get_size()` reports it.
    size = (595.0, 842.0)

    def get_size(self):
        return self.size

    def render(self, scale=None):
        _FakePdfPage.scale_requests.append(scale)
        return _FakeBitmap()

    def close(self):
        _FakePdfPage.events.append("page.close")


class _FakePdf:
    """The surface predict_document uses: len(), [index], init_forms(), close()."""

    def __init__(self, page_count):
        self._page_count = page_count

    def __len__(self):
        return self._page_count

    def __getitem__(self, index):
        return _FakePdfPage()

    def init_forms(self):
        pass

    def close(self):
        _FakePdfPage.events.append("pdf.close")


#: How many pages the fake PDF has (a test may monkeypatch it).
fake_pdf_pages = 3


@pytest.fixture
def fake_pdfium(monkeypatch):
    """Stub pypdfium2 the same way service.inference is stubbed above.

    Rasterising is not what is under test here; the per-page record, the limits and the use
    of the engine (scale, closing, the lock) are. TestRealPdfium drives the real engine.
    """
    module = types.ModuleType("pypdfium2")

    class PdfiumError(RuntimeError):
        pass

    module.PdfiumError = PdfiumError
    module.PdfDocument = lambda content: _FakePdf(sys.modules[__name__].fake_pdf_pages)
    monkeypatch.setitem(sys.modules, "pypdfium2", module)
    monkeypatch.setattr(_FakePdfPage, "scale_requests", [])
    monkeypatch.setattr(_FakePdfPage, "events", [])
    return module


class TestPredictDocumentDocumentJson:
    def test_every_pdf_page_lands_in_the_record(self, client, fake_pdfium):
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
        # no filename page-split for a whole PDF; `.pdf` is a known suffix, so the inner dot stays
        assert record["doc_id"] == "CTX01.scan"
        assert record["page_categories"] == {"1": "TEXT", "2": "TEXT", "3": "TEXT"}
        assert [p["page"] for p in record["pages"]] == ["1", "2", "3"]

    def test_a_seed_keyed_unlike_the_upload_comes_back_with_our_block(self, client, fake_pdfium):
        """(atrium-project#68) The PDF endpoint, same guarantee as the image one."""
        baseline = _upstream_baseline(SEED)
        response = client.post(
            "/predict_document",
            data={"version": "v4.3", "topn": 3},
            files={
                "file": ("scan.pdf", b"%PDF-1.4 fake", "application/pdf"),
                "document_json": ("seed.document.json", json.dumps(baseline).encode("utf-8"), "application/json"),
            },
        )
        assert response.status_code == 200
        record = response.json()["document_json"]
        assert record["doc_id"] == SEED
        assert record["page_categories"] == {"1": "TEXT", "2": "TEXT", "3": "TEXT"}
        assert record["lines"] == baseline["lines"]

    def test_upstream_blocks_survive_a_pdf_run(self, client, fake_pdfium):
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

    def test_absent_part_leaves_the_old_response_shape(self, client, fake_pdfium):
        response = client.post(
            "/predict_document",
            data={"version": "v4.3", "topn": 3},
            files={"file": ("CTX01.pdf", b"%PDF-1.4 fake", "application/pdf")},
        )
        assert response.status_code == 200
        # limits_applied (atrium-project#53) and the call's CreateAction, paradata
        # (atrium-project#71 R2), are in every response; document_json is not.
        assert set(response.json()) == {"type", "pages", "limits_applied", "paradata"}

    def test_pages_are_rasterised_at_the_training_resolution(self, client, fake_pdfium):
        """PDFium renders at 72 dpi at scale 1; the training pages were made by pdf2png.sh at
        300. Every page must be rendered at PDF_RENDER_DPI (scale = dpi / 72), and its DEFAULT
        must stay at 300 — it is an environment setting since atrium-project#53, so a
        deployment may choose otherwise."""
        from tool_limits import PDF_RENDER_DPI

        response = client.post(
            "/predict_document",
            data={"version": "v4.3", "topn": 3},
            files={"file": ("CTX01.pdf", b"%PDF-1.4 fake", "application/pdf")},
        )
        assert response.status_code == 200
        assert PDF_RENDER_DPI.default == 300
        assert _FakePdfPage.scale_requests == [300 / 72] * 3

    def test_every_bitmap_page_and_the_document_are_closed(self, client, fake_pdfium):
        """PDFium's memory is not Python's: a page, a bitmap or a document left open is held
        until the garbage collector finds it, in a service that renders gigapixels a day."""
        response = client.post(
            "/predict_document",
            data={"version": "v4.3", "topn": 3},
            files={"file": ("CTX01.pdf", b"%PDF-1.4 fake", "application/pdf")},
        )
        assert response.status_code == 200
        assert _FakePdfPage.events == ["bitmap.close", "page.close"] * 3 + ["pdf.close"]

    def test_a_refused_page_still_closes_the_page_and_the_document(self, client, fake_pdfium, monkeypatch):
        monkeypatch.setenv("MAX_IMAGE_PIXELS", "1000")
        response = client.post(
            "/predict_document",
            data={"version": "v4.3", "topn": 3},
            files={"file": ("CTX01.pdf", b"%PDF-1.4 fake", "application/pdf")},
        )
        assert response.status_code == 413
        assert _FakePdfPage.scale_requests == [], "refused before rendering"
        assert _FakePdfPage.events == ["page.close", "pdf.close"]

    def test_pdfium_is_never_entered_from_two_threads_at_once(self, client, fake_pdfium, monkeypatch):
        """PDFium is not thread-safe, and /predict_document renders in a worker thread (issue
        #55): without `_PDFIUM_LOCK`, two requests render at the same time. Four documents of
        three pages each, rendered from four threads, must never overlap inside the engine."""
        import threading
        import time

        from service import api

        guard, inside, depths = threading.Lock(), [0], []
        original = _FakePdfPage.render

        def slow_render(page, scale=None):
            with guard:
                inside[0] += 1
                depths.append(inside[0])
            time.sleep(0.01)
            with guard:
                inside[0] -= 1
            return original(page, scale=scale)

        monkeypatch.setattr(_FakePdfPage, "render", slow_render)
        threads = [threading.Thread(target=api._classify_pdf_pages, args=(b"%PDF", "v4.3", 1)) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert len(depths) == 12, "every page of every document was rendered"
        assert max(depths) == 1, f"PDFium was entered by {max(depths)} threads at once"


class TestRealPdfium:
    """The real engine, end to end (pypdfium2 is in setup/requirements-test.txt): a PDF made
    here goes through PDFium and reaches the model as the page image the classifier expects —
    RGB, at PDF_RENDER_DPI, with its colours where they were (PDFium renders BGR; a channel
    swap would hand the model a different page)."""

    #: (R, G, B) per page: asymmetric, so a red/blue swap cannot pass.
    COLOURS = [(200, 120, 40), (40, 160, 220)]

    def _pdf(self, size=(612, 792)) -> bytes:
        from PIL import Image

        pages = [Image.new("RGB", size, color=colour) for colour in self.COLOURS]
        buf = io.BytesIO()
        # resolution=72: one pixel per PDF point, so the page is `size` points.
        pages[0].save(buf, format="PDF", resolution=72, save_all=True, append_images=pages[1:])
        return buf.getvalue()

    def _client(self, monkeypatch, seen):
        from fastapi.testclient import TestClient

        from service import api

        class Recording(_MockManager):
            def predict(self, image, version, topn):
                seen.append(image)
                return PREDS

        monkeypatch.setattr(api, "manager", Recording())
        return TestClient(api.app)

    def test_pages_reach_the_model_as_rgb_images_at_the_render_dpi(self, monkeypatch):
        pytest.importorskip("pypdfium2")
        seen = []
        client = self._client(monkeypatch, seen)
        monkeypatch.setenv("PDF_RENDER_DPI", "144")  # scale 2.0: exact in floating point
        response = client.post(
            "/predict_document",
            data={"version": "v4.3", "topn": 3, "document_json_out": "true"},
            files={"file": ("CTX01.pdf", self._pdf(), "application/pdf")},
        )
        assert response.status_code == 200, response.text
        assert [page["page"] for page in response.json()["pages"]] == [1, 2]
        assert response.json()["document_json"]["page_categories"] == {"1": "TEXT", "2": "TEXT"}
        assert [(image.mode, image.size) for image in seen] == [("RGB", (1224, 1584))] * 2
        for image, colour in zip(seen, self.COLOURS, strict=True):
            centre = image.getpixel((612, 792))
            assert all(abs(a - b) <= 4 for a, b in zip(centre, colour, strict=True)), (centre, colour)

    def test_bytes_that_are_not_a_pdf_are_422_with_pdfiums_reason(self, monkeypatch):
        pytest.importorskip("pypdfium2")
        client = self._client(monkeypatch, [])
        response = client.post(
            "/predict_document",
            data={"version": "v4.3", "topn": 3},
            files={"file": ("d.pdf", b"not a pdf", "application/pdf")},
        )
        assert response.status_code == 422
        assert response.json()["detail"].startswith("The upload is not a readable PDF: Failed to load document")

    def test_the_size_checked_before_rendering_is_the_size_rendered(self):
        """MAX_IMAGE_PIXELS is checked on `pixel_size()` before PDFium renders, so it must be the
        size PDFium then produces — including its rounding. A 522 x 737 px scan stored at 300 dpi
        is 125.28 x 176.88 pt, held by PDFium as 32-bit floats (176.8800048828125 pt), which is
        737.00002 px at 300 dpi: PDFium rounds it up to 738, and so must the check."""
        pytest.importorskip("pypdfium2")
        from PIL import Image

        from service.pdf_render import open_document, pixel_size, render_page, render_scale

        buf = io.BytesIO()
        Image.new("RGB", (522, 737), color=(250, 250, 245)).save(buf, format="PDF", resolution=300)
        pdf = open_document(buf.getvalue())
        try:
            predicted = pixel_size(pdf[0], render_scale(300))
            image = render_page(pdf, 0, render_scale(300))
        finally:
            pdf.close()
        assert image.mode == "RGB"
        assert image.size == predicted == (522, 738)

    def test_a_page_over_max_image_pixels_is_refused_from_its_declared_size(self, monkeypatch):
        """A 612 x 792 pt page at 300 dpi is 2550 x 3300 = 8.4 Mpx: refused at 8 Mpx, before
        PDFium renders a pixel of it."""
        pytest.importorskip("pypdfium2")
        seen = []
        client = self._client(monkeypatch, seen)
        monkeypatch.setenv("MAX_IMAGE_PIXELS", "8000000")
        response = client.post(
            "/predict_document",
            data={"version": "v4.3", "topn": 3},
            files={"file": ("CTX01.pdf", self._pdf(), "application/pdf")},
        )
        assert response.status_code == 413 and response.json()["limit"]["key"] == "max_image_pixels"
        assert seen == []


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


class TestLimits:
    """atrium-project#53 (factor III): every limit is a setting, reported in /info, and an input
    over one is refused with the harmonised error — never cut quietly."""

    def test_info_reports_every_limit_with_the_variable_that_sets_it(self, client):
        info = client.get("/info").json()
        assert info["limits"] == {
            "max_upload_mb": 10.0,
            "max_pdf_pages": 50,
            "pdf_render_dpi": 300,
            "max_image_pixels": 178956970,
        }
        assert {k: m["env"] for k, m in info["limits_meta"].items()} == {
            "max_upload_mb": "MAX_UPLOAD_MB",
            "max_pdf_pages": "MAX_PDF_PAGES",
            "pdf_render_dpi": "PDF_RENDER_DPI",
            "max_image_pixels": "MAX_IMAGE_PIXELS",
        }

    def test_a_pdf_over_max_pdf_pages_is_refused(self, client, fake_pdfium, monkeypatch):
        monkeypatch.setenv("MAX_PDF_PAGES", "2")
        response = client.post(
            "/predict_document",
            data={"version": "v4.3", "topn": 3},
            files={"file": ("CTX01.pdf", b"%PDF-1.4 fake", "application/pdf")},
        )
        assert response.status_code == 413
        body = response.json()
        assert body["reason"] == "limit_exceeded"
        assert body["detail"] == "PDF has too many pages: 3. Limit is 2 (MAX_PDF_PAGES)."
        assert body["limit"] == {
            "key": "max_pdf_pages",
            "env": "MAX_PDF_PAGES",
            "value": 2,
            "observed": 3,
            "unit": "pages",
        }
        assert _FakePdfPage.scale_requests == [], "refused before any page was rendered"
        assert _FakePdfPage.events == ["pdf.close"]

    def test_a_page_too_large_to_render_is_refused_before_rendering(self, client, fake_pdfium, monkeypatch):
        monkeypatch.setenv("MAX_IMAGE_PIXELS", "1000000")  # A4 at 300 dpi is ~8.7 Mpx
        response = client.post(
            "/predict_document",
            data={"version": "v4.3", "topn": 3},
            files={"file": ("CTX01.pdf", b"%PDF-1.4 fake", "application/pdf")},
        )
        assert response.status_code == 413
        assert response.json()["limit"]["key"] == "max_image_pixels"
        # PDFium's own rounding (ceil of points x scale): 595 x 842 pt at 300 dpi.
        assert response.json()["detail"].startswith("Page 1 at 300 dpi is 2480 x 3509")
        assert _FakePdfPage.scale_requests == []

    def test_pdf_render_dpi_is_a_setting(self, client, fake_pdfium, monkeypatch):
        monkeypatch.setenv("PDF_RENDER_DPI", "150")
        response = client.post(
            "/predict_document",
            data={"version": "v4.3", "topn": 3},
            files={"file": ("CTX01.pdf", b"%PDF-1.4 fake", "application/pdf")},
        )
        assert response.status_code == 200
        assert _FakePdfPage.scale_requests == [150 / 72] * 3
        assert client.get("/info").json()["limits_meta"]["pdf_render_dpi"]["source"] == "env"

    def test_an_image_over_max_image_pixels_is_refused_before_decoding(self, client, monkeypatch):
        monkeypatch.setenv("MAX_IMAGE_PIXELS", "63")  # the test PNG is 8 x 8 = 64 px
        response = client.post(
            "/predict_image",
            data={"version": "v4.3", "topn": 3},
            files={"file": ("CTX01_0001.png", _png_bytes(), "image/png")},
        )
        assert response.status_code == 413
        body = response.json()
        assert body["reason"] == "limit_exceeded" and body["limit"]["observed"] == 64

    def test_an_oversized_document_json_part_is_refused(self, client, monkeypatch):
        monkeypatch.setenv("MAX_UPLOAD_MB", "0.001")
        response = client.post(
            "/predict_image",
            data={"version": "v4.3", "topn": 3},
            files={
                "file": ("CTX01_0001.png", _png_bytes(), "image/png"),
                "document_json": ("b.json", b"{" + b" " * 4096 + b"}", "application/json"),
            },
        )
        assert response.status_code == 413
        assert response.json()["detail"] == "document_json too large: over 0.001 MB (MAX_UPLOAD_MB)."

    @pytest.mark.parametrize("topn", [0, 12])
    def test_topn_out_of_range_is_a_validation_error_not_a_silent_cap(self, client, topn):
        response = client.post(
            "/predict_image",
            data={"version": "all", "topn": topn},
            files={"file": ("CTX01_0001.png", _png_bytes(), "image/png")},
        )
        assert response.status_code == 422
        assert response.json()["reason"] is None and "topn" in response.json()["detail"]

    def test_every_response_carries_limits_applied(self, client):
        response = client.post(
            "/predict_image",
            data={"version": "v4.3", "topn": 3},
            files={"file": ("CTX01_0001.png", _png_bytes(), "image/png")},
        )
        assert response.status_code == 200 and response.json()["limits_applied"] == []

    def test_a_malformed_limit_fails_at_declaration(self, monkeypatch):
        import atrium_limits

        monkeypatch.setenv("MAX_PDF_PAGES", "fifty")
        with pytest.raises(atrium_limits.LimitConfigError, match="MAX_PDF_PAGES"):
            atrium_limits.limit("MAX_PDF_PAGES", 50, unit="pages")


# ── the `pages` selection and the record's own page labels (atrium-digital-convert#2) ─────────
# digital-convert's `/describe` asks this service about the pages whose text layer is unusable,
# sending the PDF and the born-digital record it made. That record names its pages by their PDF
# page labels (`i`, `ii`, `1`, …) with the position in `pages[].page_index`; categories written
# under the positions "1", "2", … would land on the wrong rows.

from service.document_json import (  # noqa: E402
    baseline_page_labels,
    expand_page_selection,
    parse_page_selection,
    record_page_keys,
)


def _born_digital_baseline(doc_id="C-202000543A-DT-27"):
    """A born-digital record as digital-convert writes it: labelled pages with `page_index`.

    Built through `DocumentRecord` as the `digital-convert` originator, so it is the shape (and
    the validity) of a real one rather than a hand-written approximation.
    """
    from atrium_document import DocumentRecord

    canvas = {"width": 595.0, "height": 842.0, "unit": "pt"}
    record = DocumentRecord(doc_id, "digital-convert")
    record.set_source(SHA256, filename="report.pdf", media_type="application/pdf", origin="digital-born-pdf")
    record.merge_block(
        "pages",
        [
            {"page": "i", "page_index": 1, "canvas": canvas},
            {"page": "ii", "page_index": 2, "canvas": canvas},
            {"page": "1", "page_index": 3, "canvas": canvas, "needs_ocr": True, "needs_ocr_reason": "no text layer"},
        ],
        key_fields=["page"],
    )
    record.merge_block("lines", [{"page": "i", "line": 0, "text": "Zpráva o výzkumu"}], key_fields=["page", "line"])
    data = record.to_dict()
    validate_document(data)
    return data


class TestPageSelection:
    @pytest.mark.parametrize(
        "spec, ranges",
        [
            (None, None),
            ("", None),
            ("  ", None),
            ("3", [(3, 3)]),
            ("1,3,5-7", [(1, 1), (3, 3), (5, 7)]),
            (" 2 - 4 , 9 ", [(2, 4), (9, 9)]),
        ],
    )
    def test_the_field_parses_into_ranges(self, spec, ranges):
        assert parse_page_selection(spec) == ranges

    @pytest.mark.parametrize("spec", ["a", "1,,2", "0", "3-1", "1-", "-2", "1;2", "1.5"])
    def test_a_malformed_selection_is_refused(self, spec):
        with pytest.raises(ValueError):
            parse_page_selection(spec)

    def test_a_huge_range_costs_nothing_before_the_page_count_is_known(self):
        assert parse_page_selection("1-999999999") == [(1, 999999999)]

    def test_expansion_is_sorted_unique_and_bounded_by_the_pdf(self):
        assert expand_page_selection(None, 3) == [1, 2, 3]
        assert expand_page_selection([(3, 3), (1, 2), (2, 2)], 3) == [1, 2, 3]
        with pytest.raises(ValueError, match="page 5 was asked for, but the PDF has 3"):
            expand_page_selection([(1, 1), (4, 5)], 3)


class TestRecordPageKeys:
    def test_a_born_digital_baseline_names_its_pages_by_page_index(self):
        labels = baseline_page_labels(json.dumps(_born_digital_baseline()).encode())
        assert labels == {1: "i", 2: "ii", 3: "1"}

    def test_a_baseline_without_page_index_maps_nothing(self):
        assert baseline_page_labels(json.dumps(_upstream_baseline()).encode()) == {}
        assert baseline_page_labels(None) == {}
        assert baseline_page_labels(b"") == {}
        assert baseline_page_labels(b"\xff not json") == {}

    def test_positions_become_labels_and_a_colliding_position_is_left_out(self):
        labels = {1: "i", 2: "ii", 3: "1", 5: "3"}
        assert record_page_keys([1, 2, 3, 4, 5], labels) == {1: "i", 2: "ii", 3: "1", 4: "4", 5: "3"}
        # page 6 has no row, and "3" is page 5's label — it must not be written onto that row
        assert record_page_keys([6, 3], {3: "6"}) == {6: None, 3: "6"}
        assert record_page_keys([1, 2], {}) == {1: "1", 2: "2"}


class TestPredictDocumentPagesAndLabels:
    def _post(self, client, data=None, baseline=None):
        files = {"file": ("report.pdf", b"%PDF-1.4 fake", "application/pdf")}
        if baseline is not None:
            files["document_json"] = ("seed.document.json", json.dumps(baseline).encode("utf-8"), "application/json")
        return client.post("/predict_document", data={"version": "v4.3", "topn": 3, **(data or {})}, files=files)

    def test_only_the_selected_pages_are_rendered_classified_and_recorded(self, client, fake_pdfium):
        response = self._post(client, {"pages": "1,3", "document_json_out": "true"})
        assert response.status_code == 200, response.text
        body = response.json()
        assert [page["page"] for page in body["pages"]] == [1, 3]
        assert all("page_label" not in page for page in body["pages"]), "no baseline, no labels"
        assert len(_FakePdfPage.scale_requests) == 2
        assert body["document_json"]["page_categories"] == {"1": "TEXT", "3": "TEXT"}

    def test_a_malformed_selection_is_422_before_the_pdf_is_opened(self, client, fake_pdfium):
        response = self._post(client, {"pages": "two"})
        assert response.status_code == 422
        assert response.json()["detail"].startswith("pages: ")
        assert _FakePdfPage.events == []

    def test_a_page_past_the_end_is_422_and_the_pdf_is_closed(self, client, fake_pdfium):
        response = self._post(client, {"pages": "2-4"})
        assert response.status_code == 422
        assert response.json()["detail"] == "pages: page 4 was asked for, but the PDF has 3 page(s)."
        assert _FakePdfPage.scale_requests == [] and _FakePdfPage.events == ["pdf.close"]

    def test_max_pdf_pages_counts_the_pages_classified(self, client, fake_pdfium, monkeypatch):
        monkeypatch.setenv("MAX_PDF_PAGES", "1")
        assert self._post(client, {"pages": "2"}).status_code == 200
        response = self._post(client, {"pages": "1-2"})
        assert response.status_code == 413
        assert response.json()["detail"] == "Too many pages selected: 2 of the PDF's 3. Limit is 1 (MAX_PDF_PAGES)."

    def test_categories_land_on_the_born_digital_records_own_labels(self, client, fake_pdfium):
        baseline = _born_digital_baseline()
        response = self._post(client, baseline=baseline)
        assert response.status_code == 200, response.text
        body = response.json()
        assert [(p["page"], p["page_label"]) for p in body["pages"]] == [(1, "i"), (2, "ii"), (3, "1")]
        record = body["document_json"]
        assert record["page_categories"] == {"i": "TEXT", "ii": "TEXT", "1": "TEXT"}
        assert [p["page"] for p in record["pages"]] == ["i", "ii", "1"], "no page row invented"
        for before, after in zip(baseline["pages"], record["pages"]):
            assert {k: after[k] for k in before} == before, "the converter's fields are untouched"
            assert after["category"] == "TEXT"
        assert record["lines"] == baseline["lines"] and record["source"] == baseline["source"]
        validate_document(record)

    def test_a_selection_on_a_labelled_record_writes_only_those_pages(self, client, fake_pdfium):
        response = self._post(client, {"pages": "3"}, baseline=_born_digital_baseline())
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["pages"] == [{"page": 3, "page_label": "1", "predictions": PREDS}]
        record = body["document_json"]
        assert record["page_categories"] == {"1": "TEXT"}
        assert [p.get("category") for p in record["pages"]] == [None, None, "TEXT"]

    def test_a_baseline_keyed_by_position_is_unchanged(self, client, fake_pdfium):
        """An ALTO record (pages "1".."N", no page_index) gets the same keys as before."""
        response = self._post(client, baseline=_upstream_baseline())
        assert response.status_code == 200
        body = response.json()
        assert all("page_label" not in page for page in body["pages"])
        assert body["document_json"]["page_categories"] == {"1": "TEXT", "2": "TEXT", "3": "TEXT"}
