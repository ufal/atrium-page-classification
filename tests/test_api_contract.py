"""tests/test_api_contract.py — ATRIUM API meta-contract conformance (strategy §4, issue #32).

Hermetic contract test: asserts the ``/info`` envelope, ``/health``, ``/ready`` (issue #55), the advertised endpoint
set, and OpenAPI validity against the in-process app. Tolerant of missing service dependencies,
so it is a clean no-op in the fast lane and a real check in CI.
"""

import pytest

# --- per-service contract parameters -----------------------------------------------------------
SERVICE = "atrium-page-classification"
APP_IMPORT = "service.api"
PRIMARY_ENDPOINTS = ["/predict_image", "/predict_document"]
# -----------------------------------------------------------------------------------------------

try:
    from fastapi.testclient import TestClient

    # Stand a model manager in for the torch-bound service.inference when torch is absent
    # (atrium-project#32 round 2): the same stub tests/test_openapi_contract.py and the spec
    # export use, so this meta-contract test and the conformance tests at the end run in the
    # fast lane too. With torch installed it does nothing and the real manager is imported.
    from tests.openapi_contract_data import prepare

    prepare()
    app = __import__(APP_IMPORT, fromlist=["app"]).app
    client = TestClient(app)
    deps_present = True
except ImportError:
    # Only a missing dependency skips (atrium-project#53). This used to be `except Exception`,
    # which turned ANY import-time failure into a green skip — including a malformed limit
    # (atrium_limits.LimitConfigError), which must fail loudly.
    app = None
    client = None
    deps_present = False

# Apply skip to ALL tests in this file if heavy dependencies are missing.
# This allows Pytest to COLLECT the tests (avoiding Exit Code 5) but skip their execution.
pytestmark = pytest.mark.skipif(
    not deps_present, reason="Missing heavy service dependencies (inference, etc.) -> skipping cleanly"
)


def test_info_envelope_required_fields():
    """§4.1: /info always carries service, version, endpoints, limits.max_upload_mb."""
    response = client.get("/info")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == SERVICE
    assert data["version"] and data["version"] == app.version
    assert isinstance(data["endpoints"], list) and data["endpoints"]
    assert isinstance(data["limits"], dict)
    assert "max_upload_mb" in data["limits"]


def test_info_reports_every_declared_limit():
    """atrium-project#53: /info `limits` is tool_limits.LIMITS, value for value, and
    `limits_meta` names the variable that sets each one. tests/test_limits_contract.py checks
    the declaration against .env.example and the README."""
    from tool_limits import LIMITS

    data = client.get("/info").json()
    assert data["limits"] == LIMITS.values()
    assert data["limits_meta"] == LIMITS.meta()


def test_errors_have_the_harmonised_body():
    """§4.4 (atrium-project#32 item 2): every error is {status, reason, detail}."""
    body = client.get("/no-such-route").json()
    assert body == {"status": 404, "reason": None, "detail": "Not Found"}


def test_info_endpoints_match_real_routes():
    """Advertised endpoints are real routes, and every primary endpoint is advertised."""
    advertised = set(client.get("/info").json()["endpoints"])
    real = {r.path for r in app.routes if getattr(r, "methods", None)}
    assert advertised <= real
    for path in PRIMARY_ENDPOINTS:
        assert path in advertised, f"{path} missing from /info endpoints"


def test_health_shallow_ok():
    """§4.1: shallow /health is a cheap 200 liveness probe."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] in {"ok", "degraded"}


def test_primary_endpoints_documented_in_openapi():
    paths = app.openapi()["paths"]
    for path in PRIMARY_ENDPOINTS:
        assert path in paths, f"{path} missing from OpenAPI paths"


def test_openapi_document_is_spec_valid():
    """The runtime /openapi.json validates against the OpenAPI 3.x spec (§2.2)."""
    spec_validator = pytest.importorskip("openapi_spec_validator")
    spec_validator.validate(app.openapi())


# --- §4.6 readiness + shutdown contract (issue #55) --------------------------------------------
# The state-machine itself is unit-tested once, in the hub
# (atrium-project/docs/templates/shared/test_atrium_service.py). What these assert is that THIS
# repo actually wired it up: the route exists, it is advertised, and — the one that matters —
# liveness does not start failing just because the service is draining.

try:
    _state = getattr(__import__(APP_IMPORT, fromlist=["app"]), "_state", None)
except Exception:  # noqa: BLE001 - same missing-heavy-deps case this file already guards
    # Repos guard the app import two different ways (module-level pytest.skip vs a
    # `deps_present` flag + pytestmark.skipif). Under the second style this module keeps
    # loading after a failed import, so this must not raise at import time; the skip
    # marker already stops the tests below from running.
    _state = None


def test_ready_route_is_registered_and_advertised():
    """§4.6: /ready exists, and /info advertises it like any other route."""
    assert _state is not None, (
        f"{APP_IMPORT} has no module-level `_state` — the service has not adopted "
        "ServiceState/attach_health(state=...) (issue #55)"
    )
    response = client.get("/ready")
    assert response.status_code in (200, 503)
    assert response.json()["status"] in {"ready", "starting", "draining"}
    assert "/ready" in client.get("/info").json()["endpoints"]


def test_ready_reports_starting_before_warmup_and_ready_after():
    """503 until the service's own lifespan marks it warm, 200 once it has.

    `client` above is a bare TestClient, so the ASGI lifespan has NOT run and the service is
    genuinely un-warm here — which is exactly the pre-warmup state a Kubernetes startupProbe
    sees on a cold pod.
    """
    assert _state is not None
    was_warm, was_draining = _state.warm, _state.draining
    try:
        _state.draining = False
        _state.warm = False
        assert client.get("/ready").status_code == 503
        assert client.get("/ready").json()["status"] == "starting"

        _state.warm = True
        assert client.get("/ready").status_code == 200
        assert client.get("/ready").json()["status"] == "ready"
    finally:
        _state.warm, _state.draining = was_warm, was_draining


def test_liveness_stays_200_while_draining_but_readiness_does_not():
    """The load-bearing distinction of issue #55.

    If shallow /health went 503 on SIGTERM, an orchestrator's livenessProbe would SIGKILL the
    container before its drain finished — the very failure the drain exists to prevent. Routing
    traffic away from a draining pod is /ready's job.
    """
    assert _state is not None
    was_warm, was_draining = _state.warm, _state.draining
    try:
        _state.warm = True
        _state.draining = True

        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == {"status": "ok"}

        ready = client.get("/ready")
        assert ready.status_code == 503
        assert ready.json()["status"] == "draining"
    finally:
        _state.warm, _state.draining = was_warm, was_draining


def test_deep_health_reports_draining_with_operator_fields():
    """`?deep=true` had no coverage in any repo before issue #55."""
    assert _state is not None
    was_warm, was_draining = _state.warm, _state.draining
    try:
        _state.warm = True
        _state.draining = True
        response = client.get("/health?deep=true")
        assert response.status_code == 503
        body = response.json()
        assert body["status"] == "degraded"
        assert body["detail"] == "shutting down"
        assert body["draining"] is True
        assert "in_flight" in body
    finally:
        _state.warm, _state.draining = was_warm, was_draining


# --- the typed contract (atrium-project#32 round 2) --------------------------------------------
# tests/test_openapi_contract.py (canonical, vendored) checks the committed spec itself. What
# these add is the part only this repo can do: drive both endpoints (with a counting model
# manager and, for PDFs, a stand-in pypdfium2, as tests/test_service_document_json.py does) and
# hold every response — 200s and refusals alike — to the schema the PUBLISHED spec declares.

import io  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import types  # noqa: E402
from pathlib import Path  # noqa: E402

import atrium_openapi  # noqa: E402

_SPEC = atrium_openapi.load(Path(__file__).resolve().parent.parent / "service" / "openapi.json")
_SEED_ID = "C-202000543A-DT-27"
#: A #67 R1 seed: the AMČR file id and the source, nothing else.
_SEED = {"doc_id": _SEED_ID, "source": {"sha256": "9" * 64, "filename": "scan_0001.png"}}


class _CountingManager:
    device = "cpu"
    available_versions = ["v1.4", "v2.4", "v3.4", "v4.4", "v5.4"]

    def __init__(self, answer=None):
        self.calls = 0
        self.answer = answer

    def get_model_details(self, version):
        return f"stub ({version})"

    def predict(self, image, version, topn):
        self.calls += 1
        return self.answer if self.answer is not None else [{"label": "TEXT", "score": 0.91}][:topn]


@pytest.fixture
def manager(monkeypatch):
    from service import api

    counting = _CountingManager()
    monkeypatch.setattr(api, "manager", counting)
    return counting


def _png():
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color="white").save(buf, format="PNG")
    return buf.getvalue()


class _Bitmap:
    def to_pil(self):
        from PIL import Image

        return Image.new("RGB", (4, 4), color="white")

    def close(self):
        pass


class _Page:
    def get_size(self):
        return (4.0, 4.0)

    def render(self, scale=None):
        return _Bitmap()

    def close(self):
        pass


class _Pdf:
    def __len__(self):
        return 2

    def __getitem__(self, index):
        return _Page()

    def init_forms(self):
        pass

    def close(self):
        pass


@pytest.fixture
def pdfium(monkeypatch):
    """A stand-in pypdfium2: two pages, or `PdfDocument` raising what PDFium raises for a bad PDF."""
    module = types.ModuleType("pypdfium2")

    class PdfiumError(RuntimeError):
        pass

    module.PdfiumError = PdfiumError
    module.PdfDocument = lambda content: _Pdf()
    monkeypatch.setitem(sys.modules, "pypdfium2", module)
    return module


def _conforms(status, response, path, method="post"):
    pytest.importorskip("jsonschema")
    assert response.status_code == status, response.text
    atrium_openapi.validate_response(_SPEC, path, method, status, response.json())
    return response.json()


def test_an_image_response_conforms_with_every_field_present(manager):
    body = _conforms(
        200, client.post("/predict_image", files={"file": ("p.png", _png(), "image/png")}), "/predict_image"
    )
    assert body["predictions"] == [{"label": "TEXT", "score": 0.91}]
    # response_model sends every field: the nullable ones as null
    assert (body["document_json"], body["document_json_schema_error"]) == (None, None)
    # ...and the call's CreateAction on every success (atrium-project#71 R2): with no record asked
    # for, what it wrote is the predictions it answers with.
    import atrium_rocrate

    assert atrium_rocrate.action_problems(body["paradata"]) == []
    assert [item["name"] for item in body["paradata"]["result"]] == ["predictions.json"]


def test_an_image_response_with_a_seed_conforms_including_the_record(manager):
    """The returned record is held to the vendored record schema, through the spec's
    AtriumDocument component — the type AMČR's generated client deserialises it into."""
    files = {
        "file": ("scan_0001.png", _png(), "image/png"),
        "document_json": ("seed.document.json", json.dumps(_SEED).encode(), "application/json"),
    }
    body = _conforms(200, client.post("/predict_image", files=files), "/predict_image")
    assert body["document_json"]["doc_id"] == _SEED_ID and body["document_json"]["page_categories"] == {"1": "TEXT"}


def test_the_action_is_the_run_the_record_is_stamped_with(manager):
    """atrium-project#71 R2: the CreateAction's `@id` is the run_uuid stamped into the blocks this
    call wrote; `object` names the upload and the record sent with it, `result` the two blocks."""
    import atrium_rocrate

    files = {
        "file": ("scan_0001.png", _png(), "image/png"),
        "document_json": ("seed.document.json", json.dumps(_SEED).encode(), "application/json"),
    }
    body = _conforms(200, client.post("/predict_image", files=files), "/predict_image")
    action, record = body["paradata"], body["document_json"]
    assert atrium_rocrate.action_problems(action) == []
    stamps = record["assembled"]["blocks"]
    assert action["@id"] == stamps["page_categories"]["run_uuid"] == stamps["pages"]["run_uuid"]
    assert record["assembled"]["blocks"]["pages"]["paradata_ref"] == action["@id"]
    assert [item["@id"] for item in action["object"]][1:] == ["#record"]
    assert {"#block-page_categories", "#block-pages"} <= {item["@id"] for item in action["result"]}


def test_an_empty_record_part_still_originates_the_record(manager):
    """page-classification is stage 1: sending the part at all opts into the record, as it did."""
    files = {"file": ("CTX01_0007.png", _png(), "image/png"), "document_json": ("e.json", b"", "application/json")}
    body = _conforms(200, client.post("/predict_image", files=files), "/predict_image")
    assert body["document_json"]["assembled"]["had_baseline"] is False


def test_a_document_response_conforms(manager, pdfium):
    response = client.post(
        "/predict_document",
        data={"document_json_out": "true"},
        files={"file": ("CTX01.pdf", b"%PDF-1.4", "application/pdf")},
    )
    body = _conforms(200, response, "/predict_document")
    assert [page["page"] for page in body["pages"]] == [1, 2]
    assert body["document_json"]["page_categories"] == {"1": "TEXT", "2": "TEXT"}

    # The call's CreateAction (atrium-project#71 R2), with the PDF engine among its components.
    import atrium_rocrate

    action = body["paradata"]
    assert atrium_rocrate.action_problems(action) == []
    assert action["@id"] == body["document_json"]["assembled"]["blocks"]["page_categories"]["run_uuid"]
    assert "pypdfium2" in json.dumps(action["paradataRecord"])


@pytest.mark.parametrize(
    "path, name, media_type, accepted",
    [
        ("/predict_image", "a.txt", "text/plain", ["image/*"]),
        ("/predict_document", "a.png", "image/png", ["application/pdf"]),
    ],
)
def test_a_wrong_media_type_is_415_unsupported_media_type(manager, path, name, media_type, accepted):
    body = _conforms(415, client.post(path, files={"file": (name, b"x", media_type)}), path)
    assert (body["reason"], body["accepted"]) == ("unsupported_media_type", accepted)
    assert body["detail"].startswith("Invalid file type.")


@pytest.mark.parametrize("path", ["/predict_image", "/predict_document"])
def test_an_unknown_version_is_422_before_any_model_runs(manager, path):
    kind = ("p.png", _png(), "image/png") if path == "/predict_image" else ("d.pdf", b"%PDF", "application/pdf")
    body = _conforms(422, client.post(path, data={"version": "latest"}, files={"file": kind}), path)
    assert "latest" in body["detail"] and "v4.4" in body["detail"] and manager.calls == 0


def test_a_known_revision_prefix_is_not_refused(manager):
    """Only a version no revision matches is refused: a sub-revision the loader resolves passes."""
    response = client.post("/predict_image", data={"version": "v4.40"}, files={"file": ("p.png", _png(), "image/png")})
    assert response.status_code == 200 and manager.calls == 1


@pytest.mark.parametrize("record", [b"[1, 2]", b"{not json", b'{"schema_version": "9.0", "doc_id": "x"}'])
def test_a_record_that_cannot_be_opened_is_422_invalid_record_before_any_model_runs(manager, record):
    files = {"file": ("p.png", _png(), "image/png"), "document_json": ("r.json", record, "application/json")}
    body = _conforms(422, client.post("/predict_image", files=files), "/predict_image")
    assert body["reason"] == "invalid_record" and manager.calls == 0


def test_an_unreadable_image_is_422_not_500(manager):
    body = _conforms(
        422, client.post("/predict_image", files={"file": ("p.png", b"not an image", "image/png")}), "/predict_image"
    )
    assert body["detail"].startswith("The upload is not a readable image") and manager.calls == 0


def test_an_unreadable_pdf_is_422_not_500(manager, pdfium):
    def refuse(content):
        raise pdfium.PdfiumError("Failed to load document (PDFium: Data format error).")

    pdfium.PdfDocument = refuse
    response = client.post("/predict_document", files={"file": ("d.pdf", b"not a pdf", "application/pdf")})
    body = _conforms(422, response, "/predict_document")
    assert body["detail"] == "The upload is not a readable PDF: Failed to load document (PDFium: Data format error)."


def test_a_page_the_model_could_not_classify_is_a_500_not_a_200(manager, pdfium):
    """Its `{"error": ...}` used to stand in for the page's predictions inside a 200."""
    manager.answer = {"error": "All models failed."}
    body = _conforms(
        500,
        client.post("/predict_document", files={"file": ("d.pdf", b"%PDF", "application/pdf")}),
        "/predict_document",
    )
    assert body["detail"] == "Page 1: All models failed."


def test_info_conforms_to_the_published_schema(manager):
    _conforms(200, client.get("/info"), "/info", method="get")
