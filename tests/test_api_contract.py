"""tests/test_api_contract.py — ATRIUM API meta-contract conformance (strategy §4, issue #32).

Hermetic contract test: asserts the ``/info`` envelope, ``/health``, the advertised endpoint
set, and OpenAPI validity against the in-process app. ``importorskip``-guarded and tolerant of
missing service dependencies, so it is a clean no-op in the fast lane and a real check in CI.
"""

import pytest

# Tell pytest to skip this entire test file if 'inference' (or 'torch') is not installed
pytest.importorskip("inference")
from service.api.app import app  # noqa: E402

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

# --- per-service contract parameters -----------------------------------------------------------
SERVICE = "atrium-page-classification"
APP_IMPORT = "service.api"
PRIMARY_ENDPOINTS = ["/predict_image", "/predict_document"]
# -----------------------------------------------------------------------------------------------

try:
    app = __import__(APP_IMPORT, fromlist=["app"]).app
except Exception as exc:  # missing heavy service deps → skip cleanly
    pytest.skip(f"cannot import {APP_IMPORT}.app: {exc}", allow_module_level=True)

client = TestClient(app)


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
