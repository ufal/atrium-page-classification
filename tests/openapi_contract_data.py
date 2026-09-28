"""Repo-local declarations for tests/test_openapi_contract.py (atrium-project#32 round 2).

Never vendored, never in para-drift, never in the ruff [format] exclude — unlike the
canonical test that reads it, this file's content is per repo by design: which services the
repo runs, where their committed specs live, which settings could reach a spec, and which
requirement files pin fastapi and pydantic. See the canonical test's docstring.

This service imports torch through service/inference.py, and torch is in no light test lane,
so `prepare()` stands a model manager in for it first — the same stub
tests/test_service_document_json.py uses. The spec does not depend on the manager (nothing
it knows reaches a route declaration), so the stubbed and the real app generate the same
document; the docker-tool lane's smoke step compares the real image's /openapi.json with the
committed one, which is the proof for the torch-bound build.
"""

from __future__ import annotations

import sys
import types

#: One entry per HTTP service of this repo. `primary`: the domain endpoints whose JSON 200
#: must be a named model (strategy §4.2).
SERVICES = [
    {
        "service": "atrium-page-classification",
        "app": "service.api:app",
        "spec": "service/openapi.json",
        "primary": ["/predict_image", "/predict_document"],
    },
]

#: Settings besides every [limit] variable (which the test perturbs from tool_limits.LIMITS)
#: that a deployment changes and that must not change the spec: the CORS origins and where
#: the models come from.
ENV_PERTURB = {
    "ALLOWED_ORIGINS": "https://example.org",
    "HF_HOME": "/nonexistent/hf-cache",
}

#: Every requirements file a lane or an image installs fastapi or pydantic from: the images
#: (service/requirements.txt, with setup/requirements.txt) and the light and docker-tool test
#: lanes (setup/requirements-test.txt).
PIN_FILES = ["service/requirements.txt", "setup/requirements-test.txt"]

#: Run before the app is imported: stubs the torch-bound model manager where torch is absent.
PREPARE = "tests.openapi_contract_data:prepare"


class _StubManager:
    """Stands in for service.inference.manager: no weights, no torch. What the routes read of it:
    `available_versions` and `get_model_details` (/info, the `version` check); `predict` answers
    one category, as the real manager's top-N list does."""

    device = "cpu"

    def __init__(self):
        from model_registry import REVISION_BEST_MODELS  # torch-free; service/inference.py's source too

        self.available_versions = list(REVISION_BEST_MODELS)

    def get_model_details(self, version):
        return "Ensemble (Average of 5 Models)" if version == "all" else f"stub ({version})"

    def predict(self, image, version, topn):
        return [{"label": "TEXT", "score": 0.99}][:topn]

    def warmup(self, versions=None):
        pass


def prepare() -> None:
    """Put the stub in place of service.inference, only when torch is not installed."""
    try:
        import torch  # noqa: F401
    except ImportError:
        stub = types.ModuleType("service.inference")
        stub.manager = _StubManager()
        sys.modules.setdefault("service.inference", stub)
