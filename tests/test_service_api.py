# filepath: ufal/atrium-page-classification/atrium-page-classification-test/tests/test_service_api.py
import io
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.append(str(Path(__file__).parent.parent))

from service.api import MAX_UPLOAD_BYTES, app

client = TestClient(app)


def test_info_endpoint():
    response = client.get("/info")
    assert response.status_code == 200
    data = response.json()
    assert "available_models" in data
    assert "categories" in data


def test_predict_image_bad_type():
    response = client.post(
        "/predict_image",
        data={"version": "v1.3", "topn": 3},
        files={"file": ("test.txt", b"fake image data", "text/plain")}
    )
    assert response.status_code == 400
    assert "Invalid file type" in response.json()["detail"]


def test_predict_image_size_limit():
    large_data = b"0" * (MAX_UPLOAD_BYTES + 1024)
    response = client.post(
        "/predict_image",
        data={"version": "v1.3", "topn": 3},
        files={"file": ("test.jpg", large_data, "image/jpeg")}
    )
    assert response.status_code == 413
    assert "File too large" in response.json()["detail"]


def test_predict_document_bad_type():
    response = client.post(
        "/predict_document",
        data={"version": "v1.3", "topn": 3},
        files={"file": ("test.txt", b"fake pdf data", "text/plain")}
    )
    assert response.status_code == 400
    assert "Invalid file type" in response.json()["detail"]


def test_predict_document_size_limit():
    large_data = b"0" * (MAX_UPLOAD_BYTES + 1024)
    response = client.post(
        "/predict_document",
        data={"version": "v1.3", "topn": 3},
        files={"file": ("test.pdf", large_data, "application/pdf")}
    )
    assert response.status_code == 413
    assert "File too large" in response.json()["detail"]


@pytest.fixture
def mock_manager(monkeypatch):
    class MockManager:
        device = "cpu"

        def get_model_details(self, version): return "mocked_model"

        def predict(self, image, version, topn):
            return [{"label": "TEXT", "score": 0.99}]

        def warmup(self, versions): pass

    from service import api
    monkeypatch.setattr(api, "manager", MockManager())
    return MockManager()


def test_predict_image_success(mock_manager):
    img = Image.new("RGB", (10, 10), color="white")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="JPEG")

    response = client.post(
        "/predict_image",
        data={"version": "v1.3", "topn": 3},
        files={"file": ("test.jpg", img_byte_arr.getvalue(), "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "image"
    assert len(data["predictions"]) == 1
    assert data["predictions"][0]["label"] == "TEXT"
