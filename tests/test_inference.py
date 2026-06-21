from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from service.inference import ModelManager


@pytest.fixture
def manager():
    return ModelManager()


@pytest.fixture
def dummy_image():
    # Provide a minimal valid image for PIL processing
    return Image.new("RGB", (224, 224))


def test_get_base_model_id(manager):
    """Test registry dictionary lookups and error handling."""
    with patch("service.inference.REVISION_TO_BASE_MODEL", {"v4.3": "vit-base-test"}):
        assert manager._get_base_model_id("v4.3") == "vit-base-test"

    with pytest.raises(ValueError, match="Base model not found"):
        manager._get_base_model_id("unknown_version")


def test_get_model_details(manager):
    """Test the /info details generator."""
    assert "Ensemble" in manager.get_model_details("all")
    with patch.object(manager, "_get_base_model_id", return_value="test-architecture"):
        assert "test-architecture" in manager.get_model_details("v1.0")


@patch("service.inference.ImageClassifier")
def test_load_model_from_hub(mock_classifier, manager):
    """Test fallback to HF Hub if the model is not stored locally."""
    mock_clf_instance = MagicMock()
    mock_classifier.return_value = mock_clf_instance

    with patch.object(manager, "_get_base_model_id", return_value="test-base"):
        with patch("service.inference.Path.exists", return_value=False):
            clf = manager.load_model("vtest")
            mock_clf_instance.load_from_hub.assert_called_once()
            mock_clf_instance.save_model.assert_called_once()
            assert clf == mock_clf_instance


@patch.object(ModelManager, "load_model")
def test_predict_single_image(mock_load, manager, dummy_image):
    """Test single image inference path."""
    mock_clf = MagicMock()
    # Mocking standard returned scores
    mock_clf.top_n_predictions.return_value = [(0, 0.95), (1, 0.05)]
    mock_load.return_value = mock_clf

    with patch("service.inference.CATEGORIES", ["DRAW", "TEXT"]):
        res = manager.predict(dummy_image, version="vtest", topn=2)
        assert len(res) == 2
        assert res[0]["label"] == "DRAW"
        assert res[0]["score"] == 0.95


@patch.object(ModelManager, "_run_single_inference")
def test_predict_averaged_ensemble(mock_single, manager, dummy_image):
    """Test the 'all' version ensemble averaging logic."""
    manager.available_versions = ["v1", "v2"]
    mock_single.side_effect = [
        [{"label": "DRAW", "score": 0.8}, {"label": "TEXT", "score": 0.2}],
        [{"label": "DRAW", "score": 0.6}, {"label": "TEXT", "score": 0.4}],
    ]

    res = manager.predict(dummy_image, version="all", topn=1)
    assert len(res) == 1
    assert res[0]["label"] == "DRAW"
    assert res[0]["score"] == 0.70  # (0.8 + 0.6) / 2


@patch("os.listdir")
@patch.object(ModelManager, "load_model")
def test_predict_directory_batch(mock_load, mock_listdir, manager):
    """Test batch directory predictions filtering non-images."""
    mock_listdir.return_value = ["page1.png", "page2.jpg", "log.txt"]

    mock_clf = MagicMock()
    mock_clf.create_dataloader.return_value = "dummy_loader"
    mock_clf.infer_dataloader.return_value = (
        [[(0, 0.9)], [(1, 0.8)]],  # Image 1 and Image 2 predictions
        [],
    )
    mock_load.return_value = mock_clf

    with patch("service.inference.CATEGORIES", ["DRAW", "TEXT"]):
        res = manager.predict_directory("/dummy/path", version="vtest", topn=1)
        assert len(res) == 2  # txt file is ignored
        assert res[0][0]["label"] == "DRAW"
        assert res[1][0]["label"] == "TEXT"
