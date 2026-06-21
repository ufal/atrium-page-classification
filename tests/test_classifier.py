"""
tests/test_classifier.py
========================
Unit tests for the pure-Python / NumPy parts of classifier.py.

Scope
-----
* custom_collate       – batch assembly, None-item filtering
* split_data_80_10_10  – deterministic 80/10/10 stratified split

Intentionally excluded (require GPU / Hugging Face download):
  ImageClassifier, BalancedBatchSampler, average_model_weights
  → mark with @pytest.mark.slow and test in an integration environment.

No GPU, no trained model, and no network access required.
"""
import numpy as np
import pytest
import torch

from classifier import custom_collate, split_data_80_10_10


def test_custom_collate_filters_none():
    """Ensure custom_collate gracefully drops None records resulting from corrupt images."""
    mock_batch = [
        {"pixel_values": torch.zeros((3, 224, 224)), "label": torch.tensor([1.0, 0.0])},
        None,
        {"pixel_values": torch.ones((3, 224, 224)), "label": torch.tensor([0.0, 1.0])}
    ]

    result = custom_collate(mock_batch)
    assert result is not None
    assert result["pixel_values"].shape == (2, 3, 224, 224)
    assert result["labels"].shape == (2, 2)
    assert torch.all(result["labels"][1] == torch.tensor([0.0, 1.0]))


def test_custom_collate_empty():
    """Ensure custom_collate handles entirely empty/corrupt batches."""
    assert custom_collate([None, None]) == (None, None)
    assert custom_collate([]) == (None, None)


def test_split_data_80_10_10_stratification():
    """Verify temporal distribution stratification in train/val/test splits."""
    # Generate 100 fake files for 2 classes
    files = [f"file_{i}.png" for i in range(100)]

    # Class 0: 60 items, Class 1: 40 items (One-hot encoded)
    labels = [np.array([1, 0])] * 60 + [np.array([0, 1])] * 40

    train_f, val_f, test_f, train_l, val_l, test_l = split_data_80_10_10(
        files, labels, random_seed=42, max_categ=100, safe_check=False
    )

    # Check total sum matches original 100
    assert len(train_f) + len(val_f) + len(test_f) == 100

    # Class 0 should have 6 in test, 6 in val (10% of 60)
    test_class_0 = sum(1 for lbl in test_l if lbl[0] == 1)
    val_class_0 = sum(1 for lbl in val_l if lbl[0] == 1)
    assert test_class_0 == 6
    assert val_class_0 == 6

    # Class 1 should have 4 in test, 4 in val (10% of 40)
    test_class_1 = sum(1 for lbl in test_l if lbl[1] == 1)
    val_class_1 = sum(1 for lbl in val_l if lbl[1] == 1)
    assert test_class_1 == 4
    assert val_class_1 == 4


@pytest.mark.slow
def test_image_classifier_instantiation():
    """Test actual model loading logic (marked slow to skip in fast CI)."""
    from classifier import ImageClassifier
    try:
        clf = ImageClassifier(checkpoint="google/vit-base-patch16-224", num_labels=11)
        assert clf.model is not None
        assert clf.processor is not None
        assert clf.model.config.num_labels == 11
    except Exception as e:
        pytest.fail(f"Failed to load checkpoint: {e}")


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════

def _pixel_tensor(b=3, h=4, w=4):
    """Return a minimal (C, H, W) float tensor suitable for pixel_values."""
    return torch.zeros(b, h, w)


def _item(label=None, n_classes=5):
    """Build a dataset-item dict as produced by ImageDataset.__getitem__."""
    lbl = torch.zeros(n_classes) if label is None else label
    return {"pixel_values": _pixel_tensor(), "label": lbl}


def _make_data(n_per_class: int, n_classes: int):
    """
    Generate dummy file paths and one-hot labels for split tests.
    File names are unique across classes to enable set-based overlap checks.
    """
    files, labels = [], []
    for cls in range(n_classes):
        for i in range(n_per_class):
            files.append(f"/fake/cls{cls}/img_{cls}_{i:05d}.png")
            lbl = np.zeros(n_classes)
            lbl[cls] = 1.0
            labels.append(lbl)
    return files, np.array(labels)


# ════════════════════════════════════════════════════════════════════════════
# custom_collate
# ════════════════════════════════════════════════════════════════════════════
class TestCustomCollate:
    """custom_collate filters None entries and stacks pixel_values / labels."""

    def test_valid_batch_returns_dict(self):
        batch = [_item(), _item()]
        result = custom_collate(batch)
        assert isinstance(result, dict)
        assert "pixel_values" in result
        assert "labels" in result

    def test_pixel_values_stacked_correctly(self):
        batch = [_item() for _ in range(3)]
        result = custom_collate(batch)
        # Shape: (batch_size, channels, H, W)
        assert result["pixel_values"].shape == (3, 3, 4, 4)

    def test_none_items_filtered_before_stacking(self):
        batch = [None, _item(), None, _item()]
        result = custom_collate(batch)
        assert isinstance(result, dict)
        assert result["pixel_values"].shape[0] == 2

    def test_all_none_batch_returns_sentinel(self):
        """Entirely-None batch should return the (None, None) sentinel that
        infer_dataloader checks for before processing."""
        result = custom_collate([None, None])
        # The function returns a tuple (None, None) – not a dict
        assert result == (None, None)

    def test_labels_stacked_into_2d_float_tensor(self):
        labels = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])]
        batch = [{"pixel_values": _pixel_tensor(), "label": lbl} for lbl in labels]
        result = custom_collate(batch)
        assert result["labels"].shape == (2, 2)
        assert result["labels"].dtype == torch.float32

    def test_none_labels_produce_empty_tensor(self):
        batch = [
            {"pixel_values": _pixel_tensor(), "label": None},
            {"pixel_values": _pixel_tensor(), "label": None},
        ]
        result = custom_collate(batch)
        assert result["labels"].numel() == 0

    def test_numpy_label_converted_to_tensor(self):
        lbl = np.array([0.0, 1.0, 0.0])
        batch = [{"pixel_values": _pixel_tensor(), "label": lbl}]
        result = custom_collate(batch)
        assert isinstance(result["labels"], torch.Tensor)
        assert result["labels"].shape == (1, 3)

    def test_list_label_converted_to_tensor(self):
        batch = [{"pixel_values": _pixel_tensor(), "label": [1.0, 0.0]}]
        result = custom_collate(batch)
        assert isinstance(result["labels"], torch.Tensor)


# ════════════════════════════════════════════════════════════════════════════
# split_data_80_10_10
# ════════════════════════════════════════════════════════════════════════════
class TestSplitData801010:
    """
    split_data_80_10_10 must partition the dataset into non-overlapping
    train / val / test splits that together cover the full dataset.

    All tests use safe_check=False so no real image files are required.
    """

    def test_returns_six_elements(self):
        files, labels = _make_data(20, 2)
        result = split_data_80_10_10(
            files, labels, random_seed=0, max_categ=100, safe_check=False
        )
        assert len(result) == 6

    def test_splits_sum_to_total_dataset_size(self):
        n_per_class, n_classes = 30, 3   # 90 files total
        files, labels = _make_data(n_per_class, n_classes)
        train_f, val_f, test_f, *_ = split_data_80_10_10(
            files, labels, random_seed=42, max_categ=200, safe_check=False
        )
        assert len(train_f) + len(val_f) + len(test_f) == n_per_class * n_classes

    def test_no_overlap_between_any_two_splits(self):
        files, labels = _make_data(20, 2)
        train_f, val_f, test_f, *_ = split_data_80_10_10(
            files, labels, random_seed=7, max_categ=100, safe_check=False
        )
        train_set, val_set, test_set = set(train_f), set(val_f), set(test_f)
        assert train_set.isdisjoint(val_set),  "train and val overlap"
        assert train_set.isdisjoint(test_set), "train and test overlap"
        assert val_set.isdisjoint(test_set),   "val and test overlap"

    def test_file_and_label_arrays_have_matching_lengths(self):
        files, labels = _make_data(15, 4)
        train_f, val_f, test_f, train_l, val_l, test_l = split_data_80_10_10(
            files, labels, random_seed=1, max_categ=100, safe_check=False
        )
        assert len(train_f) == len(train_l)
        assert len(val_f)   == len(val_l)
        assert len(test_f)  == len(test_l)

    def test_training_split_is_the_largest(self):
        files, labels = _make_data(30, 3)
        train_f, val_f, test_f, *_ = split_data_80_10_10(
            files, labels, random_seed=0, max_categ=200, safe_check=False
        )
        assert len(train_f) > len(val_f)
        assert len(train_f) > len(test_f)

    def test_max_categ_caps_samples_per_class(self):
        """With max_categ=5 and 2 classes the total must not exceed 10."""
        files, labels = _make_data(50, 2)   # 100 files; cap to 5 each → 10
        train_f, val_f, test_f, *_ = split_data_80_10_10(
            files, labels, random_seed=42, max_categ=5, safe_check=False
        )
        total = len(train_f) + len(val_f) + len(test_f)
        assert total <= 10

    def test_same_seed_produces_identical_splits(self):
        files, labels = _make_data(20, 3)
        r1 = split_data_80_10_10(files, labels, random_seed=99, max_categ=100, safe_check=False)
        r2 = split_data_80_10_10(files, labels, random_seed=99, max_categ=100, safe_check=False)
        # Compare train split file lists
        assert list(r1[0]) == list(r2[0])
        # Compare val split
        assert list(r1[1]) == list(r2[1])

    def test_different_seeds_produce_different_splits(self):
        files, labels = _make_data(30, 2)
        r1 = split_data_80_10_10(files, labels, random_seed=0,  max_categ=100, safe_check=False)
        r2 = split_data_80_10_10(files, labels, random_seed=99, max_categ=100, safe_check=False)
        # Very unlikely that different seeds produce identical test splits
        assert list(r1[2]) != list(r2[2]), (
            "Different seeds produced identical test splits – check RNG seeding"
        )

    def test_val_and_test_sizes_approximately_10pct_each(self):
        """Each held-out split should be ≈10 % of the full dataset."""
        n_per_class, n_classes = 50, 2   # 100 total; expect ~10 per split per class
        files, labels = _make_data(n_per_class, n_classes)
        _, val_f, test_f, *_ = split_data_80_10_10(
            files, labels, random_seed=0, max_categ=200, safe_check=False
        )
        total = n_per_class * n_classes         # 100
        # Allow ±3 files tolerance around the expected 10 %
        assert abs(len(val_f)  - total * 0.10) <= 3
        assert abs(len(test_f) - total * 0.10) <= 3
