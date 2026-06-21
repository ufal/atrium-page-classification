"""
tests/test_utils.py
===================
Unit tests for utils.py – pure-Python utility functions.

Scope
-----
* directory_scraper  – filesystem traversal
* dataframe_results  – prediction → DataFrame conversion (top-1 and top-N)
* collect_images     – labelled dataset collection from a directory tree
* confusion_plot     – accuracy reporting and PNG output

No GPU, no trained model, and no network access required.
The Agg matplotlib backend is activated by conftest.py before this module
is imported, so no display is needed.
"""
from pathlib import Path

import numpy as np
import pytest

# Project root is already on sys.path via conftest.py
from utils import collect_images, confusion_plot, dataframe_results, directory_scraper

# ── shared fixture data ──────────────────────────────────────────────────────
ALL_CATEGORIES = [
    "DRAW", "DRAW_L", "LINE_HW", "LINE_P", "LINE_T",
    "PHOTO", "PHOTO_L", "TEXT", "TEXT_HW", "TEXT_P", "TEXT_T",
]


# ════════════════════════════════════════════════════════════════════════════
# directory_scraper
# ════════════════════════════════════════════════════════════════════════════
class TestDirectoryScraper:
    """directory_scraper(folder_path, file_format, file_list) should
    recursively collect all files matching file_format."""

    def test_counts_matching_files_in_flat_directory(self, tmp_path):
        (tmp_path / "a.png").touch()
        (tmp_path / "b.png").touch()
        (tmp_path / "c.jpg").touch()    # different extension – must be excluded
        result = directory_scraper(tmp_path, "png")
        assert len(result) == 2

    def test_returns_empty_list_when_no_format_match(self, tmp_path):
        (tmp_path / "a.jpg").touch()
        result = directory_scraper(tmp_path, "png")
        assert result == []

    def test_recurses_into_nested_subdirectories(self, tmp_path):
        sub = tmp_path / "subdir"
        sub.mkdir()
        (tmp_path / "top.png").touch()
        (sub / "nested.png").touch()
        result = directory_scraper(tmp_path, "png")
        assert len(result) == 2

    def test_accumulates_into_caller_supplied_list(self, tmp_path):
        (tmp_path / "new.png").touch()
        seed = ["pre_existing.png"]
        result = directory_scraper(tmp_path, "png", file_list=seed)
        assert len(result) == 2
        assert "pre_existing.png" in result

    def test_independent_calls_do_not_share_state(self, tmp_path):
        """Each call with file_list=None must start with a fresh list
        (guards against the mutable-default-argument anti-pattern)."""
        (tmp_path / "x.png").touch()
        r1 = directory_scraper(tmp_path, "png")
        r2 = directory_scraper(tmp_path, "png")
        assert len(r1) == 1
        assert len(r2) == 1


# ════════════════════════════════════════════════════════════════════════════
# dataframe_results – top_N == 1
# ════════════════════════════════════════════════════════════════════════════
class TestDataframeResultsTop1:
    """When top_N=1, predictions are plain class indices (int).
    SCORE-1 is dropped; a CATEGORY alias column is added instead."""

    def test_required_columns_present(self):
        df, _ = dataframe_results(["doc-001.png"], [0], ALL_CATEGORIES, top_N=1)
        for col in ("FILE", "PAGE", "CLASS-1", "CATEGORY"):
            assert col in df.columns, f"expected column '{col}' not found"

    def test_score_column_absent_for_top1(self):
        df, _ = dataframe_results(["doc-001.png"], [0], ALL_CATEGORIES, top_N=1)
        assert "SCORE-1" not in df.columns

    def test_category_alias_equals_class1(self):
        df, _ = dataframe_results(["doc-001.png"], [3], ALL_CATEGORIES, top_N=1)
        assert df.iloc[0]["CLASS-1"] == ALL_CATEGORIES[3]
        assert df.iloc[0]["CATEGORY"] == ALL_CATEGORIES[3]

    def test_page_number_parsed_from_hyphen_separator(self):
        df, _ = dataframe_results(["CTX193200994-24.png"], [0], ALL_CATEGORIES, top_N=1)
        assert df.iloc[0]["PAGE"] == 24
        assert df.iloc[0]["FILE"] == "CTX193200994"

    def test_page_number_parsed_from_underscore_separator(self):
        df, _ = dataframe_results(["report_012.png"], [0], ALL_CATEGORIES, top_N=1)
        assert df.iloc[0]["PAGE"] == 12
        assert df.iloc[0]["FILE"] == "report"

    def test_page_number_parsed_from_compound_name(self):
        """Greedy regex: last separator wins – 'report_2021_003' → FILE='report_2021', PAGE=3."""
        df, _ = dataframe_results(["report_2021_003.png"], [0], ALL_CATEGORIES, top_N=1)
        assert df.iloc[0]["PAGE"] == 3
        assert df.iloc[0]["FILE"] == "report_2021"

    def test_page_number_fallback_when_no_suffix(self):
        df, _ = dataframe_results(["coverpage.png"], [0], ALL_CATEGORIES, top_N=1)
        assert df.iloc[0]["PAGE"] == 1
        assert df.iloc[0]["FILE"] == "coverpage"

    def test_row_count_matches_number_of_images(self):
        images = ["a-1.png", "b-2.png", "c-3.png"]
        df, _ = dataframe_results(images, [0, 1, 2], ALL_CATEGORIES, top_N=1)
        assert len(df) == 3

    def test_raw_dataframe_is_none_when_not_requested(self):
        _, rawdf = dataframe_results(["a-1.png"], [0], ALL_CATEGORIES, top_N=1)
        assert rawdf is None


# ════════════════════════════════════════════════════════════════════════════
# dataframe_results – top_N > 1
# ════════════════════════════════════════════════════════════════════════════
class TestDataframeResultsTopN:
    """When top_N > 1, predictions are list[tuple[int, float]]."""

    def test_column_names_for_top3(self):
        preds = [[(0, 0.80), (1, 0.15), (2, 0.05)]]
        df, _ = dataframe_results(["doc-001.png"], preds, ALL_CATEGORIES, top_N=3)
        for n in range(1, 4):
            assert f"CLASS-{n}" in df.columns
            assert f"SCORE-{n}" in df.columns

    def test_scores_stored_rounded_to_3dp(self):
        preds = [[(0, 0.8123456), (1, 0.1876544)]]
        df, _ = dataframe_results(["doc-001.png"], preds, ALL_CATEGORIES, top_N=2)
        assert df.iloc[0]["SCORE-1"] == pytest.approx(0.812, abs=1e-3)
        assert df.iloc[0]["SCORE-2"] == pytest.approx(0.188, abs=1e-3)

    def test_class_labels_mapped_from_indices(self):
        # index 5 → "PHOTO", index 0 → "DRAW"
        preds = [[(5, 0.9), (0, 0.1)]]
        df, _ = dataframe_results(["doc-001.png"], preds, ALL_CATEGORIES, top_N=2)
        assert df.iloc[0]["CLASS-1"] == "PHOTO"
        assert df.iloc[0]["CLASS-2"] == "DRAW"

    def test_top1_always_adds_category_alias(self):
        """Regression: top_N=1 path must always produce the CATEGORY column."""
        df, _ = dataframe_results(["doc-001.png"], [0], ALL_CATEGORIES, top_N=1)
        assert "CATEGORY" in df.columns


# ════════════════════════════════════════════════════════════════════════════
# dataframe_results – raw scores
# ════════════════════════════════════════════════════════════════════════════
class TestDataframeResultsRawScores:

    def test_raw_dataframe_has_one_column_per_category(self):
        cats = ["A", "B"]
        raw = [[0.6, 0.4]]
        _, rawdf = dataframe_results(["doc-001.png"], [0], cats, top_N=1, raw_scores=raw)
        assert "A" in rawdf.columns
        assert "B" in rawdf.columns

    def test_raw_dataframe_row_count_matches_images(self):
        raw = [[0.6, 0.4], [0.3, 0.7]]
        _, rawdf = dataframe_results(
            ["a-1.png", "b-2.png"], [0, 1], ["A", "B"],
            top_N=1, raw_scores=raw,
        )
        assert len(rawdf) == 2

    def test_raw_scores_rounded_to_3dp(self):
        raw = [[0.999999, 0.000001]]
        _, rawdf = dataframe_results(["doc-001.png"], [0], ["A", "B"],
                                     top_N=1, raw_scores=raw)
        assert rawdf.iloc[0]["A"] == pytest.approx(1.0, abs=1e-3)


# ════════════════════════════════════════════════════════════════════════════
# collect_images
# ════════════════════════════════════════════════════════════════════════════
class TestCollectImages:
    """collect_images(directory) reads a folder of category sub-directories
    and returns (file_paths, one-hot_labels, category_names)."""

    # ── helper ────────────────────────────────────────────────────────────
    @staticmethod
    def _make_dataset(tmp_path: Path, structure: dict) -> None:
        """
        Create category sub-directories with dummy image files.

        structure = {"DRAW": ["img1.png", "img2.png"], "TEXT": ["img3.png"]}
        """
        for category, files in structure.items():
            cat_dir = tmp_path / category
            cat_dir.mkdir()
            for fname in files:
                (cat_dir / fname).touch()

    # ── tests ─────────────────────────────────────────────────────────────
    def test_categories_returned_in_alphabetical_order(self, tmp_path):
        self._make_dataset(tmp_path, {"TEXT": ["t.png"], "DRAW": ["d.png"]})
        _, _, cats = collect_images(str(tmp_path))
        assert cats == sorted(cats)

    def test_total_file_count_matches_structure(self, tmp_path):
        self._make_dataset(tmp_path, {"A": ["f1.png", "f2.png"], "B": ["f3.png"]})
        files, labels, _ = collect_images(str(tmp_path))
        assert len(files) == 3
        assert len(labels) == 3

    def test_labels_are_one_hot_vectors(self, tmp_path):
        self._make_dataset(tmp_path, {"A": ["f1.png"], "B": ["f2.png"], "C": ["f3.png"]})
        _, labels, _ = collect_images(str(tmp_path))
        for lbl in labels:
            assert lbl.sum() == 1.0
            assert set(lbl.tolist()) == {0.0, 1.0}

    def test_label_length_equals_number_of_categories(self, tmp_path):
        n_cats = 5
        structure = {f"cat{i}": ["img.png"] for i in range(n_cats)}
        self._make_dataset(tmp_path, structure)
        _, labels, cats = collect_images(str(tmp_path))
        assert len(cats) == n_cats
        for lbl in labels:
            assert len(lbl) == n_cats

    def test_first_alphabetical_category_gets_index_zero(self, tmp_path):
        """Category 'A' (index 0 after sorting) must have one-hot [1, 0]."""
        self._make_dataset(tmp_path, {"A": ["img.png"], "B": ["img.png"]})
        files, labels, cats = collect_images(str(tmp_path))
        assert cats[0] == "A"
        # With ordered=True, A/img.png sorts before B/img.png
        a_label = labels[0]
        assert a_label.tolist() == [1.0, 0.0]

    def test_return_types_are_lists(self, tmp_path):
        self._make_dataset(tmp_path, {"CAT": ["img.png"]})
        files, labels, cats = collect_images(str(tmp_path))
        assert isinstance(files, list)
        assert isinstance(labels, list)
        assert isinstance(cats, list)

    def test_ordered_flag_sorts_files_alphabetically(self, tmp_path):
        self._make_dataset(tmp_path, {"A": ["z_page.png", "a_page.png", "m_page.png"]})
        files, _, _ = collect_images(str(tmp_path), ordered=True)
        assert files == sorted(files)

    def test_two_images_same_category_share_identical_label(self, tmp_path):
        self._make_dataset(tmp_path, {"ONLY": ["p1.png", "p2.png"]})
        _, labels, _ = collect_images(str(tmp_path))
        assert np.array_equal(labels[0], labels[1])


# ════════════════════════════════════════════════════════════════════════════
# confusion_plot
# ════════════════════════════════════════════════════════════════════════════
class TestConfusionPlot:
    """confusion_plot writes a PNG to {output_dir}/plots/ and prints a
    classification report.  Tests verify the file is created and that
    the function handles both top-1 and top-N prediction formats."""

    @staticmethod
    def _output_dir(tmp_path: Path) -> str:
        """Create the expected 'plots' sub-directory and return parent as str."""
        (tmp_path / "plots").mkdir()
        return str(tmp_path)

    def test_creates_png_for_perfect_predictions(self, tmp_path):
        cats = ["DRAW", "TEXT"]
        preds = [0, 1, 0, 1]
        trues = [0, 1, 0, 1]
        confusion_plot(preds, trues, cats, "test_model",
                       top_N=1, output_dir=self._output_dir(tmp_path))
        pngs = list((tmp_path / "plots").glob("*.png"))
        assert len(pngs) == 1

    def test_creates_png_with_classification_errors(self, tmp_path):
        cats = ["A", "B", "C"]
        preds = [0, 1, 2, 0, 1, 2]
        trues = [0, 1, 1, 0, 2, 2]    # 2 misclassifications
        confusion_plot(preds, trues, cats, "test_model",
                       top_N=1, output_dir=self._output_dir(tmp_path))
        pngs = list((tmp_path / "plots").glob("*.png"))
        assert len(pngs) == 1

    def test_topn_format_handled_correctly(self, tmp_path):
        """top_N=2: predictions are list[tuple[int, float]], true class inside
        the top-2 candidates should count as correct."""
        cats = ["A", "B"]
        # each prediction: [(best_idx, score), (second_idx, score)]
        preds = [[(0, 0.9), (1, 0.1)],   # true=0, top-1 correct
                 [(1, 0.8), (0, 0.2)],   # true=1, top-1 correct
                 [(0, 0.55), (1, 0.45)]] # true=1, top-1 wrong but top-2 correct
        trues = [0, 1, 1]
        confusion_plot(preds, trues, cats, "test_model",
                       top_N=2, output_dir=self._output_dir(tmp_path))
        pngs = list((tmp_path / "plots").glob("*.png"))
        assert len(pngs) == 1

    def test_png_filename_embeds_model_name_and_topn(self, tmp_path):
        cats = ["X", "Y"]
        confusion_plot([0, 1], [0, 1], cats, "mymodel_v43",
                       top_N=1, output_dir=self._output_dir(tmp_path))
        pngs = list((tmp_path / "plots").glob("*.png"))
        assert len(pngs) == 1
        fname = pngs[0].name
        assert "mymodel_v43" in fname
        assert "TOP-1" in fname
