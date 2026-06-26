"""
tests/test_filtering.py
=======================
Unit tests for supplementary/filtering.py.

Scope
-----
* parse_stem      – naming-convention parsing (pure function, no I/O)
* build_valid_set – valid-set construction from a training directory tree

No GPU, no trained model, no network required.
"""

from pathlib import Path

from filtering import build_valid_set, parse_stem


# ════════════════════════════════════════════════════════════════════════════
# parse_stem
# ════════════════════════════════════════════════════════════════════════════
class TestParseStem:
    """parse_stem(stem) → (file_prefix, page_number) or (None, None)."""

    # ── standard naming conventions ───────────────────────────────────────
    def test_hyphen_separator_zero_padded(self):
        assert parse_stem("thesis-008") == ("thesis", 8)

    def test_hyphen_separator_single_digit(self):
        assert parse_stem("defense-1") == ("defense", 1)

    def test_underscore_separator(self):
        assert parse_stem("pages_online_13") == ("pages_online", 13)

    def test_compound_underscore_name_greedy_match(self):
        """Greedy `.+` captures everything up to the *last* separator."""
        assert parse_stem("presentation_thesis_01") == ("presentation_thesis", 1)

    def test_mixed_separators_last_wins(self):
        assert parse_stem("caa_conference-02") == ("caa_conference", 2)

    def test_number_embedded_in_prefix_not_confused(self):
        """'volume2' is the prefix; '001' is the page."""
        assert parse_stem("volume2-001") == ("volume2", 1)

    def test_year_in_filename_prefix(self):
        """'arch1985' is the prefix; '003' is the page."""
        assert parse_stem("arch1985-003") == ("arch1985", 3)

    # ── leading zeros stripped from page number ───────────────────────────
    def test_leading_zeros_stripped(self):
        _, page = parse_stem("doc-007")
        assert page == 7

    def test_large_page_number(self):
        _, page = parse_stem("bigarchive-1234")
        assert page == 1234

    # ── non-matching stems ────────────────────────────────────────────────
    def test_no_separator_returns_none_tuple(self):
        assert parse_stem("coverpage") == (None, None)

    def test_only_digits_returns_none_tuple(self):
        assert parse_stem("12345") == (None, None)

    def test_separator_with_no_prefix_returns_none(self):
        """A stem like '-5' has an empty prefix — not a valid pair."""
        assert parse_stem("-5") == (None, None)

    def test_separator_with_no_digits_returns_none(self):
        assert parse_stem("doc-") == (None, None)


# ════════════════════════════════════════════════════════════════════════════
# build_valid_set
# ════════════════════════════════════════════════════════════════════════════
class TestBuildValidSet:
    """build_valid_set(data_dir) returns (valid_set, unmatched_names)."""

    @staticmethod
    def _make_dir(root: Path, structure: dict) -> None:
        """
        Build a category-tree directory from a dict.
        structure = {"CAT": ["file-1.png", "file-2.png"], ...}
        """
        for cat, files in structure.items():
            (root / cat).mkdir(parents=True, exist_ok=True)
            for fname in files:
                (root / cat / fname).touch()

    # ── valid set contents ────────────────────────────────────────────────
    def test_count_matches_total_valid_pngs(self, tmp_path):
        self._make_dir(tmp_path, {"TEXT": ["a-1.png", "b-2.png"], "DRAW": ["c-3.png"]})
        valid, _ = build_valid_set(tmp_path)
        assert len(valid) == 3

    def test_correct_triple_in_valid_set(self, tmp_path):
        self._make_dir(tmp_path, {"TEXT": ["thesis-008.png"]})
        valid, _ = build_valid_set(tmp_path)
        assert ("thesis", 8, "TEXT") in valid

    def test_underscore_stem_indexed(self, tmp_path):
        self._make_dir(tmp_path, {"DRAW": ["map_survey_013.png"]})
        valid, _ = build_valid_set(tmp_path)
        assert ("map_survey", 13, "DRAW") in valid

    def test_same_filename_different_categories_indexed_separately(self, tmp_path):
        self._make_dir(
            tmp_path,
            {
                "TEXT": ["doc-1.png"],
                "DRAW": ["doc-1.png"],
            },
        )
        valid, _ = build_valid_set(tmp_path)
        assert ("doc", 1, "TEXT") in valid
        assert ("doc", 1, "DRAW") in valid

    def test_file_not_in_valid_set_for_wrong_class(self, tmp_path):
        self._make_dir(tmp_path, {"TEXT": ["doc-1.png"]})
        valid, _ = build_valid_set(tmp_path)
        assert ("doc", 1, "DRAW") not in valid

    # ── unmatched names ───────────────────────────────────────────────────
    def test_unmatched_filename_excluded_from_valid_set(self, tmp_path):
        self._make_dir(tmp_path, {"TEXT": ["no_page_number.png"]})
        valid, unmatched = build_valid_set(tmp_path)
        assert len(valid) == 0
        assert len(unmatched) == 1

    def test_unmatched_entry_prefixed_with_category(self, tmp_path):
        self._make_dir(tmp_path, {"TEXT": ["bad.png"]})
        _, unmatched = build_valid_set(tmp_path)
        assert unmatched[0].startswith("TEXT/")

    def test_matched_and_unmatched_counted_correctly(self, tmp_path):
        self._make_dir(tmp_path, {"TEXT": ["good-1.png", "bad.png"]})
        valid, unmatched = build_valid_set(tmp_path)
        assert len(valid) == 1
        assert len(unmatched) == 1

    # ── edge cases ────────────────────────────────────────────────────────
    def test_empty_directory_returns_empty_collections(self, tmp_path):
        valid, unmatched = build_valid_set(tmp_path)
        assert len(valid) == 0
        assert len(unmatched) == 0

    def test_non_png_files_ignored(self, tmp_path):
        (tmp_path / "TEXT").mkdir()
        (tmp_path / "TEXT" / "doc-1.jpg").touch()  # JPEG — skipped
        (tmp_path / "TEXT" / "doc-2.png").touch()  # PNG  — included
        valid, _ = build_valid_set(tmp_path)
        assert len(valid) == 1
        assert ("doc", 2, "TEXT") in valid

    def test_nested_category_files_indexed(self, tmp_path):
        self._make_dir(
            tmp_path,
            {
                "PHOTO": ["photo-001.png", "photo-002.png", "photo-003.png"],
            },
        )
        valid, _ = build_valid_set(tmp_path)
        assert len(valid) == 3

    def test_valid_set_is_a_python_set(self, tmp_path):
        self._make_dir(tmp_path, {"TEXT": ["a-1.png"]})
        valid, _ = build_valid_set(tmp_path)
        assert isinstance(valid, set)

    def test_unmatched_names_is_a_list(self, tmp_path):
        self._make_dir(tmp_path, {"TEXT": ["a-1.png"]})
        _, unmatched = build_valid_set(tmp_path)
        assert isinstance(unmatched, list)
