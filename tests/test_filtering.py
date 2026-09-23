"""
tests/test_filtering.py
=======================
Unit tests for supplementary/filtering.py.

Scope
-----
* parse_stem           – naming-convention parsing (pure function, no I/O)
* build_valid_set      – valid-set construction from a training directory tree
* resolve_label_column – CLASS / CLASS-1 / CATEGORY lookup and --class-column
* filter_rows          – keep / relabel / remove against the tree (pure function)
* the CLI              – an annotation CSV (FILE,PAGE,CLASS) end to end

No GPU, no trained model, no network required.
"""

import csv
import subprocess
import sys
from pathlib import Path

from filtering import build_valid_set, filter_rows, parse_stem, resolve_label_column

FILTERING_PY = Path(__file__).resolve().parent.parent / "supplementary" / "scripts" / "filtering.py"


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


# ════════════════════════════════════════════════════════════════════════════
# resolve_label_column
# ════════════════════════════════════════════════════════════════════════════
class TestResolveLabelColumn:
    """CLASS (annotation CSV) first, then CLASS-1 (classifier output), then CATEGORY."""

    def test_annotation_csv_uses_class(self):
        assert resolve_label_column(["FILE", "PAGE", "CLASS"]) == "CLASS"

    def test_classifier_output_uses_class_1(self):
        assert resolve_label_column(["FILE", "PAGE", "CLASS-1", "SCORE-1", "CATEGORY"]) == "CLASS-1"

    def test_class_preferred_over_class_1(self):
        assert resolve_label_column(["FILE", "PAGE", "CLASS-1", "CLASS"]) == "CLASS"

    def test_category_is_the_last_resort(self):
        assert resolve_label_column(["FILE", "PAGE", "CATEGORY"]) == "CATEGORY"

    def test_match_is_case_insensitive_and_returns_the_csv_spelling(self):
        assert resolve_label_column(["file", "page", "Class"]) == "Class"

    def test_no_label_column_returns_none(self):
        assert resolve_label_column(["FILE", "PAGE", "NOTE"]) is None

    def test_override_wins(self):
        assert resolve_label_column(["FILE", "PAGE", "CLASS", "TRUE"], override="TRUE") == "TRUE"

    def test_override_absent_from_csv_returns_none(self):
        assert resolve_label_column(["FILE", "PAGE", "CLASS"], override="NOPE") is None


# ════════════════════════════════════════════════════════════════════════════
# filter_rows
# ════════════════════════════════════════════════════════════════════════════
class TestFilterRows:
    """filter_rows(rows, valid, label_col) → (kept, relabelled, removed)."""

    VALID = {
        ("a", 1, "TEXT"),
        ("b", 2, "DRAW"),  # annotated TEXT, moved to DRAW by a reviewer
        ("c", 3, "TEXT"),  # annotated LINE_P, now in two other folders
        ("c", 3, "PHOTO"),
    }

    @staticmethod
    def _row(file, page, label, col="CLASS"):
        return {"FILE": file, "PAGE": page, col: label}

    def test_row_still_in_place_is_kept_unchanged(self):
        row = self._row("a", "1", "TEXT")
        kept, relabelled, removed = filter_rows([row], self.VALID, "CLASS")
        assert kept == [row] and kept[0] is row
        assert relabelled == [] and removed == []

    def test_page_moved_to_one_other_folder_is_relabelled(self):
        kept, relabelled, removed = filter_rows([self._row("b", "2", "TEXT")], self.VALID, "CLASS")
        assert [r["CLASS"] for r in kept] == ["DRAW"]
        assert [(old, new) for _, old, new in relabelled] == [("TEXT", "DRAW")]
        assert removed == []

    def test_zero_padded_page_matches(self):
        kept, relabelled, _ = filter_rows([self._row("b", "02", "TEXT")], self.VALID, "CLASS")
        assert kept[0]["CLASS"] == "DRAW"
        assert kept[0]["PAGE"] == "02"  # the CSV's own spelling is written back untouched

    def test_relabel_does_not_mutate_the_input_row(self):
        row = self._row("b", "2", "TEXT")
        filter_rows([row], self.VALID, "CLASS")
        assert row["CLASS"] == "TEXT"

    def test_page_in_two_other_folders_is_removed_as_ambiguous(self):
        kept, relabelled, removed = filter_rows([self._row("c", "3", "LINE_P")], self.VALID, "CLASS")
        assert kept == [] and relabelled == []
        assert [reason for _, reason in removed] == ["ambiguous: PHOTO, TEXT"]

    def test_page_listed_where_it_still_is_wins_over_other_copies(self):
        """c-3 sits in TEXT and PHOTO; a row saying TEXT is simply correct."""
        kept, relabelled, removed = filter_rows([self._row("c", "3", "TEXT")], self.VALID, "CLASS")
        assert len(kept) == 1 and relabelled == [] and removed == []

    def test_page_gone_from_the_tree_is_removed_as_missing(self):
        kept, _, removed = filter_rows([self._row("d", "4", "TEXT")], self.VALID, "CLASS")
        assert kept == []
        assert [reason for _, reason in removed] == ["missing"]

    def test_no_relabel_drops_a_moved_page(self):
        kept, relabelled, removed = filter_rows([self._row("b", "2", "TEXT")], self.VALID, "CLASS", relabel=False)
        assert kept == [] and relabelled == []
        assert [reason for _, reason in removed] == ["missing"]

    def test_non_integer_page_is_removed_not_raised(self):
        kept, _, removed = filter_rows([self._row("a", "one", "TEXT")], self.VALID, "CLASS")
        assert kept == []
        assert [reason for _, reason in removed] == ["invalid page"]

    def test_class_1_relabel_carries_the_category_alias_along(self):
        row = {"FILE": "b", "PAGE": "2", "CLASS-1": "TEXT", "SCORE-1": "0.9", "CATEGORY": "TEXT"}
        kept, _, _ = filter_rows([row], self.VALID, "CLASS-1", mirror_col="CATEGORY")
        assert kept[0]["CLASS-1"] == "DRAW"
        assert kept[0]["CATEGORY"] == "DRAW"
        assert kept[0]["SCORE-1"] == "0.9"

    def test_input_order_is_preserved(self):
        rows = [self._row("b", "2", "TEXT"), self._row("d", "4", "TEXT"), self._row("a", "1", "TEXT")]
        kept, _, _ = filter_rows(rows, self.VALID, "CLASS")
        assert [r["FILE"] for r in kept] == ["b", "a"]


# ════════════════════════════════════════════════════════════════════════════
# CLI — the README's annotation CSV, end to end
# ════════════════════════════════════════════════════════════════════════════
class TestCli:
    @staticmethod
    def _run(*args):
        return subprocess.run([sys.executable, str(FILTERING_PY), *map(str, args)], capture_output=True, text=True)

    def test_annotation_csv_is_accepted_and_relabelled(self, tmp_path):
        """The README's FILE,PAGE,CLASS format used to exit asking for CLASS-1."""
        for cls, name in (("TEXT", "a-001.png"), ("DRAW", "b-2.png")):
            (tmp_path / "tree" / cls).mkdir(parents=True, exist_ok=True)
            (tmp_path / "tree" / cls / name).touch()
        csv_in = tmp_path / "ann.csv"
        # A BOM, as a spreadsheet export writes it, must not hide the FILE column.
        csv_in.write_text("\ufeffFILE,PAGE,CLASS,NOTE\na,1,TEXT,ok\nb,2,TEXT,moved\nd,4,TEXT,gone\n", encoding="utf-8")

        result = self._run("-d", tmp_path / "tree", "-i", csv_in)
        assert result.returncode == 0, result.stderr

        with (tmp_path / "ann_filtered.csv").open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert [(r["FILE"], r["CLASS"], r["NOTE"]) for r in rows] == [("a", "TEXT", "ok"), ("b", "DRAW", "moved")]
        assert "Relabelled:   1" in result.stdout
        assert "Rows removed: 1" in result.stdout

    def test_missing_label_column_exits_with_an_error(self, tmp_path):
        (tmp_path / "tree").mkdir()
        csv_in = tmp_path / "bad.csv"
        csv_in.write_text("FILE,PAGE,NOTE\na,1,x\n", encoding="utf-8")
        result = self._run("-d", tmp_path / "tree", "-i", csv_in)
        assert result.returncode != 0
        assert "missing required columns" in result.stderr
