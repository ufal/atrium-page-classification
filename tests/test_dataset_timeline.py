"""
tests/test_dataset_timeline.py
==============================
Unit tests for supplementary/dataset_timeline.py.

Scope
-----
* parse_csv_by_category – year extraction via regex, category count aggregation,
  handling of missing-year rows, case-insensitive column names, sorted index.
* plot_stacked_timeline – PNG file created, zero-count categories handled cleanly.

Matplotlib uses the non-interactive Agg backend (activated in conftest.py).
No GPU, no trained model, no network required.
"""
import csv
from pathlib import Path

import pandas as pd
import pytest

from dataset_timeline import parse_csv_by_category, plot_stacked_timeline

DEFAULT_REGEX = r'((?:19|20)\d{2})'


# ── helpers ────────────────────────────────────────────────────────────────

def write_annotation_csv(path: Path, rows: list) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)


# ════════════════════════════════════════════════════════════════════════════
# parse_csv_by_category
# ════════════════════════════════════════════════════════════════════════════
class TestParseCsvByCategory:

    def test_year_appears_in_dataframe_index(self, tmp_path):
        f = tmp_path / "data.csv"
        write_annotation_csv(f, [
            ["file", "page", "category"],
            ["report_1985_001.png", "1", "TEXT"],
        ])
        df = parse_csv_by_category(str(f), DEFAULT_REGEX)
        assert 1985 in df.index

    def test_category_count_correct_for_single_year(self, tmp_path):
        f = tmp_path / "data.csv"
        write_annotation_csv(f, [
            ["file", "page", "category"],
            ["arch1985_001.png", "1", "TEXT"],
            ["arch1985_002.png", "2", "TEXT"],
            ["arch1985_003.png", "3", "DRAW"],
        ])
        df = parse_csv_by_category(str(f), DEFAULT_REGEX)
        assert df.loc[1985, "TEXT"] == 2
        assert df.loc[1985, "DRAW"] == 1

    def test_rows_without_year_excluded(self, tmp_path):
        f = tmp_path / "data.csv"
        write_annotation_csv(f, [
            ["file", "page", "category"],
            ["nodatefile.png", "1", "TEXT"],         # no year → dropped
            ["arch2001_001.png", "2", "DRAW"],        # year present → kept
        ])
        df = parse_csv_by_category(str(f), DEFAULT_REGEX)
        assert len(df.index) == 1
        assert 2001 in df.index

    def test_multiple_years_each_appear_in_index(self, tmp_path):
        f = tmp_path / "data.csv"
        write_annotation_csv(f, [
            ["file", "page", "category"],
            ["report1990_001.png", "1", "TEXT"],
            ["report2005_001.png", "1", "DRAW"],
        ])
        df = parse_csv_by_category(str(f), DEFAULT_REGEX)
        assert 1990 in df.index
        assert 2005 in df.index

    def test_index_sorted_ascending(self, tmp_path):
        f = tmp_path / "data.csv"
        write_annotation_csv(f, [
            ["file", "page", "category"],
            ["r2010_1.png", "1", "TEXT"],
            ["r1980_1.png", "1", "TEXT"],
            ["r1995_1.png", "1", "TEXT"],
        ])
        df = parse_csv_by_category(str(f), DEFAULT_REGEX)
        years = list(df.index)
        assert years == sorted(years)

    def test_uppercase_column_names_accepted(self, tmp_path):
        """Column names are lowercased internally so any case is accepted."""
        f = tmp_path / "data.csv"
        write_annotation_csv(f, [
            ["FILE", "PAGE", "CATEGORY"],
            ["arch1970_001.png", "1", "TEXT"],
        ])
        df = parse_csv_by_category(str(f), DEFAULT_REGEX)
        assert 1970 in df.index

    def test_class_column_accepted_as_category(self, tmp_path):
        """Both 'category' and 'class' are recognised as the label column."""
        f = tmp_path / "data.csv"
        write_annotation_csv(f, [
            ["file", "page", "class"],
            ["arch2000_001.png", "1", "DRAW"],
        ])
        df = parse_csv_by_category(str(f), DEFAULT_REGEX)
        assert 2000 in df.index

    def test_custom_regex_extracts_alternative_year_range(self, tmp_path):
        f = tmp_path / "data.csv"
        write_annotation_csv(f, [
            ["file", "page", "category"],
            ["medieval_1347_fol001.png", "1", "TEXT"],
        ])
        df = parse_csv_by_category(str(f), r'(1[0-9]{3})')
        assert 1347 in df.index

    def test_missing_categories_filled_with_zero(self, tmp_path):
        """If a known category never appears it should be present as 0."""
        f = tmp_path / "data.csv"
        write_annotation_csv(f, [
            ["file", "page", "category"],
            ["report1990_001.png", "1", "TEXT"],
        ])
        df = parse_csv_by_category(str(f), DEFAULT_REGEX)
        # 'DRAW' never appears but should be a column (filled with 0 by the function)
        assert "DRAW" in df.columns
        assert df.loc[1990, "DRAW"] == 0

    def test_returned_type_is_dataframe(self, tmp_path):
        f = tmp_path / "data.csv"
        write_annotation_csv(f, [
            ["file", "page", "category"],
            ["arch1985_001.png", "1", "TEXT"],
        ])
        result = parse_csv_by_category(str(f), DEFAULT_REGEX)
        assert isinstance(result, pd.DataFrame)

    def test_counts_are_integer_not_float(self, tmp_path):
        f = tmp_path / "data.csv"
        write_annotation_csv(f, [
            ["file", "page", "category"],
            ["r1990_1.png", "1", "TEXT"],
        ])
        df = parse_csv_by_category(str(f), DEFAULT_REGEX)
        assert df.loc[1990, "TEXT"] == int(df.loc[1990, "TEXT"])


# ════════════════════════════════════════════════════════════════════════════
# plot_stacked_timeline
# ════════════════════════════════════════════════════════════════════════════
class TestPlotStackedTimeline:

    def test_png_file_created(self, tmp_path):
        counts = pd.DataFrame(
            {"TEXT": [10, 15], "DRAW": [3, 5]},
            index=[1985, 1990],
        )
        out = tmp_path / "timeline.png"
        plot_stacked_timeline(counts, output_path=str(out), show=False)
        assert out.exists()

    def test_zero_count_category_does_not_crash(self, tmp_path):
        """A category present in the DataFrame but with all-zero values must
        be quietly excluded from the plot without raising."""
        counts = pd.DataFrame(
            {"TEXT": [5], "DRAW": [0]},
            index=[2000],
        )
        out = tmp_path / "out.png"
        plot_stacked_timeline(counts, output_path=str(out), show=False)
        assert out.exists()

    def test_single_year_single_category_produces_valid_png(self, tmp_path):
        counts = pd.DataFrame({"TEXT": [20]}, index=[2010])
        out = tmp_path / "single.png"
        plot_stacked_timeline(counts, output_path=str(out), show=False)
        assert out.exists()
        assert out.stat().st_size > 0