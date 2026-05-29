"""
tests/test_averaging.py
=======================
Unit tests for supplement_scripts/averaging.py.

Scope
-----
* load_and_melt – standard CLASS-N / SCORE-N format
* load_and_melt – wide multi-model format (CLASS-1-v1.3 …)
* load_and_melt – CATEGORY column fallback (top_N=1 output from utils.py)
* load_and_melt – filename normalisation and --no-normalize
* Aggregation helpers: intersection, score averaging, top-N ranking

No GPU, no trained model, no network required.
"""
import textwrap
from pathlib import Path

import pandas as pd
import pytest

# supplement_scripts/ is on sys.path via conftest.py
from averaging import load_and_melt


# ── helpers ────────────────────────────────────────────────────────────────

def write_csv(tmp_path: Path, name: str, content: str) -> Path:
    """Write dedented CSV content to a file and return the path."""
    p = tmp_path / name
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


# ════════════════════════════════════════════════════════════════════════════
# Standard CLASS-N / SCORE-N format
# ════════════════════════════════════════════════════════════════════════════
class TestLoadAndMeltStandard:
    """Single-model CSVs with the canonical CLASS-N / SCORE-N column layout."""

    def test_single_file_counted_as_one_model(self, tmp_path):
        write_csv(tmp_path, "model_v43_TOP-3.csv", """\
            FILE,PAGE,CLASS-1,SCORE-1,CLASS-2,SCORE-2,CLASS-3,SCORE-3
            doc,1,TEXT,0.90,LINE_P,0.08,DRAW,0.02
        """)
        _, _, _, count = load_and_melt([str(tmp_path / "model_v43_TOP-3.csv")])
        assert count == 1

    def test_melted_df_has_class_and_score_columns(self, tmp_path):
        f = write_csv(tmp_path, "m.csv", """\
            FILE,PAGE,CLASS-1,SCORE-1
            doc,1,TEXT,1.0
        """)
        long_dfs, _, _, _ = load_and_melt([str(f)])
        assert "CLASS" in long_dfs[0].columns
        assert "SCORE" in long_dfs[0].columns

    def test_two_files_counted_as_two_models(self, tmp_path):
        for n in (1, 2):
            write_csv(tmp_path, f"m{n}.csv", """\
                FILE,PAGE,CLASS-1,SCORE-1
                doc,1,TEXT,1.0
            """)
        _, _, _, count = load_and_melt([str(p) for p in tmp_path.glob("*.csv")])
        assert count == 2

    def test_category_column_accepted_as_class1_fallback(self, tmp_path):
        """top_N=1 output from utils.py writes CATEGORY rather than CLASS-1."""
        f = write_csv(tmp_path, "top1.csv", """\
            FILE,PAGE,CATEGORY
            doc,1,TEXT
        """)
        _, _, _, count = load_and_melt([str(f)])
        assert count == 1

    def test_csv_without_any_class_column_is_skipped(self, tmp_path):
        f = write_csv(tmp_path, "bad.csv", """\
            FILE,PAGE,SCORE-1
            doc,1,0.9
        """)
        long_dfs, _, _, count = load_and_melt([str(f)])
        assert count == 0
        assert long_dfs is None

    def test_hyphens_normalised_to_underscores_by_default(self, tmp_path):
        f = write_csv(tmp_path, "m.csv", """\
            FILE,PAGE,CLASS-1,SCORE-1
            my-doc,1,TEXT,1.0
        """)
        long_dfs, _, _, _ = load_and_melt([str(f)], normalize_filenames=True)
        assert "my_doc" in long_dfs[0]["FILE"].values

    def test_no_normalize_preserves_hyphens(self, tmp_path):
        f = write_csv(tmp_path, "m.csv", """\
            FILE,PAGE,CLASS-1,SCORE-1
            my-doc,1,TEXT,1.0
        """)
        long_dfs, _, _, _ = load_and_melt([str(f)], normalize_filenames=False)
        assert "my-doc" in long_dfs[0]["FILE"].values

    def test_glob_pattern_expands_to_multiple_files(self, tmp_path):
        for i in (1, 2, 3):
            write_csv(tmp_path, f"m{i}.csv", """\
                FILE,PAGE,CLASS-1,SCORE-1
                doc,1,TEXT,1.0
            """)
        _, _, _, count = load_and_melt([str(tmp_path / "*.csv")])
        assert count == 3

    def test_top1_column_captured_in_top1_dfs(self, tmp_path):
        f = write_csv(tmp_path, "model_v43_TOP-1.csv", """\
            FILE,PAGE,CLASS-1,SCORE-1
            doc,1,TEXT,1.0
        """)
        _, _, top1_dfs, _ = load_and_melt([str(f)])
        assert len(top1_dfs) == 1

    def test_nonexistent_path_returns_empty_result(self, tmp_path):
        long_dfs, _, _, count = load_and_melt([str(tmp_path / "missing.csv")])
        assert count == 0
        assert long_dfs is None

    def test_top3_produces_three_rank_entries_per_page(self, tmp_path):
        f = write_csv(tmp_path, "m.csv", """\
            FILE,PAGE,CLASS-1,SCORE-1,CLASS-2,SCORE-2,CLASS-3,SCORE-3
            doc,1,TEXT,0.8,LINE_P,0.1,DRAW,0.1
        """)
        long_dfs, _, _, _ = load_and_melt([str(f)])
        page_rows = long_dfs[0][(long_dfs[0]["FILE"] == "doc") & (long_dfs[0]["PAGE"] == 1)]
        assert len(page_rows) == 3


# ════════════════════════════════════════════════════════════════════════════
# Wide multi-model format  (CLASS-1-v1.3, CLASS-1-v2.3 …)
# ════════════════════════════════════════════════════════════════════════════
class TestLoadAndMeltWideFormat:
    """BEST_N_models_TOP-1 CSV with one column per model and no SCORE columns."""

    WIDE_CSV = """\
        FILE,PAGE,CLASS-1-v1.3,CLASS-1-v2.3,CLASS-1-v3.3,CLASS-1-v4.3,CLASS-1-v5.3
        atrium,1,TEXT_P,TEXT,LINE_P,TEXT,TEXT
        atrium,2,LINE_P,LINE_P,LINE_P,LINE_P,LINE_P
    """

    def test_five_model_columns_detected_as_five_virtual_inputs(self, tmp_path):
        f = write_csv(tmp_path, "BEST_5_models_TOP-1.csv", self.WIDE_CSV)
        _, _, _, count = load_and_melt([str(f)])
        assert count == 5

    def test_five_top1_df_entries_one_per_model_column(self, tmp_path):
        f = write_csv(tmp_path, "BEST_5_models_TOP-1.csv", self.WIDE_CSV)
        _, _, top1_dfs, _ = load_and_melt([str(f)])
        assert len(top1_dfs) == 5

    def test_display_names_have_uppercase_first_letter(self, tmp_path):
        f = write_csv(tmp_path, "BEST_5_models_TOP-1.csv", self.WIDE_CSV)
        _, _, top1_dfs, _ = load_and_melt([str(f)])
        names = {col for t in top1_dfs for col in t.columns if col not in ("FILE", "PAGE")}
        assert all(n[0].isupper() for n in names)

    def test_majority_vote_probability_correct_for_page1(self, tmp_path):
        """(atrium, 1): TEXT×3, TEXT_P×1, LINE_P×1 → TEXT gets AVG = 3/5 = 0.6."""
        f = write_csv(tmp_path, "BEST_5_models_TOP-1.csv", self.WIDE_CSV)
        long_dfs, file_page_sets, _, num_models = load_and_melt([str(f)])

        common = set.intersection(*file_page_sets)
        combined = pd.concat(long_dfs, ignore_index=True)
        mask = pd.Series(list(zip(combined["FILE"], combined["PAGE"]))).isin(common)
        combined = combined[mask.values]

        grouped = combined.groupby(["FILE", "PAGE", "CLASS"])["SCORE"].sum().reset_index()
        grouped["AVG"] = grouped["SCORE"] / num_models

        text_avg = grouped[
            (grouped["FILE"] == "atrium") &
            (grouped["PAGE"] == 1) &
            (grouped["CLASS"] == "TEXT")
        ]["AVG"].values
        assert len(text_avg) == 1
        assert text_avg[0] == pytest.approx(0.6, abs=1e-6)

    def test_unanimous_page_gives_avg_score_1(self, tmp_path):
        """(atrium, 2): all 5 models predict LINE_P → AVG = 1.0."""
        f = write_csv(tmp_path, "BEST_5_models_TOP-1.csv", self.WIDE_CSV)
        long_dfs, file_page_sets, _, num_models = load_and_melt([str(f)])

        combined = pd.concat(long_dfs, ignore_index=True)
        grouped = combined.groupby(["FILE", "PAGE", "CLASS"])["SCORE"].sum().reset_index()
        grouped["AVG"] = grouped["SCORE"] / num_models

        line_avg = grouped[
            (grouped["PAGE"] == 2) & (grouped["CLASS"] == "LINE_P")
        ]["AVG"].values
        assert line_avg[0] == pytest.approx(1.0, abs=1e-6)

    def test_file_page_sets_length_equals_model_count(self, tmp_path):
        """Each virtual model gets its own entry in file_page_sets."""
        f = write_csv(tmp_path, "BEST_5_models_TOP-1.csv", self.WIDE_CSV)
        _, file_page_sets, _, count = load_and_melt([str(f)])
        assert len(file_page_sets) == count == 5

    def test_all_virtual_models_cover_same_pages(self, tmp_path):
        f = write_csv(tmp_path, "BEST_5_models_TOP-1.csv", self.WIDE_CSV)
        _, file_page_sets, _, _ = load_and_melt([str(f)])
        assert all(s == file_page_sets[0] for s in file_page_sets)

    def test_wide_and_standard_combined_yields_correct_total(self, tmp_path):
        """6 total virtual models: 5 from the wide file + 1 standard CSV."""
        write_csv(tmp_path, "BEST_5_models_TOP-1.csv", self.WIDE_CSV)
        write_csv(tmp_path, "model_v13_TOP-1.csv", """\
            FILE,PAGE,CLASS-1,SCORE-1
            atrium,1,TEXT,1.0
            atrium,2,LINE_P,1.0
        """)
        # Fix ordering so the wide file is always processed first
        files = [
            str(tmp_path / "BEST_5_models_TOP-1.csv"),
            str(tmp_path / "model_v13_TOP-1.csv"),
        ]
        _, _, top1_dfs, count = load_and_melt(files)
        assert count == 6
        assert len(top1_dfs) == 6

    def test_collision_guard_prevents_duplicate_display_names_within_wide_file(self, tmp_path):
        """Two columns that map to the same display name must be disambiguated."""
        # Craft a synthetic wide CSV where two columns share the same model_id
        f = write_csv(tmp_path, "BEST_2_models_TOP-1.csv", """\
            FILE,PAGE,CLASS-1-v4.3,CLASS-2-v4.3
            doc,1,TEXT,LINE_P
        """)
        _, _, top1_dfs, _ = load_and_melt([str(f)])
        names = [col for t in top1_dfs for col in t.columns if col not in ("FILE", "PAGE")]
        # Both columns must produce distinct display names (collision guard)
        assert len(names) == len(set(names))


# ════════════════════════════════════════════════════════════════════════════
# Intersection and aggregation integration
# ════════════════════════════════════════════════════════════════════════════
class TestIntersectionAndAggregation:
    """Downstream set-intersection and score-averaging using load_and_melt output."""

    def test_page_absent_from_one_model_excluded_from_intersection(self, tmp_path):
        write_csv(tmp_path, "m1.csv", """\
            FILE,PAGE,CLASS-1,SCORE-1
            doc,1,TEXT,1.0
            doc,2,LINE_P,1.0
        """)
        write_csv(tmp_path, "m2.csv", """\
            FILE,PAGE,CLASS-1,SCORE-1
            doc,1,TEXT,1.0
        """)
        _, file_page_sets, _, _ = load_and_melt(
            [str(p) for p in sorted(tmp_path.glob("*.csv"))]
        )
        common = set.intersection(*file_page_sets)
        assert ("doc", 1) in common
        assert ("doc", 2) not in common

    def test_scores_averaged_across_two_models(self, tmp_path):
        write_csv(tmp_path, "m1.csv", """\
            FILE,PAGE,CLASS-1,SCORE-1
            doc,1,TEXT,0.8
        """)
        write_csv(tmp_path, "m2.csv", """\
            FILE,PAGE,CLASS-1,SCORE-1
            doc,1,TEXT,0.6
        """)
        long_dfs, _, _, num = load_and_melt(
            [str(p) for p in sorted(tmp_path.glob("*.csv"))]
        )
        combined = pd.concat(long_dfs, ignore_index=True)
        grouped = combined.groupby(["FILE", "PAGE", "CLASS"])["SCORE"].sum().reset_index()
        grouped["AVG"] = grouped["SCORE"] / num
        avg = grouped[grouped["CLASS"] == "TEXT"]["AVG"].values[0]
        assert avg == pytest.approx(0.7, abs=1e-6)

    def test_competing_classes_ranked_by_score(self, tmp_path):
        """Model 1 votes TEXT, model 2 votes DRAW → TEXT(0.5) > DRAW(0.5) is a tie;
        but 2×TEXT vs 1×DRAW gives TEXT(0.67) > DRAW(0.33)."""
        write_csv(tmp_path, "m1.csv", """\
            FILE,PAGE,CLASS-1,SCORE-1
            doc,1,TEXT,1.0
        """)
        write_csv(tmp_path, "m2.csv", """\
            FILE,PAGE,CLASS-1,SCORE-1
            doc,1,TEXT,1.0
        """)
        write_csv(tmp_path, "m3.csv", """\
            FILE,PAGE,CLASS-1,SCORE-1
            doc,1,DRAW,1.0
        """)
        long_dfs, _, _, num = load_and_melt(
            [str(p) for p in sorted(tmp_path.glob("*.csv"))]
        )
        combined = pd.concat(long_dfs, ignore_index=True)
        grouped = combined.groupby(["FILE", "PAGE", "CLASS"])["SCORE"].sum().reset_index()
        grouped["AVG"] = grouped["SCORE"] / num
        grouped.sort_values("AVG", ascending=False, inplace=True)
        assert grouped.iloc[0]["CLASS"] == "TEXT"
        assert grouped.iloc[0]["AVG"] == pytest.approx(2 / 3, abs=1e-6)