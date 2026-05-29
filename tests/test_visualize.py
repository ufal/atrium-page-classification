"""
tests/test_visualize.py
=======================
Unit tests for supplement_scripts/visualize.py.

Scope
-----
* get_model_type   – model-family detection from name prefix (pure function)
* short_model_name – display label stripping (pure function)
* plot_comparison  – PNG output creation, missing-column validation

Matplotlib uses the non-interactive Agg backend (activated in conftest.py).
No GPU, no trained model, no network required.
"""
import csv
from pathlib import Path

import pytest

from visualize import get_model_type, plot_comparison, short_model_name


# ── helpers ────────────────────────────────────────────────────────────────

def write_model_csv(path: Path, rows: list) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)


# ════════════════════════════════════════════════════════════════════════════
# get_model_type
# ════════════════════════════════════════════════════════════════════════════
class TestGetModelType:
    """get_model_type(name) maps a model identifier to its family label."""

    def test_vit_lowercase_prefix(self):
        assert get_model_type("vit-base-patch16-224") == "ViT"

    def test_vit_uppercase_prefix(self):
        assert get_model_type("Vit-large-patch16-384") == "ViT"

    def test_regnety_prefix(self):
        assert get_model_type("regnety_160.swag_ft_in1k") == "RegNetY"

    def test_efficientnetv2_tf_prefix(self):
        assert get_model_type("tf_efficientnetv2_m.in21k_ft_in1k") == "EffNetV2"

    def test_efficientnetv2_named_prefix(self):
        assert get_model_type("EffNet-v2-small") == "EffNetV2"

    def test_dit_lowercase_prefix(self):
        assert get_model_type("dit-base-finetuned-rvlcdip") == "DiT"

    def test_clip_prefix(self):
        assert get_model_type("CLIP-ViT-B/16") == "CLIP"

    def test_unknown_model_returns_other(self):
        assert get_model_type("some-unknown-model-xyz") == "Other"

    def test_empty_string_returns_other(self):
        assert get_model_type("") == "Other"


# ════════════════════════════════════════════════════════════════════════════
# short_model_name
# ════════════════════════════════════════════════════════════════════════════
class TestShortModelName:
    """short_model_name(name) strips the known type prefix for display."""

    def test_strips_vit_prefix(self):
        assert short_model_name("vit-base-patch16-224") == "base-patch16-224"

    def test_strips_regnety_prefix(self):
        assert short_model_name("regnety_160.swag_ft_in1k") == "160.swag_ft_in1k"

    def test_strips_tf_efficientnetv2_prefix(self):
        result = short_model_name("tf_efficientnetv2_m.in21k_ft_in1k")
        assert result == "m.in21k_ft_in1k"

    def test_unknown_prefix_returned_unchanged(self):
        assert short_model_name("unknown-model-abc") == "unknown-model-abc"


# ════════════════════════════════════════════════════════════════════════════
# plot_comparison
# ════════════════════════════════════════════════════════════════════════════
class TestPlotComparison:
    """plot_comparison(csv, png, title, show) writes a scatter-plot PNG."""

    def test_png_created_for_valid_csv(self, tmp_path):
        f = tmp_path / "models.csv"
        write_model_csv(f, [
            ["model", "param", "acc"],
            ["vit-base-patch16-224",   "87",  "98.79"],
            ["regnety_160.swag_ft_in1k", "84", "99.17"],
            ["vit-large-patch16-384", "305", "99.12"],
        ])
        out = tmp_path / "chart.png"
        plot_comparison(str(f), str(out), title="Test Chart", show=False)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_missing_model_column_raises_value_error(self, tmp_path):
        f = tmp_path / "bad.csv"
        write_model_csv(f, [["param", "acc"], ["84", "99.17"]])
        with pytest.raises(ValueError, match="missing required column"):
            plot_comparison(str(f), str(tmp_path / "out.png"),
                            title="Test", show=False)

    def test_missing_param_column_raises_value_error(self, tmp_path):
        f = tmp_path / "bad.csv"
        write_model_csv(f, [["model", "acc"], ["vit", "98.0"]])
        with pytest.raises(ValueError):
            plot_comparison(str(f), str(tmp_path / "out.png"),
                            title="Test", show=False)

    def test_missing_acc_column_raises_value_error(self, tmp_path):
        f = tmp_path / "bad.csv"
        write_model_csv(f, [["model", "param"], ["vit", "87"]])
        with pytest.raises(ValueError):
            plot_comparison(str(f), str(tmp_path / "out.png"),
                            title="Test", show=False)

    def test_single_row_does_not_raise(self, tmp_path):
        """One data point cannot produce a trendline, but must not crash."""
        f = tmp_path / "single.csv"
        write_model_csv(f, [
            ["model", "param", "acc"],
            ["vit-base-patch16-224", "87", "98.79"],
        ])
        out = tmp_path / "single.png"
        plot_comparison(str(f), str(out), title="Solo", show=False)
        assert out.exists()

    def test_multiple_model_families_all_plotted(self, tmp_path):
        """One row per family — each should appear without errors."""
        f = tmp_path / "multi.csv"
        write_model_csv(f, [
            ["model", "param", "acc"],
            ["vit-base-patch16-224",      "87",  "98.79"],
            ["regnety_160.swag_ft_in1k",  "84",  "99.17"],
            ["tf_efficientnetv2_m.in21k", "54",  "98.83"],
            ["dit-base-finetuned-rvlcdip","86",  "98.72"],
        ])
        out = tmp_path / "multi.png"
        plot_comparison(str(f), str(out), title="Multi-family", show=False)
        assert out.exists()