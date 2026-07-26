"""
tests/test_ensemble.py – Unit tests for ensemble.py, the softmax-averaging
helpers used by parallel_best.py (batch CSVs) and the FastAPI service
(in-memory JSON).

Distinct from test_averaging.py, which covers the standalone
supplementary/scripts/averaging.py CLI loader.
"""

import sys
from unittest.mock import MagicMock

# Mock atrium_document before any project imports to prevent collection errors
if "atrium_document" not in sys.modules:
    sys.modules["atrium_document"] = MagicMock()

import pandas as pd
import pytest

from ensemble import average_prediction_dicts, average_rdfs

CATS = ["TEXT", "PHOTO", "DRAW"]


# ── average_prediction_dicts (service path) ─────────────────────────────────
def test_prediction_dicts_empty_returns_empty():
    assert average_prediction_dicts([], CATS, top_n=3) == []


def test_prediction_dicts_averages_and_ranks():
    m1 = [{"label": "TEXT", "score": 0.8}, {"label": "PHOTO", "score": 0.2}, {"label": "DRAW", "score": 0.0}]
    m2 = [{"label": "TEXT", "score": 0.6}, {"label": "PHOTO", "score": 0.4}, {"label": "DRAW", "score": 0.0}]

    result = average_prediction_dicts([m1, m2], CATS, top_n=2)

    assert [r["label"] for r in result] == ["TEXT", "PHOTO"]
    assert result[0]["score"] == pytest.approx(0.7)
    assert result[1]["score"] == pytest.approx(0.3)


def test_prediction_dicts_top_n_truncates():
    m1 = [{"label": c, "score": s} for c, s in zip(CATS, [0.5, 0.3, 0.2])]
    result = average_prediction_dicts([m1], CATS, top_n=1)
    assert len(result) == 1
    assert result[0]["label"] == "TEXT"


def test_prediction_dicts_score_clamped_to_one():
    # A single model whose scores already sum >1 must never exceed 1.0 per label.
    m1 = [{"label": "TEXT", "score": 1.4}, {"label": "PHOTO", "score": 0.1}, {"label": "DRAW", "score": 0.0}]
    result = average_prediction_dicts([m1], CATS, top_n=3)
    assert max(r["score"] for r in result) <= 1.0


# ── average_rdfs (batch CSV path) ───────────────────────────────────────────
def _rdf(cls, score):
    return pd.DataFrame({"FILE": ["f"], "PAGE": [1], "CLASS-1": [cls], "SCORE-1": [score]})


def test_rdfs_empty_returns_empty_frame_with_expected_columns():
    out = average_rdfs({}, top_N=3)
    assert len(out) == 0
    for col in ["FILE", "PAGE", "CLASS-1", "CLASS-2", "CLASS-3", "SCORE-1", "SCORE-2", "SCORE-3"]:
        assert col in out.columns


def test_rdfs_averages_scores_across_models():
    out = average_rdfs({"v4.3": _rdf("TEXT", 0.8), "v5.3": _rdf("TEXT", 0.6)}, top_N=1)
    row = out.to_dict("records")[0]
    assert row["CLASS-1"] == "TEXT"
    assert row["SCORE-1"] == pytest.approx(0.7)


def test_rdfs_adds_one_vote_column_per_revision():
    out = average_rdfs({"v4.3": _rdf("TEXT", 0.8), "v5.3": _rdf("PHOTO", 0.9)}, top_N=1)
    # _vote_col_name upper-cases the first character of the revision.
    assert "V4.3" in out.columns
    assert "V5.3" in out.columns
    assert out.iloc[0]["V4.3"] == "TEXT"
    assert out.iloc[0]["V5.3"] == "PHOTO"


def test_rdfs_avg_score_clamped_to_one():
    # Same class from two models with scores summing well above num_models.
    out = average_rdfs({"v4.3": _rdf("TEXT", 1.0), "v5.3": _rdf("TEXT", 1.0)}, top_N=1)
    assert out.iloc[0]["SCORE-1"] <= 1.0
