"""
tests/test_folds_split.py
=========================
Unit tests for the explicit dataset-folds support added for issue #15.

Scope
-----
* split_data_from_folds        – read an explicit train/dev/test split from a folds CSV
                                 (classifier.py), including the "pages absent from the CSV are
                                 dropped" behaviour that removes the reduced-dataset pages and
                                 the safe_check=False path used by `--eval` fold-subset selection.
* REVISION_BEST_FOLDS mapping  – every retraining revision resolves to a fold column.
* base-model resolution        – the v*.4 entries precede the generic single-dot prefixes, so
                                 run.py's `next(... startswith ...)` picks the correct base model.
* resolve_fold_column          – the run.py helper shared by the train and eval branches.

No GPU, no trained model, and no network access required.
"""

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from classifier import split_data_from_folds
from model_registry import REVISION_BEST_FOLDS, REVISION_TO_BASE_MODEL


# ── helpers ──────────────────────────────────────────────────────────────────
def _write_folds_csv(path, rows, columns=("fold1", "fold2")):
    """rows: list of (png_name, {col: value}). Writes a minimal folds CSV."""
    records = []
    for png, folds in rows:
        rec = {"PNG": png, "categ": "DRAW"}
        rec.update(folds)
        records.append(rec)
    pd.DataFrame(records, columns=["PNG", "categ", *columns]).to_csv(path, index=False)
    return str(path)


def _onehot(i, n=2):
    v = np.zeros(n)
    v[i] = 1
    return v


# ── split_data_from_folds: routing ───────────────────────────────────────────
def test_routes_files_by_fold_column(tmp_path):
    csv = _write_folds_csv(
        tmp_path / "folds.csv",
        [
            ("a.png", {"fold1": "train", "fold2": "test"}),
            ("b.png", {"fold1": "dev", "fold2": "train"}),
            ("c.png", {"fold1": "test", "fold2": "dev"}),
        ],
    )
    # On-disk paths can be anywhere; matching is by basename against the PNG column.
    files = ["/data/DRAW/a.png", "/data/DRAW/b.png", "/data/DRAW/c.png"]
    labels = [_onehot(0), _onehot(1), _onehot(0)]

    tr_f, va_f, te_f, tr_l, va_l, te_l = split_data_from_folds(files, labels, csv, "fold1", safe_check=False)

    assert [f.rsplit("/", 1)[-1] for f in tr_f] == ["a.png"]
    assert [f.rsplit("/", 1)[-1] for f in va_f] == ["b.png"]
    assert [f.rsplit("/", 1)[-1] for f in te_f] == ["c.png"]
    # labels travel with their files
    assert np.array_equal(tr_l[0], _onehot(0))
    assert np.array_equal(va_l[0], _onehot(1))


def test_different_columns_give_different_partitions(tmp_path):
    csv = _write_folds_csv(
        tmp_path / "folds.csv",
        [
            ("a.png", {"fold1": "train", "fold2": "test"}),
            ("b.png", {"fold1": "dev", "fold2": "train"}),
            ("c.png", {"fold1": "test", "fold2": "dev"}),
        ],
    )
    files = ["a.png", "b.png", "c.png"]
    labels = [_onehot(0)] * 3

    _, _, te1, *_ = split_data_from_folds(files, labels, csv, "fold1", safe_check=False)
    _, _, te2, *_ = split_data_from_folds(files, labels, csv, "fold2", safe_check=False)
    assert list(te1) == ["c.png"]
    assert list(te2) == ["a.png"]


def test_files_absent_from_csv_are_dropped(tmp_path):
    """The reduced-dataset behaviour: on-disk pages not listed in the CSV are excluded entirely."""
    csv = _write_folds_csv(
        tmp_path / "folds.csv",
        [("a.png", {"fold1": "train"}), ("b.png", {"fold1": "test"})],
        columns=("fold1",),
    )
    files = ["a.png", "b.png", "removed_page.png"]  # third is NOT in the CSV
    labels = [_onehot(0)] * 3

    tr_f, va_f, te_f, *_ = split_data_from_folds(files, labels, csv, "fold1", safe_check=False)
    kept = set(tr_f) | set(va_f) | set(te_f)
    assert kept == {"a.png", "b.png"}
    assert "removed_page.png" not in kept


def test_fold_value_whitespace_and_case_insensitive(tmp_path):
    csv = _write_folds_csv(
        tmp_path / "folds.csv",
        [("a.png", {"fold1": " Test "}), ("b.png", {"fold1": "TRAIN"}), ("c.png", {"fold1": "Dev"})],
        columns=("fold1",),
    )
    files = ["a.png", "b.png", "c.png"]
    labels = [_onehot(0)] * 3
    tr_f, va_f, te_f, *_ = split_data_from_folds(files, labels, csv, "fold1", safe_check=False)
    assert list(te_f) == ["a.png"]
    assert list(tr_f) == ["b.png"]
    assert list(va_f) == ["c.png"]


def test_unknown_fold_value_is_skipped(tmp_path):
    csv = _write_folds_csv(
        tmp_path / "folds.csv",
        [("a.png", {"fold1": "train"}), ("b.png", {"fold1": "holdout"})],  # "holdout" is unknown
        columns=("fold1",),
    )
    files = ["a.png", "b.png"]
    labels = [_onehot(0)] * 2
    tr_f, va_f, te_f, *_ = split_data_from_folds(files, labels, csv, "fold1", safe_check=False)
    kept = set(tr_f) | set(va_f) | set(te_f)
    assert kept == {"a.png"}  # b.png with unknown value dropped, no crash


# ── eval fold-subset behaviour ───────────────────────────────────────────────
def test_eval_subset_is_test_bucket(tmp_path):
    """Mirrors exactly the call run.py makes for `--eval --folds_csv`: keep only the test bucket."""
    csv = _write_folds_csv(
        tmp_path / "folds.csv",
        [
            ("a.png", {"fold2": "train"}),
            ("b.png", {"fold2": "test"}),
            ("c.png", {"fold2": "dev"}),
            ("d.png", {"fold2": "test"}),
        ],
        columns=("fold2",),
    )
    files = ["a.png", "b.png", "c.png", "d.png"]
    labels = [_onehot(0)] * 4
    _, _, testfiles, _, _, testLabels = split_data_from_folds(files, labels, csv, "fold2", safe_check=False)
    assert sorted(testfiles) == ["b.png", "d.png"]
    assert len(testLabels) == 2


# ── safe_check with real images ──────────────────────────────────────────────
def test_safe_check_drops_corrupted_images(tmp_path):
    good = tmp_path / "good.png"
    Image.new("RGB", (8, 8), "white").save(good)
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not really a png")

    csv = _write_folds_csv(
        tmp_path / "folds.csv",
        [("good.png", {"fold1": "train"}), ("bad.png", {"fold1": "train"})],
        columns=("fold1",),
    )
    tr_f, *_ = split_data_from_folds([str(good), str(bad)], [_onehot(0), _onehot(0)], csv, "fold1", safe_check=True)
    assert [f.rsplit("/", 1)[-1] for f in tr_f] == ["good.png"]


# ── error handling ───────────────────────────────────────────────────────────
def test_missing_csv_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        split_data_from_folds(["a.png"], [_onehot(0)], str(tmp_path / "nope.csv"), "fold1", safe_check=False)


def test_missing_png_column_raises(tmp_path):
    csv = tmp_path / "bad.csv"
    pd.DataFrame([{"file": "a", "fold1": "train"}]).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="PNG"):
        split_data_from_folds(["a.png"], [_onehot(0)], str(csv), "fold1", safe_check=False)


def test_missing_fold_column_raises(tmp_path):
    csv = _write_folds_csv(tmp_path / "folds.csv", [("a.png", {"fold1": "train"})], columns=("fold1",))
    with pytest.raises(ValueError, match="fold9"):
        split_data_from_folds(["a.png"], [_onehot(0)], csv, "fold9", safe_check=False)


# ── registry: per-model fold + base-model resolution ─────────────────────────
def _resolve_base(revision):
    """Replicates run.py's resolution: first matching prefix wins (insertion order)."""
    return REVISION_TO_BASE_MODEL[next(k for k in REVISION_TO_BASE_MODEL if revision.startswith(k))]


@pytest.mark.parametrize(
    "revision,expected_base,expected_fold",
    [
        ("v1.4", "timm/tf_efficientnetv2_m.in21k_ft_in1k", "fold1"),
        ("v2.4", "google/vit-base-patch16-224", "fold5"),
        ("v3.4", "google/vit-base-patch16-384", "fold2"),
        ("v4.4", "timm/regnety_160.swag_ft_in1k", "fold1"),
        ("v5.4", "google/vit-large-patch16-384", "fold2"),
    ],
)
def test_revision_resolves_to_base_and_fold(revision, expected_base, expected_fold):
    # ordering caveat: v4.4 must resolve to regnety_160, NOT the generic "v4." effnetv2_l entry
    assert _resolve_base(revision) == expected_base
    assert REVISION_BEST_FOLDS[revision] == expected_fold


def test_all_best_fold_revisions_are_registered():
    for revision in REVISION_BEST_FOLDS:
        # every retraining revision must have a base-model entry resolvable by run.py
        assert _resolve_base(revision) is not None


# ── run.py helper ────────────────────────────────────────────────────────────
def test_resolve_fold_column_helper():
    from run import resolve_fold_column

    assert resolve_fold_column("v2.4", None) == "fold5"  # per-revision default
    assert resolve_fold_column("v2.4", "fold3") == "fold3"  # explicit override wins
    with pytest.raises(ValueError):
        resolve_fold_column("v99.9", None)  # unknown revision, no explicit column


# ── optional sanity check against the real (large) folds CSV ──────────────────
@pytest.mark.slow
def test_real_folds_csv_counts_match(tmp_path):
    """If the licensed folds CSV is staged, bucket sizes must match a direct column tally."""
    import os

    csv_path = os.environ.get("FOLDS_CSV")
    if not csv_path or not os.path.isfile(csv_path):
        pytest.skip("set FOLDS_CSV to the licensed folds CSV to run this check")

    df = pd.read_csv(csv_path)
    files = list(df["PNG"].astype(str))
    labels = [_onehot(0)] * len(files)
    tr_f, va_f, te_f, *_ = split_data_from_folds(files, labels, csv_path, "fold1", safe_check=False)
    counts = df["fold1"].astype(str).str.strip().str.lower().value_counts()
    assert len(tr_f) == counts.get("train", 0)
    assert len(va_f) == counts.get("dev", 0)
    assert len(te_f) == counts.get("test", 0)
