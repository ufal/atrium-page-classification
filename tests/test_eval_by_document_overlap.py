"""
tests/test_eval_by_document_overlap.py
======================================
Unit tests for data_scripts/eval_by_document_overlap.py (atrium-project#15): accuracy reported
separately for pages in training, pages of documents seen in training, and unseen documents.

No GPU, no trained model, no network required.
"""

import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data_scripts"))

import eval_by_document_overlap as ev  # noqa: E402

DATASETS = """Training set (3 images):
/x/train/TEXT/DOC1-1.png
/x/train/TEXT/DOC1-2.png
/x/train/DRAW/DOC2-01.png
Validation set (1 images):
/x/train/TEXT/DOC3-1.png
Test set (3 images):
/x/train/TEXT/DOC1-3.png
/x/train/DRAW/DOC4-1.png
/x/train/DRAW/DOC4-2.png
"""


def write_results(path: Path, rows: list) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["FILE", "PAGE", "CLASS-1", "SCORE-1", "TRUE"])
        writer.writerows(rows)
    return path


def test_pages_are_grouped_by_their_document_s_training_pages(tmp_path):
    split = tmp_path / "x_FOLD_1_DATASETS.txt"
    split.write_text(DATASETS, encoding="utf-8")
    results = write_results(
        tmp_path / "r.csv",
        [
            ["DOC1", 3, "TEXT", 0.9, "TEXT"],  # held out, its document has training pages
            ["DOC4", 1, "DRAW", 0.9, "DRAW"],  # unseen document
            ["DOC4", 2, "TEXT", 0.6, "DRAW"],  # unseen document, wrong
            ["DOC2", 1, "DRAW", 0.9, "DRAW"],  # a training page ("01" is page 1)
            ["DOC9", 1, "DRAW", 0.9, "DRAW"],  # not in the split
        ],
    )
    result = ev.evaluate(ev.read_split(split), ev.read_results(results))
    assert result == {
        "in training": {"pages": 1, "correct": 1},
        "seen document": {"pages": 1, "correct": 1},
        "unseen document": {"pages": 2, "correct": 1},
        "not in the split": {"pages": 1, "correct": 1},
    }


def test_a_folds_csv_column_is_read_like_a_datasets_file(tmp_path):
    folds = tmp_path / "folds.csv"
    folds.write_text("PNG,fold1,fold2\nDOC1-1.png,train,test\nDOC1-2.png,test,train\n", encoding="utf-8")
    assert ev.read_split(folds, "fold1") == {("DOC1", 1): "train", ("DOC1", 2): "test"}
    with pytest.raises(ValueError, match="--fold"):
        ev.read_split(folds)


def test_main_prints_json(tmp_path, capsys):
    split = tmp_path / "s_DATASETS.txt"
    split.write_text(DATASETS, encoding="utf-8")
    results = write_results(tmp_path / "r.csv", [["DOC4", 1, "DRAW", 0.9, "DRAW"]])
    assert ev.main(["--split", str(split), "--results", str(results), "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {str(results): {"unseen document": {"pages": 1, "correct": 1}}}


STORED_SPLIT = ROOT / "result" / "stats" / "20251005-0434_model_v93_FOLD_1_DATASETS.txt"
STORED_EVAL = ROOT / "result" / "tables" / "20251020-1849_5449_model_v43_TOP-1_EVAL.csv"


@pytest.mark.skipif(not (STORED_SPLIT.is_file() and STORED_EVAL.is_file()), reason="stored results not present")
def test_the_stored_5449_page_set_overlaps_fold_1_training_as_reported():
    """The numbers of atrium-project's 15.revision_brief.md, section 1."""
    result = ev.evaluate(ev.read_split(STORED_SPLIT), ev.read_results(STORED_EVAL))
    assert result["in training"]["pages"] == 4296
    assert sum(group["pages"] for group in result.values()) == 5449
