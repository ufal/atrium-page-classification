"""Tests for data_scripts/collect_evaluation.py — a run's outputs kept apart with their provenance
(atrium-project#15). The run is a fixture: the lists eval_vX4_job.sh writes, and two tables."""

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_scripts"))
import collect_evaluation as ce  # noqa: E402

STAMP = "20261022-0909"


@pytest.fixture
def run(tmp_path, monkeypatch):
    """A results folder after one run: an old vX.3 table, the run's two tables, its lists and a folds CSV."""
    monkeypatch.chdir(tmp_path)
    results = Path("result")
    (results / "tables").mkdir(parents=True)
    (results / "stats").mkdir()
    (results / "paradata").mkdir()
    old = results / "tables" / "20251020-1849_5449_model_v43_TOP-1_EVAL.csv"
    old.write_text("FILE,PAGE,CATEGORY,TRUE\nCTX1,1,TEXT,TEXT\n", encoding="utf-8")
    test_table = results / "tables" / "20261022-0924_model_v14_TOP-1_EVAL.csv"
    test_table.write_text("FILE,PAGE,CATEGORY,TRUE\nCTX1,1,TEXT,TEXT\nCTX1,2,DRAW,TEXT\n", encoding="utf-8")
    set_table = results / "tables" / "20261022-1000_5449_model_v14_TOP-1_EVAL.csv"
    set_table.write_text("FILE,PAGE,CATEGORY,TRUE\nCTX9,1,TEXT,TEXT\n", encoding="utf-8")
    (results / "paradata" / "261022-071106_page-classification.json").write_text("{}", encoding="utf-8")

    folds = tmp_path / "licensed_crossval_folds.csv"
    with folds.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["PNG", "categ", "fold1"])
        writer.writerows(
            [["CTX1-01.png", "TEXT", "test"], ["CTX1-02.png", "TEXT", "test"], ["CTX1-03.png", "TEXT", "test"]]
            + [["", "TEXT", "test"], ["", "DRAW", "test"], ["CTX2-01.png", "TEXT", "train"]]
        )
    stats = results / "stats"
    (stats / f"{STAMP}_vX4_steps.txt").write_text(
        f"vX.4 evaluation, job 7271549 on dll-8gpu2, started now\ncommit abc123 (v1.9.8-beta-1-gf14bc4b)\n"
        f"version v1.9.8-beta OK\nfolds CSV {folds}\n\nOK      stage fold1\nOK      v1.4 fold1 test top-1\n"
        "FAILED  v1.4 5,449-page set top-1 (exit 1)\n",
        encoding="utf-8",
    )
    (stats / f"{STAMP}_vX4_tables.tsv").write_text(
        "set\trevision\tfold\ttop_n\ttable\n"
        f"fold1 test\tv1.4\tfold1\t1\t./{test_table}\n5,449-page set\tv1.4\tfold1\t1\t./{set_table}\n",
        encoding="utf-8",
    )
    (stats / f"{STAMP}_vX4_summary.md").write_text("# summary\n", encoding="utf-8")
    (stats / "20261022-0901_vX4_hub_revisions.csv").write_text(
        "revision,base_model,fold,hub_repo,hub_commit\nv1.4,timm/x,fold1,ufal/vit-historical-page,93f2dc7\n",
        encoding="utf-8",
    )
    (stats / f"{STAMP}_vX4_outputs.txt").write_text(
        "\n".join(
            f"./{p}" for p in (test_table, set_table, results / "paradata" / "261022-071106_page-classification.json")
        )
        + "\n",
        encoding="utf-8",
    )
    return results, folds, old


def test_the_run_moves_apart_and_the_original_tables_stay(run):
    results, folds, old = run
    dest = ce.collect(results, STAMP, results / "evaluations", folds)
    assert dest == results / "evaluations" / f"{STAMP}_vX4"
    assert old.is_file(), "a vX.3 file must never move"
    assert sorted(p.name for p in (results / "tables").iterdir()) == [old.name]
    assert (dest / "tables" / "20261022-0924_model_v14_TOP-1_EVAL.csv").is_file()
    assert (dest / "paradata" / "261022-071106_page-classification.json").is_file()
    assert (dest / "stats" / f"{STAMP}_vX4_summary.md").is_file()
    # the prefetch's list belongs to every run of the day: copied, not moved
    assert (dest / "stats" / "20261022-0901_vX4_hub_revisions.csv").is_file()
    assert (results / "stats" / "20261022-0901_vX4_hub_revisions.csv").is_file()
    manifest = list(csv.DictReader((dest / "stats" / f"{STAMP}_vX4_tables.tsv").open(encoding="utf-8"), delimiter="\t"))
    assert all(Path(row["table"]).is_file() for row in manifest)


def test_provenance_counts_the_fold_and_names_what_ran(run):
    results, folds, _ = run
    text = (ce.collect(results, STAMP, results / "evaluations", folds) / "PROVENANCE.md").read_text(encoding="utf-8")
    assert "not** held-out accuracy" in text and "co-authors" in text
    assert "commit abc123" in text and "93f2dc7" in text
    assert ce.sha256(folds) in text and ce.LINDAT_RECORD in text
    # 5 test rows, 2 without a name, 3 named pages, 2 evaluated, 1 named but not evaluated
    assert "| fold1 test | v1.4 | fold1 | 1 | 5 | 2 | 3 | 2 | 1 |" in text
    assert "| 5,449-page set | v1.4 | fold1 | 1 | – | – | – | 1 | – |" in text
    assert "3 steps, 1 not OK." in text and "FAILED  v1.4 5,449-page set top-1 (exit 1)" in text


def test_the_folds_csv_defaults_to_the_one_the_log_names(run):
    results, folds, _ = run
    text = (ce.collect(results, STAMP, results / "evaluations") / "PROVENANCE.md").read_text(encoding="utf-8")
    assert ce.sha256(folds) in text


def test_a_run_is_collected_once_and_an_unknown_stamp_is_refused(run, capsys):
    results, folds, _ = run
    ce.collect(results, STAMP, results / "evaluations", folds)
    assert ce.main(["--stamp", STAMP, "--folds_csv", str(folds)]) == 1
    assert ce.main(["--stamp", "19990101-0000"]) == 1
    assert "not found" in capsys.readouterr().err
