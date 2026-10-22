"""
tests/test_stage_fold_pages.py
==============================
Unit tests for data_scripts/stage_fold_pages.py (atrium-project#15): one fold's test pages, out of
the LINDAT category zips, as a folder `run.py --eval --eval_dir` can read.

No GPU, no trained model, no network required.
"""

import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data_scripts"))

import stage_fold_pages as sf  # noqa: E402

from model_registry import CATEGORIES  # noqa: E402

FOLDS = """categ,fold1,fold2,PNG,BYTES
DRAW,test,train,DOC1-1.png,5
DRAW,train,test,DOC1-2.png,5
TEXT,test,dev,DOC2-01.png,6
TEXT,test,train,DOC3-10.png,6
TEXT_HW,dev,test,DOC4-1.png,7
"""


def _zips(tmp_path: Path, members: dict) -> Path:
    """{zip name: [member path, ...]}; each member holds its own name as bytes."""
    folder = tmp_path / "zips"
    folder.mkdir()
    for name, paths in members.items():
        with zipfile.ZipFile(folder / name, "w") as handle:
            for path in paths:
                handle.writestr(path, path.encode())
    return folder


@pytest.fixture
def lindat(tmp_path):
    folds = tmp_path / "folds.csv"
    folds.write_text(FOLDS, encoding="utf-8")
    zips = _zips(
        tmp_path,
        {
            "DRAW.zip": ["DRAW/DOC1-1.png", "DRAW/DOC1-2.png"],
            "TEXT_part001.zip": ["TEXT/DOC2-01.png"],
            "TEXT_part002.zip": ["TEXT/DOC3-10.png"],
            "TEXT_HW.zip": ["TEXT_HW/DOC4-1.png"],
        },
    )
    return folds, zips, tmp_path / "out"


def _run(folds, zips, out, *extra):
    return sf.main(["--folds_csv", str(folds), "--fold", "fold1", "--zips", str(zips), "--out", str(out), *extra])


def test_the_test_pages_land_in_all_eleven_category_folders(lindat):
    folds, zips, out = lindat
    assert _run(folds, zips, out) == 0
    assert sorted(entry.name for entry in out.iterdir()) == sorted(CATEGORIES)
    staged = sorted(str(path.relative_to(out)) for path in out.rglob("*.png"))
    assert staged == ["DRAW/DOC1-1.png", "TEXT/DOC2-01.png", "TEXT/DOC3-10.png"]
    assert (out / "TEXT" / "DOC2-01.png").read_bytes() == b"TEXT/DOC2-01.png"


def test_a_page_in_no_zip_fails_and_is_listed(lindat, capsys):
    folds, zips, out = lindat
    (zips / "TEXT_part002.zip").unlink()
    assert _run(folds, zips, out) == 1
    assert (out / "missing.txt").read_text(encoding="utf-8") == "DOC3-10.png\n"
    assert _run(folds, zips, out, "--allow-missing") == 0
    assert "1 pages not in any zip" in capsys.readouterr().err


def test_a_page_in_another_category_s_zip_fails(tmp_path):
    folds = tmp_path / "folds.csv"
    folds.write_text(FOLDS, encoding="utf-8")
    zips = _zips(tmp_path, {"DRAW.zip": ["DOC1-1.png", "DOC2-01.png", "DOC3-10.png"]})
    out = tmp_path / "out"
    assert _run(folds, zips, out) == 1
    assert not (out / "DRAW" / "DOC2-01.png").exists() and not (out / "TEXT" / "DOC2-01.png").exists()


def test_a_second_run_extracts_nothing(lindat, capsys):
    folds, zips, out = lindat
    _run(folds, zips, out)
    capsys.readouterr()
    assert _run(folds, zips, out) == 0
    assert "; 0 extracted" in capsys.readouterr().out


def test_a_dry_run_writes_nothing(lindat):
    folds, zips, out = lindat
    assert _run(folds, zips, out, "--dry-run") == 0
    assert not out.exists()


def test_another_partition_is_read_from_the_same_column(lindat):
    folds, _, _ = lindat
    assert sf.wanted_pages(folds, "fold2", "test") == {"DOC1-2.png": "DRAW", "DOC4-1.png": "TEXT_HW"}
    with pytest.raises(ValueError, match="fold9"):
        sf.wanted_pages(folds, "fold9")


def test_a_row_without_a_png_name_is_left_out(lindat, capsys):
    folds, zips, out = lindat
    folds.write_text(FOLDS + "TEXT,test,test,,0\n", encoding="utf-8")
    assert "" not in sf.wanted_pages(folds, "fold1")
    assert _run(folds, zips, out) == 0
    assert "1 fold1 test rows of folds.csv have no PNG name" in capsys.readouterr().err


def test_the_zip_category_drops_the_part_number():
    assert sf.zip_category(Path("LINE_T_part002.zip")) == "LINE_T"
    assert sf.zip_category(Path("TEXT_HW.zip")) == "TEXT_HW"
