"""
tests/test_run.py
=================
In-process CLI validation tests for run.py (Phase 1 / hub issue #10).

run.py defers its heavy imports (numpy/pandas/sklearn and the torch chain) into
main(), so importing it and exercising early CLI validation requires no ML
dependencies at all — these tests replace the former subprocess smoke tests,
which crashed in torch-free environments before argparse ever ran.
"""

import configparser
from pathlib import Path

import pytest

from model_registry import CATEGORIES, REVISION_TO_BASE_MODEL
from run import build_parser, main, resolve_revision_base

REPO_ROOT = Path(__file__).parent.parent


def _config():
    config = configparser.ConfigParser()
    config.read(REPO_ROOT / "setup" / "config.txt")
    return config


def test_cli_help_flag(capsys):
    """--help must print usage and exit 0 without touching heavy imports."""
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "Page sorter" in out
    assert "--revision" in out


def test_cli_invalid_revision():
    """An unsupported revision must fail fast, before any filesystem work."""
    with pytest.raises(ValueError, match="not supported"):
        main(["-rev", "v999.0", "-d", "."])


def test_cli_invalid_topn():
    """--topn outside 1..len(CATEGORIES) must fail fast with a clear message."""
    with pytest.raises(ValueError, match="topn"):
        main(["--topn", "999", "-d", "."])
    with pytest.raises(ValueError, match="topn"):
        main(["--topn", "0", "-d", "."])


def test_cli_missing_input(tmp_path, monkeypatch, capsys):
    """An empty input directory exits cleanly (rc 0) via the early-exit guard.

    chdir into tmp_path so the config's relative output folders (./result,
    ./model, ./checkpoint) and the paradata log land in the test sandbox.
    """
    monkeypatch.chdir(tmp_path)
    empty = tmp_path / "empty"
    empty.mkdir()

    rc = main(["-d", str(empty), "--no-inner", "--no-train", "--no-eval"])

    assert rc == 0
    assert "No valid image files found" in capsys.readouterr().out


def test_build_parser_defaults_follow_config():
    """build_parser derives its defaults from setup/config.txt."""
    config = _config()
    args = build_parser(config).parse_args([])
    assert args.topn == config.getint("SETUP", "top_N")
    assert args.base == config.get("SETUP", "base_model")
    assert args.file_format == config.get("SETUP", "files_format")
    assert args.train == config.getboolean("TRAIN", "Training")
    assert args.file is None and args.directory is None
    assert args.eval_dir == config.get("EVAL", "FOLDER_PAGES")


def test_eval_dir_overrides_the_config_folder():
    args = build_parser(_config()).parse_args(["--eval", "--eval_dir", "/data/fold1_test"])
    assert args.eval_dir == "/data/fold1_test"


REVISIONS = ["v1.3", "v2.3", "v3.3", "v4.3", "v5.3", "v1.4", "v2.4", "v3.4", "v4.4", "v5.4"]


@pytest.mark.parametrize("revision", REVISIONS)
def test_a_revision_builds_on_its_own_base(revision):
    """The config's base (regnety_160) or an unrelated -b never replaces the revision's own base:
    the eval transforms are built from it (atrium-project#15)."""
    expected = REVISION_TO_BASE_MODEL[revision]
    assert resolve_revision_base(revision, _config().get("SETUP", "base_model")) == expected
    assert resolve_revision_base(revision, "google/vit-base-patch16-224") == expected


def test_a_revision_without_a_registry_entry_keeps_the_requested_base():
    assert resolve_revision_base("main", "timm/regnety_160.swag_ft_in1k") == "timm/regnety_160.swag_ft_in1k"


def test_eval_reads_the_eval_dir(tmp_path, monkeypatch, capsys):
    """`--eval --eval_dir D` collects the pages of D, not of [EVAL] FOLDER_PAGES."""
    pytest.importorskip("torch")
    import utils

    class Collected(Exception):
        pass

    def collect_images(directory, ordered=True):
        raise Collected(directory)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(utils, "collect_images", collect_images)
    monkeypatch.setattr(utils, "tolerate_scan_quirks", lambda: None)  # process-wide Pillow switches
    pages = tmp_path / "fold1_test"
    with pytest.raises(Collected) as excinfo:
        main(["--eval", "--no-hf", "--eval_dir", str(pages), "-rev", "v2.4"])
    assert excinfo.value.args == (str(pages),)
    assert "Using google/vit-base-patch16-224 instead" in capsys.readouterr().out


def test_topn_upper_bound_matches_category_count():
    """The --topn ceiling tracks the model registry, not a magic number."""
    with pytest.raises(ValueError, match=str(len(CATEGORIES))):
        main(["--topn", str(len(CATEGORIES) + 1), "-d", "."])
