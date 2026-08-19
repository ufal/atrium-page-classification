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

from model_registry import CATEGORIES
from run import build_parser, main

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


def test_topn_upper_bound_matches_category_count():
    """The --topn ceiling tracks the model registry, not a magic number."""
    with pytest.raises(ValueError, match=str(len(CATEGORIES))):
        main(["--topn", str(len(CATEGORIES) + 1), "-d", "."])
