"""
tests/test_models_dir.py — `--models-dir`: the weights staged once, read by every run.

A Galaxy job (atrium-project#66) should not download the models again in every job. The admin
stages them once (`run.py --hf -rev <revision> --models-dir DIR`, one call per revision), and the
tool classifies with `--models-dir DIR`, which reads that folder only, offline. The checks below
need no torch, except the last one, which runs a whole classification with a stand-in model.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from run import build_parser, main, staged_model_paths, use_staged_models


@pytest.fixture
def clean_hf_env(monkeypatch):
    """main() writes the Hugging Face variables into os.environ; restore them after each test."""
    for name in ("HF_HOME", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.setenv(name, "sentinel")
        monkeypatch.delenv(name)
    return monkeypatch


def test_the_staged_folder_layout(tmp_path):
    model_dir, cp_dir, hf_home = staged_model_paths(tmp_path)
    assert (model_dir, cp_dir, hf_home) == (tmp_path / "model", tmp_path / "checkpoint", tmp_path / "hf")


def test_reading_a_staged_folder_goes_offline(tmp_path):
    environ = {}
    model_dir, cp_dir = use_staged_models(tmp_path, download=False, environ=environ)
    assert model_dir == str(tmp_path / "model") and cp_dir == tmp_path / "checkpoint"
    assert environ == {"HF_HOME": str(tmp_path / "hf"), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}


def test_staging_downloads_into_the_folder(tmp_path):
    environ = {}
    use_staged_models(tmp_path, download=True, environ=environ)
    assert environ == {"HF_HOME": str(tmp_path / "hf")}


def test_the_option_is_off_by_default():
    import configparser

    config = configparser.ConfigParser()
    config.read(Path(__file__).parent.parent / "setup" / "config.txt")
    assert build_parser(config).parse_args([]).models_dir is None


def test_a_missing_folder_without_hf_is_a_usage_error(tmp_path, clean_hf_env, capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--models-dir", str(tmp_path / "absent"), "-d", str(tmp_path)])
    assert excinfo.value.code == 2
    assert "stage the models into it with --hf first" in capsys.readouterr().err


def test_a_run_on_a_staged_folder_is_offline(tmp_path, clean_hf_env):
    """An empty input exits before any model loads; the environment is already set by then."""
    import os

    staged = tmp_path / "staged"
    staged.mkdir()
    empty = tmp_path / "empty"
    empty.mkdir()
    clean_hf_env.chdir(tmp_path)
    assert main(["--models-dir", str(staged), "-d", str(empty), "--no-hf"]) == 0
    assert os.environ["HF_HOME"] == str(staged / "hf")
    assert os.environ["HF_HUB_OFFLINE"] == "1" and os.environ["TRANSFORMERS_OFFLINE"] == "1"


def test_a_staged_run_loads_the_base_and_the_model_from_the_folder(tmp_path, clean_hf_env):
    pytest.importorskip("torch")
    from PIL import Image

    import classifier
    import utils

    staged = tmp_path / "staged"
    (staged / "model" / "model_v44").mkdir(parents=True)
    page = tmp_path / "DOC1-1.png"
    Image.new("RGB", (8, 8), "white").save(page)
    seen = {}

    class FakeClassifier:
        def __init__(self, checkpoint, num_labels, store_dir):
            seen["store_dir"] = store_dir

        def load_model(self, path):
            seen["model"] = path

        def process_images(self, files, labels, batch, train):
            return list(files)

        def create_dataloader(self, files, batch):
            return list(files)

        def infer_dataloader(self, loader, top_n, raw):
            return [[(0, 0.9), (1, 0.1)] for _ in loader], None

    clean_hf_env.chdir(tmp_path)
    clean_hf_env.setattr(classifier, "ImageClassifier", FakeClassifier)
    clean_hf_env.setattr(utils, "tolerate_scan_quirks", lambda: None)
    main(["--models-dir", str(staged), "-f", str(page), "-rev", "v4.4", "--no-hf", "-tn", "2"])
    assert seen["store_dir"] == str(staged / "checkpoint")
    assert seen["model"] == str(staged / "model" / "model_v44")
