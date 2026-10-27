"""
tests/test_container_entrypoint.py — the batch image's ENTRYPOINT (entrypoint.py).

Options go to run.py, as they always did; a first argument that names a program runs as given,
which is how Galaxy starts a tool's container (`docker run IMAGE /bin/sh tool_script.sh`,
atrium-project#66). Pure-Python checks: no Docker, no torch.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import entrypoint  # noqa: E402


def test_options_go_to_the_classifier():
    assert entrypoint.command_for(["-d", "/data/input", "--best"], python="py") == [
        "py",
        "/app/run.py",
        "-d",
        "/data/input",
        "--best",
    ]


def test_no_arguments_run_the_classifier():
    assert entrypoint.command_for([], python="py") == ["py", "/app/run.py"]


@pytest.mark.skipif(shutil.which("sh") is None, reason="no POSIX shell on this runner")
def test_galaxy_style_command_runs_as_given():
    shell = shutil.which("/bin/sh") or shutil.which("sh")
    command = entrypoint.command_for([shell, "/tmp/job/tool_script.sh"], python="py")
    assert command == [shell, "/tmp/job/tool_script.sh"]


@pytest.mark.skipif(shutil.which("python3") is None, reason="python3 not on PATH")
def test_a_program_name_on_path_runs_as_given():
    command = entrypoint.command_for(["python3", "/app/run.py", "--help"], python="py")
    assert command == [shutil.which("python3"), "/app/run.py", "--help"]


def test_a_word_that_is_no_program_still_goes_to_the_classifier(tmp_path):
    # run.py takes no positional argument, so this fails there, as before, with argparse's message.
    not_a_program = tmp_path / "data" / "input"
    assert entrypoint.command_for([str(not_a_program)], python="py") == ["py", "/app/run.py", str(not_a_program)]


def test_main_reports_the_gpu_only_before_the_classifier(monkeypatch):
    calls = []
    monkeypatch.setattr(entrypoint, "report_gpu", lambda: calls.append("gpu"))
    monkeypatch.setattr(entrypoint.os, "execv", lambda path, argv: calls.append(argv))
    entrypoint.main(["-f", "page.png"])
    assert calls[0] == "gpu" and calls[1][1:] == ["/app/run.py", "-f", "page.png"]

    calls.clear()
    shell = shutil.which("sh")
    if shell:
        entrypoint.main([shell, "-c", "true"])
        assert calls == [[shell, "-c", "true"]]


def test_the_dockerfile_runs_this_file():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert 'ENTRYPOINT ["python3", "/app/entrypoint.py"]' in dockerfile
    # The file is tracked now; a second, generated copy would shadow it.
    assert "> /app/entrypoint.py" not in dockerfile
