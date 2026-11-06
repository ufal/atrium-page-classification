"""
tests/test_container_entrypoint_shared.py — the batch image's shared ENTRYPOINT (container_entrypoint.py).

Canonical source: ``ufal/atrium-project`` → ``docs/templates/shared/test_container_entrypoint_shared.py``;
vendored byte-identically into every tool repo's ``tests/`` and held there by para-drift.

The image's program gets the arguments ``docker run`` passed; a first argument naming a shell or
Python runs as given, which is how Galaxy starts a tool's container (``docker run IMAGE /bin/sh
tool_script.sh``, atrium-project#66). Programs with a positional input must still receive an
input file that happens to be executable. Pure-Python checks: no Docker. Which stage of a repo's
Dockerfile uses the module is that repo's own test (``tests/test_batch_entrypoint.py``).
"""

from __future__ import annotations

import shutil

import pytest

import container_entrypoint as ce


@pytest.fixture
def program(tmp_path):
    path = tmp_path / "tool.py"
    path.write_text("", encoding="utf-8")
    return str(path)


def test_options_and_a_positional_input_go_to_the_program(program):
    args = ["/data/input", "--alto", "-o", "/data/output"]
    assert ce.command_for(program, args, python="py") == ["py", program, *args]


def test_no_arguments_run_the_program(program):
    assert ce.command_for(program, [], python="py") == ["py", program]


@pytest.mark.skipif(shutil.which("sh") is None, reason="no POSIX shell on this runner")
def test_a_galaxy_job_script_runs_as_given(program):
    shell = shutil.which("/bin/sh") or shutil.which("sh")
    job = [shell, "/tmp/job/tool_script.sh"]
    assert ce.command_for(program, job, python="py") == job


@pytest.mark.skipif(shutil.which("python3") is None, reason="python3 not on PATH")
def test_an_interpreter_named_on_path_runs_as_given(program):
    command = ce.command_for(program, ["python3", "-c", "pass"], python="py")
    assert command == [shutil.which("python3"), "-c", "pass"]


def test_an_executable_input_file_reaches_the_program(program, tmp_path):
    page = tmp_path / "page.alto.xml"
    page.write_text("<alto/>", encoding="utf-8")
    page.chmod(0o755)
    assert ce.command_for(program, [str(page)], python="py") == ["py", program, str(page)]


def test_a_folder_named_like_an_interpreter_reaches_the_program(program, tmp_path):
    folder = tmp_path / "sh"
    folder.mkdir()
    assert ce.command_for(program, [str(folder)], python="py") == ["py", program, str(folder)]


@pytest.mark.skipif(shutil.which("ls") is None, reason="no ls on this runner")
def test_a_program_that_is_no_interpreter_is_an_argument(program):
    assert ce.command_for(program, ["ls", "-o", "out"], python="py") == ["py", program, "ls", "-o", "out"]


def test_main_execs_the_command(program, monkeypatch):
    calls = []
    monkeypatch.setattr(ce.os, "execv", lambda path, argv: calls.append((path, argv)))
    ce.main([program, "in.xml", "--alto"])
    path, argv = calls[0]
    assert argv[1:] == [program, "in.xml", "--alto"] and path == argv[0]


@pytest.mark.parametrize("bad", ["tool.py", "/nonexistent/tool.py"])
def test_a_program_the_image_lacks_is_refused(bad, monkeypatch, capsys):
    monkeypatch.setattr(ce.os, "execv", lambda *a: pytest.fail("must not exec"))
    assert ce.main([bad, "--help"]) == 2
    assert "container_entrypoint.py:" in capsys.readouterr().err


def test_no_program_is_a_usage_error(capsys):
    assert ce.main([]) == 2
    assert "usage:" in capsys.readouterr().err


def test_the_selftest_passes(capsys):
    assert ce.main(["--selftest"]) == 0
    assert "selftest OK" in capsys.readouterr().out
