import subprocess
import sys
from pathlib import Path

RUN_SCRIPT = Path(__file__).parent.parent / "run.py"


def test_cli_help_flag():
    """Test that the entrypoint parses arguments correctly and can return help."""
    result = subprocess.run([sys.executable, str(RUN_SCRIPT), "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "Page sorter based on ViT" in result.stdout
    assert "--topn" in result.stdout
    assert "--best" in result.stdout


def test_cli_invalid_revision():
    """Test that using an unsupported revision raises the expected error safely."""
    result = subprocess.run(
        [sys.executable, str(RUN_SCRIPT), "-rev", "v99.9", "--eval"], capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "ValueError" in result.stderr
    assert "Revision v99.9 is not supported" in result.stderr


def test_cli_missing_input(tmp_path):
    """Test error handling when input directory is empty or missing."""
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()

    result = subprocess.run([sys.executable, str(RUN_SCRIPT), "-d", str(empty_dir)], capture_output=True, text=True)
    # The script should exit gracefully but report no files found
    assert result.returncode == 0
    assert "No valid image files" in result.stdout or "0 files" in result.stdout


def test_cli_invalid_topn():
    """Test boundary validation for the --topn argument."""
    result = subprocess.run(
        [sys.executable, str(RUN_SCRIPT), "-d", ".", "--topn", "999"], capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "Error" in result.stderr or "invalid" in result.stderr.lower()
