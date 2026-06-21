import subprocess
import sys
from pathlib import Path


def test_cli_help_flag():
    """Test that the entrypoint parses arguments correctly and can return help."""
    run_script = Path(__file__).parent.parent / "run.py"

    # Run the script with --help
    result = subprocess.run(
        [sys.executable, str(run_script), "--help"],
        capture_output=True,
        text=True
    )

    # The script should exit cleanly and show the custom arguments
    assert result.returncode == 0
    assert "Page sorter based on ViT / YOLO-cls" in result.stdout
    assert "--topn" in result.stdout
    assert "--best" in result.stdout
    assert "--parallel" in result.stdout


def test_cli_invalid_revision():
    """Test that using an unsupported revision raises the expected error safely."""
    run_script = Path(__file__).parent.parent / "run.py"

    result = subprocess.run(
        [sys.executable, str(run_script), "-rev", "v99.9", "--eval"],
        capture_output=True,
        text=True
    )

    # Expect failure due to unsupported revision
    assert result.returncode != 0
    assert "ValueError" in result.stderr
    assert "Revision v99.9 is not supported" in result.stderr
