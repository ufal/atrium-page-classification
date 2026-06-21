"""
tests/test_downscale.py
=======================
Unit tests for supplementary/downscale.py.

Scope
-----
* downscale() – file creation, output dimensions, dry-run, skip-existing,
  overwrite, extension filtering, empty-subdir handling.

Requires Pillow (already in requirements.txt).
No GPU, no trained model, no network required.
"""
from pathlib import Path

from downscale import downscale
from PIL import Image

# ── fixture helpers ────────────────────────────────────────────────────────

def make_png(path: Path, size: tuple = (100, 100)) -> None:
    """Create a minimal solid-colour RGB PNG at *path*."""
    Image.new("RGB", size, (128, 128, 128)).save(str(path))


def make_dataset(root: Path, structure: dict) -> None:
    """
    Build a training-style category directory tree with small PNG images.

    structure = {"CAT": [("img-1.png", (w, h)), ...], ...}
    """
    for cat, images in structure.items():
        (root / cat).mkdir(parents=True, exist_ok=True)
        for fname, size in images:
            make_png(root / cat / fname, size=size)


# ════════════════════════════════════════════════════════════════════════════
# Basic functionality
# ════════════════════════════════════════════════════════════════════════════
class TestDownscaleBasic:

    def test_output_file_created(self, tmp_path):
        src, dst = tmp_path / "src", tmp_path / "dst"
        make_dataset(src, {"TEXT": [("img-1.png", (100, 100))]})
        downscale(src, dst, scale=50.0, ext="png",
                  overwrite=False, dry_run=False, quiet=True)
        assert (dst / "TEXT" / "img-1.png").exists()

    def test_output_dimensions_halved_at_50pct(self, tmp_path):
        src, dst = tmp_path / "src", tmp_path / "dst"
        make_dataset(src, {"TEXT": [("img-1.png", (100, 80))]})
        downscale(src, dst, scale=50.0, ext="png",
                  overwrite=False, dry_run=False, quiet=True)
        with Image.open(dst / "TEXT" / "img-1.png") as im:
            assert im.size == (50, 40)

    def test_category_subdirectory_structure_preserved(self, tmp_path):
        src, dst = tmp_path / "src", tmp_path / "dst"
        make_dataset(src, {
            "DRAW": [("d-1.png", (50, 50))],
            "TEXT": [("t-1.png", (50, 50))],
        })
        downscale(src, dst, scale=30.0, ext="png",
                  overwrite=False, dry_run=False, quiet=True)
        assert (dst / "DRAW" / "d-1.png").exists()
        assert (dst / "TEXT" / "t-1.png").exists()

    def test_quiet_summary_reports_correct_created_count(self, tmp_path, capsys):
        src, dst = tmp_path / "src", tmp_path / "dst"
        make_dataset(src, {"TEXT": [("a-1.png", (40, 40)), ("b-2.png", (40, 40))]})
        downscale(src, dst, scale=50.0, ext="png",
                  overwrite=False, dry_run=False, quiet=True)
        out = capsys.readouterr().out
        assert "2 created" in out

    def test_minimum_output_dimension_is_one_pixel(self, tmp_path):
        """Very small scale must not produce a zero-size image."""
        src, dst = tmp_path / "src", tmp_path / "dst"
        make_dataset(src, {"TEXT": [("img-1.png", (4, 4))]})
        downscale(src, dst, scale=1.0, ext="png",
                  overwrite=False, dry_run=False, quiet=True)
        with Image.open(dst / "TEXT" / "img-1.png") as im:
            assert im.size[0] >= 1
            assert im.size[1] >= 1


# ════════════════════════════════════════════════════════════════════════════
# Dry-run
# ════════════════════════════════════════════════════════════════════════════
class TestDownscaleDryRun:

    def test_dry_run_writes_no_files(self, tmp_path):
        src, dst = tmp_path / "src", tmp_path / "dst"
        make_dataset(src, {"TEXT": [("img-1.png", (100, 100))]})
        downscale(src, dst, scale=50.0, ext="png",
                  overwrite=False, dry_run=True, quiet=True)
        assert not (dst / "TEXT" / "img-1.png").exists()

    def test_dry_run_creates_no_destination_directories(self, tmp_path):
        src, dst = tmp_path / "src", tmp_path / "dst"
        make_dataset(src, {"TEXT": [("img-1.png", (100, 100))]})
        downscale(src, dst, scale=50.0, ext="png",
                  overwrite=False, dry_run=True, quiet=True)
        assert not dst.exists()

    def test_dry_run_summary_says_would_create(self, tmp_path, capsys):
        src = tmp_path / "src"
        make_dataset(src, {"TEXT": [("img-1.png", (100, 100))]})
        downscale(src, tmp_path / "dst", scale=50.0, ext="png",
                  overwrite=False, dry_run=True, quiet=True)
        out = capsys.readouterr().out
        assert "would create" in out


# ════════════════════════════════════════════════════════════════════════════
# Skip-existing / overwrite
# ════════════════════════════════════════════════════════════════════════════
class TestDownscaleSkipAndOverwrite:

    def test_existing_file_skipped_by_default(self, tmp_path, capsys):
        src, dst = tmp_path / "src", tmp_path / "dst"
        make_dataset(src, {"TEXT": [("img-1.png", (60, 60))]})
        downscale(src, dst, 50.0, "png", overwrite=False, dry_run=False, quiet=True)
        capsys.readouterr()  # discard first-run output
        downscale(src, dst, 50.0, "png", overwrite=False, dry_run=False, quiet=True)
        out = capsys.readouterr().out
        assert "0 created" in out
        assert "1 skipped" in out

    def test_overwrite_flag_processes_existing_file_again(self, tmp_path, capsys):
        src, dst = tmp_path / "src", tmp_path / "dst"
        make_dataset(src, {"TEXT": [("img-1.png", (60, 60))]})
        downscale(src, dst, 50.0, "png", overwrite=False, dry_run=False, quiet=True)
        capsys.readouterr()
        downscale(src, dst, 50.0, "png", overwrite=True, dry_run=False, quiet=True)
        out = capsys.readouterr().out
        assert "1 created" in out
        assert "0 skipped" in out


# ════════════════════════════════════════════════════════════════════════════
# Extension filtering
# ════════════════════════════════════════════════════════════════════════════
class TestDownscaleExtensionFiltering:

    def test_only_png_processed_when_ext_is_png(self, tmp_path):
        src, dst = tmp_path / "src", tmp_path / "dst"
        (src / "TEXT").mkdir(parents=True)
        make_png(src / "TEXT" / "img-1.png")
        Image.new("RGB", (50, 50)).save(str(src / "TEXT" / "img-2.jpg"))
        downscale(src, dst, 50.0, "png", overwrite=False, dry_run=False, quiet=True)
        assert (dst / "TEXT" / "img-1.png").exists()
        assert not (dst / "TEXT" / "img-2.jpg").exists()

    def test_jpg_extension_processes_jpeg_files(self, tmp_path):
        src, dst = tmp_path / "src", tmp_path / "dst"
        (src / "TEXT").mkdir(parents=True)
        Image.new("RGB", (60, 60)).save(str(src / "TEXT" / "img-1.jpg"))
        downscale(src, dst, 50.0, "jpg", overwrite=False, dry_run=False, quiet=True)
        assert (dst / "TEXT" / "img-1.jpg").exists()

    def test_no_matching_extension_prints_skipped_message(self, tmp_path, capsys):
        src, dst = tmp_path / "src", tmp_path / "dst"
        (src / "TEXT").mkdir(parents=True)
        Image.new("RGB", (40, 40)).save(str(src / "TEXT" / "img-1.jpg"))
        # ask for PNG but only JPEG exists
        downscale(src, dst, 50.0, "png", overwrite=False, dry_run=False, quiet=False)
        out = capsys.readouterr().out
        assert "skipped" in out.lower() or "no .png" in out.lower()


# ════════════════════════════════════════════════════════════════════════════
# Edge cases
# ════════════════════════════════════════════════════════════════════════════
class TestDownscaleEdgeCases:

    def test_empty_source_subdir_produces_no_destination_dir(self, tmp_path):
        src, dst = tmp_path / "src", tmp_path / "dst"
        src.mkdir()
        (src / "EMPTY_CAT").mkdir()   # exists but contains no images
        downscale(src, dst, 50.0, "png", overwrite=False, dry_run=False, quiet=True)
        assert not (dst / "EMPTY_CAT").exists()

    def test_no_subdirectories_at_all_prints_message(self, tmp_path, capsys):
        src = tmp_path / "src"
        src.mkdir()
        downscale(src, tmp_path / "dst", 50.0, "png",
                  overwrite=False, dry_run=False, quiet=False)
        out = capsys.readouterr().out
        assert "no subdirectories" in out.lower()

    def test_multiple_images_per_category_all_created(self, tmp_path):
        src, dst = tmp_path / "src", tmp_path / "dst"
        make_dataset(src, {
            "TEXT": [(f"img-{i}.png", (50, 50)) for i in range(1, 6)]
        })
        downscale(src, dst, 50.0, "png", overwrite=False, dry_run=False, quiet=True)
        output_files = list((dst / "TEXT").glob("*.png"))
        assert len(output_files) == 5
