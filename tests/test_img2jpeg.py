"""
tests/test_img2jpeg.py
======================
Unit tests for supplementary/scripts/img2jpeg_v3.py — the only supplementary
script that previously lacked a matching test. Covers the JPEG mode-conversion
logic (transparency flattening, palette/grayscale handling) and the
single-image convert path, using tiny in-memory Pillow images.
"""

import pytest

pytest.importorskip("PIL")

from img2jpeg_v3 import convert_image_mode, get_image_files, process_single_image  # noqa: E402
from PIL import Image  # noqa: E402


def test_convert_rgba_flattens_to_rgb():
    img = Image.new("RGBA", (4, 4), (10, 20, 30, 128))
    out = convert_image_mode(img)
    assert out.mode == "RGB"
    assert out.size == (4, 4)


def test_convert_palette_to_rgb():
    img = Image.new("P", (4, 4))
    out = convert_image_mode(img)
    # Palette without transparency falls through the generic convert branch.
    assert out.mode == "RGB"


def test_convert_rgb_and_l_are_unchanged():
    rgb = Image.new("RGB", (2, 2), (1, 2, 3))
    assert convert_image_mode(rgb).mode == "RGB"
    gray = Image.new("L", (2, 2), 128)
    assert convert_image_mode(gray).mode == "L"


def test_process_single_image_png_to_jpeg(tmp_path):
    src = tmp_path / "in.png"
    Image.new("RGBA", (8, 8), (200, 100, 50, 255)).save(src, "PNG")
    dst = tmp_path / "out.jpg"

    ok, meta = process_single_image(src, dst, quality=80)

    assert ok is True
    assert meta["success"] is True
    assert dst.exists()
    assert meta["mode_after"] == "RGB"
    assert meta["dimensions"] == (8, 8)
    # Output is a readable JPEG.
    with Image.open(dst) as reopened:
        assert reopened.format == "JPEG"


def test_process_single_image_skips_existing(tmp_path):
    src = tmp_path / "in.png"
    Image.new("RGB", (4, 4), (0, 0, 0)).save(src, "PNG")
    dst = tmp_path / "out.jpg"
    Image.new("RGB", (4, 4), (0, 0, 0)).save(dst, "JPEG")

    ok, meta = process_single_image(src, dst)

    # Existing output -> conversion skipped, but metadata still records success.
    assert ok is False
    assert meta["skipped_existing"] is True
    assert meta["success"] is True


def test_get_image_files_filters_by_extension(tmp_path):
    (tmp_path / "a.png").write_bytes(b"x")
    (tmp_path / "b.tif").write_bytes(b"x")
    (tmp_path / "notes.txt").write_bytes(b"x")
    found = {p.name for p in get_image_files(tmp_path)}
    assert "a.png" in found and "b.tif" in found
    assert "notes.txt" not in found
