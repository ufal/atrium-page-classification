"""tests/test_degrade_pages.py — the synthetic defects of #58's B4 (data_scripts/degrade_pages.py).

Pillow only, on small synthetic pages: a white sheet with a dark block of "text" in its upper left.
"""

import csv
import random
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageStat

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "data_scripts"))

import degrade_pages as dg  # noqa: E402

import decide_pages as dp  # noqa: E402


def _page(size=(200, 300)):
    img = Image.new("RGB", size, (250, 250, 250))
    draw = ImageDraw.Draw(img)
    draw.rectangle((20, 20, 120, 60), fill=(10, 10, 10))
    return img


def test_every_defect_with_synthetic_gold_has_a_degrader():
    assert set(dg.DEGRADERS) | {"rotated_text"} == set(dp.SYNTHETIC_DEFECTS)


def test_paper_colour_reads_the_corners():
    assert dg.paper_colour(_page()) == (250, 250, 250)


@pytest.mark.parametrize("defect", sorted(dg.DEGRADERS))
def test_each_degrader_changes_the_page_more_at_a_higher_level(defect):
    page = _page()
    outs = [dg.DEGRADERS[defect](page, level, random.Random(f"x:{defect}:{level}")) for level in dg.LEVELS]
    assert all(out.mode == "RGB" and params for out, params in outs)
    if defect == "blurred":  # a blur keeps the mean and lowers the contrast
        change = [-ImageStat.Stat(out.convert("L")).stddev[0] for out, _ in outs]
    elif defect in ("skewed", "cropped"):  # the canvas grows or shrinks
        change = [abs(out.size[0] * out.size[1] - page.size[0] * page.size[1]) for out, _ in outs]
    else:  # the colours move
        change = [
            sum(abs(a - b) for a, b in zip(ImageStat.Stat(page).mean, ImageStat.Stat(out).mean)) for out, _ in outs
        ]
    assert change[0] < change[1] < change[2], change


def test_faint_ink_lightens_the_ink_and_keeps_the_paper():
    out, _ = dg.DEGRADERS["faint_ink"](_page(), 3, random.Random(0))
    assert out.getpixel((60, 40))[0] > 150  # the dark block is faded
    assert out.getpixel((190, 290)) == (250, 250, 250)


def test_rotations_turn_the_page_the_named_way():
    page = _page()
    variants = {v[3]: v[4] for v in dg.variants_of(page, "p.png", ["rotated_text"], [1], seed=1) if v[1]}
    assert set(variants) == {"cw90", "upside_down", "ccw90"}
    # the dark block sits upper left; turned a quarter clockwise it sits upper right
    assert variants["cw90"].size == (300, 200) and variants["cw90"].getpixel((260, 60))[0] < 50
    assert variants["ccw90"].getpixel((40, 140))[0] < 50
    assert variants["upside_down"].getpixel((140, 260))[0] < 50


def test_variants_are_deterministic_for_a_seed():
    page = _page()
    first = dg.variants_of(page, "p.png", ["skewed", "cropped"], [2], seed=7)
    again = dg.variants_of(page, "p.png", ["skewed", "cropped"], [2], seed=7)
    assert [(v[0], v[5]) for v in first] == [(v[0], v[5]) for v in again]
    assert all(a[4].tobytes() == b[4].tobytes() for a, b in zip(first, again))


def test_degrade_writes_the_label_layout_and_a_manifest(tmp_path):
    source = tmp_path / "src" / "TEXT_P"
    source.mkdir(parents=True)
    _page().save(source / "doc-01.png")
    out = tmp_path / "out"
    rows = dg.degrade([(source / "doc-01.png", "TEXT_P")], out, ["yellowed", "rotated_text"], [1, 2], seed=3)
    assert len(rows) == 1 + 2 + 3
    with (out / "degradations.csv").open(newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    assert tuple(manifest[0]) == dg.MANIFEST_COLUMNS
    assert all((out / r["image"]).is_file() and r["image"].startswith("TEXT_P/") for r in manifest)
    assert manifest[0]["defect"] == "" and manifest[0]["level"] == "0"
    assert all((out / r["source"]).resolve() == (source / "doc-01.png").resolve() for r in manifest)
    pages, variants = dp.read_degradations(out / "degradations.csv")
    assert [label for _, label in pages] == ["TEXT_P"] * 6
    assert {v["VARIANT"] for v in variants.values()} == {"original", "yellowed", "rotated_text"}


def test_main_refuses_unknown_defects_and_levels_and_an_existing_manifest(tmp_path):
    src = tmp_path / "src" / "DRAW"
    src.mkdir(parents=True)
    _page().save(src / "d-01.png")
    with pytest.raises(SystemExit):
        dg.main(["--dir", str(tmp_path / "src"), "--out", str(tmp_path / "o"), "--defects", "stained"])
    with pytest.raises(SystemExit):
        dg.main(["--dir", str(tmp_path / "src"), "--out", str(tmp_path / "o"), "--levels", "4"])
    assert dg.main(["--dir", str(tmp_path / "src"), "--out", str(tmp_path / "o"), "--defects", "blurred"]) == 0
    with pytest.raises(SystemExit):
        dg.main(["--dir", str(tmp_path / "src"), "--out", str(tmp_path / "o")])
