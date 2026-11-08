#!/usr/bin/env python3
"""
data_scripts/degrade_pages.py — controlled defects on clean pages, the synthetic gold of #58's B4.

Why it exists. B4 (``decide_pages.py --task b4``) asks a decision model about a page's condition, and
no page of the dataset is labelled for it. Some defects can be made with a filter, on pages whose
condition is otherwise unchanged; the change in the model's answer between a page and its degraded
copy then measures the question, without annotators. This script writes, for each page:

* the original, re-saved as PNG (``defect`` empty, ``level`` 0), so both halves of a pair go through
  the same encoding;
* one copy per defect and level (``SYNTHETIC_DEFECTS`` of ``decide_pages.py``; levels 1–3, mild to
  strong), and for ``rotated_text`` one copy per orientation (``cw90``, ``upside_down``, ``ccw90``).

What is done, at level 1 / 2 / 3:

* ``skewed``: rotated by 2° / 4° / 8°, the sign drawn per page; the corners filled with the paper's colour;
* ``yellowed``: multiplied by an aged-paper tint, light / medium / strong;
* ``faint_ink``: blended towards the paper's colour by 35 % / 55 % / 75 %, so the ink fades and the paper stays;
* ``blurred``: a Gaussian blur of 1 / 2 / 3.5 px per 1,000 px of the long side;
* ``cropped``: 8 % / 15 % / 25 % cut from one side, the side drawn per page;
* ``uneven_light``: a shadow gradient from one side, down to 70 % / 50 % / 30 % brightness;
* ``rotated_text``: turned a quarter clockwise, upside down, or a quarter counter-clockwise (level 1).

Stains, tears, water damage, folds, holes, foxing, bleed-through and smudges are not faked: a filter
would teach nothing about the real thing. They wait for annotated pages (ufal/atrium-project#83).
Everything is deterministic for a given ``--seed``. The output keeps the ``<out>/<CATEGORY>/`` layout
and a manifest, ``<out>/degradations.csv`` (image, source, label, defect, level, orientation,
params), which ``decide_pages.py --degradations`` reads. Run it from the repository root::

    python3 data_scripts/degrade_pages.py --dir small_data_samples --per-label 2 --out stage/degraded
    python3 data_scripts/degrade_pages.py --dir small_data_samples --defects skewed,blurred --levels 1,3 \\
        --out stage/degraded_small
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageChops, ImageFilter, ImageOps, ImageStat

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from decide_pages import SYNTHETIC_DEFECTS, pages_from_dir, sample_per_label  # noqa: E402

LEVELS = (1, 2, 3)
ORIENTATION_TURNS = {
    "cw90": Image.Transpose.ROTATE_270,  # PIL turns counter-clockwise: 270° ccw is a quarter clockwise
    "upside_down": Image.Transpose.ROTATE_180,
    "ccw90": Image.Transpose.ROTATE_90,
}
SKEW_DEGREES = {1: 2.0, 2: 4.0, 3: 8.0}
YELLOW_TINT = {1: (255, 242, 205), 2: (250, 228, 168), 3: (238, 208, 130)}
FADE = {1: 0.35, 2: 0.55, 3: 0.75}
BLUR_PER_1000PX = {1: 1.0, 2: 2.0, 3: 3.5}
CROP = {1: 0.08, 2: 0.15, 3: 0.25}
SHADOW_FLOOR = {1: 0.70, 2: 0.50, 3: 0.30}
SIDES = ("left", "right", "top", "bottom")
MANIFEST_COLUMNS = ("image", "source", "label", "defect", "level", "orientation", "params")


def paper_colour(img: Image.Image, patch: int = 12) -> Tuple[int, int, int]:
    """The mean colour of the four corner patches: the paper, on a scanned page."""
    w, h = img.size
    patch = max(1, min(patch, w // 4 or 1, h // 4 or 1))
    boxes = [(0, 0), (w - patch, 0), (0, h - patch), (w - patch, h - patch)]
    means = [ImageStat.Stat(img.crop((x, y, x + patch, y + patch))).mean for x, y in boxes]
    return tuple(round(sum(m[i] for m in means) / len(means)) for i in range(3))  # type: ignore[return-value]


def _skewed(img: Image.Image, level: int, rng: random.Random) -> Tuple[Image.Image, Dict[str, Any]]:
    angle = SKEW_DEGREES[level] * rng.choice((-1, 1))
    out = img.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=paper_colour(img))
    return out, {"degrees": angle}


def _yellowed(img: Image.Image, level: int, rng: random.Random) -> Tuple[Image.Image, Dict[str, Any]]:
    tint = YELLOW_TINT[level]
    return ImageChops.multiply(img, Image.new("RGB", img.size, tint)), {"tint": list(tint)}


def _faint_ink(img: Image.Image, level: int, rng: random.Random) -> Tuple[Image.Image, Dict[str, Any]]:
    alpha = FADE[level]
    return Image.blend(img, Image.new("RGB", img.size, paper_colour(img)), alpha), {"towards_paper": alpha}


def _blurred(img: Image.Image, level: int, rng: random.Random) -> Tuple[Image.Image, Dict[str, Any]]:
    radius = max(0.5, BLUR_PER_1000PX[level] * max(img.size) / 1000)
    return img.filter(ImageFilter.GaussianBlur(radius)), {"radius_px": round(radius, 2)}


def _cropped(img: Image.Image, level: int, rng: random.Random) -> Tuple[Image.Image, Dict[str, Any]]:
    side, share = rng.choice(SIDES), CROP[level]
    w, h = img.size
    cut_w, cut_h = round(w * share), round(h * share)
    box = {
        "left": (cut_w, 0, w, h),
        "right": (0, 0, w - cut_w, h),
        "top": (0, cut_h, w, h),
        "bottom": (0, 0, w, h - cut_h),
    }[side]
    return img.crop(box), {"side": side, "share": share}


def _uneven_light(img: Image.Image, level: int, rng: random.Random) -> Tuple[Image.Image, Dict[str, Any]]:
    side, floor = rng.choice(SIDES), SHADOW_FLOOR[level]
    gradient = Image.linear_gradient("L")  # 0 at the top, 255 at the bottom
    gradient = {
        "top": gradient,
        "bottom": ImageOps.flip(gradient),
        "left": gradient.transpose(Image.Transpose.ROTATE_90).transpose(Image.Transpose.FLIP_LEFT_RIGHT),
        "right": gradient.transpose(Image.Transpose.ROTATE_90),
    }[side]
    mask = gradient.resize(img.size).point(lambda v: round(255 * (floor + (1 - floor) * v / 255)))
    return ImageChops.multiply(img, Image.merge("RGB", (mask, mask, mask))), {"dark_side": side, "floor": floor}


DEGRADERS: Dict[str, Callable[[Image.Image, int, random.Random], Tuple[Image.Image, Dict[str, Any]]]] = {
    "skewed": _skewed,
    "yellowed": _yellowed,
    "faint_ink": _faint_ink,
    "blurred": _blurred,
    "cropped": _cropped,
    "uneven_light": _uneven_light,
}


def variants_of(
    img: Image.Image, page_name: str, defects: Sequence[str], levels: Sequence[int], seed: int
) -> List[Tuple[str, str, int, str, Image.Image, Dict[str, Any]]]:
    """``(suffix, defect, level, orientation, image, params)`` for one page, the original first."""
    out = [("", "", 0, "upright", img, {})]
    for defect in defects:
        if defect == "rotated_text":
            for orientation, turn in ORIENTATION_TURNS.items():
                out.append((f"__rotated_text_{orientation}", defect, 1, orientation, img.transpose(turn), {}))
            continue
        for level in levels:
            rng = random.Random(f"{seed}:{page_name}:{defect}:{level}")
            degraded, params = DEGRADERS[defect](img, level, rng)
            out.append((f"__{defect}_{level}", defect, level, "upright", degraded, params))
    return out


def degrade(
    pages: Sequence[Tuple[Path, str]], out: Path, defects: Sequence[str], levels: Sequence[int], seed: int
) -> List[Dict[str, Any]]:
    """Write every variant of every page under *out* and return the manifest rows."""
    rows: List[Dict[str, Any]] = []
    for source, label in pages:
        folder = out / label
        folder.mkdir(parents=True, exist_ok=True)
        with Image.open(source) as opened:
            img = ImageOps.exif_transpose(opened).convert("RGB")
        for suffix, defect, level, orientation, image, params in variants_of(img, source.name, defects, levels, seed):
            target = folder / f"{source.stem}{suffix}.png"
            image.save(target)
            rows.append(
                {
                    "image": target.relative_to(out).as_posix(),
                    "source": Path(os.path.relpath(source.resolve(), out.resolve())).as_posix(),
                    "label": label,
                    "defect": defect,
                    "level": level,
                    "orientation": orientation,
                    "params": json.dumps(params, sort_keys=True) if params else "",
                }
            )
    with (out / "degradations.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _csv_list(text: str) -> List[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    parser.add_argument("--dir", type=Path, required=True, help="a folder with one sub-folder per label")
    parser.add_argument("--out", type=Path, required=True, help="where the copies and degradations.csv go")
    parser.add_argument("--per-label", type=int, help="at most N source pages per label")
    parser.add_argument("--seed", type=int, default=420)
    parser.add_argument("--defects", default="all", help=f"comma-separated, or all: {','.join(SYNTHETIC_DEFECTS)}")
    parser.add_argument("--levels", default="1,2,3", help="comma-separated levels of 1–3 (not for rotated_text)")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    defects = list(SYNTHETIC_DEFECTS) if args.defects == "all" else _csv_list(args.defects)
    unknown = sorted(set(defects) - set(SYNTHETIC_DEFECTS))
    if unknown:
        parser.error(f"unknown defects {unknown}; choose from {list(SYNTHETIC_DEFECTS)}")
    try:
        levels = sorted({int(level) for level in _csv_list(args.levels)})
    except ValueError:
        parser.error("--levels takes integers")
    if not levels or set(levels) - set(LEVELS):
        parser.error(f"--levels must be among {LEVELS}")
    pages = pages_from_dir(args.dir)
    if args.per_label:
        pages = sample_per_label(pages, args.per_label, args.seed)
    if not pages:
        parser.error(f"no pages under {args.dir}/<CATEGORY>/")
    if (args.out / "degradations.csv").exists():
        parser.error(f"{args.out}/degradations.csv exists; choose another --out")
    rows = degrade(pages, args.out, defects, levels, args.seed)
    print(f"{len(pages)} pages -> {len(rows)} images in {args.out} (manifest: degradations.csv)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
