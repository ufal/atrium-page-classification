#!/usr/bin/env python3
"""
data_scripts/compare_pdf_rasterisers.py — the service's PDF rendering against the references.

Why it exists. ``POST /predict_document`` rasterised PDF pages with PyMuPDF (AGPL-3.0, declared
nowhere) until atrium-project#72 replaced it with pypdfium2 (PDFium; Apache-2.0 or BSD-3-Clause).
Engines anti-alias, decode images and round page sizes differently, and the classifier only ever
sees the rendered pixels, so the swap is measured, not assumed. Every page is rendered at the same
resolution, as RGB, by

* **the service** — ``service/pdf_render.py`` itself, so what is measured is what the service does;
* **pdftoppm** (poppler) — the renderer ``data_scripts/unix/pdf2png.sh`` made the TRAINING pages
  with, run the same way (``pdftoppm -png -r DPI``); the closer the service is to it, the closer
  a PDF page is to what the models learned from;
* **PyMuPDF**, optionally — the engine the service used before, for a before/after comparison.

Each reference is compared with the service's render at full size and at the classifier's input
size (224 × 224, where a difference can change a label). For PDFs built from page images
(``--from-images``), whose true pixels are known, the service's render is also compared with the
source. With ``--labels`` (torch and the model weights installed) every render is classified and
the top-1 labels are compared: the number that decides whether the engine changes an answer.

Neither reference is a dependency of this repository: pdftoppm is a system tool (poppler-utils,
GPL; the Unix data script already needs it), and PyMuPDF is in NO requirements file
(``tests/test_para_config.py`` fails if one names it). Install PyMuPDF only in a throwaway
environment, if at all::

    python3 -m venv /tmp/rast && /tmp/rast/bin/pip install "pypdfium2>=5.13.0,<6.0" pillow numpy [pymupdf]
    /tmp/rast/bin/python data_scripts/compare_pdf_rasterisers.py path/to/pdfs/ --dpi 300
    /tmp/rast/bin/python data_scripts/compare_pdf_rasterisers.py --from-images small_data_samples/ --limit 40
    /tmp/rast/bin/python data_scripts/compare_pdf_rasterisers.py --from-images small_data_samples/ --canvas 2480x3508
    /tmp/rast/bin/python data_scripts/compare_pdf_rasterisers.py --from-images small_data_samples/ \
        --canvas 4677x6622 --scan-dpi 200 --limit 3     # an A1 plan scanned at 200 dpi, rendered at 300

    # with labels: the service's own model manager (torch, timm, transformers; weights from the
    # Hugging Face Hub, or model/ when present), run from the repository root
    python3 data_scripts/compare_pdf_rasterisers.py path/to/amcr_pdfs/ --labels --version v4.3 \\
        --out amcr_labels.md

The recorded results and the decision they support are in
``data_scripts/pdf_rasteriser_comparison.md``. Exit status: 0 when the comparison ran (whatever it
found), 2 when the service's engine, every reference, or the input is missing.
"""

from __future__ import annotations

import argparse
import io
import math
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

#: A channel difference above this is visible (8 of 255 levels, ~3 %); smaller ones are
#: anti-aliasing and JPEG-decoder rounding.
VISIBLE_DIFF = 8

#: The classifiers' input side (``classifier.py`` eval transforms: ``Resize((size, size))``;
#: 224 for the published revisions). What the model sees is the page at this size, so the
#: difference there is the one that can change a label.
MODEL_INPUT = 224

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

Renderer = Callable[[bytes, int], List[Image.Image]]


# ── the renderers ──────────────────────────────────────────────────────────────────────────


def _service_module(name: str):
    """A torch-free module of this repository's ``service/`` (the repo root on ``sys.path``)."""
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return __import__(f"service.{name}", fromlist=[name])


def render_service(data: bytes, dpi: int) -> List[Image.Image]:
    """Every page as the service renders it (``service/pdf_render.py``)."""
    pdf_render = _service_module("pdf_render")
    scale = pdf_render.render_scale(dpi)
    with pdf_render.LOCK:
        pdf = pdf_render.open_document(data)
        try:
            return [pdf_render.render_page(pdf, index, scale) for index in range(len(pdf))]
        finally:
            pdf.close()


def pdftoppm_renderer(binary: str) -> Renderer:
    """Every page as ``pdftoppm -png -r DPI`` renders it — ``data_scripts/unix/pdf2png.sh``'s call."""

    def render(data: bytes, dpi: int) -> List[Image.Image]:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "in.pdf"
            source.write_bytes(data)
            subprocess.run([binary, "-png", "-r", str(dpi), str(source), str(Path(tmp) / "page")], check=True)
            # pdftoppm pads the page number to the width of the page count: page-1.png or page-01.png.
            files = sorted(Path(tmp).glob("page-*.png"), key=lambda p: int(re.findall(r"\d+", p.stem)[-1]))
            pages = []
            for path in files:
                with Image.open(path) as opened:
                    pages.append(opened.convert("RGB"))
            return pages

    return render


def render_pymupdf(data: bytes, dpi: int) -> List[Image.Image]:
    """Every page as ``service/api.py`` rendered it before atrium-project#72 (PyMuPDF)."""
    try:
        import pymupdf  # AGPL-3.0: only in the throwaway environment this script is run from
    except ImportError:  # PyMuPDF before 1.24 is importable only as `fitz`
        import fitz as pymupdf

    pages = []
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        for index in range(len(doc)):
            pix = doc.load_page(index).get_pixmap(dpi=dpi, alpha=False)
            pages.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples).copy())
    return pages


def available_references(pdftoppm: Optional[str]) -> Dict[str, Renderer]:
    """The references that can run here, in report order: pdftoppm first (the training renderer)."""
    references: Dict[str, Renderer] = {}
    binary = shutil.which(pdftoppm) if pdftoppm else None
    if binary:
        references["pdftoppm"] = pdftoppm_renderer(binary)
    if _importable("pymupdf") or _importable("fitz"):
        references["PyMuPDF"] = render_pymupdf
    return references


# ── the measures ───────────────────────────────────────────────────────────────────────────


@dataclass
class Difference:
    """How far two renders of one page are apart, over the area both cover."""

    mean_abs: float
    max_abs: int
    visible_share: float
    psnr: float


def at_model_input(a: Image.Image, b: Image.Image, side: int = MODEL_INPUT) -> Tuple[Image.Image, Image.Image]:
    """Both renders cropped to their common area and resized to ``side`` × ``side`` (bilinear,
    antialiased by Pillow when reducing, as torchvision's ``Resize`` does): the classifier's view."""
    box = (0, 0, min(a.width, b.width), min(a.height, b.height))
    return tuple(image.convert("RGB").crop(box).resize((side, side), Image.BILINEAR) for image in (a, b))


def difference(a: Image.Image, b: Image.Image) -> Difference:
    """Pixel difference of ``a`` and ``b`` (RGB), over the common top-left area.

    Engines round a page's pixel size differently (PDFium and pdftoppm round up), so two
    renders may differ by a row or a column; the comparison covers what both drew.
    """
    width, height = min(a.width, b.width), min(a.height, b.height)
    x = np.asarray(a.convert("RGB").crop((0, 0, width, height)), dtype=np.int16)
    y = np.asarray(b.convert("RGB").crop((0, 0, width, height)), dtype=np.int16)
    diff = np.abs(x - y)
    if not diff.size:
        return Difference(0.0, 0, 0.0, math.inf)
    mse = float(np.mean(diff.astype(np.float64) ** 2))
    return Difference(
        mean_abs=float(diff.mean()),
        max_abs=int(diff.max()),
        visible_share=float(np.mean(diff.max(axis=2) > VISIBLE_DIFF)),
        psnr=math.inf if mse == 0 else 10 * math.log10(255.0**2 / mse),
    )


def psnr_at_model_input(a: Image.Image, b: Image.Image) -> float:
    return difference(*at_model_input(a, b)).psnr


# ── inputs ─────────────────────────────────────────────────────────────────────────────────


@dataclass
class Document:
    name: str
    data: bytes
    #: The page images a PDF was built from (``--from-images``), else empty.
    sources: List[Image.Image] = field(default_factory=list)


def pdfs_under(paths: Sequence[Path]) -> List[Path]:
    found: List[Path] = []
    for path in paths:
        if path.is_dir():
            found.extend(sorted(p for p in path.rglob("*") if p.suffix.lower() == ".pdf"))
        elif path.is_file():
            found.append(path)
        else:
            raise FileNotFoundError(path)
    return found


def pdf_from_image(image: Image.Image, dpi: int) -> bytes:
    """A one-page PDF holding ``image`` at ``dpi``: the page is the scan, as a scanner writes it."""
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PDF", resolution=dpi)
    return buf.getvalue()


def on_canvas(image: Image.Image, canvas: Tuple[int, int]) -> Image.Image:
    """``image`` scaled to fit a white ``canvas`` (width, height) and placed top-left, to give a
    small sample the pixel geometry of a real scan of any size (a slip, an A4 page, a plan)."""
    ratio = min(canvas[0] / image.width, canvas[1] / image.height)
    page = Image.new("RGB", canvas, "white")
    page.paste(image.resize((round(image.width * ratio), round(image.height * ratio)), Image.LANCZOS), (0, 0))
    return page


def documents(
    pdfs: Iterable[Path],
    image_dirs: Sequence[Path],
    dpi: int,
    limit: Optional[int],
    canvas: Optional[Tuple[int, int]] = None,
) -> List[Document]:
    """The PDFs, then one single-page PDF per image under ``image_dirs`` stored at ``dpi`` (the
    scan's resolution: a page of ``px × 72 / dpi`` points, as a scanner writes it)."""
    docs = [Document(path.name, path.read_bytes()) for path in pdfs]
    images: List[Path] = []
    for directory in image_dirs:
        images.extend(sorted(p for p in directory.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES))
    for path in images[:limit] if limit else images:
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        if canvas:
            image = on_canvas(image, canvas)
        docs.append(Document(f"{path.parent.name}/{path.name}", pdf_from_image(image, dpi), [image]))
    return docs


# ── labels (optional) ──────────────────────────────────────────────────────────────────────


def label_function(version: str) -> Callable[[Image.Image], str]:
    """Top-1 label of a page image by the service's own model manager (torch and weights needed)."""
    manager = _service_module("inference").manager

    def top1(image: Image.Image) -> str:
        predictions = manager.predict(image, version=version, topn=1)
        if isinstance(predictions, dict):
            raise RuntimeError(predictions.get("error", "the model returned no prediction"))
        return str(predictions[0]["label"])

    return top1


# ── the report ─────────────────────────────────────────────────────────────────────────────


def _fmt_psnr(value: float) -> str:
    return "∞" if math.isinf(value) else f"{value:.1f}"


def _median(values: Sequence[float]) -> str:
    """The median, with identical pages (infinite PSNR) counted as the highest values."""
    if not values:
        return "—"
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return _fmt_psnr(ordered[middle])
    low, high = ordered[middle - 1], ordered[middle]
    return _fmt_psnr(high if math.isinf(low) else (low + high) / 2)


@dataclass
class _Tally:
    full: List[float] = field(default_factory=list)
    small: List[float] = field(default_factory=list)
    size_differs: int = 0
    agree: int = 0


def compare(
    docs: Sequence[Document],
    dpi: int,
    references: Dict[str, Renderer],
    labels: Optional[Callable[[Image.Image], str]] = None,
    service: Renderer = render_service,
) -> List[str]:
    """The markdown report: one row per page, then a summary per reference."""
    names = list(references)
    header = ["document", "page", "size (service)"]
    for name in names:
        header += [f"size ({name})", f"PSNR dB vs {name}", f"at {MODEL_INPUT} px"]
    header += [f"service vs source at {MODEL_INPUT} px"]
    if labels is not None:
        header += ["top-1: service / " + " / ".join(names)]
    rows = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    tallies = {name: _Tally() for name in names}
    source_psnrs: List[float] = []
    #: With both references: how far the old engine was from the training renderer — the bar
    #: the service's distance to pdftoppm is read against.
    old_vs_training: List[float] = []
    pages = 0
    for doc in docs:
        mine = service(doc.data, dpi)
        theirs = {name: render(doc.data, dpi) for name, render in references.items()}
        if any(len(pages_) != len(mine) for pages_ in theirs.values()):
            counts = ", ".join(f"{name} {len(pages_)}" for name, pages_ in theirs.items())
            rows.append(f"| {doc.name} | — | {len(mine)} pages | page counts differ: {counts} |")
            continue
        for index, page in enumerate(mine):
            pages += 1
            cells = [doc.name, str(index + 1), f"{page.width}×{page.height}"]
            labels_row = [labels(page)] if labels is not None else []
            for name in names:
                other = theirs[name][index]
                tally = tallies[name]
                full, small = difference(other, page).psnr, psnr_at_model_input(other, page)
                tally.full.append(full)
                tally.small.append(small)
                tally.size_differs += other.size != page.size
                cells += [f"{other.width}×{other.height}", _fmt_psnr(full), _fmt_psnr(small)]
                if labels is not None:
                    theirs_label = labels(other)
                    tally.agree += theirs_label == labels_row[0]
                    labels_row.append(theirs_label)
            if {"pdftoppm", "PyMuPDF"} <= set(theirs):
                old_vs_training.append(psnr_at_model_input(theirs["pdftoppm"][index], theirs["PyMuPDF"][index]))
            if index < len(doc.sources):
                source = doc.sources[index]
                if abs(source.width - page.width) > 1 or abs(source.height - page.height) > 1:
                    # scanned at another resolution than rendered: the ideal render is the scan resampled
                    source = source.resize(page.size, Image.LANCZOS)
                source_psnrs.append(psnr_at_model_input(source, page))
                cells.append(_fmt_psnr(source_psnrs[-1]))
            else:
                cells.append("")
            if labels is not None:
                flag = "" if len(set(labels_row)) == 1 else " ⚠️"
                cells.append(" / ".join(labels_row) + flag)
            rows.append("| " + " | ".join(cells) + " |")

    summary = ["", f"**{len(docs)} documents, {pages} pages at {dpi} dpi.**"]
    for name in names:
        tally = tallies[name]
        line = (
            f"- **vs {name}:** median PSNR {_median(tally.full)} dB at full size, {_median(tally.small)} dB at "
            f"{MODEL_INPUT} px (lowest {_fmt_psnr(min(tally.small, default=math.inf))} dB); pixel size differs on "
            f"{tally.size_differs} of {len(tally.small)} pages"
        )
        if labels is not None:
            line += f"; **top-1 labels agree on {tally.agree} of {len(tally.small)} pages**"
        summary.append(line + ".")
    if old_vs_training:
        summary.append(
            f"- **Baseline, PyMuPDF (the old engine) vs pdftoppm:** median PSNR {_median(old_vs_training)} dB at "
            f"{MODEL_INPUT} px (lowest {_fmt_psnr(min(old_vs_training))} dB)."
        )
    if source_psnrs:
        summary.append(
            f"- **vs the source images:** median PSNR {_median(source_psnrs)} dB at {MODEL_INPUT} px "
            f"(lowest {_fmt_psnr(min(source_psnrs))} dB)."
        )
    if labels is None:
        summary.append("- Labels not computed (run with `--labels` where torch and the model weights are installed).")
    return rows + summary


def _importable(module: str) -> bool:
    try:
        __import__(module)
    except ImportError:
        return False
    return True


def _canvas(value: Optional[str]) -> Optional[Tuple[int, int]]:
    if not value:
        return None
    width, height = (int(part) for part in value.lower().split("x"))
    return width, height


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("pdfs", nargs="*", type=Path, help="PDF files, or directories searched for *.pdf")
    parser.add_argument(
        "--from-images",
        type=Path,
        action="append",
        default=[],
        help="build a one-page PDF per page image under this directory, and compare with the image too",
    )
    parser.add_argument("--dpi", type=int, default=300, help="render resolution (the service's PDF_RENDER_DPI)")
    parser.add_argument("--limit", type=int, help="at most this many images from --from-images")
    parser.add_argument(
        "--scan-dpi",
        type=int,
        help="with --from-images: the resolution the images are stored at in their PDFs (default: --dpi)",
    )
    parser.add_argument(
        "--canvas",
        help="with --from-images: fit each image on a white page of WIDTHxHEIGHT pixels first, e.g. "
        "874x1240 (a 74 x 105 mm slip at 300 dpi), 2480x3508 (A4) or 4677x6622 (an A1 plan at 200 dpi)",
    )
    parser.add_argument("--pdftoppm", default="pdftoppm", help="the pdftoppm to compare with (default: on PATH)")
    parser.add_argument("--labels", action="store_true", help="classify every render (torch + model weights)")
    parser.add_argument("--version", default="v4.3", help="model revision for --labels (default: v4.3)")
    parser.add_argument("--out", type=Path, help="write the markdown report here instead of stdout")
    args = parser.parse_args(argv)

    if not _importable("pypdfium2"):
        print("pypdfium2 (the service's engine) is not installed here.", file=sys.stderr)
        return 2
    references = available_references(args.pdftoppm)
    if not references:
        print("no reference to compare with: install poppler-utils (pdftoppm) and/or PyMuPDF.", file=sys.stderr)
        return 2
    try:
        canvas = _canvas(args.canvas)
    except ValueError:
        print(f"--canvas takes WIDTHxHEIGHT, not {args.canvas!r}", file=sys.stderr)
        return 2
    try:
        docs = documents(pdfs_under(args.pdfs), args.from_images, args.scan_dpi or args.dpi, args.limit, canvas)
    except FileNotFoundError as exc:
        print(f"no such file or directory: {exc}", file=sys.stderr)
        return 2
    if not docs:
        print("nothing to compare: give PDFs or --from-images", file=sys.stderr)
        return 2
    labels = label_function(args.version) if args.labels else None
    report = "\n".join(compare(docs, args.dpi, references, labels)) + "\n"
    if args.out:
        args.out.write_text(report, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
