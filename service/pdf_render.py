"""
service/pdf_render.py — PDF pages to page images: the one place PDFium is called.

``POST /predict_document`` (``service/api.py``) renders every page of an uploaded PDF and
classifies the image. The engine is **pypdfium2** (PDFium; Apache-2.0 or BSD-3-Clause, declared
in ``setup/para_config.txt``) since atrium-project#72; PyMuPDF (AGPL-3.0, declared nowhere) did it
before. A page is rendered the way PyMuPDF's ``get_pixmap(dpi=...)`` rendered it: the whole page,
annotations and form fields included, at ``dpi / 72`` pixels per PDF point. The engines were
compared against ``pdftoppm`` (the renderer ``data_scripts/unix/pdf2png.sh`` made the training
pages with) in ``data_scripts/pdf_rasteriser_comparison.md``; the comparison script renders with
this module, so what it measures is what the service does.

Torch-free and FastAPI-free on purpose: the script and the fast test lane import it directly.

Two PDFium properties shape the code:

* **PDFium is not thread-safe.** pypdfium2 forbids calling it from two threads at once, even on
  different documents, and the service renders in a worker thread (issue #55). Every call into
  this module must hold :data:`LOCK`; the caller keeps the model outside it.
* **The pixel size is ``ceil(points × scale)``**, computed from the page size PDFium holds as a
  32-bit float. A page whose size lands a hair above a whole pixel (an A4 scan stored as
  595.2 pt wide is 2480.00005 px at 300 dpi) gets one extra column of pixels. The classifier
  resizes every page to its input size, so the column changes nothing it sees;
  :func:`pixel_size` reproduces the rounding so a size limit is checked on the real size.
"""

from __future__ import annotations

import math
import threading
from typing import Callable, Optional, Tuple

from PIL import Image

#: Held around every PDFium call (see the module docstring).
LOCK = threading.Lock()


def render_scale(dpi: int) -> float:
    """The PDFium scale that renders at ``dpi``: PDF points are 1/72 inch."""
    return dpi / 72


def pixel_size(page, scale: float) -> Tuple[int, int]:
    """``(width, height)`` in pixels that ``page`` renders to at ``scale``, rounded as PDFium does."""
    width, height = page.get_size()
    return math.ceil(width * scale), math.ceil(height * scale)


def open_document(content: bytes):
    """The PDF in ``content`` as a ``pypdfium2.PdfDocument``; hold :data:`LOCK`.

    Raises ``pypdfium2.PdfiumError`` (a RuntimeError) for bytes that are not a PDF, a damaged
    one or a locked one. Forms are initialised, so filled-in form fields render, as they did
    with PyMuPDF and do with ``pdftoppm``.
    """
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(content)
    try:
        pdf.init_forms()
    except pdfium.PdfiumError:
        pass  # a document whose form environment cannot start still renders its page content
    return pdf


def render_page(pdf, index: int, scale: float, check: Optional[Callable[[int, int], None]] = None) -> Image.Image:
    """Page ``index`` (0-based) of ``pdf`` as an RGB image at ``scale``; hold :data:`LOCK`.

    ``check(width, height)`` runs with the pixel size before anything is rendered, so a caller
    can refuse a page that would render too large (a small file can declare a huge page). The
    image is copied out of PDFium's bitmap before the bitmap is freed, and the page and the
    bitmap are closed whatever happens: PDFium's memory is not Python's.
    """
    page = pdf[index]
    try:
        if check is not None:
            check(*pixel_size(page, scale))
        bitmap = page.render(scale=scale)
        try:
            return bitmap.to_pil().convert("RGB")  # convert() copies: the bitmap is closed next
        finally:
            bitmap.close()
    finally:
        page.close()
