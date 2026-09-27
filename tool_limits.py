"""tool_limits.py — every limit atrium-page-classification has (atrium-project#53, factor III).

One declaration, read by the service (``service/api.py``) and reported by ``GET /info``
(``limits`` and ``limits_meta``). Each limit is an environment setting; a malformed value
stops the service at startup, naming the variable (``atrium_limits.LimitConfigError``).
``.env.example`` §1/§5 and ``service/README.md``'s ``## Limits`` table list the same set, and
``tests/test_api_contract.py`` checks that the three agree.

Standard library only (``atrium_limits`` is the hub's canonical module at the repo root),
so importing it costs nothing and needs no web framework.
"""

from __future__ import annotations

from atrium_limits import LimitSet, limit, upload_limit

#: §4.5 upload limit, per uploaded part (the image or PDF, and the `document_json`
#: baseline). Over it → 413 ``limit_exceeded``.
MAX_UPLOAD = upload_limit(10)

#: Most pages a PDF sent to ``/predict_document`` may have. Checked before any page is
#: rendered. Over it → 413 ``limit_exceeded``.
MAX_PDF_PAGES = limit("MAX_PDF_PAGES", 50, unit="pages", minimum=1)

#: Resolution PDF pages are rasterised at before classification. 300 matches
#: ``data_scripts/unix/pdf2png.sh``'s default, which is how the training pages were made;
#: PyMuPDF's own default is 72 dpi. Changing it changes what the models see.
PDF_RENDER_DPI = limit("PDF_RENDER_DPI", 300, unit="dpi", minimum=1)

#: Largest image, in pixels (width × height), the service will decode: an uploaded image,
#: or a PDF page at ``PDF_RENDER_DPI`` (its size is computed before it is rendered). The
#: default is Pillow's own decompression-bomb threshold. Over it → 413 ``limit_exceeded``.
MAX_IMAGE_PIXELS = limit("MAX_IMAGE_PIXELS", 178956970, unit="px", minimum=1)

LIMITS = LimitSet(MAX_UPLOAD, MAX_PDF_PAGES, PDF_RENDER_DPI, MAX_IMAGE_PIXELS)
