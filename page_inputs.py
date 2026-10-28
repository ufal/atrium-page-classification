"""
page_inputs.py — the pages a classification run reads, the same on the command line and in the service.

`run.py` used to take page IMAGES only: one extension (`-ff`, case-sensitive), and a PDF had to be
split into PNG files first with `data_scripts/unix/pdf2png.sh`. `POST /predict_document` meanwhile
rendered PDF pages itself with pypdfium2, and `POST /predict_image` took any image Pillow reads.
This module gives the command line the service's readers (atrium-project#72, the service–CLI
alignment):

* `parse_formats()` — `-ff png,JPG,tif`: several formats, any case;
* `is_pdf()` — a PDF is recognised by its content (`%PDF-` in the first KB), as the other ATRIUM
  tools recognise it, not by its name;
* `scrape()` — the files of a directory whose extension is one of the formats;
* `render_pdf()` — the PDF's pages as images, rendered by `service/pdf_render.py` (the service's
  own renderer, at `PDF_RENDER_DPI`), with the `pages` selection of `/predict_document`
  (`service/document_json.py`);
* `pdf_page_keys()` — the record key of each rendered page: a born-digital record's own page label
  when the baseline carries `page_index`, as `/predict_document` writes it;
* `flatten_on_white()` — an image as the models see it: RGB, with any transparency put on white
  (both sides since v1.11.0-beta; the service used to keep the colour stored under a transparent
  pixel, often black).

Torch-free and FastAPI-free, like the two service modules it calls: the fast test lane imports it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

#: The format `-ff` takes when a PDF should be rendered rather than opened as an image.
PDF_FORMAT = "pdf"


def flatten_on_white(image):
    """`image` in RGB, a transparent or semi-transparent pixel composited onto white.

    The command line's loaders always did this; the service converted straight to RGB, so the same
    PNG with an alpha channel reached the models with different pixels (atrium-project#72). Decoding
    errors (a damaged or truncated file) propagate, as from `Image.convert`.
    """
    from PIL import Image

    if image.mode == "RGB":
        return image
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, "WHITE")
    background.paste(rgba, (0, 0), rgba)
    return background.convert("RGB")


def parse_formats(spec: Optional[str]) -> Set[str]:
    """`"png,JPG, .tif"` -> `{"png", "jpg", "tif"}`; empty -> `{"png"}`."""
    formats = {part.strip().lstrip(".").lower() for part in str(spec or "").split(",")}
    formats.discard("")
    return formats or {"png"}


def is_pdf(path) -> bool:
    """Whether the file is a PDF, by its content: a PDF header may sit anywhere in the first KB."""
    try:
        with open(path, "rb") as handle:
            return b"%PDF-" in handle.read(1024)
    except OSError:
        return False


def scrape(directory, formats: Iterable[str], inner: bool = True) -> List[str]:
    """The files of `directory` (and its sub-directories when `inner`) whose extension, in any
    case, is one of `formats`. Sorted, so a run is reproducible."""
    wanted = {f".{fmt.lower()}" for fmt in formats}
    root = Path(directory)
    candidates = root.rglob("*") if inner else root.iterdir()
    files = [str(p) for p in candidates if p.is_file() and p.suffix.lower() in wanted]
    return sorted(files)


def render_pdf(
    pdf_path,
    out_dir,
    pages: Optional[str] = None,
    dpi: Optional[int] = None,
) -> Tuple[str, List[Tuple[int, str]]]:
    """Render the pages of a PDF (or the `pages` selection, "1,3,5-7") into PNG files in `out_dir`.

    Returns `(doc_id, [(physical page number, image path), ...])`. The doc_id is the one
    `/predict_document` derives (`canonical_doc_id` of the PDF's name), and each image is named
    `<doc_id>-<n>.png`, so `utils.doc_id_and_page()` reads back the same document and page.
    Raises `ValueError` for a selection that is malformed or reaches past the last page, and
    `pypdfium2.PdfiumError` for a file PDFium cannot open.
    """
    import tool_limits
    from service.document_json import doc_id_for_document, expand_page_selection, parse_page_selection
    from service.pdf_render import LOCK, open_document, render_page, render_scale

    selection = parse_page_selection(pages)
    doc_id = doc_id_for_document(Path(pdf_path).name)
    dpi = int(dpi if dpi is not None else tool_limits.PDF_RENDER_DPI.get())
    scale = render_scale(dpi)
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    rendered: List[Tuple[int, str]] = []
    with LOCK:
        document = open_document(Path(pdf_path).read_bytes())
    try:
        with LOCK:
            numbers = expand_page_selection(selection, len(document))
        for number in numbers:
            with LOCK:
                image = render_page(document, number - 1, scale)
            path = target / f"{doc_id}-{number}.png"
            image.save(path)
            rendered.append((number, str(path)))
    finally:
        with LOCK:
            document.close()
    return doc_id, rendered


def baseline_labels(baseline_path) -> Dict[int, str]:
    """{page_index: page label} of a baseline record file, as `/predict_document` reads its part."""
    from service.document_json import baseline_page_labels

    if not baseline_path or not Path(baseline_path).is_file():
        return {}
    return baseline_page_labels(Path(baseline_path).read_bytes())


def pdf_page_keys(numbers: Sequence[int], baseline_path=None) -> Dict[int, Optional[str]]:
    """The record key of each rendered page: the baseline's label for its `page_index`, else
    `str(n)`, or None when `str(n)` is another page's label (`record_page_keys`)."""
    from service.document_json import record_page_keys

    return record_page_keys(list(numbers), baseline_labels(baseline_path))


def remap_pages(rdf, doc_id: str, keys: Dict[int, Optional[str]]):
    """The rows of `rdf` (FILE/PAGE/...) with the rendered PDF's pages under their record keys.

    Rows of other documents pass through. A page whose key is None is left out of what is
    written to the record (it stays in the CSV), as `/predict_document` leaves it out.
    """
    if not keys:
        return rdf
    rows = rdf.copy()
    rows["PAGE"] = rows["PAGE"].astype(object)  # a label replaces a number: no int column
    mine = rows["FILE"].astype(str) == str(doc_id)
    mapped = rows.loc[mine, "PAGE"].map(lambda page: keys.get(int(page)))
    rows.loc[mine, "PAGE"] = mapped
    return rows[~(mine & rows["PAGE"].isna())]
