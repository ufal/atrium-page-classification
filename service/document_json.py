"""
service/document_json.py — the service's half of accretion contract rule 1.

`service/api.py` neither accepted nor returned a `document_json` part; there were zero hits
for the string anywhere under `service/` while `run.py` implemented the contract fully
(atrium-project#10, J2). This module is the wiring, kept OUT of `service/api.py` on purpose:
`api.py` imports `service.inference`, which imports torch, so anything living there can only
be tested where the full ML stack exists — and `tests/test_service_api.py` duly
`importorskip`s torch, which is precisely the shape of gate the review found blind to J2/G3
in the first place. Everything here imports pandas/atrium_document and nothing heavier, so
`tests/test_service_document_json.py` exercises the real accretion in the torch-free fast
lane and only the thin HTTP layer stays behind a skip.

The accretion logic itself is NOT re-implemented here. It goes through
`atrium_document_adapter.write_document_record()` — the same function `run.py`'s `-f` path
calls, already the ecosystem's reference implementation of the set_block/merge_block split,
the exact field grant and (since D4/D8) the Layer D schema gate and the field-survival
assertion. A second copy of that logic in the service is exactly how alto's `/process`
endpoint ended up writing junk (J1).

Two more helpers serve `/predict_document` when a caller sends a record that already has
pages (atrium-digital-convert#2's `/describe`):

* `parse_page_selection()` / `expand_page_selection()` — the optional `pages` form field
  ("1,3,5-7", 1-based physical pages): only those pages are rendered and classified.
* `baseline_page_labels()` / `record_page_keys()` — a PDF's pages are numbered 1..N by
  position, but a born-digital record names them by their PDF page labels (`i`, `ii`, `1`, …)
  and carries the position in `pages[].page_index`. Writing categories under "1", "2", … into
  such a record would attach them to the wrong page rows (or invent new ones), so the page
  keys are mapped through `page_index` to the record's own labels first.
"""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


def doc_id_for_image(filename: Optional[str]) -> Tuple[str, str]:
    """(doc_id, page key) for a single per-page IMAGE upload — `/predict_image`.

    Delegates to `utils.doc_id_and_page()`, the one composition of `canonical_doc_id()` and
    this repo's page-suffix split (atrium-project#10, D3), so a service upload and a CLI run
    over the same file land on the same record instead of forking it. Imported lazily
    because `utils` pulls in matplotlib/sklearn at module level and nothing else here needs
    them.
    """
    from utils import doc_id_and_page

    doc_id, page = doc_id_and_page(filename or "upload.png")
    # A filename of "" or "." can leave nothing behind; the record is keyed on doc_id and
    # DocumentRecord refuses an empty one, so degrade to a placeholder rather than 500.
    return (doc_id or "upload"), str(page if page is not None else 1)


def doc_id_for_document(filename: Optional[str]) -> str:
    """doc_id for a whole multi-page PDF upload — `/predict_document`.

    No page-suffix split here: the pages are the PDF's own 1..N, not a label in the
    filename, so this is `canonical_doc_id()` alone.
    """
    from atrium_document import canonical_doc_id

    return canonical_doc_id(filename or "upload.pdf") or "upload"


_SELECTION_PART = re.compile(r"^(\d+)(?:\s*-\s*(\d+))?$")


def parse_page_selection(spec: Optional[str]) -> Optional[List[Tuple[int, int]]]:
    """The `pages` form field as inclusive 1-based ranges, or None for every page.

    "1,3,5-7" -> [(1, 1), (3, 3), (5, 7)]. Blank (or absent) means every page. Raises
    `ValueError` on anything else: a part that is not `N` or `N-M`, a page below 1, a range
    whose end is before its start. The ranges are not expanded here, so "1-99999999" costs
    nothing before the PDF's page count is known (`expand_page_selection`).
    """
    if spec is None or not spec.strip():
        return None
    ranges: List[Tuple[int, int]] = []
    for part in spec.split(","):
        part = part.strip()
        match = _SELECTION_PART.match(part)
        if not match:
            raise ValueError(f"{part!r} is not a page number or a range like 5-7")
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) else start
        if start < 1:
            raise ValueError(f"pages are numbered from 1, not {start}")
        if end < start:
            raise ValueError(f"the range {part!r} ends before it starts")
        ranges.append((start, end))
    return ranges


def expand_page_selection(ranges: Optional[Sequence[Tuple[int, int]]], page_count: int) -> List[int]:
    """The physical page numbers to classify, sorted and unique.

    Every page (1..page_count) when `ranges` is None. Raises `ValueError` when a range
    reaches past the document's last page: a caller asking for page 12 of a 10-page PDF has
    the wrong PDF, which a silently shorter answer would hide.
    """
    if ranges is None:
        return list(range(1, page_count + 1))
    beyond = [end for _, end in ranges if end > page_count]
    if beyond:
        raise ValueError(f"page {max(beyond)} was asked for, but the PDF has {page_count} page(s)")
    return sorted({n for start, end in ranges for n in range(start, end + 1)})


def baseline_page_labels(baseline_bytes: Optional[bytes]) -> Dict[int, str]:
    """{page_index: page label} for the baseline's page rows that carry both, else {}.

    Read leniently: the part was already opened by the service (`parse_record_part`), and a
    record with no pages, or pages without `page_index` (an ALTO record keyed "1".."N"), simply
    has nothing to map.
    """
    if not baseline_bytes:
        return {}
    try:
        record = json.loads(baseline_bytes.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}
    pages = record.get("pages") if isinstance(record, dict) else None
    labels: Dict[int, str] = {}
    for row in pages if isinstance(pages, list) else []:
        if not isinstance(row, dict):
            continue
        index, label = row.get("page_index"), row.get("page")
        if isinstance(index, int) and not isinstance(index, bool) and index >= 1 and label not in (None, ""):
            labels.setdefault(index, str(label))
    return labels


def record_page_keys(page_numbers: Sequence[int], labels: Mapping[int, str]) -> Dict[int, Optional[str]]:
    """The record key each classified physical page is written under.

    * the baseline's label when its row carries this `page_index`;
    * else `str(n)`, as before — unless that string is ANOTHER page's label in the baseline
      (page "3" of a record whose third page is labelled "1" and whose fifth is "3"): then
      None, and the page is left out of the record rather than written onto the wrong row.
      Its prediction is still in the response.
    """
    taken = set(labels.values())
    keys: Dict[int, Optional[str]] = {}
    for n in page_numbers:
        if n in labels:
            keys[n] = labels[n]
        elif str(n) in taken:
            keys[n] = None
        else:
            keys[n] = str(n)
    return keys


def _top_rows(doc_id: str, pages: Sequence[Tuple[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten (page key, predictions) pairs into the adapter's FILE/PAGE/CLASS-1/SCORE-1 rows.

    `manager.predict()` returns either a top-N list of `{label, score}` or, when every model
    failed, a bare `{"error": ...}` dict. Pages of the second kind are dropped rather than
    turned into a row with no category: a `pages[]` row needs only `page` to satisfy the
    schema, so an empty one would pass Layer D and hand the next tool a page it believes was
    classified.
    """
    rows: List[Dict[str, Any]] = []
    for page_key, predictions in pages:
        if not isinstance(predictions, list) or not predictions:
            continue
        top = predictions[0]
        label = top.get("label") if isinstance(top, dict) else None
        if not label:
            continue
        row: Dict[str, Any] = {"FILE": doc_id, "PAGE": str(page_key), "CLASS-1": label}
        score = top.get("score") if isinstance(top, dict) else None
        if score is not None:
            row["SCORE-1"] = float(score)
        rows.append(row)
    return rows


def build_document_record(
    doc_id: str,
    pages: Sequence[Tuple[str, Any]],
    baseline_bytes: Optional[bytes] = None,
    paradata_logger: Any = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Accrete this run's `page_categories` / `pages[]` onto an optional baseline record.

    Returns `(record, schema_error)`:

    * `record` is the updated document JSON, ready to hand back in the response body, or
      None when no page produced a usable prediction (nothing to contribute → rule 3 says
      emit nothing rather than an empty block).
    * `schema_error` is None normally. It is non-None only in the one case Layer D lets
      through: the CALLER's uploaded baseline did not validate, so the adapter warned and
      emitted anyway (the defect is inherited, not ours) instead of raising. Surfacing it as
      a field means an automated caller can test for it instead of grepping the service log.

    Raises `RuntimeError` when the record this tool built is itself invalid — the adapter's
    Layer D refusal. `api.py` maps that to a 500, because a record page-classification
    cannot emit is a defect on this side.

    `paradata_logger` is the call's run (atrium-project#71): its `run_id`, `run_uuid` and
    licence block are stamped into the record, and its `run_uuid` is the `@id` of the
    CreateAction the response carries.

    Everything happens in a TemporaryDirectory: the adapter's contract is file-in/file-out
    (matching the CLI flags exactly), and re-plumbing it for in-memory use would be a second
    code path to keep correct for no gain.
    """
    from atrium_document import FILE_SUFFIX, load_document

    rows = _top_rows(doc_id, pages)
    if not rows:
        return None, None

    import pandas as pd

    from atrium_document_adapter import schema_error, write_document_record

    with tempfile.TemporaryDirectory() as tmp_dir:
        work = Path(tmp_dir)
        # Separate in/ and out/ dirs: a client whose upload happens to be named
        # <doc_id>.document.json would otherwise have the baseline and the output resolve to
        # the same path.
        in_dir, out_dir = work / "in", work / "out"
        in_dir.mkdir()
        out_dir.mkdir()

        baseline_path: Optional[Path] = None
        if baseline_bytes is not None:
            baseline_path = in_dir / f"{doc_id}{FILE_SUFFIX}"
            baseline_path.write_bytes(baseline_bytes)

        out_path = out_dir / f"{doc_id}{FILE_SUFFIX}"
        write_document_record(
            rdf=pd.DataFrame(rows),
            document_json=str(baseline_path) if baseline_path is not None else None,
            document_json_out=str(out_path),
            paradata_logger=paradata_logger,
        )
        record = load_document(str(out_path))

    return record, schema_error(record, f"{doc_id}{FILE_SUFFIX}")
