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
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


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
        )
        record = load_document(str(out_path))

    return record, schema_error(record, f"{doc_id}{FILE_SUFFIX}")
