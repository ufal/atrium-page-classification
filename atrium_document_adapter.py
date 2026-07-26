"""
atrium_document_adapter.py — page-classification's contribution to the ATRIUM "paradata pair".

Bridges the classifier's per-page top-N predictions (a FILE/PAGE/CLASS-N/SCORE-N DataFrame,
see utils.dataframe_results / parallel_best.average_rdfs) into AtriumDocument records via
atrium_document.DocumentRecord, writing only this tool's owned block(s):

  * `page_categories`             — whole block, owned outright (BLOCK_OWNERS).
  * `pages[].category`            — field-level, shared with alto-postprocess/nlp-enrich
  * `pages[].category_confidence`   (BLOCK_FIELD_OWNERS), written via merge_block() so a
                                    co-contributor's fields on the same page row survive.

Two entry points mirror run.py's two CLI shapes exactly, so neither has to guess at the other's
path semantics from a shared, overloaded pair of kwargs:

  * write_document_record()      — single-document run. `--document-json` / `--document-json-out`
                                    are themselves FILE paths (accretion contract rule 1).
  * write_document_records_dir() — batch/dir run. `--document-json-dir` / `--document-json-out-dir`
                                    are DIRECTORIES, one `<doc_id>.document.json` per document.

write_document_records_dir() also handles the case where a document's pages are split across two
chunks of the same run: for doc_id's already written earlier in *this* run, the baseline is read
back from the output dir (which already carries this run's own earlier pages) rather than from the
original (pre-classification) input dir, so a later chunk cannot clobber an earlier chunk's pages
for the same document.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from atrium_document import DocumentRecord

PROGRAM = "page-classification"


def _page_patches(group: pd.DataFrame) -> tuple[dict, list]:
    """Build (page_categories, pages[]-patch) from one document's prediction rows."""
    page_categories: dict = {}
    page_patches: list = []

    for _, row in group.iterrows():
        page_str = str(row["PAGE"])
        cat = row["CLASS-1"]

        page_categories[page_str] = cat
        patch = {"page": page_str, "category": cat}
        # Only add confidence if it exists (handles top_N == 1 missing score safely)
        if "SCORE-1" in row and pd.notna(row["SCORE-1"]):
            patch["category_confidence"] = float(row["SCORE-1"])

        page_patches.append(patch)

    return page_categories, page_patches


def _run_id_and_ref(paradata_logger: Any) -> tuple[Optional[str], str]:
    if paradata_logger is None:
        return None, ""
    run_id = getattr(paradata_logger, "_run_id", None)
    ref = f"paradata/{run_id}_{PROGRAM}.json" if run_id else ""
    return run_id, ref


def _write_one(
    doc_id: str,
    group: pd.DataFrame,
    *,
    baseline: Optional[str],
    out_path: str,
    classification_csv_ref: Optional[str],
    paradata_logger: Any,
    strict: bool,
) -> str:
    """Build one document's record and atomically write it to out_path."""
    page_categories, page_patches = _page_patches(group)
    run_id, paradata_ref = _run_id_and_ref(paradata_logger)

    doc = DocumentRecord.open(
        doc_id,
        PROGRAM,
        baseline=baseline,
        run_id=run_id,
        paradata_ref=paradata_ref,
        strict=strict,
    )

    # `page_categories` is a whole-block, single-owner block (rule 2: "own block only"), so a
    # set_block() call replaces it entirely. That's only correct if the payload already reflects
    # this tool's FULL knowledge of the block. Across a single run that's true — but when a
    # document's pages are split across chunks (see write_document_records_dir's baseline
    # resolution), a later chunk's set_block() would otherwise wipe out an earlier chunk's
    # entries for the same doc_id. Merge with whatever this tool itself already wrote, so a
    # chunked self-run behaves like one full-block write.
    existing_categories = doc.to_dict().get("page_categories") or {}
    doc.set_block("page_categories", {**existing_categories, **page_categories})
    doc.merge_block(
        "pages",
        page_patches,
        key_fields=["page"],
        own_fields=["category", "category_confidence"],
    )

    if classification_csv_ref:
        doc.add_derived_from("classification", classification_csv_ref)

    if paradata_logger is not None:
        license_block = paradata_logger.get_license_block()
        if license_block:
            doc.add_license_detail(license_block)

    # Atomic write: build the record ourselves (doc.finalize() also works, but writes
    # in place — a crash mid-write would leave a corrupt record for the next tool to trip
    # over `load_document()` on).
    out_p = Path(out_path)
    os.makedirs(out_p.parent or ".", exist_ok=True)
    tmp_p = out_p.with_suffix(out_p.suffix + ".tmp")
    with open(tmp_p, "w", encoding="utf-8") as fh:
        json.dump(doc.to_dict(), fh, ensure_ascii=False, indent=2)
    tmp_p.replace(out_p)
    return str(out_p)


def write_document_record(
    rdf: pd.DataFrame,
    *,
    document_json: Optional[str] = None,
    document_json_out: Optional[str] = None,
    classification_csv_ref: Optional[str] = None,
    paradata_logger: Any = None,
    strict: bool = False,
) -> Optional[str]:
    """
    Single-document run: `document_json` / `document_json_out` are FILE paths
    (`--document-json` / `--document-json-out`). `rdf` is expected to describe exactly
    one document (one distinct FILE value) — the -f/--file single-image CLI path.
    """
    if not document_json_out or rdf.empty:
        return None

    doc_id = str(rdf.iloc[0]["FILE"])
    return _write_one(
        doc_id,
        rdf,
        baseline=document_json,
        out_path=document_json_out,
        classification_csv_ref=classification_csv_ref,
        paradata_logger=paradata_logger,
        strict=strict,
    )


def write_document_records_dir(
    rdf: pd.DataFrame,
    *,
    document_json_dir: Optional[str] = None,
    document_json_out_dir: Optional[str] = None,
    classification_csv_ref: Optional[str] = None,
    paradata_logger: Any = None,
    strict: bool = False,
) -> list:
    """
    Batch run: `document_json_dir` / `document_json_out_dir` are DIRECTORIES
    (`--document-json-dir` / `--document-json-out-dir`), one `<doc_id>.document.json` per
    document. `rdf` may cover several documents (-d/--directory, chunked or not) — grouped
    by FILE so each gets its own record.
    """
    if not document_json_out_dir:
        return []

    written = []
    for doc_id, group in rdf.groupby("FILE"):
        doc_id = str(doc_id)
        out_path = str(Path(document_json_out_dir) / f"{doc_id}.document.json")

        # Prefer this run's own prior chunk output for this doc_id (so a document whose
        # pages straddle two chunks doesn't have its earlier chunk's pages clobbered by a
        # later one starting over from the pristine upstream baseline) and fall back to the
        # upstream baseline dir otherwise.
        baseline = None
        if os.path.exists(out_path):
            baseline = out_path
        elif document_json_dir:
            candidate = Path(document_json_dir) / f"{doc_id}.document.json"
            if candidate.exists():
                baseline = str(candidate)

        written.append(
            _write_one(
                doc_id,
                group,
                baseline=baseline,
                out_path=out_path,
                classification_csv_ref=classification_csv_ref,
                paradata_logger=paradata_logger,
                strict=strict,
            )
        )

    return written
