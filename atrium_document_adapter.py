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

Both entry points funnel into _write_one(), which is therefore this repo's SINGLE Layer D
chokepoint (atrium-project#10, D4/D8): the schema gate and the field-survival assertion live
there once, rather than once per caller.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from atrium_document import FILE_SUFFIX, SCHEMA_FILENAME, DocumentRecord, load_document, validate_document

PROGRAM = "page-classification"

#: This tool's field-level grant in the `pages` block — must stay a subset of
#: BLOCK_FIELD_OWNERS["pages"]["page-classification"]. Named once, rather than repeated as a
#: literal at the merge_block() call, because it is also what _page_patches() is allowed to
#: emit: the D8 assertion below compares the two, and two copies of the same literal would
#: make that comparison compare a list with itself.
OWN_PAGE_FIELDS = ["category", "category_confidence"]

#: `schema_error()` warns ONCE per process when the gate itself is unavailable, not once per
#: document — a batch run over 400 pages must not bury the one line that matters under 400
#: copies of it.
_schema_gate_disabled_warned = False


def schema_error(record: dict, what: str) -> Optional[str]:
    """Validate one record against `atrium_document.schema.json`.

    Returns None when it validates, or a one-line description of the schema error when it does
    not. This is plan §2's **Layer D** — "no doc.json is emitted if validation fails" — adopted
    here for atrium-project#10 (D4), which found `validate_document()` called from no production
    path in any of the five repos: the gate `docs/document_schema.md` documents as normative was
    protecting nothing at all.

    Deliberately only answers *"is it valid"*. The POLICY — who raises and who merely warns —
    lives at the two call sites in `_write_one()`, because it differs for an inherited baseline
    and for this tool's own output.

    A missing `jsonschema` (RuntimeError from `validate_document()`), a module vendored without
    its schema (FileNotFoundError from `load_schema()`) or an unparseable schema (JSONDecodeError)
    all mean the GATE is absent, not that the record is bad — a `jsonschema.ValidationError` is
    none of those three, so nothing real is swallowed here. They degrade to ONE loud warning and
    a pass, because a gate that silently no-ops is indistinguishable in the output from a gate
    that passed, which is the precise failure mode D4 is about. `jsonschema` is declared in
    setup/requirements.txt (the runtime install every image builds from) and in
    setup/requirements-test.txt, so the degraded path should never be taken in a supported
    deployment.
    """
    global _schema_gate_disabled_warned
    try:
        validate_document(record)
    except (RuntimeError, FileNotFoundError, json.JSONDecodeError) as exc:
        if not _schema_gate_disabled_warned:
            print(
                f"[document] WARNING - schema validation is DISABLED for {what} and every record after it: {exc}",
                file=sys.stderr,
            )
            _schema_gate_disabled_warned = True
        return None
    except Exception as exc:
        # jsonschema.ValidationError: `.message` is the human-readable half and `.json_path`
        # points at the offending node. Both are absent on any other validator, hence getattr.
        detail = getattr(exc, "message", None) or str(exc)
        path = getattr(exc, "json_path", "") or ""
        return f"{detail}{f' at {path}' if path else ''}"
    return None


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
    run_id = getattr(paradata_logger, "run_id", None)
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

    # Layer D, first half (atrium-project#10, D4): judge the baseline as it ARRIVED, before
    # DocumentRecord.open() re-reads it below, so the verdict is about the UPSTREAM tool's output
    # and not about anything this run has since applied to it. An invalid baseline only warns:
    # refusing to run because alto-postprocess wrote something the schema rejects would turn one
    # bad record into a stalled pipeline, and rule 6 already commits to passing unknown content
    # through. It also sets the severity of the second half — a schema error we inherited is not
    # ours to fail on.
    baseline_was_invalid = False
    if baseline and os.path.exists(baseline):
        baseline_error = schema_error(load_document(baseline), f"baseline {Path(baseline).name}")
        if baseline_error:
            baseline_was_invalid = True
            print(
                f"[document] WARNING - inherited baseline {Path(baseline).name} does not validate "
                f"against {SCHEMA_FILENAME} ({baseline_error}) - continuing anyway (rule 6), and "
                f"demoting this run's own output check to a warning",
                file=sys.stderr,
            )

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
        own_fields=OWN_PAGE_FIELDS,
    )
    # Issue #18 §1b, adopted for atrium-project#10 (D8). merge_block() filters silently by
    # design, and `jsonschema` cannot see the loss either — `pages[]` requires only `page`, so a
    # row stripped of its `category` is a *valid* row. That is how one careless edit would
    # produce records with page rows and no categories in them that still pass Layer D.
    #
    # No `fields=` argument on purpose: the default is "every field present on the incoming
    # rows", i.e. whatever _page_patches() actually built. Passing OWN_PAGE_FIELDS here instead
    # would compare the grant against itself and could never fail. As written, the moment
    # _page_patches() starts emitting a field OWN_PAGE_FIELDS does not cover — the realistic
    # regression, since the two live in different functions — this raises instead of dropping it.
    # Let it raise: that is a code bug to catch at dev time, not data variance. Preferred over
    # the global warn_dropped_fields=True, which the module's own docstring notes needs a
    # call-site cleanup pass first.
    doc.assert_fields_survived("pages", page_patches)

    if classification_csv_ref:
        doc.add_derived_from("classification", classification_csv_ref)

    if paradata_logger is not None:
        license_block = paradata_logger.get_license_block()
        if license_block:
            doc.add_license_detail(license_block)

    record = doc.to_dict()

    # Layer D, second half (atrium-project#10, D4): never EMIT an invalid record. This runs
    # BEFORE the write below, which is the whole point — raising here means nothing reaches disk
    # and the next tool never loads a record this one knew was broken. The one exception is a
    # baseline that was already invalid on arrival: the defect is then inherited rather than
    # ours, and failing on it would contradict the warn-and-continue decision taken above.
    own_error = schema_error(record, f"{doc_id}{FILE_SUFFIX}")
    if own_error:
        if baseline_was_invalid:
            print(
                f"[document] WARNING - {doc_id}{FILE_SUFFIX} does not validate against "
                f"{SCHEMA_FILENAME} ({own_error}) - emitting it anyway because the baseline was "
                f"already invalid; fix the upstream record first",
                file=sys.stderr,
            )
        else:
            raise RuntimeError(
                f"page-classification's own document record for {doc_id} does not validate "
                f"against {SCHEMA_FILENAME}: {own_error} - refusing to emit it (Layer D)"
            )

    # Atomic write: build the record ourselves (doc.finalize() also works, but writes
    # in place — a crash mid-write would leave a corrupt record for the next tool to trip
    # over `load_document()` on).
    out_p = Path(out_path)
    os.makedirs(out_p.parent or ".", exist_ok=True)
    tmp_p = out_p.with_suffix(out_p.suffix + ".tmp")
    with open(tmp_p, "w", encoding="utf-8") as fh:
        json.dump(record, fh, ensure_ascii=False, indent=2)
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
