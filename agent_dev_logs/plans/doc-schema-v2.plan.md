# Strategy — [atrium-project#84](https://github.com/ufal/atrium-project/issues/84) (doc-schema-v2): page-classification's part

> _**Refreshed 2026-11-14**: "Status — PR (11-14)": the PR's checks and round 5's fixes._
> _**Refreshed 2026-11-13**: "Status — round 2" (the code, pushed 11-13 to `doc-schema-v2`) and round 4's notes. The file keeps
> the name `doc-schema-v2.plan.md` (it was `84.plan.md` on 11-12 and 11-13)._
> _**Written 2026-11-12** on ARÚP's output contract of 12 November and atrium-project#84 (motyc, 11-12), at this
> checkout's head `cc35617` (= `test`). The hub carries the whole change:
> [plan](https://github.com/ufal/atrium-project/blob/test/agent_dev_logs/plans/doc-schema-v2.plan.md) ·
> [digest](https://github.com/ufal/atrium-project/blob/test/agent_dev_logs/digests/doc-schema-v2.digest.md) ·
> [contract digest](https://github.com/ufal/atrium-project/blob/test/agent_dev_logs/digests/project_state_1211.md).
> This file is the repository's slice only. It replaces the 11-11 slice of the same name, which had the classifier
> on every PDF and image; 12 November changes that._

**Status (2026-11-14)**: 🟡 **[atrium-page-classification#71](https://github.com/ufal/atrium-page-classification/pull/71)
is open (`doc-schema-v2` into `test`); see "Status — PR". Drift clears once the hub's PR merges and the pins move; the
record-major check clears at this tool's next release, which raises its major version.**

**Status (2026-11-13)**: 🟡 **Implemented, pushed 11-13 to `doc-schema-v2`; see "Status — round 2". Deferred items are listed there.** _(superseded 11-14: the PR is open)_

**Status (2026-11-12)**: 📋 **Planned, plans only; small. page-classification runs on the pages that need ATR only,
beside the ATR (H3): the `needs_atr` pages of a PDF, and every page of an image. Its output is unchanged —
`page_categories` and each page's `category`/`category_confidence` in the record — and may be offered to the ATR as
an optional weak signal, never as routing (D33). Code follows the hub's shared modules.**

## Its fields

| Record node                                   | In v2                                                                                       |
|-----------------------------------------------|---------------------------------------------------------------------------------------------|
| `page_categories`                             | written                                                                                     |
| `pages[].category`, `category_confidence`     | written, for the pages that needed ATR; joined on `pages[].page_index` (H2)                 |
| `derived_from.classification`                 | stops (D1): `atrium_document_adapter.py:223`                                                |
| run data, licence detail                      | none (D1, D2), through the re-vendored `atrium_document.py`                                 |

## Tasks (next rounds)

1. **The `needs_atr` pages only** (H3). `/predict_document` (`service/api.py:795`) already takes a `pages` selection
   (its `pages` form field; `expand_page_selection()`); the pipeline sends the original and the `needs_atr` page
   numbers from digital-convert's routing. For an image, the whole image. Nothing in the tool decides which pages.
2. **Page keys by `page_index`** (H2). The rows are keyed by the record's page label today
   (`page_inputs.pdf_page_keys()`, `page_inputs.py:129`; `service/document_json.record_page_keys()`,
   `service/document_json.py:138`); with `page_index` on every page row the label mapping and the baseline input are
   not needed. The record builder writes the page rows first (ocr-postprocess), so this step adds fields to existing
   rows.
3. **`MAX_PDF_PAGES`** (`tool_limits.py:24`, 50; checked on the selection, `service/api.py:337-361`) refuses rather
   than trims, as D13 wants. A mixed or scanned document with more than 50 `needs_atr` pages would fail the job: the
   limit is raised for the pipeline, or the pipeline sends the pages in chunks (H1: the cluster side chunks). To
   settle with aiscr-docs-pipeline when the step is assembled.
4. **Paradata** (#84 §4). One action in `paradata/atrium/document-json`: object = the original by distribution name
   and SHA-512, with the pages classified as a parameter; result = `page_categories` and `pages` with the fields
   written. `predictions.json` (`service/api.py:626`) is not a stored output: it leaves the result.
5. **The weak signal** (D33): if Ronald takes it up, the pipeline passes the categories to the ATR; nothing to build
   here beyond the existing output.

## Status — round 2 (11-13)

Local branch from `test`, on the re-vendored shared modules (record 2.0, paradata 3.0,
`ContentLimitError`); delivered as an archive with the commands to run; pushed 11-13 (below). Nothing posted.

- **Task 2, `page_index`:** the adapter writes `{page_index, category, category_confidence}` rows keyed by
  `page_index`; `page_categories` is keyed by the baseline's label for that page, else `str(page_index)`.
  `page_inputs.py` and `service/document_json.py` key by `page_index` and read a label only from a baseline.
- **Task 4, paradata:** one action. `object`: the upload as `original` and the record read; `result`: the record's
  fields written; `pages` classified as a parameter; `predictions.json` left out. No `derived_from`, no licence.
- **Tasks 1, 3, 5:** no code. The pipeline sends the pages; `MAX_PDF_PAGES` still refuses (question for the
  pipeline step).
- **Docs and spec:** how-it-works, usage, outputs, paradata, licences, service README; `service/openapi.json`
  regenerated.
- **Tests:** 873 passed, 16 skipped (offline lane, -m "not slow").
- **Pushed** to `doc-schema-v2` (11-13). Round 4 re-vendors the addendum's shared files; the suite then passes (873,
  offline lane).
- **[atrium-digital-convert#9](https://github.com/ufal/atrium-digital-convert/issues/9#issuecomment-6838301982)**
  (point 5, agreed 11-13): of an office file, as of a PDF, only the pages that need the ATR come here (H3). Nothing
  changes in the tool.

## Status — PR (11-14)

[atrium-page-classification#71](https://github.com/ufal/atrium-page-classification/pull/71), head `d37a8bb`. Failing
checks: check-drift (canonical), because the callers read the hub at `v1` (`4435ffb`) while the copies are v2;
api-contract, because the record schema went from 1.0 to 2.0 without a major version. The causes across all seven PRs:
[hub plan of #84](https://github.com/ufal/atrium-project/blob/test/agent_dev_logs/plans/doc-schema-v2.plan.md),
"Status — PRs".

Round 5 (not pushed):

- `atrium_rocrate.py` and `tests/test_atrium_rocrate.py` re-vendored: a crate root that asserts no licence names
  `#no-licence-asserted` again (RO-Crate 1.2 requires a root licence; atrium-project#6, motyc 11-12). The hub's PR
  failed its RO-Crate check without it.

The suite passes (offline lane): 875. After the hub's PR merges, `v1` moves to its merge commit and
`scripts/pin_hub_reusables.py --sha` pins this repository's callers to it: drift clears. The record-major check stays
red until the release, which raises the major version in `setup/para_config.txt` (`v1.14.2-beta` now), `CITATION.cff`
and `service/openapi.json`'s `info.version`.

## Depends on

- Hub tasks 1 (grants; `page_index`), 3 (`file_entity` with SHA-512), re-vendoring.
- ocr-postprocess's slice: a page row for every page before this step writes.
- aiscr-docs-pipeline: how the `needs_atr` pages and large documents are passed (task 3).

## Superseded in this repository's logs

- The 11-11 slice (not committed): page classification on every PDF and image, bundled with the ATR call →
  `needs_atr` pages only, beside the ATR (H3).
- `plans/58.plan.md`, `digests/58.digest.md`: parked (atrium-project#83); their 11-11 link to the slice now points
  here; the TrueOpen.AI note of 11-12.

## Not in scope

- #58's decision models (parked); the TrueOpen.AI request (option B, parked; Kate answers them).
- Routing (the pipeline's, from digital-convert's verdicts); the ATR (ARÚP's).
