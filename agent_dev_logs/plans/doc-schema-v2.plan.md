# Strategy — [atrium-project#84](https://github.com/ufal/atrium-project/issues/84) (doc-schema-v2): page-classification's part

> _**Refreshed 2026-11-28**: Status — 11-28 (merged and released as `v2.0.0-beta`; the hub's lanes, the chain sample at the tags and `doc-schema-v2` next)._
> _**Refreshed 2026-11-27 (evening)**: Status — 11-27 (evening) (ARÚP's answers on atrium-project#84, comment 7; the PR adjusted)._
> _**Refreshed 2026-11-27**: Status — 11-27 (the non-A options of atrium-project#84 prepared, not pushed)._
> _**Refreshed 2026-11-25**: Status — 11-25 (the v2.0.0-beta release prep)._
> _**Refreshed 2026-11-19**: "Status — 11-19": #74 and #75 merged; D33 resolved (no categories to the ATR); the
> wording that has the classifier route pages._
> _**Refreshed 2026-11-14 (afternoon)**: "Status — round 6 (11-14)": the hub merged, the pins moved, round 6._
> _**Refreshed 2026-11-14**: "Status — PR (11-14)": the PR's checks and round 5's fixes._
> _**Refreshed 2026-11-13**: "Status — round 2" (the code, pushed 11-13 to `doc-schema-v2`) and round 4's notes. The file keeps
> the name `doc-schema-v2.plan.md` (it was `84.plan.md` on 11-12 and 11-13)._
> _**Written 2026-11-12** on ARÚP's output contract of 12 November and atrium-project#84 (motyc, 11-12), at this
> checkout's head `cc35617` (= `test`). The hub carries the whole change:
> [plan](https://github.com/ufal/atrium-project/blob/test/agent_dev_logs/plans/84.plan.md) ·
> [digest](https://github.com/ufal/atrium-project/blob/test/agent_dev_logs/digests/84.digest.md) ·
> contract digest.
> This file is the repository's slice only. It replaces the 11-11 slice of the same name, which had the classifier
> on every PDF and image; 12 November changes that._

**Status (2026-11-28)**: 🟢 **Released as `v2.0.0-beta` (`9d5f4b8`, 11-28): #79 squash-merged (`a7ccb60`, atrium-project#84's decisions 1, 4, 5) after the re-pin to the hub's `ff0a8ea` (`v1`), with the two demonstration runs from the cluster (runner `f457430`); the release run is green and the images are certified. Next, in the hub: the chain sample re-run at this tag, both E2E lanes on the `2.0.0-beta` images, and `doc-schema-v2`'s tag (motyc is asked on atrium-project#84). See the DEVLOG entry of 11-28.**

**Status (2026-11-27, evening)**: 🟡 **ARÚP answered atrium-project#84 (comment 7). #79 is adjusted: the caller selects the pages (forced ∪ pages with `figure` blocks, the hub's shared `atrium_page_selection`) and `/predict_document` records `pages` and `page_selection` with the rule's statement (decision 4); the rendered PDF is named `stored: false` (decision 5); `run.py` keeps settings and models in `config` and names the files it reads and writes as entries (decision 1). A bug found on the way: the record adapter hashed an in-place baseline after overwriting it. See the DEVLOG entry of 11-27 (evening).** _(superseded 11-28: merged and released)_

**Status (2026-11-27)**: 🟡 **The non-A options of atrium-project#84 prepared (not pushed; ARÚP has not answered): `/predict_document` names its PDF by the distribution it is (`original` or `rendered-pdf`, from the record's media type; decision 5, R7); in the pipeline it reads the forced pages and the pages that show graphics (decision 4); the command line's paradata name files, not paths, and record `pages` (decision 1). One PR, after the hub's. See the DEVLOG entry of 11-27.** _(superseded 11-27 evening: ARÚP answered)_

**Status (2026-11-25)**: 🟢 **v2.0.0-beta release prep in #78 (`v2-beta-prep` → `test`), CI green; the two demonstration runs are this version's (runner `466aeb7`, checked 11-25). Tag 26 November after the merge.** _(superseded 11-28: tagged `v2.0.0-beta` on 11-28, after ARÚP's answers)_

**Status (2026-11-19)**: 🟢 **This repository's part is on `test` (#71, #74), pinned to `02cbd01`. D33 is resolved:
the page categories are not sent to the ATR (Ronald, ask 6; motyc agreed); they stay in the record. Left: two
sentences that still have the classifier decide the processing (a docs PR under atrium-project#84); the release
(`v2.0.0-beta`) with the batch (atrium-project#77).** _(superseded 11-25: the release prep)_

**Status (2026-11-14, afternoon)**: 🟡 **The hub's PR is merged and this repository is pinned to it: drift is green.
Round 6 completes this repository's part where no answer is pending; see "Status — round 6". The release (v2.0.0-beta)
is cut at K4TEL's cadence.** _(superseded 11-19: the answers in)_

**Status (2026-11-14)**: 🟡 **[atrium-page-classification#71](https://github.com/ufal/atrium-page-classification/pull/71)
is open (`doc-schema-v2` into `test`); see "Status — PR". Drift clears once the hub's PR merges and the pins move; the
record-major check clears at this tool's next release, which raises its major version.** _(superseded 11-14 afternoon)_

**Status (2026-11-13)**: 🟡 **Implemented, pushed 11-13 to `doc-schema-v2`; see "Status — round 2". Deferred items are listed there.** _(superseded 11-14: the PR is open)_

**Status (2026-11-12)**: 📋 **Planned, plans only; small. page-classification runs on the pages that need ATR only,
beside the ATR (H3): the `needs_atr` pages of a PDF, and every page of an image. Its output is unchanged —
`page_categories` and each page's `category`/`category_confidence` in the record — and may be offered to the ATR as
an optional weak signal, never as routing (D33). Code follows the hub's shared modules.** _(11-19: not offered at
all — D33 resolved)_

## Its fields

| Record node                                   | In v2                                                                                       |
|-----------------------------------------------|---------------------------------------------------------------------------------------------|
| `page_categories`                             | written                                                                                     |
| `pages[].category`, `category_confidence`     | written, for the selected pages (decision 4); joined on `pages[].page_index` (H2)           |
| `derived_from.classification`                 | stops (D1): `atrium_document_adapter.py:223`                                                |
| run data, licence detail                      | none (D1, D2), through the re-vendored `atrium_document.py`                                 |

## Tasks (next rounds)

1. **The forced and `figure` pages** (H3; decision 4, 11-27; the `needs_atr` pages only until then).
   `/predict_document` (`service/api.py:795`) already takes a `pages` selection (its `pages` form field;
   `expand_page_selection()`); the pipeline sends the original and the selected page numbers (forced ∪ `figure` pages,
   the shared `atrium_page_selection`) from digital-convert's docproc; the tool records `pages` and `page_selection`.
   For an image, the whole image. Nothing in the tool decides which pages.
2. **Page keys by `page_index`** (H2). The rows are keyed by the record's page label today
   (`page_inputs.pdf_page_keys()`, `page_inputs.py:129`; `service/document_json.record_page_keys()`,
   `service/document_json.py:138`); with `page_index` on every page row the label mapping and the baseline input are
   not needed. The record builder writes the page rows first (ocr-postprocess), so this step adds fields to existing
   rows.
3. **`MAX_PDF_PAGES`** (`tool_limits.py:24`, 50; checked on the selection, `service/api.py:337-361`) refuses rather
   than trims, as D13 wants. A mixed or scanned document with more than 50 selected pages would fail the job: the
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
[hub plan of #84](https://github.com/ufal/atrium-project/blob/test/agent_dev_logs/plans/84.plan.md),
"Status — PRs".

Round 5 (not pushed):

- `atrium_rocrate.py` and `tests/test_atrium_rocrate.py` re-vendored: a crate root that asserts no licence names
  `#no-licence-asserted` again (RO-Crate 1.2 requires a root licence; atrium-project#6, motyc 11-12). The hub's PR
  failed its RO-Crate check without it.

The suite passes (offline lane): 875. After the hub's PR merges, `v1` moves to its merge commit and
`scripts/pin_hub_reusables.py --sha` pins this repository's callers to it: drift clears. The record-major check stays
red until the release, which raises the major version in `setup/para_config.txt` (`v1.14.2-beta` now), `CITATION.cff`
and `service/openapi.json`'s `info.version`.

## Status — round 6 (11-14)

The hub's PR (atrium-project#86) merged; `v1` moved to it and K4TEL pinned this repository's callers: drift green. Round
6 (local, delivered as an archive):

- `atrium_document.py`, its schema and `tests/test_document_originators.py` re-vendored: rows may carry
  `lines[].unit: "block"` (rule 5, written by ocr-postprocess only), and `quality_summary.lines` counts them as
  `block_units`. `v2.0.0-beta` as the unreleased version: `setup/para_config.txt`, `CITATION.cff`,
  `service/openapi.json` (re-exported) and a CONTRIBUTING.md row marked Unreleased; api-contract passes with it.

The suite passes: 876 (offline lane). Next: the hub's second PR for #84 (the E2E lanes, rule 5's mark), then `v1` and
the pins again; then this PR. Hub:
[plan of #84](https://github.com/ufal/atrium-project/blob/test/agent_dev_logs/plans/84.plan.md),
"Status — round 6".

## Status — round 8 (11-14)

The hub's second PR (atrium-project#87) merged and the six repositories' PRs were re-pinned to it; four have merged. On
this PR, `docker-build-smoke (api)` failed with exit 139 when the image was stopped during its model warm-up: in the
container the service is PID 1, so the SIGTERM uvicorn raises again after its graceful exit is ignored, `uvicorn.run()`
returns, and Python's teardown met the warm-up thread inside torch (dependabot's #69 had hit the same on 11-10). The
service now ends with `os._exit(0)` when it stops while the warm-up still runs (`service/api.py`,
`_exit_now_if_warming()`; three tests in `tests/test_service_warmup.py`). Dependabot's #69 (`/service`) and #70
(`/setup`) each raise `pypdfium2` in one manifest only, which `tests/test_service_runtime_deps.py` refuses: #70 takes
the `/service` raise too after this PR merges, and #69 is closed.

## Status — round 9 (11-14)

#71 merged (`bfebf36`); the `/service` raise went to `test` as `111def2`, then #70 merged (`ed9105d`), and the `api`
smoke passed. #69 did not close: dependabot rebased it to `pypdfium2>=5.15.2` in `/service` only, the same split.
`.github/dependabot.yml` now has one pip entry for both directories; a `shared-deps` group with
`group-by: dependency-name` (Pillow, pypdfium2) gives each dependency declared in both manifests one PR that raises
both, and `tests/test_service_runtime_deps.py` checks that every such dependency is in it. #69 is closed once this is
merged; dependabot opens the combined `pypdfium2` PR on its next run.

## Status — the meeting's follow-ups (11-16 night)

atrium-project#84 round 8 (comment 3's update): `forms` gone, decode verdicts apart, one line key, the services' JSON in the record's names. Paradata per distribution waits for the next meeting (Q23). Not pushed.

- **The record's names.** `/predict_document`'s page entries: `page_index` (was `page`) and `page`, the label, when the baseline has one (was `page_label`); the frontends show the label, else the index.
- Re-vendored shared modules; `service/openapi.json`; service README, CONTRIBUTING.
- **Checked:** 882 passed.
- **Next:** the hub's PR first; then this repository's, with the pins moved to it.

## Status — 11-19: D33 resolved

- **On `test`:** #74 (round 8, 11-16) and #75 (dependabot: `ultralytics` in `setup/requirements.txt`, 11-18).
  `master` is `1e80e7a`, not `test`.
- **Ronald, ask 6** (11-17, [hub digest](https://github.com/ufal/atrium-project/blob/test/agent_dev_logs/digests/84.digest.md#-2026-11-19--ronalds-answers-the-open-atrs-draft-the-storage-contract)): the categories are of no use to the ATR; not sent. motyc agreed.
  So the classifier's output goes only into the record (`page_categories`, `pages[].category`,
  `category_confidence`), for description and later analysis; nothing here sent it anywhere else.
- **Ask 8:** the ATR has no handwriting type; the `_HW` categories remain the only handwriting signal in the record.
- **Wording to fix (later, one PR under atrium-project#84):** `README.md:5` ("The category tells which processing
  the page needs next, for example OCR or table extraction") and `CITATION.cff:7` ("tells which processing the page
  needs next") contradict H3 and D33: digital-convert's verdict decides which pages go to the ATR. The CAA paper has
  the same point (hub `digests/13.findings.md`, M2).
- **aiscr-docs-pipeline#2** (this step) still says "a category for every page": a question for the meeting.

## Depends on

_(11-19: hub tasks done; the pipeline's hand-over of the selected pages (forced ∪ `figure` pages since
decision 4) is aiscr-docs-pipeline#12, due 12-04.)_

- Hub tasks 1 (grants; `page_index`), 3 (`file_entity` with SHA-512), re-vendoring.
- ocr-postprocess's slice: a page row for every page before this step writes.
- aiscr-docs-pipeline: how the selected pages and large documents are passed (task 3).

## Superseded in this repository's logs

- The 11-11 slice (not committed): page classification on every PDF and image, bundled with the ATR call →
  the selected pages beside the ATR (H3; forced ∪ `figure` pages, decision 4).
- `plans/58.plan.md`, `digests/58.digest.md`: parked (atrium-project#83); their 11-11 link to the slice now points
  here; the TrueOpen.AI note of 11-12.

## Not in scope

- #58's decision models (parked); the TrueOpen.AI request (option B, parked; Kate answers them).
- Routing (the pipeline's, from digital-convert's verdicts); the ATR (ARÚP's).
