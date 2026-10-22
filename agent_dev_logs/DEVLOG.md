# 📓 atrium-page-classification — agent_dev_logs/DEVLOG.md (timeline index)
> _Historical page-image classification. Open issue: #58 (decision-model experiments, 2026-10-20). #61 (the
> directory-run bug) closed 2026-10-22, fixed in **v1.9.6-beta**. #48 closed 2026-09-17; #15 and #26 closed; four open
> Dependabot PRs (#56, #57, #59, #60). Its pilot work is tracked in hub issues (atrium-project #6, #67, #72) and in
> atrium-digital-convert#5. `test` = `f6ef687` (2026-10-22, the `vX.4` results of atrium-project#15, run with
> **v1.9.8-beta**) · **v1.9.9-beta** prepared (the results' documentation, the evaluation paradata) · **v1.9.5-beta**
> (2026-10-19, the runbook) is the first immutable release (atrium-project#72)._
> _Per-issue detail: `digests/58.digest.md` · `plans/58.plan.md`, `digests/61.digest.md` · `plans/61.plan.md`._
> _Cross-repo/hub history lives in `ufal/atrium-project/agent_dev_logs/DEVLOG.md` (deduplicated out of this file)._

## 2026-06-25
- **#15 Retrain 5 best models on the new dataset (N−318 pages)** — Opened by K4TEL: retrain the five best models —
EffNet `model_1241` (CV split 1, seed 420), RegN `model_741` (split 1, seed 420), Vb-2 `model_245` (split 5, seed 424),
Vb-3 `model_342` (split 2, seed 421), Vl-3 `model_542` (split 2, seed 421) — on the licensed dataset minus the 318 removed
pages; attached `licensed_crossval_folds_CUT.csv`. Confirms the atrium-project **#21** dataset release: it becomes the
key reference once the re-finetuned models reach the same accuracy on the same (minus-318) eval data.

## 2026-06-27
- **#15** — Eval fold path committed (`b86c431`): `--eval --folds_csv` scores only the chosen fold's `test` pages from
the global EVAL dir (single-model only, not `--best` — each best model uses a different fold). Builds on the fold-CSV
**training** path already merged on `test` (`992c5df`, `split_data_from_folds` + `REVISION_BEST_FOLDS` + the v*.4 registry
, released in **v1.5.0-beta** with the paradata-template sync and `agent_dev_logs/`). Second fine-tuning round **launched**
on the cropped dataset.

## 2026-06-28
- **#15** — `f0643a0` fixes the train-finishing evaluation path.

## 2026-06-29
- **#15** — Retrain **evaluated**: vX.4 vs vX.3 prediction diff = **24 mismatches of 229 samples** (`diff_models_3-4.csv`,
down from 25) — **all ambiguous cases**. Conclusion: removing the 318 pages only affects cases the vX.3 best-5 ensemble
already disagreed on; obvious category-specific classification does not suffer. Pushing the vX.4 models to the HF hub
is possible if needed — **awaiting the call**.

## 2026-07-12
- **#15** — `55e604c` lands `tests/test_folds_split.py` (18 tests: routing, −318/absent-page drop, NA/whitespace/case,
eval subset, registry ordering, real-CSV `slow` check) + the shared `tests/test_para_licenses.py`; fixed a pandas
≥3.0 NA-preservation crash in `split_data_from_folds` (~225 blank `fold1` cells — now `pd.isna`-guarded). Released
**v1.5.1-beta** (licenses test per template, fixed automatic version reading via `_read_tool_version()`, dependency
bumps, GHA fixes). Digest+plan refreshed to the committed reality — remaining decision: the HF-hub push / canonical vX.4 release.
- **#26 Agent-skill branch for page classifier prediction** — Opened by K4TEL: package the classifier as an agent
skill on the [`honzas83/uwebasr-skill`](https://github.com/honzas83/uwebasr-skill) pattern — `SKILL.md` (frontmatter + agent guidelines) + a **stdlib-only
client** (`scripts/atrium_classify.py`) wrapping the existing FastAPI service, with a two-phase workflow (ensure-server-
up via `/info`, then classify) so the same client flips to kosarko's future LINDAT endpoint via `--base-url` / `ATRIUM_PC_URL`
only. The [`agent-skill`](https://github.com/ufal/atrium-page-classification/tree/agent-skill) branch with cleaned-up files created 08:14; digest+plan pair added (client, `SKILL.md`, `serve.sh`,
multi-agent install docs, smoke fixtures = next).

## 2026-08-02
- **#15** — HF-hub push completed and canonical vX.4 release finalized. Issue closed.
- **#26** — Completed agent-skill integration tasks including the client, `SKILL.md`, `serve.sh`,
multi-agent install docs, and smoke fixtures. Branch merged and issue closed.

## 2026-08-03 – 2026-08-04

* Hub template (`atrium_document.py` / `atrium_document.schema.json`) synced across three commits (`8a4ecce`
"clean up", `bf935ee`/`b56781e` template scripts, `2ed7ba1` formatting), then re-aligned again (`e2d0375`, `f6de640`)
— the same expanding-shared-contract pattern seen ecosystem-wide this window.
* **Real defect fixed**: `469b1d7`/`a1700ed` — `setup/requirements.txt`'s `numpy` pin had drifted past the
Python-3.12-only `2.5.x` line while every image and CI job here still runs **3.11**, exactly the break the hub's
cross-repo audit caught live on `test` on 07-30 (`project_state_3007.md`, finding N1 — `pip` failing outright with
"No matching distribution found"). Fixed here on 08-04 (five days after the hub flagged it) by floor-pinning numpy
back to `<2.5` and adding a `dependabot.yml` ignore rule so the same bump can't recur.

## 2026-08-06

* **v1.7.4-beta.** A further "LLM review+fix round by Opus" (`fe14d24`) lands several things at once:
`atrium_document_adapter.py` +121 lines, `run.py`, a new `service/document_json.py` splitting the per-page-image
(`/predict_image`) and whole-document (`/predict_document`) upload paths onto one shared derivation, and
`service/api.py`/`README.md` updated to match. **Real defect found and fixed in the same commit**:
`service/requirements.txt` had held six pytest/contract dependencies and **no ASGI server at all** since it was
last touched — the same class of defect the hub's cross-repo audit later flagged as N2 for `atrium-page-classification`
in `project_state_3007.md` (07-30), except this repo had already caught and fixed its own instance four days
earlier. Nothing had noticed because the test suite drives the app in-process via `TestClient` (no server needed)
and the hub's docker-build-smoke job only *builds* the image — `docker compose --profile api up api` would have
died at container start with "executable file not found" the first time anyone actually ran it. Fixed by rewriting
the file as a documented runtime manifest and adding `tests/test_service_runtime_deps.py`, which parses
`docker-compose.yml`/`setup_api_service.sh` for the console entrypoints they invoke and asserts each is declared in
an installed requirements file — so pruning the list again fails the fast lane, not just the deployment.
`ruff.toml` gains a full 79-line config (`550fdcd`) — pc had none before this. Two more `atrium_document.py` fix
passes (`ae61664`) and a version bump (`c86037a`); `DocumentRecord` now inherits `doc_id` from the baseline instead
of letting a later stage re-key it.

## 2026-08-18 – 2026-08-19

* Dependabot bumps `pymupdf` (service) and `ultralytics` (setup). **v1.7.5-beta** ships: GHA fixes (`a1e5ba5` —
`gpu-inference.yml`/`release.yml` timeout guards), a further Opus-reviewed round (`7439514`) hardening `codeql.yml`/
`scheduled-smoke.yml`/`security.yml` concurrency scoping (so a push can no longer cancel a cron) plus another
`atrium_document.py` alignment with a new `tests/test_document_originators.py`; `4548deb` fixes a test the previous
commit's changes broke. `857307b` cleans up `.coveragerc`/`.gitignore`.
* Confirmed live: the hub reusable-workflow references are pinned to the tagged **`@v1`** release across all seven
workflows (api-contract, codeql, docker-tool, para-drift, pre-commit, security, workflow-lint) — this repo closed
the cross-repo N8 finding (the `@test`→`@v1` repin) in the same window the other four tool repos did.

## 2026-09-03 – 2026-09-06

* Routine dependency/action bumps (`335bd02` — `softprops/action-gh-release`). `60f58bc` fixes the scheduled-smoke
GHA workflow (adding the timeout/failure-notification pattern the other four repos picked up around the same date).
No functional or categorization changes; no issue activity — the tracker has been at zero open issues since #26
closed on 08-02.

## 2026-09-07

* **State**: 0 open issues. `test` and `vit` (default) both at `60f58bc`, **v1.7.5-beta**. Since the last refresh
(08-02) all work has been infrastructure hardening driven by the hub's ongoing cross-repo LLM-review passes
("atrium-project#10" in commit messages/comments across the ecosystem) — the shared `atrium_document.py` template,
the `@v1` reusable-workflow repin, and GHA concurrency/timeout hardening — plus one real, quickly-fixed defect (the
numpy/Python-3.12 pin mismatch). Recommend this repo gain a `digests/`/`plans/` pair the next time a substantive
issue opens, matching its four siblings.

## 2026-09-25 — doc ids of dotted PDF names (atrium-alto-postprocess#31)

* The hub's `KNOWN_PIPELINE_SUFFIXES` gains `.pdf` and the other text-lines input suffixes. `doc_id_for_document()`
  (`service/document_json.py`) is `canonical_doc_id()` alone, so `CTX01.scan.pdf` is now `CTX01.scan` (it was `CTX01`
  by the first-dot fallback). Decided with K4TEL; the two tests in `tests/test_service_document_json.py` that pinned the
  old answer now pin the new one and must land together with the re-vendored `atrium_document.py`. Image names are
  unchanged (`doc_id_and_page()` strips the image extension and the page label first; no image suffix is listed).
* With the new `atrium_document.py`: `tests/test_service_document_json.py` has exactly the 9 environment failures it has
  without it (no scikit-learn/torch here), and nothing else in the suite moves. Not pushed: files delivered in chat.

## 2026-09-26 — AMČR baseline (atrium-project#67): this repository's pilot items, tracked in the hub

* **What arrived:** [atrium-project#67](https://github.com/ufal/atrium-project/issues/67) (motyc, AMČR) — no issue of
  this repository is in the baseline (#48 closed 09-17), but four of its requests land here. All are tracked in hub
  pairs; nothing is changed in code by this entry.
  * **Licences (hub #6, motyc 16:08):** `/predict_document` rasterises PDFs with PyMuPDF (`import fitz`,
    `service/api.py:120`; `PyMuPDF>=1.28.2` in `service/requirements.txt`, installed in `base`), which is AGPL-3.0 and
    not declared in `setup/para_config.txt [components]`. Recommended: swap to `pypdfium2` (render at
    `PDF_RENDER_DPI / 72`), with a regression run over the PDF samples; the fallback is to declare it.
  * **Limits (hub #53, motyc 16:03):** `MAX_PDF_PAGES = 50` (`service/api.py:58`) and `PDF_RENDER_DPI = 300` (`:64`)
    become environment settings; `/info` reports both (today it shows `max_upload_mb` and `max_pdf_pages`).
  * **Seeded record, paradata (hub #67 R1, R2):** the service already accepts a `document_json` part and writes to an
    explicit path (correct when the seed's `doc_id` differs from the file name; a regression test is planned); it has no
    `ParadataLogger`, so the plan is to add one and return the `CreateAction`.
  * **Record-only (hub #67 R3):** build `page_categories` from an existing predictions table
    (`atrium_document_adapter.py` has the logic, not the entry point).
* **Also 2026-09-26:** the `doc-schema-v1` freeze files and the re-vendored freeze test (blob `7c35fbf1`, `2e27073`).
* **Pair to remove:** `48.*` (issue closed 2026-09-17).
* Not pushed: files delivered in chat.

## 2026-09-30 — atrium-project#72 round 1: PyMuPDF → pypdfium2, the production image declared
* **PDF engine** — `service/pdf_render.py` renders with pypdfium2 (Apache-2.0/BSD-3-Clause) at `PDF_RENDER_DPI / 72`,
  under a process lock (PDFium is not thread-safe); `service/api.py` opens, counts and renders under it and classifies
  outside it; a PDF PDFium cannot open is a 422. `MAX_IMAGE_PIXELS` is checked on PDFium's own pixel size
  (`pixel_size`: it rounds up from a 32-bit page size, so a 595.2 pt page at 300 dpi is 2481 px). PyMuPDF (AGPL-3.0,
  installed and declared nowhere) left `service/requirements.txt`; pypdfium2 declared `conditional` in
  `setup/para_config.txt`.
* **Decision record** — `data_scripts/pdf_rasteriser_comparison.md`: why not PyMuPDF, pdftoppm or Ghostscript (the
  renderer's licence becomes every PDF record's), and the pixel comparison against pdftoppm (the training renderer)
  over twelve geometries, made with `data_scripts/compare_pdf_rasterisers.py` (`--canvas`, `--scan-dpi`, `--labels`).
  The label run on AMČR samples is the user's; its table is empty until then.
* **Production image** — `.github/production-image.json` (the `api` target's first-party files), checked by the hub's
  `tools/ci/image_closure.py` in workflow-lint; `docker.yml` names `api` as the production target. Revendored
  `atrium_openapi.py`, `service/atrium_service.py`, `tests/test_openapi_contract.py` (declared-rename rule).
* Tests: fake-PDFium fixtures for the contract and document tests, real-PDFium tests (size checked = size rendered,
  thread safety, closes), `tests/test_compare_pdf_rasterisers.py`. Fast suite 645 passed, 7 skipped.
* Tag: `v1.9.1-beta` (drafted here as `v1.10.0-beta`; the maintainer set the number). Pushed as `db834e6`.

## 2026-10-04 — `/predict_document` for digital-convert's `/describe` (atrium-digital-convert#2)
* **Why:** digital-convert `v1.1.0-beta`'s `POST /describe` asks this service about the pages of a born-digital PDF
  whose text layer is unusable, and sends the record it made. That record names its pages by their PDF page labels
  (`i`, `ii`, `1`, …) with the position in `pages[].page_index`; `/predict_document` keyed every page "1".."N", so on a
  labelled PDF its categories landed on the wrong rows (or on new ones).
* **Code:**
  * `pages` form field (`1,3,5-7`, 1-based physical): only those pages are rendered and classified; malformed or past
    the end → 422; `MAX_PDF_PAGES` counts the pages classified.
  * Categories go under the baseline row's own label wherever it carries `page_index`
    (`service/document_json.baseline_page_labels` / `record_page_keys`, torch-free). A position with no row whose
    number is another row's label is left out of the record. ALTO records are unchanged.
  * `page_label` in each page entry. The spec stays compatible (additive).
* **Tests:** `test_service_document_json.py` (+26, a real `digital-convert`-built baseline). Full suite 710 passed.
  The real app ran live against digital-convert's `/describe`.
* **Docs:** `service/README.md`, `.env.example` / `tool_limits.py` wording of `MAX_PDF_PAGES`, CONTRIBUTING row.
* Tag: `v1.9.2-beta` (drafted here as `v1.10.0-beta`; the maintainer set the number). Pushed as `41c84ca`.
* 2026-10-04, evening: the CONTRIBUTING row now also lists the shared modules re-vendored since `v1.9.1-beta`
  (`f3e8f34`, `0e6af27`, `9bf252b`: the AMČR seed profile, `source.sha512` and `run_uuid` in the record schema,
  `run_uuid`/`run_agent` in CLI paradata, RO-Crate 1.2, the program successors; `41c84ca`: `SCORING_FIELDS`).

## 2026-10-05 (evening) — The `v1.9.2-beta` tag smoke; tag line corrected
* **Red tag run:** the `v1.9.2-beta` Docker run (37269822039) failed "Run container on a NON-DEFAULT port
  (issue #58)". The cause was not the port. Each probe container downloaded the five models again, and the
  `PORT=9000` container was still loading the fifth when the HEALTHCHECK budget ran out. The images were promoted
  anyway, because the smoke job does not gate publishing.
* **Fix, in the hub:** `4cbf900` (in `v1`) shares one named volume at `HF_HOME` across a job's probe containers and
  names the warm-up in the error. This repository's first run on it (37298437914) is green.
* **Still open here:** `paradata` is not returned yet. `/predict_image` returns `null` and `/predict_document`
  omits it (atrium-project#71).
* **DEVLOG:** the 2026-09-30 entry's tag draft now says what shipped: `v1.9.1-beta`, pushed as `db834e6`.

  Files delivered in chat.

## 2026-10-07 — `paradata` from both routes (atrium-project#71 R2)
* **Why:** the one service without a `CreateAction` (#71, plan C.1): `/predict_image` returned `null`,
  `/predict_document` omitted it.
* **`service/api.py`:** one `ParadataLogger(program="page-classification", paradata_dir=None)` per call
  (`_open_run`); `_action()` builds `atrium_rocrate.create_action()` as digital-convert does: `object` is the upload
  (and `#record` when one was sent), `result` the blocks the call stamped and `predictions.json`.
  `/predict_document` logs `pypdfium2`. The two field descriptions say what is returned.
* **The record:** `service/document_json.build_document_record()` passes the run to
  `atrium_document_adapter.write_document_record()`, which now stamps `run_uuid`; for a run that writes no paradata
  file, `paradata_ref` is the run_uuid, as in the other services.
* **Tests:** `tests/test_api_contract.py` requires a valid action (`action_problems()`) on both routes; a new test
  checks `@id` = the stamps' run_uuid and the `object`/`result` entries. `tests/test_service_document_json.py`: the
  response key set includes `paradata`.
* **Spec:** `service/openapi.json` regenerated (two descriptions); `atrium_openapi.py check`: current; `compare`
  against the committed spec: compatible.
* **Docs:** `service/README.md` (*Provenance*), the CONTRIBUTING row.
* **Checks:** `-m "not slow"`: 733 passed, 7 skipped, 2 xfailed; ruff clean.
* Tag draft: `v1.9.3-beta`. **Not pushed: files delivered in chat.**

## 2026-10-07 (later) — The release bundle ships `atrium_rocrate.py`
* **Why:** the `v1.9.3-beta` tag run
  ([37624299589](https://github.com/ufal/atrium-page-classification/actions/runs/37624299589)) stopped at "Verify
  bundle is self-contained". `service/api.py` imports `atrium_rocrate` since the CreateAction change, and the `cp`
  list in `.github/workflows/release.yml` did not ship it. No release was published.
* **Fix:** `atrium_rocrate.py` joins the `cp` list; `atrium_vocab.py`, which it imports, was already there.
* **Checked locally** on `test` (`c8be0eb`) with the fix: the bundle check passes (23 modules), and so do the steps
  after it: the version guard for `v1.9.3-beta`, the spec check, the stamp, and the compare against `v1.9.2-beta`
  (compatible; oasdiff runs in CI only).
* **Missed this morning:** the CreateAction change was checked with the test suite and the spec check, but not with
  the bundle check, which only runs on a tag.
* Tag: `v1.9.3-beta`, to be re-cut on the commit carrying this fix. **Not pushed: files delivered in chat.**

## 2026-10-07 (evening) — A truncated image is refused for the life of the process
* **Found in the atrium-project#53 audit (09-27), checked today:** `utils.py` set `ImageFile.LOAD_TRUNCATED_IMAGES =
  True` and `Image.MAX_IMAGE_PIXELS = 4221790634` at import. The service imports `utils` lazily, in
  `doc_id_for_image()` on the first `/predict_image`, so the first request of a process refused a truncated upload (422
  from `_decode_image`) and every later one classified its grey fill. `service/api.py` had already worked around the
  pixel half of the same history (`Image.MAX_IMAGE_PIXELS = None`, its own `MAX_IMAGE_PIXELS` check); the truncated
  half had no such guard.
* **Fix:** the two assignments left the import. `utils.tolerate_scan_quirks()` sets them, and the batch tools call it:
  `run.py` after its `utils` import, `parallel_best.py` in both of its engines. The CLI decodes truncated scans as
  before; the service no longer changes Pillow when it imports `utils`.
* **`tests/test_pillow_switches.py` (5):** each case in a fresh interpreter, since the switches are process-wide:
  importing `utils` and `doc_id_for_image()` leave them alone; a truncated JPEG is still refused after `utils` has been
  imported; `tolerate_scan_quirks()` decodes it and sets both switches; `run.py` and `parallel_best.py` make the call.
  All five fail on `test` (`5bc982c`) and pass now. No torch needed.
* **Docs:** `service/README.md`: the published image is `ghcr.io/ufal/atrium-page-classification-api:<version>`, not
  `…:<version>-api`; the dataset handle is `1-6184`, as in the README (hub #21).
* **Checks:** full suite 750 tests, 0 failures, 12 skipped; ruff clean.
* Not released: the fix rides the next page-classification tag (the maintainer's).
* **Dev logs:** the pair of #48 belongs to a closed issue and can be deleted.

  Files delivered in chat.

## 2026-10-09 — Shared modules re-vendored (atrium-project#73, atrium-nlp-enrich#41)
* `atrium_document.py`, `atrium_document.schema.json`, `service/atrium_service.py`,
  `tests/test_document_originators.py` and `tests/test_schema_freeze.py` re-vendored from the hub: the record's
  `keywords` and `quality_summary` blocks (owned by keyword-extract and ocr-postprocess; this repository writes
  neither) and the reason `upstream_unavailable`. `service/openapi.json` regenerated (the hoisted record schema, the
  reason list). No behaviour change, no release.
* **Checks:** 751 passed, 10 skipped, 2 xfailed; ruff check and format clean; spec current; image closure OK.

  Files delivered in chat.

## 2026-10-13 — Shared modules re-vendored (atrium-project#51, atrium-nlp-enrich#41)
* `atrium_rocrate.py`, `atrium_vocab.py`, `atrium_document.schema.json`, `service/atrium_service.py`,
  `tests/test_atrium_rocrate.py` and `tests/test_schema_freeze.py` re-vendored from the hub:
  * the crate names AMČR/TEATER concepts by their source URIs, from `teater_category_ids`, and no longer mints
    `w3id.org/atrium/theme/<label>`;
  * the schema declares `enrichment.items[].teater_category_ids`;
  * the registry's theme maps follow keyword-extract's taxonomy;
  * the shared 502 text says when to retry.
* `service/openapi.json` regenerated (the record schema and its sha, the 502 description). No behaviour change, no
  release.
* **Checks:** the API-contract and vendored shared tests, 223 passed; spec current.

  Files delivered in chat.

## 2026-10-15 — The Hub loader puts its model on the device (atrium-digital-convert#5)
* **Why:** in digital-convert#5's run of 2026-10-14 (job 7100515, stage 61), `classifier.load_from_hub` left the
  three ViT models (v2.4, v3.4, v5.4) on the CPU on a GPU node. Every inference failed: "Input type
  (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same". The `all` ensemble answered
  with the two models left, and nothing in its answer said so. 21 of the 533 routes there differ from the run with
  all five models.
* **`f6c1aeb` (14 October):**
  * `classifier.py`: `load_from_hub` moves the model to `self.device`, as every other load does;
  * `service/inference.py`: a failing model is logged once, not once per page (a 529-page document had logged
    1,599 identical errors). An ensemble answer from fewer models logs "N of M models answered; failing: …" at
    WARNING the first time.
  * Tests: `tests/test_classifier.py`
    `test_a_model_loaded_from_the_hub_is_moved_to_the_classifier_s_device` and `tests/test_inference.py`
    `test_a_failing_model_is_reported_once_and_the_others_still_answer`.
* **CI:** green on `f6c1aeb`.
* **Not released.** It ships with this repository's next tag. Turn immutable releases on before that tag
  (atrium-project#72). digital-convert#5's clean re-run (C2) checks it: no model failure in the log, and the 533
  flagged pages routed alike in its stages 61 and 62.
* **Dev logs:** this entry and the header (the `test` head, the latest tag, where the pilot work is tracked).

  The code was delivered in chat on 2026-10-14 with digital-convert#5's R2; this entry is written after the push.

## 2026-10-18 — v1.9.4-beta prepared: the loader fix, the Pillow switches, the shared modules
* **Release row and files** (the tag waits for immutable releases, atrium-project#72): `setup/para_config.txt`,
  `CITATION.cff`, the spec (`--prepare tests.openapi_contract_data:prepare`; compare: compatible). The row covers
  the five commits since `v1.9.3-beta`:
  * `f6c1aeb`: Hub-loaded models on the classifier's device; a failing model logged once; a reduced ensemble said
    once (atrium-digital-convert#5);
  * `cc07344`: `utils.py` no longer sets Pillow's globals on import, so a truncated upload stays a 422 on every
    request; the batch tools call `tolerate_scan_quirks()`;
  * `29bd955`, `4160a1a`: the shared modules of 5 and 13 October.
* **#72's label run** (D1-c): stage 30 of atrium-digital-convert's `issue5_extras_job.sh` runs
  `data_scripts/compare_pdf_rasterisers.py --labels --version all` on the 15 AMČR PDFs, in parts of 40 pages (the
  script holds every render of a PDF in memory), with PyMuPDF only in a throwaway `--target` folder. Its table
  goes into `data_scripts/pdf_rasteriser_comparison.md` from the archive.
* **Checks:** 757 passed, 10 skipped, 2 xfailed (without torch); ruff clean; `check_version` OK
  (`--para-config setup/para_config.txt`).

  Files delivered in chat.

## 2026-10-19 — #72's label run: the renderer changes 5 of 2,514 AMČR page labels
* **The run** (atrium-digital-convert's extras job 7147054, RTX A4000, `v1.9.4-beta` at `d0fac40`): the 15
  born-digital AMČR reports of atrium-digital-convert#5, 2,514 pages in parts of 40, every published model
  (`--version all`), the service's pypdfium2 render against PyMuPDF. pdftoppm was not on the node.
* **Result:** top-1 labels agree on 2,509 pages (99.8 %); the five that differ move between neighbouring classes
  (`TEXT_P`/`TEXT`, `LINE_P`/`DRAW`, `LINE_P`/`TEXT_P`, `DRAW`/`DRAW_L`). Median PSNR against PyMuPDF 28.1 dB at full
  size, 45.8 dB at the 224 px input (lowest 30.9). The table per report and the five pages are in
  `data_scripts/pdf_rasteriser_comparison.md`, whose "still to run" section this replaces. The fallback of #6
  (declare the training renderer) is not needed.
* **On the way:** the job's first two attempts stopped on the project's disk quota (job 7147053), which the
  scripts now test for before they start.
* **Releases:** immutable releases are on here since 10-19 (atrium-project#72, @stranak); `v1.9.4-beta` (10-18)
  stays mutable. Nothing in this entry needs a release.

  Files delivered in chat.

## 2026-10-19 (evening) — v1.9.5-beta: the service README as a runbook (atrium-project#77)
* **`service/README.md`** gets the five runbook sections of the hub's outline:
  - Start: the published image first, with the model cache as a volume;
  - Models and caches: the five classifiers from `ufal/vit-historical-page` by revision name into `HF_HOME`, loaded before `/ready`;
  - a "What to do" column on all six error rows;
  - Partial results;
  - Known limits.

  README.md now links it. The hub's `tools/runbook_check.py` passes.
* **Found while writing it, not fixed here:** with `version=all`, a model that cannot be loaded is left out
  and the others are averaged, but the response does not say so (only the log does). Stated in Known
  limits; reporting it in the response is an additive API change for the next code release
  (atrium-project#77).
* **Release `v1.9.5-beta`:** no code change; the first immutable release here (switched on 10-19), so the S2
  pinned set can name it. `setup/para_config.txt`, `CITATION.cff`, the spec (version only; compatible), the
  CONTRIBUTING row.

  Files delivered in chat.

## 2026-10-21 — The renderer note in README.md; two directory-run bugs found (atrium-project#66, #15)
* `README.md` (§ licences, the PDF route): the two renderers are not identical; on 2,514 pages of 15 born-digital
  AMČR reports, 2,509 kept their top-1 label and five moved to a neighbouring class. Linked to
  `data_scripts/pdf_rasteriser_comparison.md`.
* **Found while writing the Galaxy wrapper (atrium-project#66), not fixed here:**
  * a directory run drops an unreadable image in `custom_collate`, and `dataframe_results` then pairs the
    predictions with the file names by position: every later image gets its neighbour's row, and the last one is
    lost, with no error. With `--no-inner`, any non-image file in the folder does it;
  * with `-tn 1`, `infer_dataloader` returns class numbers, so `SCORE-1` holds the class number, not a score. The
    service avoids it (`internal_topn = max(topn, 2)`).
  Both need a run with torch to fix and test; this session had none. Issue text drafted for the user.
* **Found for the paper (atrium-project#15):** in `result/stats/*FOLD_*_DATASETS.txt`, 4,296 of the 5,449 pages of
  the manually checked evaluation set are in fold 1's training partition (the other folds are alike), and no stored
  table gives 99.16 %. The held-out figures are each fold's 4,823-page test partition.
* Docs only. **Checks:** none needed beyond the link check.

## 2026-10-21 (round 2) — #61 fixed: the two directory-run bugs; the document-overlap evaluation (atrium-project#15)
* **Alignment:** `ImageDataset.__getitem__` returns the image's index; inference uses `inference_collate`, which
  keeps the indices of the images that loaded (training keeps `custom_collate`: the Trainer passes every collated
  key to the model). `infer_dataloader` returns one entry per dataset item, `None` for an unreadable image.
  `utils.keep_readable` drops those before `dataframe_results` and `run.py` logs each to the paradata as skipped;
  `dataframe_results` refuses lists of different lengths. `parallel_best.py` (both paths) and
  `service/inference.py` `predict_directory` do the same; YOLO models share the scoring.
* **Top-N scores:** `classifier.topn_pairs` returns (class, probability) pairs for every N, renormalised over the
  best max(N, 2) classes, the service's rule. `-tn 1` wrote the class number to `SCORE-1`, and
  `top_n_predictions(top_n=1)` failed. README's score note says so.
* **#15:** `data_scripts/eval_by_document_overlap.py` reports top-1 accuracy for pages in training, held-out pages
  of documents seen in training, and unseen documents, from a folds CSV or a `*_DATASETS.txt`. On the stored fold 1
  split it reproduces the 4,296 of 5,449 pages in training.
* CONTRIBUTING: an Unreleased row. No version bump: the Galaxy wrapper stays on `1.9.5-beta` with its guard.
* **#61** was opened at 07:59 from the issue text drafted the day before; its milestone is CAA Proceedings, because
  the `vX.4` re-run for the papers (atrium-project#13, #15) needs the fix. Digest and plan:
  `digests/61.digest.md`, `plans/61.plan.md`.
* **The agent-skill branch** (atrium-project#77 C, archive only): re-derived with this fix; the client retries a 502
  only when it is not final and exits 1 on a usage error; `scripts/server.sh` waits for `/ready`; `service/README.md`
  is the v1.9.5 runbook, with the LINDAT frontend pointed at `test`. Skill-validate steps 1–4a and the drift check
  pass. Delivered as `atrium-page-classification-agent-skill.zip`.
* **Checks:** a py3.11 venv with torch 2.7.1 (PyPI). Full suite: 884 passed, 3 skipped, 2 xfailed; 2 failed, the
  same 2 as on unmodified `test` in this sandbox (one needs huggingface.co; one fails only after ultralytics has
  tried to install `pi-heif` offline). `ruff check` and `ruff format --check` clean.

## 2026-10-21 (round 3) — The 5,449-page set's selection bug; v1.9.6-beta prepared (atrium-project#15, #61)
* **Why 4,296 of the 5,449 "held-out" pages are training pages.** The set is
  `result/stats/unused_entries_da5_f12345.csv`, made by `result/stats/unused.sh` from the five `*_FOLD_*_DATASETS.txt`
  files as the pages no fold trained on: held out by design. The script matched only `CTX<digits>-<digits>` and
  compared the page as text, so fold 1's training pages `CTX192100040-01.png` (3,679 zero-padded) and
  `CTX193401761A-11.png` (617 ids with a letter) never met the CSV's `CTX192100040,1`. Every literal match it saw
  is in a dev or test partition, as for a run with `-t`.
* **Fixed:** `extract_doc_page` uses the rule of `utils.doc_id_and_page` (the label after the last `-` or `_`, as a
  decimal number; a name without one is page 1), and the CSV's page is compared as a number. Re-run on the stored
  splits and the old CSV, it leaves **130** of 5,450 pages, the ones in no fold's training partition. Two tests in
  `tests/test_data_scripts.py` (`TestUnusedSh`) fail on the old script and pass on the new one.
* README: a note under the accuracy table (the `vX.3` figures come from that set; the `vX.4` re-run on the LINDAT
  test sets replaces them). CONTRIBUTING's model table points to it. `eval_by_document_overlap.py` names the cause.
* **v1.9.6-beta prepared:** `setup/para_config.txt`, `CITATION.cff` and `service/openapi.json` (version only);
  `check_version.py` agrees. The GitHub notes and the CONTRIBUTING row, which replaces the Unreleased row, are
  drafted in chat. Tagging is the maintainer's.
* **Dev logs:** the #48 pair is removed (closed 2026-09-17; the hub's history page links the copy on `vit`).
* **Checks:** see the round's summary.

## 2026-10-21 (round 4) — The vX.4 re-run as one SLURM job; `-rev` builds on its own base (atrium-project#15)
* **Why:** the user re-runs the `vX.4` models on the test sets of the LINDAT record on the cluster, and the results
  replace the stored `vX.3` ones (atrium-project#15, #13).
* **Found while tracing the run:** `run.py -rev vN.M` kept the config's base (`timm/regnety_160`) unless `-b` named
  another (`if args.base != config_base_model`). The eval transforms (resize, mean/std) are built from that base and
  are not rebuilt when the weights load. So `--eval -rev v2.4` gave the 224 px ViT 384 px images with ImageNet
  statistics, and the other ViTs and EfficientNet got ImageNet statistics; `--train -rev v2.4` would have trained a
  RegNetY. **Fixed:** `run.resolve_revision_base()` takes the registry's base, exact key first
  (`model_registry.resolve_base_model`, as in the service), and the paradata records it.
* **`--eval_dir`** (default `[EVAL] FOLDER_PAGES`): the folder to evaluate, from the command line.
* **`data_scripts/stage_fold_pages.py`:** a fold's test pages out of the LINDAT category zips, into all 11
  category folders. The cluster has no folder with the CSV's names: `PNG/data/` is per document with unpadded page
  numbers, and `test_F12345` holds JPEGs. A page in no zip, or in another category's zip, stops it; a re-run
  extracts nothing it already has. A row without a PNG name is left out with a note: the published
  `licensed_crossval_folds.csv` has one, `test` in folds 1, 2 and 5, which the user's prefetch of 10-22 reported as
  a page missing from the zips (`licensed_crossval_folds_CUT.csv` has no such row).
* **`supplementary/scripts/eval_vX4_job.sh`:** `prefetch` on the login node (Hub revisions to `model/`, their
  commits, the folds against `licensed_crossval_folds_CUT.csv`, a dry run of the staging); then `sbatch`, offline.
  It runs each model on its fold's test pages (Top-1; Top-3 with `--raw`), the 5,449-page set, the sample pages and
  the `--best` ensemble. Each step is logged and the job goes on after a failure. The summary gives accuracy and the
  per-group Top-1 of `eval_by_document_overlap.py`.
* `supplementary/scripts/result_analysis.sh`: the `v14`–`v54` names (`v44` parsed as `v4`, fold 4).
* **v1.9.8-beta prepared:** `setup/para_config.txt`, `CITATION.cff`, `service/openapi.json` (version only), the
  CONTRIBUTING row. Tag it before the cluster run, so the paradata names the release.
* **Checks:** a py3.11 venv with torch 2.7.1 (PyPI), `HF_HUB_OFFLINE=1`. Full suite (`-m "not slow"`): 905
  passed, 2 xfailed, 1 failed; untouched `test` gives 885 passed and the same failure
  (`test_an_unreadable_image_is_422_not_500`, which fails only in the full run, after ultralytics has tried to
  install `pi-heif` offline). `ruff check` and `ruff format --check` clean; shellcheck clean; `check_version.py`
  and `atrium_openapi.py check` OK. The job ran end to end on a fixture (zips built from `small_data_samples`, a
  stand-in for `run.py`): 25 steps OK, 20 tables in the summary, the staged folds removed. The Hub lookup of the
  prefetch needs network and was not run here.

  Files delivered in chat.

## 2026-10-22 — The vX.4 results kept, the old ones explained; evaluation paradata (atrium-project#15)
* **The run** (job 7271549, `v1.9.8-beta` at `f14bc4b`; the user's push `582938c`…`f6ef687`): 25 steps OK, no
  unreadable page, no fold-test page in training. Top-1 on each model's fold test pages 94.64–95.87 %, unseen documents
  93.87–95.60 %, Top-3 99.75–99.90 %. Checked before writing:
  * TOP-1 and TOP-3 tables agree page by page; RAW agrees except one tie at 0.496/0.496;
  * Wilson intervals about ±0.6 points;
  * McNemar on the shared folds: `v1.4`/`v4.4` p = 0.86, `v3.4`/`v5.4` p = 0.76.
* **Kept and removed:**
  * kept: the fold tables, plots and RAW; the 5,449-set tables (by group only); the ensemble demo; the run records;
  * removed: the 5,449-set confusion plots, the 229-page sample evaluations (outside the training data, labels never
    checked), the prefetch paradata, and `20260723-1659_model_v43_TOP-3.csv`. That last file holds `v4.4`'s
    predictions: the same top-3 classes on all 229 pages, unlike the May `v4.3` table.
* **The old results** (preserved at `789ff04`, as motyc asked on #15), in `result/README.md`:
  * every `vX.3` figure, the README's "Best_Acc" included, is the 2025 sweep's on the 5,449-page set;
  * the sweep chose each base model's fold (the highest figure, the lowest fold on a tie), and `REVISION_BEST_FOLDS`
    matches it for all five;
  * the README's plots contradicted its text: `v4.3` 98.92 % against 99.16 %, `v1.3` 96.81 % against 98.83 %;
  * the 96.81 % run is consistent with the `-rev` base bug fixed in `v1.9.8-beta`;
  * no two models' tables are identical, so the brief's "shifted labels" guess is wrong;
  * the `a5` models had seen all but 129 of the 5,449 pages.
* **README, CONTRIBUTING:** the held-out table leads; the sweep table stays, labelled; the results section shows the
  fold matrices. The `vX.2` table had `v4.2`/`v5.2` swapped and `v3.2` 96.49 %, which no stored table gives (97.41 %).
  The four `vX.2` RAW links all pointed to `v2.2`'s file, and the paradata example link was dead.
* **`run.py`:**
  * `--eval` counts its pages and tables in the paradata (`input_files_total` was 0, "processed" was the plot
    count) and declares `lindat_dataset`, as `--train` does;
  * `result/tables` and `result/plots` are made on their own: the paradata logger makes `result/` first, so an
    evaluation into a new output folder failed on its first table.

  Test: `tests/test_run.py::test_an_evaluation_counts_its_pages_and_declares_the_dataset` (it fails on the old
  code).
* **Not fixed, noted:** `utils.confusion_plot` needs every category in the evaluated set (`classification_report`
  without `labels=`).
* **v1.9.9-beta prepared:** `setup/para_config.txt`, `CITATION.cff`, `service/openapi.json` (version only), the
  CONTRIBUTING row.
* **Checks:** a py3.11 venv with torch 2.7.1 (PyPI), `HF_HUB_OFFLINE=1`. Full suite (`-m "not slow"`): 907
  passed, 2 xfailed, 1 failed, the same `test_an_unreadable_image_is_422_not_500` as on untouched `test` in this
  sandbox. `ruff check` and `ruff format --check` clean; `check_version.py` and `atrium_openapi.py check` OK; every
  relative link in `README.md` and `result/README.md` resolves. README also: RegNetY-160's input is 384 px (its Hub
  config, `MODEL_STATIC`), not 224.

  Files delivered in chat.

---
_Timeline index refreshed 2026-09-26 (AMČR baseline entry and header); earlier 2026-09-07 against live `test`/`vit` HEAD, the `CONTRIBUTING.md` changelog table, open-issue
state via the GitHub API (zero open), and the confirmed `@v1` reusable-workflow pin. Nothing removed from the issues
themselves (per hub #29); this file is a derived reading aid in `agent_dev_logs/`._
