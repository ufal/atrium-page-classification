# 📓 atrium-page-classification — agent_dev_logs/DEVLOG.md (timeline index)
> _Historical page-image classification. No open issue here (#48 closed 2026-09-17; #15 and #26 closed). Its pilot work is tracked in hub issues (atrium-project #6, #53, #67). `test` HEAD `2e27073` (2026-09-26) · **v1.8.0-beta** (2026-09-16)._
> _Per-issue detail: `digests/48.digest.md` · `plans/48.plan.md` (the issue closed 2026-09-17 and its export was removed; the pair is to be removed too). This header previously claimed no such exports existed; they have since 2026-09-07. The 2026-08-02 → 09-06 entries below predate them and are reconstructed from `CONTRIBUTING.md`'s release-note table (source of truth for that window) and commit history, not from a digest._
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

---
_Timeline index refreshed 2026-09-26 (AMČR baseline entry and header); earlier 2026-09-07 against live `test`/`vit` HEAD, the `CONTRIBUTING.md` changelog table, open-issue
state via the GitHub API (zero open), and the confirmed `@v1` reusable-workflow pin. Nothing removed from the issues
themselves (per hub #29); this file is a derived reading aid in `agent_dev_logs/`._
