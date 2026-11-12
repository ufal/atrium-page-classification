# Strategy — doc-schema-v2 (atrium-project, issue to be opened): page-classification's part

> _Written **2026-11-11** after the meeting of that day (ARÚP's output contract). Hub:
> [plan](https://github.com/ufal/atrium-project/blob/test/agent_dev_logs/plans/doc-schema-v2.plan.md) ·
> [digest](https://github.com/ufal/atrium-project/blob/test/agent_dev_logs/digests/doc-schema-v2.digest.md) ·
> [meeting digest](https://github.com/ufal/atrium-project/blob/test/agent_dev_logs/digests/project_state_1111.md).
> Read against the local checkout `16a2e52`. The issue is not opened; rename this file to the issue number once it
> is._

**Status (2026-11-11)**: 📋 **Planned, no code. The classifier runs on every PDF and image, bundled with the ATR call
at the AIS CR's ATR service, and without ATR when no page needs OCR (K4TEL). It writes its page rows under
`page_index`; its categories are hints to the route and the ATR, not gates (H3). Code waits for the hub's shared
modules.**

## Its fields

| Field                                             | Note                                                                                             |
|---------------------------------------------------|--------------------------------------------------------------------------------------------------|
| `page_categories`                                 | whole block; today keyed by the page label (`atrium_document_adapter._page_patches()`, :112-131) |
| `pages[].page_index`                              | on its own rows, the join key (H2); today only digital-convert writes it                         |
| `pages[].category`, `pages[].category_confidence` | the top class and its score (3 decimals)                                                         |

## Tasks (next rounds)

1. **Re-vendor** the hub's shared modules once hub tasks 1–5 land (`scripts/revendor_shared.sh`).
2. **`page_index` on its rows** (H2): `_page_patches()` (`atrium_document_adapter.py:112-131`) writes the physical
   page number the classifier already has (the rendered PDF page, or `utils.doc_id_and_page()` for an image). The
   label mapping then has nothing to do: `page_inputs.pdf_page_keys()` and `remap_pages()` (:129-150),
   `service/document_json.record_page_keys()` (:138-156). The original becomes the only input.
3. **No run data or licence in the record** (D1, D2): `add_derived_from("classification", …)`
   (`atrium_document_adapter.py:223`) and `add_license_detail()` (:228) go.
4. **The action** (`service/api.py:604-627`): `object` the original, without the baseline record (:622-623);
   `result` the record blocks `page_categories` and `pages`. `predictions.json` (:625-626) is not a stored output, so it leaves the result. Both `file_entity` calls
   (:621, :626) take the hub's SHA-512 form (hub task 3). `run.py`'s paradata follow.
5. **Limits** (D13): `MAX_PDF_PAGES` (50, `tool_limits.py:24`) already refuses with a 413 instead of trimming;
   chunking is the cluster's (D3).
6. **CI** (atrium-project#81): the daily `scheduled-smoke.yml` cron (:5) goes; `workflow_dispatch` stays.
   `gpu-inference.yml` has had no cron since 2026-07-29; its "RESTORE" note (:17-19) goes.
7. **Docs:** `docs/usage.md` (:395, :519 the page labels; :539 `derived_from.classification`),
   `docs/reference/cli.md:92`, `docs/how-it-works.md:100`, `service/README.md` (:172-178; :294 `predictions.json`).

## Paradata

One action per job in `paradata/atrium/document-json`:

- `object`: the original (distribution name, SHA-512);
- `result`: the record blocks `page_categories` and `pages`;
- `parameters`: the model revision and the ensemble members;
- `components` (facts, never combined): the fine-tuned ViT models `ufal/vit-historical-page` (CC BY-NC 4.0) and their
  base models' licences (`model_registry.py`, `setup/para_config.txt`).

## Depends on

- Hub tasks 1–5 (shared modules), 6 (docs), 7 (E2E lanes); the proof gate (D12).
- The intake (H1, D3): pages are rendered on the cluster side. The models were trained on 300-dpi renders
  (`tool_limits.PDF_RENDER_DPI`, :26-30), so another resolution changes what they see.
- The ATR bundle and its service are ARÚP's (aiscr-atr).
- Q1 (the per-block run map).

## Superseded in this repository's logs

All in the open #58's files (no closed issue's files remain here); marked in place on 11-11 (line numbers after the
marks; before them in brackets):

- `plans/58.plan.md:77` (:71) and `digests/58.digest.md:79` (:73): "Phase A can start there": #58 is parked during
  the pilot phase (meeting §0).
- `plans/58.plan.md:89` (:82): the "Next round" list (probe, Phase A smoke run): parked likewise.
- `plans/58.plan.md:498` (:480): digital-convert's "`/describe` stage waits until after 11-11": `/describe` is not
  used in the pipeline (H11).
- `digests/58.digest.md:274` (:254): "`/describe` already reports per page … a route": the same (H11); the route is
  the pipeline's step.

## Not in scope

- The decision models of #58: parked during the pilot phase. `v1.14.0-beta` and `v1.14.1-beta` are tagged, not
  released; `v1.14.2-beta` is prepared. Releases are cut deliberately, with notes (working agreement), so the DM
  CLIs wait.
- The TrueOpen.AI hosted pilot: the contract's §9 has ARÚP prefer parking it until the chain runs, or a limited pilot
  on the published CC BY-NC 4.0 dataset only; no archival images beyond it; non-commercial.
- New categories, retraining, gating on a category.
- The document's licence: AMČR's RO-Crate only.
