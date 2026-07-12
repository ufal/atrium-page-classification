# 📓 atrium-page-classification — agent_dev_logs/DEVLOG.md (timeline index)
> _Historical page-image classification. 2 open issues (#15, #26). `test` HEAD `14af859` (2026-07-12) · **v1.5.1-beta**._
> _Per-issue detail: `digests/{id}.digest.md` · `plans/{id}.plan.md` · `issues/` exports (source of truth). Cross-repo/hub history lives in `ufal/atrium-project/agent_dev_logs/DEVLOG.md` (deduplicated out of this file)._

## 2026-06-25
- **#15 Retrain 5 best models on the new dataset (N−318 pages)** — Opened by K4TEL: retrain the five best models — EffNet `model_1241` (CV split 1, seed 420), RegN `model_741` (split 1, seed 420), Vb-2 `model_245` (split 5, seed 424), Vb-3 `model_342` (split 2, seed 421), Vl-3 `model_542` (split 2, seed 421) — on the licensed dataset minus the 318 removed pages; attached `licensed_crossval_folds_CUT.csv`. Confirms the atrium-project **#21** dataset release: it becomes the key reference once the re-finetuned models reach the same accuracy on the same (minus-318) eval data.

## 2026-06-27
- **#15** — Eval fold path committed (`b86c431`): `--eval --folds_csv` scores only the chosen fold's `test` pages from the global EVAL dir (single-model only, not `--best` — each best model uses a different fold). Builds on the fold-CSV **training** path already merged on `test` (`992c5df`, `split_data_from_folds` + `REVISION_BEST_FOLDS` + the v*.4 registry, released in **v1.5.0-beta** with the paradata-template sync and `agent_dev_logs/`). Second fine-tuning round **launched** on the cropped dataset.

## 2026-06-28
- **#15** — `f0643a0` fixes the train-finishing evaluation path.

## 2026-06-29
- **#15** — Retrain **evaluated**: vX.4 vs vX.3 prediction diff = **24 mismatches of 229 samples** (`diff_models_3-4.csv`, down from 25) — **all ambiguous cases**. Conclusion: removing the 318 pages only affects cases the vX.3 best-5 ensemble already disagreed on; obvious category-specific classification does not suffer. Pushing the vX.4 models to the HF hub is possible if needed — **awaiting the call**.

## 2026-07-12
- **#15** — `55e604c` lands `tests/test_folds_split.py` (18 tests: routing, −318/absent-page drop, NA/whitespace/case, eval subset, registry ordering, real-CSV `slow` check) + the shared `tests/test_para_licenses.py`; fixed a pandas ≥3.0 NA-preservation crash in `split_data_from_folds` (~225 blank `fold1` cells — now `pd.isna`-guarded). Released **v1.5.1-beta** (licenses test per template, fixed automatic version reading via `_read_tool_version()`, dependency bumps, GHA fixes). Digest+plan refreshed to the committed reality — remaining decision: the HF-hub push / canonical vX.4 release.
- **#26 Agent-skill branch for page classifier prediction** — Opened by K4TEL: package the classifier as an agent skill on the [`honzas83/uwebasr-skill`](https://github.com/honzas83/uwebasr-skill) pattern — `SKILL.md` (frontmatter + agent guidelines) + a **stdlib-only client** (`scripts/atrium_classify.py`) wrapping the existing FastAPI service, with a two-phase workflow (ensure-server-up via `/info`, then classify) so the same client flips to kosarko's future LINDAT endpoint via `--base-url` / `ATRIUM_PC_URL` only. The [`agent-skill`](https://github.com/ufal/atrium-page-classification/tree/agent-skill) branch with cleaned-up files created 08:14; digest+plan pair added (client, `SKILL.md`, `serve.sh`, multi-agent install docs, smoke fixtures = next).

---
_Timeline index refreshed 2026-07-12 against `test` HEAD and the refreshed digests/plans. Nothing removed from the issues themselves (per hub #29); this file is a derived reading aid in `agent_dev_logs/`._
