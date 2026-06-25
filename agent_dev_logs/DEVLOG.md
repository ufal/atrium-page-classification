# 📓 atrium-page-classification — agent_dev_logs/DEVLOG.md (history seed)
> _Historical page-image classification. 1 open issue. `test` HEAD `08582b6` (2026-06-23)._

## 2026-06-25
- **#15 Retrain 5 best models on the new dataset (N−318 pages)** — Opened by K4TEL: retrain the five best models — EffNet `model_1241` (CV split 1, seed 420), RegN `model_741` (split 1, seed 420), Vb-2 `model_245` (split 5, seed 424), Vb-3 `model_342` (split 2, seed 421), Vl-3 `model_542` (split 2, seed 421) — on the licensed dataset minus the 318 removed pages; attached `licensed_crossval_folds_CUT.csv`. Confirms the atrium-project **#21** dataset release: it becomes the key reference once the re-finetuned models reach the same accuracy on the same (minus-318) eval data.

---

# 📓 atrium-project — agent_dev_logs/DEVLOG.md (history seed)
> _Hub/planning repo. Reconstructed from 15 open issues. `test` HEAD `11ba0ff` (2026-06-24)._

## 2026-03-13
- **#4 SSH Open Marketplace records** — Opened by stranak: create SSHOMP records for every tool in our workflows (UDPipe ✅, NameTag ✅, rest TBD).
- **#6 Review & summarise licenses** — Opened by stranak (review tool+model licenses, check where CC-BY-NC-SA is required). K4TEL posted the first license inventory: FastText/AMCR-vocab CC BY-NC, layoutreader CC BY-NC-SA, distilgpt2/alto-tools/GLM/Qwen2.5 Apache-2.0, ViT/EffNet/RegNet/CLIP MIT, NameTag3/CUBBITT CC BY-NC-SA, UDP2 MPL-2.0, AISCR Teater GPL-3.0.
- **#9 Paradata of outputs** — Opened by K4TEL: unified run-logging (incl. output license) across all four tool repos.

## 2026-03-15
- **#4** — Page classifier added to SSHOMP as a Suggested Tool under the `ATRIUM catalogue` keyword.
- **#9** — Translator, textline & page classifiers tested with paradata output; basic `.json` paradata in all repos via a shared `atrium_paradata.py`.
- **#10 LLM validation of source code** — Opened by K4TEL (validate every repo's source with an LLM).

## 2026-03-22
- **#10** — All projects checked with Sonnet 4.6 Extended, then re-checked with Gemini 3.

## 2026-03-25
- **#13 CAA Proceedings paper to PCJ** — Opened by K4TEL: submit a paper to the CAA2026 proceedings / PCI Archaeology; text draft posted (5000-word limit, no figures yet).

## 2026-03-26
- **#13** — Added the full project diagram, an updated report PDF with the diagram inserted, and a Zenodo submission draft.

## 2026-04-04
- **#13** — Overleaf editor invites sent to David and Dana; CAA-proceedings project + Springer extended-version project to be reformatted into CAA styles.

## 2026-04-11
- **#4** — The remaining three repositories suggested as SSHOMP tools.

## 2026-04-16
- **#4** — ALTO post-processor, NLP enrichment, translator and page classifier all uploaded as tool-or-service under the **ATRIUM catalogue** tag.

## 2026-05-13
- **#13** — motyc: proceedings deadline is **31 October 2026**.

## 2026-05-27
- **#15 Submission to IJDL** — Opened by motyc (review ASAP, link in the minutes).
- **#16 List ARUP/B data storage locations** — Opened by motyc (so ARUP/B can later remove all copies). K4TEL listed the `data_samples` dirs across repos, the LINDAT annotated dataset, thesis/presentation page samples, and the UFAL filesystem.
- **#17 Review SSHOMP workflow descriptions** — Opened by motyc.
- **#18 Docker compose + GH action wrapper for CU forks** — Opened by motyc (links the four ARUP-CAS forks).

## 2026-05-28
- **#9** — Mass→single-file paradata records merged per repo; open questions on license source, missing tool-version tag, dynamic runner reference, and a Docker-image placeholder.
- **#10** — Slated for re-examination by Opus 4.7 and Sonnet 4.6 across all four repos.
- **#17** — K4TEL posted the four marketplace tool links; motyc thanked; noted relation to #4.

## 2026-05-29
- **#9** — Detailed per-repo license breakdown: the tool-vs-model split (NameTag3/UDPipe engines MPL-2.0 but their models CC BY-NC-SA), Teater app GPL vs data CC BY-NC, and the internal-academic-use vs external-commercial-use distinction.
- **#10** — motyc: "Opus 4.8 is just out :)".
- **#16** — Posted per-repo licensed-asset tables (alto 39, nlp 34, translator 14, page-classification 84 documents) mapped to licenses from the global metadata collection.

## 2026-06-02
- **#9** — The two easy repos (translator, page-classification) updated with paradata licenses; the two multi-step repos (alto, nlp) remain (sequential-log aggregation); alto full-pipeline commit landed.

## 2026-06-03
- **#9** — nlp-enrich commit adds licensed paradata for API scripts + keyword extraction (LLM samples to follow).

## 2026-06-08
- **#16** — Full current-state inventory of every `data_samples/` dir; alto & nlp **resolved to contain only synthetic data**; translator still holds 16 real ARUP/B source documents; page-classification has ~245 PNGs across 11 category folders.

## 2026-06-10
- **#6** — License summary (from #9) implemented for all four repos; TODO to attach the list to the SSHOMP workflows.
- **#9** — Only nlp-enrich remains (LLM samples); all-stage merging done.
- **#18** — Opus strategy: repos are already pre-wired — `atrium_paradata.py` reads `ATRIUM_RUNNER_IMAGE/REPO/REF`, so GHCR-published self-identifying containers are the plan.

## 2026-06-12
- **#9** — Merged paradata for nlp stages 1–4 + one keyword method; all seven checklist items marked done.
- **#10** — Plan to review each repo with Fable by 22 June.
- **#18** — Per-repo Docker drafts summarised (shared template, per-repo knobs); motyc: discuss orchestration with rharasim, no overall wrapper needed (containers reachable via API).
- **#21 LINDAT annotated dataset release** — Opened by K4TEL: two ways to fix the licensing problem (modify old handle vs publish new + redirect); per-file metadata fields; sample JSON/CSV; motyc OK with option 1, notes some files can't be openly published (metadata-only).

## 2026-06-14
- **#21** — Posted the 82 GB ready-to-publish `licensed_archives/` listing: `CITATION.cff`, CC BY-NC `LICENSE`, per-document licensed CSV/JSON, cross-val folds, category ZIPs, and a `not_included` CSV for disallowed-license files.

## 2026-06-15
- **#10** — Released alto v0.17.0, page-classification v1.4.0-beta, translator v0.6.0, nlp v0.12.0 with LLM-review edits applied (Fable was unavailable 😮‍💨).
- **#18** — translator & page-classification passed GH Actions; posted the "Align & Expand Docker + GHA" strategy (one reusable workflow template + thin per-repo callers).

## 2026-06-16
- **#9** — Old paradata files to be replaced and `para_config` versions bumped across all four repos.
- **#10** — Defined the next review round's aspects: Docker+GHA, merged pipeline & API, per-function test coverage, architecture, file tree, CONTRIBUTING release history, + a per-repo review plan.
- **#18** — Commit `676a1fe` lands the centralized DRY CI/CD (`ci-cd-strategy.md`, `docker-tool.reusable.yml`, caller example, shared `.coveragerc`/`ruff.toml`, dependabot appendix); all four repos pass GHA; docs updated for rharasim to test.

## 2026-06-17
- **#10** — Combined per-repo review plan committed (`aba539e`); posted the post-review validation matrix (Tier-1 compileall/ruff + Tier-2 pytest/coverage, run pc→alto→nlp→translator).
- **#22 Document Understanding eval** — Opened by K4TEL (benchmark for document understanding — OmniDocBench?). Gemini "Deep Research" report posted; Opus 4.8 follow-up corrected its fabrications/mis-attributions, separated parsing-fidelity from semantic understanding, flagged **CHURRO/CHURRO-DS** as the real historical-doc match, and recommended an OOTB-VLM-vs-legacy-pipeline comparison first.

## 2026-06-19
- **#4** — SSHOMP tool records updated with license tables.
- **#6** — TODO to attach license lists to the marketplace workflows; new versions to be set by admins on the default tool views.
- **#21** — Major licensing discussion: 318 unpublishable files (<0.01%) removed; CC BY-NC vs BY-NC-SA debated (stranak/motyc lean to dropping SA → plain NC, citing EOSC/Open-Access policy); tombstone + "incomplete dataset" metadata text drafted; link replacements queued for arXiv/README/Zenodo. stranak already published the record; license to be swapped.
- **#22** — stranak: when running a big VLM (e.g. MiniMax-M3), contact Viktor about vLLM on the reserved Grace Hopper machine.
- **#24 LLM applications to data** — Opened by K4TEL (various local/remote LLM tasks).

## 2026-06-20
- **#21** — motyc proposed Description-field text (318 files, accessible under conditions at digiarchiv, GitHub repo for full pipeline).
- **#26 Run models larger than GPU memory via CPU** — Opened by K4TEL (explore unified-memory mechanism).
- **#27 H100 multi-GPU runs** — Opened by K4TEL (MiniMax-M3 FP8 ~440 GB on a single multi-GPU node).

## 2026-06-21
- **#6** — Admins to update default tool versions; license tables added to each SSHOMP description.
- **#10** — Opus 4.8 review round: new findings — `/info` version drift, `para_licenses.py` diverged + zero tests, nlp ruff blocking, secret-scanning unverified; posted a phased strategy.
- **#18** — Further GHA-integration strategy (Codecov gate bug, `@main` vs `@test` pin drift, action version floor, per-repo P0/P1/P2); all four repos released as vX.Y.Z+1 passing ruff/pre-commit.
- **#26** — Opus recommendation: vLLM `cpu_offload_gb` (UVA zero-copy) over Ollama layer-split or raw CUDA UVM — memory-only offload keeps CPU cores free for the existing queue.
- **#27** — Opus recommendation: 8×80 GB H100 SXM5, vLLM/SGLang tensor-parallel-size 8 + expert-parallel + fp8 KV cache, capped `max-model-len`; support is brand-new (use nightly/Docker).

## 2026-06-22
- **#21** — kosarko, stranak and motyc debate whether the corrected dataset record even needs the 318-files warning (agreed it belongs on the *models*/tombstone, while keeping it discoverable via the `not_included` CSV).

## 2026-06-23
- **#10** — `docs/plan_repo_review.md` declared the canonical plan to execute across the whole ecosystem.
- **#13** — Handle/DOI to be replaced in the Overleaf bibliography (marked DONE).
- **#15** — Dataset reference to be replaced in the post-review IJDL edit; arXiv preprint to be updated.
- **#16** — #21 designated the canonical "where licensed samples are shared" reference; motyc: keep open until end of project.
- **#21** — Links updated in both arXiv papers, the README, and the Zenodo DOI (one un-editable spot remains: the official CU MFF thesis record).

## 2026-06-24
- **#21** — kosarko refined the Description wording (bolded "318 files" claim to fact-check); K4TEL confirmed the 318 count via `wc -l` on the CSVs; stranak proposed keeping the original dataset (restricted, incl. the 318 files) **plus** a CC-only derived subset, linked together.

## 2026-06-25
- **#15** — arXiv `2606.07558` updated with the new dataset link (references only).
- **#16** — Both arXiv versions (`2507.21114`, `2606.07558`) updated with the new dataset licensing link.
- **#29 Add `agent_dev_logs` directory per repo** — Opened by K4TEL (this initiative): per-repo markdown dev logs on `test`, seeded from issue history, replacing agent work-documentation in issue comments.
