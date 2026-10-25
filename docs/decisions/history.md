# History and decisions

For contributors and reviewers: why page-classification is shaped the way it is, in the order it happened. The
release-by-release list is the Release History in [CONTRIBUTING.md](../../CONTRIBUTING.md#release-history); the
chronology across the ATRIUM repositories is the hub's
[Development history](https://ufal.github.io/atrium-project/development-history/).

This page is condensed from the repository's own [DEVLOG.md](../../agent_dev_logs/DEVLOG.md), itself a timeline index
over the issue exports, digests and plans in `agent_dev_logs/`. Those raw records are not republished here: they are
linked, and they remain the source of truth.

On this page:
[The five arcs](#the-five-arcs) ·
[2026-06 to 08: retraining on the licensed dataset](#2026-06-to-08-retraining-on-the-licensed-dataset-15) ·
[2026-07 to 08: the Agent Skill](#2026-07-to-08-the-agent-skill-26) ·
[2026-08: guarding the install and the runtime](#2026-08-guarding-the-install-and-the-runtime) ·
[2026-08 to 09: hardening](#2026-08-to-09-hardening-and-the-hub-taking-over-the-plumbing) ·
[2026-09: the ensemble moves to `vX.4`](#2026-09-the-ensemble-moves-to-vx4-48) ·
[2026-10: the evaluation corrected](#2026-10-the-evaluation-corrected-15) ·
[The raw records](#the-raw-records)

## The five arcs

Versions below `v1.0` were research milestones; from `v1.0.3-beta` on, each release is a usable program, published as
a GitHub pre-release. Read top to bottom, the releases tell this story.

**1. Finding a method that works** (`v0.1.0` → `v0.9.0`). Two approaches were tried and abandoned, and the release
notes say so: DeepDoctection layout analysis plus a hand-written categorisation rule (`v0.1.0`, marked `[FAILED]`),
then a Random Forest over manually extracted texture and colour features (`v0.2.0`, also `[FAILED]`). `v0.5.0` is the
turn: fine-tuning Hugging Face base models on a labelled page dataset. `v0.7.0` widens that to the ViT family, and
`v0.9.0` to EfficientNetV2, RegNetY and DiT, with cross-validation.

**2. Picking the five** (`v0.11.0` → `v0.12.1`). A twelve-model sweep settles on RegNetY as the leader and on five
models for the ensemble ([the sweep](../evaluation/sweep-2025.md)); result averaging, the `--best` path and a first
API draft land together. The two version namespaces of
[reference/models.md](../reference/models.md#two-version-namespaces-and-where-they-collide) originate here.

**3. Becoming a program rather than a study** (`v1.0.3-beta` → `v1.4.3-beta`). Pytest arrives; the sample images
are replaced with CC BY-NC 4.0 ones; the data-processing scripts get their own unit tests; a Docker wrapper appears
(`v1.2.1-beta`, explicitly "draft, untested"); YOLO-cls is added to the entry point and the paradata gains licence
resolution (`v1.2.0-beta`); the memory-aware parallel `--best` lands (`v1.3.0-beta`); the files are reorganised into
sub-folders and the source goes through LLM review (`v1.4.0-beta` to `v1.4.3-beta`).

**4. Joining the ecosystem** (`v1.5.0-beta` → `v1.7.5-beta`). The repository stops being standalone. `folds_csv`
replaces by-seed split selection, so that a model's original split can be reproduced; the paradata template is shared
across the repositories; `agent_dev_logs/` replaces issue comments. Then the `atrium_document` standard is integrated
(`v1.7.0-beta`), the workflows move to the hub's reusable templates pinned at `@v1` (`v1.7.2-beta`), and the release
bundle gains an AST-based completeness guard (`v1.7.1-beta`). `v1.7.4-beta` and `v1.7.5-beta` re-vendor the shared
`atrium_document.py` as its contract tightened: `doc_id` inherited from the baseline rather than derived again, and
`set_source()` filling sub-keys a partial first write left unset.

**5. Standards and service contracts** (`v1.8.0-beta` → `v1.9.10-beta`). `v1.8.0-beta` is the densest single release
([below](#2026-09-the-ensemble-moves-to-vx4-48)); the `v1.9` series adds the OpenAPI contract, the pypdfium2 renderer,
the page selection digital-convert needs, the call's paradata in every answer, the operator runbook, the corrected
evaluation, and `models_used`.

## 2026-06 to 08: retraining on the licensed dataset (#15)

The dataset had 318 pages removed for licensing reasons, and the five best models had to be retrained without them.
The question was whether that changed anything.

The path taken was to make the *split* reproducible rather than regenerate it: `--folds_csv` reads the train, dev and
test assignments from an explicit CSV and leaves out pages absent from it, so a model can be retrained on the split it
originally saw ([training.md](../training.md#retraining-on-explicit-dataset-folds)). The registry gained
`REVISION_BEST_FOLDS` and the rule `splitN ↔ foldN ↔ seed 420 + (N−1)`.

**The answer, measured on 2026-06-29:** run page by page on a 229-page sample, the `vX.4` ensemble disagreed with
`vX.3` on **24 pages**, and all 24 were pages on which the `vX.3` models already disagreed among themselves. Removing
the 318 pages moved only decisions that were ambiguous to begin with.

The same work made `split_data_from_folds` robust to pandas 3.0, which no longer coerces blank fold cells silently;
`tests/test_folds_split.py` covers the split.

## 2026-07 to 08: the Agent Skill (#26)

The classifier was packaged as an Agent Skill on the `agent-skill` branch, following an existing pattern rather than
inventing one: a `SKILL.md` with front matter and agent guidelines, and a **stdlib-only client**
(`scripts/atrium_classify.py`) that wraps the existing FastAPI service. The two-phase workflow (check `/info`, then
classify) means the same client points at any hosted endpoint by changing `--base-url` or `ATRIUM_PC_URL` and nothing
else. Both issues closed on 2026-08-02. How to use it: the hub's
[Agent skills](https://ufal.github.io/atrium-project/agent-skills/) page.

## 2026-08: guarding the install and the runtime

Two dependency rules date from this month, and each exists because of a failure the test suite could not see on its
own.

**numpy stays below 2.5.** numpy 2.5 requires Python 3.12, while every image and CI job here runs 3.11, so a floating
pin breaks the install. `setup/requirements.txt` pins numpy below 2.5 and `dependabot.yml` ignores the bump, so the
same upgrade cannot arrive by pull request.

**The service's requirements must include its server.** The test suite drives the application in-process through
`TestClient`, which needs no ASGI server, so a missing `uvicorn` would only show when a container started.
`tests/test_service_runtime_deps.py` parses `docker-compose.yml` and `setup_api_service.sh` for the console entry
points they invoke and asserts that each is declared in an installed requirements file, so pruning that list fails
the fast test lane rather than the deployment.

## 2026-08 to 09: hardening, and the hub taking over the plumbing

From 08-02 on, almost all activity was infrastructure, driven by the hub's cross-repository review passes: the shared
`atrium_document.py` re-vendored as its contract tightened, all seven workflow references repinned from `@test` to
the tagged `@v1`, and CI concurrency scoped by `github.event_name` so that a push can no longer cancel a scheduled run.

One ordering detail from this window is invisible on inspection and worth knowing: the Dockerfile's `apt-get upgrade`
layer must sit **below** the `ENV` block that embeds `ATRIUM_RUNNER_REF`. CI passes that as `github.ref_name`, unique
per release tag, which invalidates the layer on every release and only on a release. With the build using
`cache-from: type=gha`, a layer in the wrong place looks correct and patches nothing.
`tests/test_dockerfile_security_layer.py` pins the order; it became a canonical shared file across the ecosystem after
the base-image CVE incident that the translator hit first (the hub's
[Development history](https://ufal.github.io/atrium-project/development-history/), 09-13 to 09-16).

## 2026-09: the ensemble moves to `vX.4` (#48)

Release **v1.8.0-beta** switched the canonical ensemble (`--best`, and the service's `version=all`) from `vX.3` to the
retrained `vX.4` generation. The switch is one dict in `model_registry.py`, and it was held back until the Hub was
right: for three days all five `vX.4` revisions served the same `regnety_160` checkpoint, so flipping the keys would
have averaged one model with itself five times while still reporting a five-model ensemble, and nothing would have
raised. The revisions were uploaded again and checked (five distinct architectures, five distinct checkpoint sizes)
before the switch, and `tests/test_best_ensemble_distinct.py` reads each revision's configuration from the Hub, so the
same mistake cannot ship.

The same release brought two ecosystem standards (SKOS, through `atrium_vocab.py` and
`model_registry.category_uri()`, and RO-Crate, through `atrium_rocrate.py`), the service's drain semantics, and two
breaking changes; the full entry is in [CONTRIBUTING.md](../../CONTRIBUTING.md#release-history).

At the time, the published accuracy tables described `vX.3`, and the retraining comparison above was the measurement
that connected the two generations. That changed in October.

## 2026-10: the evaluation corrected (#15)

The `vX.3` figures that the README published (98–99 %) turned out to be measured on a 5,449-page set most of whose
pages were training pages: the script that selected it missed zero-padded page numbers and ids with a letter. The fix
came in steps: `v1.9.6-beta` fixed the selection script and added the split of results by document overlap;
`v1.9.8-beta` made `-rev` build on the revision's own base model (an EfficientNet revision had been fed RegNetY's
normalisation) and added the staging and SLURM scripts; `v1.9.9-beta` published the five `vX.4` models scored on their
folds' held-out test pages (94.64–95.87 % top-1) and described what was wrong with the old results. The full account:
[result/README.md](../../result/README.md); the results: [evaluation/results.md](../evaluation/results.md).

## The raw records

Not republished here:

* [`agent_dev_logs/DEVLOG.md`](../../agent_dev_logs/DEVLOG.md): the timeline index this page condenses;
* [`48.digest.md`](https://github.com/ufal/atrium-page-classification/blob/5db7d1c/agent_dev_logs/digests/48.digest.md):
  the `vX.3` and `vX.4` comparison, and
  [`48.plan.md`](https://github.com/ufal/atrium-page-classification/blob/5db7d1c/agent_dev_logs/plans/48.plan.md),
  both as they were in the last commit that held them (issue #48 closed on 2026-09-17).

This page was first written for the hub's site, condensed from `vit` at `adee922` (2026-09-23): from `DEVLOG.md`
(2026-06-25 to 2026-09-07; the window from 2026-08-02 reconstructed from `CONTRIBUTING.md` and the commit history),
`CONTRIBUTING.md`'s Release History, `model_registry.py` and the #48 digest and plan. It moved here on 2026-10-25
(atrium-project#79).
