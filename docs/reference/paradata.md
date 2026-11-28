# Paradata and the record schema

For integrators and anyone who audits a run: what a run records about itself, and the schema of the document record
it reads and writes. The licences of the
components a run records: [licences.md](licences.md).

## Paradata logging

Every command-line run writes a paradata log that records its provenance, configuration and statistics. It is
written by the `ParadataLogger` of [atrium_paradata.py](../../atrium_paradata.py), a module shared by all ATRIUM
tools, and driven by this repository's [para_config.txt](../../setup/para_config.txt).

* **One file per run,** named `YYMMDD-HHmmss_page-classification.json`, written at the end of the run into
  `paradata/` under the run's output folder (`[OUTPUT] FOLDER_RESULTS`, so `./result/paradata` by default and
  `./data/output/paradata` under Docker Compose), so that the paradata travels with the data it describes. The
  logs of the two sample runs over `small_data_samples/` (one model, and the `--best` ensemble) are committed in
  `result/paradata/`, beside the tables they wrote ([result/README.md](../../result/README.md)). The logs of the
  2026-10-22 evaluation runs predate paradata 3.0 and carried a calculated licence; git history up to
  `v1.14.2-beta` keeps them.
* **Metrics:** the exact start and end times, the duration in seconds, the speed (files per minute), and the counts
  of files written (CSVs, PNGs) and of inputs skipped. An image that cannot be read is logged as a skip with its
  reason; `--eval` counts the pages it scored and the tables it wrote.
* **A configuration snapshot:** the settings only (atrium-project#84 decision 1): the model revision, the base
  model actually used, the model folder by its name (a Hugging Face id as given), the batch size and the
  command-line flags of the run; never a path or the name of a file read or written. A `--best` run names its five
  ensemble revisions and their base models.
* **Parameters:** `revisions`, the model revisions that made the predictions (the five `vX.4` with `--best`);
  `pages`, the page selection of a PDF (`--pages`); and `page_selection`, the rule the pages were selected by with
  its statement (`--page-selection`: `forced+figures`, `explicit`, `all`; decision 4), as the service records them.
* **Tool version and runner:** the `tool_version` (from `para_config.txt`), the `repository` and `runner_ref`, and
  a `docker_image` field. In a published container the repository and runner come from the `ATRIUM_RUNNER_REPO`,
  `ATRIUM_RUNNER_REF` and `ATRIUM_RUNNER_IMAGE` environment variables, so the reference names the actual runner
  rather than a fixed fork URL; outside a container `docker_image` is empty.
* **Components:** each component the run used, with its own licence, as declared in `para_config.txt`: the models
  always; the LINDAT dataset with `--train` or `--eval`; Ultralytics with `--yolo`; pypdfium2 when a PDF is
  rendered; the decision model in a run of the parked `decide_pages.py` experiment. The list is `components`, a list of
  facts: no licence is computed from it, and none of these licences is the licence of the output
  ([licences.md](licences.md)).
* **Inputs and outputs** (paradata 3.0): the record version read and the record written, each by its AMČR
  distribution name (`atrium/document-json`) and SHA-512; the record written names the blocks and fields the run
  wrote (`record_fields`). The images and PDFs a command-line run read, and the tables and plots it wrote under the
  results folder, are entries too, by file name and SHA-512, `stored: false` (decision 1).

The HTTP service writes no paradata file. Each successful answer carries the call's paradata instead, as
`paradata`: its Process Run Crate `CreateAction`. `object` is the upload (the original, or an office document's
rendered PDF, `rendered-pdf`, `stored: false`) and, when one was sent, the record version read; `result` is the record
written, when one was asked for. The pages classified are a parameter, with their `page_selection`, and `pypdfium2`
is among the components when `/predict_document` rendered a PDF
([service/README.md](../../service/README.md#how-it-works)).

## The document record schema

The per-document record this tool reads and writes, `<doc_id>.document.json`, the pair of the paradata above,
follows [atrium_document.schema.json](../../atrium_document.schema.json), schema version **`2.0`**
(doc-schema-v2, [atrium-project#84](https://github.com/ufal/atrium-project/issues/84)): content only, no run data
and no licence, page rows keyed by `pages[].page_index`. A 1.x record is read and migrated. `doc-schema-v2` is tagged
only after the end-to-end proof; until then the published reference is `doc-schema-v1`, frozen as the hub tag
[`doc-schema-v1`](https://github.com/ufal/atrium-project/releases/tag/doc-schema-v1)
(ufal/atrium-project@`544298b`):

* [tests/test_schema_freeze.py](../../tests/test_schema_freeze.py) checks this repository's schema against the frozen
  copy beside it, [atrium_document.schema.doc-schema-v1.json](../../atrium_document.schema.doc-schema-v1.json): the
  copy is exactly the tagged file, nothing declared at the freeze has been removed or renamed, and every change made
  since is registered;
* all three files, like `atrium_document.py`, are vendored from the hub and kept byte-identical to its `v1` tag by
  the `para-drift` CI check, so they are never edited in this repository. The hub also checks that every record
  shape the tools write validates under both the frozen and the current schema.

What may change after the freeze, and what a new major version takes, is in the hub's
[Freeze & conformance](https://github.com/ufal/atrium-project/blob/main/docs/document_schema.md#freeze--conformance).
[CITATION.cff](../../CITATION.cff) carries the same reference under `references`. Which fields this tool writes:
[outputs.md](outputs.md#the-document-record).
