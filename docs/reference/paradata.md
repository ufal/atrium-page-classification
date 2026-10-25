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
  committed [result/paradata](../../result/paradata) holds the logs of the 2026-10-22 evaluation runs.
* **Metrics:** the exact start and end times, the duration in seconds, the speed (files per minute), and the counts
  of files written (CSVs, PNGs) and of inputs skipped. An image that cannot be read is logged as a skip with its
  reason; `--eval` counts the pages it scored and the tables it wrote.
* **A configuration snapshot:** the model revision, the base model actually used, the batch size, the input path and
  the command-line flags of the run.
* **Tool version and runner:** the `tool_version` (from `para_config.txt`), the `repository` and `runner_ref`, and
  a `docker_image` field. In a published container the repository and runner come from the `ATRIUM_RUNNER_REPO`,
  `ATRIUM_RUNNER_REF` and `ATRIUM_RUNNER_IMAGE` environment variables, so the reference names the actual runner
  rather than a fixed fork URL; outside a container `docker_image` is empty.
* **Components:** each component the run used, with its licence, as declared in `para_config.txt`: the models
  always; the LINDAT dataset with `--train` or `--eval`; Ultralytics with `--yolo`. The log also carries the label
  that `para_licenses.py` computes from them, the most restrictive of their licences, with the supporting detail
  under `license_detail`; the label is not the licence of the output ([licences.md](licences.md)).

The HTTP service writes no paradata file. Each successful answer carries the call's provenance instead, as
`paradata`: its Process Run Crate `CreateAction`, whose `@id` is the `run_uuid` stamped into the record, with
`pypdfium2` among the components when `/predict_document` rendered a PDF
([service/README.md](../../service/README.md#api-usage)).

Example: the paradata of the [small_data_samples](../../small_data_samples) run of the evaluation,
[261022-101849_page-classification.json](../../result/paradata/261022-101849_page-classification.json).

## The document record schema

The per-document record this tool reads and writes, `<doc_id>.document.json`, the pair of the paradata above,
follows [atrium_document.schema.json](../../atrium_document.schema.json), schema version **`1.0`**. That version is
frozen as the hub tag [`doc-schema-v1`](https://github.com/ufal/atrium-project/releases/tag/doc-schema-v1)
(ufal/atrium-project@`544298b`), and it is the baseline this tool implements:

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
