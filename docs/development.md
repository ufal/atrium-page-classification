# Development

For contributors, and for anyone who wants to reuse this code as the base of their own image classification task:
what each file does, the helper scripts, the branches, and where the rules for tests and releases are. The rules
themselves (branches, pull requests, commit messages, code conventions, the test suite) are in
[CONTRIBUTING.md](../CONTRIBUTING.md), whose Release History is the changelog.

On this page:
[The files](#the-files) ·
[Command-line utility scripts](#command-line-utility-scripts) ·
[Branches](#branches) ·
[Shared and vendored files](#shared-and-vendored-files) ·
[Tests and checks](#tests-and-checks) ·
[The documentation](#the-documentation)

## The files

| File                                        | What it is                                                                                                                  |
|---------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| `run.py`                                    | the starting point of the program, with its main function; extend it for new flags and function arguments                   |
| `classifier.py`                             | the model-specific classes and functions, including predefined values for the training arguments                            |
| `utils.py`                                  | the task-related algorithms: collecting images, file names to document and page, result tables, confusion plots             |
| `model_registry.py`                         | the one source of truth for the categories, the revisions and their base models, the ensemble and its folds, `MODEL_STATIC` |
| `parallel_best.py`, `ensemble.py`           | the `--best` engine and the averaging of the ensemble                                                                       |
| `yolo_classifier.py`                        | the optional YOLO-cls backend                                                                                               |
| `atrium_document_adapter.py`                | writes this tool's blocks into the ATRIUM document record                                                                   |
| `tool_limits.py`                            | the service's limits, each an environment setting                                                                           |
| `service/`                                  | the HTTP service ([service/README.md](../service/README.md))                                                                |
| `setup/config.txt`                          | the changeable variables of the program ([configuration.md](configuration.md))                                              |
| `setup/para_config.txt`                     | the tool's version and its components for the paradata                                                                      |
| `data_scripts/`                             | data preparation ([data-preparation.md](data-preparation.md)) and the evaluation helpers                                    |
| `supplementary/scripts/job_run.sh`          | runs the program on a server node                                                                                           |
| `supplementary/scripts/result_analysis.sh`  | computes performance scores for saved model results                                                                         |
| `supplementary/scripts/dataset_timeline.py` | plots the distribution of the categories over time, from the file names                                                     |
| `supplementary/scripts/img2jpeg_v3.py`      | turns any images into JPEG                                                                                                  |
| `supplementary/scripts/logs_stat.py`        | makes a table of statistics for each TensorBoard directory of event logs                                                    |
| `supplementary/scripts/visualize.py`        | plots a comparison of model types from an input CSV such as [model_accuracies_new.csv](../model_accuracies_new.csv)         |

For finer adjustments of training, see the functions in [classifier.py](../classifier.py), which hold predefined values
that [run.py](../run.py) does not use ([training.md](training.md#hyperparameters)).

## Command-line utility scripts

The analysis scripts in [supplementary/scripts](../supplementary/scripts) run in one line, without editing paths
inside them; call any of them with `--help` for its arguments:

* [dataset_timeline.py](../supplementary/scripts/dataset_timeline.py) plots the chronological distribution of the
  categories, with `-i` (the input CSV), `-o` (the output plot) and `--regex` (the grouping).
* [visualize.py](../supplementary/scripts/visualize.py) plots model parameters against accuracy, with `-i` and `-o`.
* [logs_stat.py](../supplementary/scripts/logs_stat.py) parses binary TFRecord event logs into a CSV of training
  metrics. It accepts external JSON mappings (`--gpu-map`, `--revision-map`), so you can track your own nodes and
  architectures.
* [averaging.py](../supplementary/scripts/averaging.py), [per_doc_split.py](../supplementary/scripts/per_doc_split.py)
  and [result_analysis.sh](../supplementary/scripts/result_analysis.sh) work on result tables
  ([usage.md](usage.md#combining-and-post-processing-results));
  [filtering.py](../supplementary/scripts/filtering.py) and [downscale.py](../supplementary/scripts/downscale.py) on
  the dataset ([data-preparation.md](data-preparation.md#dataset-maintenance));
  [eval_vX4_job.sh](../supplementary/scripts/eval_vX4_job.sh) re-runs the evaluation
  ([training.md](training.md#re-running-the-vx4-evaluation)).

## Branches

| Branch        | What it is                                                                                                          |
|---------------|---------------------------------------------------------------------------------------------------------------------|
| `vit`         | the default branch and the code (ViT, RegNetY and EfficientNetV2 models)                                            |
| `test`        | the integration branch that changes are staged on; branch from it ([CONTRIBUTING.md](../CONTRIBUTING.md))           |
| `master`      | a short index pointing at the `vit` and `clip` variants                                                             |
| `clip`        | a CLIP-based variant, with its own Hugging Face repository                                                          |
| `agent-skill` | the Agent Skill packaging of the service (#26)                                                                      |
| `gh-pages`    | the landing card at [ufal.github.io/atrium-page-classification](https://ufal.github.io/atrium-page-classification/) |

The `vit` and `clip` names are model families, not development stages: the repository began as a comparison of the
two approaches.

## Shared and vendored files

The `atrium_*.py` modules, `para_licenses.py`, `check_version.py`, the record schemas
(`atrium_document.schema.json` and its frozen copy), `service/atrium_service.py`, `service/healthcheck.py` and some
tests are vendored from the hub (its `docs/templates/shared/MANIFEST.json`). The `para-drift` CI check keeps them
byte-identical to the hub, so they are never edited here; the hub's `scripts/revendor_shared.sh` refreshes them. The
record schema and its freeze: [reference/paradata.md](reference/paradata.md#the-document-record-schema).

## Tests and checks

The test suite needs no model and no GPU for its default run; tests that need models, a GPU or the network are
marked `slow`. How to install and run it, the fixture conventions, and the checks to run before every commit are in
[CONTRIBUTING.md](../CONTRIBUTING.md#code-conventions--testing). The tests check that the code does what it says;
they do not measure model quality ([evaluation/results.md](evaluation/results.md)).

The service has its own contract tests: the OpenAPI document (`tests/test_openapi_contract.py`), the limits table
(`tests/test_limits_contract.py`) and the environment ledger (`tests/test_env_contract.py`) are checked against
[service/README.md](../service/README.md), [service/openapi.json](../service/openapi.json) and
[.env.example](../.env.example). After an API change, regenerate the OpenAPI document
([service/README.md](../service/README.md#openapi-the-typed-contract)).

## The documentation

| File                                                                   | For whom                            | What it holds                                               |
|------------------------------------------------------------------------|-------------------------------------|-------------------------------------------------------------|
| [README.md](../README.md)                                              | a first-time visitor                | what it does, a quick start, limits, the documentation map  |
| [usage.md](usage.md)                                                   | a user                              | install, classify, read the results, Docker                 |
| [service/README.md](../service/README.md)                              | a deployer or integrator            | the service runbook: start, models, limits, errors, the API |
| [configuration.md](configuration.md)                                   | a user                              | every setting of `setup/config.txt`                         |
| [how-it-works.md](how-it-works.md)                                     | anyone                              | how a page is classified, the ensemble, the `--best` engine |
| [reference/](reference/categories.md)                                  | a developer or expert               | categories, models, flags, outputs, paradata, licences      |
| [evaluation/](evaluation/results.md)                                   | anyone judging the results          | results, data, the 2025 sweep                               |
| [data-preparation.md](data-preparation.md), [training.md](training.md) | someone building or training models | the dataset and the training runs                           |
| [decisions/history.md](decisions/history.md)                           | a contributor                       | why it is shaped the way it is                              |
| [CONTRIBUTING.md](../CONTRIBUTING.md)                                  | a contributor                       | rules, tests, the release history                           |

The pages follow the ATRIUM
[documentation style](https://github.com/ufal/atrium-project/blob/test/docs/documentation-style.md); the hub's
`tools/docs_check.py --repo atrium-page-classification` checks their links, anchors and headings.
