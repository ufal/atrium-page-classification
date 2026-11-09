# Command line: `run.py` and `decide_pages.py`

For anyone who runs the command line and wants the exact flag: every flag of `run.py`, its default and what it does,
as `build_parser()` in [run.py](../../run.py) defines them.
Examples of their use: [../usage.md](../usage.md) and [../training.md](../training.md). `python3 run.py -h` prints
the same list. The decision-model experiment, `decide_pages.py`, has its own section at the end.

Defaults come from [setup/config.txt](../../setup/config.txt), read before the parser is built
([../configuration.md](../configuration.md)). Flags written `--x / --no-x` are `argparse.BooleanOptionalAction`:
`--no-x` overrides a `True` in the config file, which a bare `store_true` could not do.

On this page:
[Input and output](#input-and-output) ·
[Model selection](#model-selection) ·
[Ensemble](#ensemble) ·
[Training and evaluation](#training-and-evaluation) ·
[The document record](#the-document-record) ·
[Decision models on page images: `decide_pages.py`](#decision-models-on-page-images-decide_pagespy)

## Input and output

| Flag                   | Default                        | What it does                                                                                                                       |
|------------------------|--------------------------------|------------------------------------------------------------------------------------------------------------------------------------|
| `-f`, `--file`         | none                           | a single page image (its one-row table is written too), or a PDF, recognised by its content: its pages are rendered and classified |
| `-d`, `--directory`    | none                           | a folder of unprocessed pages                                                                                                      |
| `--dir`                | off                            | process the folder named in `[INPUT] FOLDER_INPUT` instead of passing `-d`                                                         |
| `-ff`, `--file_format` | `png` (`[SETUP] files_format`) | the extensions to collect from a directory, comma-separated, any case (`png,jpg,tif`); `pdf` adds the PDFs, rendered               |
| `--pages`              | none (every page)              | PDF inputs only: the pages to classify, `1,3,5-7`, as `/predict_document`'s `pages`; a malformed value exits 2                     |
| `--inner / --no-inner` | `True` (`[SETUP] inner`)       | recurse into nested folders                                                                                                        |
| `--chunk / --no-chunk` | `False` (`[INPUT] chunking`)   | write predictions in chunks of `[INPUT] chunk_size` (100), appending to the output file as the run proceeds                        |
| `-tn`, `--topn`        | `3` (`[SETUP] top_N`)          | how many categories to report, 1–11; another value exits 2                                                                         |
| `--raw / --no-raw`     | `False` (`[SETUP] raw`)        | also write the scores of all eleven categories                                                                                     |

Input is chosen only by `-f`, `-d` or `--dir`; without one of them nothing is processed. A PDF is rendered at
`PDF_RENDER_DPI` (300) by the service's renderer, its pages named `<doc_id>-<page>`.

## Model selection

| Flag                 | Default                                                     | What it does                                                                                        |
|----------------------|-------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| `-m`, `--model`      | `./model/model_<[HF] revision without dots>` (`model_main`) | the folder of the model to load; with `-rev` it becomes `./model/model_<revision without dots>`     |
| `-b`, `--base`       | `timm/regnety_160.swag_ft_in1k` (`[SETUP] base_model`)      | the base model of a revision that `model_registry.py` does not list                                 |
| `-rev`, `--revision` | `[HF] revision` (`main`)                                    | the Hugging Face revision, `vN.M`; a revision no key of `REVISION_TO_BASE_MODEL` matches is refused |
| `--hf / --no-hf`     | `False` (`[HF] use_hf`)                                     | take the model and processor from the Hugging Face Hub, and save them under `./model/`              |
| `--models-dir`       | unset                                                       | a staged folder (`model/`, `checkpoint/`, `hf/`): `--hf` fills it, runs without it read it offline  |
| `--yolo / --no-yolo` | `False` (`[YOLO] use_yolo`)                                 | use a YOLO-cls model instead of ViT/CNN; overrides `--base` and `--revision`                        |
| `--yolo_base`        | `yolov8s-cls.pt` (`[YOLO] yolo_base`)                       | a short tag (`yv8s`), an Ultralytics id, or a local `.pt` path                                      |

A listed revision builds on its own base model, whatever `-b` or the config says
([models](models.md#how-a-revision-is-named)). The registry's keys are the exact revisions `v1.3`–`v5.3` and
`v1.4`–`v5.4`, and the prefixes `v3.`, `v4.`, `v5.` and `v6.`–`v12.`; a revision that matches none of them is refused
before anything loads (`Revision … is not supported`). That includes `main`, although `-h` names it (leave `-rev` out
to use `[HF] revision`), and the early `v1.2` and `v2.0`–`v2.2`. `--yolo` does not support `--hf` or `--best`.

## Ensemble

| Flag                   | Default | What it does                                                                                                |
|------------------------|---------|-------------------------------------------------------------------------------------------------------------|
| `--best`               | off     | run the five models of the ensemble; the result is averaged into one Top-N CSV                              |
| `--parallel`           | off     | the memory-aware grouped engine for `--best`; CUDA only, and it falls back to the sequential path otherwise |
| `--no-average-best`    | off     | skip the averaging; write only the wide `BEST_5_models_TOP-1.csv` for manual processing                     |
| `--save-intermediates` | off     | also write each model's own `{date-time}_{revision}_TOP-{N}.csv` before averaging                           |

How the engine schedules the models: [../how-it-works.md](../how-it-works.md#the-memory-aware---best-engine).

## Training and evaluation

| Flag                       | Default                      | What it does                                                                                                                                                       |
|----------------------------|------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--train / --no-train`     | `False` (`[TRAIN] Training`) | train                                                                                                                                                              |
| `--eval / --no-eval`       | `False` (`[TRAIN] Testing`)  | evaluate                                                                                                                                                           |
| `--folds`                  | `0` (`[TRAIN] cross_runs`)   | the number of cross-validation folds, 80/10/10 split; used only with `--train`                                                                                     |
| `--folds_csv`              | `[TRAIN] folds_csv`          | an explicit folds CSV (a `PNG` column and `foldN` columns of `train`/`dev`/`test`); it takes precedence over `--folds`, and pages absent from the CSV are left out |
| `--fold_column`            | per revision                 | which fold column to read; without it, resolved per revision from `REVISION_BEST_FOLDS`                                                                            |
| `--eval_dir`               | `[EVAL] FOLDER_PAGES`        | the folder, with one sub-folder per category, to evaluate on; with `--folds_csv`, only the fold's `test` pages in it count                                         |
| `--average`                | off                          | average the weights of existing fold models (needs `-ap`)                                                                                                          |
| `-ap`, `--average_pattern` | none                         | which weights to average, such as `model_v4`                                                                                                                       |

## The document record

| Flag                                                 | Default                            | What it does                                                    |
|------------------------------------------------------|------------------------------------|-----------------------------------------------------------------|
| `--document-json`                                    | `[DOCUMENT] document_json`         | the baseline record, for a single-file run                      |
| `--document-json-out`                                | `[DOCUMENT] document_json_out`     | where to write the updated record                               |
| `--document-json-dir`                                | `[DOCUMENT] document_json_dir`     | a baseline directory, for a batch run                           |
| `--document-json-out-dir`                            | `[DOCUMENT] document_json_out_dir` | the output directory, one `<doc_id>.document.json` per document |
| `--strict-document-json / --no-strict-document-json` | `False` (`[DOCUMENT] strict`)      | turn ownership and schema warnings into errors                  |

The record is opt-in and a no-op unless an output path or directory is set, on the command line or in the config.
`-f` with `--document-json-out-dir` writes `<doc_id>.document.json` there (its baseline from `--document-json-dir`);
`-d` with `--document-json-out` exits 2, as one path cannot hold a folder's records. The pages of a PDF are written
under a born-digital baseline's own page labels (`pages[].page_index`), as `/predict_document` writes them.
What the record holds: [outputs.md](outputs.md#the-document-record).

## Decision models on page images: `decide_pages.py`

An experiment, not part of the classifier. `decide_pages.py` sends page images to a decision model and writes its
answers to a table. It is not in the image and not in the service, so it has no API counterpart. The model runs on a
server that answers `POST /v1/systemone` (Clef, Clef-Flash or Strands Decider 2B), reached through
[decision_client.py](../../decision_client.py), so this side needs no torch and no GPU. Only the page image is sent,
never its file name, label or text. Which archive pages may go to which server is still being settled
([atrium-project#83](https://github.com/ufal/atrium-project/issues/83)); the public `small_data_samples` may go
anywhere. `python3 decide_pages.py -h` prints the flags.

```bash
python3 decide_pages.py --write-question-sets                       # regenerate decisions/*.json
export DECISION_MODEL_URL=<server> DECISION_MODEL_NAME=<model>     # or --url and --model on each run
python3 decide_pages.py --task triage --dir small_data_samples --per-label 3
python3 data_scripts/degrade_pages.py --dir small_data_samples --per-label 2 --out stage/degraded
python3 decide_pages.py --task b4 --degradations stage/degraded/degradations.csv
```

### What each task asks

| Task     | What the model is asked                                                                                                                                                                                                                                 | Question set                       |
|----------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------|
| `b1`     | the eleven categories, each with its definition: one choice                                                                                                                                                                                             | `decisions/page_category.v1.json`  |
| `b2`     | the same choice behind the classifier: only the pages whose `run.py --raw` top-1 score is below `--tau`, and only among its top 3                                                                                                                       | `decisions/page_category.v1.json`  |
| `b3`     | seven yes/no page elements (printed text, handwriting, a table or form, filled-in fields, a photograph, a drawing, a stamp) and the processor that matters most; the four processor tags (OCR, HTR, tables, graphics) are computed from the elements    | `decisions/page_routes.v2.json`    |
| `b4`     | fifteen yes/no defects of the paper, the content and the capture, the orientation of the main text, and three damage scores from 0 (none) to 4 (severe)                                                                                                 | `decisions/page_condition.v1.json` |
| `triage` | `b3` and `b4` in one request: 27 questions                                                                                                                                                                                                              | both of the above                  |

The question sets are generated from `decide_pages.py` and never edited by hand: `--write-question-sets` rewrites
them, and the tests fail when a committed file differs from the generator.

### Flags

| Flag                    | Default                                                   | What it does                                                                                                                                                                          |
|-------------------------|-----------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--write-question-sets` | off                                                       | regenerate the three files in `decisions/` and exit                                                                                                                                   |
| `--task`                | none                                                      | `b1`, `b2`, `b3`, `b4` or `triage`; required unless `--write-question-sets`                                                                                                           |
| `--dir`                 | none                                                      | a folder with one sub-folder per category; the folder names the true label                                                                                                            |
| `--folds_csv`           | none                                                      | a folds CSV (`PNG`, `categ` and `foldN` columns); needs `--root`                                                                                                                      |
| `--fold`                | `fold1`                                                   | the fold column of `--folds_csv`                                                                                                                                                      |
| `--part`                | `dev`                                                     | the partition of that fold: `train`, `dev` or `test`                                                                                                                                  |
| `--root`                | none                                                      | where `data_scripts/stage_fold_pages.py` put that partition's pages                                                                                                                   |
| `--degradations`        | none                                                      | `b3`, `b4`, `triage` only: the manifest `data_scripts/degrade_pages.py` wrote ([../data-preparation.md](../data-preparation.md#controlled-defects-for-the-page-condition-experiment)) |
| `--per-label`           | all pages                                                 | at most N pages per category, drawn with `--seed`; refused with `--degradations`                                                                                                      |
| `--seed`                | `420`                                                     | the seed of `--per-label`                                                                                                                                                             |
| `--limit`               | none                                                      | only the first N jobs, one job being one page at one `--max-side`, whether it sends a request or not; with `--resume`, the jobs already in the table count                            |
| `--cnn-raw`             | none                                                      | `b2`: the `run.py --raw` table of the same pages                                                                                                                                      |
| `--tau`                 | none                                                      | `b2`: the pages whose top-1 score in `--cnn-raw` is below this go to the model                                                                                                        |
| `--accept-at`           | none                                                      | the lower bound of the `accept` lane ([Lanes](#lanes))                                                                                                                                |
| `--review-below`        | none                                                      | the upper bound of the `review` lane ([Lanes](#lanes))                                                                                                                                |
| `--max-side`            | `1024`                                                    | the longest side of the image as sent, in pixels: a larger page is scaled down and sent as JPEG; repeatable, one pass per size                                                        |
| `--url`                 | `$DECISION_MODEL_URL`                                     | the server; `/v1/systemone` is added unless the URL ends with it                                                                                                                      |
| `--model`               | `$DECISION_MODEL_NAME`                                    | the model the server is asked for                                                                                                                                                     |
| `--model-revision`      | empty                                                     | the served build or Hugging Face revision, for the paradata and the summary                                                                                                           |
| `--served-on`           | `local` for a loopback URL, else `other`                  | where the pages went: `cluster-job`, `ufal-endpoint`, `local` or `other`; recorded instead of the URL                                                                                 |
| `--timeout`             | `$DECISION_TIMEOUT_S`, else `120`                         | seconds to wait for one answer                                                                                                                                                        |
| `--retries`             | `2`                                                       | how often a request is retried after a 429, a 5xx or a failed connection; a timeout is not retried                                                                                    |
| `--image-style`         | `data_uri`                                                | how the image travels: `data_uri`, `base64`, `url` or `image_url`; `python3 decision_client.py --probe` finds the one a server reads                                                  |
| `--no-proxy`            | off                                                       | do not use the proxy that the environment variables name                                                                                                                              |
| `--out`                 | `result/decide/<date-time>_<task>_<model>_DECIDE_RAW.csv` | the table; an existing file is refused without `--resume`                                                                                                                             |
| `--resume`              | off                                                       | continue `--out`, skipping the page and size pairs already in it                                                                                                                      |

Pages come from `--degradations`, from `--dir`, or from `--folds_csv` with `--root`, in that order of precedence. `b2`
needs `--cnn-raw` and `--tau`. A missing input, an empty page list, an unset `--url` or `--model`, or `--resume`
without `--out` exits 2, before any request is sent.

### Lanes

`--accept-at` and `--review-below` sort each answer by its confidence into a lane: `accept` at or above
`--accept-at`, `review` below `--review-below`, and `escalate` in between. The confidence is that of the category
choice in `b1` and `b2`, and of the primary processor in `b3` and `triage`; `b4` has no lane. The thresholds belong to
one model revision and one wording of the questions, so set them on dev data.

Give both or neither; without them the `lane` column is empty. Each must lie within 0–1, and `--review-below` must
not exceed `--accept-at`. Otherwise the run exits 2 before any request is sent.

### What a run writes

* **The table,** `DECIDE_RAW`, at `--out`; the default name carries the UTC date and time, the task and the model. It
  has one row per page and image size, appended as the answers come, so a stopped run can be resumed. A failed
  request leaves its answer columns empty, and the run goes on.
* **The summary,** `<table>.summary.json` beside it, also printed at the end: the count of each status, then per
  image size the latency (median and 95th percentile), the mean input tokens and the task's figures. It covers the
  whole table, resumed rows included.
* **The paradata,** in `paradata/` beside the table: the mode `decide_<task>`, the model and its revision,
  `--served-on`, the question set's digest, the image style and sizes, the thresholds and the pages. It lists the
  `decision_model` component, Apache-2.0 ([licences.md](licences.md)). The server's URL is recorded nowhere.

The columns of the table, by task:

| Columns              | What they hold                                                                                                                                                                                                                                                                                                                             |
|----------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| every task           | `FILE`, `PAGE` (as every result table names a page), `TRUE` (the label of the page's folder)                                                                                                                                                                                                                                               |
| `b1`, `b2`           | `CLASS-1`, the answer; `P_<category>` for each of the eleven; `confidence`; `lane`                                                                                                                                                                                                                                                         |
| `b2`                 | also `CNN_CLASS` and `CNN_SCORE`, the classifier's answer; `deferred` (1 when the page went to the model) and `options`, its top 3. `CLASS-1` keeps the classifier's answer when the page is not deferred or the request fails                                                                                                             |
| `b3`, `triage`       | `P_<element>` and `GOLD_<element>` (1, 0, or empty when the true label does not say); `P_route_<tag>`, the largest P of its elements, and `GOLD_route_<tag>`; `primary_route`, `P_primary_<tag>`, `GOLD_primary` and `lane`                                                                                                                |
| `b4`, `triage`       | `P_<defect>`; `orientation`, `P_orient_<orientation>` and `GOLD_orientation`; `E_paper_condition`, `E_content_condition` and `E_overall_damage`, the expected scores 0–4; `damage_score`, `E_overall_damage` / 4; `paper_index`, `content_index`, `capture_index` and `damage_index`, the mean defect probabilities, a check on the scores |
| `b3`, `b4`, `triage` | also `VARIANT` (the injected defect, or `original`), `LEVEL` and `SOURCE`, from `--degradations`                                                                                                                                                                                                                                           |
| every task           | `status`, `error`, `latency_ms`, `input_tokens`, `attempts`, `schema_digest` (the question set's digest, the prompt's version), `max_side`, `image`                                                                                                                                                                                        |

`status` is `ok`, `invalid` (an answer that does not fit its question), `timeout`, `error`, `rejected` or
`not_configured`. A page that `b2` does not defer is `not_requested`.

The task's figures in the summary: top-1 accuracy in `b1` and `b2`; in `b2` also the deferred pages, those the model
fixed and broke, McNemar's exact p and the classifier's own top-1. In `b3`, precision, recall and F1 of each element
and tag against what the true label implies, and the primary processor's accuracy. In `b4`, the rate of each defect,
the mean scores and the indices' rank correlation with the scores; with `--degradations`, also the paired change of
each injected defect's probability against the original page, and the orientation accuracy on text pages.
