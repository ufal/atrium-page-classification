# Configuration

For anyone who changes how the command line runs: every setting of [setup/config.txt](../setup/config.txt), its
shipped value and what it does. The HTTP service is configured by environment variables instead, listed in
[../service/README.md](../service/README.md#configuration-environment) and, in full, in
[../.env.example](../.env.example).

On this page:
[Where settings come from](#where-settings-come-from) ·
[`[OUTPUT]`](#output) · [`[EVAL]`](#eval) · [`[TRAIN]`](#train) · [`[SETUP]`](#setup) · [`[YOLO]`](#yolo) ·
[`[INPUT]`](#input) · [`[HF]`](#hf) · [`[DOCUMENT]`](#document) ·
[The paradata configuration](#the-paradata-configuration)

## Where settings come from

`run.py` reads `setup/config.txt` before it builds its argument parser, so the file supplies the defaults and a flag
overrides them for one run ([reference/cli.md](reference/cli.md)). Most of the variables you may want to change are
in `[TRAIN]`, `[HF]` and `[SETUP]`. The boolean switches of the file (`Training`, `Testing`, `raw`, `inner`,
`use_hf`, `use_yolo`, `chunking`, `strict`) default to `False` or `True` there; rather than editing them, turn them on
for a run with the flag that each one has (`--train`, `--eval`, `--raw`, …), and off with its `--no-` form.

When the set of variables changes, the documentation changes with it ([CONTRIBUTING.md](../CONTRIBUTING.md)). Some
predefined training values are not in the file at all but in [classifier.py](../classifier.py)
([training.md](training.md#hyperparameters)).

## `[OUTPUT]`

| Key              | Shipped        | What it is                                                              |
|------------------|----------------|-------------------------------------------------------------------------|
| `FOLDER_RESULTS` | `./result`     | where `tables/`, `plots/`, `stats/` and `paradata/` are written         |
| `FOLDER_CPOINTS` | `./checkpoint` | the Hugging Face cache of base models and checkpoints                   |
| `FOLDER_MODELS`  | `./model`      | the fine-tuned models, one folder each, `model_<revision without dots>` |

## `[EVAL]`

| Key            | Shipped                | What it is                                                                                                                |
|----------------|------------------------|---------------------------------------------------------------------------------------------------------------------------|
| `FOLDER_PAGES` | `./small_data_samples` | the default of `--eval_dir`: a folder of pages to evaluate on, with one sub-folder per category, like the training folder |

## `[TRAIN]`

| Key            | Shipped                | What it is                                                                                                                                                               |
|----------------|------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `FOLDER_PAGES` | `./small_data_samples` | the training pages: one sub-folder per category, whose sorted names become the labels                                                                                    |
| `cross_runs`   | `0`                    | the number of cross-validation folds (`--folds`); above 0 with `--train`, one model per fold                                                                             |
| `folds_csv`    | empty                  | an explicit folds CSV (`--folds_csv`), staged locally and not committed; when set, the split is read from it instead of being regenerated, and absent pages are left out |
| `test_size`    | `0.1`                  | the share of each category taken for the test set, and again for the development set (the 80/10/10 split)                                                                |
| `log_step`     | `100`                  | how often training logs, in steps                                                                                                                                        |
| `epochs`       | `3`                    | the number of training epochs                                                                                                                                            |
| `lr`           | `5e-5`                 | the learning rate                                                                                                                                                        |
| `max_categ`    | `14000`                | the maximum number of pages taken per category; set it above the largest category to use all pages                                                                       |
| `Training`     | `False`                | the default of `--train`                                                                                                                                                 |
| `Testing`      | `False`                | the default of `--eval`                                                                                                                                                  |

The training folder must hold the category sub-folders with images inside, and nothing else at that level. Their
names, sorted alphabetically, become the label names and replace the default category list. The label folder names
must not contain special characters such as spaces, tabs or line breaks. `max_categ` exists for over-represented
labels that dominate the data: `TEXT` was capped at 14,000 pages ([evaluation/data.md](evaluation/data.md)).

## `[SETUP]`

| Key            | Shipped                         | What it is                                                                                      |
|----------------|---------------------------------|-------------------------------------------------------------------------------------------------|
| `files_format` | `png`                           | the default of `-ff`: the extension collected from a folder                                     |
| `seed`         | `420`                           | the random seed of the data split; incremented by one for each cross-validation fold            |
| `base_model`   | `timm/regnety_160.swag_ft_in1k` | the default of `-b`; do **not** change it unless you know what you are doing                    |
| `batch`        | `16`                            | the batch size of folder runs, training and evaluation; set it to your machine's memory (below) |
| `top_N`        | `3`                             | the default of `-tn`, 1–11                                                                      |
| `raw`          | `False`                         | the default of `--raw`                                                                          |
| `inner`        | `True`                          | the default of `--inner`                                                                        |

Rough memory use per batch size, on CPU or GPU alike:

| Batch size | CPU or GPU memory |
|------------|-------------------|
| 4          | 2 GB              |
| 8          | 3 GB              |
| 16         | 5 GB              |
| 32         | 9 GB              |
| 64         | 17 GB             |

A batch below 12 is safe for an office desktop, and 4 for an old one; on a high-performance cluster, values above 20
work for training. For test runs on a CPU-only machine set it below 4, and even then a machine with more than average
memory is needed, or the system may crash.

## `[YOLO]`

The optional YOLO-cls backend (`--yolo`); see [reference/models.md](reference/models.md#alternatives-beside-the-main-line).

| Key             | Shipped          | What it is                                                                                               |
|-----------------|------------------|----------------------------------------------------------------------------------------------------------|
| `use_yolo`      | `False`          | the default of `--yolo`                                                                                  |
| `yolo_base`     | `yolov8s-cls.pt` | the default of `--yolo_base`; available: `yolov8n yv8s yv8m yv8l yv8x` and `yolo11n y11s y11m y11l y11x` |
| `yolo_imgsz`    | `224`            | the input size                                                                                           |
| `yolo_epochs`   | `30`             | training epochs                                                                                          |
| `yolo_patience` | `20`             | early-stopping patience                                                                                  |
| `yolo_lr0`      | `0`              | the initial learning rate; `0` takes `[TRAIN] lr`                                                        |
| `yolo_dropout`  | `0.1`            | the dropout of the classifier head                                                                       |
| `yolo_cache`    | `True`           | cache the images in memory or on disk for faster epochs                                                  |

## `[INPUT]`

| Key            | Shipped                | What it is                      |
|----------------|------------------------|---------------------------------|
| `FOLDER_INPUT` | `./small_data_samples` | the folder `--dir` processes    |
| `chunk_size`   | `100`                  | images per chunk with `--chunk` |
| `chunking`     | `False`                | the default of `--chunk`        |

## `[HF]`

| Key         | Shipped                    | What it is                                                                                                |
|-------------|----------------------------|-----------------------------------------------------------------------------------------------------------|
| `repo_name` | `ufal/vit-historical-page` | the Hugging Face repository of the fine-tuned models                                                      |
| `token`     | empty                      | left blank in version control: the token is read from `$HF_TOKEN` at run time                             |
| `use_hf`    | `False`                    | the default of `--hf`                                                                                     |
| `revision`  | `main`                     | the revision used when `-rev` is not given, and the name of the default model folder (`model/model_main`) |
| `latest`    | `v4.4`                     | the newest published single-model revision                                                                |

Public models need no token; only gated models and pushing to the Hub need one (`export HF_TOKEN=hf_xxx` on Unix,
`setx HF_TOKEN hf_xxx` on Windows). `--best` does not read `latest`: the ensemble is fixed in code
([reference/models.md](reference/models.md#the-ensemble)). The Hub's `main` is still `v4.3`, so name a revision
([usage.md](usage.md#get-a-model)).

## `[DOCUMENT]`

The ATRIUM document record ([usage.md](usage.md#write-a-document-record)). A single-file run (`-f`) uses
`document_json` and `document_json_out` as file paths; a folder run (`-d`) uses `document_json_dir` and
`document_json_out_dir` as folders, with one `<doc_id>.document.json` per document. All four are blank by default:
the record is opt-in, and nothing is written unless an output path or folder is set, here or on the command line.

| Key                     | Shipped | What it is                               |
|-------------------------|---------|------------------------------------------|
| `document_json`         | empty   | the default of `--document-json`         |
| `document_json_out`     | empty   | the default of `--document-json-out`     |
| `document_json_dir`     | empty   | the default of `--document-json-dir`     |
| `document_json_out_dir` | empty   | the default of `--document-json-out-dir` |
| `strict`                | `False` | the default of `--strict-document-json`  |

## The paradata configuration

[setup/para_config.txt](../setup/para_config.txt) is not a run setting: it names the program and its version for the
paradata, and declares the components and their licences ([reference/licences.md](reference/licences.md)).
