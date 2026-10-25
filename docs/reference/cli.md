# Command line: `run.py`

For anyone who runs the command line and wants the exact flag: every flag of `run.py`, its default and what it does,
as `build_parser()` in [run.py](../../run.py) defines them.
Examples of their use: [../usage.md](../usage.md) and [../training.md](../training.md). `python3 run.py -h` prints
the same list.

Defaults come from [setup/config.txt](../../setup/config.txt), read before the parser is built
([../configuration.md](../configuration.md)). Flags written `--x / --no-x` are `argparse.BooleanOptionalAction`:
`--no-x` overrides a `True` in the config file, which a bare `store_true` could not do.

On this page:
[Input and output](#input-and-output) ·
[Model selection](#model-selection) ·
[Ensemble](#ensemble) ·
[Training and evaluation](#training-and-evaluation) ·
[The document record](#the-document-record)

## Input and output

| Flag                   | Default                        | What it does                                                                                                |
|------------------------|--------------------------------|-------------------------------------------------------------------------------------------------------------|
| `-f`, `--file`         | none                           | a single page image                                                                                         |
| `-d`, `--directory`    | none                           | a folder of unprocessed pages                                                                               |
| `--dir`                | off                            | process the folder named in `[INPUT] FOLDER_INPUT` instead of passing `-d`                                  |
| `-ff`, `--file_format` | `png` (`[SETUP] files_format`) | the extension to collect from a directory                                                                   |
| `--inner / --no-inner` | `True` (`[SETUP] inner`)       | recurse into nested folders                                                                                 |
| `--chunk / --no-chunk` | `False` (`[INPUT] chunking`)   | write predictions in chunks of `[INPUT] chunk_size` (100), appending to the output file as the run proceeds |
| `-tn`, `--topn`        | `3` (`[SETUP] top_N`)          | how many categories to report; checked at start: it must be 1–11                                            |
| `--raw / --no-raw`     | `False` (`[SETUP] raw`)        | also write the scores of all eleven categories                                                              |

Input is chosen only by `-f`, `-d` or `--dir`; without one of them nothing is processed.

## Model selection

| Flag                 | Default                                                     | What it does                                                                                        |
|----------------------|-------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| `-m`, `--model`      | `./model/model_<[HF] revision without dots>` (`model_main`) | the folder of the model to load; with `-rev` it becomes `./model/model_<revision without dots>`     |
| `-b`, `--base`       | `timm/regnety_160.swag_ft_in1k` (`[SETUP] base_model`)      | the base model of a revision that `model_registry.py` does not list                                 |
| `-rev`, `--revision` | `[HF] revision` (`main`)                                    | the Hugging Face revision, `vN.M`; a revision no key of `REVISION_TO_BASE_MODEL` matches is refused |
| `--hf / --no-hf`     | `False` (`[HF] use_hf`)                                     | take the model and processor from the Hugging Face Hub, and save them under `./model/`              |
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
What the record holds: [outputs.md](outputs.md#the-document-record).
