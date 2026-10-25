# Outputs and their columns

For anyone who processes the result files further: what a command-line run writes, and the columns of each table. The service answers in JSON instead; its fields are
in [service/README.md](../../service/README.md#api-usage) and [service/openapi.json](../../service/openapi.json).
How to read a result: [../usage.md](../usage.md#reading-the-output).

On this page:
[The files a run writes](#the-files-a-run-writes) ·
[The Top-N table](#the-top-n-table) ·
[The raw table](#the-raw-table) ·
[The ensemble tables](#the-ensemble-tables) ·
[The document record](#the-document-record) ·
[Example tables in this repository](#example-tables-in-this-repository)

## The files a run writes

Tables and plots go under `[OUTPUT] FOLDER_RESULTS` (default `./result`; under Docker Compose, `./data/output` on
the host), into `tables/`, `plots/`, `stats/` and `paradata/`. `{stamp}` is the start time, `YYYYMMDD-HHMM`; `{model}`
is the local model folder name, such as `model_v44`; `{n}` is the number of pages scored.

| File                                                                    | Written by                         | Columns or content                                                                                                                                              |
|-------------------------------------------------------------------------|------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `tables/{stamp}_{model}_TOP-{N}.csv`                                    | a directory run                    | `FILE, PAGE, CLASS-1…N, SCORE-1…N`. At `-tn 1` a `CATEGORY` column repeats `CLASS-1`; `CLASS-1` and `SCORE-1` are kept, because the record needs the confidence |
| `tables/{stamp}_{model}_RAW.csv`                                        | a directory run with `--raw`       | `FILE, PAGE` and the eleven category columns, in the canonical order                                                                                            |
| `tables/{YYYYMMDD}_{model}_TOP-{N}.csv`, `…_RAW.csv`                    | a directory run with `--chunk`     | the same, one file per day, appended chunk by chunk and sorted at the end                                                                                       |
| `tables/{stamp}_{n}_{model}_TOP-{N}_EVAL.csv`                           | `--eval`                           | as Top-N, plus `TRUE`, the gold label                                                                                                                           |
| `tables/{stamp}_{n}_{model}_EVAL_RAW.csv`                               | `--eval --raw`                     | the eleven scores, plus `TRUE`                                                                                                                                  |
| `plots/{stamp}_{n}_{model}_conf_mat_TOP-{N}.png`                        | `--eval`                           | the confusion matrix                                                                                                                                            |
| `tables/{stamp}_BEST_5_models_TOP-1.csv`                                | `--best`                           | `FILE, PAGE, CLASS-1-v1.4 … CLASS-1-v5.4`: each model's top-1 vote                                                                                              |
| `tables/{stamp}_BEST_5_models_AVG_TOP-{N}.csv`                          | `--best` (not `--no-average-best`) | `FILE, PAGE, V1.4 … V5.4, CLASS-1, SCORE-1, …`: the votes, then the probability-averaged Top-N                                                                  |
| `tables/{stamp}_v14_TOP-{N}.csv` … `_v54_…`                             | `--best --save-intermediates`      | each model's own Top-N table                                                                                                                                    |
| `tables/{stamp}_{n}_{model}_TEST.csv`, `…_TEST_RAW.csv`                 | `--train --folds_csv`              | the predictions on the fold's test pages                                                                                                                        |
| `tables/{stamp}_{n}_{model}_TEST_FOLD_{i}.csv`                          | `--train --folds N`                | the predictions on each fold's test pages                                                                                                                       |
| `stats/{stamp}_{model}_{foldN}_DATASETS.txt`, `…_FOLD_{i}_DATASETS.txt` | training                           | the train, dev and test split actually used                                                                                                                     |
| `model/gpu_profile.json`                                                | `--best --parallel`                | the measured memory profile used to group the models ([how it works](../how-it-works.md#the-memory-aware---best-engine))                                        |
| `paradata/YYMMDD-HHmmss_page-classification.json`                       | every run                          | the run's paradata ([paradata.md](paradata.md))                                                                                                                 |
| `<doc_id>.document.json`                                                | with `--document-json-out(-dir)`   | the ATRIUM document record ([below](#the-document-record))                                                                                                      |

Top-N tables are sorted by `FILE` and `PAGE`, in ascending order. A directory run's `RAW` table is sorted by the
category columns, highest score first, so its rows come grouped by category.

## The Top-N table

- **FILE**: the name of the file (the document), the page image's name without its page number.
- **PAGE**: the number of the page.
- **CLASS-N**: the label of the category, guess N.
- **SCORE-N**: the score of the category, guess N.

and optionally

- **TRUE**: the actual label of the category (evaluation tables).

All scores are normalised: the model's probabilities renormalised over its best max(N, 2) categories and rounded to
three decimals, so with `-tn 1` the score is the best category's share of the best two, as the service's `topn=1`
gives. An image that cannot be read gets no row: it is recorded as skipped in the run's paradata, and every other
page keeps its own result (#61). How `FILE` and `PAGE` are split from the file name:
[../usage.md](../usage.md#name-the-page-images).

## The raw table

- **FILE**: the name of the file.
- **PAGE**: the number of the page.
- **\<CATEGORY_LABEL\>**: one column for each of the eleven categories, with its probability.
- **TRUE**: the actual label of the category (evaluation tables).

The reason to use `--raw` is convenience in reviewing the results: the rows are sorted by the category scores, so
they come grouped by category, and the most ambiguous pages have several small probabilities where the clear ones
have zeros.

## The ensemble tables

**Wide per-model votes**, `{stamp}_BEST_5_models_TOP-1.csv`:

- **FILE**: the name of the file (document).
- **PAGE**: the number of the page.
- **CLASS-1-vM.4**: the top-1 class predicted by model `vM.4` (`M` = 1…5), one column per model.

**Averaged result**, `{stamp}_BEST_5_models_AVG_TOP-{N}.csv`, the same columns as the example
[ARUP_averaged_SHORT.csv](../../result/ARUP_averaged_SHORT.csv):

- **FILE**: the name of the file (document).
- **PAGE**: the number of the page.
- **VM.4**: the top-1 vote of model `vM.4` (`M` = 1…5), in the canonical model order.
- **CLASS-K**: the `K`-th class (`K` = 1…`N`) of the probability-averaged ensemble.
- **SCORE-K**: the averaged probability of `CLASS-K`, from 0 to 1; empty when it is zero, and the paired `CLASS-K` is
  then blank too, so a class never appears with a blank score.

The vote columns are always top-1; the averaged part keeps `-tn` classes per page. How the scores are averaged:
[../how-it-works.md](../how-it-works.md#one-model-or-five).

## The document record

With `--document-json-out` (one file) or `--document-json-out-dir` (one `<doc_id>.document.json` per document), the
run writes or updates an ATRIUM document record. page-classification writes only its own parts:

- the whole `page_categories` block, which maps each page to its top-1 label;
- each page's `pages[].category` and `pages[].category_confidence`, the label and its `SCORE-1`.

Every other block passes through unchanged. The record's schema and its freeze: [paradata.md](paradata.md#the-document-record-schema);
the contract: the hub's [document contract](https://ufal.github.io/atrium-project/ecosystem/document-contract/).
`atrium_rocrate.py`, vendored from the hub, exports finished records as RO-Crate 1.2 (`ro-crate-metadata.json`,
Process Run Crate 0.5) as a separate step with its own command line (`--document`, `--run`, `--paradata`,
`--out-dir`, `--selftest`); see the hub's [RO-Crate export](https://ufal.github.io/atrium-project/contracts/rocrate/).

## Example tables in this repository

What each stored table measures, and what it does not, is in [result/README.md](../../result/README.md).

**The `vX.4` models on their folds' held-out test pages** (2026-10-22):

| Model  | Fold | Top-1                                                                         | Top-3                                                                         | Raw                                                                       |
|--------|------|-------------------------------------------------------------------------------|-------------------------------------------------------------------------------|---------------------------------------------------------------------------|
| `v1.4` | 1    | [TOP-1_EVAL](../../result/tables/20261022-0911_4795_model_v14_TOP-1_EVAL.csv) | [TOP-3_EVAL](../../result/tables/20261022-0924_4795_model_v14_TOP-3_EVAL.csv) | [EVAL_RAW](../../result/tables/20261022-0924_4795_model_v14_EVAL_RAW.csv) |
| `v4.4` | 1    | [TOP-1_EVAL](../../result/tables/20261022-0938_4795_model_v44_TOP-1_EVAL.csv) | [TOP-3_EVAL](../../result/tables/20261022-0951_4795_model_v44_TOP-3_EVAL.csv) | [EVAL_RAW](../../result/tables/20261022-0951_4795_model_v44_EVAL_RAW.csv) |
| `v3.4` | 2    | [TOP-1_EVAL](../../result/tables/20261022-1007_4792_model_v34_TOP-1_EVAL.csv) | [TOP-3_EVAL](../../result/tables/20261022-1021_4792_model_v34_TOP-3_EVAL.csv) | [EVAL_RAW](../../result/tables/20261022-1021_4792_model_v34_EVAL_RAW.csv) |
| `v5.4` | 2    | [TOP-1_EVAL](../../result/tables/20261022-1036_4792_model_v54_TOP-1_EVAL.csv) | [TOP-3_EVAL](../../result/tables/20261022-1053_4792_model_v54_TOP-3_EVAL.csv) | [EVAL_RAW](../../result/tables/20261022-1053_4792_model_v54_EVAL_RAW.csv) |
| `v2.4` | 5    | [TOP-1_EVAL](../../result/tables/20261022-1112_4791_model_v24_TOP-1_EVAL.csv) | [TOP-3_EVAL](../../result/tables/20261022-1126_4791_model_v24_TOP-3_EVAL.csv) | [EVAL_RAW](../../result/tables/20261022-1126_4791_model_v24_EVAL_RAW.csv) |

**The same five models on the 5,449-page set**, most of whose pages are training pages, kept for the report by group in
[result/README.md](../../result/README.md#the-5449-page-set), not as an accuracy:
[`v1.4`](../../result/tables/20261022-1140_5449_model_v14_TOP-1_EVAL.csv),
[`v2.4`](../../result/tables/20261022-1147_5449_model_v24_TOP-1_EVAL.csv),
[`v3.4`](../../result/tables/20261022-1154_5449_model_v34_TOP-1_EVAL.csv),
[`v4.4`](../../result/tables/20261022-1202_5449_model_v44_TOP-1_EVAL.csv),
[`v5.4`](../../result/tables/20261022-1208_5449_model_v54_TOP-1_EVAL.csv).

**The ensemble on the sample pages** ([small_data_samples](../../small_data_samples), no `TRUE` column), a demonstration
of the format: the votes [BEST_5_models_TOP-1.csv](../../result/tables/20261022-1218_BEST_5_models_TOP-1.csv) and the
averaged result [BEST_5_models_AVG_TOP-3.csv](../../result/tables/20261022-1218_BEST_5_models_AVG_TOP-3.csv), with each
member's own table, such as [v44_TOP-3.csv](../../result/tables/20261022-1218_v44_TOP-3.csv).

**Averaged results on two unannotated collections:** [ARUB_averaged_SHORT.csv](../../result/ARUB_averaged_SHORT.csv)
and [ARUP_averaged_SHORT.csv](../../result/ARUP_averaged_SHORT.csv), each from the `vX.3` best five models with a
Top-3 setting; about one thousandth of each collection's results is shared, for demonstration.

**The `vX.2` demonstration files of 2025-07.** Their evaluation set's overlap with training was never checked, so they
are format examples, not accuracy. "Checked" tables are on the manually checked evaluation set; "unchecked" ones are
small runs with `TRUE` values.

| Model  | Checked, TOP-1                                                                | Checked, TOP-3                                                                | Checked, RAW                                                              | Unchecked, TOP-1                                                    | Unchecked, RAW                                                  |
|--------|-------------------------------------------------------------------------------|-------------------------------------------------------------------------------|---------------------------------------------------------------------------|---------------------------------------------------------------------|-----------------------------------------------------------------|
| `v2.2` | [TOP-1_EVAL](../../result/tables/20250701-1057_model_v220105p_TOP-1_EVAL.csv) | [TOP-3_EVAL](../../result/tables/20250710-1925_model_v220105p_TOP-3_EVAL.csv) | [EVAL_RAW](../../result/tables/20250710-1925_model_v220105p_EVAL_RAW.csv) | [TOP-1](../../result/tables/20250710-1939_model_v220105p_TOP-1.csv) | [RAW](../../result/tables/20250710-1939_model_v220105p_RAW.csv) |
| `v3.2` | [TOP-1_EVAL](../../result/tables/20250701-1057_model_v320105p_TOP-1_EVAL.csv) | [TOP-3_EVAL](../../result/tables/20250710-1927_model_v320105p_TOP-3_EVAL.csv) | [EVAL_RAW](../../result/tables/20250710-1927_model_v320105p_EVAL_RAW.csv) | [TOP-1](../../result/tables/20250710-1936_model_v320105p_TOP-1.csv) | [RAW](../../result/tables/20250710-1936_model_v320105p_RAW.csv) |
| `v5.2` | [TOP-1_EVAL](../../result/tables/20250701-1057_model_v520105p_TOP-1_EVAL.csv) | [TOP-3_EVAL](../../result/tables/20250710-1928_model_v520105p_TOP-3_EVAL.csv) | [EVAL_RAW](../../result/tables/20250710-1928_model_v520105p_EVAL_RAW.csv) | [TOP-1](../../result/tables/20250710-1938_model_v520105p_TOP-1.csv) | [RAW](../../result/tables/20250710-1938_model_v520105p_RAW.csv) |
| `v1.2` | [TOP-1_EVAL](../../result/tables/20250709-1825_model_v120106s_TOP-1_EVAL.csv) | [TOP-3_EVAL](../../result/tables/20250710-1924_model_v120106s_TOP-3_EVAL.csv) | [EVAL_RAW](../../result/tables/20250710-1924_model_v120106s_EVAL_RAW.csv) | [TOP-1](../../result/tables/20250710-1941_model_v120106s_TOP-1.csv) | [RAW](../../result/tables/20250710-1941_model_v120106s_RAW.csv) |
| `v4.2` | [TOP-1_EVAL](../../result/tables/20250709-1823_model_v120106l_TOP-1_EVAL.csv) | [TOP-3_EVAL](../../result/tables/20250710-1921_model_v120106l_TOP-3_EVAL.csv) | [EVAL_RAW](../../result/tables/20250710-1921_model_v120106l_EVAL_RAW.csv) | [TOP-1](../../result/tables/20250710-1942_model_v120106l_TOP-1.csv) | [RAW](../../result/tables/20250710-1942_model_v120106l_RAW.csv) |

The `v4.2` files carry the internal name `model_v120106l` and the `v1.2` files `model_v120106s`.
