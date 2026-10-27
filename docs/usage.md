# Using page-classification

For anyone who wants to classify their own page images: how to install the tool, get a model, classify one page or a
whole folder, read and combine the results, and run it from the Docker images. Deploying and operating the HTTP
service, with every endpoint, limit and error, is in [../service/README.md](../service/README.md); every flag is in
[reference/cli.md](reference/cli.md).

On this page:
[Requirements](#requirements) ·
[Install](#install) ·
[Get a model](#get-a-model) ·
[Prepare the input](#prepare-the-input) ·
[Classify one page](#classify-one-page) ·
[Classify a folder](#classify-a-folder) ·
[Reading the output](#reading-the-output) ·
[Combining and post-processing results](#combining-and-post-processing-results) ·
[Write a document record](#write-a-document-record) ·
[Docker](#docker) ·
[The HTTP service](#the-http-service) ·
[In the ATRIUM pipeline and from a coding agent](#in-the-atrium-pipeline-and-from-a-coding-agent) ·
[Troubleshooting](#troubleshooting)

## Requirements

- **Python 3.11**, the version the container images and CI use across the ATRIUM repositories. `numpy` is pinned
  below 2.5, because 2.5 requires Python 3.12.
- **A machine.** The minimum is a CPU with a decent (above average) amount of memory: prediction is slow, training
  and evaluation very slow. For fast prediction, and relatively fast training and evaluation, add an Nvidia GPU with
  [CUDA](https://developer.nvidia.com/cuda-python) support.
- **Disk:** up to 1 GB for the model files and checkpoints, and up to 7 GB for the Python libraries (PyTorch and its
  dependencies).

The commands below are written for a Unix shell; on Windows use `python` instead of `python3`.

## Install

```bash
git clone https://github.com/ufal/atrium-page-classification.git
cd atrium-page-classification
python3 -m venv .venv && source .venv/bin/activate      # how to: https://docs.python.org/3/library/venv.html
pip install -r setup/requirements.txt                    # takes a while
python3 run.py -h                                        # a description of every flag
```

There is no `pip install .`: the repository ships no `pyproject.toml` and no console script, and every command is
`python3 run.py` from the repository root, with the virtual environment active. PyTorch's CUDA support comes with
the requirements; whether a GPU is present is checked on first use and again before each model is initialised (for
training, evaluation or prediction).

> ⚠️ **`transformers` stays on 4.x.** On `transformers` 5.x, model construction moves to the `meta` device inside
> `from_pretrained`, and the timm-based architectures used here (RegNetY `v4.3`/`v4.4`, EfficientNetV2 `v1.3`/`v1.4`)
> run shape-deriving operations (`torch.unique`, `.item()`) while they are built. These have no meta kernel and raise
> an error. `setup/requirements.txt` therefore pins `transformers>=4.57.6,<5`, and `.github/dependabot.yml` ignores
> the 5.x bump.

The HTTP service has its own setup script, `setup/setup_api_service.sh`, which also installs
`service/requirements.txt` and downloads the weights:
[../service/README.md](../service/README.md#installation--setup).

### Branches

The code is on the default branch, `vit`; changes are staged on `test` ([CONTRIBUTING.md](../CONTRIBUTING.md)).
`clip` holds a separate, CLIP-based classifier for the same categories, with its own Hugging Face repository; switch
to it with `git checkout clip`. The other branches: [development.md](development.md#branches).

### Updating a checkout

To keep your local changes and merge the latest files over them:

```bash
cd /local/folder/for/this/project/atrium-page-classification
git add <changed_file>
git commit -m 'local changes'
git pull -X theirs
```

To update one branch (`vit` or `clip`) by fast-forward:

```bash
git fetch origin
git checkout vit
git pull --ff-only origin vit
```

To drop every local change and take the branch as it is on GitHub:

```bash
git reset --hard HEAD
git clean -fd
git fetch origin
git checkout vit
git reset --hard origin/vit
```

## Get a model

The fine-tuned models are revisions of [`ufal/vit-historical-page`](https://huggingface.co/ufal/vit-historical-page)
on the Hugging Face Hub. Download one with `--hf` and name it with `-rev`:

```bash
python3 run.py --hf -rev v4.4         # saved to model/model_v44
```

You see a message about loading the model from the Hub, and then about saving it locally. Downloading takes less
time than installing the dependencies. Once the files are in `model/model_<revision without dots>`, later runs load
them from there and `--hf` is needed only to update the model.

> ⚠️ **Name the revision.** The Hub's `main` revision still resolves to `v4.3`, the generation before `vX.4`, and
> `--hf` without `-rev` takes `[HF] revision`, which is `main`. Pass `-rev v4.4` (or another `vX.4`) until `main` is
> moved. Which revisions exist: [reference/models.md](reference/models.md).

A revision builds on the base model that [model_registry.py](../model_registry.py) lists for it (for example `v5.4`
on `google/vit-large-patch16-384`), whatever `--base` or the config says, because the image transforms are made from
the base. `--base` matters only for a revision the registry does not list. The command line refuses a revision no
registry key matches, among them `main` given as `-rev` and the early `v1.2` and `v2.0`–`v2.2`
([reference/cli.md](reference/cli.md#model-selection)).

`--best` runs the five `vX.4` models from `model/model_v14` … `model/model_v54`, so download all five first:

```bash
for r in v1.4 v2.4 v3.4 v4.4 v5.4; do python3 run.py --hf -rev "$r"; done
```

### Stage the models once, run offline

On a shared server (a Galaxy instance, a cluster) the weights should be downloaded once, not in every job.
`--models-dir DIR` puts everything a run reads in one folder: the fine-tuned models in `DIR/model`, the base-model
cache in `DIR/checkpoint` and the Hugging Face cache of the image processors in `DIR/hf`. With `--hf` the revision is
downloaded into it; without `--hf` the run reads only that folder and switches the Hugging Face libraries offline
(`HF_HUB_OFFLINE=1`), so a missing file stops the run instead of starting a download.

```bash
# once, with network: the five vX.4 models (about 2.5 GB)
for r in v1.4 v2.4 v3.4 v4.4 v5.4; do python3 run.py --hf -rev "$r" --models-dir /srv/atrium-page-models; done

# every run, offline: one model, or the five-model ensemble
python3 run.py --models-dir /srv/atrium-page-models -rev v4.4 -d pages/
python3 run.py --models-dir /srv/atrium-page-models --best -d pages/
```

The results still go to `result/` under the working directory. The staged folder can be read-only; a `--best` run on
a GPU then re-measures its memory profile each time instead of saving it.

After a download the checkout looks like this (`model/`, `result/` and `checkpoint/` are git-ignored runtime folders,
created when the pipeline first runs; under Docker they are volumes, so data and models persist between runs;
`model_output/` appears only after training):

```text
atrium-page-classification
├── model
│   └── model_<revision>
│       ├── config.json
│       ├── model.safetensors
│       └── preprocessor_config.json
├── checkpoint
│   ├── models--google--vit-base-patch16-224
│   │   ├── blobs
│   │   ├── snapshots
│   │   └── refs
│   └── .locs
│       └── models--google--vit-base-patch16-224
├── data_scripts
│   ├── windows
│   │   ├── move_single.bat
│   │   ├── pdf2png.bat
│   │   └── sort.bat
│   └── unix
│       ├── move_single.sh
│       ├── pdf2png.sh
│       └── sort.sh
├── result
│   ├── plots
│   │   ├── date-time_<#samples>_model_<revision>_conf_mat_TOP-<top_N>.png
│   │   └── ...
│   ├── tables
│   │   ├── date-time_model_<revision>_TOP-<top_N>.csv
│   │   ├── date-time_model_<revision>_RAW.csv
│   │   ├── date-time_<#samples>_model_<revision>_TOP-<top_N>_EVAL.csv
│   │   ├── date-time_<#samples>_model_<revision>_EVAL_RAW.csv
│   │   ├── date-time_BEST_<#models>_models_TOP-<top_N>.csv
│   │   └── ...
│   └── stats
│       ├── date-time_model_<revision>_FOLD_<n>_DATASETS.txt
│       └── ...
├── setup
│   ├── config.txt
│   ├── para_config.txt
│   ├── requirements.txt
│   ├── requirements-test.txt
│   └── setup_api_service.sh
├── small_data_samples
│   ├── DRAW
│   ├── DRAW_L
│   └── ...
├── supplementary
│   ├── figures
│   └── scripts
├── run.py
├── classifier.py
├── utils.py
├── README.md
└── ...
```

Every output file and its columns: [reference/outputs.md](reference/outputs.md).

## Prepare the input

The classifier reads **page images**, one file per page, and **PDFs**. A PDF given with `-f`, or found by `-d` when
`-ff` names `pdf`, is rendered page by page with the service's renderer (pypdfium2, at `PDF_RENDER_DPI`, 300 dpi),
so the command line and `POST /predict_document` see the same pixels; `--pages 1,3,5-7` classifies a selection.
A file is taken as a PDF by its content, whatever its name. Splitting the PDFs into files first is still possible,
with the bundled converter, which renders every page at 300 dpi with poppler's `pdftoppm`, in parallel, into one
sub-folder per PDF:

```bash
data_scripts/unix/pdf2png.sh --dir /data/pdfs --output /data/pages      # the PDFs are kept
data_scripts/unix/pdf2png.sh --dir /data/pdfs --format jpg --dpi 200    # other options
```

`data_scripts/windows/pdf2png.bat` does the same with ImageMagick and Ghostscript, one file at a time. Neither the
command line nor the HTTP service needs them any more for classification. The converters, and how to turn the pages
into a training set: [data-preparation.md](data-preparation.md).

### Name the page images

A page image is expected to be named `<document>-<page>.png` or `<document>_<page>.png`. The tool splits on the
*last* separator, so `report_2021_003.png` is document `report_2021`, page 3. The two halves become the `FILE` and
`PAGE` columns of every result table and the key of the document record. A name without a trailing page number is
treated as page 1, with a warning. The pages of a PDF are named for you: `<document>-<page>`, where `<document>` is
the PDF's name as `canonical_doc_id()` reads it and `<page>` its physical position, 1-based.

## Configure a run

Defaults come from [setup/config.txt](../setup/config.txt). Before a first run, set the input folder in its
`[INPUT]` section, and optionally `top_N` and `batch` in `[SETUP]`.

> 💡 **Top-3** covers most images; **Top-5** helps with a small number of pages that are hard to classify.

`batch` depends on your machine's memory: below 12 is safe for an office desktop, and 4 for an old one; on a
high-performance cluster you can go above 20. The memory each batch size needs:
[configuration.md](configuration.md#setup). Do not change `base_model` and the other sections unless you know what
you are doing. Every setting: [configuration.md](configuration.md).

## Classify one page

Pass the image with `-f`; `-tn` sets how many guesses you get, and `-m` the model folder:

```bash
python3 run.py -tn 3 -f '/full/path/to/file.png' -m '/full/path/to/model/folder'   # exactly Top-3, to the console
python3 run.py -f '/full/path/to/file.png'                                        # the config's defaults
python3 run.py -f '/full/path/to/file.png' -tn 1                                  # the single best guess
python3 run.py -f '/full/path/to/file.png' -rev v4.4                              # a named revision from model/
python3 run.py -f '/full/path/to/file.png' --best                                 # all five models, averaged
python3 run.py -f '/full/path/to/report.pdf' -rev v4.4 --pages 1-3                # a PDF: pages 1 to 3
```

The result is printed to the console, one line per category with its score as a percentage
([the Quick start](../README.md#quick-start) shows one), and written as the one-row table a folder run writes
(`result/tables/<time>_<model>_TOP-<N>.csv`; before v1.10.0-beta `-f` only printed). A PDF takes the folder path:
one row per page.

> ℹ️ The console output and every result table give **normalised** scores: the model's probabilities renormalised over
> its best max(N, 2) categories, so with `-tn 1` the score is the best category's share of the best two (as the
> service's `topn=1`). An image that cannot be read gets no row: it is recorded as skipped in the run's paradata,
> and every other page keeps its own result (#61).

## Classify a folder

`-d` names the folder; `--dir` takes the folder in `[INPUT] FOLDER_INPUT` instead. Input is chosen only by `-f`,
`-d` or `--dir`: without one of them nothing happens. `--inner` (the default) walks nested folders, `-ff` sets the
file extensions to collect, comma-separated and in any case (default `png`; `png,jpg,tif`; add `pdf` to render and
classify the PDFs too), and `--raw` also writes all eleven scores. The same `-tn` and `-m` flags apply as for one
page. A PDF is taken only when `-ff` names `pdf`, so a folder that holds a PDF beside its own page images is not
classified twice; with `--no-inner`, which takes every file of the folder, a PDF is always rendered.

```bash
python3 run.py -d /data/pages --inner                      # recurse into sub-folders
python3 run.py --dir --inner --raw                         # the config's folder, all eleven scores too
python3 run.py --dir --inner --best                        # all five models, one averaged Top-N table
python3 run.py --dir --inner --best --parallel             # the same in one pass on a CUDA GPU
python3 run.py --dir --inner --best --save-intermediates   # also each model's own Top-N table
python3 run.py --dir --inner --best --no-average-best      # only the wide per-model votes file
python3 run.py -d /data/pages --chunk                      # write in chunks of [INPUT] chunk_size images
python3 run.py -d /data/scans -ff png,jpg,pdf              # images of three kinds and every PDF's pages
```

- A folder is processed **in batches** (`[SETUP] batch`), so mind the memory each batch size needs.
- With more than about 500,000 files, even listing them across the sub-folders takes a while before any page is
  classified.
- Progress is printed every 50 batches, as `<i>-th batch  Processed <n> images in <t> min`, where `n` grows by the
  batch size (`[SETUP] batch`) with each batch. The output table is saved in `result/tables` only after every image
  is processed (with `--chunk`, chunk by chunk).
- `--parallel` changes only how the models are scheduled on the GPU: its output files are identical to those of a
  plain `--best` run, so the two can be diffed byte for byte ([how-it-works.md](how-it-works.md#the-memory-aware---best-engine)).
- `--best` honours `-tn`: the averaged table keeps `top_N` ranked classes per page, while the per-model vote columns
  are always top-1.

## Reading the output

A Top-3 run writes one row per page, best guess first; the scores are normalised probabilities, rounded to three
decimals. An example (made-up rows):

```text
FILE,PAGE,CLASS-1,CLASS-2,CLASS-3,SCORE-1,SCORE-2,SCORE-3
report_2021,1,TEXT_T,TEXT,TEXT_P,0.962,0.031,0.007
report_2021,2,LINE_T,LINE_P,TEXT_T,0.884,0.101,0.015
report_2021,3,TEXT,TEXT_T,DRAW,0.521,0.468,0.011
```

`CLASS-1` is the category to act on. A low `SCORE-1` with a close `CLASS-2`, as on page 3, marks a page worth a human
look. `--raw` adds a table with one probability column per category, convenient for review because an ambiguous page
spreads its probability over several columns. With `--best` the run also writes each model's top-1 vote, so
disagreement between the five is visible page by page. Real tables from this repository's runs are listed in
[reference/outputs.md](reference/outputs.md#example-tables-in-this-repository); the confusion matrices that `--eval`
draws are explained in [evaluation/results.md](evaluation/results.md#reading-a-confusion-matrix).

When a record is requested, the same top-1 lands in the document record: `page_categories` maps each page to its
label, and `pages[].category` and `pages[].category_confidence` carry the label and `SCORE-1`.

## Combining and post-processing results

> 💡 **The best way to classify a collection of messy files is to combine several models.** The five models give
> different perspectives on the data, and their ensemble evens out each one's errors. With `--best` the ensembling
> happens inside the program.

`--best` loads each of the five models, runs them, and writes two files to `result/tables/`:

| File                                        | Contents                                                                                                    |
|---------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| `{date-time}_BEST_5_models_TOP-1.csv`       | the wide per-model votes: one top-1 column per model (`CLASS-1-v1.4` … `CLASS-1-v5.4`)                      |
| `{date-time}_BEST_5_models_AVG_TOP-{N}.csv` | the final averaged result: the per-model votes (`V1.4` … `V5.4`) and the probability-averaged Top-N classes |

The averaging is a mean of the models' probabilities, not a majority vote, so confident models weigh more than
uncertain ones, and a page on which all five agree with certainty scores `1.0`
([how-it-works.md](how-it-works.md#one-model-or-five)).

The wide `BEST_5_models_TOP-1.csv` is byte-compatible with
[averaging.py](../supplementary/scripts/averaging.py), which stays available for manual re-averaging. It is no longer
a required step. Reach for it instead of `--best` when:

* you already have saved per-model tables and want to combine them again without re-running inference
  (`--best --save-intermediates` writes those per-model files);
* you want to ensemble another set of models or runs than the default five, or recompute with a different `top_N`.

```bash
# combine saved per-model results into a Top-3 summary
python3 supplementary/scripts/averaging.py --files "result/tables/*_TOP-3.csv" --top_n 3 --output ensemble_results.csv
# re-average a wide votes file (a majority vote when it holds no scores)
python3 supplementary/scripts/averaging.py --files result/tables/20261022-1218_BEST_5_models_TOP-1.csv -n 1
```

`--keep-zeros` keeps null scores and `--no-normalize` turns off the hyphen-to-underscore normalisation of file names;
see `python3 supplementary/scripts/averaging.py --help`. Examples shipped in the repository: the wide
[BEST_5_models_TOP-1.csv](../result/tables/20261022-1218_BEST_5_models_TOP-1.csv) and the averaged
[ARUB_averaged_SHORT.csv](../result/ARUB_averaged_SHORT.csv) and [ARUP_averaged_SHORT.csv](../result/ARUP_averaged_SHORT.csv),
each from the `vX.3` best five models with a Top-3 setting (about one thousandth of each collection's results).

Two more scripts take result tables apart:

* **Per-document tables:** split any result table into one CSV per document; the script expects the file name in
  the first column:

  ```bash
  python3 supplementary/scripts/per_doc_split.py -i result_table.csv -o /path/to/output_dir
  ```

* **Scores across runs:** compute accuracy over several saved evaluation tables in a folder:

  ```bash
  supplementary/scripts/result_analysis.sh result/tables/ --pattern "*_TOP-1_EVAL.csv"
  ```

## Write a document record

The record of the ATRIUM pipeline is opt-in: set an output path, on the command line or in `[DOCUMENT]`.

```bash
python3 run.py -rev v4.4 -f pages/doc0001-7.png --document-json-out records/doc0001.document.json
python3 run.py -rev v4.4 -d pages/ --document-json-dir baseline/ --document-json-out-dir records/
```

`--document-json` (or `--document-json-dir` for a folder) names a record to add to, written by an earlier stage;
without one, page-classification starts the record. `--strict-document-json` turns ownership and schema warnings into
errors. What the tool writes into the record: [reference/outputs.md](reference/outputs.md#the-document-record).

- **A born-digital record names its pages by their PDF page labels** (`i`, `ii`, `1`, …) and keeps the position in
  `pages[].page_index`. The pages of a PDF are written under those labels, as `/predict_document` writes them; a page
  whose position is another page's label is left out of the record (it stays in the table).
- `-f` with `--document-json-out-dir` writes `<doc_id>.document.json` into that directory (taking the baseline from
  `--document-json-dir`, as a folder run does); `-d` with `--document-json-out` is refused, as one path cannot hold a
  folder's records (it used to be ignored).

## Docker

Two images are published to GHCR from the same [Dockerfile](../Dockerfile). `<version>` is the release without its
leading `v` (for example `1.10.0-beta`), or `latest`; when each tag moves is on the hub's
[Operations](https://ufal.github.io/atrium-project/operations/) page.

| Image                                                   | Stage  | Entry point                                       | What it is                    |
|---------------------------------------------------------|--------|---------------------------------------------------|-------------------------------|
| `ghcr.io/ufal/atrium-page-classification:<version>`     | `base` | `python3 /app/entrypoint.py`, which runs `run.py` | the batch command line        |
| `ghcr.io/ufal/atrium-page-classification-api:<version>` | `api`  | `python -m service.api`                           | the HTTP service on port 8000 |

The `-api` suffix belongs to the image name. An image built locally with Compose carries its own local tag
(`${ATRIUM_VERSION:-dev}`). Neither image holds model weights.

The batch image passes options to `run.py` (`docker run IMAGE -d /data/input --best`). Since `v1.10.4-beta`, a first
argument that names a program runs that program instead (`docker run IMAGE /bin/sh job.sh`,
`docker run IMAGE python3 /app/run.py --help`), which is how Galaxy starts a tool's container; before, it reached
`run.py` and failed with "unrecognized arguments" (atrium-project#66).

```bash
# the batch command line, over ./data/input, writing to ./data/output
docker compose run --rm classify

# the HTTP service
docker compose --profile api up --build      # then http://localhost:8000/info
```

> ⚠️ The batch image's default command is `-d /data/input --hf -rev v4.3`, the generation before `vX.4`. Pass your own
> arguments to choose the revision, for example `docker compose run --rm classify -d /data/input --hf -rev v4.4`.

**On a GPU,** overlay the second compose file and build against the CUDA wheel index:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm classify
#   build with: --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126
```

**Volumes:** `./data:/data` (the inputs, from `./data/input`), `./data/output:/app/result` (the results and the
paradata), and the named volumes `page-model:/app/model` and `hf-cache:/cache/huggingface`. The last two keep the
downloaded weights, so a container start does not download them again.

**Run as yourself on Linux.** `./data` and `./data/output` are part of the clone and belong to you, while the images
run as the non-root user `atrium`, uid 10001. Compose runs the services as `user: "${ATRIUM_UID:-10001}:0"`, so put
your uid in `.env` once, `echo "ATRIUM_UID=$(id -u)" >> .env`, and the container writes the results as you. With
`docker run`, pass `--user "$(id -u):0"`. Docker Desktop (macOS, Windows) needs neither (atrium-project#69). A host
folder the container's uid cannot write fails with `EACCES` before any page is classified.

**`HF_TOKEN`** is optional: the models are public, but anonymous Hugging Face downloads are rate-limited, so an
automated run should pass one (`-e HF_TOKEN`). It is read from the environment, never from `setup/config.txt`.

The service image is hardened for deployment (upload size limits, a PDF page cap) and stops cleanly on `SIGTERM`; a
clean stop exits with **143**, which is expected. The operator runbook (start, models and caches, errors and what to
do, limits, shutdown): [../service/README.md](../service/README.md).

## The HTTP service

The service classifies one image (`POST /predict_image`) or a PDF (`POST /predict_document`) and answers in JSON,
with the same models. A minimal call is in the [Quick start](../README.md#quick-start):

```bash
curl -F file=@page.png -F version=v4.4 -F topn=3 http://localhost:8000/predict_image
curl -F file=@report.pdf -F version=all http://localhost:8000/predict_document
curl -F file=@page.png -F document_json=@record.json http://localhost:8000/predict_image   # add to a record
```

### The command line and the service

They read the same inputs the same way, and write the same record:

| What                             | Command line (`run.py`)                            | Service                                                                              |
|----------------------------------|----------------------------------------------------|--------------------------------------------------------------------------------------|
| a page image                     | `-f page.png`; any format Pillow reads             | `POST /predict_image`; recognised by content when sent as `application/octet-stream` |
| a PDF, a page selection          | `-f report.pdf --pages 1-3`; `-d` with `-ff …,pdf` | `POST /predict_document`, `pages=1-3`                                                |
| a record to add to, a new record | `--document-json`, `--document-json-out`           | `document_json`, `document_json_out=true`                                            |
| each model's own top-N           | `--best --save-intermediates` (one CSV per model)  | `per_model=true`                                                                     |
| all eleven scores                | `--raw`                                            | `topn=11`                                                                            |

What only one side does, on purpose: the command line takes folders, trains and evaluates models, runs YOLO and
writes CSV tables and a paradata file; the service takes one file per request, refuses uploads over its limits
(`MAX_UPLOAD_MB`, `MAX_PDF_PAGES`, `MAX_IMAGE_PIXELS`) and returns its paradata as a `CreateAction`.

Kept as they are, documented in [how-it-works.md](how-it-works.md#one-model-or-five): the two sides average the five
models differently, the tables round scores to three decimals (the service does not), the default model differs
(the CLI's config `main` and the batch image's `v4.3`, the service's `all`), and a transparent image is put on white
by the command line but not by the service.

`service/api_client.py` wraps the same calls and prints the JSON answer
(`python3 service/api_client.py -f page.png -v v4.4 --top 3`), and a static page for trying it in a browser is served
at `/frontend`. The request fields, the response, the limits and every error are in
[../service/README.md](../service/README.md); the typed contract is [../service/openapi.json](../service/openapi.json).

## In the ATRIUM pipeline and from a coding agent

Each pipeline stage runs as its own container, and the document record passed with `--document-json` /
`--document-json-out` is the hand-off between them. The reference sequence, with page-classification as its first
stage, is on the hub's [workflow page](https://ufal.github.io/atrium-project/workflows/page-classification/) and its
[Pipelines](https://ufal.github.io/atrium-project/pipelines/) page. The `agent-skill` branch packages the HTTP service
as an Agent Skill, so a coding agent can classify a page or a PDF by calling a running service; installing it, and
the contract it relies on, are on the hub's [Agent skills](https://ufal.github.io/atrium-project/agent-skills/) page.

## Troubleshooting

| Symptom                                                               | Cause                                                                                         | Fix                                                                             |
|-----------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| `Revision … is not supported`                                         | `-rev` names a revision `model_registry.py` does not list (`main`, `v1.2`, `v2.0`–`v2.2`)     | name a listed revision, such as `v4.4`; leave `-rev` out to use `[HF] revision` |
| `No valid image files found to process … Exiting.`                    | no `-f`, `-d` or `--dir`, or no file with the `-ff` extension in the folder                   | pass the input; check `-ff` (default `png`) and `--inner`                       |
| `pip install` aborts with "No matching distribution found" for numpy  | numpy 2.5 or later requires Python 3.12                                                       | stay on 3.11 and keep the `<2.5` pin; `dependabot.yml` already ignores the bump |
| a crash on the meta device when a timm base model loads               | `transformers` 5.x has no meta kernel for these bases                                         | keep `transformers` below 5.0; `dependabot.yml` ignores that bump too           |
| the service's first call for a revision is slow                       | only the ensemble's revisions `v1.4`–`v5.4` are loaded at start; others download on first use | ask for one of `v1.4`–`v5.4`, or accept one slow first call                     |
| the container reports healthy, but nothing can reach it               | `HOST=127.0.0.1` binds inside the container only                                              | leave `HOST` unset (it defaults to `0.0.0.0`)                                   |
| every browser request is blocked, with no error in the log            | `ALLOWED_ORIGINS` set to the empty string means "no origins"; leaving it out means `*`        | set it explicitly, or remove the line                                           |
| the browser works from `localhost:8080` but not from `127.0.0.1:8080` | they are different origins to a browser                                                       | list both, as the compose default does                                          |
| `EACCES` writing results under Docker                                 | the host folder is not writable by uid 10001                                                  | set `ATRIUM_UID` as above, or `chown 10001` the mounted folder                  |
| the container exits with 143                                          | a clean `SIGTERM` drain                                                                       | expected; not a failure                                                         |
