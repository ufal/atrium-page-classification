# ATRIUM page-classification: scanned archive pages sorted into eleven page types

page-classification reads the image of a scanned archival page and says what kind of page it is. It chooses one of
eleven categories, such as running text, a table or form, a drawing or a photograph, with handwritten, printed or
typed text. The category describes the page for review, search and later analysis; in the AMČR pipeline it is
recorded with the page, not used to route it. Use it on scans of paper documents; the labels it gives born-digital
pages have not been checked.

[Releases](https://github.com/ufal/atrium-page-classification/releases) · [MIT licence](LICENSE) ·
[ATRIUM documentation](https://ufal.github.io/atrium-project/workflows/page-classification/) ·
[Models on Hugging Face](https://huggingface.co/ufal/vit-historical-page)

## Inputs and outputs

| Input                                                                                                                 | Output                                                                                                                                                                                                       |
|-----------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| a page image (any format Pillow reads: PNG, JPEG, TIFF, …), a folder or a tree of them, named `<document>-<page>.png` | per page, the best categories with their scores: a CSV (`FILE, PAGE, CLASS-1…N, SCORE-1…N`; at `-tn 1` also `CATEGORY`, a copy of `CLASS-1`: [outputs](docs/reference/outputs.md)), or JSON from the service |
| a PDF, recognised by its content: every page, or a `pages` selection, rendered at 300 dpi                             | optionally all eleven scores (`--raw`, or `topn=11`) and each model's own vote (`--best --save-intermediates`, or `per_model=true`)                                                                          |
| optionally, an ATRIUM document record to add to                                                                       | the record with its `page_categories` block and each page's `category` and `category_confidence`; the run's paradata                                                                                         |

The tool reads the image only, so the language of the page does not matter. The command line and the service read the
same inputs the same way; what differs, and why, is in
[docs/usage.md](docs/usage.md#the-command-line-and-the-service).

The eleven categories and their definitions: [docs/reference/categories.md](docs/reference/categories.md). The
record follows the hub's [document contract](https://ufal.github.io/atrium-project/ecosystem/document-contract/)
(record schema 2.0): page-classification adds its categories to the page rows that ocr-postprocess, the record
builder, wrote, keyed by `pages[].page_index`; run on its own, it starts a record holding only its own blocks. The
run's paradata, what it read and wrote, come with every answer.

## Quick start

**What you need:** Python 3.11 and a checkout of this repository, or Docker. Network access to `huggingface.co`, once,
to download the models (about 0.3 GB for one revision, 2.44 GB for the five of the ensemble); the tool calls no other
host. A GPU is optional: a CPU works, more slowly.

Classify a sample page of this repository with one model, `v4.4`. The run writes into the folder it runs in, so start
it from a scratch folder:

```bash
git clone https://github.com/ufal/atrium-page-classification.git && cd atrium-page-classification
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r setup/requirements.txt
mkdir -p out && cd out
python3 ../run.py --hf -rev v4.4 --models-dir models -f ../small_data_samples/TEXT_T/pages_online_2.png
```

After the argument list and the download messages it prints (the last digit can differ between machines):

```text
File ../small_data_samples/TEXT_T/pages_online_2.png predicted:
	TEXT_T:  90.0%
	TEXT:  10.0%
	TEXT_P:  0.1%
Results for TOP-3 predictions are recorded into result/tables/<date-time>_model_v44_TOP-3.csv
```

The table and the run's paradata are in `out/result/`, and the model in `out/models/`.

The same page through the service, from the published image. Start it in the background, wait until its five
models are loaded, call it, and stop it:

```bash
docker run -d --name atrium-page-classification -p 8000:8000 -v hf-cache:/cache/huggingface \
  ghcr.io/ufal/atrium-page-classification-api:<version>     # 2.0.0-beta or later, without its v
until curl -fsS localhost:8000/ready >/dev/null; do sleep 5; done
curl --fail-with-body -sS -X POST localhost:8000/predict_image \
  -F "file=@small_data_samples/TEXT_T/pages_online_2.png" -F version=v4.4 -F topn=3
docker rm -f atrium-page-classification
```

Expected result (abridged):

```json
{"type": "image",
 "predictions": [{"label": "TEXT_T", "score": 0.8996}, {"label": "TEXT", "score": 0.0996}, {"label": "TEXT_P", "score": 0.0009}],
 "models_used": ["v4.4"], "per_model": null, "limits_applied": [], "document_json": null,
 "paradata": {"@type": "CreateAction", "…": "…"}}
```

The service answers its scores unrounded; they are shown here to four decimals. Without `version` it uses the
ensemble (`all`), and `models_used` lists the five revisions that answered. The image holds no weights: on its first
start it downloads the five models into the `hf-cache` volume. `/health` answers at once, and `/ready` returns 200 once
the models are loaded. Every endpoint, limit and error: [service/README.md](service/README.md). Releases before
2.0.0-beta key a PDF's pages by `page` instead of `page_index` and are described by the README at their tag.

## How it works

The classifiers are fine-tuned Vision Transformer, RegNetY and EfficientNetV2 models, published as revisions of
[`ufal/vit-historical-page`](https://huggingface.co/ufal/vit-historical-page). They were trained on annotated pages of
archaeological reports from Czech archives, 1920–2020, published as a dataset on LINDAT. One revision can run alone,
or the five `vX.4` revisions run as an ensemble that averages their scores; the ensemble is the default of the HTTP
service.

Each page image is resized to the model's square input (224 or 384 pixels) and scored for all eleven categories;
the scores become probabilities, and the best ones are kept, best first. With `--best` (or `version=all`) five
models score every page and their probabilities are averaged, so a page on which they disagree gets a lower top
score. The command line writes CSV tables and a paradata log; the service answers in JSON. More, with the diagrams:
[docs/how-it-works.md](docs/how-it-works.md).

## Where it fits in ATRIUM

In the AMČR pipeline, page-classification classifies the pages that go to the ATR, AIS CR's text-recognition
service, beside the ATR, and the pages that show pictures: the pages forced to the ATR and the pages with `figure`
blocks (pictures, charts, drawings) in the docproc, of a PDF or of an office document, or every page of a scanned
image (atrium-project#84 decision 4). For an office document it reads digital-convert's rendered PDF, whose pages
the docproc numbers, and names it in its paradata as that distribution, not stored (`distribution`). The pipeline
selects the pages (the hub's `atrium_page_selection`) and passes them as `pages` with the rule
(`page_selection=forced+figures`); the paradata record both, so a page without a category reads as not selected.
The categories join the record's page rows after ocr-postprocess builds them. They may be offered to the ATR as a weak signal about the content, never as routing. The whole route:
the hub's [Pipelines](https://ufal.github.io/atrium-project/pipelines/#the-route-through-the-pipeline) page and this
tool's [workflow page](https://ufal.github.io/atrium-project/workflows/page-classification/).

## Limits and maturity

- **Accuracy is measured for single models, not for the ensemble.** Each `vX.4` model scored 94.64–95.87 % top-1 and
  99.75–99.90 % top-3 on about 4,800 held-out pages of the fold it was trained on. The split is by page, so about a
  quarter of those pages come from documents with other pages in training; on pages of unseen documents top-1 is
  93.87–95.60 %. The ensemble has no held-out figure, because its members were trained on different folds
  ([docs/evaluation/results.md](docs/evaluation/results.md)).
- **Some pages were not measured.** Born-digital pages, pages unlike the archaeological reports of the training data,
  and the optional YOLO backend have no held-out figure, so treat their labels as unchecked. The 2025 sweep's 98–99 %
  were measured mostly on training pages and are not held-out accuracy ([the sweep](docs/evaluation/sweep-2025.md)).
  The test suite checks that the code does what it says; it does not measure accuracy.
- **Neighbouring categories are the hard cases.** `TEXT_T` is most often taken for `TEXT` (recall 87–92 %), and
  `LINE_HW` for `LINE_T`, and one category is given per page. A low top score with a close second marks a page worth a
  look ([reading the output](docs/usage.md#reading-the-output)).
- **Name a revision.** The Hub's `main` revision still resolves to `v4.3`, the generation before `vX.4`, so pass
  `-rev v4.4` (or `--best`) rather than relying on it. `setup/config.txt` has `[HF] revision = main`, and the batch
  image's default command runs `-rev v4.3` ([models](docs/reference/models.md)).
- **PDFs are rendered with pypdfium2** at 300 dpi, by the service and by the command line, while the training pages
  were made with pdftoppm ([the service's known limits](service/README.md#known-limits)).
- **One service request is limited in size.** It classifies at most `MAX_PDF_PAGES` pages (50 by default; a deployment
  can change it) and takes at most 10 MB (`MAX_UPLOAD_MB`). Send a longer PDF in parts with `pages`. The command line
  has no such limit ([Limits](service/README.md#limits)).
- **The service's ensemble needs all five models.** With `version=all`, a model that fails to load or to answer
  fails the request (500, naming it) instead of averaging the others ([Errors](service/README.md#errors)). The
  command line's `--best` still leaves a failing model out, with a warning.
- **The service classifies nothing while it loads its models.** The first start downloads about 2.44 GB. Until the
  models are loaded, `/health` answers 200 while `/ready` and the two `/predict_*` endpoints answer 503, so wait for
  `/ready` to return 200 ([Start](service/README.md#start)).

## Documentation

| For                 | To                                                                      | Read                                                                                                                                                                                     |
|---------------------|-------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| using it            | see one result at once                                                  | the [Quick start](#quick-start) above                                                                                                                                                    |
|                     | learn what the eleven labels mean                                       | [docs/reference/categories.md](docs/reference/categories.md)                                                                                                                             |
|                     | classify your own pages, a folder at a time, and read the result tables | [docs/usage.md](docs/usage.md)                                                                                                                                                           |
|                     | look up models, flags, outputs, paradata and licences                   | [models](docs/reference/models.md), [flags](docs/reference/cli.md), [outputs](docs/reference/outputs.md), [paradata](docs/reference/paradata.md), [licences](docs/reference/licences.md) |
|                     | set every option of `setup/config.txt`                                  | [docs/configuration.md](docs/configuration.md)                                                                                                                                           |
| running the service | deploy and operate it, and read its limits and errors                   | [service/README.md](service/README.md)                                                                                                                                                   |
|                     | see the API contract                                                    | [service/openapi.json](service/openapi.json)                                                                                                                                             |
| judging the results | decide how far to rely on them                                          | [Limits and maturity](#limits-and-maturity) above                                                                                                                                        |
|                     | see what was measured, and on which data                                | [results](docs/evaluation/results.md), [data](docs/evaluation/data.md), [the 2025 sweep](docs/evaluation/sweep-2025.md)                                                                  |
| changing the code   | understand how a page is classified and how the ensemble runs           | [docs/how-it-works.md](docs/how-it-works.md)                                                                                                                                             |
|                     | set up, test, release; find the history                                 | [docs/development.md](docs/development.md), [CONTRIBUTING.md](CONTRIBUTING.md) (its Release History is the changelog)                                                                    |
|                     | read why it is built this way                                           | [docs/decisions/history.md](docs/decisions/history.md)                                                                                                                                   |
|                     | turn PDFs into an annotated training set                                | [docs/data-preparation.md](docs/data-preparation.md)                                                                                                                                     |
|                     | train, retrain on the published folds, evaluate                         | [docs/training.md](docs/training.md)                                                                                                                                                     |
|                     | see what is parked during the pilot (`decide_pages.py`)                 | [docs/reference/cli.md](docs/reference/cli.md#decision-models-on-page-images-decide_pagespy)                                                                                             |

Terms used across the tools are explained in the [ATRIUM glossary](https://ufal.github.io/atrium-project/glossary/).
The documentation follows the ATRIUM
[documentation style](https://github.com/ufal/atrium-project/blob/main/docs/documentation-style.md).

## Licence, citation and acknowledgements

The code is MIT ([LICENSE](LICENSE)). The fine-tuned models, the LINDAT training dataset they were trained on and the
sample pages in [small_data_samples](small_data_samples) are CC BY-NC 4.0, the optional YOLO backend (Ultralytics)
is AGPL-3.0, and the service renders PDFs with pypdfium2 (Apache-2.0 or BSD-3-Clause). The models are therefore
for non-commercial use only: the code's MIT licence does not extend to them. Each component, with its
licence and where that is stated: [docs/reference/licences.md](docs/reference/licences.md). Each run lists the
components it used, each with its own licence or terms. No licence of an output is computed: the rights in an output
follow the source document ([Licences and paradata](https://ufal.github.io/atrium-project/ecosystem/licences/)).

Cite the method with the preprints
[Page image classifier fine-tuned on century-spanning archives of scanned documents for further content-specific processing](https://arxiv.org/abs/2606.07558)
(arXiv 2606.07558, 2026) and
[Page image classification for content-specific data processing](https://arxiv.org/abs/2507.21114) (arXiv
2507.21114, 2025), the models with their [Hugging Face repository](https://huggingface.co/ufal/vit-historical-page)
and revision, the data with its [LINDAT record](http://hdl.handle.net/20.500.12800/1-6184), and the software with
[CITATION.cff](CITATION.cff), which lists the authors with their ORCIDs. The tool is listed in the SSH Open Marketplace as
[Historical documents page classifier - ATRIUM](https://marketplace.sshopencloud.eu/tool-or-service/RER7Fw).

Developed by [ÚFAL](https://ufal.mff.cuni.cz/), Charles University, with AMČR (the Archaeological Map of the Czech
Republic) as the pipeline partner. Funded by the European Union under Horizon Europe
([ATRIUM](https://atrium-research.eu/), GA 101132163), with the support of the LINDAT/CLARIAH-CZ (LM2023062) and AIS
CR (LM2023031) research infrastructures, funded by the Ministry of Education, Youth and Sports of the Czech Republic.
Contact: lutsai@ufal.mff.cuni.cz, or the [issue tracker](https://github.com/ufal/atrium-page-classification/issues).

© 2026 ÚFAL & ATRIUM
