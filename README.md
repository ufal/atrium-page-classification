# ATRIUM page-classification: scanned archive pages sorted into eleven page types

page-classification looks at the image of a scanned archival page and says what kind of page it is: one of eleven
categories that combine running text, tables or forms, drawings and photographs with handwritten, printed or typed
text. The category tells which processing a page needs next (OCR, handwritten text recognition, table extraction or
image extraction) before any of it is run. It reads the image only, so the language of the page does not matter.
Use it on scans of paper documents; born-digital pages are classified too, but those labels have not been checked.

[Releases](https://github.com/ufal/atrium-page-classification/releases) · [MIT licence](LICENSE) ·
[ATRIUM documentation](https://ufal.github.io/atrium-project/workflows/page-classification/) ·
[Models on Hugging Face](https://huggingface.co/ufal/vit-historical-page)

The classifiers are fine-tuned Vision Transformers, RegNetY and EfficientNetV2 models, published as revisions of
`ufal/vit-historical-page`. One revision can run alone, or the five `vX.4` revisions run as an ensemble that
averages their scores; the ensemble is the default of the HTTP service. They were trained on annotated pages of
archaeological reports from Czech archives, 1920–2020, published as a dataset on LINDAT.

## Start here

1. Run the [Quick start](#quick-start) below: one sample page, through the service or the command line.
2. Learn what the eleven labels mean: [docs/reference/categories.md](docs/reference/categories.md).
3. Classify your own pages, a folder at a time, and read the result tables: [docs/usage.md](docs/usage.md).
4. Before you rely on the results, read [Limits and maturity](#limits-and-maturity) and what was measured:
   [docs/evaluation/results.md](docs/evaluation/results.md).
5. Everything else is in the [documentation map](#documentation).

## Inputs and outputs

| Input                                                                                                                 | Output                                                                                                                              |
|-----------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|
| a page image (any format Pillow reads: PNG, JPEG, TIFF, …), a folder or a tree of them, named `<document>-<page>.png` | per page, the best categories with their scores: a CSV (`FILE, PAGE, CLASS-1…N, SCORE-1…N`), or JSON from the service               |
| a PDF, recognised by its content: every page, or a `pages` selection, rendered at 300 dpi                             | optionally all eleven scores (`--raw`, or `topn=11`) and each model's own vote (`--best --save-intermediates`, or `per_model=true`) |
| optionally, an ATRIUM document record to add to                                                                       | the record with its `page_categories` block and each page's `category` and `category_confidence`; the run's paradata                |

The command line and the service read the same inputs the same way (since v1.10.0-beta): a PDF is rendered by the
service's own renderer on both sides, with the same `pages` selection, doc_id and page keys, and a born-digital
record's page labels are honoured on both. What differs, and why, is in [docs/usage.md](docs/usage.md#the-command-line-and-the-service):
directories and training are the command line's, the upload limits the service's, and the two still average and
round scores differently.

The eleven categories and their definitions: [docs/reference/categories.md](docs/reference/categories.md). The
record follows the hub's [document contract](https://ufal.github.io/atrium-project/ecosystem/document-contract/);
page-classification is its first stage, so it creates the record as often as it adds to one.

## Quick start

The service, from the published image, on a sample page of this repository:

```bash
docker run --rm -p 8000:8000 -v hf-cache:/cache/huggingface \
  ghcr.io/ufal/atrium-page-classification-api:<version>     # the release without its v, e.g. 1.10.4-beta
curl -s localhost:8000/ready                                 # 200 once the five models are loaded
curl --fail-with-body -sS -X POST localhost:8000/predict_image \
  -F "file=@small_data_samples/TEXT_T/pages_online_2.png" -F version=v4.4 -F topn=3
```

Expected result (abridged):

```json
{"type": "image",
 "predictions": [{"label": "TEXT_T", "score": 0.9}, {"label": "TEXT", "score": 0.099}, {"label": "TEXT_P", "score": 0.001}],
 "models_used": ["v4.4"], "limits_applied": [], "document_json": null,
 "paradata": {"@type": "CreateAction", "…": "…"}}
```

The scores are those `v4.4` gave this page in the stored run
[result/tables/20261022-1218_v44_TOP-3.csv](result/tables/20261022-1218_v44_TOP-3.csv), rounded. Without `version`
the service uses the ensemble (`all`), and `models_used` lists the five revisions that answered. The image holds no
weights: on its first start it downloads the five models from the Hugging Face Hub into the `hf-cache` volume, and
`/ready` answers 503 until they are loaded. Every endpoint, limit and error: [service/README.md](service/README.md).

The same page on the command line, from a checkout with its [setup](docs/usage.md#install) done:

```bash
python3 run.py --hf -rev v4.4 -f small_data_samples/TEXT_T/pages_online_2.png
```

After the argument list and the download messages it prints:

```text
File small_data_samples/TEXT_T/pages_online_2.png predicted:
	TEXT_T:  90.0%
	TEXT:  9.9%
	TEXT_P:  0.1%
Results for TOP-3 predictions are recorded into ./result/tables/<date-time>_model_v44_TOP-3.csv
```

## How it works

Each page image is resized to the model's square input (224 or 384 pixels) and scored for all eleven categories;
the scores become probabilities, and the best ones are kept, best first. With `--best` (or `version=all`) five
models score every page and their probabilities are averaged, so a page on which they disagree gets a lower top
score. The command line writes CSV tables and a paradata log; the service answers in JSON. More, with the diagrams:
[docs/how-it-works.md](docs/how-it-works.md).

## Where it fits in ATRIUM

page-classification is the first stage of the route for scanned documents. Its categories inform a routing
decision, taken by a person or a script reading the results: which pages go to OCR, to handwriting recognition, to
table extraction or to image extraction. The next stage, ocr-postprocess, works from the OCR output and does not
read the categories. On the born-digital route, digital-convert's `/describe` asks `/predict_document` about the
pages that have no usable text layer. The whole chain: the
[hub's workflow page](https://ufal.github.io/atrium-project/workflows/page-classification/).

## Limits and maturity

- **What was measured.** Each of the five `vX.4` models was scored on the held-out test pages of the fold it was
  trained on: 94.64–95.87 % top-1 and 99.75–99.90 % top-3, about 4,800 pages per model, one fold per model. The
  split is by page: about a quarter of the test pages come from documents with other pages in training, and on pages
  of unseen documents top-1 is 93.87–95.60 %. The ensemble has no held-out figure, because its members were trained
  on different folds ([docs/evaluation/results.md](docs/evaluation/results.md)).
- **What was not.** Born-digital pages, pages unlike the archaeological reports of the training data, and the
  optional YOLO backend. The 2025 sweep's 98–99 % were measured mostly on training pages and are not held-out
  accuracy ([the sweep](docs/evaluation/sweep-2025.md)). The test suite checks that the code does what it says;
  it does not measure accuracy.
- **Neighbouring categories are the hard cases.** `TEXT_T` is most often taken for `TEXT` (recall 87–92 %), and
  `LINE_HW` for `LINE_T`. One category is given per page.
- **Name a revision.** The Hub's `main` revision still resolves to `v4.3`, the generation before `vX.4`; pass
  `-rev v4.4` (or `--best`) rather than relying on it. `setup/config.txt` has `[HF] revision = main`, and the batch
  image's default command runs `-rev v4.3` ([models](docs/reference/models.md)).
- **PDFs are rendered with pypdfium2** at 300 dpi, by the service and, since v1.10.0-beta, by the command line,
  while the training pages were made with pdftoppm. One service request classifies at most `MAX_PDF_PAGES` pages (50
  by default; a deployment can change it) and takes at most 10 MB (`MAX_UPLOAD_MB`); a longer PDF is sent in parts
  with `pages` ([Limits](service/README.md#limits)). The command line has no such limit.
- **The ensemble can answer with fewer models** when one fails to load; `models_used` then lists fewer than five
  ([Partial results](service/README.md#partial-results)).

## Documentation

| To                                                                 | Read                                                                                                                                                                                                                                 |
|--------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| install it, classify pages, read the results, use the images       | [docs/usage.md](docs/usage.md)                                                                                                                                                                                                       |
| deploy and operate the service, its limits and errors              | [service/README.md](service/README.md)                                                                                                                                                                                               |
| see the API contract                                               | [service/openapi.json](service/openapi.json)                                                                                                                                                                                         |
| set every option of `setup/config.txt`                             | [docs/configuration.md](docs/configuration.md)                                                                                                                                                                                       |
| understand how a page is classified and how the ensemble runs      | [docs/how-it-works.md](docs/how-it-works.md)                                                                                                                                                                                         |
| look up the categories, models, flags, outputs, paradata, licences | [categories](docs/reference/categories.md), [models](docs/reference/models.md), [flags](docs/reference/cli.md), [outputs](docs/reference/outputs.md), [paradata](docs/reference/paradata.md), [licences](docs/reference/licences.md) |
| see what was measured, and on which data                           | [results](docs/evaluation/results.md), [data](docs/evaluation/data.md), [the 2025 sweep](docs/evaluation/sweep-2025.md)                                                                                                              |
| turn PDFs into an annotated training set                           | [docs/data-preparation.md](docs/data-preparation.md)                                                                                                                                                                                 |
| train, retrain on the published folds, evaluate                    | [docs/training.md](docs/training.md)                                                                                                                                                                                                 |
| contribute, test, release                                          | [docs/development.md](docs/development.md), [CONTRIBUTING.md](CONTRIBUTING.md) (its Release History is the changelog)                                                                                                                |
| read why it is built this way                                      | [docs/decisions/history.md](docs/decisions/history.md)                                                                                                                                                                               |

The documentation follows the ATRIUM
[documentation style](https://github.com/ufal/atrium-project/blob/test/docs/documentation-style.md).

## Licence, citation and acknowledgements

The code is MIT ([LICENSE](LICENSE)). The fine-tuned models are declared MIT, the LINDAT training dataset and the
sample pages in [small_data_samples](small_data_samples) are CC BY-NC 4.0, the optional YOLO backend (Ultralytics)
is AGPL-3.0, and the service renders PDFs with pypdfium2 (Apache-2.0 or BSD-3-Clause). Each component, with its
licence and where that is stated: [docs/reference/licences.md](docs/reference/licences.md). The paradata of a run
lists the components it used; it does not set the rights in what the run produces, which depend first on the
source document ([Licences and paradata](https://ufal.github.io/atrium-project/ecosystem/licences/)).

Cite the method with the preprints
[Page image classifier fine-tuned on century-spanning archives of scanned documents for further content-specific processing](https://arxiv.org/abs/2606.07558)
(arXiv 2606.07558, 2026) and
[Page image classification for content-specific data processing](https://arxiv.org/abs/2507.21114) (arXiv
2507.21114, 2025), the models with their [Hugging Face repository](https://huggingface.co/ufal/vit-historical-page)
and revision, the data with its [LINDAT record](http://hdl.handle.net/20.500.12800/1-6184), and the software with
[CITATION.cff](CITATION.cff), which lists the authors with their ORCIDs. The tool is listed in the SSH Open Marketplace as
[Historical documents page classifier - ATRIUM](https://marketplace.sshopencloud.eu/tool-or-service/RER7Fw).

Developed by [ÚFAL](https://ufal.mff.cuni.cz/), Charles University, with AMČR (the Archaeological Map of the Czech
Republic) as the pipeline partner. Funded by [ATRIUM](https://atrium-research.eu/) (GA 101132163), with the support
of the LINDAT/CLARIAH-CZ research infrastructure and the AIS CR infrastructure (LM2023031). Contact:
lutsai@ufal.mff.cuni.cz, or the [issue tracker](https://github.com/ufal/atrium-page-classification/issues).

© 2026 ÚFAL & ATRIUM
