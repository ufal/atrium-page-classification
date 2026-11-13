# Licences

For anyone who publishes or reuses what page-classification produces, or redistributes the tool: the licences of
the code, the models, the data and the libraries it uses, and where each is stated.
None of them is the licence of what a run produces: the rights in an output depend first on the source document,
and no tool knows those ([Licences and paradata](https://ufal.github.io/atrium-project/ecosystem/licences/) on the
hub).

## The components a run records

[setup/para_config.txt](../../setup/para_config.txt) declares the components; the paradata of each run lists the
ones it used, each with its licence ([paradata.md](paradata.md)).

| Component (`para_config.txt`) | What it is                                                                                                | Licence                    | Where the licence is stated                                                                                                                                              | Recorded when                             |
|-------------------------------|-----------------------------------------------------------------------------------------------------------|----------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------|
| `vit_models`                  | the fine-tuned classifiers, [`ufal/vit-historical-page`](https://huggingface.co/ufal/vit-historical-page) | CC BY-NC 4.0               | `para_config.txt`: trained on the dataset below; `v4.3` and `v4.4` also fine-tuned from Meta's SWAG weights (CC BY-NC 4.0)                                               | always                                    |
| `lindat_dataset`              | the annotated page dataset, 48,499 pages                                                                  | CC BY-NC 4.0               | the [LINDAT record](http://hdl.handle.net/20.500.12800/1-6184)                                                                                                           | `--train`, and `--eval` since v1.9.9-beta |
| `ultralytics`                 | the YOLO-cls backend and its base weights                                                                 | AGPL-3.0                   | [Ultralytics](https://github.com/ultralytics/ultralytics)                                                                                                                | `--yolo`                                  |
| `pypdfium2`                   | PDFium page rendering of a PDF, on the command line and in `/predict_document`                            | Apache-2.0 or BSD-3-Clause | [pypdfium2](https://github.com/pypdfium2-team/pypdfium2) (PDFium: BSD-3-Clause); recorded as Apache-2.0, because `para_licenses.py` does not parse SPDX `OR` expressions | a PDF is rendered                         |
| `decision_model`              | the decision models of the `decide_pages.py` experiment: Clef, Clef-Flash, Strands Decider 2B             | Apache-2.0                 | `para_config.txt`: all three are Apache-2.0; they run on the server `decide_pages.py` calls, never in `run.py` or the service                                            | every `decide_pages.py` run               |

## Other licences in the repository

| What                                                    | Licence                                 | Where it is stated                                                                                                    |
|---------------------------------------------------------|-----------------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| the code                                                | MIT                                     | [LICENSE](../../LICENSE), [CITATION.cff](../../CITATION.cff)                                                          |
| the sample pages in `small_data_samples/`               | CC BY-NC 4.0                            | [small_data_samples/LICENSE](../../small_data_samples/LICENSE)                                                        |
| result tables with a `TRUE` column                      | CC BY-NC 4.0 (the dataset's annotation) | [result/README.md](../../result/README.md#license-of-these-files)                                                     |
| the base models (Google's ViTs, timm's EfficientNetV2)  | their model cards'                      | the Hugging Face pages linked from [models.md](models.md#where-they-are-published); not declared in `para_config.txt` |
| the RegNetY base of `v4.3`/`v4.4` (Meta's SWAG weights) | CC BY-NC 4.0                            | its [model card](https://huggingface.co/timm/regnety_160.swag_ft_in1k); carried by `vit_models` (below)               |
| Windows data preparation: ImageMagick, Ghostscript      | Ghostscript is AGPL                     | their download pages ([data-preparation.md](../data-preparation.md)); used only to make page images                   |

**The fine-tuned models are CC BY-NC 4.0.** MIT is a software licence, and it does not carry the non-commercial
clause of what the models are made from: every generation is trained on the LINDAT dataset (CC BY-NC 4.0), and
`v4.3`, `v4.4` and the ensemble, which includes `v4.4`, also start from the SWAG weights. So `vit_models` is declared
CC BY-NC 4.0 and listed in every run ([atrium-project#6](https://github.com/ufal/atrium-project/issues/6), 29 October
2026).

## What the list does not say

The components a run lists, with their licences, are not the licence of what the run produces: the rights in an
output depend first on the source document. What weights trained on the dataset may be used for is for whoever
publishes them to establish. Since paradata 3.0 no licence is calculated from the list
([Licences and paradata](https://ufal.github.io/atrium-project/ecosystem/licences/) on the hub).

## The PDF renderer

Since v1.9.1-beta the service renders PDF pages with **pypdfium2**. It replaced PyMuPDF, an AGPL-3.0 library that
was installed in both images and declared nowhere
([atrium-project#72](https://github.com/ufal/atrium-project/issues/72)). The two renderers are not identical: on
2,514 pages of 15 born-digital AMČR reports they gave the same top-1 label on 2,509, and the other five moved to a
neighbouring class. The training pages were rendered with pdftoppm, which neither engine reproduces exactly.
Measurements and the five pages:
[data_scripts/pdf_rasteriser_comparison.md](../../data_scripts/pdf_rasteriser_comparison.md).
