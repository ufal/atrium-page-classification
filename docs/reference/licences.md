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
| `vit_models`                  | the fine-tuned classifiers, [`ufal/vit-historical-page`](https://huggingface.co/ufal/vit-historical-page) | MIT                        | `para_config.txt`                                                                                                                                                        | always                                    |
| `lindat_dataset`              | the annotated page dataset, 48,499 pages                                                                  | CC BY-NC 4.0               | the [LINDAT record](http://hdl.handle.net/20.500.12800/1-6184)                                                                                                           | `--train`, and `--eval` since v1.9.9-beta |
| `ultralytics`                 | the YOLO-cls backend and its base weights                                                                 | AGPL-3.0                   | [Ultralytics](https://github.com/ultralytics/ultralytics)                                                                                                                | `--yolo`                                  |
| `pypdfium2`                   | PDFium page rendering of a PDF, on the command line and in `/predict_document`                            | Apache-2.0 or BSD-3-Clause | [pypdfium2](https://github.com/pypdfium2-team/pypdfium2) (PDFium: BSD-3-Clause); recorded as Apache-2.0, because `para_licenses.py` does not parse SPDX `OR` expressions | a PDF is rendered                         |

## Other licences in the repository

| What                                                    | Licence                                 | Where it is stated                                                                                                    |
|---------------------------------------------------------|-----------------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| the code                                                | MIT                                     | [LICENSE](../../LICENSE), [CITATION.cff](../../CITATION.cff)                                                          |
| the sample pages in `small_data_samples/`               | CC BY-NC 4.0                            | [small_data_samples/LICENSE](../../small_data_samples/LICENSE)                                                        |
| result tables with a `TRUE` column                      | CC BY-NC 4.0 (the dataset's annotation) | [result/README.md](../../result/README.md#license-of-these-files)                                                     |
| the base models (Google's ViTs, timm's EfficientNetV2)  | their model cards'                      | the Hugging Face pages linked from [models.md](models.md#where-they-are-published); not declared in `para_config.txt` |
| the RegNetY base of `v4.3`/`v4.4` (Meta's SWAG weights) | CC BY-NC 4.0                            | its [model card](https://huggingface.co/timm/regnety_160.swag_ft_in1k); not in `para_config.txt` (below)              |
| Windows data preparation: ImageMagick, Ghostscript      | Ghostscript is AGPL                     | their download pages ([data-preparation.md](../data-preparation.md)); used only to make page images                   |

**The SWAG base is not a recorded component.** `para_config.txt` declares the fine-tuned models as `vit_models`
(MIT), so a classification run's label stays MIT even when `v4.3`, `v4.4` or the ensemble, which includes `v4.4`, does
the work. Whether the CC BY-NC 4.0 of the SWAG base, and of the dataset the `vX.4` models were trained on, carries
over to the fine-tuned models, and so should be declared, is asked on
[atrium-project#6](https://github.com/ufal/atrium-project/issues/6); the declaration changes with its answer.

## The label the paradata computes

From the components a run used, the shared module [para_licenses.py](../../para_licenses.py) computes one label, the
most restrictive of their licences, and the paradata carries it. For this tool:

- a classification run, on the command line or through the service, including a PDF rendered with pypdfium2:
  **MIT**;
- `--train`, and `--eval` since v1.9.9-beta (the `TRUE` column of every evaluation table is the dataset's
  annotation): **CC BY-NC 4.0**;
- `--yolo`: an inference run gets **AGPL-3.0**; with `--train`, CC BY-NC 4.0 ranks above it.

**The label is not the licence of the output.** It was added as a warning: it says at a glance that a run used, for
example, a dataset published for non-commercial use. It does not say whether the licences used can be combined, or
whether an output is a derived work of a component. Training uses the LINDAT dataset (CC BY-NC 4.0: non-commercial,
not share-alike); what that means for weights trained on it is for whoever publishes them to establish, and the label
does not decide it. The hub's [Licences and paradata](https://ufal.github.io/atrium-project/ecosystem/licences/)
explains what replaces the label ([atrium-project#6](https://github.com/ufal/atrium-project/issues/6)).

## The PDF renderer

Since v1.9.1-beta the service renders PDF pages with **pypdfium2**. It replaced PyMuPDF, an AGPL-3.0 library that
was installed in both images and declared nowhere
([atrium-project#72](https://github.com/ufal/atrium-project/issues/72)). The two renderers are not identical: on
2,514 pages of 15 born-digital AMČR reports they gave the same top-1 label on 2,509, and the other five moved to a
neighbouring class. The training pages were rendered with pdftoppm, which neither engine reproduces exactly.
Measurements and the five pages:
[data_scripts/pdf_rasteriser_comparison.md](../../data_scripts/pdf_rasteriser_comparison.md).
