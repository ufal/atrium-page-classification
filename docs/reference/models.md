# The models

For anyone who picks a model or reads a revision name: the fine-tuned classifiers, how their revisions are named,
which five make the ensemble, and how big they are.
This page is the one home of the model list; [service/README.md](../../service/README.md) links here. How well the
models do: [../evaluation/results.md](../evaluation/results.md). The code-side source of truth is
[model_registry.py](../../model_registry.py).

On this page:
[Where they are published](#where-they-are-published) ·
[How a revision is named](#how-a-revision-is-named) ·
[The published revisions](#the-published-revisions) ·
[The ensemble](#the-ensemble) ·
[Two version namespaces](#two-version-namespaces-and-where-they-collide) ·
[Size](#size) ·
[Alternatives beside the main line](#alternatives-beside-the-main-line)

## Where they are published

All fine-tuned models are revisions of one Hugging Face repository,
[`ufal/vit-historical-page`](https://huggingface.co/ufal/vit-historical-page). Their base models are Google's
[vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224),
[vit-base-patch16-384](https://huggingface.co/google/vit-base-patch16-384) and
[vit-large-patch16-384](https://huggingface.co/google/vit-large-patch16-384), and timm's
[regnety_160.swag_ft_in1k](https://huggingface.co/timm/regnety_160.swag_ft_in1k) and
[tf_efficientnetv2_m.in21k_ft_in1k](https://huggingface.co/timm/tf_efficientnetv2_m.in21k_ft_in1k); earlier
generations also used [tf_efficientnetv2_s.in21k](https://huggingface.co/timm/tf_efficientnetv2_s.in21k) and
[tf_efficientnetv2_l.in21k_ft_in1k](https://huggingface.co/timm/tf_efficientnetv2_l.in21k_ft_in1k).

**The Hub's `main` revision still resolves to `v4.3`,** the generation before `vX.4`. Pull a `vX.4` revision
explicitly (`-rev v4.4`) until `main` is moved. `setup/config.txt`'s `[HF] latest` already names `v4.4`, but
`[HF] revision` is `main`, and the batch image's default command is `-d /data/input --hf -rev v4.3`.

## How a revision is named

A revision is named `vX.Y`. `X` is the model slot, a base model:

| `X`: model slot | Base model                               | Input  |
|-----------------|------------------------------------------|--------|
| `v1.Y`          | `timm/tf_efficientnetv2_m.in21k_ft_in1k` | 384 px |
| `v2.Y`          | `google/vit-base-patch16-224`            | 224 px |
| `v3.Y`          | `google/vit-base-patch16-384`            | 384 px |
| `v4.Y`          | `timm/regnety_160.swag_ft_in1k`          | 384 px |
| `v5.Y`          | `google/vit-large-patch16-384`           | 384 px |

`Y` is the generation: which annotated data the model was trained on. Generations `.0`–`.2` are the early, smaller
annotation rounds (up to 14,270 pages), in which slots 1 and 4 held EfficientNetV2-S and EfficientNetV2-L; `.3` is
the full 48,499-page dataset with five-fold cross-validation; `.4` is the same five base models retrained on the
subset whose licence is CC BY-NC 4.0. The data of each generation: [../evaluation/data.md](../evaluation/data.md).

A revision builds on the base model `model_registry.py` lists for it, whatever `--base` or the config says, because
the image transforms are made from the base (exact key first, then the first key the revision starts with). `--base`
matters only for a revision the registry does not list, such as `main`.

## The published revisions

All share the same eleven categories, but not the same annotations. Since `v1.8.0-beta` the default ensemble is the
`vX.4` generation: what `--best` averages and what the service returns for `version=all`.

| Version | Base                             | Pages | PDFs      | Description                                                                        |
|--------:|----------------------------------|:-----:|:---------:|:-----------------------------------------------------------------------------------|
|  `v2.0` | `vit-base-patch16-224`           | 10073 | **3896**  | annotations with mistakes, more heterogenous data                                  |
|  `v2.1` | `vit-base-patch16-224`           | 11940 | **5002**  | once `main`: more diverse pages in each category, less annotation mistakes         |
|  `v2.2` | `vit-base-patch16-224`           | 14270 | **5730**  | same data as `v2.1` + some restored pages from `v2.0`                              |
|  `v3.2` | `vit-base-patch16-384`           | 14270 | **5730**  | same data as `v2.2`, but a bit larger model base with higher resolution            |
|  `v5.2` | `vit-large-patch16-384`          | 14270 | **5730**  | same data as `v2.2`, but the largest model base with higher resolution             |
|  `v1.2` | `efficientnetv2_s.in21k`         | 14270 | **5730**  | same data as `v2.2`, but the smallest model base (CNN)                             |
|  `v4.2` | `efficientnetv2_l.in21k_ft_in1k` | 14270 | **5730**  | same data as `v2.2`, CNN base model smaller than the largest, may be more accurate |
|  `v2.4` | `vit-base-patch16-224`           | 38307 | **37316** | only the CC BY-NC 4.0 licensed data samples were kept in the `vX.4` versions' data |
|  `v3.4` | `vit-base-patch16-384`           | 38307 | **37316** | same data as `v2.4`, but a bit larger model base with higher resolution            |
|  `v5.4` | `vit-large-patch16-384`          | 38307 | **37316** | same data as `v2.4`, but the largest model base with higher resolution             |
|  `v1.4` | `efficientnetv2_m.in21k_ft_in1k` | 38307 | **37316** | same data as `v2.4`, but the smallest model base (CNN)                             |
|  `v4.4` | `regnety_160.swag_ft_in1k`       | 38307 | **37316** | same data as `v2.4`, CNN base model bigger than the smallest, may be more accurate |

The `vX.3` generation (`v1.3`–`v5.3`, the same five base models as `vX.4`, trained on 38,625 pages) is also on the
Hub; `v4.3` is what `main` serves.

## The ensemble

`--best`, `parallel_best.run_best_models` and the service's `version=all` all read one dict,
`REVISION_BEST_MODELS`. Since `v1.8.0-beta` it names the `vX.4` generation:

| Revision | Base model                               | Fold    | Resolution | Parameters (bytes) | Described as                                                 |
|----------|------------------------------------------|---------|------------|--------------------|--------------------------------------------------------------|
| `v1.4`   | `timm/tf_efficientnetv2_m.in21k_ft_in1k` | `fold1` | 384        | 211,489,788        | CNN-based, faster inference                                  |
| `v2.4`   | `google/vit-base-patch16-224`            | `fold5` | 224        | 343,228,460        | standard Transformer baseline                                |
| `v3.4`   | `google/vit-base-patch16-384`            | `fold2` | 384        | 344,395,820        | higher-resolution Transformer baseline                       |
| `v4.4`   | `timm/regnety_160.swag_ft_in1k`          | `fold1` | 384        | 322,393,660        | balanced option; the configured single model (`[HF] latest`) |
| `v5.4`   | `google/vit-large-patch16-384`           | `fold2` | 384        | 1,214,808,108      | the largest model, slowest inference                         |

The fold is the cross-validation split each model was trained on (`REVISION_BEST_FOLDS`); the rule is
`splitN ↔ foldN column ↔ seed = 420 + (N−1)`. Resolution and parameter size come from `MODEL_STATIC`, which
`--parallel` reads to group models by memory before running them. The service loads these five at start and exposes
them as `available_models` in `GET /info`; any other `version` that resolves to a base model is downloaded on its
first request.

On their held-out test pages the five do not differ much: 94.64–95.87 % top-1, and models scored on the same fold do
not differ significantly ([results](../evaluation/results.md)). The service's descriptions used to call `v5.4` the most
accurate and `v4.4` the best performing small model; the held-out evaluation does not support either.

[tests/test_best_ensemble_distinct.py](../../tests/test_best_ensemble_distinct.py) guards the ensemble: its static
half asserts that the five revisions name five distinct base models, and its `-m slow` half reads each revision's
`config.json` from the Hub and asserts that the published architectures are distinct too. Why that test exists:
[the v1.8.0-beta switch](../decisions/history.md#2026-09-the-ensemble-moves-to-vx4-48).

## Two version namespaces, and where they collide

`REVISION_TO_BASE_MODEL` holds two numberings, and reading it as one is the mistake:

* **Sweep numbering:** one number per candidate base model of the 2025 sweep, as recorded in
  [model_accuracies_new.csv](../../model_accuracies_new.csv): `v4.` = EfficientNetV2-L, `v6.` = RegNetY-120,
  `v7.` = RegNetY-160, `v8.` = RegNetY-640, `v9.`/`v10.`/`v11.` = the DiT family, `v12.` = EfficientNetV2-M.
* **Published numbering:** the five sweep winners, re-published under compact names: `v1.3` = EfficientNetV2-M
  (sweep `v12.3.1`), `v2.3` = ViT-B/16-224, `v3.3` = ViT-B/16-384, `v4.3` = RegNetY-160 (sweep `v7.3.1`),
  `v5.3` = ViT-L/16-384.

They collide on `v1.3` and `v4.3`. The published meaning wins, by exact-key-first resolution rather than by the order
of the keys. `model_accuracies_new.csv` records the sweep namespace; the Hugging Face model card and this
documentation use the published one. The sweep itself: [../evaluation/sweep-2025.md](../evaluation/sweep-2025.md).

## Size

| Base model                       | Parameters (M) | Resolution (px) | Revision       | Disk space of the fine-tuned model |
|----------------------------------|----------------|-----------------|----------------|------------------------------------|
| `efficientnetv2_s.in21k`         | 48             | 300             | `v1.2`         |                                    |
| `efficientnetv2_m.in21k_ft_in1k` | 54             | 384             | `v1.3`, `v1.4` | 213 MB                             |
| `regnety_160.swag_ft_in1k`       | 84             | 384             | `v4.3`, `v4.4` | 323 MB                             |
| `vit-base-patch16-224`           | 87             | 224             | `v2.X`         | 344 MB                             |
| `vit-base-patch16-384`           | 87             | 384             | `v3.X`         | 345 MB                             |
| `vit-large-patch16-384`          | 305            | 384             | `v5.X`         | 1.2 GB                             |

Up to 1 GB of disk is needed for the model files and checkpoints of a run, and up to 7 GB for the Python libraries.
A model downloaded with `--hf` is saved to `model/model_<revision without dots>`, for example `model/model_v44`.

## Alternatives beside the main line

* **YOLO-cls.** `--yolo` swaps the Hugging Face models for an Ultralytics YOLO classification model (`--yolo_base`,
  for example `yv8s`; the `[YOLO]` section of [the configuration](../configuration.md#yolo)). It is an option for
  training and inference with a different speed and accuracy trade-off; it does not support `--hf` or `--best`, and
  it brings the AGPL-3.0 licence of Ultralytics into the run ([licences](licences.md)). It has not been evaluated
  here.
* **CLIP.** A separate, CLIP-based classifier for the same categories lives on the repository's `clip` branch, with
  its own Hugging Face repository.
