# Training and evaluation

For anyone who fine-tunes a model on their own pages, retrains the published models on their original folds, or
scores a model: the commands, the defaults behind them, and what each run writes. The flags are in
[reference/cli.md](reference/cli.md#training-and-evaluation); the measured results of the published models are in
[evaluation/results.md](evaluation/results.md).

On this page:
[Before you start](#before-you-start) ·
[Training](#training) ·
[Hyperparameters](#hyperparameters) ·
[Image augmentation](#image-augmentation) ·
[Where the model is saved](#where-the-model-is-saved) ·
[How the data are split](#how-the-data-are-split) ·
[Averaging fold models](#averaging-fold-models) ·
[Retraining on explicit dataset folds](#retraining-on-explicit-dataset-folds) ·
[Evaluation](#evaluation) ·
[Publishing a model on the Hub](#publishing-a-model-on-the-hub) ·
[Re-running the `vX.4` evaluation](#re-running-the-vx4-evaluation)

## Before you start

```bash
python3 run.py --train                                         # train, with the config's settings
python3 run.py --eval -rev v4.4                                # score one revision
python3 run.py --train --folds 5                               # 5-fold cross-validation, 80/10/10
python3 run.py --train -rev v4.4 --folds_csv folds.csv         # retrain on a published fold
python3 run.py --average -ap model_v4                          # average existing fold weights
```

- **The training folder.** For training and evaluation the pages folder in [setup/config.txt](../setup/config.txt)
  (`[TRAIN] FOLDER_PAGES`; `[EVAL] FOLDER_PAGES` or `--eval_dir` for evaluation) must hold one sub-folder per
  category with the images inside, and nothing else at that level. The sub-folder names, sorted alphabetically,
  become the label names and replace the default category list. How to make such a folder:
  [data-preparation.md](data-preparation.md).
- **A machine.** A CPU of any kind with enough memory, and a CUDA GPU (an Nvidia card) for any real training:
  efficient training is possible only on a CUDA-capable GPU. For test runs on a CPU-only machine set the batch size
  below 4, and even then more than average memory is needed. Memory per batch size:
  [configuration.md](configuration.md#setup).
- **Time.** Training logs its progress to the console and takes about 5–12 hours, depending on the CPU or GPU memory
  and the size of the dataset.

## Training

```bash
python3 run.py --train
```

> [!TIP]
> Train with the **default hyperparameters** if you have at least about 10,000 and fewer than 50,000 pages similar
> to the original data. Fine-tuning for the same task on an extended or new dataset of document pages needs no other
> change, and the number of categories does not matter while it stays under 20.

## Hyperparameters

The defaults, the `TrainingArguments`
([Hugging Face documentation](https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments))
of the training run:

* `eval_strategy` "epoch"
* `save_strategy` "epoch"
* `learning_rate` **5e-5**
* `per_device_train_batch_size` 8
* `per_device_eval_batch_size` 8
* `num_train_epochs` **3**
* `warmup_ratio` **0.1**
* `logging_steps` **10**
* `load_best_model_at_end` True
* `metric_for_best_model` "accuracy"

`epochs`, `log_step` and `lr` in the `[TRAIN]` section of [setup/config.txt](../setup/config.txt), and `batch` in
`[SETUP]`, change the epochs, the logging steps, the learning rate and the batch size of a run; the warm-up ratio and
the other hyperparameters are set only in [classifier.py](../classifier.py), which also has predefined values that
[run.py](../run.py) does not use.

Change the hyperparameters only if the **training loss** descends too slowly to reach values of about 0.001 by the end
of the third (by default, the last) epoch. If the **evaluation loss** starts to rise steadily after descending, you
have reached the limit of useful epochs: next time set `epochs` to the last epoch that ended before the evaluation
loss started to grow.

## Image augmentation

During training, image transformations
([torchvision transforms](https://pytorch.org/vision/0.20/transforms.html)) are applied one after another, each with
a 50 % chance:

* `transforms.ColorJitter(brightness=0.5)`
* `transforms.ColorJitter(contrast=0.5)`
* `transforms.ColorJitter(saturation=0.5)`
* `transforms.ColorJitter(hue=0.5)`
* `transforms.Lambda(lambda img: ImageEnhance.Sharpness(img).enhance(random.uniform(0.5, 1.5)))`
* `transforms.Lambda(lambda img: img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 2))))`

No rotation, reshaping or flipping is applied, only colour, sharpness and blur. The reasons are the pages that hold
particular kinds of forms, the general orientation of text on a page, and the reshaping of every input to the model's
square resolution (224×224 for the base ViT): page orientation is part of what the model is asked to read.

## Where the model is saved

After training the model is saved to its own folder in `model/`. By default the folder is named after `revision` in
the `[HF]` section of the config, shortened by removing dots: `model_<revision>`. That name does **not** affect the
trained model itself; after training, `-m` (`--model`) points evaluation or prediction at any model folder.

```text
atrium-page-classification
├── model
│   ├── model_<HFrevision1>
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   └── preprocessor_config.json
│   ├── model_<HFrevision2>
│   └── ...
├── checkpoint
│   ├── models--google--vit-base-patch16-224
│   │   ├── blobs
│   │   ├── snapshots
│   │   └── refs
│   └── .locs
│       └── models--google--vit-base-patch16-224
├── model_output
│   ├── checkpoint-version1
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   ├── trainer_state.json
│   │   ├── optimizer.pt
│   │   ├── scheduler.pt
│   │   ├── rng_state.pth
│   │   └── training_args.bin
│   ├── checkpoint-version2
│   └── ...
├── data_scripts
├── result
│   ├── plots
│   └── tables
├── setup
├── small_data_samples
└── ...
```

## How the data are split

The project is adapted to file names that carry date stamps: the files of each category are sorted by name, and a
randomised periodic selection takes the final evaluation (test) and the training-time evaluation (development)
subsets, both of `test_size` (10 %); the rest is training data, 80/10/10. The method is described in
[evaluation/data.md](evaluation/data.md#how-the-dataset-is-split). This happens on a single `--train` run and on every
fold of a cross-validation (`--folds N`, or `[TRAIN] cross_runs` above 0).

> [!TIP]
> Cross-validation takes longer and selects the subsets again for each run, with the `seed` of `[SETUP]`
> incremented by one for each fold. The split of each fold is written as a `.txt` file to `result/stats`, and the
> fold's test-set predictions to `result/tables`. The trained models are saved as `model_<revision><fold>`.

## Averaging fold models

Models trained in cross-validation on the same base model can be averaged into one model, for evaluation or
prediction:

```bash
python3 run.py --average -ap model_<revision>
```

`model_<revision>` is the common part of the model folders' names. The result is a new model, saved as
`model_<revision>a<#folds>` next to its parents in the model folder. Averaging the five folds' weights usually gives a
more robust model; for the categories whose pages look alike (`TEXT` and `TEXT_T`) it can also lower their accuracy
([evaluation/data.md](evaluation/data.md)).

## Retraining on explicit dataset folds

To **reproduce a model's original train/dev/test split** rather than regenerate it (for example, to fine-tune the
five best models again on a slightly reduced dataset and confirm they reach the original accuracy on the **same
evaluation pages**), use `--folds_csv`. Instead of the random 80/10/10 selection, the split is read from a CSV that
encodes the original folds.

The CSV has a `PNG` column with the **exact file name on disk** (such as `CTX192700593-08.png`, already zero-padded)
and one `foldN` column per split, whose cells are `train`, `dev` or `test`. Run one command per model; the fold column
is chosen per revision, so you pass only the CSV:

```bash
python3 run.py --train -rev v1.4 --folds_csv path/to/licensed_crossval_folds_CUT.csv
python3 run.py --train -rev v2.4 --folds_csv path/to/licensed_crossval_folds_CUT.csv
python3 run.py --train -rev v3.4 --folds_csv path/to/licensed_crossval_folds_CUT.csv
python3 run.py --train -rev v4.4 --folds_csv path/to/licensed_crossval_folds_CUT.csv
python3 run.py --train -rev v5.4 --folds_csv path/to/licensed_crossval_folds_CUT.csv
```

The mapping per revision (one source of truth: `REVISION_BEST_FOLDS` in [model_registry.py](../model_registry.py))
follows the rule `splitN ↔ foldN column ↔ seed = 420 + (N−1)`:

| Revision | Base model                               | Fold column |
|----------|------------------------------------------|-------------|
| `v1.4`   | `timm/tf_efficientnetv2_m.in21k_ft_in1k` | `fold1`     |
| `v2.4`   | `google/vit-base-patch16-224`            | `fold5`     |
| `v3.4`   | `google/vit-base-patch16-384`            | `fold2`     |
| `v4.4`   | `timm/regnety_160.swag_ft_in1k`          | `fold1`     |
| `v5.4`   | `google/vit-large-patch16-384`           | `fold2`     |

- **Pages absent from the CSV are left out** of every subset. This is how removed pages (and any image on disk that
  the CSV does not list) are dropped, so the retraining runs on the reduced dataset by itself. The chosen split is
  written as `…_<foldN>_DATASETS.txt` to `result/stats`, and the test-set predictions as `…_TEST.csv` to
  `result/tables`, as in cross-validation.
- To read another column than the revision's default, pass `--fold_column foldN`. A default path can be set as
  `[TRAIN] folds_csv` in the config. `--folds_csv` takes precedence over the random `--folds N` loop. The folds CSV is
  **not committed**: stage it locally and point the flag at it.

The same `--folds_csv` works with `--eval`: the model is scored on **only the chosen fold's `test` pages**, picked at
run time out of the `--eval_dir` folder (default `[EVAL] FOLDER_PAGES`). This is what confirms that a re-fine-tuned
model reaches the original accuracy on the *same held-out pages* (minus the removed ones):

```bash
python3 run.py --eval -rev v1.4 --folds_csv path/to/licensed_crossval_folds.csv --eval_dir path/to/pages
# evaluates ONLY the fold1 'test' pages found under path/to/pages
```

`--eval --folds_csv` is **single-model only**: the fold column is resolved from `-rev` (or an explicit
`--fold_column`). It is not combined with `--best`, because each best model was trained on a different fold, so an
ensemble has no single shared held-out test set.

## Evaluation

After the fine-tuned model is saved, `--eval` scores it and writes a table of the Top-N classes of the evaluation
pages. With the unchanged configuration it also draws a confusion matrix, and `--raw` adds the table of all class
probabilities:

```bash
python3 run.py --eval --raw
python3 run.py --eval -m './model/model_<your_model_number_code>'    # when you do not remember the run's settings
```

The evaluation pages come from `--eval_dir` (default `[EVAL] FOLDER_PAGES`): a folder with the same structure as the
training folder, one sub-folder per category. Within a training run, the test subset is the semi-randomly composed
10 % of the training folder, with the class proportions kept and spread evenly across the time span of the data.

An evaluation writes:

- `{stamp}_{n}_{model}_TOP-N_EVAL.csv`: the results on the evaluation pages, with the Top-N guesses (by default);
- `{stamp}_{n}_{model}_conf_mat_TOP-N.png`: the confusion matrix, also with Top-N guesses (by default);
- `{stamp}_{n}_{model}_EVAL_RAW.csv`: the raw probabilities of all classes (with `--raw`).

Their columns: [reference/outputs.md](reference/outputs.md). How to read a confusion matrix:
[evaluation/results.md](evaluation/results.md#reading-a-confusion-matrix). An evaluation run declares the LINDAT
dataset as a component of its paradata ([reference/licences.md](reference/licences.md)).

## Publishing a model on the Hub

When a model is trained and you are happy with its tests, it can be pushed to your own Hub repository.
`ImageClassifier.push_to_hub()` in [classifier.py](../classifier.py) uploads the model and its processor;
[run.py](../run.py) does not call it, and the branch-preparation lines for a push (with the `[HF]` section's
`repo_name` and `token`) are commented out in its `--hf` block, so a push needs a small edit there. First **rename the
trained model folder** after the `revision` value, without its dots (revision `v1.9.22` becomes the folder
`model_v1922`), and only then push.

> [!WARNING]
> Set `repo_name` to an empty repository of your own on the Hub, and create a token under **Settings → Access
> Tokens** of your Hugging Face account. Provide it as `HF_TOKEN` in the environment, which the Hugging Face libraries
> read; `[HF] token` stays blank in version control. Never commit a token.

## Re-running the `vX.4` evaluation

The LINDAT record (<http://hdl.handle.net/20.500.12800/1-6184>) holds the pages only in its category zips.
[stage_fold_pages.py](../data_scripts/stage_fold_pages.py) extracts one fold's test pages from them into a folder that
`--eval_dir` reads, with all eleven category sub-folders:

```bash
python3 data_scripts/stage_fold_pages.py --folds_csv licensed_crossval_folds.csv --fold fold1 \
    --zips path/to/lindat_zips --out stage/fold1
```

[eval_vX4_job.sh](../supplementary/scripts/eval_vX4_job.sh) is the SLURM job that runs the five `vX.4` models this
way, one after another: each on its fold's test pages (Top-1, and Top-3 with `--raw`), then on the 5,449-page set and
the sample pages, then as the `--best` ensemble. Its summary splits each top-1 result into pages in training, pages of
documents seen in training and unseen documents
([eval_by_document_overlap.py](../data_scripts/eval_by_document_overlap.py)). The weights are fetched first, on a node
with network access, and the job itself runs offline:

```bash
bash supplementary/scripts/eval_vX4_job.sh prefetch          # Hub revisions to model/, their commits, the zips checked
sbatch supplementary/scripts/eval_vX4_job.sh                 # result/tables, result/plots, result/stats/<stamp>_vX4_summary.md
bash supplementary/scripts/eval_vX4_job.sh collect <stamp>   # optional: a later run's files apart
```

The run of 2026-10-22 is in `result/` and described in [result/README.md](../result/README.md). A later run writes
into the same folders; `collect <stamp>` moves only the files that run wrote to `result/evaluations/<stamp>_vX4/` and
writes `PROVENANCE.md` beside them: the commit and release, the model revisions and their Hub commits, the folds CSV's
checksum and the LINDAT record, and per table the fold's test rows, the rows without a PNG name, and the pages
evaluated and not evaluated.
