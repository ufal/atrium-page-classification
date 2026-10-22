# Results

What the files in this folder are, what each one measures, and what was wrong with the results kept here before
2026-10-22 ([atrium-project#15](https://github.com/ufal/atrium-project/issues/15)).

## The `vX.4` evaluation (2026-10-22)

Each of the five published `vX.4` models was scored on the **test partition of the fold it was trained on**, as
published in the LINDAT record [1-6184](http://hdl.handle.net/20.500.12800/1-6184).

| Revision | Base model                               | Fold | Test pages | Top-1 % (95 % CI)     | Top-3 % | Top-1 %, documents seen in training (pages) | Top-1 %, unseen documents (pages) |
|----------|------------------------------------------|------|-----------:|-----------------------|--------:|---------------------------------------------|-----------------------------------|
| `v1.4`   | `timm/tf_efficientnetv2_m.in21k_ft_in1k` | 1    |      4,795 | 95.87 (95.27–96.40)   |   99.90 | 96.70 (1,181)                               | 95.60 (3,614)                     |
| `v4.4`   | `timm/regnety_160.swag_ft_in1k`          | 1    |      4,795 | 95.81 (95.20–96.34)   |   99.79 | 97.71 (1,181)                               | 95.19 (3,614)                     |
| `v3.4`   | `google/vit-base-patch16-384`            | 2    |      4,792 | 95.49 (94.87–96.04)   |   99.77 | 97.16 (1,164)                               | 94.96 (3,628)                     |
| `v5.4`   | `google/vit-large-patch16-384`           | 2    |      4,792 | 95.58 (94.96–96.12)   |   99.75 | 96.48 (1,164)                               | 95.29 (3,628)                     |
| `v2.4`   | `google/vit-base-patch16-224`            | 5    |      4,791 | 94.64 (93.96–95.24)   |   99.81 | 96.96 (1,183)                               | 93.87 (3,608)                     |

- **The split is by page.** About a quarter of each fold's test pages come from documents that have other pages in
  training ("seen"); their accuracy is 1–3 points higher than that of pages from unseen documents. No test page is
  a training page (checked with [eval_by_document_overlap.py](../data_scripts/eval_by_document_overlap.py)).
- **Models on the same pages do not differ significantly.** On the fold 1 pages, `v1.4` and `v4.4` differ in
  correctness on 121 pages, 62 to 59 (exact McNemar p = 0.86); on fold 2, `v3.4` and `v5.4` on 98, 47 to 51
  (p = 0.76). `v2.4` is about one point lower than the rest, but it is the only model on fold 5, so the model and the
  fold cannot be told apart.
- **Where the errors are.** `TEXT_T` has the lowest recall (87–92 %, mostly taken for `TEXT`), then `TEXT_P`,
  `LINE_HW` and `DRAW` (91–96 %; `LINE_HW` → `LINE_T` in 4–9 % of its pages). Top-3 contains the true category for
  99.75–99.90 % of pages.

**Provenance.**
- **Code:** page-classification `v1.9.8-beta` at
  [`f14bc4b`](https://github.com/ufal/atrium-page-classification/commit/f14bc4b8c953e4e5c0f6149225c101e20de52a21), run
  as one SLURM job ([eval_vX4_job.sh](../supplementary/scripts/eval_vX4_job.sh), job 7271549, 2026-10-22).
- **Models:** `ufal/vit-historical-page` at the Hub commits in
  [20261022-0901_vX4_hub_revisions.csv](stats/20261022-0901_vX4_hub_revisions.csv).
- **Split:** the `fold1`/`fold2`/`fold5` columns of the record's `licensed_crossval_folds.csv`. These are the folds
  the `vX.4` models were trained on (`model_registry.REVISION_BEST_FOLDS`): the named test pages are the same as in
  the training-time CSV.
- **Excluded rows:** the published CSV also has rows with no file name, 28, 31 and 32 of them in the fold 1, 2 and
  5 test cells. They name no page and were left out.
- **Pages:** extracted from the record's category zips
  ([stage_fold_pages.py](../data_scripts/stage_fold_pages.py)). The true category is the folder the record puts the
  page in. No page was unreadable or skipped.
- **The run's own record:** [20261022-0909_vX4_summary.md](stats/20261022-0909_vX4_summary.md), with its steps,
  tables and output list beside it in [stats/](stats).

| Files (in [tables/](tables) and [plots/](plots))                                     | What they are                                                                                                                                                        |
|--------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `20261022-*_479[1-5]_model_v?4_TOP-1_EVAL.csv`, `…_TOP-3_EVAL.csv`, `…_EVAL_RAW.csv` | the per-fold test results above: top-1, top-3 with scores, all 11 scores; `TRUE` is the dataset's category                                                           |
| `20261022-*_479[1-5]_model_v?4_conf_mat_TOP-1.png`, `…_TOP-3.png`                    | their confusion matrices                                                                                                                                             |
| `20261022-*_5449_model_v?4_TOP-1_EVAL.csv`                                           | the 5,449-page set (below); kept for the by-group report, not as an accuracy                                                                                         |
| `20261022-1218_BEST_5_models_*.csv`, `20261022-1218_v?4_TOP-3.csv`                   | the `--best` ensemble and its five members on [small_data_samples](../small_data_samples), a demonstration of the output format: those pages are not evaluation data |
| [paradata/](paradata)                                                                | one log per kept run                                                                                                                                                 |

The 229 pages of `small_data_samples` were scored too (70–79 % top-1), and those tables were removed. The pages come
from theses, slides, our own papers and web pages: they are outside the training data, and their folders were never
checked as evaluation labels.

## How the folds were chosen

The `vX.3` sweep of 2025 trained every base model on five folds and kept, per base model, the fold with the highest
top-1 on the 5,449-page set; a tie went to the lowest fold. The figures are in
[model_accuracies_top1.csv](https://github.com/ufal/atrium-page-classification/blob/789ff043dc5e81604e0b6cbb4a1003a730a200d2/result/stats/model_accuracies_top1.csv).
These are the folds of `REVISION_BEST_FOLDS`, and the `vX.4` models were retrained on them:

- EfficientNetV2-M: fold 1 (98.83);
- RegNetY-160: fold 1 (99.16);
- ViT-B/16-224: fold 5 (98.79);
- ViT-B/16-384: fold 2 (98.92, tied with folds 4 and 5);
- ViT-L/16-384: fold 2 (99.12).

So the folds were chosen on a set that is mostly training pages (next section). The `vX.4` test partitions played
no part in any choice.

## The 5,449-page set

[unused_entries_da5_f12345.csv](stats/unused_entries_da5_f12345.csv) was selected in 2025 by
[unused.sh](stats/unused.sh) as the pages that no fold used for training, and was manually checked. The script
compared file names as text: it missed zero-padded page numbers and ids with a letter, so most of the set is
training data. The script is fixed since `v1.9.6-beta`. Split by the `vX.4` model's own fold:

| Revision | Top-1 % on all 5,449 | In training      | Seen document    | Unseen document | Not in the licensed CSV |
|----------|---------------------:|------------------|------------------|-----------------|-------------------------|
| `v1.4`   |                98.26 | 4,250 (99.06 %)  | 1,020 (97.06 %)  | 119 (97.48 %)   | 60 (63.33 %)            |
| `v2.4`   |                98.57 | 4,235 (99.43 %)  | 1,025 (96.98 %)  | 128 (92.97 %)   | 61 (77.05 %)            |
| `v3.4`   |                98.79 | 4,230 (99.46 %)  | 1,037 (97.01 %)  | 121 (95.04 %)   | 61 (90.16 %)            |
| `v4.4`   |                99.16 | 4,250 (99.48 %)  | 1,020 (98.14 %)  | 119 (95.80 %)   | 60 (100.00 %)           |
| `v5.4`   |                98.94 | 4,230 (99.62 %)  | 1,037 (97.30 %)  | 121 (95.04 %)   | 61 (86.89 %)            |

The figures on all 5,449 pages are **not held-out accuracy**. That `v4.4` reaches 99.16 % here, the figure published
for `v4.3`, is a coincidence of the set, not a reproduction. These pages are stored as JPEG; the last column counts
the pages that are not in the public record's folds CSV, so no `vX.4` model trained on them.

## Earlier results and their problems

The results kept here until 2026-10-22 are preserved at
[`789ff04/result`](https://github.com/ufal/atrium-page-classification/tree/789ff043dc5e81604e0b6cbb4a1003a730a200d2/result)
(the last commit with them). They are:
- `*_5449_model_v?3_*` (2025-10-19/20), the `vX.3` models on the 5,449-page set;
- `*_model_v?3a5_*` (2025-10-04), the five-fold averaged models;
- `*_115_*` and `*_BEST_6_*`, sample-page runs;
- `20260530-*`, `vX.3` on `small_data_samples`;
- the sweep's `stats/model_accuracies_top1.csv` / `top3.csv` and older one-off tables.

What was wrong with them:

1. **They were not held-out results.** Every `vX.3` figure, including the README's "Best_Acc" column and the `vX.3`
   rows of [model_accuracies_new.csv](../model_accuracies_new.csv), was measured on the 5,449-page set. For each fold, about
   4,300 of those pages are training pages.
2. **The published figures and the stored files disagree.**
   - The README gave `v4.3` 99.16 %, the sweep's figure for RegNetY fold 1. No stored table has it: the stored
     `v4.3` table and its confusion matrix (the one the README showed next to 99.16 %) give 98.92 %.
   - The README gave `v1.3` 98.83 % (the sweep's figure, also given by a run of 2025-10-19), next to the matrix
     of a run of 2025-10-20 that gives 96.81 % on the same pages.
3. **The 96.81 % run is likely a preprocessing error.** `run.py -rev` kept the config's RegNetY base unless `-b`
   named another, and the image transforms are built from the base, so an EfficientNet revision was fed RegNetY's
   normalisation. The 96.81 % is consistent with that; it was fixed in `v1.9.8-beta`.
4. **The averaged `a5` models (96.83–99.01 %) had seen almost every page.** Their five folds' training partitions
   together cover 5,320 of the 5,449 pages (all but 129 of the 48,498 in the splits).
5. **A mislabelled table.** `20260723-1659_model_v43_TOP-3.csv` (added and removed on 2026-10-22) held `v4.4`'s
   predictions: the same top-3 classes as `v4.4` on all 229 pages, against the earlier `v4.3` table that differs.

Not a problem, for the record: no two models' tables are identical. The 2025-10-20 batch is not a set of shifted
labels, as atrium-project#15's brief first supposed.

Still here from before 2026-10-22:
- The `v*.2` demonstration tables of 2025-07 (`*_model_v[1-5]2010*`). Their evaluation set's overlap with training
  was never checked, so they are format examples, not accuracy.
- [stats/](stats): the five `vX.3` fold splits (`*_model_v93_FOLD_*_DATASETS.txt`), the 5,449-page list and the
  scripts that read them.
- `20251020-1849_5449_model_v43_TOP-1_EVAL.csv`, which `tests/test_eval_by_document_overlap.py` reads.
- [ARUB_averaged_SHORT.csv](ARUB_averaged_SHORT.csv) and [ARUP_averaged_SHORT.csv](ARUP_averaged_SHORT.csv), `vX.3`
  ensemble predictions on two unannotated collections (no accuracy).

## License of these files

Tables with a `TRUE` column carry the dataset's annotation, which is under CC BY-NC 4.0 like the
[LINDAT record](http://hdl.handle.net/20.500.12800/1-6184); the code that made them is MIT. The paradata of the
2026-10-22 run says MIT, because `--eval` did not declare the dataset; from `v1.9.9-beta` it does.
