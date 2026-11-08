# Data preparation

For anyone who builds a dataset of pages: how to turn PDF documents into page images, annotate them, sort them into
the category folders that training and evaluation read, and keep the dataset clean. Classifying pages needs only the
first step; training needs all of them ([training.md](training.md)).

On this page:
[The scripts](#the-scripts) ·
[PDF to PNG](#pdf-to-png) ·
[PNG pages annotation](#png-pages-annotation) ·
[PNG pages routing for training](#png-pages-routing-for-training) ·
[Dataset maintenance](#dataset-maintenance) ·
[Controlled defects for the page-condition experiment](#controlled-defects-for-the-page-condition-experiment)

## The scripts

The [data_scripts](../data_scripts) folder has scripts for the whole process. The `.sh` scripts are for Unix and the
`.bat` scripts for Windows; their functionality is the same. All of them take command-line arguments: `--help` on
Unix, `/?` on Windows, lists the options.

On **Windows**, install two programs before converting PDF documents to PNG images:

- [ImageMagick](https://imagemagick.org/script/download.php#windows): download and install the latest version;
- [Ghostscript](https://www.ghostscript.com/releases/gsdnld.html): download and install the latest version (32 or
  64-bit); it is AGPL-licensed.

## PDF to PNG

The source PDF documents must be converted to page-specific PNG images before any processing. Copy the converter
script to the folder with the PDF documents, or call it from anywhere with the folder as an argument:

```bash
cp /local/folder/for/this/project/data_scripts/unix/pdf2png.sh /full/path/to/your/folder/with/pdf/files     # Unix
```

```bat
move \local\folder\for\this\project\data_scripts\windows\pdf2png.bat \full\path\to\your\folder\with\pdf\files
```

Run it with your preferences: the source folder, the output format and the DPI. The source PDFs are **kept** by
default; add `--delete` (Unix) or `/delete` (Windows) to remove each PDF once it has been converted. On Unix, `.PDF`
files are found as well as `.pdf`, and the conversion runs in parallel on all CPU cores.

```bash
/path/to/pdf2png.sh --dir /folder/with/pdfs --format png --dpi 300                # Unix; --output DIR for another root
```

```bat
\path\to\pdf2png.bat /d \folder\with\pdfs /f png /r 300
```

The Windows batch version runs one file at a time; for parallel processing on Windows, run the Unix script under WSL.
The defaults are PNG at 300 dpi, which is how the training pages were made.

Each PDF gets its own folder of pages. On Unix:

```text
/full/path/to/your/folder/with/pdf/files
├── PdfFile1Name
│   ├── PdfFile1Name-001.png
│   ├── PdfFile1Name-002.png
│   └── ...
├── PdfFile2Name
│   ├── PdfFile2Name-01.png
│   ├── PDFFile2Name-02.png
│   └── ...
├── PdfFile3Name
│   └── PdfFile3Name-1.png
├── PdfFile4Name
└── ...
```

The page numbers are padded with zeros on the left to the length of the last page number of each PDF; `pdftoppm`,
used on Unix, does this itself. ImageMagick's `convert`, used on Windows, does **not** pad them:

```text
\full\path\to\your\folder\with\pdf\files
├── PdfFile1Name
│   ├── PdfFile1Name-1.png
│   ├── PdfFile1Name-2.png
│   └── ...
├── PdfFile2Name
│   ├── PdfFile2Name-1.png
│   ├── PDFFile2Name-2.png
│   └── ...
├── PdfFile3Name
│   └── PdfFile3Name-1.png
├── PdfFile4Name
└── ...
```

### Gathering one-page documents

Optionally, [move_single.sh](../data_scripts/unix/move_single.sh) or
[move_single.bat](../data_scripts/windows/move_single.bat) moves the PNG files of every folder that holds a single
page into one common folder of one-pagers. By default the scripts take `onepagers` as the back-off folder for PDF
document names without a folder of their own among the converted pages. Set the folders with `--source` and
`--target` (`/s` and `/t` on Windows), and add `--dry-run` to preview the files that would be moved.

```bash
cp /local/folder/for/this/project/atrium-page-classification/data_scripts/unix/move_single.sh /full/path/to/your/folder/with/pdf/files
cd /full/path/to/your/folder/with/pdf/files
./move_single.sh
```

```bat
move \local\folder\for\this\project\atrium-page-classification\data_scripts\windows\move_single.bat \full\path\to\your\folder\with\pdf\files
cd \full\path\to\your\folder\with\pdf\files
move_single.bat
```

The move is for convenience in the annotation that follows; [sort.sh](../data_scripts/unix/sort.sh) and
[sort.bat](../data_scripts/windows/sort.bat) take it into account.

## PNG pages annotation

The page images are used to make the annotated gold data. It takes a lot of time to collect at least several hundred
examples per category.

Prepare a CSV table with a header row and at least these three columns:

- **FILE**: the name of the PDF document the page comes from;
- **PAGE**: the number of the page (plain, such as `8`; a zero-padded `08` is accepted too);
- **CLASS**: the label of the category.

Extra columns (a title, a DOI, notes) are ignored. `sort.sh` finds the three columns by their header names, in any
order; `sort.bat` reads them by position, so on Windows keep `FILE`, `PAGE`, `CLASS` as the first three columns.

```text
FILE,PAGE,CLASS
PdfFile1Name,1,Label1
PdfFile2Name,9,Label1
PdfFile1Name,11,Label3
...
```

> [!TIP]
> Make the categories equal in size if you can, so that the model is not biased towards the over-represented
> labels.

On Windows, MS Excel is **not** recommended for writing CSV tables; a free alternative is Apache's
[OpenOffice](https://www.openoffice.org/download/). On Unix, LibreOffice Calc writes a correct comma-separated table.

## PNG pages routing for training

[sort.sh](../data_scripts/unix/sort.sh) or [sort.bat](../data_scripts/windows/sort.bat) copies the annotated pages
from the source folder to a training folder in which each category has its own sub-folder. This division of the
images is the gold data for training and evaluation. It does not matter which folder you start the script from:
give the paths with `-c` for the annotated CSV table, `-i` for the folder of converted pages and `-o` for the target
training folder.

```bash
sort.sh -c annotations.csv -i /path/to/pages -o /path/to/train_dir --move          # Unix
```

```bat
sort.bat /c annotations.csv /i \path\to\pages /o \path\to\train_dir
```

`--dry-run` (Unix) or `/n` (Windows) previews the distribution; leave out `--move` (`/move`) to copy the files instead
of moving them. A label folder is created only once a page has been found for it, and a row whose `PAGE` is not a
number, or whose `CLASS` is empty or contains `/` or `\`, is reported and skipped.

The result is a folder of label sub-folders holding the pages of many documents. On Unix:

```text
/full/path/to/your/folder/with/train/pages
├── Label1
│   ├── PdfFileAName-00N.png
│   ├── PdfFileBName-0M.png
│   └── ...
├── Label2
├── Label3
├── Label4
└── ...
```

On Windows the page numbers are not padded (`PdfFileAName-N.png`, `PdfFileBName-M.png`).

The sorting helps to catch mislabelled pages before training; accurate annotation directly affects the model's
performance. Before training, set the path of this folder in the `[TRAIN]` section of
[setup/config.txt](../setup/config.txt), and make sure the label folder names contain no special characters such as
spaces, tabs or line breaks ([configuration.md](configuration.md#train)).

> [!TIP]
> If some labels are significantly over-represented, lower `max_categ` in `[TRAIN]`, the maximum number of pages
> taken per category. Set it above the size of the largest category to use **all** the pages.

From here you can train or evaluate a model ([training.md](training.md)). The round trip of annotation, from
PDFs to a labelled training set and corrections back into it, is drawn as workflow W12 on the hub's
[Pipelines](https://ufal.github.io/atrium-project/pipelines/) page.

## Dataset maintenance

Once the dataset is sorted, two scripts in [supplementary/scripts](../supplementary/scripts) help to clean it or to
prototype with it:

* **Filtering.** If you move mislabelled images between label folders by hand, or delete them, re-sync your CSV with
  the folders:

  ```bash
  python3 supplementary/scripts/filtering.py -i annotations.csv -d train_dir
  ```

  A row whose page now sits in exactly one other label folder is **relabelled** to it, and a row whose page is gone
  (or sits in several other folders) is removed and reported. The result is written to `annotations_filtered.csv`
  next to the input (`-o` for another path; `--no-relabel` only drops rows). The label column is `CLASS`, or
  `CLASS-1` for a classifier result table.
* **Downscaling.** To prototype quickly with lower-resolution images:

  ```bash
  python3 supplementary/scripts/downscale.py --src train_dir --dst small_train_dir --scale 50
  ```

  It shrinks the images and keeps the whole category-folder hierarchy.

## Controlled defects for the page-condition experiment

[degrade_pages.py](../data_scripts/degrade_pages.py) makes copies of clean pages with one known defect each, for the
decision-model experiment `decide_pages.py --task b4`
([reference/cli.md](reference/cli.md#decision-models-on-page-images-decide_pagespy)). Classifying and training do not
need it. No page of the dataset is labelled for its condition. A page and its degraded copy differ by the defect alone,
so the change in the model's answer between them measures the question without annotators.

```bash
python3 data_scripts/degrade_pages.py --dir small_data_samples --per-label 2 --out stage/degraded
python3 decide_pages.py --task b4 --degradations stage/degraded/degradations.csv
```

For each page it writes the original, re-saved as PNG so that both halves of a pair go through the same encoding, and
one copy per defect and level, from mild (1) to strong (3):

| Defect         | Level 1 / 2 / 3                                                                                   |
|----------------|---------------------------------------------------------------------------------------------------|
| `skewed`       | rotated by 2° / 4° / 8°, the direction drawn per copy; the corners filled with the paper's colour |
| `yellowed`     | multiplied by an aged-paper tint, light / medium / strong                                         |
| `faint_ink`    | blended towards the paper's colour by 35 % / 55 % / 75 %, so the ink fades and the paper stays    |
| `blurred`      | a Gaussian blur of 1 / 2 / 3.5 px per 1,000 px of the long side                                   |
| `cropped`      | 8 % / 15 % / 25 % cut from one side, the side drawn per copy                                      |
| `uneven_light` | a shadow from one side, down to 70 % / 50 % / 30 % of the brightness                              |
| `rotated_text` | no levels: one copy turned a quarter clockwise, one upside down, one a quarter counter-clockwise  |

The paper's colour is the mean colour of the four corners. Stains, tears, water damage, folds, holes, foxing,
bleed-through and smudges are not faked: a filter would teach nothing about the real thing.

| Flag          | Default   | What it does                                                                                      |
|---------------|-----------|---------------------------------------------------------------------------------------------------|
| `--dir`       | required  | a folder with one sub-folder per category                                                         |
| `--out`       | required  | where the copies and the manifest go; a folder that already holds a manifest is refused           |
| `--per-label` | all pages | at most N source pages per category, drawn with `--seed`                                          |
| `--seed`      | `420`     | the seed of `--per-label` and of every drawn direction and side, so a run can be repeated exactly |
| `--defects`   | `all`     | the defects to make, comma-separated, from the table above                                        |
| `--levels`    | `1,2,3`   | the levels to make, comma-separated; `rotated_text` ignores them                                  |

The copies keep the `<out>/<CATEGORY>/` layout, as `<page>__<defect>_<level>.png` and
`<page>__rotated_text_<orientation>.png`. The manifest, `<out>/degradations.csv`, has one row per image written:

| Column        | What it holds                                                                                                                                                 |
|---------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `image`       | the image, relative to the manifest's folder                                                                                                                  |
| `source`      | the clean page it was made from, relative to the manifest's folder                                                                                            |
| `label`       | the category folder of the page                                                                                                                               |
| `defect`      | the defect; empty for the original                                                                                                                            |
| `level`       | `1`–`3`; `0` for the original and `1` for `rotated_text`                                                                                                      |
| `orientation` | the way up of the page's text: `upright`, or the turn of a `rotated_text` copy (`cw90`, `upside_down`, `ccw90`)                                               |
| `params`      | what was done, as JSON (the angle, the tint, the blur radius, the side and share cut, the shadow's side and floor); empty for the original and `rotated_text` |

`decide_pages.py --degradations <manifest>`, with `b3`, `b4` or `triage`, takes its pages from the manifest instead of a
folder; a path in it may also be absolute. Every image is named by its source page's `FILE` and `PAGE`, so a copy pairs
with its original. Its row carries `VARIANT` (the defect, or `original`), `LEVEL` and `SOURCE`, and in `b4` and `triage`
also `GOLD_orientation`. The summary of `b4` and `triage` then reports, per injected defect and image size, the defect's
mean probability on the copies and on their originals, the paired change, how often it was detected (P ≥ 0.5), and the
false alarms on the originals. It adds the rank correlation of the level with the overall damage score, and the
orientation accuracy on pages whose category has text. `--per-label` is refused with a manifest, because it could split
a copy from its original: sample with `degrade_pages.py --per-label` instead.
