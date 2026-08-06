import os
import re
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image, ImageFile
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

from atrium_document import canonical_doc_id

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 4221790634

#: Trailing PAGE NUMBER on a per-page image filename: `CTX01_0007.png` → doc `CTX01`,
#: page `0007`. Greedy on purpose (last separator wins), so `report_2021_003` is
#: document `report_2021` page 3 and not document `report` page 2021.
#:
#: Compiled once, module-level, and used by BOTH entry points. It used to be two
#: independent copies of the same expression — here and inline in `run.py`'s single-file
#: branch — which is the same drift risk that produced the eleven hand-rolled doc_id
#: derivations issue #13 was opened about (atrium-project#10, D3).
_PAGE_SUFFIX_RE = re.compile(r"(.*)[-_](\d+)$")


def doc_id_and_page(image_path) -> tuple[str, Optional[int]]:
    """Split one per-page image filename into (doc_id, page number).

    page-classification is the DOCUMENTED EXCEPTION to "call `canonical_doc_id()` and
    nothing else" (atrium-project#10, D3): its inputs are per-page IMAGES, and the page
    number is carried in the filename rather than in the content, so a doc_id derivation
    here has to strip a page label that `canonical_doc_id()` knows nothing about —
    `KNOWN_PIPELINE_SUFFIXES` deliberately lists no image extensions. So this COMPOSES
    the two halves rather than replacing either:

      1. drop the image extension (`.png`/`.jpg`/…), which only `Path.stem` can do here;
      2. split the trailing page label off — the genuinely pc-specific half;
      3. hand what remains to `canonical_doc_id()`, so a multi-dot document name resolves
         to the SAME id every other tool in the pipeline computes for it.

    Step 3 is the fix: `CTX01.scan_0007.png` used to yield doc_id `CTX01.scan`, while
    alto-postprocess/nlp-enrich key the very same document off `CTX01.alto.xml` /
    `CTX01.teitok.xml` → `CTX01`. A fork like that does not fail — it silently writes a
    second record under a key no other stage ever reads, and this tool's `page_categories`
    are then lost to the rest of the pipeline.

    Steps 1–2 run BEFORE step 3, not after, and the order is load-bearing: fed the whole
    filename, `canonical_doc_id()` finds no known suffix on a `.png`, falls through to its
    `split(".")[0]` fallback and returns `CTX01` — page label and all — so a
    canonical-first composition would collapse every page of a multi-dot document onto
    page 1, i.e. trade a doc_id fork for a worse page collision. Splitting first keeps the
    page and still lands on the canonical id.

    Returns `page = None` when the name carries no page label; the caller decides what to
    substitute (both call sites use page 1) and whether to warn.
    """
    stem = Path(image_path).stem
    match = _PAGE_SUFFIX_RE.match(stem)
    if match:
        return canonical_doc_id(match.group(1)), int(match.group(2))
    return canonical_doc_id(stem), None


# get list of all files in the folder and nested folders by file format
def directory_scraper(folder_path: Path, file_format: str = "png", file_list: list = None) -> list[str]:
    if file_list is None:
        file_list = []
    file_list += list(folder_path.rglob(f"*.{file_format}"))
    print(f"[ {file_format.upper()} ] \tFrom directory {folder_path} collected {len(file_list)} {file_format} files")
    return file_list


def dataframe_results(
    test_images: list, test_predictions: list, categories: list, top_N: int, raw_scores: list = None
) -> (pd.DataFrame, pd.DataFrame):
    results = []
    raws = []

    for image_file, predict_scores in zip(test_images, test_predictions):
        # One shared derivation for both CLI shapes — see doc_id_and_page (D3).
        document, page_num = doc_id_and_page(image_file)

        if page_num is None:
            # Fallback if file doesn't match the format (e.g., "cover_page.png")
            warnings.warn(
                f"Ambiguous filename without page suffix: '{Path(image_file).stem}'. "
                f"Assigning the canonical doc_id '{document}' as FILE and '1' as PAGE.",
                UserWarning,
            )
            page_num = 1

        labels = [categories[i[0]] for i in predict_scores] if top_N > 1 else [categories[predict_scores]]
        scores = [round(i[1], 3) for i in predict_scores] if top_N > 1 else [round(predict_scores, 3)]

        res = [document, page_num] + labels + scores
        results.append(res)
        if raw_scores is not None:
            raws.append([document, page_num])

    col = ["FILE", "PAGE"] + [f"CLASS-{j + 1}" for j in range(top_N)] + [f"SCORE-{j + 1}" for j in range(top_N)]
    rdf = pd.DataFrame(results, columns=col)

    if top_N == 1:
        # Keep CLASS-1 / SCORE-1 naming so downstream tools (e.g. averaging.py)
        # can always expect consistent column names regardless of top_N value.
        # A CATEGORY alias column is added for human-readable CSV output.
        rdf["CATEGORY"] = rdf["CLASS-1"]
        # Do not drop SCORE-1 here to prevent losing the confidence metric needed for atrium_document JSON.

    rawdf = None
    if raw_scores is not None:
        col = ["FILE", "PAGE"]
        rawdf = pd.DataFrame(raws, columns=col)
        # Ensure raw_scores is a numpy array before rounding
        raw_weights = np.array(raw_scores).round(3)
        rawdf[categories] = raw_weights

    return rdf, rawdf


def collect_images(directory: str, ordered: bool = True) -> (list, list, list):
    print(f"Collecting images from {directory}...")

    categories = sorted(os.listdir(directory))
    print(f"Category input directories found: {categories}")

    total_files, total_labels, total_classes = [], [], []
    for category_idx, category in enumerate(categories):
        all_category_files = os.listdir(os.path.join(directory, category))

        total_files += [os.path.join(directory, category, file) for file in all_category_files]

        label_template = np.zeros(len(categories))
        label_template[category_idx] = 1

        total_labels += [label_template] * len(all_category_files)
        total_classes += [category_idx] * len(all_category_files)

    label, count = np.unique(total_classes, return_counts=True)
    for label_id, label_count in zip(label, count):
        print(f"{categories[int(label_id)]}:\t{label_count}\t{round(label_count / len(total_labels) * 100, 2)}%")

    if ordered:
        # sorting in alphabetical order to ensure consistent order
        sorted_pairs = sorted(zip(total_files, total_labels), key=lambda pair: pair[0])
        total_files, total_labels = zip(*sorted_pairs)
        total_files, total_labels = list(total_files), list(total_labels)

    return total_files, total_labels, categories


def confusion_plot(
    predictions: list, trues: list, categories: list, out_model: str, top_N: int = 1, output_dir: str = None
):
    single_pred = []
    correct = 0
    for j, pred_scores in enumerate(predictions):
        true_class = trues[j]

        if top_N > 1:
            classes = [i[0] for i in pred_scores]

            if true_class in classes:
                correct += 1
                single_pred.append(true_class)
            else:
                single_pred.append(classes[0])

        else:
            single_pred.append(pred_scores)
            if pred_scores == true_class:
                correct += 1

    print("=" * 40)
    print("Percentage correct: ", round(100 * correct / len(trues), 2))
    print("=" * 40)
    print(classification_report(trues, single_pred, target_names=categories, digits=4, zero_division=0))

    # Confusion matrix display and normalized output
    disp = ConfusionMatrixDisplay.from_predictions(
        trues, single_pred, cmap="inferno", normalize="true", display_labels=np.array(categories)
    )

    short_labels = [f"{label[0]}{label.split('_')[-1][0] if '_' in label else ''}" for label in disp.display_labels]

    print(f"\t{' '.join(disp.display_labels)}")
    for ir, row in enumerate(disp.confusion_matrix):
        print(f"{disp.display_labels[ir]}\t{'   '.join([str(val) if val > 0 else ' -- ' for val in np.round(row, 2)])}")

    # Customize x-axis tick labels to show only the first character of each label
    tick_positions = disp.ax_.get_xticks()
    disp.ax_.set_xticks(tick_positions)
    disp.ax_.set_xticklabels(short_labels)

    time_stamp = time.strftime("%Y%m%d-%H%M")
    disp.ax_.set_title(f"TOP-{top_N} Confusion matrix {out_model} - {round(100 * correct / len(trues), 2)}%")
    out = (
        f"{output_dir if output_dir else 'result'}/plots/{time_stamp}_{len(trues)}_{out_model}_conf_mat_TOP-{top_N}.png"
    )
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.close()
