from pathlib import Path
import pandas as pd
import os
import numpy as np
import random
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay,
)
from matplotlib import pyplot as plt
import time
import datetime
import re
import scipy.special


# ── filesystem helpers ────────────────────────────────────────────────────────

def directory_scraper(
    folder_path: Path,
    file_format: str = "png",
    file_list: list = None,
) -> list:
    """Recursively collect all files matching *file_format* under *folder_path*."""
    if file_list is None:
        file_list = []
    file_list += list(Path(folder_path).rglob(f"*.{file_format}"))
    print(
        f"[ {file_format.upper()} ] \tFrom directory {folder_path} "
        f"collected {len(file_list)} {file_format} files"
    )
    return file_list


# ── dataset collection ────────────────────────────────────────────────────────

def collect_images(
    directory: str,
    max_categ: int = None,
    ordered: bool = False,
) -> tuple:
    """
    Walk *directory* for category sub-directories and return
    (file_paths, one-hot_labels, category_names).

    Parameters
    ----------
    directory  : root folder whose immediate sub-directories are category names
    max_categ  : maximum images per category (None = unlimited)
    ordered    : if True, sort the returned file list alphabetically
    """
    categories = sorted(os.listdir(directory))
    print(f"Category input directories found: {categories}")

    total_files, total_labels, total_classes = [], [], []

    for category_idx, category in enumerate(categories):
        cat_dir = os.path.join(directory, category)
        if not os.path.isdir(cat_dir):
            continue

        all_category_files = [
            os.path.join(cat_dir, f) for f in os.listdir(cat_dir)
        ]

        if max_categ is not None and len(all_category_files) > max_categ:
            random.shuffle(all_category_files)
            all_category_files = all_category_files[:max_categ]

        print(f"Collected {len(all_category_files)} {category} category images")

        label_template = np.zeros(len(categories))
        label_template[category_idx] = 1.0

        total_files  += all_category_files
        total_labels += [label_template.copy() for _ in all_category_files]
        total_classes += [category_idx] * len(all_category_files)

    if ordered and total_files:
        paired = sorted(zip(total_files, total_labels, total_classes),
                        key=lambda x: str(x[0]))
        total_files, total_labels, total_classes = map(list, zip(*paired))

    label, count = np.unique(total_classes, return_counts=True)
    for label_id, label_count in zip(label, count):
        print(
            f"{categories[int(label_id)]}:\t{label_count}\t"
            f"{round(label_count / max(len(total_labels), 1) * 100, 2)}%"
        )

    return total_files, total_labels, categories


# ── prediction → DataFrame ────────────────────────────────────────────────────

def _parse_image_filename(image_file) -> tuple:
    """
    Extract (document_name, page_number) from an image filename.

    Supports separators ``-`` and ``_`` before the trailing page digits.
    Falls back to PAGE=1 when no trailing digits are found.

    Examples
    --------
    ``CTX193200994-24.png``  → (``CTX193200994``, 24)
    ``report_012.png``       → (``report``, 12)
    ``report_2021_003.png``  → (``report_2021``, 3)
    ``coverpage.png``        → (``coverpage``, 1)
    """
    stem = Path(str(image_file)).stem
    m = re.search(r"(\d+)$", stem)
    if m:
        page_num = int(m.group(1))
        document = stem[: m.start()].rstrip("-_")
    else:
        page_num = 1
        document = stem
    return document, page_num


def dataframe_results(
    test_images: list,
    test_predictions,
    categories: list,
    top_N: int,
    raw_scores: list = None,
) -> tuple:
    """
    Convert image prediction results into two pandas DataFrames.

    Accepted formats for *test_predictions*
    ----------------------------------------
    1. **int-index list** – ``[0, 2, 1, …]``
       One integer class index per image (typically top-1 predictions).

    2. **tuple list** – ``[[(idx, score), …], …]``
       One inner list per image, each containing ``(class_index, score)``
       pairs ordered best-first (top-N predictions with explicit scores).

    3. **score matrix** – 2-D ``numpy.ndarray`` *or* list of 1-D score arrays
       Raw per-class logits / probabilities; softmax is applied internally.

    Returns
    -------
    (results_df, raw_df)
        *results_df* always has ``FILE``, ``PAGE``, ``CLASS-1`` … ``CLASS-N``
        columns.  When ``top_N == 1`` a ``CATEGORY`` alias column is added and
        ``SCORE-1`` is omitted.
        *raw_df* is ``None`` unless *raw_scores* is provided.
    """
    print(
        f"Processing {len(test_images)} images with top_N={top_N} "
        f"predictions of {len(categories)} possible labels..."
    )

    n_categories = len(categories)
    effective_N  = max(1, min(top_N, n_categories))
    results: list = []

    # ── detect prediction format ──────────────────────────────────────────────
    def _is_int_format(preds):
        return (
            len(preds) > 0
            and isinstance(preds[0], (int, np.integer))
        )

    def _is_tuple_format(preds):
        return (
            len(preds) > 0
            and isinstance(preds[0], (list, tuple))
            and len(preds[0]) > 0
            and isinstance(preds[0][0], (list, tuple))
        )

    # ── format 1 : plain class-index list ────────────────────────────────────
    if _is_int_format(test_predictions):
        for img, idx in zip(test_images, test_predictions):
            doc, page = _parse_image_filename(img)
            label = (
                categories[idx] if 0 <= int(idx) < n_categories else str(idx)
            )
            results.append([doc, page, label])

        cols = ["FILE", "PAGE", "CLASS-1"]
        rdf  = pd.DataFrame(results, columns=cols)
        rdf["CATEGORY"] = rdf["CLASS-1"]

    # ── format 2 : list of (idx, score) tuples ────────────────────────────────
    elif _is_tuple_format(test_predictions):
        for img, pred_list in zip(test_images, test_predictions):
            doc, page = _parse_image_filename(img)
            row = [doc, page]
            # class columns first
            for j in range(effective_N):
                if j < len(pred_list):
                    idx = pred_list[j][0]
                    row.append(
                        categories[idx]
                        if 0 <= int(idx) < n_categories
                        else str(idx)
                    )
                else:
                    row.append("")
            # score columns second
            for j in range(effective_N):
                row.append(
                    round(float(pred_list[j][1]), 3)
                    if j < len(pred_list)
                    else 0.0
                )
            results.append(row)

        cols = (
            ["FILE", "PAGE"]
            + [f"CLASS-{j + 1}" for j in range(effective_N)]
            + [f"SCORE-{j + 1}" for j in range(effective_N)]
        )
        rdf = pd.DataFrame(results, columns=cols)

        if effective_N == 1:
            rdf = rdf.drop(columns=["SCORE-1"])
            rdf["CATEGORY"] = rdf["CLASS-1"]

    # ── format 3 : raw score matrix (original behaviour) ─────────────────────
    else:
        # Flatten to a list of 1-D per-image score arrays
        flat: list = []
        for batch in test_predictions:
            batch = np.atleast_2d(np.asarray(batch))
            for row in batch:
                flat.append(row)

        if not flat:
            print("[ERROR] Flat predictions list is empty.")
            return pd.DataFrame(), None

        try:
            preds_matrix = np.vstack(flat)
        except ValueError as exc:
            print(f"[CRITICAL ERROR] Could not stack predictions: {exc}")
            return pd.DataFrame(), None

        # Align lengths
        n_images = min(preds_matrix.shape[0], len(test_images))
        preds_matrix = preds_matrix[:n_images]
        images_slice = list(test_images)[:n_images]

        # Align category dimension
        n_scores = preds_matrix.shape[1]
        if n_scores > n_categories:
            preds_matrix = preds_matrix[:, :n_categories]
        elif n_scores < n_categories:
            effective_N = max(1, min(top_N, n_scores))

        for img, scores in zip(images_slice, preds_matrix):
            if np.all(scores == 0):
                continue
            doc, page = _parse_image_filename(img)

            try:
                probs = scipy.special.softmax(scores)
            except Exception:
                exp_s = np.exp(scores - np.max(scores))
                probs = exp_s / np.sum(exp_s)

            top_idx    = probs.argsort()[::-1][:effective_N]
            labels     = [categories[i] for i in top_idx]
            score_vals = [round(float(probs[i]), 3) for i in top_idx]
            results.append([doc, page] + labels + score_vals)

        if not results:
            print("[ERROR] No valid prediction rows found.")
            return pd.DataFrame(), None

        cols = (
            ["FILE", "PAGE"]
            + [f"CLASS-{j + 1}" for j in range(effective_N)]
            + [f"SCORE-{j + 1}" for j in range(effective_N)]
        )
        rdf = pd.DataFrame(results, columns=cols)

        if effective_N == 1:
            rdf = rdf.rename(columns={"CLASS-1": "CATEGORY"}).drop(
                columns=["SCORE-1"]
            )
            rdf.insert(2, "CLASS-1", rdf["CATEGORY"])

    # ── raw-scores DataFrame ──────────────────────────────────────────────────
    rawdf = None
    if raw_scores is not None:
        raws: list = []
        for img, scores in zip(test_images, raw_scores):
            doc, page = _parse_image_filename(img)
            raws.append([doc, page] + [round(float(s), 3) for s in scores])

        if raws:
            n_score_cols = len(raws[0]) - 2
            raw_col = ["FILE", "PAGE"] + list(categories[:n_score_cols])
            rawdf = pd.DataFrame(raws, columns=raw_col)

    print(
        f"Created results table with {len(rdf)} rows "
        f"and columns: {rdf.columns.tolist()}"
    )
    if rawdf is not None:
        print(f"Created RAW results table with shape {rawdf.shape}")

    return rdf, rawdf


# ── visualisation ─────────────────────────────────────────────────────────────

def confusion_plot(
    predictions,
    true_labels: list,
    categories: list,
    model_name: str,
    top_N: int = 1,
    output_dir: str = None,
) -> float:
    """
    Generate a normalised confusion matrix PNG and print a classification report.

    Parameters
    ----------
    predictions  : list of int  *or*  list of [(idx, score), …]
        For ``top_N == 1`` pass plain class indices.
        For ``top_N > 1`` pass per-image lists of ``(class_index, score)``
        pairs; the top-ranked index is used for the confusion matrix.
    true_labels  : list of int – ground-truth class indices
    categories   : list of str – ordered category names
    model_name   : str – embedded in the output filename and plot title
    top_N        : int – used in filename / title only
    output_dir   : directory that contains (or will contain) a ``plots/``
                   sub-directory.  Defaults to the current directory.

    Returns
    -------
    float : top-1 accuracy in percent
    """
    # ── extract top-1 predicted index from any input format ──────────────────
    if (
        predictions
        and isinstance(predictions[0], (list, tuple))
        and predictions[0]
        and isinstance(predictions[0][0], (list, tuple))
    ):
        # format: [[(idx, score), …], …]
        pred_indices = [int(p[0][0]) for p in predictions]
    else:
        pred_indices = [int(p) for p in predictions]

    # ── output directory ──────────────────────────────────────────────────────
    base_dir = Path(output_dir) if output_dir else Path(".")
    plot_dir = base_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    time_stamp = time.strftime("%Y%m%d-%H%M%S")
    plot_path  = plot_dir / f"{time_stamp}_{model_name}_TOP-{top_N}.png"

    # ── metrics ───────────────────────────────────────────────────────────────
    acc = accuracy_score(true_labels, pred_indices) * 100

    print("=" * 40)
    print(f"\t*\tAccuracy: {acc:.2f}%")
    print("=" * 40)
    print(
        classification_report(
            true_labels,
            pred_indices,
            target_names=categories,
            zero_division=0,
        )
    )

    # ── plot ──────────────────────────────────────────────────────────────────
    disp = ConfusionMatrixDisplay.from_predictions(
        true_labels,
        pred_indices,
        display_labels=categories,
        normalize="true",
        cmap="inferno",
    )
    disp.ax_.set_title(f"TOP {top_N} {model_name} – {acc:.2f}%")
    plt.savefig(plot_path, bbox_inches="tight", dpi=150)
    plt.close()

    print(f"Confusion matrix saved to {plot_path}")
    return acc


# ── misc ──────────────────────────────────────────────────────────────────────

def append_to_csv(df, filepath):
    """Append a DataFrame to a CSV file, or create it if it does not exist."""
    if not os.path.exists(filepath):
        df.to_csv(filepath, index=False, sep=",")
    else:
        df.to_csv(filepath, mode="a", header=False, index=False, sep=",")