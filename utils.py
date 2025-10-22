from pathlib import Path
import pandas as pd
import os
import numpy as np
import random
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
import time
import datetime
import re
import scipy.special

# get list of all files in the folder and nested folders by file format
def directory_scraper(folder_path: Path, file_format: str = "png", file_list: list = None) -> list[str]:
    if file_list is None:
        file_list = []
    file_list += list(folder_path.rglob(f"*.{file_format}"))
    print(f"[ {file_format.upper()} ] \tFrom directory {folder_path} collected {len(file_list)} {file_format} files")
    return file_list


def dataframe_results(
        test_images: list,
        test_predictions: list,
        categories: list,
        top_N: int,
        raw_scores: list = None
) -> (pd.DataFrame, pd.DataFrame):
    """
    Processes image prediction results into two pandas DataFrames:
    one for formatted top-N predictions and another for raw scores.

    FIXED VERSION:
      - Ensures test_predictions is flattened into an array of per-image scores.
      - Safely handles cases where the number of prediction values does not match
        the number of categories.
    """
    print(f"Processing {len(test_images)} images with top_N={top_N} predictions "
          f"of {len(categories)} possible labels...")

    # --- FIX 1: Flatten test_predictions into a single array of per-image scores ---
    # Convert list of batch arrays into a single list of per-image score arrays
    flat_predictions = []
    for batch_preds in test_predictions:
        # Assuming batch_preds is a numpy array or torch tensor of shape (batch_size, n_classes)
        # Convert to numpy array and ensure it's 2D
        batch_preds = np.atleast_2d(np.asarray(batch_preds))
        for row in batch_preds:
            flat_predictions.append(row)

    if not flat_predictions:
        print("[ERROR] Flat predictions list is empty.")
        return pd.DataFrame(), None


    # Vertically stack all per-image score arrays
    try:
        preds = np.vstack(flat_predictions)
    except ValueError as e:
        print(f"[CRITICAL ERROR] Could not stack flat predictions: {e}")
        return pd.DataFrame(), None

    # Check if the number of images in predictions matches test_images
    if preds.shape[0] != len(test_images):
        print(f"[ERROR] Mismatch: {len(test_images)} images, but {preds.shape[0]} predictions.")
        # If mismatch, truncate to the minimum count, which is usually the length of the images list
        min_len = min(preds.shape[0], len(test_images))
        preds = preds[:min_len]
        test_images = test_images[:min_len]
        # raw_scores would also need truncation if it was used, but we'll assume it's aligned or None

    n_images, n_raw_scores = preds.shape
    n_categories = len(categories)
    print(f"Final prediction array shape: {preds.shape}")

    # --- FIX 2: Check and align categories and scores ---
    # If the number of raw scores (columns) is greater than the number of categories, truncate the scores.
    if n_raw_scores > n_categories:
        # print(
        #     f"[WARN] {n_raw_scores} prediction values but only {n_categories} categories. Truncating prediction scores to match categories.")
        preds = preds[:, :n_categories]
    # If the number of raw scores is less than the number of categories, truncate the categories list.
    elif n_raw_scores < n_categories:
        # print(f"[WARN] {n_categories} categories but only {n_raw_scores} prediction scores. Truncating category list.")
        # categories = categories[:n_raw_scores]
        n_categories = n_raw_scores

    # Re-evaluate n_images, n_categories after potential truncation
    if preds.ndim == 2:
        n_images, n_categories = preds.shape
    else:  # Should not happen after np.vstack
        print("[ERROR] Predictions are not a 2D array.")
        return pd.DataFrame(), None

    results, raws = [], []
    valid_rows = 0

    for image_file, scores in zip(test_images, preds):
        # Skip fully zero/padded rows
        if np.all(scores == 0) and not np.any(scores):
            continue

        # --- Robust filename parsing ---
        image_name = Path(image_file).stem
        match = re.search(r'(\d+)$', image_name)
        if match:
            page_num = int(match.group(1))
            document = image_name[:match.start()].rstrip("-_")
        else:
            page_num = None
            document = image_name

        # --- Stable softmax normalization ---
        # scores should already be clipped to n_categories from the checks above
        valid_scores = scores
        # Use scipy.special.softmax for robustness if available, otherwise the custom one
        try:
            probs = scipy.special.softmax(valid_scores)
        except AttributeError:
            exp_scores = np.exp(valid_scores - np.max(valid_scores))
            probs = exp_scores / np.sum(exp_scores)
        except FloatingPointError as e:
            print(f"[WARN] Floating point error during softmax: {e}. Using uniform distribution.")
            probs = np.full_like(valid_scores, 1.0 / len(valid_scores))

        # --- Top-N selection ---
        top_N = max(1, min(top_N, n_categories))
        top_idx = probs.argsort()[::-1][:top_N]
        labels = [categories[i] for i in top_idx]
        score_vals = [round(float(probs[i]), 3) for i in top_idx]

        results.append([document, page_num] + labels + score_vals)
        valid_rows += 1

        # --- Raw scores ---
        if raw_scores is not None:
            raws.append([document, page_num] + [round(float(s), 3) for s in valid_scores])

    # ... (rest of the function for DataFrame construction remains the same) ...

    if valid_rows == 0:
        print("[ERROR] No valid prediction rows found — check input shapes or category count.")
        return pd.DataFrame(), None

    # --- Construct formatted results DataFrame ---
    col = ["FILE", "PAGE"] + [f"CLASS-{j + 1}" for j in range(top_N)] + [f"SCORE-{j + 1}" for j in range(top_N)]
    rdf = pd.DataFrame(results, columns=col)

    if top_N == 1:
        rdf = rdf.rename(columns={"CLASS-1": "CATEGORY"}).drop(columns=["SCORE-1"])

    # --- Construct raw scores DataFrame ---
    rawdf = None
    if raw_scores is not None:
        raw_col = ["FILE", "PAGE"] + categories
        rawdf = pd.DataFrame(raws, columns=raw_col)

    print(f"Created results table with {len(rdf)} rows and columns: {rdf.columns.tolist()}")
    if rawdf is not None:
        print(f"Created RAW results table with shape {rawdf.shape}")

    return rdf, rawdf


def collect_images(directory: str, max_categ: int) -> (list, list, list):
    categories = sorted(os.listdir(directory))
    print(f"Category input directories found: {categories}")

    total_files, total_labels, total_classes = [], [], []
    for category_idx, category in enumerate(categories):
        all_category_files = os.listdir(os.path.join(directory, category))
        if len(all_category_files) > max_categ:
            random.shuffle(all_category_files)
            all_category_files = all_category_files[:max_categ]

        print(f"Collected {len(all_category_files)} {category} category images")

        total_files += [os.path.join(directory, category, file) for file in all_category_files]

        label_template = np.zeros(len(categories))
        label_template[category_idx] = 1

        total_labels += [label_template] * len(all_category_files)
        total_classes += [category_idx] * len(all_category_files)

    label, count = np.unique(total_classes, return_counts=True)
    for label_id, label_count in zip(label, count):
        print(f"{categories[int(label_id)]}:\t{label_count}\t{round(label_count / len(total_labels) * 100, 2)}%")

    return total_files, total_labels, categories




def append_to_csv(df, filepath):
    """
    Appends a DataFrame to a CSV file, or creates a new file if it doesn't exist.
    """
    if not os.path.exists(filepath):
        df.to_csv(filepath, index=False, sep=",")
    else:
        df.to_csv(filepath, mode="a", header=False, index=False, sep=",")
