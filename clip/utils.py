from pathlib import Path
import pandas as pd
import os
import numpy as np
import random
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
import time
import scipy.special

# get list of all files in the folder and nested folders by file format
def directory_scraper(folder_path: Path, file_format: str = "png", file_list: list = None) -> list[str]:
    if file_list is None:
        file_list = []
    file_list += list(folder_path.rglob(f"*.{file_format}"))
    print(f"[ {file_format.upper()} ] \tFrom directory {folder_path} collected {len(file_list)} {file_format} files")
    return file_list


# def dataframe_results(test_images: list, test_predictions: list, categories: list, top_N: int,
#                       raw_scores: list = None) -> (pd.DataFrame, pd.DataFrame):
#     results = []
#     raws = []
#     # print(test_predictions, "test_predictions")
#
#     for image_file, predict_scores in zip(test_images, test_predictions):
#         image_name = Path(image_file).stem
#         name_parts = image_name.split("-")
#         page_num = int(name_parts[-1])
#         document = name_parts[0] if len(name_parts) == 2 else "-".join(name_parts[:-1])
#
#         norm_scores = np.hstack(predict_scores, axis=0) if isinstance(predict_scores, list) else predict_scores
#
#         # Apply softmax using LogSumExp for numerical stability
#         if norm_scores.ndim > 1:
#             # For a 2D array (e.g., batch_size, num_classes)
#             log_sum_exp = scipy.special.logsumexp(norm_scores, axis=-1, keepdims=True)
#             norm_scores = np.exp(norm_scores - log_sum_exp)
#         else:
#             # For a 1D array (single prediction)
#             log_sum_exp = scipy.special.logsumexp(norm_scores)
#             norm_scores = np.exp(norm_scores - log_sum_exp)
#
#         best_indices = np.argsort(norm_scores)[-top_N:][::-1] if top_N > 1 else np.argmax(norm_scores)
#         best_indices = best_indices[:, :top_N] if top_N > 1 else best_indices
#
#         print(best_indices, "best_indices after slicing")
#
#         # labels = [categories[i] for i in best_indices] if top_N > 1 else [categories[best_indices]]
#         # scores = [round(i[1], 3) for i in predict_scores] if top_N > 1 else [round(predict_scores, 3)]
#
#         res = [document, page_num] + labels + scores
#         results.append(res)
#         if raw_scores is not None:
#             raws.append([document, page_num])
#
#     col = ["FILE", "PAGE"] + [f"CLASS-{j + 1}" for j in range(top_N)] + [f"SCORE-{j + 1}" for j in range(top_N)]
#     rdf = pd.DataFrame(results, columns=col)
#
#     if top_N == 1:
#         rdf.drop(columns=["SCORE-1"], inplace=True)
#         rdf.rename(columns={"CLASS-1": "CATEGORY"}, inplace=True)
#
#     rawdf = None
#     if raw_scores is not None:
#         col = ["FILE", "PAGE"]
#         rawdf = pd.DataFrame(raws, columns=col)
#         raw_weights = np.array(raw_scores).round(3)
#         rawdf[categories] = raw_weights
#
#     return rdf, rawdf


def dataframe_results(test_images: list, test_predictions: list, categories: list, top_N: int,
                      raw_scores: list = None) -> (pd.DataFrame, pd.DataFrame):
    """
    Processes image prediction results into two pandas DataFrames:
    one for formatted top-N predictions and another for raw scores.

    Args:
        test_images (list): List of image file paths.
        test_predictions (list): List of prediction scores (logits or raw scores) for each image.
                                 Each element is a 1D numpy array corresponding to an image's prediction.
        categories (list): List of category names, where the index corresponds to the class ID.
        top_N (int): The number of top predictions to include for each image.
        raw_scores (list, optional): List of raw scores for each image, if available.
                                     Defaults to None.

    Returns:
        tuple: A tuple containing two pandas DataFrames:
               - rdf (pd.DataFrame): Formatted DataFrame with 'FILE', 'PAGE', and top-N 'CLASS' and 'SCORE' columns.
                                     If top_N is 1, it will have 'FILE', 'PAGE', and 'CATEGORY'.
               - rawdf (pd.DataFrame): DataFrame with 'FILE', 'PAGE', and raw scores for all categories.
                                       Returns None if raw_scores is not provided.
    """
    results = []
    raws = []

    current_labels = []
    current_scores = []

    print(f"Processing {len(test_images)} images with top_N={top_N} predictions...")
    print("Predictions shape:")
    for pred in test_predictions:
        print(f"\t{len(pred)} categories")
    test_predictions = np.vstack(test_predictions) if isinstance(test_predictions, list) else np.array(test_predictions)
    print(f"Total predictions shape: {test_predictions.shape}")

    for image_file, predict_scores in zip(test_images, test_predictions):
        image_name = Path(image_file).stem
        name_parts = image_name.split("-")
        page_num = int(name_parts[-1])
        document = name_parts[0] if len(name_parts) == 2 else "-".join(name_parts[:-1])

        # Ensure predict_scores is a numpy array for consistent processing
        norm_scores = np.array(predict_scores)

        # Apply softmax using LogSumExp for numerical stability
        # Since predict_scores for a single image is always 1D here,
        # norm_scores will also be 1D.
        log_sum_exp = scipy.special.logsumexp(norm_scores)
        norm_scores = np.exp(norm_scores - log_sum_exp)

        # Determine the best indices based on top_N
        if top_N > 1:
            # Get indices of top_N highest scores, sorted in descending order of score
            best_indices = np.argsort(norm_scores)[-top_N:][::-1]
        else:
            # Get the index of the single highest score
            best_indices = [np.argmax(norm_scores)] # Wrap in a list for consistent iteration

        # Extract labels and scores for the current image

        current_labels = []
        current_scores = []
        for idx in best_indices:
            current_labels.append(categories[idx])
            current_scores.append(round(norm_scores[idx], 3))

        # Combine document, page_num, labels, and scores for the current result row
        res = [document, page_num] + current_labels + current_scores
        results.append(res)

        # Collect raw scores if provided
        if raw_scores is not None:
            raws.append([document, page_num])

    # Create the main results DataFrame
    col = ["FILE", "PAGE"] + [f"CLASS-{j + 1}" for j in range(top_N)] + [f"SCORE-{j + 1}" for j in range(top_N)]
    rdf = pd.DataFrame(results, columns=col)

    # Adjust column names if top_N is 1
    if top_N == 1:
        rdf.drop(columns=["SCORE-1"], inplace=True)
        rdf.rename(columns={"CLASS-1": "CATEGORY"}, inplace=True)

    # Create the raw scores DataFrame if raw_scores were provided
    rawdf = None
    if raw_scores is not None:
        raw_col = ["FILE", "PAGE"]
        rawdf = pd.DataFrame(raws, columns=raw_col)
        # Ensure raw_scores are processed correctly to match the DataFrame structure
        # Assuming raw_scores is a list of 1D arrays, similar to test_predictions
        raw_weights = np.array(raw_scores).round(3)
        rawdf[categories] = raw_weights

    print(f"DataFrame created with {len(rdf)} entries and columns: {rdf.columns.tolist()}")
    print(rdf.head())

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


def confusion_plot(predictions: list, trues: list, categories: list, out_model: str, top_N: int = 1, output_dir: str = None):
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

    print('Percentage correct: ',round(100 * correct / len(trues), 2))

    # Confusion matrix display and normalized output
    disp = ConfusionMatrixDisplay.from_predictions(
        trues, single_pred, cmap='inferno',
        normalize="true", display_labels=np.array(categories)
    )

    print(f"\t{' '.join(disp.display_labels)}")
    for ir, row in enumerate(disp.confusion_matrix):
        print(
            f"{disp.display_labels[ir]}\t{'   '.join([str(val) if val > 0 else ' -- ' for val in np.round(row, 2)])}")

    # Customize x-axis tick labels to show only the first character of each label
    tick_positions = disp.ax_.get_xticks()
    short_labels = [f"{label[0]}{label.split('_')[-1][0] if '_' in label else ''}" for label in disp.display_labels]
    disp.ax_.set_xticks(tick_positions)
    disp.ax_.set_xticklabels(short_labels)

    time_stamp = time.strftime("%Y%m%d-%H%M")
    disp.ax_.set_title(
        f"TOP {top_N} Confusion matrix {out_model}")
    out = f"{output_dir if output_dir else 'result'}/plots/{time_stamp}_{out_model}_conf_mat_TOP-{top_N}.png"
    plt.savefig(out, bbox_inches='tight', dpi=300)
    plt.close()
