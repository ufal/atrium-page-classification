from pathlib import Path
import pandas as pd
import os
import numpy as np
import random
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
import time


# get list of all files in the folder and nested folders by file format
def directory_scraper(folder_path: Path, file_format: str = "png", file_list: list = None) -> list[str]:
    if file_list is None:
        file_list = []
    file_list += list(folder_path.rglob(f"*.{file_format}"))
    print(f"[ {file_format.upper()} ] \tFrom directory {folder_path} collected {len(file_list)} {file_format} files")
    return file_list


def dataframe_results(test_images: list, test_predictions: list, categories: list, top_N: int):
    results = []
    for image_file, predict_scores in zip(test_images, test_predictions):
        image_name = Path(image_file).stem
        document, page_num = image_name.split("-")

        labels = [categories[i[0]] for i in predict_scores]
        scores = [round(i[1], 3) for i in predict_scores]

        res = [document, page_num] + labels + scores
        # print(res)

        results.append(res)

    col = ["FILE", "PAGE"] + [f"CLASS-{j + 1}" for j in range(top_N)] + [f"SCORE-{j + 1}" for j in range(top_N)]
    rdf = pd.DataFrame(results, columns=col)

    return rdf


def collect_images(directory: str, max_categ: int) -> (list, list, list):
    categories = sorted(os.listdir(directory))
    print(f"Category input directories found: {categories}")

    total_files = []
    total_labels = []
    total_classes = []
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


def confusion_plot(predictions: list, trues: list, categories: list, top_N: int = 1, output_dir: str = None):

    single_pred = []

    correct = 0
    for j, pred_scores in enumerate(predictions):
        labels = [categories[i[0]] for i in pred_scores]
        scores = [round(i[1], 3) for i in pred_scores]
        classes = [i[0] for i in pred_scores]

        true_class = trues[j]

        if top_N > 1:

            if true_class in classes:
                correct += 1
                single_pred.append(true_class)
            else:
                single_pred.append(classes[0])

        else:
            single_pred.append(classes[0])
            if classes[0] == true_class:
                correct += 1

    print('Percentage correct: ',round(100 * correct / len(trues), 2))
    # Confusion matrix display and normalized output
    disp = ConfusionMatrixDisplay.from_predictions(
        trues, single_pred,
        normalize="true", display_labels=np.array(categories)
    )

    print(f"\t{' '.join(disp.display_labels)}")
    for ir, row in enumerate(disp.confusion_matrix):
        print(
            f"{disp.display_labels[ir]}\t{'   '.join([str(val) if val > 0 else ' -- ' for val in np.round(row, 2)])}")

    time_stamp = time.strftime("%Y%m%d-%H%M")
    disp.ax_.set_title(
        f"TOP {top_N} Full Confusion matrix")
    plt.savefig(f"{output_dir if output_dir else 'result'}/plots/{time_stamp}_conf_mat.png", bbox_inches='tight')
    plt.close()
