from pathlib import Path
import pandas as pd
import os
import numpy as np
import random


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
        print(res)

        results.append(res)

    col = ["FILE", "PAGE"] + [f"CLASS-{j + 1}" for j in range(top_N)] + [f"SCORE-{j + 1}" for j in range(top_N)]
    rdf = pd.DataFrame(results, columns=col)

    return rdf


def collect_images(directory: str, max_categ: int):
    categories = sorted(os.listdir(directory))
    print(f"Category input directories found: {categories}")

    total_files = []
    total_labels = []
    for category_idx, category in enumerate(categories):
        all_category_files = os.listdir(os.path.join(directory, category))
        if len(all_category_files) > max_categ:
            random.shuffle(all_category_files)
            all_category_files = all_category_files[:max_categ]

        total_files += [os.path.join(directory, category, file) for file in all_category_files]

        label_template = np.zeros(len(categories))
        label_template[category_idx] = 1

        total_labels += [label_template] * len(all_category_files)

    return total_files, total_labels, categories
