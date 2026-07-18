from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════════
# agent-skill branch: inference-only utilities.
#
# The full utils.py (confusion_plot, collect_images and the matplotlib /
# scikit-learn imports they require) lives on the `test` branch. This branch
# keeps only the two helpers consumed at inference time:
#
#   directory_scraper()  - run.py input collection
#   dataframe_results()  - run.py + parallel_best.py CSV table construction
# ═══════════════════════════════════════════════════════════════════════════


# get list of all files in the folder and nested folders by file format
def directory_scraper(
    folder_path: Path, file_format: str = "png", file_list: list = None
) -> list[str]:
    if file_list is None:
        file_list = []
    file_list += list(folder_path.rglob(f"*.{file_format}"))
    print(
        f"[ {file_format.upper()} ] \tFrom directory {folder_path} collected {len(file_list)} {file_format} files"
    )
    return file_list


def dataframe_results(
    test_images: list,
    test_predictions: list,
    categories: list,
    top_N: int,
    raw_scores: list = None,
) -> (pd.DataFrame, pd.DataFrame):
    results = []
    raws = []

    # 2. Compile the regex pattern once for efficiency
    # Pattern explanation:
    # (.*)   -> Capture group 1: Everything up until the separator (greedy match)
    # [-_]   -> Match a single hyphen OR underscore
    # (\d+)  -> Capture group 2: One or more digits (the page number)
    # $      -> End of the string
    pattern = re.compile(r"(.*)[-_](\d+)$")

    for image_file, predict_scores in zip(test_images, test_predictions):
        image_name = Path(image_file).stem

        # 3. Apply the regex match
        match = pattern.match(image_name)

        if match:
            document = match.group(1)
            page_num = int(match.group(2))
        else:
            # Fallback if file doesn't match the format (e.g., "cover_page.png")
            document = image_name
            page_num = 1  # Default value or handle error as needed

        # --- Logic below remains unchanged ---
        labels = (
            [categories[i[0]] for i in predict_scores]
            if top_N > 1
            else [categories[predict_scores]]
        )
        scores = (
            [round(i[1], 3) for i in predict_scores]
            if top_N > 1
            else [round(predict_scores, 3)]
        )

        res = [document, page_num] + labels + scores
        results.append(res)
        if raw_scores is not None:
            raws.append([document, page_num])

    col = (
        ["FILE", "PAGE"]
        + [f"CLASS-{j + 1}" for j in range(top_N)]
        + [f"SCORE-{j + 1}" for j in range(top_N)]
    )
    rdf = pd.DataFrame(results, columns=col)

    if top_N == 1:
        # Keep CLASS-1 / SCORE-1 naming so downstream tools (e.g. averaging.py)
        # can always expect consistent column names regardless of top_N value.
        # A CATEGORY alias column is added for human-readable CSV output.
        rdf["CATEGORY"] = rdf["CLASS-1"]
        rdf.drop(columns=["SCORE-1"], inplace=True)

    rawdf = None
    if raw_scores is not None:
        col = ["FILE", "PAGE"]
        rawdf = pd.DataFrame(raws, columns=col)
        # Ensure raw_scores is a numpy array before rounding
        raw_weights = np.array(raw_scores).round(3)
        rawdf[categories] = raw_weights

    return rdf, rawdf
