from pathlib import Path
import pandas as pd
import os
import numpy as np
import random
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
import time


def evaluate_multiple_models(model_dir: str, eval_dir: str, vis: bool = True,
                                 batch_size: int = 16, cp_dir: str = "checkpoint",):
    """
    Evaluates multiple saved models in a directory and records their Top-1 accuracy.
    :param model_dir: Directory containing the saved model files.
    :param eval_dir: Directory for evaluation data.
    :param model_suffix: Suffix to filter model files (e.g., ".pt", "_cp.pt").
    :param vis: If True, visualize results in a bar graph.
    :param batch_size: Batch size for evaluation.
    """
    map_base_name = {
        "google/vit-base-patch16-224": "ViT-B/16",
        "google/vit-base-patch16-384": "ViT-B/16-384",
        "google/vit-large-patch16-384": "ViT-L/16",
        "microsoft/dit-base-finetuned-rvlcdip": "DiT-B/RVL",
        "microsoft/dit-large-finetuned-rvlcdip": "DiT-L/RVL",
        "microsoft/dit-large": "DiT-L",
        "timm/tf_efficientnetv2_s.in21k": "EffNetV2-S",
        "timm/tf_efficientnetv2_m.in21k_ft_in1k": "EffNetV2-M",
        "timm/tf_efficientnetv2_l.in21k_ft_in1k": "EffNetV2-L",
        "timm/regnety_120.sw_in12k_ft_in1k": "RegNetY-12GF",
        "timm/regnety_160.pycls_in1k": "RegNetY-16GF",
        "timm/regnety_640.seer_ft_in1k": "RegNetY-64GF",
    }

    model_bases = {
        "_v22010": "google/vit-base-patch16-224",
        "_v32010": "google/vit-base-patch16-384",
        "_v52010": "google/vit-large-patch16-384",
        "_v72010": "microsoft/dit-base-finetuned-rvlcdip",
        "_v92010": "microsoft/dit-large-finetuned-rvlcdip",
        "_v82010": "microsoft/dit-large",
        "_v120105s": "timm/tf_efficientnetv2_s.in21k",
        "_v120105m": "timm/tf_efficientnetv2_m.in21k_ft_in1k",
        "_v120105l": "timm/tf_efficientnetv2_l.in21k_ft_in1k",
        "_v12010512": "timm/regnety_120.sw_in12k_ft_in1k",
        "_v12010516": "timm/regnety_160.pycls_in1k",
        "_v12010564": "timm/regnety_640.seer_ft_in1k",
    }

    scheduler_ending = {
        "106": "linear",
        "105": "cosine",
        "105p": "poly",
        "105fp": "cosine fp16",
    }

    cur = Path(__file__).parent if '__file__' in globals() else Path.cwd()
    output_dir = cur / "result"
    output_dir.mkdir(exist_ok=True, parents=True)

    model_dir_path = cur / Path(model_dir)
    if not model_dir_path.is_dir():
        print(f"Error: Model directory not found at {model_dir}")
        return

    model_files = list(model_dir_path.rglob(f"*2010*"))
    if not model_files:
        print(f"No model files found with suffix '2010' in {model_dir}")
        return

    print(f"Found {len(model_files)} models with suffix '2010'.")
    accuracies = {}  # To store filename_stem: top1_accuracy



    testfiles, testLabels, categories = collect_images(eval_dir, 100000)

    for model_path in sorted(model_files):
        model_name_stem = model_path.stem
        print(f"\nEvaluating model: {model_name_stem}")

        base_name = None
        for model_base_encoding, model_base_name in model_bases.items():
            if model_base_encoding in model_name_stem:
                base_name = model_base_name
                break

        sched_categ = "UNK"  # Default category
        for scheduler_enc, categ in scheduler_ending.items():
            if model_name_stem.endswith(scheduler_enc):
                sched_categ = categ
                break

        vis_base = map_base_name.get(base_name, base_name)

        if base_name is None:
            vis_model_name = f"{vis_base} {sched_categ}"
            accuracies[vis_model_name] = "Error"  # Indicate error
        else:
            vis_model_name = f"{vis_base} {sched_categ}"
            print(vis_model_name)
            try:
                # Initialize the classifier
                classifier = ImageClassifier(checkpoint=base_name, num_labels=len(categories), store_dir=str(cp_dir))
                classifier.load_model(str(model_path))


                eval_loader = classifier.process_images(testfiles,
                                                        testLabels,
                                                        batch_size,
                                                        False)
                eval_predictions, raw_prediction = classifier.infer_dataloader(eval_loader, top_N, False)

                test_labels = np.argmax(testLabels, axis=-1).tolist()

                rdf, raw_df = dataframe_results(testfiles,
                                                eval_predictions,
                                                categories,
                                                top_N,
                                                raw_prediction)

                rdf["TRUE"] = [categories[i] for i in test_labels]
                rdf.sort_values(['FILE', 'PAGE'], ascending=[True, True], inplace=True)
                rdf.to_csv(f"{output_dir}/tables/{time_stamp}_{model_name_stem}_TOP-{top_N}_EVAL.csv", sep=",",
                           index=False)
                print(
                    f"Evaluation results for TOP-{top_N} predictions are recorded into {output_dir}/tables/ directory")

                confusion_plot(eval_predictions,
                               test_labels,
                               categories,
                               model_name_stem,
                               top_N)

                single_pred = []
                correct = 0
                for j, pred_scores in enumerate(eval_predictions):
                    true_class = test_labels[j]

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

                accuracies[vis_model_name] = round(100 * correct / len(test_labels), 2)


                print(f"Top 1 Accuracy for {vis_model_name} {model_name_stem}: {accuracies[vis_model_name]:.2f}%")

            except Exception as e:
                print(f"Error evaluating model {model_name_stem}: {e}")
                if vis_model_name not in accuracies.keys():
                    accuracies[vis_model_name] = "Error"  # Indicate error

    # Save results to a table
    if accuracies:
        results_df = pd.DataFrame(list(accuracies.items()), columns=['model_name', 'accuracy'])

        # Create a dedicated directory for evaluation statistics
        eval_stats_output_dir = Path(output_dir) / 'stats'
        eval_stats_output_dir.mkdir(parents=True, exist_ok=True)

        csv_output_path = eval_stats_output_dir / f"model_accuracies_trans.csv"
        results_df.to_csv(csv_output_path, index=False)
        print(f"Evaluation results saved to {csv_output_path}")

        if vis:
            # sort results by vis order
            visualize_results(str(csv_output_path), str(Path(output_dir) / 'stats'))
    else:
        print("No models were successfully evaluated.")

def visualize_results(csv_file: str, output_dir: str):
    """
    Generate a bar plot from a CSV file of model accuracies.

    :param csv_file: Path to the CSV file containing model accuracies.
    :param output_dir: Directory where the plot will be saved.
    :param vis_orders: Dictionary to define custom sorting order.
    :param base_model_colors: Dictionary to map base model names to specific colors.
    """
    base_model_colors = {
        "linear ": "steelblue",
        "cosine ": "indigo",
        "poly ": "orange",
        "cosine fp16 ": "gold"
    }

    category_codes = {
        "vit": 4,  # 4
        "regnet": 2,  # 2
        "dit": 3,  # 3
        "eff": 1,  # 1
    }

    vis_order = {}

    results_df = pd.read_csv(csv_file)

    for vis_model_name in results_df['model_name'].tolist():
        for code, order in category_codes.items():
            if code in vis_model_name.lower():
                vis_order[vis_model_name] = order
                break

    # Load the CSV into a DataFrame


    results_df['vis_order'] = results_df['model_name'].apply(lambda x: vis_order.get(x, 0))
    results_df.sort_values(by='vis_order', inplace=True, ascending=True)
    results_df.drop(columns='vis_order', inplace=True)

    # Assign colors based on base model
    results_df['color'] = results_df['model_name'].apply(
        lambda x: next((color for base, color in base_model_colors.items() if base in x), 'black')
    )

    # Generate the bar plot
    plt.figure(figsize=(12, 7))
    plt.bar(results_df['model_name'], results_df['accuracy'], color=results_df['color'])
    plt.xlabel("Model Name")
    plt.ylabel("Top-1 Accuracy (%)")
    plt.title("Model Accuracy Comparison")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # set min-max y-axis values
    plt.ylim(results_df['accuracy'].min()-1, 100 if results_df['accuracy'].max() == 100 else results_df['accuracy'].max()+1)

    # Save the plot
    plot_output_dir = Path(output_dir)
    plot_output_dir.mkdir(parents=True, exist_ok=True)
    plot_output_path = plot_output_dir / f"model_accuracy_plot_trans.png"
    plt.savefig(plot_output_path, dpi=300)
    plt.close()
    print(f"Accuracy plot saved to {plot_output_path}")

# get list of all files in the folder and nested folders by file format
def directory_scraper(folder_path: Path, file_format: str = "png", file_list: list = None) -> list[str]:
    if file_list is None:
        file_list = []
    file_list += list(folder_path.rglob(f"*.{file_format}"))
    print(f"[ {file_format.upper()} ] \tFrom directory {folder_path} collected {len(file_list)} {file_format} files")
    return file_list


def dataframe_results(test_images: list, test_predictions: list, categories: list, top_N: int,
                      raw_scores: list = None) -> (pd.DataFrame, pd.DataFrame):
    results = []
    raws = []
    for image_file, predict_scores in zip(test_images, test_predictions):
        image_name = Path(image_file).stem
        name_parts = image_name.split("-")
        page_num = int(name_parts[-1])
        document = name_parts[0] if len(name_parts) == 2 else "-".join(name_parts[:-1])

        labels = [categories[i[0]] for i in predict_scores] if top_N > 1 else [categories[predict_scores]]
        scores = [round(i[1], 3) for i in predict_scores] if top_N > 1 else [round(predict_scores, 3)]

        res = [document, page_num] + labels + scores
        results.append(res)
        if raw_scores is not None:
            raws.append([document, page_num])

    col = ["FILE", "PAGE"] + [f"CLASS-{j + 1}" for j in range(top_N)] + [f"SCORE-{j + 1}" for j in range(top_N)]
    rdf = pd.DataFrame(results, columns=col)

    if top_N == 1:
        rdf.drop(columns=["SCORE-1"], inplace=True)
        rdf.rename(columns={"CLASS-1": "CATEGORY"}, inplace=True)

    rawdf = None
    if raw_scores is not None:
        col = ["FILE", "PAGE"]
        rawdf = pd.DataFrame(raws, columns=col)
        raw_weights = np.array(raw_scores).round(3)
        rawdf[categories] = raw_weights

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
