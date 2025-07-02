import argparse
import os

import configparser
from classifier import *
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

                # confusion_plot(eval_predictions,
                #                test_labels,
                #                categories,
                #                model_name_stem,
                #                top_N)

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

if __name__ == "__main__":
    # Initialize the parser
    config = configparser.ConfigParser()
    # Read the configuration file
    config.read('config.txt')

    def_categ = ["DRAW", "DRAW_L", "LINE_HW", "LINE_P", "LINE_T", "PHOTO", "PHOTO_L", "TEXT", "TEXT_HW", "TEXT_P", "TEXT_T"]

    seed = config.getint('SETUP', 'seed')
    batch = config.getint('SETUP', 'batch')  # depends on GPU/CPU capabilities
    top_N = config.getint('SETUP', 'top_N')  # top N predictions, 3 is enough, 11 for "raw" scores (most scores are 0)

    base_model = config.get('SETUP', 'base_model')  # do not change

    raw = config.getboolean('SETUP', 'raw')
    inner = config.getboolean('SETUP', 'inner')

    Training = config.getboolean('TRAIN', 'Training')
    Testing = config.getboolean('TRAIN', 'Testing')
    HF = config.getboolean('HF', 'use_hf')
    hf_version = config.get("HF", "revision")

    # setting main to latest version by default
    # hf_version = hf_version if hf_version != 'main' else config.get('HF', 'latest')

    model_name_local = f"model_{hf_version.replace('.', '')}"
    model_dir = config.get('OUTPUT', 'FOLDER_MODELS')
    model_path = f"{model_dir}/{model_name_local}"

    test_dir = config.get('INPUT', 'FOLDER_INPUT')

    # cur = Path.cwd()  # directory with this script
    cur = Path(__file__).resolve().parent  # directory with this script
    output_dir = Path(config.get('OUTPUT', 'FOLDER_RESULTS'))
    cp_dir = Path(config.get('OUTPUT', 'FOLDER_CPOINTS'))

    time_stamp = time.strftime("%Y%m%d-%H%M")  # for results files

    parser = argparse.ArgumentParser(description='Page sorter based on ViT')
    parser.add_argument('-f', "--file", type=str, default=None, help="Single PNG page path")
    parser.add_argument('-d', "--directory", type=str, default=None, help="Path to folder with PNG pages")
    parser.add_argument('-m', "--model", type=str, default=model_path, help="Path to folder with model")
    parser.add_argument('-b', "--base", type=str, default=base_model, help="Repository of the base model")
    parser.add_argument('-rev', "--revision", type=str, default=hf_version, help="HuggingFace revision (e.g. `main`, `vN.0` or `vN.M`)")
    parser.add_argument('-tn', "--topn", type=int, default=top_N, help="Number of top result categories to consider")
    parser.add_argument("--dir", help="Process whole directory (if -d not used)", action="store_true")
    parser.add_argument("--inner", help="Process subdirectories of the given directory as well (FALSE by default)", default=inner, action="store_true")
    parser.add_argument("--train", help="Training model", default=Training, action="store_true")
    parser.add_argument("--eval", help="Evaluating model", default=Testing, action="store_true")
    parser.add_argument("--eval_dir", help="Evaluating models", default=False, action="store_true")
    parser.add_argument("--hf", help="Use model and processor from the HuggingFace repository", default=HF, action="store_true")
    parser.add_argument("--raw", help="Output raw scores for all categories", default=raw, action="store_true")

    args = parser.parse_args()

    input_dir = Path(test_dir) if args.directory is None else Path(args.directory)
    Training, top_N, raw = args.train, args.topn, args.raw

    if args.revision == hf_version and args.base == base_model:
        model_path = Path(args.model)
    else:
        new_model_name_local = f"model_{args.revision.replace('.', '')}"
        model_path = f"{model_dir}/{new_model_name_local}"
        model_path = Path(model_path)

    # locally creating new directory paths instead of context.txt variables loaded with mistakes
    if not output_dir.is_dir():
        os.makedirs(output_dir)

        os.makedirs(f"{output_dir}/tables")
        os.makedirs(f"{output_dir}/plots")

    if not cp_dir.is_dir():
        os.makedirs(cp_dir)

    if not Path(model_dir).is_dir():
        os.makedirs(model_dir)

    if args.train or args.eval or args.eval_dir:
        epochs = config.getint("TRAIN", "epochs")
        max_categ = config.getint("TRAIN", "max_categ")  # max number of category samples
        log_step = config.getint("TRAIN", "log_step")
        test_size = config.getfloat("TRAIN", "test_size")
        learning_rate = config.getfloat("TRAIN", "lr")

        data_dir = config.get("TRAIN", "FOLDER_PAGES")

        total_files, total_labels, categories = collect_images(data_dir, max_categ)

        (trainfiles, testfiles,
         trainLabels, testLabels) = train_test_split(total_files,
                                                     np.array(total_labels),
                                                     test_size=test_size,
                                                     random_state=seed,
                                                     stratify=np.array(total_labels))

        # Initialize the classifier
        classifier = ImageClassifier(checkpoint=args.base, num_labels=len(categories), store_dir=str(cp_dir))

    elif args.eval:
        data_dir = config.get("EVAL", "FOLDER_PAGES")

        testfiles, testLabels, categories = collect_images(data_dir, 100000)


        # Initialize the classifier
        classifier = ImageClassifier(checkpoint=args.base, num_labels=len(categories), store_dir=str(cp_dir))

    else:
        categories = def_categ
        print(f"Category input directories found: {categories}")

        # Initialize the classifier
        classifier = ImageClassifier(checkpoint=args.base, num_labels=len(categories), store_dir=str(cp_dir))

    if args.train:
        train_loader = classifier.process_images(trainfiles,
                                                 trainLabels,
                                                 batch,
                                                 True,
                                                 testfiles)
        eval_loader = classifier.process_images(testfiles,
                                                testLabels,
                                                batch,
                                                False)

        print(f"Training on {len(trainfiles)} images, evaluating on {len(testfiles)} images")
        print(f"Base model: {args.base}, local model name: {model_name_local}")
        classifier.train_model(train_loader,
                               eval_loader,
                               output_dir="./model_output",
                               out_model=model_name_local,
                               num_epochs=epochs,
                               learning_rate=learning_rate,
                               logging_steps=log_step
                               )

    if args.hf:
        # ----------------------------------------------
        # ----- UNCOMMENT for pushing to HF repo -------
        # ----------------------------------------------
        # classifier.load_model(str(model_path))
        # classifier.push_to_hub(str(model_path), config.get("HF", "repo_name"), False, config.get("HF", "token"), config.get("HF", "revision"))
        # ----------------------------------------------

        # loading from repo
        classifier.load_from_hub(config.get("HF", "repo_name"), args.revision)

        # hf_model_name_local = f"model_{args.revision.replace('.', '')}"
        # hf_model_path = f"{model_dir}/{hf_model_name_local}"

        classifier.save_model(str(model_path))

        classifier.load_model(str(model_path))

    elif not args.eval_dir:
        classifier.load_model(str(model_path))

    if args.eval:
        eval_loader = classifier.process_images(testfiles,
                                                testLabels,
                                                batch,
                                                False)
        eval_predictions, raw_prediction = classifier.infer_dataloader(eval_loader, top_N, raw)

        test_labels = np.argmax(testLabels, axis=-1).tolist()

        rdf, raw_df = dataframe_results(testfiles,
                                        eval_predictions,
                                        categories,
                                        top_N,
                                        raw_prediction)

        rdf["TRUE"] = [categories[i] for i in test_labels]
        rdf.sort_values(['FILE', 'PAGE'], ascending=[True, True], inplace=True)
        rdf.to_csv(f"{output_dir}/tables/{time_stamp}_{model_name_local}_TOP-{top_N}_EVAL.csv", sep=",", index=False)
        print(f"Evaluation results for TOP-{top_N} predictions are recorded into {output_dir}/tables/ directory")

        if raw:
            raw_df["TRUE"] = [categories[i] for i in test_labels]
            raw_df.sort_values(categories, ascending=[False] * len(categories), inplace=True)
            raw_df.to_csv(f"{output_dir}/tables/{time_stamp}_{model_name_local}_EVAL_RAW.csv", sep=",", index=False)
            print(f"RAW Evaluation results are recorded into {output_dir}/tables/ directory")


        confusion_plot(eval_predictions,
                       test_labels,
                       categories,
                       model_name_local,
                       top_N)
    elif args.eval_dir:
        data_dir = config.get("EVAL", "FOLDER_PAGES")

        evaluate_multiple_models(model_dir, data_dir, True, batch, str(cp_dir))

    if args.file is not None:
        pred_scores = classifier.top_n_predictions(args.file, top_N)

        labels = [categories[i[0]] for i in pred_scores]
        scores = [round(i[1], 3) for i in pred_scores]

        print(f"File {args.file} predicted:")
        for lab, sc in zip(labels, scores):
            print(f"\t{lab}:  {round(sc * 100, 2)}%")

    if args.dir or args.directory is not None:
        print(f"Start of the directory processing at {time.strftime('%Y-%m-%d %H:%M:%S')}")

        if args.inner:
            test_images = sorted(directory_scraper(Path(test_dir), "png"))
        else:
            test_images = sorted(os.listdir(test_dir))
            test_images = [os.path.join(test_dir, img) for img in test_images]

        test_loader = classifier.create_dataloader(test_images, batch)

        test_predictions, raw_prediction = classifier.infer_dataloader(test_loader, top_N, raw)

        rdf, raw_df = dataframe_results(test_images,
                                        test_predictions,
                                        categories,
                                        top_N,
                                        raw_prediction)

        rdf.sort_values(['FILE', 'PAGE'], ascending=[True, True], inplace=True)
        rdf.to_csv(f"{output_dir}/tables/{time_stamp}_{model_name_local}_TOP-{top_N}.csv", sep=",", index=False)
        print(f"Results for TOP-{top_N} predictions are recorded into {output_dir}/tables/ directory")

        if raw:
            raw_df.sort_values(categories, ascending=[False] * len(categories), inplace=True)
            raw_df.to_csv(f"{output_dir}/tables/{time_stamp}_{model_name_local}_RAW.csv", sep=",", index=False)
            print(f"RAW Results are recorded into {output_dir}/tables/ directory")



