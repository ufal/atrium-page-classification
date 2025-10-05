import argparse

import configparser
import math

from classifier import *
import time
from huggingface_hub import create_branch


if __name__ == "__main__":
    # Initialize the parser
    config = configparser.ConfigParser()
    # Read the configuration file
    config.read('config.txt')

    revision_to_base_model = {
        "v1.": "timm/tf_efficientnetv2_s.in21k",
        "v2.": "google/vit-base-patch16-224",
        "v3.": "google/vit-base-patch16-384",
        "v4.": "timm/tf_efficientnetv2_l.in21k_ft_in1k",
        "v5.": "google/vit-large-patch16-384",
        "v6.": "timm/regnety_120.sw_in12k_ft_in1k",
        "v7.": "timm/regnety_160.swag_ft_in1k",
        "v8.": "timm/regnety_640.seer",
        "v9.": "microsoft/dit-base-finetuned-rvlcdip",
        "v10.": "microsoft/dit-large-finetuned-rvlcdip",
        "v11.": "microsoft/dit-large"
    }

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

    cross_runs = config.get("TRAIN", "cross_runs")

    # setting main to latest version by default
    # hf_version = hf_version if hf_version != 'main' else config.get('HF', 'latest')

    model_name_local = f"model_{hf_version.replace('.', '')}"
    model_dir = config.get('OUTPUT', 'FOLDER_MODELS')
    model_path = f"{model_dir}/{model_name_local}"

    test_dir = config.get('INPUT', 'FOLDER_INPUT')
    dir_format = config.get('INPUT', 'INPUT_FORMAT')

    # cur = Path.cwd()  # directory with this script
    cur = Path(__file__).resolve().parent  # directory with this script
    output_dir = Path(config.get('OUTPUT', 'FOLDER_RESULTS'))
    cp_dir = Path(config.get('OUTPUT', 'FOLDER_CPOINTS'))

    time_stamp = time.strftime("%Y%m%d-%H%M")  # for results files

    parser = argparse.ArgumentParser(description='Page sorter based on ViT')
    parser.add_argument('-f', "--file", type=str, default=None, help="Single page path")
    parser.add_argument('-ff', "--file_format", type=str, default=dir_format, help="File format to look for in the directory")
    parser.add_argument('-d', "--directory", type=str, default=None, help="Path to folder with PNG pages")
    parser.add_argument('-m', "--model", type=str, default=model_path, help="Path to folder with model")
    parser.add_argument('-b', "--base", type=str, default=base_model, help="Repository of the base model")
    parser.add_argument('-rev', "--revision", type=str, default=None, help="HuggingFace revision (e.g. `main`, `vN.0` or `vN.M`)")
    parser.add_argument('-tn', "--topn", type=int, default=top_N, help="Number of top result categories to consider")
    parser.add_argument("--dir", help="Process whole directory (if -d not used)", action="store_true")
    parser.add_argument("--chunk", help="Process input directory and save predictions in chunks", action="store_true")
    parser.add_argument("--inner", help="Process subdirectories of the given directory as well (FALSE by default)", default=inner, action="store_true")
    parser.add_argument("--train", help="Training model", default=Training, action="store_true")
    parser.add_argument("--eval", help="Evaluating model", default=Testing, action="store_true")
    parser.add_argument("--hf", help="Use model and processor from the HuggingFace repository", default=HF, action="store_true")
    parser.add_argument("--raw", help="Output raw scores for all categories", default=raw, action="store_true")
    parser.add_argument("--folds", type=int, default=cross_runs, help="Number of folds for cross-validation with 80/10/10 split. Default is 0 (no cross-validation).")
    parser.add_argument("--average", help="Average existing fold models", action="store_true")
    parser.add_argument("-ap", "--average_pattern", type=str, default=None,
                        help="Pattern for models to average (e.g., 'model_v4')")

    args = parser.parse_args()

    input_dir = Path(test_dir) if args.directory is None else Path(args.directory)
    Training, top_N, raw = args.train, args.topn, args.raw

    if args.revision is None: # using config file revision
        args.revision = hf_version
    else:
        if not any(args.revision.startswith(key) for key in revision_to_base_model.keys()):
            raise ValueError(f"Revision {args.revision} is not supported. Available revisions: {list(revision_to_base_model.keys())}")

        base_model = revision_to_base_model[args.revision]
        if args.base != base_model:
            print(f"Base model {args.base} does not match the revision {args.revision}. Using {base_model} instead.")
            args.base = base_model

    print("Arguments:")
    for arg in vars(args):
        if getattr(args, arg) is not None and getattr(args, arg) != False:
            print(arg, "\t=\t", getattr(args, arg))

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

    if args.train or args.eval:
        epochs = config.getint("TRAIN", "epochs")
        max_categ = config.getint("TRAIN", "max_categ")  # max number of category samples
        log_step = config.getint("TRAIN", "log_step")
        test_size = config.getfloat("TRAIN", "test_size")
        learning_rate = config.getfloat("TRAIN", "lr")

        data_dir = config.get("TRAIN", "FOLDER_PAGES")

        if args.train:
            total_files, total_labels, categories = collect_images(data_dir)


    if args.eval:
        data_dir = config.get("EVAL", "FOLDER_PAGES")
        testfiles, testLabels, categories = collect_images(data_dir)
        # Initialize the classifier
        classifier = ImageClassifier(checkpoint=args.base, num_labels=len(categories), store_dir=str(cp_dir))

    else:
        categories = def_categ
        print(f"Category input directories found: {categories}")
        # Initialize the classifier
        classifier = ImageClassifier(checkpoint=args.base, num_labels=len(categories), store_dir=str(cp_dir))


    if args.train:

        if args.folds > 0:
            for i in range(args.folds):
                print(f"--- Cross-Validation Fold {i + 1}/{args.folds} ---")
                fold_seed = seed + i  # Use a different seed for each fold

                (trainfiles, valfiles, testfiles,
                 trainLabels, valLabels, testLabels) = split_data_80_10_10(total_files, total_labels, fold_seed, max_categ)

                # record datasets
                with open(f"{output_dir}/stats/{time_stamp}_{model_name_local}_FOLD_{i + 1}_DATASETS.txt", "w") as f:
                    f.write(f"Training set ({len(trainfiles)} images):\n")
                    for file in trainfiles:
                        f.write(f"{file}\n")
                    f.write(f"\nValidation set ({len(valfiles)} images):\n")
                    for file in valfiles:
                        f.write(f"{file}\n")
                    f.write(f"\nTest set ({len(testfiles)} images):\n")
                    for file in testfiles:
                        f.write(f"{file}\n")

                # Initialize a new classifier for each fold
                classifier = ImageClassifier(checkpoint=args.base, num_labels=len(categories), store_dir=str(cp_dir))

                train_loader = classifier.process_images(trainfiles, trainLabels, batch, True)
                eval_loader = classifier.process_images(valfiles, valLabels, batch, False)
                test_loader = classifier.process_images(testfiles, testLabels, batch, False)

                print(
                    f"Fold {i + 1}: Training on {len(trainfiles)}, validating on {len(valfiles)}, testing on {len(testfiles)}.")

                # Train the model
                classifier.train_model(
                    train_loader,
                    eval_loader,
                    output_dir=f"./model_output_fold_{i + 1}",
                    out_model=f"{model_name_local}{i + 1}",
                    num_epochs=epochs,
                    learning_rate=learning_rate,
                    logging_steps=log_step
                )

                # Evaluate on the test set for the current fold
                print(f"--- Evaluating on test set for fold {i + 1} ---")
                test_predictions, raw_prediction = classifier.infer_dataloader(test_loader, top_N, raw)
                # testLabels = [t for t in test_loader.image_labels if t is not None ]
                test_labels_indices = np.argmax(testLabels, axis=-1).tolist()

                # print(f"Head and tail of the predicted labels:")
                # print(test_predictions[:3])
                # print(test_predictions[-3:])

                print(f"TEST SET's correct percentage:\t{round(100 * sum([1 for true, pred in zip(test_labels_indices, test_predictions) if true == pred]) / len(test_labels_indices), 2)}%")

                rdf, raw_df = dataframe_results(testfiles, test_predictions, categories, top_N, raw_prediction)
                rdf["TRUE"] = [categories[label] for label in test_labels_indices]
                rdf.sort_values(['FILE', 'PAGE'], ascending=[True, True], inplace=True)
                rdf.to_csv(f"{output_dir}/tables/{time_stamp}_{model_name_local}_TEST_FOLD_{i + 1}.csv", index=False)

                if raw:
                    raw_df["TRUE"] = [categories[label] for label in test_labels_indices]
                    raw_df.sort_values(categories, ascending=[False] * len(categories), inplace=True)
                    raw_df.to_csv(f"{output_dir}/tables/{time_stamp}_{model_name_local}_TEST_RAW_FOLD_{i + 1}.csv",
                                  index=False)

                print(f"Test results for fold {i + 1} saved.")
        else:

            (trainfiles, valfiles, testfiles,
             trainLabels, valLabels, testLabels) = split_data_80_10_10(total_files, total_labels, seed, max_categ)

            # classifier = ImageClassifier(checkpoint=args.base, num_labels=len(categories), store_dir=str(cp_dir))

            train_loader = classifier.process_images(trainfiles, trainLabels, batch, True)
            eval_loader = classifier.process_images(valfiles, valLabels, batch, False)

            print(f"Training on {len(trainfiles)} images, evaluating on {len(valfiles)} images")
            print(f"Base model: {args.base}, local model name: {model_name_local}")
            classifier.train_model(
                train_loader,
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
        classifier.load_model(str(model_path))
        create_branch(config.get("HF", "repo_name"), repo_type="model", branch=args.revision, exist_ok=True, token=config.get("HF", "token"))

        classifier.push_to_hub(str(model_path), config.get("HF", "repo_name"), False, config.get("HF", "token"), config.get("HF", "revision"))
        # ----------------------------------------------

        # loading from repo
        classifier.load_from_hub(config.get("HF", "repo_name"), args.revision)

        #hf_model_name_local = f"model_{args.revision.replace('.', '')}"
        #hf_model_path = f"{model_dir}/{hf_model_name_local}"

        classifier.save_model(str(model_path))

        classifier.load_model(str(model_path))

    else:
        classifier.load_model(str(model_path))

    if args.eval:
        print(f"--- Evaluating on the test set ({len(testfiles)} images) ---")
        test_loader = classifier.process_images(testfiles, testLabels, batch, False)
        eval_predictions, raw_prediction = classifier.infer_dataloader(test_loader, top_N, raw)
        test_labels_indices = np.argmax(testLabels, axis=-1).tolist()

        rdf, raw_df = dataframe_results(testfiles, eval_predictions, categories, top_N, raw_prediction)

        rdf["TRUE"] = [categories[i] for i in test_labels_indices]
        rdf.sort_values(['FILE', 'PAGE'], ascending=[True, True], inplace=True)
        rdf.to_csv(f"{output_dir}/tables/{time_stamp}_{model_name_local}_TOP-{top_N}_EVAL.csv", sep=",", index=False)
        print(f"Evaluation results for TOP-{top_N} predictions are recorded into {output_dir}/tables/ directory")

        if raw:
            raw_df["TRUE"] = [categories[i] for i in test_labels_indices]
            raw_df.sort_values(categories, ascending=[False] * len(categories), inplace=True)
            raw_df.to_csv(f"{output_dir}/tables/{time_stamp}_{model_name_local}_EVAL_RAW.csv", sep=",", index=False)
            print(f"RAW Evaluation results are recorded into {output_dir}/tables/ directory")


        confusion_plot(eval_predictions,
                       test_labels_indices,
                       categories,
                       model_name_local,
                       top_N)

    if args.average:
        print("\n" + "=" * 60)
        print("AVERAGING EXISTING FOLD MODELS")
        print("=" * 60)

        # Average specific pattern
        base_model_for_pattern = None
        for version_key, model_path in revision_to_base_model.items():
            if version_key.rstrip('.') in args.average_pattern:
                base_model_for_pattern = model_path
                break

        if base_model_for_pattern:
            try:
                averaged_path = average_model_weights(
                    model_dir=str(model_dir),
                    model_name_pattern=args.average_pattern,
                    base_model=base_model_for_pattern,
                    num_labels=len(categories)
                )
                print(f"Averaged model saved to: {averaged_path}")
            except Exception as e:
                print(f"Error averaging models: {e}")
        else:
            print(f"Could not determine base model for pattern: {args.average_pattern}")


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
            test_images = sorted(directory_scraper(Path(test_dir), args.file_format))
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

    elif (args.dir or args.directory is not None) and args.chunk:
        print(f"Start of the directory processing at {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # build list of image paths (unchanged)
        if args.inner:
            test_images = sorted(directory_scraper(Path(test_dir), args.file_format))
        else:
            test_images = sorted(os.listdir(test_dir))
            test_images = [os.path.join(test_dir, img) for img in test_images]

        # ensure output dir exists
        os.makedirs(f"{output_dir}/tables", exist_ok=True)

        # chunking params
        chunk_size = 10000  # change this if you want a different frequency
        total = len(test_images)
        chunks = math.ceil(total / chunk_size)

        # daily date-based filenames (YYYYMMDD)
        date_stamp = time.strftime('%Y%m%d')
        top_out_path = f"{output_dir}/tables/{date_stamp}_{model_name_local}_TOP-{top_N}.csv"
        raw_out_path = f"{output_dir}/tables/{date_stamp}_{model_name_local}_RAW.csv"

        for chunk_idx, start in enumerate(range(0, total, chunk_size), start=1):
            end = min(start + chunk_size, total)
            chunk_images = test_images[start:end]
            print(f"Processing images {start + 1}–{end} (chunk {chunk_idx}/{chunks})")

            # create dataloader and run inference for this chunk
            test_loader = classifier.create_dataloader(chunk_images, batch)
            test_predictions, raw_prediction = classifier.infer_dataloader(test_loader, top_N, raw)

            # convert to dataframes for this chunk
            rdf_chunk, raw_df_chunk = dataframe_results(
                chunk_images,
                test_predictions,
                categories,
                top_N,
                raw_prediction
            )

            # sort chunk for nicer local ordering (optional)
            rdf_chunk.sort_values(['FILE', 'PAGE'], ascending=[True, True], inplace=True)

            # append chunk to the daily TOP file (write header only if file doesn't exist)
            write_header = not os.path.exists(top_out_path)
            rdf_chunk.to_csv(top_out_path, sep=",", index=False, mode='a', header=write_header)
            if write_header:
                print(f"Created and wrote TOP-{top_N} daily file: {top_out_path} (chunk {chunk_idx})")
            else:
                print(f"Appended TOP-{top_N} chunk {chunk_idx} to {top_out_path}")

            if raw:
                # sort raw chunk by category scores (descending) if possible
                if raw_df_chunk is not None and not raw_df_chunk.empty:
                    raw_df_chunk.sort_values(categories, ascending=[False] * len(categories), inplace=True)

                write_header_raw = not os.path.exists(raw_out_path)
                raw_df_chunk.to_csv(raw_out_path, sep=",", index=False, mode='a', header=write_header_raw)
                if write_header_raw:
                    print(f"Created and wrote RAW daily file: {raw_out_path} (chunk {chunk_idx})")
                else:
                    print(f"Appended RAW chunk {chunk_idx} to {raw_out_path}")

        print(f"Processing complete. Daily files are in {output_dir}/tables/:")
        print(f" - TOP file: {top_out_path}")
        if raw:
            print(f" - RAW file: {raw_out_path}")




