import argparse

import configparser
from classifier import *
import time

if __name__ == "__main__":
    # Initialize the parser
    config = configparser.ConfigParser()
    # Read the configuration file
    config.read('config.txt')

    revision_to_base_model = {
        "v4.2": "timm/tf_efficientnetv2_l.in21k_ft_in1k",
        "v1.2": "timm/tf_efficientnetv2_s.in21k",
        "v2.0": "google/vit-base-patch16-224",
        "v2.1": "google/vit-base-patch16-224",
        "v2.2": "google/vit-base-patch16-224",
        "v3.2": "google/vit-base-patch16-384",
        "v5.2": "google/vit-large-patch16-384",
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
    parser.add_argument('-rev', "--revision", type=str, default=None, help="HuggingFace revision (e.g. `main`, `vN.0` or `vN.M`)")
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

    if args.revision is None: # using config file revision
        args.revision = hf_version
    else:
        if args.revision not in revision_to_base_model:
            raise ValueError(f"Revision {args.revision} is not supported. Available revisions: {list(revision_to_base_model.keys())}")

        base_model = revision_to_base_model[args.revision]
        if args.base != base_model:
            print(f"Base model {args.base} does not match the revision {args.revision}. Using {base_model} instead.")
            args.base = base_model


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



