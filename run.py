import argparse
import os

import configparser
from classifier import *
import time

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

    Training = config.getboolean('TRAIN', 'Training')
    Testing = config.getboolean('TRAIN', 'Testing')
    HF = config.getboolean('HF', 'use_hf')

    model_folder = "model_1119_3"  # change if needed
    model_dir = config.get('OUTPUT', 'FOLDER_MODELS')
     # change if needed
    model_path = f"{model_dir}/{model_folder}"

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
    parser.add_argument('-tn', "--topn", type=int, default=top_N, help="Number of top result categories to consider")
    parser.add_argument("--dir", help="Process whole directory (if -d not used)", action="store_true")
    parser.add_argument("--inner", help="Process subdirectories of the given directory as well (FALSE by default)", default=False, action="store_true")
    parser.add_argument("--train", help="Training model", default=Training, action="store_true")
    parser.add_argument("--eval", help="Evaluating model", default=Testing, action="store_true")
    parser.add_argument("--hf", help="Use model and processor from the HuggingFace repository", default=HF, action="store_true")
    parser.add_argument("--raw", help="Output raw scores for all categories", default=raw, action="store_true")

    args = parser.parse_args()

    input_dir = Path(test_dir) if args.directory is None else Path(args.directory)
    Training = args.train
    model_path = Path(args.model)
    top_N = args.topn
    raw = args.raw

    # locally creating new directory paths instead of .env variables loaded with mistakes
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

        data_dir = config.get("TRAIN", "FOLDER_PAGES")

        total_files, total_labels, categories = collect_images(data_dir, max_categ)

        (trainfiles, testfiles,
         trainLabels, testLabels) = train_test_split(total_files,
                                                     np.array(total_labels),
                                                     test_size=test_size,
                                                     random_state=seed,
                                                     stratify=np.array(total_labels))

        # Initialize the classifier
        classifier = ImageClassifier(checkpoint=base_model, num_labels=len(categories), store_dir=str(cp_dir))

    else:
        categories = def_categ
        print(f"Category input directories found: {categories}")

        # Initialize the classifier
        classifier = ImageClassifier(checkpoint=base_model, num_labels=len(categories), store_dir=str(cp_dir))

    if args.train:
        train_loader = classifier.process_images(trainfiles,
                                                 trainLabels,
                                                 batch,
                                                 True)
        eval_loader = classifier.process_images(testfiles,
                                                testLabels,
                                                batch,
                                                False)

        classifier.train_model(train_loader,
                               eval_loader,
                               output_dir="./model_output",
                               num_epochs=epochs,
                               logging_steps=log_step)

    if args.hf:
        # pushing to repo
        # classifier.push_to_hub(str(model_path), config.get("HF", "repo_name"), False, config.get("HF", "token"))

        # loading from repo
        classifier.load_from_hub(config.get("HF", "repo_name"))

        classifier.save_model(str(model_path))

    else:
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
        rdf.to_csv(f"{output_dir}/tables/{time_stamp}_{model_folder}_TOP-{top_N}_EVAL.csv", sep=",", index=False)

        if raw:
            raw_df["TRUE"] = [categories[i] for i in test_labels]
            raw_df.to_csv(f"{output_dir}/tables/{time_stamp}_{model_folder}_EVAL_RAW.csv", sep=",", index=False)

        confusion_plot(eval_predictions,
                       test_labels,
                       categories,
                       top_N)

    if args.file is not None:
        pred_scores = classifier.top_n_predictions(args.file, top_N)

        labels = [categories[i[0]] for i in pred_scores]
        scores = [round(i[1], 3) for i in pred_scores]

        print(f"File {args.file} predicted:")
        for lab, sc in zip(labels, scores):
            print(f"\t{lab}:  {round(sc * 100, 2)}%")

    if args.dir or args.directory is not None:

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

        rdf.to_csv(f"{output_dir}/tables/{time_stamp}_{model_folder}_TOP-{top_N}.csv", sep=",", index=False)
        if raw:
            raw_df.to_csv(f"{output_dir}/tables/{time_stamp}_{model_folder}_RAW.csv", sep=",", index=False)


