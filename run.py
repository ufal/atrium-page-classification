import argparse
import os

from dotenv import load_dotenv
from classifier import *
import time

if __name__ == "__main__":
    data_dir = f'/lnet/work/people/lutsai/pythonProject/pages/train_final'  # data for training
    test_dir = f'/lnet/work/people/lutsai/pythonProject/pages/test_data'  # considered as default input folder

    def_categ = ["DRAW", "DRAW_L", "LINE_HW", "LINE_P", "LINE_T", "PHOTO", "PHOTO_L", "TEXT", "TEXT_HW", "TEXT_P", "TEXT_T"]

    seed = 420
    max_categ = 1400  # max number of category samples
    test_size = 0.1
    log_step = 10  # for training only

    epochs = 5  # training only
    batch = 12  # depends on GPU/CPU capabilities
    top_N = 3  # top N predictions, 3 is enough, 11 for "raw" scores (most scores are 0)

    Training = False  # training control
    Testing = False  # evaluation control

    base_model = "google/vit-base-patch16-224"  # do not change
    model_folder = "model_1119_3"  # change if needed
    model_path = f"model/{model_folder}"  # change if needed

    time_stamp = time.strftime("%Y%m%d-%H%M")  # for results files

    parser = argparse.ArgumentParser(description='Page sorter based on ViT')
    parser.add_argument('-f', "--file", type=str, default=None, help="Single PNG page path")
    parser.add_argument('-d', "--directory", type=str, default=None, help="Path to folder with PNG pages")
    parser.add_argument('-m', "--model", type=str, default=model_path, help="Path to folder with model")
    parser.add_argument('-tn', "--topn", type=int, default=top_N, help="Number of top result categories to consider")
    parser.add_argument("--dir", help="Process whole directory", action="store_true")
    parser.add_argument("--train", help="Training model", default=Training, action="store_true")
    parser.add_argument("--eval", help="Evaluating model", default=Testing, action="store_true")

    args = parser.parse_args()

    load_dotenv()

    page_images_folder = Path(os.environ.get('FOLDER_PAGES', Path(data_dir)))
    input_dir = Path(os.environ.get('FOLDER_INPUT', Path(test_dir))) if args.directory is None else Path(args.directory)

    cur = Path.cwd()  # directory with this script
    output_dir = Path(os.environ.get('FOLDER_RESULTS', cur / "result"))
    cp_dir = Path(os.environ.get('FOLDER_CP', cur / "ckeckpoint"))

    # locally creating new directory paths instead of .env variables loaded with mistakes
    if not output_dir.is_dir():
        os.makedirs(output_dir)

        os.makedirs(f"{output_dir}/tables")
        os.makedirs(f"{output_dir}/plots")

    if not cp_dir.is_dir():
        os.makedirs(cp_dir)

    if args.train or args.eval:
        total_files, total_labels, categories = collect_images(data_dir, max_categ)

        (trainfiles, testfiles,
         trainLabels, testLabels) = train_test_split(total_files,
                                                     np.array(total_labels),
                                                     test_size=test_size,
                                                     random_state=seed,
                                                     stratify=np.array(total_labels))

        # Initialize the classifier
        classifier = ImageClassifier(checkpoint=base_model, num_labels=len(categories))

    else:
        categories = def_categ
        print(f"Category input directories found: {categories}")

        # Initialize the classifier
        classifier = ImageClassifier(checkpoint=base_model, num_labels=len(categories), store_dir=cp_dir)

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

    classifier.load_model(model_path)

    if args.eval:
        eval_loader = classifier.process_images(testfiles,
                                                testLabels,
                                                batch,
                                                False)
        eval_predictions = classifier.infer_dataloader(eval_loader, top_N)

        test_labels = np.argmax(testLabels, axis=-1).tolist()

        rdf = dataframe_results(testfiles,
                                eval_predictions,
                                categories,
                                top_N)

        rdf["TRUE"] = [categories[i] for i in test_labels]

        rdf.to_csv(f"{output_dir}/tables/{time_stamp}_{model_folder}_TOP-{top_N}_EVAL.csv", sep=",", index=False)

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
            print(f"\t{lab}:  {sc}")

    if args.dir or args.directory is not None:
        test_images = sorted(os.listdir(test_dir))
        test_images = [os.path.join(test_dir, img) for img in test_images]
        test_loader = classifier.create_dataloader(test_images, batch)

        test_predictions = classifier.infer_dataloader(test_loader, top_N)

        rdf = dataframe_results(test_images,
                                test_predictions,
                                categories,
                                top_N)

        rdf.to_csv(f"{output_dir}/tables/{time_stamp}_{model_folder}_TOP-{top_N}.csv", sep=",", index=False)


