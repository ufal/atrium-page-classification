import argparse
from dotenv import load_dotenv
from classifier import *
import time

if __name__ == "__main__":
    data_dir = f'/lnet/work/people/lutsai/pythonProject/pages/train_final'
    test_dir = f'/lnet/work/people/lutsai/pythonProject/pages/test_data'

    top_N = 3
    seed = 420
    max_categ = 1400
    batch = 12
    test_size = 0.1
    epochs = 2
    log_step = 10

    Training = False

    checkpoint_anem = "google/vit-base-patch16-224"
    model_folder = "model_1119_3"
    model_path = f"../trans/model/{model_folder}"

    time_stamp = time.strftime("%Y%m%d-%H%M")

    parser = argparse.ArgumentParser(description='Page sorter based on RFC')
    parser.add_argument('-f', "--file", type=str, default=None, help="Single PNG page path")
    parser.add_argument('-d', "--directory", type=str, default=None, help="Path to folder with PNG pages")
    parser.add_argument('-tn', "--topn", type=int, default=top_N, help="Number of top result categories to consider")
    parser.add_argument("--dir", help="Process whole directory", action="store_true")
    parser.add_argument("--train", help="Process PDF files into layouts", default=Training, action="store_true")

    args = parser.parse_args()

    load_dotenv()

    cur = Path.cwd() #  directory with this script
    # locally creating new directory pathes instead of .env variables loaded with mistakes
    output_dir = Path(os.environ.get('FOLDER_RESULTS', cur / "result"))
    page_images_folder = Path(os.environ.get('FOLDER_PAGES', Path(data_dir)))
    input_dir = Path(os.environ.get('FOLDER_INPUT', Path(test_dir))) if args.directory is None else Path(args.directory)

    if not output_dir.is_dir():
        os.makedirs(output_dir)

        os.makedirs(f"{output_dir}/tables")
        os.makedirs(f"{output_dir}/plots")

    if args.train:
        total_files, total_labels, categories = collect_images(data_dir, max_categ)

        (trainfiles, testfiles,
         trainLabels, testLabels) = train_test_split(total_files,
                                                     np.array(total_labels),
                                                     test_size=test_size,
                                                     random_state=seed,
                                                     stratify=np.array(
                                                         total_labels))

        # Initialize the classifier
        classifier = ImageClassifier(checkpoint=checkpoint_anem, num_labels=len(categories))

        train_loader = classifier.process_images(trainfiles, trainLabels, batch, True)
        eval_loader = classifier.process_images(testfiles, testLabels, batch, False)

        classifier.train_model(train_loader, eval_loader, output_dir="./model_output", num_epochs=epochs,
                               logging_steps=log_step)

        eval_predictions = classifier.infer_dataloader(eval_loader, top_N)

        confusion_plot(eval_predictions, testLabels, categories, top_N)

    else:
        categories = sorted(os.listdir(data_dir))
        print(f"Category input directories found: {categories}")

        # Initialize the classifier
        classifier = ImageClassifier(checkpoint=checkpoint_anem, num_labels=len(categories))

        classifier.load_model(model_path)

    if args.file is not None:
        pred_scores = classifier.top_n_predictions(args.file, top_N)

        labels = [categories[i[0]] for i in pred_scores]
        scores = [round(i[1], 3) for i in pred_scores]

        print(f"File {args.file} predicted:")
        for lab, sc in zip(labels, scores):
            print(f"\t{lab}:  {sc}")

    if args.dir:
        test_images = sorted(os.listdir(test_dir))
        test_images = [os.path.join(test_dir, img) for img in test_images]
        test_loader = classifier.create_dataloader(test_images, batch)

        test_predictions = classifier.infer_dataloader(test_loader, top_N)

        rdf = dataframe_results(test_images, test_predictions, categories, top_N)

        rdf.to_csv(f"result/tables/{time_stamp}_{model_folder}.csv", sep=",", index=False)


