from classifier import *

# # weighting priorities
# category_map = {
#     1: {
#         "PHOTO": ["PHOTO", "PHOTO_L"]
#     },
#     2: {
#         "DRAW": ["DRAW", "DRAW_L"]
#     },
#     3: {
#         "LINE": ["LINE_HW", "LINE_P", "LINE_T"]
#     },
#     4: {
#         "TEXT": ["TEXT", "TEXT_HW", "TEXT_P", "TEXT_T"]
#     }
# }


# base_dir = "/lnet/work/people/lutsai/pythonProject/OCR/ltp-ocr/trans"

data_dir = f'/lnet/work/people/lutsai/pythonProject/pages/train_final'

test_dir = f'/lnet/work/people/lutsai/pythonProject/pages/testing'

top_N = 5
seed = 420
max_categ = 1400
batch = 16
test_size = 0.1
epochs = 2
log_step = 10

Training = False

model_folder = "model_1119_3"

if Training:
    total_files, total_labels, categories = collect_images(data_dir, max_categ)

    (trainfiles, testfiles,
     trainLabels, testLabels) = train_test_split(total_files,
                                                 np.array(total_labels),
                                                 test_size=test_size,
                                                 random_state=seed,
                                                 stratify=np.array(
                                                     total_labels))

    # Initialize the classifier
    classifier = ImageClassifier(checkpoint="google/vit-base-patch16-224", num_labels=len(categories))

    train_loader = classifier.process_images(trainfiles, trainLabels, batch, True)
    eval_loader = classifier.process_images(testfiles, testLabels, batch, False)

    classifier.train_model(train_loader, eval_loader, output_dir="./model_output", num_epochs=epochs, logging_steps=log_step)
else:
    categories = sorted(os.listdir(data_dir))
    print(f"Category input directories found: {categories}")

    # Initialize the classifier
    classifier = ImageClassifier(checkpoint="google/vit-base-patch16-224", num_labels=len(categories))

    classifier.load_model(f"model/{model_folder}")

test_images = sorted(os.listdir(test_dir))
test_images = [os.path.join(test_dir, img) for img in test_images]
test_loader = classifier.create_dataloader(test_images, batch)

test_predictions = classifier.infer_dataloader(test_loader, top_N)

rdf = dataframe_results(test_images, test_predictions, categories, top_N)

rdf.to_csv(f"result/tables/{model_folder}.csv", sep=",", index=False)
