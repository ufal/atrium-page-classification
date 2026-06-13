import argparse

import configparser
import math

from classifier import *
from yolo_classifier import YOLOClassifier
import time
from huggingface_hub import create_branch, delete_branch
from atrium_paradata import ParadataLogger
from parallel_best import run_best_models  # [add] memory-aware best-models engine + averaging

if __name__ == "__main__":
    # Initialize the parser
    config = configparser.ConfigParser()
    # Read the configuration file
    config.read('config.txt')


    revision_to_base_model = {
        "v10.": "microsoft/dit-large-finetuned-rvlcdip",
        "v11.": "microsoft/dit-large",
        "v12.": "timm/tf_efficientnetv2_m.in21k_ft_in1k",
        "v1.3": "timm/tf_efficientnetv2_s.in21k",
        "v2.3": "google/vit-base-patch16-224",
        "v3.": "google/vit-base-patch16-384",
        "v4.": "timm/tf_efficientnetv2_l.in21k_ft_in1k",
        "v5.": "google/vit-large-patch16-384",
        "v6.": "timm/regnety_120.sw_in12k_ft_in1k",
        "v7.": "timm/regnety_160.swag_ft_in1k",
        "v8.": "timm/regnety_640.seer",
        "v9.": "microsoft/dit-base-finetuned-rvlcdip",
    }

    revision_best_models = {
        "v1.3": "timm/tf_efficientnetv2_m.in21k_ft_in1k",
        "v2.3": "google/vit-base-patch16-224",
        "v3.3": "google/vit-base-patch16-384",
        "v4.3": "timm/regnety_160.swag_ft_in1k",
        "v5.3": "google/vit-large-patch16-384",
        # "v6.3": "timm/regnety_640.seer",
    }

    def_categ = ["DRAW", "DRAW_L", "LINE_HW", "LINE_P", "LINE_T", "PHOTO", "PHOTO_L", "TEXT", "TEXT_HW", "TEXT_P",
                 "TEXT_T"]

    seed = config.getint('SETUP', 'seed')
    batch = config.getint('SETUP', 'batch')  # depends on GPU/CPU capabilities
    top_N = config.getint('SETUP', 'top_N')  # top N predictions, 3 is enough, 11 for "raw" scores (most scores are 0)

    config_base_model = config.get('SETUP', 'base_model')  # do not change
    config_format = config.get('SETUP', 'files_format')

    raw = config.getboolean('SETUP', 'raw')
    inner = config.getboolean('SETUP', 'inner')

    Training = config.getboolean('TRAIN', 'Training')
    Testing = config.getboolean('TRAIN', 'Testing')
    HF = config.getboolean('HF', 'use_hf')
    hf_version = config.get("HF", "revision")

    # FIX: use getint so the default is an int, not a str "0"
    cross_runs = config.getint("TRAIN", "cross_runs")

    # setting main to latest version by default
    # hf_version = hf_version if hf_version != 'main' else config.get('HF', 'latest')

    config_model_name_local = f"model_{hf_version.replace('.', '')}"
    model_dir = config.get('OUTPUT', 'FOLDER_MODELS')
    config_model_path = f"{model_dir}/{config_model_name_local}"

    config_input_dir = config.get('INPUT', 'FOLDER_INPUT')
    chunk_size = config.getint('INPUT', 'chunk_size')  # number of batches to process and save at once
    config_chunking = config.getboolean('INPUT', 'chunking')

    # cur = Path.cwd()  # directory with this script
    cur = Path(__file__).resolve().parent  # directory with this script
    output_dir = Path(config.get('OUTPUT', 'FOLDER_RESULTS'))
    cp_dir = Path(config.get('OUTPUT', 'FOLDER_CPOINTS'))

    time_stamp = time.strftime("%Y%m%d-%H%M")  # for results files

    parser = argparse.ArgumentParser(description='Page sorter based on ViT / YOLO-cls')
    parser.add_argument('-f', "--file", type=str, default=None, help="Single page image path")
    parser.add_argument('-ff', "--file_format", type=str, default=config_format,
                        help="File format to look for in the directory (e.g., png or jpeg)")
    parser.add_argument('-d', "--directory", type=str, default=None, help="Path to folder with unprocessed pages")
    parser.add_argument('-m', "--model", type=str, default=config_model_path,
                        help="Path to the folder with model subfolders")
    parser.add_argument('-b', "--base", type=str, default=config_base_model, help="Repository of the base model")
    parser.add_argument('-rev', "--revision", type=str, default=None,
                        help="HuggingFace revision (e.g. `main`, `vN.0` or `vN.M`)")
    parser.add_argument('-tn', "--topn", type=int, default=top_N,
                        help="Number of the best result categories to consider")
    parser.add_argument("--dir", help="Process whole directory (if -d not used) but input set in CONFIG",
                        action="store_true")
    parser.add_argument("--chunk", default=config_chunking, help="Process input directory and write predictions in chunks", action="store_true")
    parser.add_argument("--inner", help="Process nested folders of the given directory (FALSE by default)",
                        default=inner, action="store_true")
    parser.add_argument("--train", help="Training model", default=Training, action="store_true")
    parser.add_argument("--eval", help="Evaluating model", default=Testing, action="store_true")
    parser.add_argument("--hf", help="Use model and processor from the HuggingFace repository", default=HF,
                        action="store_true")
    parser.add_argument("--raw", help="Output raw scores for all categories", default=raw, action="store_true")
    parser.add_argument("--best",
                        help=f"Output all ({len(revision_best_models.keys())}) best models' scores. Result is automatically averaged into a final TOP-N CSV.",
                        default=False, action="store_true")
    parser.add_argument("--parallel",
                        help="Enable memory-aware grouped parallel execution when running with --best (requires CUDA).",
                        default=False, action="store_true")

    # [add] Averaging open-choice flags
    parser.add_argument("--no-average-best", action="store_true",
                        help="Skip automatically averaging the results when running with --best.")
    parser.add_argument("--save-intermediates", action="store_true",
                        help="Save the individual Top-N CSVs for each model during a --best run.")

    parser.add_argument("--folds", type=int, default=cross_runs,
                        help="Number of folds for cross-validation with 80/10/10 split. Default is 0 (no cross-validation).")
    parser.add_argument("--average", help="Averaging existing fold models", action="store_true")
    parser.add_argument("-ap", "--average_pattern", type=str, default=None,
                        help="Pattern for models weights to average (e.g., 'model_v4')")
    # --- YOLO arguments ---
    parser.add_argument(
        "--yolo",
        help="Use YOLO-cls model instead of ViT/CNN (overrides --base and --revision)",
        action="store_true",
        default=config.getboolean("YOLO", "use_yolo", fallback=False),
    )
    parser.add_argument(
        "--yolo_base",
        type=str,
        default=config.get("YOLO", "yolo_base", fallback="yolov8s-cls.pt"),
        help="YOLO base weights identifier (e.g. yolov8s-cls.pt)",
    )

    args = parser.parse_args()

    input_dir = Path(config_input_dir) if args.directory is None else Path(args.directory)
    Training, top_N, raw, chunked_result_record = args.train, args.topn, args.raw, args.chunk
    args.folds = 0 if not args.train else args.folds
    args.average = False if args.average_pattern is None else args.average

    # ── FIX 6: resolve YOLO name/path BEFORE paradata init so the log is accurate ──
    if args.yolo:
        yolo_tag = Path(args.yolo_base).stem.replace(".", "").replace("-", "")
        revision_model_name_local = f"model_{yolo_tag}"
        args.model = f"{model_dir}/{revision_model_name_local}"
    elif args.revision is None:  # using config file revision
        args.revision = hf_version
        args.base = config_base_model

        if args.model != config_model_path:
            revision_model_name_local = Path(args.model).name
        else:
            args.model = config_model_path
            revision_model_name_local = config_model_name_local

    else:  # using command line argument revision from flag --revision / -rev
        if not any(args.revision.startswith(key) for key in revision_to_base_model.keys()):
            raise ValueError(
                f"Revision {args.revision} is not supported. Available revisions: {list(revision_to_base_model.keys())}")

        revision_model_name_local = f"model_{args.revision.replace('.', '')}"
        args.model = f"{model_dir}/{revision_model_name_local}"
        rev_code = key = next(key for key in revision_to_base_model.keys() if args.revision.startswith(key))

        if args.base != config_base_model:  # flag argument provided
            print(
                f"Base model {config_base_model} does not match the revision {args.revision}. Using {revision_to_base_model[rev_code]} instead.")
            args.base = revision_to_base_model[rev_code]
        else:
            print(f"Using base model\t{config_base_model} from CONFIG,\trevision\t{args.revision}.")

        # Warn if revision="main" since revision_to_base_model lookup will fall through silently
        if args.revision == "main":
            print(f"WARNING: revision='main' — base model lookup fell back to config value '{config_base_model}'. "
                  f"Consider specifying an explicit version tag.")

    # ── paradata init ─────────────────────────────────────────────────────────
    # Placed AFTER full arg resolution (including YOLO path) so logged paths are accurate.
    _paradata_cfg = {
        # argparse / config.txt values – extend as needed
        "model_path":    args.model      if hasattr(args, "model")      else config.get("SETUP", "model",    fallback=""),
        # FIX: was args.rev (AttributeError) – correct attribute name is args.revision
        "revision":      args.revision   if hasattr(args, "revision")   else config.get("HF",    "revision", fallback=""),
        "base_model":    args.yolo_base  if args.yolo else config.get("SETUP", "base_model", fallback=""),
        "top_n":         args.topn       if hasattr(args, "topn")       else config.get("SETUP", "top_n",    fallback=""),
        "batch_size":    config.get("SETUP", "batch",       fallback=""),
        "input_path":    str(args.file or args.directory or config.get("INPUT", "folder", fallback="")),
        "inner_dirs":    config.get("SETUP", "inner",       fallback=""),
        "file_format":   args.file_format if hasattr(args, "file_format") else "png",
        "mode":          "file" if (hasattr(args, "file") and args.file) else "directory",
        "raw_output":    str(getattr(args, "raw",      False)),
        "best_models":   str(getattr(args, "best",     False)),
        "parallel_best": str(getattr(args, "parallel", False)),
        "yolo":          str(args.yolo),
    }
    _paradata_logger = ParadataLogger(
        program="page-classification",
        config=_paradata_cfg,
        paradata_dir=str(output_dir / "paradata"),
        output_types=["csv", "png"],
        config_dir=str(cur),  # cur = Path(__file__).resolve().parent — find para_config.txt reliably
    )
    # ── end paradata init ─────────────────────────────────────────────────────

    print("Arguments:")
    for arg in vars(args):
        if getattr(args, arg) is not None and getattr(args, arg) != False and getattr(args, arg) != 0:
            print(arg, "\t=\t", getattr(args, arg))

    # locally creating new directory paths instead of context.txt variables loaded with mistakes
    if not output_dir.is_dir():
        os.makedirs(output_dir)

        os.makedirs(f"{output_dir}/tables")
        os.makedirs(f"{output_dir}/plots")

    if not cp_dir.is_dir():
        os.makedirs(cp_dir)

    if not Path(model_dir).is_dir():
        os.makedirs(model_dir)

    # ── data loading (train / eval) ───────────────────────────────────────────
    if args.train or args.eval:
        epochs = config.getint("TRAIN", "epochs")
        max_categ = config.getint("TRAIN", "max_categ")  # max number of category samples
        log_step = config.getint("TRAIN", "log_step")
        test_size = config.getfloat("TRAIN", "test_size")
        learning_rate = config.getfloat("TRAIN", "lr")

        data_dir = config.get("TRAIN", "FOLDER_PAGES")

        if args.train:
            total_files, total_labels, categories = collect_images(data_dir)
            # Training consumes the CC BY-NC-SA 4.0 LINDAT dataset, which makes
            # the trained model (and this run's outputs) non-commercial + share-alike.
            # Inference-only runs never reach here and stay MIT.
            _paradata_logger.log_component("lindat_dataset")

    # ── FIX 1: eval data loaded once; classifier created once below ───────────
    if args.eval:
        data_dir = config.get("EVAL", "FOLDER_PAGES")
        testfiles, testLabels, categories = collect_images(data_dir)
        # categories is now set from eval data; classifier constructed below

    # ── single classifier instantiation (YOLO or standard) ───────────────────
    if args.yolo:
        # categories may have been set from training/eval data above; fall back
        # to defaults if neither --train nor --eval was requested
        if not (args.train or args.eval):
            categories = def_categ
        print(f"[YOLO] Using YOLO-cls backend: {args.yolo_base}")
        classifier = YOLOClassifier(
            checkpoint=args.yolo_base,
            num_labels=len(categories),
            categories=categories,
            store_dir=str(cp_dir),
            imgsz=config.getint("YOLO", "yolo_imgsz", fallback=224),
        )
    else:
        if not (args.train or args.eval):
            categories = def_categ
        print(f"Category input directories found: {categories}")
        classifier = ImageClassifier(
            checkpoint=args.base,
            num_labels=len(categories),
            store_dir=str(cp_dir),
        )

    # ── training ──────────────────────────────────────────────────────────────
    if args.train:
        if args.yolo:
            # ── FIX 2 / FIX 3: YOLO single-run training (no DataLoader) ──────
            (trainfiles, valfiles, testfiles,
             trainLabels, valLabels, testLabels) = split_data_80_10_10(
                total_files, total_labels, seed, max_categ
            )
            print(f"[YOLO] Training on {len(trainfiles)} images, validating on {len(valfiles)} images")

            # ── YOLO-specific training hyperparameters from [YOLO] config ──────
            # yolo_epochs / yolo_lr0 override the generic [TRAIN] epochs / lr when
            # set; yolo_lr0 == 0 is treated as "unset" → fall back to TRAIN.lr
            # (lr0=0 would be a degenerate learning rate).
            yolo_epochs   = config.getint("YOLO", "yolo_epochs", fallback=epochs)
            yolo_patience = config.getint("YOLO", "yolo_patience", fallback=100)
            yolo_lr0_cfg  = config.getfloat("YOLO", "yolo_lr0", fallback=0.0)
            yolo_lr0      = yolo_lr0_cfg if yolo_lr0_cfg > 0 else learning_rate
            yolo_dropout  = config.getfloat("YOLO", "yolo_dropout", fallback=0.0)
            yolo_cache    = config.getboolean("YOLO", "yolo_cache", fallback=False)

            classifier.train_model(
                trainfiles=list(trainfiles),
                trainLabels=trainLabels,
                valfiles=list(valfiles),
                valLabels=valLabels,
                out_model=revision_model_name_local,
                num_epochs=yolo_epochs,
                batch_size=batch,
                learning_rate=yolo_lr0,
                output_dir="./yolo_output",
                logging_steps=log_step,
                patience=yolo_patience,
                dropout=yolo_dropout,
                cache=yolo_cache,
            )

        else:
            # ── standard ViT/CNN path ─────────────────────────────────────────
            if args.folds > 0:
                for i in range(args.folds):
                    print(f"--- Cross-Validation Fold {i + 1}/{args.folds} ---")
                    fold_seed = seed + i  # Use a different seed for each fold

                    (trainfiles, valfiles, testfiles,
                     trainLabels, valLabels, testLabels) = split_data_80_10_10(total_files, total_labels, fold_seed,
                                                                               max_categ)

                    # record datasets
                    with open(f"{output_dir}/stats/{time_stamp}_{revision_model_name_local}_FOLD_{i + 1}_DATASETS.txt",
                              "w") as f:
                        f.write(f"Training set ({len(trainfiles)} images):\n")
                        for file in trainfiles:
                            f.write(f"{file}\n")
                        f.write(f"\nValidation set ({len(valfiles)} images):\n")
                        for file in valfiles:
                            f.write(f"{file}\n")
                        f.write(f"\nTest set ({len(testfiles)} images):\n")
                        for file in testfiles:
                            f.write(f"{file}\n")

                    # Initialize a fresh classifier for each fold
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
                        out_model=f"{revision_model_name_local}{i + 1}",
                        num_epochs=epochs,
                        learning_rate=learning_rate,
                        logging_steps=log_step
                    )

                    # Evaluate on the test set for the current fold
                    print(f"--- Evaluating on test set for fold {i + 1} ---")
                    test_predictions, raw_prediction = classifier.infer_dataloader(test_loader, top_N, raw)
                    test_labels_indices = np.argmax(testLabels, axis=-1).tolist()

                    print("=" * 40)
                    print(
                        f"TEST SET's correct percentage:\t{round(100 * sum([1 for true, pred in zip(test_labels_indices, test_predictions) if true == pred]) / len(test_labels_indices), 2)}%")
                    print("=" * 40)
                    print(classification_report(test_labels_indices, test_predictions,
                                                target_names=categories, labels=list(range(len(categories))),
                                                zero_division=0))

                    rdf, raw_df = dataframe_results(testfiles, test_predictions, categories, top_N, raw_prediction)
                    rdf["TRUE"] = [categories[label] for label in test_labels_indices]
                    rdf.sort_values(['FILE', 'PAGE'], ascending=[True, True], inplace=True)
                    rdf.to_csv(f"{output_dir}/tables/{time_stamp}_{len(test_labels_indices)}_{revision_model_name_local}_TEST_FOLD_{i + 1}.csv",
                               index=False)

                    if raw:
                        raw_df["TRUE"] = [categories[label] for label in test_labels_indices]
                        raw_df.sort_values(categories, ascending=[False] * len(categories), inplace=True)
                        raw_df.to_csv(
                            f"{output_dir}/tables/{time_stamp}_{len(test_labels_indices)}_{revision_model_name_local}_TEST_RAW_FOLD_{i + 1}.csv",
                            index=False)

                    print(f"Test results for fold {i + 1} saved.")

            else:
                # single-run (no cross-validation)
                (trainfiles, valfiles, testfiles,
                 trainLabels, valLabels, testLabels) = split_data_80_10_10(total_files, total_labels, seed, max_categ)

                train_loader = classifier.process_images(trainfiles, trainLabels, batch, True)
                eval_loader = classifier.process_images(valfiles, valLabels, batch, False)

                print(f"Training on {len(trainfiles)} images, evaluating on {len(valfiles)} images")
                print(f"Base model: {args.base}, local model name: {revision_model_name_local}")
                classifier.train_model(
                    train_loader,
                    eval_loader,
                    output_dir="./model_output",
                    out_model=revision_model_name_local,
                    num_epochs=epochs,
                    learning_rate=learning_rate,
                    logging_steps=log_step
                )

    # ── HuggingFace hub ───────────────────────────────────────────────────────
    if args.hf:
        # ── FIX 4: YOLO models are not on HF hub ─────────────────────────────
        if args.yolo:
            print("[YOLO] --hf is not supported for YOLO models. Skipping hub download.")
        else:
            # ----------------------------------------------
            # ----- UNCOMMENT for pushing to HF repo -------
            # ----------------------------------------------
            #print(f"Deleting {args.revision} branch")
            #delete_branch(config.get("HF", "repo_name"), repo_type="model", branch=args.revision,
            #              token=config.get("HF", "token"))
            # print(f"Creating fresh {args.revision} branch")
            # create_branch(config.get("HF", "repo_name"), repo_type="model", branch=args.revision, exist_ok=True,
            #               token=config.get("HF", "token"))
            #
            # print(f"Loading {args.model} model")
            #
            # classifier.load_model(str(args.model))
            #
            # classifier.push_to_hub(str(args.model), config.get("HF", "repo_name"), False, config.get("HF", "token"),
            #                        config.get("HF", "revision"))
            # ----------------------------------------------

            # loading from repo
            classifier.load_from_hub(config.get("HF", "repo_name"), args.revision)

            hf_model_name_local = f"model_{args.revision.replace('.', '')}"
            hf_model_path = f"{model_dir}/{hf_model_name_local}"

            classifier.save_model(hf_model_path)

            classifier.load_model(hf_model_path)

    else:
        if not args.average and not args.best:
            classifier.load_model(args.model)

    # ── evaluation ────────────────────────────────────────────────────────────
    if args.eval:
        print(f"\tModel loaded:\t{revision_model_name_local}\t{args.model}")
        print(f"\t*\t--- Evaluating on the test set ({len(testfiles)} images) ---")

        # ── FIX 5: YOLO uses infer_dataloader(image_paths); ViT uses DataLoader ──
        if args.yolo:
            yolo_loader = classifier.create_dataloader(list(testfiles), batch)
            eval_predictions, raw_prediction = classifier.infer_dataloader(
                yolo_loader, top_N, raw
            )
        else:
            test_loader = classifier.process_images(testfiles, testLabels, batch, False)
            eval_predictions, raw_prediction = classifier.infer_dataloader(test_loader, top_N, raw)

        test_labels_indices = np.argmax(testLabels, axis=-1).tolist()

        rdf, raw_df = dataframe_results(testfiles, eval_predictions, categories, top_N, raw_prediction)
        number_of_rows = len(rdf.index)

        rdf["TRUE"] = [categories[i] for i in test_labels_indices]
        rdf.sort_values(['FILE', 'PAGE'], ascending=[True, True], inplace=True)
        rdf.to_csv(
            f"{output_dir}/tables/{time_stamp}_{number_of_rows}_{revision_model_name_local}_TOP-{top_N}_EVAL.csv",
            sep=",", index=False)
        print(f"Evaluation results for TOP-{top_N} predictions are recorded into {output_dir}/tables/ directory")

        if raw:
            raw_df["TRUE"] = [categories[i] for i in test_labels_indices]
            raw_df.sort_values(categories, ascending=[False] * len(categories), inplace=True)
            raw_df.to_csv(f"{output_dir}/tables/{time_stamp}_{number_of_rows}_{revision_model_name_local}_EVAL_RAW.csv",
                          sep=",", index=False)
            print(f"RAW Evaluation results are recorded into {output_dir}/tables/ directory")

        confusion_plot(eval_predictions,
                       test_labels_indices,
                       categories,
                       revision_model_name_local,
                       top_N,
                       output_dir=str(output_dir))
        _paradata_logger.log_success("png")

        print(f"\t*\t--- Evaluation of {revision_model_name_local} completed ---")

    # ── model averaging (ViT/CNN only) ────────────────────────────────────────
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
                    model_name_pattern=str(args.average_pattern),
                    base_model=base_model_for_pattern,
                    num_labels=len(categories)
                )
                print(f"Averaged model saved to: {averaged_path}")
            except Exception as e:
                print(f"Error averaging models: {e}")
        else:
            print(f"Could not determine base model for pattern: {args.average_pattern}")

    # ── inference ─────────────────────────────────────────────────────────────
    _total_inputs = 0
    try:
        if args.file is not None:
            _total_inputs += 1
            if not args.best:
                pred_scores = classifier.top_n_predictions(args.file, top_N)

                labels = [categories[i[0]] for i in pred_scores]
                scores = [round(i[1], 3) for i in pred_scores]

                print(f"File {args.file} predicted:")
                for lab, sc in zip(labels, scores):
                    print(f"\t{lab}:  {round(sc * 100, 2)}%")
                _paradata_logger.log_success("csv")
            else:
                # --best is a ViT-only feature; guard against YOLO misuse
                if args.yolo:
                    print("[YOLO] --best is not supported for YOLO models. Run without --best.")
                else:
                    avg_csv_path = run_best_models(
                        test_images=[args.file],
                        categories=categories,
                        revision_best_models=revision_best_models,
                        model_dir=str(model_dir),
                        cp_dir=str(cp_dir),
                        batch=batch,
                        top_N=args.topn,
                        output_dir=str(output_dir),
                        time_stamp=time_stamp,
                        paradata_logger=_paradata_logger,
                        parallel=args.parallel,
                        save_intermediates=args.save_intermediates,
                        average_best=not args.no_average_best,
                    )

                    if avg_csv_path:
                        avg_df = pd.read_csv(avg_csv_path)
                        print(f"\nFile {args.file} — averaged predictions from {len(revision_best_models)} best models:")
                        for _, row in avg_df.iterrows():
                            for n in range(1, args.topn + 1):
                                cls_col, scr_col = f"CLASS-{n}", f"SCORE-{n}"
                                if cls_col in avg_df.columns and pd.notna(row.get(cls_col)) and row.get(cls_col) != "":
                                    print(f"\t{row[cls_col]}:  {round(float(row[scr_col]) * 100, 2)}%")

        if args.dir or args.directory is not None:
            print(f"Starting inference of {input_dir}, saving results in chunks...")

            if args.inner:
                test_images = sorted(directory_scraper(Path(input_dir), args.file_format))
            else:
                test_images = sorted(os.listdir(input_dir))
                test_images = [os.path.join(input_dir, img) for img in test_images]

            _total_inputs = len(test_images)

            if not args.best:

                if not chunked_result_record:  # all at once (no chunking)
                    test_loader = classifier.create_dataloader(test_images, batch)

                    test_predictions, raw_prediction = classifier.infer_dataloader(test_loader, top_N, raw)
                    rdf, raw_df = dataframe_results(test_images,
                                                    test_predictions,
                                                    categories,
                                                    top_N,
                                                    raw_prediction)

                    _paradata_logger.log_success("csv", len(rdf.index))

                    rdf.sort_values(['FILE', 'PAGE'], ascending=[True, True], inplace=True)
                    rdf.to_csv(f"{output_dir}/tables/{time_stamp}_{revision_model_name_local}_TOP-{top_N}.csv", sep=",",
                               index=False)
                    print(f"Results for TOP-{top_N} predictions are recorded into {output_dir}/tables/ directory")

                    if raw:
                        raw_df.sort_values(categories, ascending=[False] * len(categories), inplace=True)
                        raw_df.to_csv(f"{output_dir}/tables/{time_stamp}_{revision_model_name_local}_RAW.csv", sep=",",
                                      index=False)
                        print(f"RAW Results are recorded into {output_dir}/tables/ directory")

                else:  # chunked processing and saving
                    print(f"Starting inference of {input_dir}, saving results in chunks of {chunk_size * batch} images...")

                    total = len(test_images)
                    chunks = math.ceil(total / chunk_size)

                    # daily date-based filenames (YYYYMMDD)
                    date_stamp = time.strftime('%Y%m%d')
                    top_out_path = f"{output_dir}/tables/{date_stamp}_{revision_model_name_local}_TOP-{top_N}.csv"
                    raw_out_path = f"{output_dir}/tables/{date_stamp}_{revision_model_name_local}_RAW.csv"

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

                        _paradata_logger.log_success("csv", len(rdf_chunk.index))

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

                    # ensure ascending order in the final daily files
                    if os.path.exists(top_out_path):
                        final_top_df = pd.read_csv(top_out_path)
                        final_top_df.sort_values(['FILE', 'PAGE'], ascending=[True, True], inplace=True)
                        final_top_df.to_csv(top_out_path, sep=",", index=False)
                        print(f"Final TOP-{top_N} daily file sorted by FILE and PAGE.")

                    # ensure raw_df.sort_values(categories, ascending=[False] * len(categories), inplace=True)
                    if raw and os.path.exists(raw_out_path):
                        final_raw_df = pd.read_csv(raw_out_path)
                        final_raw_df.sort_values(categories, ascending=[False] * len(categories), inplace=True)
                        final_raw_df.to_csv(raw_out_path, sep=",", index=False)
                        print(f"Final RAW daily file sorted by category scores.")

            else:  # args.best == True
                if args.yolo:
                    print("[YOLO] --best is not supported for YOLO models. Run without --best.")
                else:
                    avg_csv_path = run_best_models(
                        test_images=test_images,
                        categories=categories,
                        revision_best_models=revision_best_models,
                        model_dir=str(model_dir),
                        cp_dir=str(cp_dir),
                        batch=batch,
                        top_N=top_N,
                        output_dir=str(output_dir),
                        time_stamp=time_stamp,
                        paradata_logger=_paradata_logger,
                        parallel=args.parallel,
                        save_intermediates=args.save_intermediates,
                        average_best=not args.no_average_best,
                    )
                    if avg_csv_path:
                        print(f"Averaged results for TOP-{top_N} predictions → {avg_csv_path}")

    finally:
        _paradata_logger.finalize(_total_inputs)