import argparse
from huggingface_hub import create_branch, delete_branch

import configparser
from classifier import *
from utils import *
import time

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")


    revision_to_base_model = {
        "v1.1.": "ViT-B/16",
        "v1.2.": "ViT-B/32",
        "v2.1.": "ViT-L/14",
        "v2.2.": "ViT-L/14@336px",
    }


    revision_best_models = {
        "v1.1.3.1": "ViT-B/16",
        "v1.1.3.3": "ViT-B/16",
        "v1.1.3.4": "ViT-B/16",
        "v1.1.3.6": "ViT-B/16",
        "v1.1.3.7": "ViT-B/16",
        # "v1.2.3": "ViT-B/32",
        # "v2.1.3.1": "ViT-L/14",
        # "v2.2.3.4": "ViT-L/14@336px",
    }

    category_sufix = {
        "113": "average",
        "123": "average",
        "213": "average",
        "223": "average",
        "31": "init",
        "32": "details",
        "33": "extra",
        "34": "gemini",
        "35": "gpt",
        "36": "large",
        "37": "mid",
        "38": "min",
        "39": "short",
    }



    # Initialize the parser
    config = configparser.ConfigParser()
    # Read the configuration file
    config.read('config.txt')

    def_categ = ["DRAW", "DRAW_L", "LINE_HW", "LINE_P", "LINE_T", "PHOTO", "PHOTO_L", "TEXT", "TEXT_HW", "TEXT_P", "TEXT_T"]

    seed = config.getint('SETUP', 'seed')
    batch = config.getint('SETUP', 'batch')  # depends on GPU/CPU capabilities
    top_N = config.getint('SETUP', 'top_N')  # top N predictions, 3 is enough, 11 for "raw" scores (most scores are 0)

    base_model = config.get('SETUP', 'base_model')  # do not change
    config_format = config.get('SETUP', 'files_format')  # input image format, e.g. PNG
    raw = config.getboolean('SETUP', 'raw')
    avg = config.getboolean('SETUP', 'avg')  # average scores from multiple category description files

    categ_prefix = config.get('SETUP', 'categories_prefix')  # prefix for category description files
    categ_file = config.get('SETUP', 'categories_file')  # file with category descriptions
    categ_directory = config.get('SETUP', 'description_folder')  # directory with category description files

    Training = config.getboolean('TRAIN', 'Training')
    Testing = config.getboolean('TRAIN', 'Testing')
    HF = config.getboolean('HF', 'use_hf')
    hf_version = config.get("HF", "revision")

    # setting main to latest version by default
    # hf_version = hf_version if hf_version != 'main' else config.get('HF', 'latest')

    model_name_local = f"{base_model.replace('/', '').replace('@', '-')}_rev_{hf_version.replace('.', '')}"
    hf_models_directory = config.get('OUTPUT', 'FOLDER_MODELS')
    model_path = Path(f"{hf_models_directory}/{model_name_local}")

    test_dir = config.get('INPUT', 'FOLDER_INPUT')
    chunk_size = config.getint('INPUT', 'chunk_size')

    epochs = config.getint("TRAIN", "epochs")
    max_categ = config.getint("TRAIN", "max_categ")  # max number of category samples
    max_categ_e = config.getint("TRAIN", "max_categ_e")  # max number of category samples for evaluation
    log_step = config.getint("TRAIN", "log_step")
    test_size = config.getfloat("TRAIN", "test_size")
    learning_rate = config.getfloat("TRAIN", "lr")

    zero_shot = config.getboolean('SETUP', 'zero_shot')  # zero-shot prediction without training
    visualize = config.getboolean('SETUP', 'visualize')  # visualize model accuracy statistics
    download_root = config.get('SETUP', 'model_storage')  # root directory for downloading datasets

    # cur = Path.cwd()  # directory with this script
    cur = Path(__file__).resolve().parent  # directory with this script
    output_dir = Path(config.get('OUTPUT', 'FOLDER_RESULTS'))
    cp_dir = Path(config.get('OUTPUT', 'FOLDER_CPOINTS'))

    time_stamp = time.strftime("%Y%m%d-%H%M")  # for results files

    parser = argparse.ArgumentParser(description='Page sorter based on ViT')

    # Prediction arguments
    parser.add_argument('-f', "--file", type=str, default=None, help="Single PNG page path for prediction.")
    parser.add_argument('-ff', "--file_format", type=str, default=config_format, help="File format to look for in the directory (e.g., png or jpeg)")
    parser.add_argument('-d', "--directory", type=str, default=None,
                        help="Path to folder with PNG pages for prediction.")
    parser.add_argument("--dir", help="Predict a whole directory of images recursively.", action="store_true")
    parser.add_argument('-m', "--model", type=str, default=base_model,
                        help="CLIP model name to use. Default is ViT-B/32.")

    # Training arguments
    parser.add_argument("--train", action="store_true", help="Run model fine-tuning.")
    parser.add_argument('--epochs', type=int, default=epochs, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=learning_rate, help='Learning rate for the optimizer.')
    parser.add_argument('--batch_size', type=int, default=batch, help='Batch size for training and evaluation.')
    parser.add_argument('-mc', "--max_categ", type=int, default=max_categ,
                        help="Maximum number of samples per category for training.")
    parser.add_argument('-mce', "--max_categ_eval", type=int, default=max_categ_e,
                        help="Maximum number of samples per category for evaluation.")
    parser.add_argument("--safe", action="store_true", help="Safely load images skipping the corrupted ones.")


    # Category file arguments
    parser.add_argument('--cat_prefix', type=str, default=categ_prefix,
                        help='Prefix for category description TSV files.')
    parser.add_argument('-cc', '--cat_csv', type=str, default=categ_file,
                        help='Category descriptions file in TSV or CSV format.')
    parser.add_argument('--cat_dir', type=str, default=categ_directory, help='Directory with category description files.'),
    parser.add_argument('--avg', action='store_true', default=avg, help='Average scores from multiple category description files.')
    parser.add_argument("--best",
                        help=f"Output all ({len(revision_best_models.keys())}) best models' scores (more time needed, and NO chunk, raw, or top_N>1 is available)",
                        default=raw, action="store_true")
    parser.add_argument('--zero_shot', action='store_true', default=zero_shot, help='Perform zero-shot prediction (no training).')
    parser.add_argument('--vis', action='store_true',default=visualize, help='Visualize model accuracy statistics.')

    # Common arguments
    parser.add_argument('-tn', "--topn", type=int, default=top_N, help="Number of top result categories to consider.")

    # Evaluation arguments
    parser.add_argument("--eval", action="store_true", help="Evaluate a saved model.")
    parser.add_argument("--eval_dir", action="store_true", help="Evaluate a directory of saved models.")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to the saved model checkpoint (.pt file) or a model folder (inside `models` folder).")
    parser.add_argument("--model_dir", type=str, default=hf_models_directory,
                        help="Path to the directory of saved model checkpoints (.pt files) for evaluation.")

    parser.add_argument('-rev', "--revision", type=str, default=None, help="HuggingFace revision (e.g. `main`, `vN.0` or `vN.M`)")
    parser.add_argument("--hf", help="Use model and processor from the HuggingFace repository", default=HF, action="store_true")
    parser.add_argument("--raw", help="Output raw scores for each category", default=raw, action="store_true")


    args = parser.parse_args()

    input_dir = Path(test_dir) if args.directory is None else Path(args.directory)
    Training, top_N, raw, safety, categ_file = args.train, args.topn, args.raw, args.safe, args.cat_csv
    model_path = args.model_path if args.model_path is not None else model_path

    if args.revision is None: # using config file revision
        args.revision = hf_version
    else:
        if not any(link for link in revision_to_base_model if args.revision.startswith(link)):
            raise ValueError(f"Revision {args.revision} is not supported. Available revisions: {list(revision_to_base_model.keys())}")

        base_model = revision_to_base_model[args.revision[:len(revision_to_base_model.keys().__iter__().__next__())]]
        if args.model != base_model:
            print(f"Base model {args.base} does not match the revision {args.revision}. Using {base_model} instead.")
            args.model = base_model

    # locally creating new directory paths instead of context.txt variables loaded with mistakes
    if not output_dir.is_dir():
        os.makedirs(output_dir)

        os.makedirs(f"{output_dir}/tables")
        os.makedirs(f"{output_dir}/stats")
        os.makedirs(f"{output_dir}/plots")

    if not cp_dir.is_dir():
        os.makedirs(cp_dir)

    if not Path(hf_models_directory).is_dir():
        os.makedirs(hf_models_directory)

    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items()) if v
                  is not None and k not in (
                      "file", "directory", "dir", "eval", "train", "model_path", "model", "cat_prefix", "model_dir",
                      "eval_dir", "vis", "raw", "safe", "cat_dir", "hf", "cat_csv", "file_format")))
    ))

    args.logdir += f"-{args.model.replace('/', '_')}" if args.model else ""

    print("Arguments:")
    for arg in vars(args):
        if getattr(args, arg) is not None and getattr(args, arg) != False:
            print(arg, "\t=\t", getattr(args, arg))

    cur = Path(__file__).parent
    output_dir = cur / "result"
    output_dir.mkdir(exist_ok=True)

    cat_directory = str(cur / args.cat_dir)
    if not args.vis and not args.best and not args.eval_dir:
        clip_instance = CLIP(max_category_samples=args.max_categ, test_ratio=test_size,
                             eval_max_category_samples=args.max_categ_eval,
                             top_N=args.topn, model_name=args.model, device=device,
                             categories_tsv=args.cat_csv, seed=seed, input_format=args.file_format,
                             output_dir=str(output_dir), categories_dir=cat_directory,
                             model_dir=str(model_path), cp_dir=str(cp_dir), model_revision=args.revision.replace('.', ''),
                             cat_prefix=args.cat_prefix, avg=args.avg, zero_shot=args.zero_shot)

    data_dir = config.get("TRAIN", "FOLDER_PAGES")
    data_dir_eval = config.get("EVAL", "FOLDER_PAGES")

    categories = def_categ
    print(f"Category input directories found: {categories}")

    if model_path is None:
        model_name_local = f"model_{args.model.replace('/', '').replace('@', '-')}_rev_{args.revision.replace('.', '')}"
        model_path = Path(cp_dir.parent / args.model_dir / model_name_local)
        print(f"Working with a local folder \t{model_path}")
    else:
        print(f"Working with a local folder \t{model_path}")
        model_name_local = Path(model_path).stem

    model_cp_path = Path(cp_dir / f"{model_name_local}_{args.epochs}e.pt")
    print(f"Model checkpoints folder \t{model_cp_path}")

    if args.hf:
        # saving model to local path
        if model_path is None:
            model_name_local = f"{args.model.replace('/', '').replace('@', '-')}_rev_{args.revision.replace('.', '')}"
            model_path = Path(cp_dir.parent / args.model_dir / model_name_local)
            local_revision = args.revision
        else:
            model_name_local = Path(model_path).stem
            local_revision = args.model_path.split("_")[-1]

        hf_model_name_local = f"model_{config.get('HF', 'revision').replace('.', '')}"
        hf_model_path = f"{config.get('OUTPUT', 'FOLDER_MODELS')}/{hf_model_name_local}"

        # ----------------------------------------------
        # ----- UNCOMMENT for pushing to HF repo -------
        # ----------------------------------------------
        #print(f"Deleting branch {config.get('HF', 'revision')}")
        #delete_branch(config.get("HF", "repo_name"), repo_type="model", branch=config.get("HF", "revision"), token=config.get("HF", "token"))
        # print(f"Creating fresh branch {config.get('HF', 'revision')}")
        # create_branch(config.get("HF", "repo_name"), repo_type="model", branch=config.get("HF", "revision"),
        #              exist_ok=True,
        #              token=config.get("HF", "token"))
        # print(f"Loading local model from {model_path} for pushing to the HuggingFace hub {config.get('HF', 'repo_name')}")
        # clip_instance.load_model(str(model_path), local_revision)
        # print(f"Pushing model to the HuggingFace hub branch {config.get('HF', 'revision')}")
        # clip_instance.pushing_to_hub(config.get("HF", "repo_name"), False, config.get("HF", "token"),
        #                             config.get("HF", "revision"))
        # ----------------------------------------------


        # loading from repo
        clip_instance.load_from_hub(config.get("HF", "repo_name"), args.revision)

        clip_instance.save_model(hf_model_path)

        clip_instance.load_model(hf_model_path, config.get("HF", "revision"))

    if args.train:
        if not os.path.isdir(data_dir):
            raise ValueError(f"Train directory not found at: {data_dir}")
        if not os.path.isdir(data_dir_eval):
            print(f"Warning: Evaluation directory not found at: {data_dir_eval}. Using training directory for evaluation.")
            data_dir_eval = data_dir

        clip_instance.train(
            train_dir=data_dir,
            eval_dir=data_dir_eval if args.eval else None,
            log_dir=args.logdir,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            batch_size=args.batch_size,
        )
    elif args.vis:
        csv = output_dir / 'stats' / f"model_accuracies_total.csv"
        if not csv.exists():
            print(f"CSV file for visualization of all models not found at {csv}. Please run model evaluation first.")
        else:
            visualize_all_results(str(csv), str(output_dir / 'stats'))

        csv = output_dir / 'stats' / f"model_accuracies{'_zero' if args.zero_shot else ''}.csv"
        if not csv.exists():
            print(f"CSV file for visualization not found at {csv}. Please run model evaluation first.")
        else:
            visualize_results(str(csv), str(output_dir / 'stats'), args.zero_shot)


    if args.eval:
        if args.zero_shot:
            model_path_str = None
        else:
            model_path_str = args.model_path
            print(f"Model path provided: {model_path_str}")
            if model_path_str is None:
                if model_cp_path.is_file():
                    model_path_str = str(model_cp_path)
                else:
                    model_cp_path = Path(str(model_cp_path).replace("e.pt", "e.cp.pt"))

                    if not model_cp_path.is_file():
                        raise ValueError(
                            f"Model file or checkpoint {model_cp_path} are not found at default paths. Please provide a path using --model_path.")
                    else:
                        model_path_str = str(model_cp_path)


        if not os.path.isdir(data_dir_eval):
            print(f"Warning: Evaluation directory not found at: {data_dir_eval}. Using training directory for evaluation.")
            data_dir_eval = data_dir

        clip_instance.evaluate_saved_model(
            model_path=model_path_str,
            eval_dir=data_dir_eval,
            batch_size=args.batch_size
        )
    elif args.eval_dir:
        if not os.path.isdir(data_dir_eval):
            print(f"Warning: Evaluation directory not found at: {data_dir_eval}. Using training directory for evaluation.")
            data_dir_eval = data_dir
        if not os.path.isdir(args.model_dir):
            raise ValueError(f"Model directory not found at: {args.model_dir}. Please provide a valid path.")

        evaluate_multiple_models(
            model_dir=args.model_dir,
            eval_dir=data_dir_eval,
            categ_dir=cat_directory,
            batch_size=args.batch_size,
            device=device,
            cat_prefix=args.cat_prefix,
            vis=True,
            random_seed=seed,
            input_format=args.file_format,
            test_fraction=test_size,
            zero_shot=args.zero_shot,
            top_N=args.topn
        )
    elif args.zero_shot:  # New branch for zero-shot prediction
        if args.file:
            prediction = clip_instance.predict_single(args.file)
            print(f"Zero-shot prediction for {args.file}: {prediction}")
        elif args.dir:
            input_dir_pred = Path(args.directory) if args.directory is not None else cur / 'category_samples'
            clip_instance.predict_directory(str(input_dir_pred), raw=raw, chunk_size=chunk_size,)
        else:
            print("Please specify a file (-f) or a directory (-d) for zero-shot prediction.")
    else:
        if args.file:
            if not args.best:
                if args.topn > 1:
                    scores, labels = clip_instance.predict_top_N(args.file)
                    print(f"File {args.file} predicted:")
                    for lab, sc in zip(labels, scores):
                        print(f"\t{lab}:  {round(sc * 100, 2)}%")
                else:
                    prediction = clip_instance.predict_single_best(args.file)
                    print(f"Prediction for {args.file}:\n{prediction}")
            else:
                all_best_predictions = {}

                for rev, base_model in revision_best_models.items():
                    print(f"\nLoading best model for revision {rev} based on {base_model}...")
                    model_name_local = f"{base_model.replace('/', '').replace('@', '-')}_rev_{rev}"
                    hf_model_name_local = f"model_{rev.replace('.', '')}"
                    model_path = Path(f"{hf_models_directory}/{model_name_local}")
                    hf_model_path = Path(f"{hf_models_directory}/{hf_model_name_local}")

                    if hf_model_path.is_dir():
                        model_path = hf_model_path

                    temp_avg = ".3." not in rev  # averaging is not used for .3.* best models

                    # check if revision ends with one of the category sufix keys
                    categ_file = None
                    for sufix_key, sufix_value in category_sufix.items():
                        if rev.replace(".", "").endswith(sufix_key):
                            if temp_avg:
                                categ_file = f"TOTAL_{args.cat_prefix}.tsv"
                            else:
                                categ_file = f"{args.cat_prefix}_{sufix_value}.tsv"
                            print(f"Using category description file {categ_file} for revision {rev}")
                            break

                    category_csv_path = Path(cat_directory) / categ_file

                    clip_instance = CLIP(max_category_samples=args.max_categ, test_ratio=test_size,
                             eval_max_category_samples=args.max_categ_eval,
                             top_N=args.topn, model_name=base_model, device=device,
                             categories_tsv=str(category_csv_path), seed=seed, input_format=args.file_format,
                             output_dir=str(output_dir), categories_dir=cat_directory,
                             model_dir=str(model_path), cp_dir=str(cp_dir), model_revision=rev,
                             cat_prefix=args.cat_prefix, avg=temp_avg, zero_shot=False)

                    clip_instance.load_model(str(model_path), rev)

                    if args.topn > 1:
                        scores, labels = clip_instance.predict_top_N(args.file)
                        all_best_predictions[rev] = (labels, scores)
                    else:
                        prediction = clip_instance.predict_single_best(args.file)
                        all_best_predictions[rev] = (prediction, 1)

                print(f"\nFile {args.file} predictions from best models:")
                for rev, (labels, scores) in all_best_predictions.items():
                    printed = 0
                    print(f"\n--- Revision {rev} --- {revision_best_models[rev]} ---")
                    for lab, sc in zip(labels, scores):
                        if printed >= args.topn:
                            break
                        print(f"\t{lab}:  {round(sc * 100, 2)}%")
                        printed += 1

        if args.dir or args.directory is not None:
            if not args.best:
                clip_instance.predict_directory(str(input_dir), raw=raw, chunk_size=chunk_size)
            else:
                all_best_directory_output = {}

                for rev, base_model in revision_best_models.items():
                    print(f"\nLoading best model for revision {rev} based on {base_model}...")
                    model_name_local = f"{base_model.replace('/', '').replace('@', '-')}_rev_{rev.replace('.', '')}"
                    hf_model_name_local = f"model_{rev.replace('.', '')}"
                    model_path = Path(f"{hf_models_directory}/{model_name_local}")
                    hf_model_path = Path(f"{hf_models_directory}/{hf_model_name_local}")

                    if hf_model_path.is_dir():
                        model_path = hf_model_path

                    temp_avg = ".3." not in rev

                    categ_file = None
                    for sufix_key, sufix_value in category_sufix.items():
                        if rev.replace(".", "").endswith(sufix_key):
                            if temp_avg:
                                categ_file = f"TOTAL_{args.cat_prefix}.tsv"
                            else:
                                categ_file = f"{args.cat_prefix}_{sufix_value}.tsv"
                            print(f"Using category description file {categ_file} for revision {rev}")
                            break

                    category_csv_path = Path(cat_directory) / categ_file

                    clip_instance = CLIP(max_category_samples=args.max_categ, test_ratio=test_size,
                                         eval_max_category_samples=args.max_categ_eval,
                                         top_N=1, model_name=base_model, device=device,
                                         categories_tsv=str(category_csv_path), seed=seed, input_format=args.file_format,
                                         output_dir=str(output_dir), categories_dir=cat_directory,
                                         model_dir=str(model_path), cp_dir=str(cp_dir), model_revision=rev,
                                         cat_prefix=args.cat_prefix, avg=temp_avg, zero_shot=False)

                    clip_instance.load_model(str(model_path), rev)

                    rdf = clip_instance.predict_directory(str(input_dir), raw=raw, chunk_size=chunk_size)
                    all_best_directory_output[rev] = rdf

                # combine all best predictions into single file by first 2 columns
                combined_df = pd.DataFrame()
                for rev, rdf in all_best_directory_output.items():
                    # rename columns to include revision
                    renamed_columns = {col: f"{col}-{rev}" for col in rdf.columns if col not in ["FILE", "PAGE"]}
                    rdf_renamed = rdf.rename(columns=renamed_columns)

                    if combined_df.empty:
                        combined_df = rdf_renamed
                    else:
                        combined_df = pd.merge(combined_df, rdf_renamed, on=["FILE", "PAGE"], how="outer")

                combined_df.to_csv(
                    f"{output_dir}/tables/{time_stamp}_BEST_{len(revision_best_models.keys())}_models_TOP-1.csv",
                    sep=",",
                    index=False)
                print(f"Results for TOP-{top_N} predictions are recorded into {output_dir}/tables/ directory")






