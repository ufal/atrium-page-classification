import argparse
import os
from huggingface_hub import create_branch

import configparser
from classifier import *
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
        "v1.1": "ViT-B/16",
        "v1.2": "ViT-B/32",
        "v2.1": "ViT-L/14",
        "v2.2": "ViT-L/14@336",
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

    model_name_local = f"{base_model.replace('/', '')}_rev_{hf_version.replace('.', '')}"
    hf_models_directory = config.get('OUTPUT', 'FOLDER_MODELS')
    model_path = Path(f"{hf_models_directory}/{model_name_local}")

    test_dir = config.get('INPUT', 'FOLDER_INPUT')

    epochs = config.getint("TRAIN", "epochs")
    max_categ = config.getint("TRAIN", "max_categ")  # max number of category samples
    max_categ_e = config.getint("TRAIN", "max_categ_e")  # max number of category samples for evaluation
    log_step = config.getint("TRAIN", "log_step")
    test_size = config.getfloat("TRAIN", "test_size")
    learning_rate = config.getfloat("TRAIN", "lr")

    raw = config.getboolean('SETUP', 'raw')
    zero_shot = config.getboolean('SETUP', 'zero_shot')  # zero-shot prediction without training
    visualize = config.getboolean('SETUP', 'visualize')  # visualize model accuracy statistics


    # cur = Path.cwd()  # directory with this script
    cur = Path(__file__).resolve().parent  # directory with this script
    output_dir = Path(config.get('OUTPUT', 'FOLDER_RESULTS'))
    cp_dir = Path(config.get('OUTPUT', 'FOLDER_CPOINTS'))

    time_stamp = time.strftime("%Y%m%d-%H%M")  # for results files

    parser = argparse.ArgumentParser(description='Page sorter based on ViT')

    # Prediction arguments
    parser.add_argument('-f', "--file", type=str, default=None, help="Single PNG page path for prediction.")
    parser.add_argument('-d', "--directory", type=str, default=None,
                        help="Path to folder with PNG pages for prediction.")
    parser.add_argument("--dir", help="Predict a whole directory of images.", action="store_true")
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

    # Category file arguments
    parser.add_argument('--cat_prefix', type=str, default=categ_prefix,
                        help='Prefix for category description TSV files.')
    parser.add_argument('--cat_dir', type=str, default=categ_directory, help='Directory with category description files.'),
    parser.add_argument('--avg', action='store_true', default=avg, help='Average scores from multiple category description files.')
    parser.add_argument('--zero_shot', action='store_true', default=zero_shot, help='Perform zero-shot prediction (no training).')
    parser.add_argument('--vis', action='store_true',default=visualize, help='Visualize model accuracy statsistics.')

    # Common arguments
    parser.add_argument('-tn', "--topn", type=int, default=top_N, help="Number of top result categories to consider.")

    # Evaluation arguments
    parser.add_argument("--eval", action="store_true", help="Evaluate a saved model.")
    parser.add_argument("--eval_dir", action="store_true", help="Evaluate a directory of saved models.")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to the saved model checkpoint (.pt file) for evaluation.")
    parser.add_argument("--model_dir", type=str, default=hf_models_directory,
                        help="Path to the directory of saved model checkpoints (.pt files) for evaluation.")


    parser.add_argument('-rev', "--revision", type=str, default=None, help="HuggingFace revision (e.g. `main`, `vN.0` or `vN.M`)")
    parser.add_argument("--hf", help="Use model and processor from the HuggingFace repository", default=HF, action="store_true")
    parser.add_argument("--raw", help="Output raw scores for each category", default=raw, action="store_true")


    args = parser.parse_args()

    input_dir = Path(test_dir) if args.directory is None else Path(args.directory)
    Training, top_N, raw = args.train, args.topn, args.raw


    if args.revision is None: # using config file revision
        args.revision = hf_version
    else:
        if args.revision not in revision_to_base_model:
            raise ValueError(f"Revision {args.revision} is not supported. Available revisions: {list(revision_to_base_model.keys())}")

        base_model = revision_to_base_model[args.revision]
        if args.model != base_model:
            print(f"Base model {args.base} does not match the revision {args.revision}. Using {base_model} instead.")
            args.model = base_model

    # locally creating new directory paths instead of context.txt variables loaded with mistakes
    if not output_dir.is_dir():
        os.makedirs(output_dir)

        os.makedirs(f"{output_dir}/tables")
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
                      "eval_dir", "vis")))
    ))

    args.logdir += f"-{args.model.replace('/', '_')}" if args.model else ""

    cur = Path(__file__).parent
    output_dir = cur / "results"
    output_dir.mkdir(exist_ok=True)

    cat_directory = str(cur / args.cat_dir)
    clip_instance = CLIP(max_category_samples=args.max_categ,
                         eval_max_category_samples=args.max_categ_eval,
                         top_N=args.topn, model_name=args.model, device=device,
                         categories_tsv=categ_file,
                         output_dir=str(output_dir), categories_dir=cat_directory,
                         cat_prefix=args.cat_prefix, avg=args.avg, zero_shot=args.zero_shot)

    data_dir = config.get("TRAIN", "FOLDER_PAGES")
    data_dir_eval = config.get("EVAL", "FOLDER_PAGES")

    categories = def_categ
    print(f"Category input directories found: {categories}")

    if args.hf:
        weights_path = Path("model_checkpoints")

        # -------------------------------------------------------------
        # ----- UNCOMMENT for saving trained model in HF format -------
        # -------------------------------------------------------------
        # model_path_str = args.model_path
        # print(f"Model path provided: {model_path_str}")
        # if model_path_str is None:
        #     remove_punctuation = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
        #     model_name_sanitized = args.model.translate(remove_punctuation).replace(" ", "")
        #     model_path = weights_path / f"model_{model_name_sanitized}_{args.max_categ}c_{str(args.lr)}.pt"
        #     if not model_path.exists():
        #         model_path = weights_path / f"model_{model_name_sanitized}_{args.max_categ}c_{str(args.lr)}_cp.pt"
        #
        #     if not model_path.exists():
        #         raise ValueError(
        #             "Model file or checkpoint not found at default paths. Please provide a path using --model_path.")
        #
        # model_path_str = str(weights_path / model_path_str) if model_path_str else None
        #
        # if model_path_str is not None:
        #     print(f"Loading model from {model_path_str} for evaluation...")
        #     checkpoint = torch.load(model_path_str, map_location=device)
        #     clip_instance.model.load_state_dict(checkpoint['model_state_dict'])
        #     print(f"Model loaded from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.4f}.")
        # -------------------------------------------------------------

        # saving model to local path
        model_name_local = f"{args.model.replace('/', '')}_rev_{args.revision.replace('.', '')}"
        model_path = Path(weights_path.parent / args.model_dir / model_name_local)


        # ----------------------------------------------
        # ----- UNCOMMENT for saving in HF format -------
        # ----------------------------------------------
        # if not model_path.is_dir():
        #     os.makedirs(model_path, exist_ok=True)
        # clip_instance.save_model(str(model_path))
        # ----------------------------------------------

        # clip_instance.load_model(str(model_path))
        # ----------------------------------------------
        # ----- UNCOMMENT for pushing to HF repo -------
        # ----------------------------------------------
        # create_branch(config.get("HF", "repo_name"), repo_type="model", branch=config.get("HF", "revision"),
        #               exist_ok=True,
        #               token=config.get("HF", "token"))
        # clip_instance.pushing_to_hub(config.get("HF", "repo_name"), False, config.get("HF", "token"),
        #                              config.get("HF", "revision"))
        # ----------------------------------------------

        # raise NotImplementedError
        #
        # loading from repo
        clip_instance.load_from_hub(config.get("HF", "repo_name"), args.revision)

        # hf_model_name_local = f"model_{args.revision.replace('.', '')}"
        # hf_model_path = f"{model_dir}/{hf_model_name_local}"

        clip_instance.save_model(str(model_path))

        clip_instance.load_model(str(model_path))

    if args.train:
        if not os.path.isdir(data_dir):
            raise ValueError(f"Train directory not found at: {data_dir}")
        if not os.path.isdir(data_dir_eval):
            raise ValueError(f"Evaluation directory not found at: {data_dir_eval}")

        clip_instance.train(
            train_dir=data_dir,
            eval_dir=data_dir_eval,
            log_dir=args.logdir,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            batch_size=args.batch_size
        )
    elif args.vis:
        csv = output_dir / 'stats' / f"model_accuracies{'_zero' if args.zero_shot else ''}.csv"
        if not csv.exists():
            print(f"CSV file for visualization not found at {csv}. Please run model evaluation first.")
        else:
            visualize_results(str(csv), str(output_dir / 'stats'), args.zero_shot)
    elif args.eval:
        weights_path = Path("model_checkpoints")
        if args.zero_shot:
            model_path_str = None
        else:
            model_path_str = args.model_path
            print(f"Model path provided: {model_path_str}")
            if model_path_str is None:
                remove_punctuation = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
                model_name_sanitized = args.model.translate(remove_punctuation).replace(" ", "")
                model_path = weights_path / f"model_{model_name_sanitized}_{args.max_categ}c_{str(args.lr)}.pt"
                if not model_path.exists():
                    model_path = weights_path / f"model_{model_name_sanitized}_{args.max_categ}c_{str(args.lr)}_cp.pt"

                if not model_path.exists():
                    raise ValueError(
                        "Model file or checkpoint not found at default paths. Please provide a path using --model_path.")

            model_path_str = str(weights_path / model_path_str)

        if not os.path.isdir(data_dir_eval):
            raise ValueError(f"Evaluation directory not found at: {data_dir_eval}")

        clip_instance.evaluate_saved_model(
            model_path=model_path_str,
            eval_dir=data_dir_eval,
            batch_size=args.batch_size
        )
    elif args.eval_dir:
        if not os.path.isdir(data_dir_eval):
            raise ValueError(f"Evaluation directory not found at: {data_dir_eval}")
        if not os.path.isdir(args.model_dir):
            raise ValueError(f"Model directory not found at: {args.model_dir}. Please provide a valid path.")

        evaluate_multiple_models(
            model_dir=args.model_dir,
            eval_dir=data_dir_eval,
            batch_size=args.batch_size,
            device=device,
            cat_prefix=args.cat_prefix,
            vis=True,
            zero_shot=args.zero_shot,
        )
    elif args.zero_shot:  # New branch for zero-shot prediction
        if args.file:
            prediction = clip_instance.predict_single(args.file)
            print(f"Zero-shot prediction for {args.file}: {prediction}")
        elif args.dir:
            input_dir_pred = Path(args.directory) if args.directory is not None else cur / 'test-images' / 'pages'
            table_out_path = output_dir / 'tables'
            table_out_path.mkdir(exist_ok=True, parents=True)
            directory_result_output = str(table_out_path / f'zero_shot_raw_result_{args.model.replace("/", "")}_{args.topn}n_{max_categ}c.csv')
            clip_instance.predict_directory(str(input_dir_pred), raw=True, out_table=directory_result_output)
        else:
            print("Please specify a file (-f) or a directory (-d) for zero-shot prediction.")
    else:
        if args.file:
            if args.topn > 1:
                scores, labels = clip_instance.predict_top(args.file)
                print(f"File {args.file} predicted:")
                for lab, sc in zip(labels, scores):
                    print(f"\t{lab}:  {round(sc * 100, 2)}%")
            else:
                prediction = clip_instance.predict_single(args.file)
                print(f"Prediction for {args.file}:\n{prediction}")

        if args.dir or args.directory is not None:
            input_dir_pred = Path(args.directory) if args.directory is not None else cur / 'test-images' / 'pages'
            table_out_path = output_dir / 'tables'
            table_out_path.mkdir(exist_ok=True, parents=True)
            directory_result_output = str(table_out_path / f'result_{time_stamp}_{args.model.replace("/", "")}_{args.topn}n_{max_categ}c.csv')
            clip_instance.predict_directory(str(input_dir_pred), raw=True, out_table=directory_result_output)




