from __future__ import annotations

import argparse
import configparser
import os
import sys
import time
from pathlib import Path

import torch

from classifier import ImageClassifier
from model_registry import CATEGORIES as def_categ
from model_registry import REVISION_BEST_MODELS, REVISION_TO_BASE_MODEL
from parallel_best import run_best_models  # memory-aware best-models engine + averaging
from utils import dataframe_results, directory_scraper

# ═══════════════════════════════════════════════════════════════════════════
# agent-skill branch: inference-only CLI.
#
# The full run.py (training, cross-validation, evaluation, checkpoint
# averaging, YOLO backend, paradata logging) lives on the `test` branch.
# This branch keeps the prediction surface only:
#
#   python3 run.py --hf -rev v4.3                # download-only (setup script)
#   python3 run.py -f page.png [--hf]            # single image, top-N
#   python3 run.py -d scans/ [--hf]              # directory → CSV table
#   python3 run.py -f page.png --best [--hf]     # best-5 ensemble, averaged
#   python3 run.py -d scans/ --best --parallel   # ensemble, memory-aware GPU
# ═══════════════════════════════════════════════════════════════════════════


def resolve_base_model(revision: str, explicit_base: str = None) -> str:
    """Map a model revision tag to its base checkpoint (exact key, then prefix)."""
    if explicit_base:
        return explicit_base
    if revision in REVISION_TO_BASE_MODEL:
        return REVISION_TO_BASE_MODEL[revision]
    for key, base_model in REVISION_TO_BASE_MODEL.items():
        if revision.startswith(key):
            return base_model
    raise ValueError(f"Base model not found for revision: {revision} (pass -b/--base explicitly)")


def load_or_download(
    revision: str,
    base_model: str,
    model_dir: str,
    cp_dir: str,
    hf_repo: str,
    allow_download: bool,
) -> ImageClassifier:
    """Load model_{revision} from model_dir, downloading it from the HF hub if allowed."""
    local_name = f"model_{revision.replace('.', '')}"
    local_path = Path(model_dir) / local_name

    classifier = ImageClassifier(checkpoint=base_model, num_labels=len(def_categ), store_dir=str(cp_dir))

    if local_path.is_dir():
        classifier.load_model(str(local_path))
        return classifier

    if not allow_download:
        sys.exit(
            f"Model {revision} not found locally at {local_path} - rerun with --hf to download it "
            f"from the Hugging Face hub ({hf_repo})."
        )

    classifier.load_from_hub(hf_repo, revision=revision)
    classifier.save_model(str(local_path))
    return classifier


def ensure_best_models(model_dir: str, cp_dir: str, hf_repo: str, allow_download: bool) -> None:
    """Make sure every best-5 ensemble member exists locally (downloading if allowed)."""
    missing = [
        rev
        for rev in REVISION_BEST_MODELS
        if not (Path(model_dir) / f"model_{rev.replace('.', '')}").is_dir()
    ]
    if not missing:
        return
    if not allow_download:
        sys.exit(
            f"Missing local model(s) for --best: {', '.join(missing)} - rerun with --hf to download "
            f"them from the Hugging Face hub ({hf_repo})."
        )
    for rev in missing:
        clf = load_or_download(rev, REVISION_BEST_MODELS[rev], model_dir, cp_dir, hf_repo, allow_download=True)
        del clf


def print_single_prediction(header: str, labeled_scores: list) -> None:
    print(header)
    for label, score in labeled_scores:
        print(f"\t{label}\t{score:.3f}")


if __name__ == "__main__":
    # Initialize the parser
    config = configparser.ConfigParser()
    # Read the configuration file
    config.read(os.path.join(os.path.dirname(__file__), "setup", "config.txt"))

    revision_best_models = REVISION_BEST_MODELS

    batch = config.getint("SETUP", "batch")  # depends on GPU/CPU capabilities
    top_N = config.getint("SETUP", "top_N")  # top N predictions, 3 is enough, 11 for "raw" scores

    config_base_model = config.get("SETUP", "base_model")  # explicit -b/--base override wins
    config_format = config.get("SETUP", "files_format")
    raw = config.getboolean("SETUP", "raw")

    HF = config.getboolean("HF", "use_hf")
    hf_version = config.get("HF", "revision")
    hf_repo = config.get("HF", "repo_name")

    model_dir = config.get("OUTPUT", "FOLDER_MODELS")
    config_input_dir = config.get("INPUT", "FOLDER_INPUT")

    output_dir = Path(config.get("OUTPUT", "FOLDER_RESULTS"))
    cp_dir = Path(config.get("OUTPUT", "FOLDER_CPOINTS"))

    time_stamp = time.strftime("%Y%m%d-%H%M")  # for results files

    parser = argparse.ArgumentParser(description="Page sorter based on ViT / RegNetY / EffNetV2 (inference-only)")
    parser.add_argument("-f", "--file", type=str, default=None, help="Single page image path")
    parser.add_argument("-d", "--directory", type=str, default=None, help="Path to folder with unprocessed pages")
    parser.add_argument(
        "--dir", help=f"Process the configured input folder ({config_input_dir})", action="store_true"
    )
    parser.add_argument("-b", "--base", type=str, default=None, help="Repository of the base model (override)")
    parser.add_argument(
        "-rev", "--revision", type=str, default=hf_version, help="HF revision tag of the fine-tuned model"
    )
    parser.add_argument("-tn", "--topn", type=int, default=top_N, help="Number of top predictions per page")
    parser.add_argument("--hf", help="Download missing models from the HF hub", action="store_true", default=HF)
    parser.add_argument("--raw", help="Also save raw per-category scores CSV", action="store_true", default=raw)
    parser.add_argument(
        "--file_format", type=str, default=config_format, help="Image file format for directory scraping"
    )
    parser.add_argument("--best", help="Run the best-5 models ensemble with averaging", action="store_true")
    parser.add_argument(
        "--parallel", help="Memory-aware parallel loading of the best models (CUDA only)", action="store_true"
    )
    parser.add_argument(
        "--no-average-best",
        dest="average_best",
        help="Skip averaging - keep only the wide per-model votes table",
        action="store_false",
    )
    parser.add_argument(
        "--save_intermediates", help="Save per-model TOP-N CSVs next to the averaged result", action="store_true"
    )
    args = parser.parse_args()

    if args.topn < 1 or args.topn > len(def_categ):
        sys.exit(f"-tn/--topn must be between 1 and {len(def_categ)}")

    device_note = "CUDA" if torch.cuda.is_available() else ("MPS" if torch.backends.mps.is_available() else "CPU")
    print(f"Device: {device_note}\tRevision: {args.revision}\tTop-N: {args.topn}")

    # ── Input resolution ─────────────────────────────────────────────────────
    test_images: list = []
    input_dir = Path(config_input_dir) if args.directory is None else Path(args.directory)

    if args.file is not None:
        if not Path(args.file).is_file():
            sys.exit(f"File not found: {args.file}")
        test_images = [args.file]
    elif args.dir or args.directory is not None:
        if not input_dir.is_dir():
            sys.exit(f"Input directory not found: {input_dir}")
        test_images = sorted(str(p) for p in directory_scraper(Path(input_dir), args.file_format))
        if not test_images:
            sys.exit(f"No valid image files found to process in {input_dir}. Exiting.")
    elif args.hf and not args.best:
        # Download-only mode (used by setup/setup_api_service.sh):
        #   python3 run.py --hf -rev v4.3
        base_model = resolve_base_model(args.revision, args.base)
        load_or_download(args.revision, base_model, model_dir, cp_dir, hf_repo, allow_download=True)
        print(f"Download-only mode: model {args.revision} is ready under {model_dir}. Exiting.")
        sys.exit(0)
    elif args.hf and args.best:
        # Download-only mode for the whole ensemble:
        #   python3 run.py --hf --best
        ensure_best_models(model_dir, cp_dir, hf_repo, allow_download=True)
        print(f"Download-only mode: all best-5 models are ready under {model_dir}. Exiting.")
        sys.exit(0)
    else:
        parser.error("Provide -f FILE or -d DIR (or --dir), or use --hf for download-only mode.")

    out_tables = output_dir / "tables"

    # ── Best-5 ensemble path ─────────────────────────────────────────────────
    if args.best:
        ensure_best_models(model_dir, cp_dir, hf_repo, allow_download=args.hf)

        avg_csv_path = run_best_models(
            test_images=test_images,
            categories=def_categ,
            revision_best_models=revision_best_models,
            model_dir=model_dir,
            cp_dir=str(cp_dir),
            batch=batch,
            top_N=args.topn,
            output_dir=str(output_dir),
            time_stamp=time_stamp,
            paradata_logger=None,
            parallel=args.parallel,
            save_intermediates=args.save_intermediates,
            average_best=args.average_best,
        )

        if args.file is not None and avg_csv_path:
            import pandas as pd

            avg_df = pd.read_csv(avg_csv_path)
            row = avg_df.iloc[0]
            labeled = [
                (row[f"CLASS-{j + 1}"], float(row[f"SCORE-{j + 1}"]))
                for j in range(args.topn)
                if f"CLASS-{j + 1}" in avg_df.columns
            ]
            print_single_prediction(
                f"\nFile {args.file} - averaged predictions from {len(revision_best_models)} best models:", labeled
            )
        sys.exit(0)

    # ── Single-model path ────────────────────────────────────────────────────
    base_model = resolve_base_model(args.revision, args.base)
    classifier = load_or_download(args.revision, base_model, model_dir, cp_dir, hf_repo, allow_download=args.hf)

    if args.file is not None:
        if args.topn == 1:
            idx = classifier.infer(args.file)
            print(f"File {args.file} predicted:\n\t{def_categ[idx]}")
        else:
            pred_scores = classifier.top_n_predictions(args.file, top_n=args.topn)
            labeled = [(def_categ[i], s) for i, s in pred_scores]
            print_single_prediction(f"File {args.file} predicted:", labeled)
        sys.exit(0)

    # Directory → CSV table(s)
    dataloader = classifier.create_dataloader(test_images, batch)
    test_predictions, raw_prediction = classifier.infer_dataloader(dataloader, top_n=args.topn, raw=args.raw)

    if not test_predictions:
        sys.exit("No images were successfully processed - nothing to save.")

    rdf, raw_df = dataframe_results(test_images, test_predictions, def_categ, args.topn, raw_prediction)

    out_tables.mkdir(parents=True, exist_ok=True)
    rev_tag = args.revision.replace(".", "")
    rdf_path = out_tables / f"{time_stamp}_{rev_tag}_TOP-{args.topn}.csv"
    rdf.to_csv(rdf_path, index=False)
    print(f"Result CSV → {rdf_path}")

    if args.raw and raw_df is not None:
        raw_path = out_tables / f"{time_stamp}_{rev_tag}_RAW.csv"
        raw_df.to_csv(raw_path, index=False)
        print(f"Raw scores CSV → {raw_path}")