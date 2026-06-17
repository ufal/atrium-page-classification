import sys
import os
from pathlib import Path
from PIL import Image
import torch
import logging

# Add parent directory to path to import original modules
sys.path.append(str(Path(__file__).parent.parent))

try:
    from classifier import ImageClassifier
    from model_registry import REVISION_TO_BASE_MODEL, REVISION_BEST_MODELS, CATEGORIES
    from ensemble import average_prediction_dicts
except ImportError:
    from classifier import ImageClassifier
    from model_registry import REVISION_TO_BASE_MODEL, REVISION_BEST_MODELS, CATEGORIES
    from ensemble import average_prediction_dicts

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
MODEL_BASE_PATH = Path(__file__).parent.parent / "model"
HF_REPO_NAME = "ufal/vit-historical-page"

AVAILABLE_VERSIONS = list(REVISION_BEST_MODELS.keys())


class ModelManager:
    def __init__(self):
        self.models = {}
        self.available_versions = AVAILABLE_VERSIONS
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _get_base_model_id(self, version: str) -> str:
        if version in REVISION_TO_BASE_MODEL:
            return REVISION_TO_BASE_MODEL[version]
        for key, base_model in REVISION_TO_BASE_MODEL.items():
            if version.startswith(key):
                return base_model
        raise ValueError(f"Base model not found for version: {version}")

    def get_model_details(self, version: str) -> str:
        if version == "all":
            return "Ensemble (Average of 5 Models)"
        try:
            base = self._get_base_model_id(version)
            return f"{base} ({version})"
        except:
            return f"Unknown Base ({version})"

    def load_model(self, version: str):
        if version in self.models:
            return self.models[version]

        base_model_id = self._get_base_model_id(version)
        local_model_name = f"model_{version.replace('.', '')}"
        local_model_path = MODEL_BASE_PATH / local_model_name

        logger.info(f"Initializing {version} using base '{base_model_id}' on {self.device}...")
        clf = ImageClassifier(checkpoint=base_model_id, num_labels=len(CATEGORIES))

        if local_model_path.exists():
            logger.info(f"Loading fine-tuned weights locally from {local_model_path}...")
            clf.load_model(str(local_model_path))
        else:
            logger.info(
                f"Model not found locally at {local_model_path}. Attempting download from Hugging Face ({HF_REPO_NAME}, revision {version})...")
            try:
                clf.load_from_hub(HF_REPO_NAME, revision=version)
                logger.info(f"Saving downloaded model to {local_model_path}...")
                clf.save_model(str(local_model_path))
            except Exception as e:
                logger.error(f"Failed to download model {version} from Hugging Face: {e}")
                raise RuntimeError(f"Model {version} not found locally and could not be downloaded: {e}")

        self.models[version] = clf
        return clf

    def warmup(self, versions: list = None):
        for v in (versions or self.available_versions):
            try:
                self.load_model(v)
                logger.info(f"Warmed up model {v}")
            except Exception as e:
                logger.warning(f"Could not pre-load model {v}: {e}")

    def predict(self, image: Image.Image, version: str, topn: int = 1):
        if version == "all":
            return self._predict_averaged(image, topn)
        else:
            return self._run_single_inference(version, image, topn)

    def predict_directory(self, dir_path: str, version: str, topn: int = 3, batch_size: int = 16):
        image_paths = sorted([os.path.join(dir_path, f) for f in os.listdir(dir_path)
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if not image_paths:
            return []

        if version == "all":
            n = len(image_paths)
            all_model_predictions = []

            for v in self.available_versions:
                try:
                    clf = self.load_model(v)
                    dataloader = clf.create_dataloader(image_paths, batch_size=batch_size)
                    predictions, _ = clf.infer_dataloader(
                        dataloader, top_n=len(CATEGORIES), raw=False
                    )

                    model_preds = []
                    for pred_item in predictions:
                        model_preds.append(
                            [{"label": CATEGORIES[idx], "score": float(score)} for idx, score in pred_item])
                    all_model_predictions.append(model_preds)
                except Exception as e:
                    logger.warning(f"Ensemble: model {v} failed on directory, skipping: {e}")
                    continue

            if not all_model_predictions:
                logger.error("Ensemble batch inference failed: all models errored.")
                return [[] for _ in image_paths]

            formatted_batch_results = []
            for i in range(n):
                preds_for_image = []
                for model_preds in all_model_predictions:
                    if i < len(model_preds):
                        preds_for_image.append(model_preds[i])

                formatted_batch_results.append(average_prediction_dicts(preds_for_image, CATEGORIES, topn))
            return formatted_batch_results

        try:
            clf = self.load_model(version)
            internal_topn = max(topn, 2)
            dataloader = clf.create_dataloader(image_paths, batch_size=batch_size)
            predictions, _ = clf.infer_dataloader(dataloader, top_n=internal_topn, raw=False)

            formatted_batch_results = []
            for pred_item in predictions:
                row_preds = []
                for idx, score in pred_item[:topn]:
                    row_preds.append({
                        "label": CATEGORIES[idx],
                        "score": float(score)
                    })
                formatted_batch_results.append(row_preds)

            return formatted_batch_results

        except Exception as e:
            logger.error(f"Batch inference error for {version}: {e}")
            raise e

    def _predict_averaged(self, image, topn):
        predictions_list = []

        for v in self.available_versions:
            try:
                results = self._run_single_inference(v, image, topn=len(CATEGORIES))
                if isinstance(results, dict) and "error" in results: continue
                predictions_list.append(results)
            except Exception:
                continue

        if not predictions_list: return {"error": "All models failed."}
        return average_prediction_dicts(predictions_list, CATEGORIES, topn)

    def _run_single_inference(self, version, image, topn):
        try:
            clf = self.load_model(version)
            predictions = clf.top_n_predictions(image, top_n=topn)
            return [{"label": CATEGORIES[i], "score": float(s)} for i, s in predictions]
        except Exception as e:
            return {"error": str(e)}


manager = ModelManager()