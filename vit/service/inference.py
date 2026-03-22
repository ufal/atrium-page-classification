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
except ImportError:
    from classifier import ImageClassifier

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONFIGURATION FROM RUN.PY ---

MODEL_BASE_PATH = Path(__file__).parent.parent / "model"
HF_REPO_NAME = "ufal/vit-historical-page"

CATEGORIES = ["DRAW", "DRAW_L", "LINE_HW", "LINE_P", "LINE_T", "PHOTO", "PHOTO_L", "TEXT", "TEXT_HW", "TEXT_P",
              "TEXT_T"]

REVISION_TO_BASE_MODEL = {
    "v10.": "microsoft/dit-large-finetuned-rvlcdip",
    "v11.": "microsoft/dit-large",
    "v12.": "timm/tf_efficientnetv2_m.in21k_ft_in1k",
    "v1.3": "timm/tf_efficientnetv2_s.in21k",
    "v2.3": "google/vit-base-patch16-224",
    "v3.": "google/vit-base-patch16-384",
    "v3.3": "google/vit-base-patch16-384",
    "v4.": "timm/tf_efficientnetv2_l.in21k_ft_in1k",
    "v4.3": "timm/regnety_160.swag_ft_in1k",
    "v5.": "google/vit-large-patch16-384",
    "v5.3": "google/vit-large-patch16-384",
    "v6.": "timm/regnety_120.sw_in12k_ft_in1k",
    "v7.": "timm/regnety_160.swag_ft_in1k",
    "v8.": "timm/regnety_640.seer",
    "v9.": "microsoft/dit-base-finetuned-rvlcdip",
}

AVAILABLE_VERSIONS = ["v1.3", "v2.3", "v3.3", "v4.3", "v5.3"]


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
        """Returns string: base model repository plus version"""
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

        # Initialize the classifier wrapper
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
        """
        Pre-load models at startup to avoid first-request latency spikes.
        Call from the FastAPI startup event with the default version(s).
        """
        for v in (versions or self.available_versions):
            try:
                self.load_model(v)
                logger.info(f"Warmed up model {v}")
            except Exception as e:
                logger.warning(f"Could not pre-load model {v}: {e}")

    def predict(self, image: Image.Image, version: str, topn: int = 1):
        """
        Runs prediction on a single image.
        """
        if version == "all":
            return self._predict_averaged(image, topn)
        else:
            return self._run_single_inference(version, image, topn)

    def predict_directory(self, dir_path: str, version: str, topn: int = 3, batch_size: int = 16):
        """
        Runs batch prediction on a directory of images using the classifier's dataloader logic.
        """
        if version == "all":
            # Batch averaging is complex, strictly required?
            # Fallback to simple iteration for 'all' to ensure correctness without complex refactor
            results = []
            files = sorted([os.path.join(dir_path, f) for f in os.listdir(dir_path)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            for f in files:
                img = Image.open(f).convert('RGB')
                preds = self._predict_averaged(img, topn)
                results.append(preds)
            return results

        try:
            clf = self.load_model(version)

            # 1. Get all image paths
            image_paths = sorted([os.path.join(dir_path, f) for f in os.listdir(dir_path)
                                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

            if not image_paths:
                return []

            # 2. FIX: always request at least 2 predictions internally so we always
            #    have real confidence scores. Results are then truncated to the
            #    caller-requested topn before returning.
            internal_topn = max(topn, 2)

            # 3. Create Data Loader
            dataloader = clf.create_dataloader(image_paths, batch_size=batch_size)

            # 4. Infer (returns list of predictions)
            predictions, _ = clf.infer_dataloader(dataloader, top_n=internal_topn, raw=False)

            formatted_batch_results = []

            # 5. Map indices to labels and truncate to the requested topn
            for pred_item in predictions:
                # With internal_topn >= 2, pred_item is always a list of (idx, score) tuples
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
        aggregated_scores = {cat: 0.0 for cat in CATEGORIES}
        successful_models = 0

        for v in self.available_versions:
            try:
                results = self._run_single_inference(v, image, topn=len(CATEGORIES))
                if isinstance(results, dict) and "error" in results: continue
                for item in results:
                    aggregated_scores[item['label']] += item['score']
                successful_models += 1
            except Exception:
                continue

        if successful_models == 0: return {"error": "All models failed."}

        final_results = [{"label": l, "score": s / successful_models} for l, s in aggregated_scores.items()]
        final_results.sort(key=lambda x: x["score"], reverse=True)
        return final_results[:topn]

    def _run_single_inference(self, version, image, topn):
        try:
            clf = self.load_model(version)
            predictions = clf.top_n_predictions(image, top_n=topn)
            return [{"label": CATEGORIES[i], "score": float(s)} for i, s in predictions]
        except Exception as e:
            return {"error": str(e)}


manager = ModelManager()