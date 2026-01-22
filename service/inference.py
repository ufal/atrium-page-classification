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

        if not local_model_path.exists():
            logger.error(f"Local model directory not found: {local_model_path}")
            raise FileNotFoundError(f"Model directory not found: {local_model_path}")

        logger.info(f"Initializing {version} using base '{base_model_id}' on {self.device}...")

        clf = ImageClassifier(checkpoint=base_model_id, num_labels=len(CATEGORIES))
        logger.info(f"Loading fine-tuned weights from {local_model_path}...")
        clf.load_model(str(local_model_path))

        self.models[version] = clf
        return clf

    def predict(self, image: Image.Image, version: str, topn: int = 1):
        """
        Runs prediction.
        If version is 'all', averages scores from all available versions.
        """
        if version == "all":
            return self._predict_averaged(image, topn)
        else:
            return self._run_single_inference(version, image, topn)

    def _predict_averaged(self, image, topn):
        """
        Runs all models, aggregates scores, averages them, and returns top N.
        """
        # Dictionary to store accumulated scores: { 'TEXT': 0.0, 'DRAW': 0.0, ... }
        aggregated_scores = {cat: 0.0 for cat in CATEGORIES}

        successful_models = 0

        for v in self.available_versions:
            try:
                # We request ALL categories (len(CATEGORIES)) to ensure we can average correctly
                results = self._run_single_inference(v, image, topn=len(CATEGORIES))

                # Check for error in individual result
                if isinstance(results, dict) and "error" in results:
                    continue

                for item in results:
                    aggregated_scores[item['label']] += item['score']

                successful_models += 1
            except Exception as e:
                logger.error(f"Error including version {v} in average: {e}")

        if successful_models == 0:
            return {"error": "All models failed to run."}

        # Calculate Average
        final_results = []
        for label, total_score in aggregated_scores.items():
            avg_score = total_score / successful_models
            final_results.append({"label": label, "score": avg_score})

        # Sort by score descending
        final_results.sort(key=lambda x: x["score"], reverse=True)

        # Return Top N
        return final_results[:topn]

    def _run_single_inference(self, version, image, topn):
        try:
            clf = self.load_model(version)

            # top_n_predictions returns list of tuples (index, score)
            predictions = clf.top_n_predictions(image, top_n=topn)

            formatted_results = []
            for idx, score in predictions:
                formatted_results.append({
                    "label": CATEGORIES[idx],
                    "score": float(score)
                })

            return formatted_results
        except Exception as e:
            logger.error(f"Inference error for {version}: {e}")
            return {"error": str(e)}


# Singleton instance
manager = ModelManager()