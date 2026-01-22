import sys
import os
from pathlib import Path
from PIL import Image
import torch
import logging

# Add parent directory to path to import original modules
# Assumes this script is in ./service/ or ./api/ and classifier.py is in ./
sys.path.append(str(Path(__file__).parent.parent))

try:
    from classifier import ImageClassifier
except ImportError:
    # Fallback if running from the root directory
    from classifier import ImageClassifier

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONFIGURATION FROM RUN.PY ---

# Local directory where fine-tuned models are stored
MODEL_BASE_PATH = Path(__file__).parent.parent / "model"

# Category definitions
CATEGORIES = ["DRAW", "DRAW_L", "LINE_HW", "LINE_P", "LINE_T", "PHOTO", "PHOTO_L", "TEXT", "TEXT_HW", "TEXT_P", "TEXT_T"]

# Mappings from run.py to ensure correct base model initialization
REVISION_TO_BASE_MODEL = {
    "v10.": "microsoft/dit-large-finetuned-rvlcdip",
    "v11.": "microsoft/dit-large",
    "v12.": "timm/tf_efficientnetv2_m.in21k_ft_in1k",
    "v1.3": "timm/tf_efficientnetv2_s.in21k",
    "v2.3": "google/vit-base-patch16-224",
    "v3.": "google/vit-base-patch16-384",
    "v3.3": "google/vit-base-patch16-384", # Added specific mapping for v3.3
    "v4.": "timm/tf_efficientnetv2_l.in21k_ft_in1k",
    "v4.3": "timm/regnety_160.swag_ft_in1k", # Specific mapping from best_models
    "v5.": "google/vit-large-patch16-384",
    "v5.3": "google/vit-large-patch16-384", # Specific mapping from best_models
    "v6.": "timm/regnety_120.sw_in12k_ft_in1k",
    "v7.": "timm/regnety_160.swag_ft_in1k",
    "v8.": "timm/regnety_640.seer",
    "v9.": "microsoft/dit-base-finetuned-rvlcdip",
}

# The list of versions exposed to the API (matching revision_best_models in run.py)
AVAILABLE_VERSIONS = ["v1.3", "v2.3", "v3.3", "v4.3", "v5.3"]

class ModelManager:
    def __init__(self):
        self.models = {}
        self.available_versions = AVAILABLE_VERSIONS
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def _get_base_model_id(self, version: str) -> str:
        """
        Retrieves the base model ID for a given version string.
        Matches logic in run.py (prefix matching or exact match).
        """
        # 1. Try exact match from known best models or map
        if version in REVISION_TO_BASE_MODEL:
            return REVISION_TO_BASE_MODEL[version]
            
        # 2. Try prefix matching (as seen in run.py args.revision logic)
        for key, base_model in REVISION_TO_BASE_MODEL.items():
            if version.startswith(key):
                return base_model
                
        # 3. Fallback/Error
        raise ValueError(f"Base model not found for version: {version}")

    def load_model(self, version: str):
        """
        Loads the model using the initialization logic from run.py:
        1. Initialize ImageClassifier with the *base* model (sets up transforms/config correct).
        2. Load the *local* fine-tuned weights.
        """
        if version in self.models:
            return self.models[version]

        # 1. Determine paths and IDs
        base_model_id = self._get_base_model_id(version)
        local_model_name = f"model_{version.replace('.', '')}"
        local_model_path = MODEL_BASE_PATH / local_model_name
        
        if not local_model_path.exists():
            logger.error(f"Local model directory not found: {local_model_path}")
            raise FileNotFoundError(f"Model directory not found: {local_model_path}")

        logger.info(f"Initializing {version} using base '{base_model_id}' on {self.device}...")
        
        # 2. Initialize with BASE model ID (This ensures 'timm' checks in ImageClassifier work)
        # We pass store_dir="." or similar just to satisfy the init, but we rely on local load later
        clf = ImageClassifier(checkpoint=base_model_id, num_labels=len(CATEGORIES))
        
        # 3. Load Local Weights
        logger.info(f"Loading fine-tuned weights from {local_model_path}...")
        clf.load_model(str(local_model_path))
        
        self.models[version] = clf
        return clf

    def predict(self, image: Image.Image, version: str, topn: int = 1):
        """Runs prediction on a specific version."""
        if version == "all":
            results = {}
            for v in self.available_versions:
                results[v] = self._run_single_inference(v, image, topn)
            return results
        else:
            return {version: self._run_single_inference(version, image, topn)}

    def _run_single_inference(self, version, image, topn):
        try:
            clf = self.load_model(version)
            
            # Use top_n_predictions from classifier.py
            predictions = clf.top_n_predictions(image, top_n=topn)
            
            # Predictions are returned as list of tuples: (index, score)
            # We map index back to category name
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
