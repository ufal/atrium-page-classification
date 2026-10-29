import logging
import sys
from pathlib import Path

import torch
from PIL import Image

# Add parent directory to path to import original modules
sys.path.append(str(Path(__file__).parent.parent))

try:
    from classifier import ImageClassifier
    from ensemble import average_prediction_dicts
    from model_registry import CATEGORIES, REVISION_BEST_MODELS, resolve_base_model
except ImportError:
    from classifier import ImageClassifier
    from ensemble import average_prediction_dicts
    from model_registry import CATEGORIES, REVISION_BEST_MODELS, resolve_base_model

# (12-factor XI) No basicConfig() here. This module is imported as a library by
# api.py and by tests; configuring the ROOT logger at import time is a side effect
# that silently overrides whatever the host application chose — and because
# basicConfig() is a no-op once the root logger has handlers, the first import
# wins over the real entrypoint. Emit to a named logger; let __main__ decide
# handlers and level. (issue #61)
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
        # Versions whose inference failed and has not succeeded since: their error is logged
        # once, not once per page (a 529-page document logged 1,599 identical errors).
        self._failing: set = set()
        # Versions an ensemble answer has already been reported without (WARNING once).
        self._warned: set = set()

    def _get_base_model_id(self, version: str) -> str:
        # One resolution rule, shared with service/api.py's up-front `version` check
        # (model_registry.resolve_base_model; atrium-project#32 round 2).
        base_model = resolve_base_model(version)
        if base_model is None:
            raise ValueError(f"Base model not found for version: {version}")
        return base_model

    def get_model_details(self, version: str) -> str:
        if version == "all":
            return "Ensemble (Average of 5 Models)"
        try:
            base = self._get_base_model_id(version)
            return f"{base} ({version})"
        except ValueError as e:
            logger.warning(f"Unknown model revision requested: {e}")
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
                f"Model not found locally at {local_model_path}. Attempting download from Hugging Face ({HF_REPO_NAME}, revision {version})..."
            )
            try:
                clf.load_from_hub(HF_REPO_NAME, revision=version)
                logger.info(f"Saving downloaded model to {local_model_path}...")
                clf.save_model(str(local_model_path))
            except Exception as e:
                logger.error(f"Failed to download model {version} from Hugging Face: {e}")
                raise RuntimeError(f"Model {version} not found locally and could not be downloaded: {e}") from e

        self.models[version] = clf
        return clf

    def warmup(self, versions: list = None):
        for v in versions or self.available_versions:
            try:
                self.load_model(v)
                logger.info(f"Warmed up model {v}")
            except Exception as e:
                logger.warning(f"Could not pre-load model {v}: {e}")

    def predict(self, image: Image.Image, version: str, topn: int = 1):
        """The top-N predictions, or `{"error": ...}`; `predict_detailed` also names the models."""
        result = self.predict_detailed(image, version, topn)
        return result if "error" in result else result["predictions"]

    def predict_detailed(self, image: Image.Image, version: str, topn: int = 1) -> dict:
        """`{"predictions": [...], "models_used": [...]}`, or `{"error": ...}` when no model answered.

        `models_used` lists the revisions that actually contributed (atrium-project#77): the one
        asked for, or every ensemble member that answered. A model that failed to load or to infer
        is left out of the `all` average, and this is where a caller sees it.
        """
        if version == "all":
            return self._predict_averaged(image, topn)
        results = self._run_single_inference(version, image, topn)
        if isinstance(results, dict):
            return results
        return {"predictions": results, "models_used": [version], "per_model": {version: results}}

    def _predict_averaged(self, image, topn):
        predictions_list = []
        answered = []
        failed = []

        # Each member's own top-N (renormalised over its N best), summed per category and divided by
        # the members that answered: the command line's `--best` average (ensemble.average_rdfs), so
        # both sides give the same scores since v1.11.0-beta, and from unrounded scores since
        # v1.12.0-beta (atrium-project#72). Before, the service averaged all eleven probabilities.
        for v in self.available_versions:
            try:
                results = self._run_single_inference(v, image, topn=topn)
                if isinstance(results, dict) and "error" in results:
                    failed.append(v)
                    continue
                predictions_list.append(results)
                answered.append(v)
            except Exception as e:
                logger.warning(f"Ensemble: model {v} failed, skipping: {e}")
                failed.append(v)
                continue

        if failed and predictions_list:
            # The answer is still an ensemble's, of fewer models: say so where it shows.
            level = logging.DEBUG if set(failed) <= self._warned else logging.WARNING
            logger.log(
                level,
                f"Ensemble: {len(predictions_list)} of {len(self.available_versions)} models "
                f"answered; failing: {', '.join(failed)}",
            )
            self._warned |= set(failed)
        if not predictions_list:
            return {"error": "All models failed."}
        # Each member's own top-N beside the average (`per_model=true`, atrium-project#72): what the
        # CLI's `--best --save-intermediates` writes as one CSV per model.
        per_model = {v: results[:topn] for v, results in zip(answered, predictions_list)}
        return {
            "predictions": average_prediction_dicts(predictions_list, CATEGORIES, topn),
            "models_used": answered,
            "per_model": per_model,
        }

    def _run_single_inference(self, version, image, topn):
        try:
            clf = self.load_model(version)
            predictions = clf.top_n_predictions(image, top_n=topn)
            self._failing.discard(version)
            return [{"label": CATEGORIES[i], "score": float(s)} for i, s in predictions]
        except Exception as e:
            if version in self._failing:
                logger.debug(f"Inference failed for model {version} again: {e}")
            else:
                logger.error(f"Inference failed for model {version}: {e}")
                self._failing.add(version)
            return {"error": str(e)}


manager = ModelManager()
