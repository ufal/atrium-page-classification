"""
model_registry.py - Single source of truth for ATRIUM Page Classification model definitions.
"""

CATEGORIES = [
    "DRAW",
    "DRAW_L",
    "LINE_HW",
    "LINE_P",
    "LINE_T",
    "PHOTO",
    "PHOTO_L",
    "TEXT",
    "TEXT_HW",
    "TEXT_P",
    "TEXT_T",
]

REVISION_TO_BASE_MODEL = {
    "v10.": "microsoft/dit-large-finetuned-rvlcdip",
    "v11.": "microsoft/dit-large",
    "v12.": "timm/tf_efficientnetv2_m.in21k_ft_in1k",
    "v1.3": "timm/tf_efficientnetv2_m.in21k_ft_in1k",
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

# Best models subset for ensemble
REVISION_BEST_MODELS = {
    "v1.3": "timm/tf_efficientnetv2_m.in21k_ft_in1k",
    "v2.3": "google/vit-base-patch16-224",
    "v3.3": "google/vit-base-patch16-384",
    "v4.3": "timm/regnety_160.swag_ft_in1k",
    "v5.3": "google/vit-large-patch16-384",
}

# Hardware/torch/batch-independent model facts (fp32 weights, params only).
MODEL_STATIC = {
    "v1.3": {"base_model": "timm/tf_efficientnetv2_m.in21k_ft_in1k", "resolution": 384, "params_bytes": 211489788},
    "v2.3": {"base_model": "google/vit-base-patch16-224", "resolution": 224, "params_bytes": 343228460},
    "v3.3": {"base_model": "google/vit-base-patch16-384", "resolution": 384, "params_bytes": 344395820},
    "v4.3": {"base_model": "timm/regnety_160.swag_ft_in1k", "resolution": 384, "params_bytes": 322393660},
    "v5.3": {"base_model": "google/vit-large-patch16-384", "resolution": 384, "params_bytes": 1214808108},
}
