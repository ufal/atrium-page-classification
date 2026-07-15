"""
tests/test_model_registry.py – Data-integrity + revision-resolution guards for
model_registry.py.

These are pure-data tests: model_registry.py imports nothing heavy, so unlike
test_folds_split.py (which is torch-gated because it imports `classifier`) these
run in the CPU fast lane.

`_resolve` mirrors the production resolver in service/inference.py
(`ModelManager._get_base_model_id`) and run.py: try an exact key first, then the
first `startswith` match in dict-insertion order. Because the registry relies on
that "first match wins" ordering, a data-only reordering can silently mis-route a
revision to the wrong base model — exactly what these tests pin down.
"""

import pytest

from model_registry import (
    CATEGORIES,
    MODEL_STATIC,
    REVISION_BEST_FOLDS,
    REVISION_BEST_MODELS,
    REVISION_TO_BASE_MODEL,
)

EFFNETV2_L = "timm/tf_efficientnetv2_l.in21k_ft_in1k"
REGNETY_160 = "timm/regnety_160.swag_ft_in1k"


def _resolve(revision: str) -> str:
    """Replicate service/inference.py::_get_base_model_id exactly."""
    if revision in REVISION_TO_BASE_MODEL:
        return REVISION_TO_BASE_MODEL[revision]
    for key, base_model in REVISION_TO_BASE_MODEL.items():
        if revision.startswith(key):
            return base_model
    raise ValueError(f"Base model not found for version: {revision}")


# ── Exact-key resolution ────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "revision, expected",
    [
        ("v1.3", "timm/tf_efficientnetv2_m.in21k_ft_in1k"),
        ("v2.3", "google/vit-base-patch16-224"),
        ("v3.3", "google/vit-base-patch16-384"),
        ("v4.3", REGNETY_160),
        ("v5.3", "google/vit-large-patch16-384"),
    ],
)
def test_exact_revision_resolves(revision, expected):
    assert _resolve(revision) == expected


# ── v*.4 retraining block: must precede the generic single-dot prefixes ──────
@pytest.mark.parametrize(
    "revision, expected",
    [
        ("v1.4", "timm/tf_efficientnetv2_m.in21k_ft_in1k"),
        ("v3.4", "google/vit-base-patch16-384"),
        ("v4.4", REGNETY_160),
        ("v5.4", "google/vit-large-patch16-384"),
    ],
)
def test_v4_block_exact_resolves(revision, expected):
    assert _resolve(revision) == expected


def test_v4_block_ordering_beats_generic_prefix():
    """A non-exact v*.4 sub-revision must still reach the specific entry.

    "v4.40" is not an exact key, so it falls to the startswith loop. Because the
    v*.4 block is inserted before the generic "v4." entry, it correctly resolves
    to regnety_160 rather than the generic effnetv2_l. This guards against a
    regression that moves the v*.4 block below the generic prefixes.
    """
    assert _resolve("v4.40") == REGNETY_160


def test_unknown_revision_raises():
    with pytest.raises(ValueError):
        _resolve("v99.9-does-not-exist")


# ── Known latent bug (issue #15 follow-up) ──────────────────────────────────
@pytest.mark.xfail(
    strict=True,
    reason=(
        "Generic 'v4.' is inserted before 'v4.3' in REVISION_TO_BASE_MODEL, so a "
        "non-exact sub-revision like 'v4.3.1' matches 'v4.' first and resolves to "
        "effnetv2_l instead of regnety_160. Reorder the generic '.3' block "
        "(specific before generic, like the v*.4 block) to fix, then drop xfail."
    ),
)
def test_sub_revision_of_v4_3_resolves_to_specific_model():
    assert _resolve("v4.3.1") == REGNETY_160


# ── Cross-table consistency ─────────────────────────────────────────────────
def test_best_models_subset_of_registry_with_matching_base():
    for revision, base in REVISION_BEST_MODELS.items():
        assert revision in REVISION_TO_BASE_MODEL
        assert REVISION_TO_BASE_MODEL[revision] == base


def test_model_static_base_matches_registry():
    for revision, facts in MODEL_STATIC.items():
        assert revision in REVISION_TO_BASE_MODEL
        assert facts["base_model"] == REVISION_TO_BASE_MODEL[revision]


def test_best_fold_revisions_are_registered():
    for revision in REVISION_BEST_FOLDS:
        assert revision in REVISION_TO_BASE_MODEL


def test_categories_are_eleven_and_unique():
    assert len(CATEGORIES) == 11
    assert len(set(CATEGORIES)) == len(CATEGORIES)
