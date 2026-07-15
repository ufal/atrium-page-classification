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

Current ordering facts (verified against live data):
- the v*.4 retraining block precedes the generic single-dot prefixes → safe;
- 'v3.' precedes 'v3.3' and 'v5.' precedes 'v5.3', but each pair maps to the
  SAME base model, so the shadowing is currently harmless;
- 'v4.' precedes 'v4.3' AND they map to DIFFERENT models — the one live bug.
"""

import pytest

from model_registry import (
    CATEGORIES,
    MODEL_STATIC,
    REVISION_BEST_FOLDS,
    REVISION_BEST_MODELS,
    REVISION_TO_BASE_MODEL,
)

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


# ── Shadowed '.3' keys: v3/v5 are harmless, v4 is the live bug ───────────────
def test_v3_v5_shadowing_is_currently_harmless():
    """'v3.' precedes 'v3.3' and 'v5.' precedes 'v5.3' (same wrong order as the
    v4 pair), but each generic/specific pair maps to the SAME base model, so
    resolution of e.g. 'v3.3.1' is unaffected. This pins that safety: if either
    specific model is ever changed without also fixing the key ordering, this
    test fails immediately instead of the bug shipping silently.
    """
    assert REVISION_TO_BASE_MODEL["v3."] == REVISION_TO_BASE_MODEL["v3.3"]
    assert REVISION_TO_BASE_MODEL["v5."] == REVISION_TO_BASE_MODEL["v5.3"]


# ── Known latent bug (issue #15 follow-up) ──────────────────────────────────
@pytest.mark.xfail(
    strict=True,
    reason=(
        "Generic 'v4.' is inserted before 'v4.3' in REVISION_TO_BASE_MODEL and they "
        "map to different models, so a non-exact sub-revision like 'v4.3.1' matches "
        "'v4.' first and resolves to effnetv2_l instead of regnety_160 (v4 is the "
        "only divergent pair — see test_v3_v5_shadowing_is_currently_harmless). "
        "Reorder specific-before-generic, like the v*.4 block, then drop this xfail."
    ),
)
def test_sub_revision_of_v4_3_resolves_to_specific_model():
    assert _resolve("v4.3.1") == REGNETY_160


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Structural form of the same bug: no generic key may precede a specific key "
        "it shadows while mapping to a different model. Currently violated only by "
        "('v4.', 'v4.3'). Fixing the ordering flips this to XPASS — remove both "
        "xfail markers together."
    ),
)
def test_no_generic_key_shadows_a_divergent_specific_key():
    keys = list(REVISION_TO_BASE_MODEL)
    shadowed = []
    for i, spec in enumerate(keys):
        for gen in keys[:i]:
            if spec != gen and spec.startswith(gen):
                if REVISION_TO_BASE_MODEL[spec] != REVISION_TO_BASE_MODEL[gen]:
                    shadowed.append((gen, spec))
    assert shadowed == [], f"generic keys shadow divergent specific keys: {shadowed}"


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
