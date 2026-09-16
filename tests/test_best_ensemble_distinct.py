"""
tests/test_best_ensemble_distinct.py — `--best` must average FIVE models, not one
model five times.

Why this file exists
--------------------
`REVISION_BEST_MODELS` (model_registry.py) is the single source of truth for the
ensemble. `run.py --best` averages it, and `service/inference.py:33` derives
`AVAILABLE_VERSIONS` from the same dict, so the API's `version="all"` path follows
it automatically. One five-key edit changes both.

That makes it a one-line edit away from a defect that CANNOT be seen in the output:
if two revisions resolve to the same weights, averaging still runs, still returns
well-formed Top-N predictions, and still reports "Ensemble (Average of 5 Models)".
Nothing raises. The result is simply a single model wearing an ensemble's label,
with the confidence profile of an ensemble it isn't.

This is not hypothetical, and it is why the flip to `v*.4` has not happened.
Measured against the Hub on 2026-09-16, all five `v*.4` revisions of
`ufal/vit-historical-page` serve the SAME checkpoint::

    revision   config.json "architecture"   model.safetensors
    v1.4       regnety_160                  322,925,148
    v2.4       regnety_160                  322,925,148
    v3.4       regnety_160                  322,925,148
    v4.4       regnety_160                  322,925,148
    v5.4       regnety_160                  322,925,148

against a correctly heterogeneous `v*.3` control (v1.3 = 212,764,964 B
efficientnetv2_m; v2.3 = ViT `hidden_size` 768; v5.3 = 1,214,854,652 B ViT
`hidden_size` 1024). A nonexistent revision (`v9.9`) is rejected by the Hub, so the
five `v*.4` refs do exist — they just do not hold what
`REVISION_TO_BASE_MODEL` says they hold.

So `REVISION_BEST_MODELS` stays on `v*.3` until the four wrong uploads are
replaced. These tests are what make that decision enforceable rather than a
comment someone edits past.

Two layers
----------
1. **Static** (always runs, no network): the registry's own declaration must
   describe five distinct base models. Catches a key swap that collapses the
   ensemble *on paper*.
2. **Published** (``@pytest.mark.slow``, network): the revisions the registry
   names must actually serve distinct architectures on the Hub. Catches the
   defect above — a registry that reads correctly against artifacts that are not.
   It SKIPS on any transport failure so it can never flake a release, and fails
   only when it has really read the configs and they really disagree.

Run the network layer deliberately::

    python -m pytest tests/test_best_ensemble_distinct.py -m slow -v
"""

import json
import urllib.error
import urllib.request

import pytest

from model_registry import MODEL_STATIC, REVISION_BEST_MODELS, REVISION_TO_BASE_MODEL

# Same repo the production loader uses (service/inference.py:31). Duplicated rather
# than imported because service/inference.py imports torch, which this CPU-fast-lane
# test must not drag in.
HF_REPO_NAME = "ufal/vit-historical-page"

_CONFIG_URL = "https://huggingface.co/{repo}/resolve/{revision}/config.json"
_TIMEOUT_S = 20


def _fetch_config(revision: str) -> dict:
    """The revision's config.json, or raise urllib.error.URLError/HTTPError."""
    url = _CONFIG_URL.format(repo=HF_REPO_NAME, revision=revision)
    req = urllib.request.Request(url, headers={"User-Agent": "atrium-page-classification-tests"})
    with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as response:
        return json.loads(response.read().decode("utf-8"))


def _architecture_identity(config: dict) -> str:
    """A stable identity for the checkpoint's architecture.

    timm-wrapped checkpoints (regnety, efficientnetv2) carry a top-level
    ``architecture``; native ViT checkpoints do not, and are distinguished by
    ``architectures`` + ``hidden_size`` (768 = base, 1024 = large) and by the
    patch resolution in ``image_size``. Both shapes appear in this repo, so the
    identity has to span them.
    """
    if "architecture" in config:
        return str(config["architecture"])
    parts = [str(config.get("architectures", ["?"])[0])]
    for key in ("hidden_size", "image_size", "patch_size"):
        if key in config:
            parts.append(f"{key}={config[key]}")
    return "|".join(parts)


# ── layer 1: the registry's own declaration ──────────────────────────────────


def test_best_ensemble_declares_distinct_base_models():
    """Averaging N copies of one model is not an ensemble."""
    by_base: dict[str, list[str]] = {}
    for revision, base_model in REVISION_BEST_MODELS.items():
        by_base.setdefault(base_model, []).append(revision)
    collisions = {base: revs for base, revs in by_base.items() if len(revs) > 1}
    assert not collisions, (
        f"REVISION_BEST_MODELS maps several revisions to the same base model: {collisions}. "
        f"`run.py --best` and the API's version='all' both average this dict, so the result "
        f"would be one model weighted N times while still reporting an N-model ensemble."
    )


def test_best_ensemble_revisions_resolve_to_their_declared_base():
    """The ensemble dict must agree with REVISION_TO_BASE_MODEL, exact keys only.

    An ensemble revision resolved by `startswith` fallback rather than an exact key
    is the `v4.`-shadows-`v4.3` trap (see tests/test_model_registry.py) waiting to
    happen to the ensemble specifically.
    """
    for revision, declared in REVISION_BEST_MODELS.items():
        assert revision in REVISION_TO_BASE_MODEL, (
            f"ensemble revision {revision!r} has no EXACT key in REVISION_TO_BASE_MODEL; "
            f"it would resolve by prefix fallback, which is order-dependent"
        )
        assert REVISION_TO_BASE_MODEL[revision] == declared, (
            f"{revision!r}: REVISION_BEST_MODELS says {declared!r} but "
            f"REVISION_TO_BASE_MODEL says {REVISION_TO_BASE_MODEL[revision]!r}"
        )


def test_every_ensemble_revision_has_model_static_facts():
    """`parallel_best.profile_best_models` needs resolution/params to group by VRAM.

    A missing row does not raise — it degrades the grouping silently, which on a
    shared GPU reads as an OOM in an unrelated job.
    """
    missing = [revision for revision in REVISION_BEST_MODELS if revision not in MODEL_STATIC]
    assert not missing, (
        f"MODEL_STATIC has no rows for ensemble revisions {missing}; "
        f"parallel_best.py loses its VRAM-budgeting hints for them"
    )


# ── layer 2: what the Hub actually serves ────────────────────────────────────


@pytest.mark.slow
def test_published_revisions_hold_distinct_architectures():
    """The registry can be self-consistent and still describe artifacts that aren't.

    Measured 2026-09-16: every `v*.4` revision serves `regnety_160`. Had the flip to
    `v*.4` been made on the strength of the registry alone, `--best` would have
    averaged one model five times, silently. This test is the reason that cannot
    happen again.
    """
    identities: dict[str, str] = {}
    for revision in REVISION_BEST_MODELS:
        try:
            config = _fetch_config(revision)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
            pytest.skip(f"Hugging Face unreachable for revision {revision}: {exc}")
        identities[revision] = _architecture_identity(config)

    by_identity: dict[str, list[str]] = {}
    for revision, identity in identities.items():
        by_identity.setdefault(identity, []).append(revision)
    collisions = {identity: revs for identity, revs in by_identity.items() if len(revs) > 1}

    assert not collisions, (
        f"revisions of {HF_REPO_NAME} that share one architecture: {collisions}. "
        f"Full map: {identities}. The registry names distinct base models, so the "
        f"UPLOADS are wrong, not the registry — re-push the affected revisions before "
        f"making them the `--best` default."
    )


@pytest.mark.slow
def test_published_revisions_match_their_declared_base_family():
    """A revision must serve the family REVISION_TO_BASE_MODEL claims for it."""
    # substrings that must appear in the architecture identity for each family
    families = {
        "timm/regnety_160.swag_ft_in1k": "regnety",
        "timm/tf_efficientnetv2_m.in21k_ft_in1k": "efficientnet",
        "google/vit-base-patch16-224": "hidden_size=768",
        "google/vit-base-patch16-384": "hidden_size=768",
        "google/vit-large-patch16-384": "hidden_size=1024",
    }
    mismatches = []
    for revision, base_model in REVISION_BEST_MODELS.items():
        expected = families.get(base_model)
        if expected is None:
            continue
        try:
            identity = _architecture_identity(_fetch_config(revision))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
            pytest.skip(f"Hugging Face unreachable for revision {revision}: {exc}")
        if expected not in identity:
            mismatches.append(f"{revision}: declared {base_model!r}, published {identity!r}")
    assert not mismatches, "published checkpoints disagree with the registry:\n  " + "\n  ".join(mismatches)
