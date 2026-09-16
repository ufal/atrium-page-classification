"""
model_registry.py - Single source of truth for ATRIUM Page Classification model definitions.
"""

import sys

# ── the 11 page categories ────────────────────────────────────────────────────
# THE ORDER OF THIS LIST IS LOAD-BEARING. It is the label→index binding every
# checkpoint was trained against: `utils.collect_images()` derives the same order by
# alphabetically sorting the dataset's category sub-directories, and the fine-tuned
# models store `LABEL_0`..`LABEL_10` against exactly these positions. Reordering the
# list fails nothing loudly -- it silently relabels every prediction. So the literal
# stays spelled out here, in training order, and is built from nothing.
#
# The same 11 labels are ALSO declared hub-side in `atrium_vocab.py`, as the
# `page-category` SKOS concept scheme:
#
#   * `atrium_vocab.PAGE_CATEGORIES`              -- the same 11 values, in this order;
#   * `atrium_vocab.CONCEPTS["page-category"]`    -- one definition per label, lifted
#     verbatim from README.md's "Categories 🪧" table. The README and the registry carry
#     the prose; this list carries the binding.
#   * `atrium_vocab.COLLECTIONS["page-category"]` -- the three orthogonal criteria the
#     README names, as five `skos:Collection` facets: graphical / tabular / handwritten
#     / printed / typed. Collections rather than `skos:broader`, because a label belongs
#     to several facets at once and so has no single parent.
#
# The registry is deliberately NOT imported to BUILD this list: `labels_for()` returns a
# SORTED tuple, which is not the training order, so substituting it would be a silent
# behaviour change of exactly the kind described above. Only the advisory SET comparison
# below crosses over, and a set has no order.
#
# Every label has a stable URI -- `atrium_vocab.concept_uri("page-category", label)`,
# e.g. `https://w3id.org/atrium/page-category/TEXT_HW`. Go through `category_uri()`
# below rather than formatting one by hand, so that the day the registry's `SKOS_BASE`
# is repointed there is nothing to update here.
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


def category_uri(label: str) -> str | None:
    """Stable SKOS URI for one page category, or `None` when the registry is unavailable.

    One place for downstream code to turn a bare `CATEGORIES` value into an identifier,
    so no caller hard-codes `SKOS_BASE` or the `page-category` scheme name.

    Returns `None` rather than raising when `atrium_vocab.py` has not been vendored
    alongside this file: an absent registry must never be able to break inference.

    No membership check is performed -- the URI is derived from the label as spelled,
    which is what keeps `URI -> value` a plain suffix split. Use
    `atrium_vocab.validate_labels("page-category", values)` when you want validation.
    """
    try:
        from atrium_vocab import concept_uri
    except ImportError:  # registry not vendored here -- advisory, never fatal
        return None
    return concept_uri("page-category", label)


# ── advisory consistency check (never fatal) ──────────────────────────────────
# MEMBERSHIP ONLY, against the hub-canonical registry, reported on stderr in the house
# idiom of `atrium_document.py::_note()`: visible, but it neither raises nor touches
# `CATEGORIES`. A registry that was never vendored here, or one that has drifted, must
# not be able to stop a training or inference run -- but two declarations of the same
# controlled set drifting apart in complete silence is the failure mode the registry
# exists to prevent, so the disagreement is at least said out loud.
try:
    from atrium_vocab import labels_for as _vocab_labels_for
except ImportError:  # atrium_vocab.py not vendored alongside -- abstain quietly
    pass
else:
    _only_here = sorted(set(CATEGORIES) - set(_vocab_labels_for("page-category")))
    _only_registry = sorted(set(_vocab_labels_for("page-category")) - set(CATEGORIES))
    if _only_here or _only_registry:
        print(
            f"[model_registry] NOTE – CATEGORIES and the atrium_vocab 'page-category' scheme disagree: "
            f"only here {_only_here}; only in the registry {_only_registry}. "
            f"Definitions live in atrium_vocab.CONCEPTS['page-category'] and README.md. "
            f"Nothing is changed: this list stays in training order and remains authoritative here.",
            file=sys.stderr,
        )

# TWO NUMBERING NAMESPACES LIVE IN THIS DICT. Reading it as one is the mistake.
#
#  (a) SWEEP numbering -- one number per candidate base model, as recorded in
#      model_accuracies_new.csv and matched by the GENERIC single-dot prefixes below:
#          v4.  = tf_efficientnetv2_L (csv v4.3.5, 98.62)   v6.  = regnety_120  (v6.3.5)
#          v7.  = regnety_160         (csv v7.3.1, 99.16)   v8.  = regnety_640  (v8.3.5)
#          v9./v10./v11. = the dit family              v12. = tf_efficientnetv2_M (v12.3.1)
#
#  (b) PUBLISHED numbering -- the five sweep WINNERS, re-published on the Hub under
#      compact names v1.3..v5.3 (and v1.4..v5.4). Matched by the EXACT keys below:
#          v1.3 = tf_efficientnetv2_m   (sweep v12.3.1)  <-- renumbered
#          v2.3 = vit-base-patch16-224  (sweep v2.3.5)
#          v3.3 = vit-base-patch16-384  (sweep v3.3.2)
#          v4.3 = regnety_160           (sweep v7.3.1)   <-- renumbered
#          v5.3 = vit-large-patch16-384 (sweep v5.3.2)
#
# The two namespaces COLLIDE on "v1.3" and "v4.3": in (a) those strings mean
# efficientnetv2_S and efficientnetv2_L, in (b) efficientnetv2_m and regnety_160. The
# published meaning wins, and it wins by EXACT-KEY-FIRST resolution, not by ordering
# luck -- see _resolve in tests/test_model_registry.py and the ordering note below.
#
# So model_accuracies_new.csv is NOT stale and must not be "corrected" to match this
# dict: it is a faithful record of namespace (a). The Hub model card records namespace
# (b) and agrees with the exact keys here (regnety_160 = v4.3 = 99.17, "Best & Small").
# Anyone reconciling the CSV against REVISION_BEST_MODELS is comparing two different
# numbering schemes. (atrium-project#53)
REVISION_TO_BASE_MODEL = {
    "v10.": "microsoft/dit-large-finetuned-rvlcdip",
    "v11.": "microsoft/dit-large",
    "v12.": "timm/tf_efficientnetv2_m.in21k_ft_in1k",
    "v1.3": "timm/tf_efficientnetv2_m.in21k_ft_in1k",
    "v2.3": "google/vit-base-patch16-224",
    # ── v*.4 retraining set (issue #15) ───────────────────────────────────────
    # These MUST precede the generic single-dot prefixes ("v3.", "v4.", "v5."):
    # run.py resolves the base model with `next(key ... if revision.startswith(key))`
    # (first match wins), so e.g. "v4.4".startswith("v4.") would otherwise grab the
    # generic effnetv2_l entry instead of regnety_160. Explicit entries first → correct.
    "v1.4": "timm/tf_efficientnetv2_m.in21k_ft_in1k",
    "v2.4": "google/vit-base-patch16-224",
    "v3.4": "google/vit-base-patch16-384",
    "v4.4": "timm/regnety_160.swag_ft_in1k",
    "v5.4": "google/vit-large-patch16-384",
    # ──────────────────────────────────────────────────────────────────────────
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
# NOTE (issue #15): the v*.4 models are retrained on the new dataset (N−318 pages). They share
# the same base models as v*.3, so once they become the canonical ensemble default, swap the
# keys above to "v1.4".."v5.4". Kept on v*.3 for now so `--best` stays unchanged.
#
# DO NOT MAKE THAT SWAP YET -- the blocker is upstream, not here (measured 2026-09-16).
# All five v*.4 revisions of ufal/vit-historical-page currently serve the SAME checkpoint:
# config.json "architecture": "regnety_160", model.safetensors 322,925,148 bytes, on every
# one of v1.4/v2.4/v3.4/v4.4/v5.4. Only v4.4 is what its name claims; the other four are
# copies of it. The v*.3 control is correctly heterogeneous (v1.3 tf_efficientnetv2_m,
# v2.3 ViT hidden_size 768 @224, v3.3 ViT 768 @384, v5.3 ViT 1024 @384), and a nonexistent
# ref (v9.9) is rejected by the Hub, so the five v*.4 refs do exist -- they just do not hold
# what REVISION_TO_BASE_MODEL declares for them.
#
# Why that is worth a paragraph rather than a one-line TODO: swapping the keys in this state
# fails SILENTLY. `run.py --best` and the API's version="all" would average one model with
# itself five times, return well-formed Top-N predictions, and still report "Ensemble
# (Average of 5 Models)". Nothing raises; only the confidence profile changes.
#
# Sequence to unblock, in order:
#   1. ./data_scripts/unix/hf_reupload_v4_revisions.sh --audit
#   2. ./data_scripts/unix/hf_reupload_v4_revisions.sh --models-dir ./model --apply
#   3. python -m pytest tests/test_best_ensemble_distinct.py -m slow -v     # must be GREEN
#   4. only then swap the five keys above, and update setup/config.txt [HF] latest,
#      README.md's "latest v4.3 is considered to be default", service/README.md's
#      `run.py --hf -rev vX.3`, para_config.txt, and the agent-skill branch copy of this
#      file (model_registry.py is para-drift-adjacent: it is compared byte-for-byte
#      between the default and agent-skill branches by tools/skill_drift_check.py).
# tests/test_best_ensemble_distinct.py is the standing guard for steps 1-3.

# Explicit per-model fold columns for retraining on the pre-computed cross-validation split
# (issue #15). Single source of truth: revision -> column in the folds CSV. Rule:
# splitN ↔ foldN column ↔ seed = 420 + (N−1).
REVISION_BEST_FOLDS = {
    "v1.4": "fold1",
    "v2.4": "fold5",
    "v3.4": "fold2",
    "v4.4": "fold1",
    "v5.4": "fold2",
}

# Hardware/torch/batch-independent model facts (fp32 weights, params only).
MODEL_STATIC = {
    "v1.3": {"base_model": "timm/tf_efficientnetv2_m.in21k_ft_in1k", "resolution": 384, "params_bytes": 211489788},
    "v2.3": {"base_model": "google/vit-base-patch16-224", "resolution": 224, "params_bytes": 343228460},
    "v3.3": {"base_model": "google/vit-base-patch16-384", "resolution": 384, "params_bytes": 344395820},
    "v4.3": {"base_model": "timm/regnety_160.swag_ft_in1k", "resolution": 384, "params_bytes": 322393660},
    "v5.3": {"base_model": "google/vit-large-patch16-384", "resolution": 384, "params_bytes": 1214808108},
    # ── v*.4, staged ahead of the ensemble swap (see the NOTE above) ──────────────────────
    # parallel_best.profile_best_models reads these to group models by VRAM before running
    # them; a revision with no row does not raise, it just loses its budgeting hint, which
    # on a shared GPU surfaces as an OOM in someone else's job. Adding the rows now means
    # the swap is a five-key edit rather than a five-key edit plus a silent regression.
    #
    # Values MIRROR the v*.3 row of the same base model, because that is what
    # REVISION_TO_BASE_MODEL declares the v*.4 set to be. They deliberately do NOT describe
    # what the Hub serves today (all five are regnety_160, 322,925,148 B) -- this table
    # states the contract; tests/test_best_ensemble_distinct.py checks reality against it.
    "v1.4": {"base_model": "timm/tf_efficientnetv2_m.in21k_ft_in1k", "resolution": 384, "params_bytes": 211489788},
    "v2.4": {"base_model": "google/vit-base-patch16-224", "resolution": 224, "params_bytes": 343228460},
    "v3.4": {"base_model": "google/vit-base-patch16-384", "resolution": 384, "params_bytes": 344395820},
    "v4.4": {"base_model": "timm/regnety_160.swag_ft_in1k", "resolution": 384, "params_bytes": 322393660},
    "v5.4": {"base_model": "google/vit-large-patch16-384", "resolution": 384, "params_bytes": 1214808108},
}
