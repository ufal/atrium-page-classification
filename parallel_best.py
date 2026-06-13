"""
parallel_best.py — opt-in alternative engine for the ``--best`` multi-model run.

Resolves ATRIUM issue #3 (Phase 1, single GPU).

What this engine does (and does NOT do)
----------------------------------------
The honest framing matters here.  On a *single* GPU, holding several models in
VRAM and calling them does **not** reduce GPU math time — CUDA serialises the
kernels, so total FLOPs are unchanged.  The win is purely in the work *around*
the math: today's sequential ``--best`` makes **N passes** over the input (one
DataLoader per model), so every page is read + PIL-decoded N times and the
models are loaded/unloaded N times.

This engine reorganises that into **one data pass**: each page is decoded once
and the shared RGB image is fed to every model resident in VRAM (each model
still applies its own resize/normalise transform, since the 5 best models use
different resolutions/processors — only the *decode* is shareable).  The saving
is therefore the I/O + decode + load/unload-churn portion, which is hardware-
and workload-dependent and should be **measured, not promised**.

The genuine wall-clock win lives in multi-GPU (Phase 2, not implemented here).

Design (mirrors the issue #3 strategy)
--------------------------------------
A. Per-model GPU-size registry written to ``model/gpu_profile.json``, keyed by
   hardware + batch so it self-invalidates when anything changes.
B. On-demand profiling that measures each model's peak footprint on the first
   2 batches at the configured batch size.
C. Memory-aware greedy packing of models into groups that fit the live budget.
D. Grouped single-pass execution with a runtime overflow guard; on overflow a
   group falls back to per-model sequential execution.
E. Aggregation is **unchanged**: every engine produces the same per-model
   ``rdf`` (``[FILE, PAGE, CLASS-1]``) and the same merge, so the combined CSV
   ``{ts}_BEST_{N}_models_TOP-1.csv`` is byte-identical and ``averaging.py`` is
   unaffected.  A parallel run can be diffed against a sequential run to prove
   correctness.
G. Everything is opt-in behind ``--parallel`` (only meaningful with ``--best``)
   and degrades to the guaranteed sequential path on CPU/MPS, on a stale/failed
   registry, or whenever the overflow guard cannot keep a group safe.

torch and classifier are imported lazily inside the functions that need them so
the pure helpers (``pack_models``, ``registry_is_fresh``, ``merge_best``) can be
imported and unit-tested without a GPU or a torch install.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Tunable constants
# ──────────────────────────────────────────────────────────────────────────────

GPU_PROFILE_FILENAME = "gpu_profile.json"

# Fraction of *currently free* VRAM we are willing to fill with co-resident
# model weights + activations, minus a fixed headroom for fragmentation and the
# CUDA context.  Deliberately conservative — overflow forces a per-group
# sequential fallback, which is slower, so it is cheaper to under-pack.
BUDGET_FRACTION = 0.9
HEADROOM_BYTES = 512 * 1024 ** 2          # 512 MiB

# Runtime overflow guard: if free VRAM drops below this after the first 2
# batches of a multi-model group, the static estimate was optimistic — abort
# the group and run its models sequentially instead.
OVERFLOW_MIN_FREE_BYTES = 256 * 1024 ** 2  # 256 MiB

# How many batches to look at when profiling / guarding (the issue's
# "judging by the first 2 batches").
PROFILE_BATCHES = 2


class GroupOverflow(RuntimeError):
    """Raised mid-group when the live VRAM headroom falls into the danger zone."""


# ──────────────────────────────────────────────────────────────────────────────
# A/C — registry freshness + packing (pure, torch-free, unit-testable)
# ──────────────────────────────────────────────────────────────────────────────

def registry_is_fresh(
    profile: Optional[dict],
    gpu_name: str,
    total_vram_bytes: int,
    batch: int,
    required_revs: List[str],
) -> bool:
    """Return True iff *profile* was measured on the same GPU + batch and covers
    every required model revision.

    Any mismatch (different card, different total VRAM, different batch size, or
    a missing model) means the recorded peak footprints no longer apply and the
    registry must be re-measured (or we fall back to sequential).
    """
    if not profile:
        return False
    gpu = profile.get("gpu") or {}
    if gpu.get("name") != gpu_name:
        return False
    if gpu.get("total_vram_bytes") != total_vram_bytes:
        return False
    if profile.get("batch") != batch:
        return False
    models = profile.get("models") or {}
    return all(rev in models for rev in required_revs)


def pack_models(
    model_sizes: Dict[str, int],
    budget_bytes: float,
    max_group: Optional[int] = None,
) -> List[List[str]]:
    """Greedily pack models into groups whose summed ``peak_bytes`` ≤ budget.

    First-fit-decreasing: models are sorted by recorded peak footprint
    (descending, ties broken by revision name for determinism), then each is
    placed into the first existing group it still fits.  A model larger than the
    whole budget lands in a group of 1 (= one extra data pass, still safe).
    ``max_group`` caps the number of models per group — ``max_group=2``
    reproduces the issue's "two at a time".

    The returned grouping controls *execution* order only; the final column
    order in the combined CSV is fixed separately by ``merge_best`` so output
    stays byte-identical regardless of how models are packed.
    """
    if budget_bytes <= 0:
        # No usable budget recorded → every model on its own (degrades to the
        # one-model-per-pass behaviour, which is what sequential already does).
        return [[rev] for rev in sorted(model_sizes)]

    ordered = sorted(model_sizes.items(), key=lambda kv: (-kv[1], kv[0]))
    groups: List[List[str]] = []

    for rev, size in ordered:
        placed = False
        for group in groups:
            if max_group is not None and len(group) >= max_group:
                continue
            group_size = sum(model_sizes[r] for r in group)
            if group_size + size <= budget_bytes:
                group.append(rev)
                placed = True
                break
        if not placed:
            groups.append([rev])

    return groups


# ──────────────────────────────────────────────────────────────────────────────
# E — aggregation (pure, torch-free, unit-testable, byte-identical)
# ──────────────────────────────────────────────────────────────────────────────

def merge_best(
    revision_best_models: Dict[str, str],
    rdf_by_rev: Dict[str, "pd.DataFrame"],
) -> "pd.DataFrame":
    """Merge per-model ``rdf``s into the wide combined frame.

    Iterates revisions in the canonical ``revision_best_models`` order (NOT
    group/execution order) so the column layout is exactly
    ``FILE, PAGE, CLASS-1-v1.3, CLASS-1-v2.3, …`` — identical to the original
    inline merge, which ``supplement_scripts/averaging.py`` parses via
    ``^CLASS-(\\d+)-(.+)$``.
    """
    combined_df = pd.DataFrame()
    for rev in revision_best_models:
        rdf = rdf_by_rev[rev]
        renamed_columns = {col: f"{col}-{rev}" for col in rdf.columns if col not in ["FILE", "PAGE"]}
        rdf_renamed = rdf.rename(columns=renamed_columns)
        if combined_df.empty:
            combined_df = rdf_renamed
        else:
            combined_df = pd.merge(combined_df, rdf_renamed, on=["FILE", "PAGE"], how="outer")
    return combined_df


def _build_rdf(test_images: list, predictions: list, categories: list) -> "pd.DataFrame":
    """Top-1 predictions → the single-model frame used by ``merge_best``.

    Reproduces exactly what the original ``--best`` loop did: build via
    ``dataframe_results(..., 1, None)``, drop the human-readable ``CATEGORY``
    alias (which would otherwise become a redundant ``CATEGORY-{rev}`` column),
    and sort by ``[FILE, PAGE]``.
    """
    from utils import dataframe_results  # lazy: utils pulls matplotlib/sklearn

    rdf, _ = dataframe_results(test_images, predictions, categories, 1, None)
    rdf.drop(columns=["CATEGORY"], inplace=True, errors="ignore")
    rdf.sort_values(["FILE", "PAGE"], ascending=[True, True], inplace=True)
    return rdf


# ──────────────────────────────────────────────────────────────────────────────
# B — on-demand profiling
# ──────────────────────────────────────────────────────────────────────────────

def _model_resolution(clf) -> Optional[int]:
    """Best-effort input resolution for the registry (informational only)."""
    try:
        if clf.processor is not None:
            return int(clf.processor.size["height"])
    except Exception:
        pass
    try:
        return int(clf.model.config.pretrained_cfg["input_size"][-1])
    except Exception:
        return None


def profile_best_models(
    revision_best_models: Dict[str, str],
    model_dir: str,
    cp_dir: str,
    num_labels: int,
    sample_images: list,
    batch: int,
    registry_path: str,
) -> dict:
    """Measure each best model's peak VRAM footprint on the first 2 batches and
    persist a registry keyed by hardware + batch.

    Records weights + activation peak at the *actual* configured batch, because
    activations dominate and scale with batch × resolution, so batch must be
    part of the key (see ``registry_is_fresh``).
    """
    import torch
    from classifier import ImageClassifier

    dev = torch.device("cuda")
    profile: dict = {
        "gpu": {
            "name": torch.cuda.get_device_name(dev),
            "total_vram_bytes": int(torch.cuda.mem_get_info(dev)[1]),
            "device_index": torch.cuda.current_device(),
            "torch": torch.__version__,
        },
        "batch": batch,
        "measured_at": datetime.now(tz=timezone.utc).isoformat(),
        "models": {},
    }

    probe = sample_images[: PROFILE_BATCHES * batch]

    for rev, base_model in revision_best_models.items():
        print(f"[parallel-best] Profiling {rev} ({base_model}) on {PROFILE_BATCHES} batches…")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)

        clf = ImageClassifier(checkpoint=base_model, num_labels=num_labels, store_dir=cp_dir)
        clf.load_model(f"{model_dir}/model_{rev.replace('.', '')}")
        clf.model.eval()

        loader = clf.create_dataloader(probe, batch)
        with torch.no_grad():
            for i, b in enumerate(loader):
                if b is None or (isinstance(b, tuple) and b[0] is None):
                    continue
                clf.model(pixel_values=b["pixel_values"].to(dev))
                if i >= PROFILE_BATCHES - 1:
                    break

        peak = int(torch.cuda.max_memory_allocated(dev))
        params_bytes = int(sum(p.numel() * p.element_size() for p in clf.model.parameters()))

        profile["models"][rev] = {
            "base_model": base_model,
            "resolution": _model_resolution(clf),
            "peak_bytes": peak,
            "params_bytes": params_bytes,
        }

        del clf
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(registry_path) or ".", exist_ok=True)
    with open(registry_path, "w", encoding="utf-8") as fh:
        json.dump(profile, fh, ensure_ascii=False, indent=2)
    print(f"[parallel-best] GPU profile written → {registry_path}")
    return profile


def load_or_profile_registry(
    revision_best_models: Dict[str, str],
    model_dir: str,
    cp_dir: str,
    num_labels: int,
    sample_images: list,
    batch: int,
    registry_path: str,
) -> dict:
    """Load the registry if present and fresh, otherwise (re)profile."""
    import torch

    dev = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(dev)
    total_vram = int(torch.cuda.mem_get_info(dev)[1])

    profile = None
    if os.path.exists(registry_path):
        try:
            with open(registry_path, "r", encoding="utf-8") as fh:
                profile = json.load(fh)
        except Exception as e:
            print(f"[parallel-best] Could not read registry ({e}); will re-profile.")
            profile = None

    if registry_is_fresh(profile, gpu_name, total_vram, batch, list(revision_best_models)):
        print(f"[parallel-best] Reusing fresh GPU profile → {registry_path}")
        return profile

    print("[parallel-best] GPU profile missing/stale → profiling now.")
    return profile_best_models(
        revision_best_models, model_dir, cp_dir, num_labels, sample_images, batch, registry_path
    )


# ──────────────────────────────────────────────────────────────────────────────
# D — grouped single-pass execution (one data pass, shared decode)
# ──────────────────────────────────────────────────────────────────────────────

def _build_multi_loader(image_paths: list, transforms_by_rev: Dict[str, object], batch: int):
    """Build a DataLoader that decodes each image **once** and applies every
    resident model's own transform, yielding ``{rev: pixel_values}`` per batch.

    None-handling mirrors ``ImageDataset`` / ``custom_collate`` exactly so a
    corrupted image is skipped identically across engines, keeping per-model
    prediction order — and therefore the output CSV — byte-identical.
    torch + torchvision are imported here so the module stays importable without
    a GPU.
    """
    import torch
    from torch.utils.data import DataLoader, Dataset
    from PIL import Image

    class _MultiTransformDataset(Dataset):
        def __init__(self, paths, transforms_by_rev):
            self.paths = paths
            self.transforms_by_rev = transforms_by_rev

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            path = self.paths[idx]
            try:
                image = Image.open(path)
                if image.mode != "RGB":
                    # Identical white-background RGBA→RGB conversion to ImageDataset
                    image_alpha = image.convert("RGBA")
                    new_image = Image.new("RGBA", image_alpha.size, "WHITE")
                    new_image.paste(image_alpha, (0, 0), image_alpha)
                    image = new_image.convert("RGB")
                return {rev: t(image) for rev, t in self.transforms_by_rev.items()}
            except Exception as e:
                print(path, e)
                return None

    def _multi_collate(batch):
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            return None
        revs = list(batch[0].keys())
        return {rev: torch.stack([b[rev] for b in batch]) for rev in revs}

    dataset = _MultiTransformDataset(image_paths, transforms_by_rev)
    loader = DataLoader(dataset, batch_size=batch, shuffle=False, collate_fn=_multi_collate)
    print(
        f"[parallel-best] Shared-decode dataloader ready: {len(image_paths)} images, "
        f"{len(transforms_by_rev)} model(s), batch={batch}"
    )
    return loader


def _run_group_single_pass(
    group_revs: List[str],
    revision_best_models: Dict[str, str],
    test_images: list,
    categories: list,
    batch: int,
    model_dir: str,
    cp_dir: str,
) -> Dict[str, "pd.DataFrame"]:
    """Load every model in *group_revs* into VRAM, stream the input once, and
    return ``{rev: rdf}``.  Raises :class:`GroupOverflow` if the live VRAM
    headroom enters the danger zone after the first 2 batches."""
    import torch
    from classifier import ImageClassifier

    dev = torch.device("cuda")
    num_labels = len(categories)

    clfs: Dict[str, object] = {}
    transforms_by_rev: Dict[str, object] = {}
    try:
        for rev in group_revs:
            base_model = revision_best_models[rev]
            print(f"[parallel-best] Loading {rev} ({base_model}) into VRAM…")
            clf = ImageClassifier(checkpoint=base_model, num_labels=num_labels, store_dir=cp_dir)
            clf.load_model(f"{model_dir}/model_{rev.replace('.', '')}")
            clf.model.eval()
            clfs[rev] = clf
            transforms_by_rev[rev] = clf.eval_transforms

        loader = _build_multi_loader(test_images, transforms_by_rev, batch)

        preds_by_rev: Dict[str, list] = {rev: [] for rev in group_revs}
        torch.cuda.reset_peak_memory_stats(dev)

        with torch.no_grad():
            for ib, batch_data in enumerate(loader):
                if batch_data is None:
                    continue
                for rev in group_revs:
                    pixel_values = batch_data[rev].to(dev)
                    logits = clfs[rev].model(pixel_values=pixel_values).logits
                    # top-1 == argmax of logits == argmax of softmax (matches
                    # ImageClassifier.infer_dataloader for top_n==1)
                    preds_by_rev[rev].extend(logits.argmax(-1).tolist())

                # Runtime overflow guard (the "first 2 batches" safety net).
                if ib == (PROFILE_BATCHES - 1) and len(group_revs) > 1:
                    free, _ = torch.cuda.mem_get_info(dev)
                    if free < OVERFLOW_MIN_FREE_BYTES:
                        raise GroupOverflow(
                            f"only {free / 1e6:.0f} MB free after {PROFILE_BATCHES} "
                            f"batches with {len(group_revs)} co-resident models"
                        )

        out = {rev: _build_rdf(test_images, preds_by_rev[rev], categories) for rev in group_revs}
        return out
    finally:
        for rev in list(clfs):
            del clfs[rev]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ──────────────────────────────────────────────────────────────────────────────
# Sequential engine (today's loop, extracted) + parallel engine
# ──────────────────────────────────────────────────────────────────────────────

def run_best_sequential(
    revision_best_models: Dict[str, str],
    test_images: list,
    categories: list,
    batch: int,
    model_dir: str,
    cp_dir: str,
) -> Dict[str, "pd.DataFrame"]:
    """The guaranteed fallback: one DataLoader per model, N passes over the
    input.  Behaviourally identical to the original directory ``--best`` loop
    (minus the inline merge/paradata, which the caller now owns)."""
    import torch
    from classifier import ImageClassifier

    num_labels = len(categories)
    rdf_by_rev: Dict[str, "pd.DataFrame"] = {}

    for rev, base_model in revision_best_models.items():
        print(f"\nLoading best model for revision {rev} based on {base_model}...")
        clf = ImageClassifier(checkpoint=base_model, num_labels=num_labels, store_dir=cp_dir)
        clf.load_model(f"{model_dir}/model_{rev.replace('.', '')}")

        loader = clf.create_dataloader(test_images, batch)
        predictions, _ = clf.infer_dataloader(loader, 1, False)
        rdf_by_rev[rev] = _build_rdf(test_images, predictions, categories)

        # Explicitly release GPU memory after each model to avoid accumulation.
        del clf
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return rdf_by_rev


def run_best_parallel(
    revision_best_models: Dict[str, str],
    test_images: list,
    categories: list,
    batch: int,
    model_dir: str,
    cp_dir: str,
    registry_path: str,
) -> Dict[str, "pd.DataFrame"]:
    """Profile → pack → grouped single-pass, with a per-group sequential
    fallback on overflow.  Returns ``{rev: rdf}`` for every best model."""
    import torch

    num_labels = len(categories)
    dev = torch.device("cuda")

    profile = load_or_profile_registry(
        revision_best_models, model_dir, cp_dir, num_labels, test_images, batch, registry_path
    )
    sizes = {rev: int(profile["models"][rev]["peak_bytes"]) for rev in revision_best_models}

    free, total = torch.cuda.mem_get_info(dev)
    budget = free * BUDGET_FRACTION - HEADROOM_BYTES
    groups = pack_models(sizes, budget)  # default: as many as fit

    print(
        f"[parallel-best] Free {free / 1e9:.2f} GB / total {total / 1e9:.2f} GB, "
        f"budget {budget / 1e9:.2f} GB → {len(groups)} group(s): "
        + ", ".join("[" + "+".join(g) + "]" for g in groups)
    )

    rdf_by_rev: Dict[str, "pd.DataFrame"] = {}
    for group in groups:
        if len(group) == 1:
            # Single-model group: a single-pass over one model == sequential.
            sub = {group[0]: revision_best_models[group[0]]}
            rdf_by_rev.update(run_best_sequential(sub, test_images, categories, batch, model_dir, cp_dir))
            continue
        try:
            rdf_by_rev.update(
                _run_group_single_pass(group, revision_best_models, test_images, categories,
                                       batch, model_dir, cp_dir)
            )
        except GroupOverflow as e:
            print(f"[parallel-best] Overflow guard tripped for [{'+'.join(group)}] ({e}); "
                  f"running this group sequentially instead.")
            sub = {rev: revision_best_models[rev] for rev in group}
            rdf_by_rev.update(run_best_sequential(sub, test_images, categories, batch, model_dir, cp_dir))

    return rdf_by_rev


# ──────────────────────────────────────────────────────────────────────────────
# Public entry points used by run.py
# ──────────────────────────────────────────────────────────────────────────────

def cuda_available() -> bool:
    """True only when a CUDA device with the mem-info APIs is usable.  MPS/CPU
    return False, forcing the sequential path (no ``mem_get_info`` there)."""
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def run_best_directory(
    revision_best_models: Dict[str, str],
    test_images: list,
    categories: list,
    batch: int,
    model_dir: str,
    cp_dir: str,
    output_dir: str,
    time_stamp: str,
    use_parallel: bool = False,
    paradata_logger=None,
    registry_path: Optional[str] = None,
) -> "pd.DataFrame":
    """Directory-level ``--best``: choose the engine, run it, log paradata, write
    the combined CSV.  Returns the combined DataFrame.

    The parallel engine is attempted only when explicitly requested *and* CUDA
    is available; any failure falls back to sequential so a run never regresses.
    """
    if registry_path is None:
        registry_path = os.path.join(model_dir, GPU_PROFILE_FILENAME)

    engine = "sequential"
    rdf_by_rev: Optional[Dict[str, "pd.DataFrame"]] = None

    if use_parallel and cuda_available():
        try:
            rdf_by_rev = run_best_parallel(
                revision_best_models, test_images, categories, batch, model_dir, cp_dir, registry_path
            )
            engine = "parallel"
        except Exception as e:  # never let the opt-in path break a run
            print(f"[parallel-best] Parallel engine failed ({e}); falling back to sequential.")
            rdf_by_rev = None
    elif use_parallel:
        print("[parallel-best] --parallel requested but no CUDA device available; "
              "using sequential --best.")

    if rdf_by_rev is None:
        rdf_by_rev = run_best_sequential(revision_best_models, test_images, categories, batch, model_dir, cp_dir)
        engine = "sequential"

    # Paradata: one csv success per model, in canonical order — identical total
    # count to the original loop, regardless of engine.
    if paradata_logger is not None:
        for rev in revision_best_models:
            paradata_logger.log_success("csv", len(rdf_by_rev[rev].index))

    combined_df = merge_best(revision_best_models, rdf_by_rev)
    out_path = f"{output_dir}/tables/{time_stamp}_BEST_{len(revision_best_models.keys())}_models_TOP-1.csv"
    combined_df.to_csv(out_path, sep=",", index=False)
    print(f"[best:{engine}] Combined predictions written → {out_path}")
    return combined_df


def predict_file_best(
    revision_best_models: Dict[str, str],
    file_path: str,
    categories: list,
    model_dir: str,
    cp_dir: str,
) -> Dict[str, tuple]:
    """Single-file ``--best``: one model at a time, returns
    ``{rev: (labels, scores)}`` for console output.

    There is no parallel benefit for a single image, so this stays sequential;
    extracting it here still de-duplicates the model-load/predict pattern that
    was previously copy-pasted between the file and directory ``--best`` loops.
    """
    import torch
    from classifier import ImageClassifier

    num_labels = len(categories)
    out: Dict[str, tuple] = {}

    for rev, base_model in revision_best_models.items():
        print(f"\nLoading best model for revision {rev} based on {base_model}...")
        clf = ImageClassifier(checkpoint=base_model, num_labels=num_labels, store_dir=cp_dir)
        clf.load_model(f"{model_dir}/model_{rev.replace('.', '')}")

        pred_scores = clf.top_n_predictions(file_path, len(categories))
        labels = [categories[i[0]] for i in pred_scores]
        scores = [round(i[1], 3) for i in pred_scores]
        out[rev] = (labels, scores)

        del clf
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return out
