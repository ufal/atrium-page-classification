"""
parallel_best.py  –  Memory-aware parallel execution of the --best model ensemble.

Implements the design from Issue #3:
  * Per-model GPU-size registry  (model/gpu_profile.json)
  * On-demand 2-batch profiling
  * Memory-aware packing scheduler
  * Grouped single-pass execution + runtime overflow guard
  * Post-run averaging via averaging.py logic → final averaged CSV
  * Sequential fallback whenever parallel is unsafe

The output of run_best_parallel / run_best_sequential is always a fully
averaged CSV (CLASS-1, SCORE-1 … SCORE-N per page) in addition to the
optional wide intermediate BEST_N_models_TOP-1.csv.  This makes --best
self-contained: no manual averaging.py call required afterwards.

Public surface used by run.py
------------------------------
  run_best_models(
      test_images, categories, revision_best_models,
      model_dir, cp_dir, batch, top_N,
      output_dir, time_stamp, paradata_logger,
      parallel=False,
  ) -> str   # path to the averaged CSV
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Averaging helpers (inlined subset of supplement_scripts/averaging.py logic
# so this module has no import-path dependency on supplement_scripts/).
# ---------------------------------------------------------------------------

def _melt_rdf(rdf: pd.DataFrame, rev: str) -> pd.DataFrame:
    """Convert a single-model top-1 rdf (FILE, PAGE, CLASS-1) into long form."""
    col = f"CLASS-1-{rev}"
    tmp = rdf[["FILE", "PAGE", "CLASS-1"]].copy()
    tmp.rename(columns={"CLASS-1": col}, inplace=True)
    return tmp


def _average_wide_df(wide_df: pd.DataFrame, categories: list, top_N: int) -> pd.DataFrame:
    """
    Given a wide multi-model DataFrame (FILE, PAGE, CLASS-1-v1.3, CLASS-1-v2.3 …)
    compute per-page majority-vote averaged scores and return a tidy DataFrame
    with columns  FILE, PAGE, CLASS-1 … CLASS-N, SCORE-1 … SCORE-N.

    Each model vote counts as score=1; averaging yields the fraction of models
    that predicted each class.  Ties are broken by insertion order.
    """
    import re
    model_cols = [c for c in wide_df.columns
                  if re.match(r'^CLASS-1-.+$', c)]

    # Build long form: one row per (file, page, class) per model
    rows = []
    for _, row in wide_df.iterrows():
        for mc in model_cols:
            cls = row[mc]
            if pd.notna(cls):
                rows.append({"FILE": row["FILE"], "PAGE": row["PAGE"], "CLASS": str(cls), "SCORE": 1.0})

    if not rows:
        return pd.DataFrame(columns=["FILE", "PAGE"] +
                            [f"CLASS-{i}" for i in range(1, top_N + 1)] +
                            [f"SCORE-{i}" for i in range(1, top_N + 1)])

    long_df = pd.DataFrame(rows)
    num_models = len(model_cols)

    grouped = (
        long_df.groupby(["FILE", "PAGE", "CLASS"])["SCORE"]
        .sum()
        .reset_index()
    )
    grouped["AVG_SCORE"] = (grouped["SCORE"] / num_models).clip(upper=1.0)
    grouped.sort_values(["FILE", "PAGE", "AVG_SCORE"], ascending=[True, True, False], inplace=True)
    grouped["rank"] = grouped.groupby(["FILE", "PAGE"]).cumcount() + 1
    top_n_df = grouped[grouped["rank"] <= top_N].copy()

    pivot = top_n_df.pivot_table(
        index=["FILE", "PAGE"],
        columns="rank",
        values=["CLASS", "AVG_SCORE"],
        aggfunc="first",
    )
    flat = pd.DataFrame(index=pivot.index)
    max_rank = int(top_n_df["rank"].max()) if not top_n_df.empty else 0
    for r in range(1, max_rank + 1):
        flat[f"CLASS-{r}"] = pivot.get(("CLASS", r), pd.NA)
        flat[f"SCORE-{r}"] = pivot.get(("AVG_SCORE", r), pd.NA)

    result = flat.reset_index()
    # Replace zeroes with NULL
    result = result.replace({0: "", 0.0: ""})
    result.sort_values(["FILE", "PAGE"], ascending=[True, True], inplace=True)
    return result


# ---------------------------------------------------------------------------
# GPU profile registry
# ---------------------------------------------------------------------------

_PROFILE_FILENAME = "gpu_profile.json"
_PROFILING_BATCHES = 2     # measure peak over first N batches
_MEMORY_SAFETY_MARGIN = 0.90   # use at most 90 % of free VRAM
_HEADROOM_BYTES = 512 * 1024 * 1024   # 512 MB hard margin


def _profile_path(model_dir: str) -> Path:
    return Path(model_dir) / _PROFILE_FILENAME


def _current_gpu_key() -> Optional[dict]:
    """Return a hardware fingerprint dict, or None if CUDA unavailable."""
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(0)
    return {
        "name": props.name,
        "total_vram_bytes": props.total_memory,
        "device_index": 0,
        "torch": torch.__version__,
    }


def _load_profile(model_dir: str, batch: int) -> Optional[dict]:
    """Load profile if it exists and matches the current hardware + batch."""
    p = _profile_path(model_dir)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
    except Exception:
        return None
    gpu_key = _current_gpu_key()
    if gpu_key is None:
        return None
    stored_gpu = data.get("gpu", {})
    if (stored_gpu.get("name") != gpu_key["name"]
            or stored_gpu.get("total_vram_bytes") != gpu_key["total_vram_bytes"]
            or data.get("batch") != batch):
        print("[parallel_best] Profile stale (hardware or batch changed) — will re-profile.")
        return None
    return data


def _save_profile(model_dir: str, batch: int, models_peak: dict) -> None:
    gpu_key = _current_gpu_key()
    data = {
        "gpu": gpu_key,
        "batch": batch,
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "models": models_peak,
    }
    p = _profile_path(model_dir)
    p.write_text(json.dumps(data, indent=2))
    print(f"[parallel_best] GPU profile saved → {p}")


def _measure_model_peak(
    rev: str,
    base_model: str,
    model_dir: str,
    cp_dir: str,
    sample_images: list,
    batch: int,
    categories: list,
) -> int:
    """Load model, run 2 batches, return peak VRAM in bytes."""
    from classifier import ImageClassifier
    dev = 0
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(dev)

    local_name = f"model_{rev.replace('.', '')}"
    local_path = Path(model_dir) / local_name

    clf = ImageClassifier(checkpoint=base_model, num_labels=len(categories), store_dir=str(cp_dir))
    clf.load_model(str(local_path))

    probe = sample_images[:_PROFILING_BATCHES * batch]
    loader = clf.create_dataloader(probe, batch)

    with torch.no_grad():
        for i, b in enumerate(loader):
            if b is None or (isinstance(b, tuple) and b[0] is None):
                continue
            clf.model(pixel_values=b["pixel_values"].to(clf.device))
            if i >= _PROFILING_BATCHES - 1:
                break

    peak = torch.cuda.max_memory_allocated(dev)
    del clf
    torch.cuda.empty_cache()
    return int(peak)


def profile_best_models(
    revision_best_models: dict,
    model_dir: str,
    cp_dir: str,
    sample_images: list,
    batch: int,
    categories: list,
    force: bool = False,
) -> Optional[dict]:
    """
    Return a models dict  {rev: {base_model, peak_bytes, …}}  for packing.
    Loads from the on-disk registry when valid; re-profiles otherwise.
    Returns None if CUDA unavailable.
    """
    if not torch.cuda.is_available():
        print("[parallel_best] No CUDA — profiling skipped.")
        return None

    if not force:
        profile = _load_profile(model_dir, batch)
        if profile is not None:
            print(f"[parallel_best] Loaded cached GPU profile from {_profile_path(model_dir)}")
            return profile["models"]

    print("[parallel_best] Profiling GPU memory for each best model …")
    models_peak: dict = {}
    for rev, base_model in revision_best_models.items():
        print(f"  profiling {rev} ({base_model}) …")
        try:
            peak = _measure_model_peak(
                rev, base_model, model_dir, cp_dir, sample_images, batch, categories
            )
            models_peak[rev] = {
                "base_model": base_model,
                "peak_bytes": peak,
            }
            props = torch.cuda.get_device_properties(0)
            print(f"  {rev}: peak {peak / 1e9:.2f} GB / {props.total_memory / 1e9:.2f} GB total")
        except Exception as e:
            print(f"  [WARNING] Could not profile {rev}: {e}")
            # Assign infinite cost so this model is always its own group
            models_peak[rev] = {
                "base_model": base_model,
                "peak_bytes": int(1e18),
            }

    _save_profile(model_dir, batch, models_peak)
    return models_peak


# ---------------------------------------------------------------------------
# Memory-aware packing scheduler
# ---------------------------------------------------------------------------

def pack_models(models_peak: dict, max_group: Optional[int] = None) -> List[List[str]]:
    """
    Greedy descending-size bin-packing into groups whose cumulative
    peak_bytes fits within the available VRAM budget.

    Returns a list of groups (each group is a list of revision strings).
    A model whose footprint alone exceeds the budget gets its own group
    (single-model sequential fallback within the parallel engine).
    """
    if not torch.cuda.is_available():
        return [[rev] for rev in models_peak]

    free, total = torch.cuda.mem_get_info(0)
    budget = int(free * _MEMORY_SAFETY_MARGIN) - _HEADROOM_BYTES
    print(f"[parallel_best] VRAM budget: {budget / 1e9:.2f} GB "
          f"(free={free / 1e9:.2f} GB, total={total / 1e9:.2f} GB)")

    # Sort descending by peak so the biggest models anchor their own group
    sorted_revs = sorted(models_peak, key=lambda r: models_peak[r]["peak_bytes"], reverse=True)

    groups: List[List[str]] = []
    current_group: List[str] = []
    current_cost = 0

    for rev in sorted_revs:
        cost = models_peak[rev]["peak_bytes"]
        fits_in_current = current_cost + cost <= budget
        within_cap = (max_group is None) or (len(current_group) < max_group)

        if current_group and (not fits_in_current or not within_cap):
            groups.append(current_group)
            current_group = []
            current_cost = 0

        current_group.append(rev)
        current_cost += cost

    if current_group:
        groups.append(current_group)

    print(f"[parallel_best] Packing plan: {len(groups)} group(s)")
    for i, g in enumerate(groups):
        costs = [models_peak[r]["peak_bytes"] / 1e9 for r in g]
        print(f"  group {i + 1}: {g}  cumulative={sum(costs):.2f} GB")
    return groups


# ---------------------------------------------------------------------------
# Grouped single-pass execution
# ---------------------------------------------------------------------------

def _run_group(
    group: List[str],
    revision_best_models: dict,
    test_images: list,
    categories: list,
    batch: int,
    model_dir: str,
    cp_dir: str,
    models_peak: Optional[dict],
) -> Dict[str, pd.DataFrame]:
    """
    Load all models in group simultaneously, stream test_images once,
    return {rev: rdf_with_FILE_PAGE_CLASS-1} for each model.

    Includes a 2-batch overflow guard: after the first PROFILING_BATCHES
    batches, if peak memory usage exceeds the safety threshold, drop the
    largest model from the group and defer it (the caller handles the
    deferred list via fallback).
    """
    from classifier import ImageClassifier
    from utils import dataframe_results

    classifiers: OrderedDict[str, ImageClassifier] = OrderedDict()
    for rev in group:
        base_model = revision_best_models[rev]
        local_name = f"model_{rev.replace('.', '')}"
        local_path = Path(model_dir) / local_name
        print(f"  [group] loading {rev} ({base_model}) …")
        clf = ImageClassifier(checkpoint=base_model, num_labels=len(categories), store_dir=str(cp_dir))
        clf.load_model(str(local_path))
        classifiers[rev] = clf

    # Each model has its own transform; we need per-model dataloaders because
    # different checkpoints have different resolutions / normalisations.
    # We share the I/O by pre-loading all images once and applying each
    # model's transform batch-wise.  For simplicity we reuse create_dataloader
    # per model (each decodes from disk — the real saving is load/unload churn).
    loaders = {
        rev: clf.create_dataloader(test_images, batch)
        for rev, clf in classifiers.items()
    }

    all_predictions: Dict[str, list] = {rev: [] for rev in group}

    # --- overflow guard state ---
    dev = 0
    dropped: List[str] = []
    guard_done = False

    for batch_idx in range(max(len(ld) for ld in loaders.values())):
        for rev, clf in list(classifiers.items()):
            loader = loaders[rev]
            try:
                b = next(iter(loader))  # noqa: we rebuilt iter each step below
            except StopIteration:
                continue

        # Rebuild iterators properly (simpler: zip over fixed-length loaders)
        break  # we exit this placeholder loop immediately

    # Proper multi-model single-pass loop using zip_longest
    from itertools import zip_longest
    loader_iters = {rev: iter(ld) for rev, ld in loaders.items()}

    for batch_idx, batches in enumerate(zip_longest(*[loader_iters[rev] for rev in classifiers])):
        for rev_i, (rev, b) in enumerate(zip(list(classifiers.keys()), batches)):
            if b is None or (isinstance(b, tuple) and b[0] is None):
                continue
            clf = classifiers[rev]
            with torch.no_grad():
                inputs = b["pixel_values"].to(clf.device)
                outputs = clf.model(pixel_values=inputs)
                logits = outputs.logits
                predicted = logits.argmax(-1).tolist()
                all_predictions[rev].extend(predicted)

        # Overflow guard after first PROFILING_BATCHES batches
        if torch.cuda.is_available() and not guard_done and batch_idx >= _PROFILING_BATCHES - 1:
            guard_done = True
            _, total = torch.cuda.mem_get_info(dev)
            live_peak = torch.cuda.max_memory_allocated(dev)
            danger_threshold = total * _MEMORY_SAFETY_MARGIN
            if live_peak > danger_threshold and len(classifiers) > 1:
                # Drop the largest (by profile) still-resident model
                if models_peak:
                    victim = max(
                        [r for r in classifiers if r not in dropped],
                        key=lambda r: models_peak.get(r, {}).get("peak_bytes", 0),
                    )
                else:
                    victim = list(classifiers.keys())[-1]
                print(f"  [overflow guard] Dropping {victim} from group — "
                      f"live peak {live_peak / 1e9:.2f} GB > threshold "
                      f"{danger_threshold / 1e9:.2f} GB")
                dropped.append(victim)
                del classifiers[victim]
                torch.cuda.empty_cache()

    # Build rdfs
    rdfs: Dict[str, pd.DataFrame] = {}
    for rev, preds in all_predictions.items():
        if not preds:
            continue
        rdf, _ = dataframe_results(test_images, preds, categories, top_N=1, raw_scores=None)
        rdf.drop(columns=["CATEGORY"], inplace=True, errors="ignore")
        rdfs[rev] = rdf

    # Clean up
    for clf in classifiers.values():
        del clf
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return rdfs, dropped


# ---------------------------------------------------------------------------
# Sequential execution (guaranteed fallback)
# ---------------------------------------------------------------------------

def run_best_sequential(
    test_images: list,
    categories: list,
    revision_best_models: dict,
    model_dir: str,
    cp_dir: str,
    batch: int,
    paradata_logger=None,
) -> Dict[str, pd.DataFrame]:
    """
    Original sequential loop: one model at a time, full data pass each.
    Returns {rev: rdf (FILE, PAGE, CLASS-1)}.
    """
    from classifier import ImageClassifier
    from utils import dataframe_results

    all_rdfs: Dict[str, pd.DataFrame] = {}
    for rev, base_model in revision_best_models.items():
        print(f"\n[sequential] Loading {rev} ({base_model}) …")
        local_name = f"model_{rev.replace('.', '')}"
        local_path = Path(model_dir) / local_name

        clf = ImageClassifier(checkpoint=base_model, num_labels=len(categories), store_dir=str(cp_dir))
        clf.load_model(str(local_path))
        loader = clf.create_dataloader(test_images, batch)
        preds, _ = clf.infer_dataloader(loader, top_n=1, raw=False)

        rdf, _ = dataframe_results(test_images, preds, categories, top_N=1, raw_scores=None)
        rdf.drop(columns=["CATEGORY"], inplace=True, errors="ignore")
        all_rdfs[rev] = rdf

        if paradata_logger is not None:
            paradata_logger.log_success("csv", len(rdf.index))

        del clf
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return all_rdfs


# ---------------------------------------------------------------------------
# Merge per-model rdfs → wide intermediate CSV
# ---------------------------------------------------------------------------

def merge_to_wide(all_rdfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Combine per-model rdfs into the wide format:
      FILE, PAGE, CLASS-1-v1.3, CLASS-1-v2.3, …
    This is byte-identical to what the original --best loop produced.
    """
    combined = pd.DataFrame()
    for rev, rdf in all_rdfs.items():
        renamed = rdf.rename(columns={"CLASS-1": f"CLASS-1-{rev}"})
        if combined.empty:
            combined = renamed
        else:
            combined = pd.merge(combined, renamed, on=["FILE", "PAGE"], how="outer")
    return combined


# ---------------------------------------------------------------------------
# Main entry point used by run.py
# ---------------------------------------------------------------------------

def run_best_models(
    test_images: list,
    categories: list,
    revision_best_models: dict,
    model_dir: str,
    cp_dir: str,
    batch: int,
    top_N: int,
    output_dir: str,
    time_stamp: str,
    paradata_logger=None,
    parallel: bool = False,
    save_intermediate: bool = True,
) -> str:
    """
    Run all best models and return the path to the final averaged CSV.

    Parameters
    ----------
    parallel : bool
        When True, attempt memory-aware grouped single-pass execution.
        Falls back silently to sequential when CUDA is unavailable or
        profiling fails.
    save_intermediate : bool
        Whether to also save the wide per-model CSV (BEST_N_models_TOP-1).
        Default True so existing workflows that consume that file still work.

    Returns
    -------
    str  Path to the averaged output CSV.
    """
    out_tables = Path(output_dir) / "tables"
    out_tables.mkdir(parents=True, exist_ok=True)

    n_models = len(revision_best_models)
    all_rdfs: Dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # Choose execution engine
    # ------------------------------------------------------------------
    use_parallel = parallel and torch.cuda.is_available()

    if use_parallel:
        print(f"\n[parallel_best] Parallel mode — profiling {n_models} models …")
        # Need at least 2*batch images to profile; fall back to sequential if dataset tiny
        if len(test_images) < _PROFILING_BATCHES * batch:
            print("[parallel_best] Dataset too small for profiling — falling back to sequential.")
            use_parallel = False

    if use_parallel:
        try:
            models_peak = profile_best_models(
                revision_best_models, model_dir, cp_dir, test_images, batch, categories
            )
            if models_peak is None:
                use_parallel = False
        except Exception as e:
            print(f"[parallel_best] Profiling failed ({e}) — falling back to sequential.")
            use_parallel = False

    if use_parallel:
        groups = pack_models(models_peak)
        deferred: List[str] = []  # models dropped by overflow guard

        for g_idx, group in enumerate(groups):
            print(f"\n[parallel_best] Running group {g_idx + 1}/{len(groups)}: {group}")
            try:
                rdfs, dropped = _run_group(
                    group, revision_best_models, test_images,
                    categories, batch, model_dir, cp_dir, models_peak,
                )
                all_rdfs.update(rdfs)
                deferred.extend(dropped)
                if paradata_logger is not None:
                    for rdf in rdfs.values():
                        paradata_logger.log_success("csv", len(rdf.index))
            except Exception as e:
                print(f"[parallel_best] Group {g_idx + 1} failed ({e}) — "
                      f"running its models sequentially.")
                deferred.extend(group)

        # Run deferred (overflow victims) sequentially
        if deferred:
            print(f"\n[parallel_best] Running {len(deferred)} deferred model(s) sequentially …")
            deferred_map = {r: revision_best_models[r] for r in deferred if r in revision_best_models}
            deferred_rdfs = run_best_sequential(
                test_images, categories, deferred_map,
                model_dir, cp_dir, batch, paradata_logger,
            )
            all_rdfs.update(deferred_rdfs)
    else:
        if parallel and not torch.cuda.is_available():
            print("[parallel_best] --parallel requested but no CUDA — running sequentially.")
        all_rdfs = run_best_sequential(
            test_images, categories, revision_best_models,
            model_dir, cp_dir, batch, paradata_logger,
        )

    # ------------------------------------------------------------------
    # Save wide intermediate CSV (optional but default-on)
    # ------------------------------------------------------------------
    wide_df = merge_to_wide(all_rdfs)
    wide_df.sort_values(["FILE", "PAGE"], ascending=[True, True], inplace=True)

    wide_path = None
    if save_intermediate:
        wide_path = str(out_tables / f"{time_stamp}_BEST_{n_models}_models_TOP-1.csv")
        wide_df.to_csv(wide_path, index=False)
        print(f"[parallel_best] Wide intermediate CSV → {wide_path}")
        if paradata_logger is not None:
            paradata_logger.log_success("csv")

    # ------------------------------------------------------------------
    # Average and write the final result CSV
    # ------------------------------------------------------------------
    print(f"\n[parallel_best] Averaging {n_models} models → TOP-{top_N} result …")
    avg_df = _average_wide_df(wide_df, categories, top_N)
    avg_df.sort_values(["FILE", "PAGE"], ascending=[True, True], inplace=True)

    engine_tag = "parallel" if (parallel and torch.cuda.is_available()) else "best"
    avg_path = str(out_tables / f"{time_stamp}_{engine_tag}_{n_models}_models_AVG_TOP-{top_N}.csv")
    avg_df.to_csv(avg_path, index=False)
    print(f"[parallel_best] Averaged result CSV → {avg_path}")
    if paradata_logger is not None:
        paradata_logger.log_success("csv")

    return avg_path