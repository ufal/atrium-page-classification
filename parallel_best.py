"""
parallel_best.py  –  Memory-aware parallel execution of the --best model ensemble.

Implements the design from Issue #3:
  * Per-model GPU-size registry  (model/gpu_profile.json)
  * On-demand 2-batch profiling
  * Memory-aware packing scheduler
  * Grouped single-pass execution + runtime overflow guard
  * True Top-N in-memory probability averaging → final averaged CSV
  * Sequential fallback whenever parallel is unsafe

The output of run_best_models is always a fully averaged CSV
(CLASS-1, SCORE-1 … SCORE-N per page).  Intermediate per-model CSVs
are written only when --save-intermediates is passed.

Public surface used by run.py
------------------------------
  run_best_models(
      test_images, categories, revision_best_models,
      model_dir, cp_dir, batch, top_N,
      output_dir, time_stamp, paradata_logger,
      parallel=False,
      save_intermediates=False,
      average_best=True,
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
# GPU profile registry
# ---------------------------------------------------------------------------

_PROFILE_FILENAME = "gpu_profile.json"
_PROFILING_BATCHES = 2          # measure peak over first N batches
_MEMORY_SAFETY_MARGIN = 0.90    # use at most 90 % of free VRAM
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
    top_N: int,
) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """
    Load all models in group simultaneously, stream test_images once,
    return {rev: rdf} containing Top-N classes and raw probability scores.

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

    # Each model has its own transform / resolution; per-model dataloaders
    # are required.  The real saving is avoiding repeated load/unload churn.
    loaders = {
        rev: clf.create_dataloader(test_images, batch)
        for rev, clf in classifiers.items()
    }

    all_predictions: Dict[str, list] = {rev: [] for rev in group}
    all_raw_scores: Dict[str, list] = {rev: [] for rev in group}

    dev = 0
    dropped: List[str] = []
    guard_done = False

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
                probs = torch.nn.functional.softmax(logits, dim=-1)

                # Store Top-N indices and full softmax probabilities
                topk = torch.topk(probs, top_N, dim=-1)
                all_predictions[rev].extend(topk.indices.cpu().tolist())
                all_raw_scores[rev].extend(probs.cpu().tolist())

        # Overflow guard after first PROFILING_BATCHES batches
        if torch.cuda.is_available() and not guard_done and batch_idx >= _PROFILING_BATCHES - 1:
            guard_done = True
            _, total = torch.cuda.mem_get_info(dev)
            live_peak = torch.cuda.max_memory_allocated(dev)
            danger_threshold = total * _MEMORY_SAFETY_MARGIN
            if live_peak > danger_threshold and len(classifiers) > 1:
                if models_peak:
                    victim = max(
                        [r for r in classifiers if r not in dropped],
                        key=lambda r: models_peak.get(r, {}).get("peak_bytes", 0),
                    )
                else:
                    victim = list(classifiers.keys())[-1]
                print(f"  [overflow guard] Dropping {victim} from group — "
                      f"live peak {live_peak / 1e9:.2f} GB > "
                      f"{danger_threshold / 1e9:.2f} GB")
                dropped.append(victim)
                del classifiers[victim]
                torch.cuda.empty_cache()

    # Build Top-N rdfs with actual softmax scores
    rdfs: Dict[str, pd.DataFrame] = {}
    for rev in all_predictions.keys():
        preds = all_predictions[rev]
        raws = all_raw_scores[rev]
        if not preds:
            continue
        rdf, _ = dataframe_results(test_images, preds, categories, top_N=top_N, raw_scores=raws)
        rdf.drop(columns=["CATEGORY"], inplace=True, errors="ignore")
        rdfs[rev] = rdf

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
    top_N: int,
    paradata_logger=None,
) -> Dict[str, pd.DataFrame]:
    """
    Original sequential loop: one model at a time, full data pass each.
    Captures full Top-N softmax probabilities for downstream averaging.
    Returns {rev: rdf (FILE, PAGE, CLASS-1 … CLASS-N, SCORE-1 … SCORE-N)}.
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

        # raw=True returns full softmax vector per image for probability averaging
        preds, raw_scores = clf.infer_dataloader(loader, top_n=top_N, raw=True)

        rdf, _ = dataframe_results(test_images, preds, categories, top_N=top_N, raw_scores=raw_scores)
        rdf.drop(columns=["CATEGORY"], inplace=True, errors="ignore")
        all_rdfs[rev] = rdf

        if paradata_logger is not None:
            paradata_logger.log_success("csv", len(rdf.index))

        del clf
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return all_rdfs


# ---------------------------------------------------------------------------
# In-Memory Probability Averaging
# ---------------------------------------------------------------------------

def average_rdfs(all_rdfs: Dict[str, pd.DataFrame], top_N: int) -> pd.DataFrame:
    """
    Aggregates per-model DataFrames by computing the mean softmax probability
    for each class across all models, then re-ranking to yield a Top-N DataFrame.

    This replaces the legacy wide-CSV majority-vote approach: instead of each
    model casting a binary vote, its full probability distribution is averaged,
    so confident models carry more weight than uncertain ones.
    """
    long_dfs = []
    num_models = len(all_rdfs)

    # 1. Melt each per-model DataFrame into long form (FILE, PAGE, CLASS, SCORE)
    for rev, rdf in all_rdfs.items():
        class_cols = [c for c in rdf.columns if str(c).startswith('CLASS-')]
        indices = [int(c.split('-')[1]) for c in class_cols if c.split('-')[1].isdigit()]

        if not indices:
            continue

        cols_to_keep = (
            ['FILE', 'PAGE']
            + [f'CLASS-{i}' for i in indices]
            + [f'SCORE-{i}' for i in indices if f'SCORE-{i}' in rdf.columns]
        )
        cols_to_keep = [c for c in cols_to_keep if c in rdf.columns]
        df_subset = rdf[cols_to_keep].copy()

        melted = pd.wide_to_long(
            df_subset,
            stubnames=['CLASS', 'SCORE'],
            i=['FILE', 'PAGE'],
            j='rank',
            sep='-',
            suffix=r'\d+',
        ).reset_index().dropna(subset=['CLASS'])

        # Keep the highest score seen for each (file, page, class) within this model
        melted = melted.groupby(['FILE', 'PAGE', 'CLASS'], as_index=False)['SCORE'].max()
        long_dfs.append(melted)

    if not long_dfs:
        return pd.DataFrame(
            columns=["FILE", "PAGE"]
            + [f"CLASS-{i}" for i in range(1, top_N + 1)]
            + [f"SCORE-{i}" for i in range(1, top_N + 1)]
        )

    combined = pd.concat(long_dfs, ignore_index=True)

    # 2. Average the raw probabilities across all models
    grouped = combined.groupby(['FILE', 'PAGE', 'CLASS'])['SCORE'].sum().reset_index()
    grouped['AVG_SCORE'] = (grouped['SCORE'] / num_models).clip(upper=1.0)

    # 3. Re-rank Top-N by averaged score
    grouped.sort_values(['FILE', 'PAGE', 'AVG_SCORE'], ascending=[True, True, False], inplace=True)
    grouped['rank'] = grouped.groupby(['FILE', 'PAGE']).cumcount() + 1
    top_n_df = grouped[grouped['rank'] <= top_N].copy()

    # 4. Pivot back to standard CLASS-N / SCORE-N column format
    pivot = top_n_df.pivot_table(
        index=['FILE', 'PAGE'],
        columns='rank',
        values=['CLASS', 'AVG_SCORE'],
        aggfunc='first',
    )

    flat = pd.DataFrame(index=pivot.index)
    max_rank = int(top_n_df['rank'].max()) if not top_n_df.empty else 0
    for r in range(1, max_rank + 1):
        flat[f'CLASS-{r}'] = pivot.get(('CLASS', r), pd.NA)
        flat[f'SCORE-{r}'] = pivot.get(('AVG_SCORE', r), pd.NA)

    result = flat.reset_index()
    result = result.replace({0: "", 0.0: ""})
    result.sort_values(['FILE', 'PAGE'], ascending=[True, True], inplace=True)
    return result


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
    save_intermediates: bool = False,
    average_best: bool = True,
) -> str:
    """
    Run all best models and return the path to the final averaged CSV.

    Parameters
    ----------
    parallel : bool
        When True, attempt memory-aware grouped single-pass execution.
        Falls back silently to sequential when CUDA is unavailable or
        profiling fails.
    save_intermediates : bool
        When True, write one standard Top-N CSV per model before averaging.
        Off by default; individual CSVs are natively compatible with
        averaging.py for manual re-averaging if needed.
    average_best : bool
        When False (--no-average-best), skip the averaging step and return
        an empty string.  Intermediate CSVs are still written when
        save_intermediates is True.

    Returns
    -------
    str  Path to the averaged output CSV, or "" when average_best is False.
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
        # Need at least 2*batch images to profile; fall back if dataset tiny
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
        deferred: List[str] = []

        for g_idx, group in enumerate(groups):
            print(f"\n[parallel_best] Running group {g_idx + 1}/{len(groups)}: {group}")
            try:
                rdfs, dropped = _run_group(
                    group, revision_best_models, test_images,
                    categories, batch, model_dir, cp_dir, models_peak, top_N,
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

        # Run overflow victims sequentially
        if deferred:
            print(f"\n[parallel_best] Running {len(deferred)} deferred model(s) sequentially …")
            deferred_map = {r: revision_best_models[r] for r in deferred if r in revision_best_models}
            deferred_rdfs = run_best_sequential(
                test_images, categories, deferred_map,
                model_dir, cp_dir, batch, top_N, paradata_logger,
            )
            all_rdfs.update(deferred_rdfs)
    else:
        if parallel and not torch.cuda.is_available():
            print("[parallel_best] --parallel requested but no CUDA — running sequentially.")
        all_rdfs = run_best_sequential(
            test_images, categories, revision_best_models,
            model_dir, cp_dir, batch, top_N, paradata_logger,
        )

    # ------------------------------------------------------------------
    # Optional: save individual per-model CSVs before averaging
    # ------------------------------------------------------------------
    if save_intermediates:
        for rev, rdf in all_rdfs.items():
            path = out_tables / f"{time_stamp}_{rev.replace('.', '')}_TOP-{top_N}.csv"
            rdf.to_csv(path, index=False)
            print(f"[parallel_best] Saved intermediate model CSV → {path}")

    # ------------------------------------------------------------------
    # Average and write the final result CSV
    # ------------------------------------------------------------------
    if not average_best:
        print("\n[parallel_best] Averaging bypassed (--no-average-best).")
        return ""

    print(f"\n[parallel_best] Averaging {n_models} models → TOP-{top_N} result …")
    avg_df = average_rdfs(all_rdfs, top_N)

    engine_tag = "parallel" if (parallel and torch.cuda.is_available()) else "best"
    avg_path = str(out_tables / f"{time_stamp}_{engine_tag}_{n_models}_models_AVG_TOP-{top_N}.csv")
    avg_df.to_csv(avg_path, index=False)
    print(f"[parallel_best] Averaged result CSV → {avg_path}")

    if paradata_logger is not None:
        paradata_logger.log_success("csv")

    return avg_path