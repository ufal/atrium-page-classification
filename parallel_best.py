from __future__ import annotations

import json
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import torch

from model_registry import MODEL_STATIC
from ensemble import average_rdfs

# ---------------------------------------------------------------------------
# GPU profile registry
# ---------------------------------------------------------------------------
_PROFILE_FILENAME = "gpu_profile.json"
_PROFILING_BATCHES = 2
_MEMORY_SAFETY_MARGIN = 0.90
_HEADROOM_BYTES = 512 * 1024 * 1024

def _profile_path(model_dir: str) -> Path:
    return Path(model_dir) / _PROFILE_FILENAME

# ═══════════════════════════════════════════════════════════════════════════
# Memory-aware packing scheduler
# ═══════════════════════════════════════════════════════════════════════════
def pack_models(
    sizes: Dict[str, int],
    budget_bytes: int,
    max_group: Optional[int] = None,
) -> List[List[str]]:
    sorted_revs = sorted(sizes, key=lambda r: sizes[r], reverse=True)

    if budget_bytes <= 0:
        return [[rev] for rev in sorted_revs]

    groups: List[List[str]] = []
    current: List[str] = []
    current_cost = 0

    for rev in sorted_revs:
        cost = sizes[rev]
        fits = (current_cost + cost) <= budget_bytes
        within_cap = (max_group is None) or (len(current) < max_group)

        if current and (not fits or not within_cap):
            groups.append(current)
            current = []
            current_cost = 0

        current.append(rev)
        current_cost += cost

    if current:
        groups.append(current)

    return groups


# ═══════════════════════════════════════════════════════════════════════════
# Registry freshness
# ═══════════════════════════════════════════════════════════════════════════
def registry_is_fresh(
    profile: Optional[dict],
    gpu_name: str,
    total_vram_bytes: int,
    batch: int,
    required_revs: Iterable[str],
) -> bool:
    if not profile:
        return False
    gpu = profile.get("gpu", {})
    if gpu.get("name") != gpu_name:
        return False
    if gpu.get("total_vram_bytes") != total_vram_bytes:
        return False
    if profile.get("batch") != batch:
        return False

    recorded = set(profile.get("models", {}).keys())
    if not set(required_revs).issubset(recorded):
        return False
    return True

def _current_gpu_key() -> Optional[dict]:
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(0)
    return {
        "name": props.name,
        "total_vram_bytes": props.total_memory,
        "device_index": 0,
        "torch": torch.__version__,
    }

def _load_profile(
    model_dir: str,
    batch: int,
    required_revs: Iterable[str],
) -> Optional[dict]:
    p = _profile_path(model_dir)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
    except Exception:
        return None

    gpu = _current_gpu_key()
    if gpu is None:
        return None

    if not registry_is_fresh(data, gpu["name"], gpu["total_vram_bytes"], batch, required_revs):
        print("[parallel_best] Profile stale (hardware / batch / coverage) — will re-profile.")
        return None

    return data["models"]

def _save_profile(model_dir: str, batch: int, models_peak: dict) -> None:
    data = {
        "gpu": _current_gpu_key(),
        "batch": batch,
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "models": models_peak,
    }
    p = _profile_path(model_dir)
    p.write_text(json.dumps(data, indent=2))
    print(f"[parallel_best] GPU profile saved → {p}")


# ═══════════════════════════════════════════════════════════════════════════
# On-demand profiling
# ═══════════════════════════════════════════════════════════════════════════
def _measure_model_peak(
    rev: str,
    base_model: str,
    model_dir: str,
    cp_dir: str,
    sample_images: list,
    batch: int,
    categories: list,
) -> int:
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
    if not torch.cuda.is_available():
        print("[parallel_best] No CUDA — profiling skipped.")
        return None

    required_revs = list(revision_best_models.keys())

    if not force:
        models = _load_profile(model_dir, batch, required_revs)
        if models is not None:
            print(f"[parallel_best] Loaded cached GPU profile from {_profile_path(model_dir)}")
            return models

    print("[parallel_best] Profiling GPU memory for each best model …")
    models_peak: dict = {}
    for rev, base_model in revision_best_models.items():
        print(f"  profiling {rev} ({base_model}) …")
        try:
            peak = _measure_model_peak(
                rev, base_model, model_dir, cp_dir, sample_images, batch, categories
            )
            models_peak[rev] = {"base_model": base_model, "peak_bytes": peak}
            props = torch.cuda.get_device_properties(0)
            print(f"  {rev}: peak {peak / 1e9:.2f} GB / {props.total_memory / 1e9:.2f} GB total")
        except Exception as e:
            print(f"  [WARNING] Could not profile {rev}: {e}")
            models_peak[rev] = {"base_model": base_model, "peak_bytes": int(1e18)}

    _save_profile(model_dir, batch, models_peak)
    return models_peak


# ═══════════════════════════════════════════════════════════════════════════
# Grouped single-pass execution
# ═══════════════════════════════════════════════════════════════════════════
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
    from classifier import ImageClassifier
    from utils import dataframe_results

    classifiers: "OrderedDict[str, ImageClassifier]" = OrderedDict()
    for rev in group:
        base_model = revision_best_models[rev]
        local_name = f"model_{rev.replace('.', '')}"
        local_path = Path(model_dir) / local_name
        print(f"  [group] loading {rev} ({base_model}) …")
        clf = ImageClassifier(checkpoint=base_model, num_labels=len(categories), store_dir=str(cp_dir))
        clf.load_model(str(local_path))
        classifiers[rev] = clf

    loaders = {rev: clf.create_dataloader(test_images, batch) for rev, clf in classifiers.items()}

    all_predictions: Dict[str, list] = {rev: [] for rev in group}
    all_raw_scores: Dict[str, list] = {rev: [] for rev in group}

    dev = 0
    dropped: List[str] = []
    guard_done = False

    from itertools import zip_longest
    loader_iters = {rev: iter(ld) for rev, ld in loaders.items()}

    for batch_idx, batches in enumerate(zip_longest(*[loader_iters[rev] for rev in classifiers])):
        for rev, b in zip(list(classifiers.keys()), batches):
            if b is None or (isinstance(b, tuple) and b[0] is None):
                continue
            clf = classifiers[rev]
            with torch.no_grad():
                inputs = b["pixel_values"].to(clf.device)
                outputs = clf.model(pixel_values=inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

                all_raw_scores[rev].extend(probs.cpu().tolist())
                if top_N > 1:
                    topk_probs, topk_idx = torch.topk(probs, top_N, dim=-1)
                    for idxs, prs in zip(topk_idx, topk_probs):
                        prs_norm = prs / prs.sum()
                        all_predictions[rev].append(
                            list(zip(idxs.cpu().tolist(), prs_norm.cpu().tolist()))
                        )
                else:
                    all_predictions[rev].extend(probs.argmax(dim=-1).cpu().tolist())

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
                      f"live peak {live_peak / 1e9:.2f} GB > {danger_threshold / 1e9:.2f} GB")
                dropped.append(victim)
                del classifiers[victim]
                torch.cuda.empty_cache()

    rdfs: Dict[str, pd.DataFrame] = {}
    for rev in list(all_predictions.keys()):
        preds = all_predictions[rev]
        raws = all_raw_scores[rev]
        if not preds or rev in dropped:
            continue
        rdf, _ = dataframe_results(test_images, preds, categories, top_N=top_N, raw_scores=raws)
        rdf.drop(columns=["CATEGORY"], inplace=True, errors="ignore")
        rdfs[rev] = rdf

    for clf in classifiers.values():
        del clf
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return rdfs, dropped


# ═══════════════════════════════════════════════════════════════════════════
# Sequential execution (guaranteed fallback)
# ═══════════════════════════════════════════════════════════════════════════
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


# ═══════════════════════════════════════════════════════════════════════════
# Wide combined frame  (BEST_{N}_models_TOP-1.csv)
# ═══════════════════════════════════════════════════════════════════════════
def merge_best(
    revision_best_models: dict,
    rdf_by_rev: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    combined = pd.DataFrame()
    for rev in revision_best_models:
        if rev not in rdf_by_rev:
            continue
        rdf = rdf_by_rev[rev]
        renamed = {c: f"{c}-{rev}" for c in rdf.columns if c not in ("FILE", "PAGE")}
        rdf_renamed = rdf.rename(columns=renamed)
        if combined.empty:
            combined = rdf_renamed
        else:
            combined = pd.merge(combined, rdf_renamed, on=["FILE", "PAGE"], how="outer")
    return combined


# ═══════════════════════════════════════════════════════════════════════════
# Main entry point used by run.py
# ═══════════════════════════════════════════════════════════════════════════
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
    out_tables = Path(output_dir) / "tables"
    out_tables.mkdir(parents=True, exist_ok=True)

    n_models = len(revision_best_models)
    all_rdfs: Dict[str, pd.DataFrame] = {}

    use_parallel = parallel and torch.cuda.is_available()
    if use_parallel and len(test_images) < _PROFILING_BATCHES * batch:
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
        free, total = torch.cuda.mem_get_info(0)
        budget = int(free * _MEMORY_SAFETY_MARGIN) - _HEADROOM_BYTES
        sizes = {rev: info["peak_bytes"] for rev, info in models_peak.items()}
        groups = pack_models(sizes, budget)

        print(f"[parallel_best] VRAM budget {budget / 1e9:.2f} GB → {len(groups)} group(s)")
        for i, g in enumerate(groups):
            print(f"  group {i + 1}: {g}  "
                  f"cumulative={sum(sizes[r] for r in g) / 1e9:.2f} GB")

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
                print(f"[parallel_best] Group {g_idx + 1} failed ({e}) — running sequentially.")
                deferred.extend(group)

        if deferred:
            print(f"\n[parallel_best] Running {len(deferred)} deferred model(s) sequentially …")
            deferred_map = {r: revision_best_models[r] for r in deferred if r in revision_best_models}
            all_rdfs.update(run_best_sequential(
                test_images, categories, deferred_map,
                model_dir, cp_dir, batch, top_N, paradata_logger,
            ))
    else:
        if parallel and not torch.cuda.is_available():
            print("[parallel_best] --parallel requested but no CUDA — running sequentially.")
        all_rdfs = run_best_sequential(
            test_images, categories, revision_best_models,
            model_dir, cp_dir, batch, top_N, paradata_logger,
        )

    top1_rdfs = {
        rev: rdf[["FILE", "PAGE", "CLASS-1"]].copy()
        for rev, rdf in all_rdfs.items()
        if "CLASS-1" in rdf.columns
    }
    wide_df = merge_best(revision_best_models, top1_rdfs)
    wide_df.sort_values(["FILE", "PAGE"], ascending=[True, True], inplace=True)
    wide_path = str(out_tables / f"{time_stamp}_BEST_{n_models}_models_TOP-1.csv")
    wide_df.to_csv(wide_path, index=False)
    print(f"[parallel_best] Wide per-model votes → {wide_path}")

    if save_intermediates:
        for rev, rdf in all_rdfs.items():
            path = out_tables / f"{time_stamp}_{rev.replace('.', '')}_TOP-{top_N}.csv"
            rdf.to_csv(path, index=False)
            print(f"[parallel_best] Saved intermediate model CSV → {path}")

    if not average_best:
        print("\n[parallel_best] Averaging bypassed (--no-average-best). "
              f"Wide votes available at {wide_path}")
        return ""

    print(f"\n[parallel_best] Averaging {n_models} models → TOP-{top_N} result …")
    avg_df = average_rdfs(all_rdfs, top_N, revision_best_models)

    avg_path = str(out_tables / f"{time_stamp}_BEST_{n_models}_models_AVG_TOP-{top_N}.csv")
    avg_df.to_csv(avg_path, index=False)
    print(f"[parallel_best] Averaged result CSV → {avg_path}")

    if paradata_logger is not None:
        paradata_logger.log_success("csv")

    return avg_path