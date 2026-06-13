"""
parallel_best.py  –  Memory-aware execution of the --best model ensemble.

Implements the design from Issue #3:
  * Per-model GPU-size registry  (model/gpu_profile.json)
  * On-demand 2-batch profiling
  * Memory-aware packing scheduler (pack_models)
  * Registry freshness check (registry_is_fresh)
  * Grouped single-pass execution + runtime overflow guard
  * Wide combined frame builder (merge_best) -> BEST_{N}_models_TOP-1.csv
  * True Top-N in-memory probability averaging -> final averaged CSV
  * Sequential fallback whenever parallel is unsafe

Output contract
---------------
`run_best_models` writes (and the engine therefore guarantees):

  1. `{ts}_BEST_{n}_models_TOP-1.csv`     (always)
     Wide per-model votes: FILE, PAGE, CLASS-1-v1.3, CLASS-1-v2.3, …
     Byte-compatible with supplement_scripts/averaging.py's _WIDE_MODEL_RE.

  2. `{ts}_BEST_{n}_models_AVG_TOP-{N}.csv`  (when average_best, the default)
     The same column set as ARUP_averaged.csv:
        FILE, PAGE, V1.3, V2.3, V3.3, V4.3, V5.3,
        CLASS-1, SCORE-1, CLASS-2, SCORE-2, CLASS-3, SCORE-3
     i.e. per-model top-1 vote columns followed by the averaged Top-N.

Public surface used by run.py is unchanged:
    run_best_models(...) -> str   # path to the averaged CSV (or "" if skipped)
"""
from __future__ import annotations

import json
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import torch

# ---------------------------------------------------------------------------
# GPU profile registry
# ---------------------------------------------------------------------------

_PROFILE_FILENAME = "gpu_profile.json"
_PROFILING_BATCHES = 2          # measure peak over first N batches
_MEMORY_SAFETY_MARGIN = 0.90    # use at most 90 % of free VRAM
_HEADROOM_BYTES = 512 * 1024 * 1024   # 512 MB hard margin

# Hardware/torch/batch-independent model facts (fp32 weights, params only).
# Safe to commit; bootstraps packing before any GPU profile exists.
MODEL_STATIC = {
    "v1.3": {"base_model": "timm/tf_efficientnetv2_m.in21k_ft_in1k", "resolution": 384, "params_bytes": 211489788},
    "v2.3": {"base_model": "google/vit-base-patch16-224",            "resolution": 224, "params_bytes": 343228460},
    "v3.3": {"base_model": "google/vit-base-patch16-384",            "resolution": 384, "params_bytes": 344395820},
    "v4.3": {"base_model": "timm/regnety_160.swag_ft_in1k",          "resolution": 384, "params_bytes": 322393660},
    "v5.3": {"base_model": "google/vit-large-patch16-384",           "resolution": 384, "params_bytes": 1214808108},
}


def _profile_path(model_dir: str) -> Path:
    return Path(model_dir) / _PROFILE_FILENAME


# ---------------------------------------------------------------------------
# Helper: per-model top-1 vote column name  (v1.3 -> V1.3)
# ---------------------------------------------------------------------------

def _vote_col_name(rev: str) -> str:
    """Map a revision string to its ARUP-style vote column header.

    'v1.3' -> 'V1.3'  (uppercase the leading 'v'); matches averaging.py's
    display-name convention so the engine output and the standalone post-
    processor stay column-compatible.
    """
    return (rev[0].upper() + rev[1:]) if rev else rev


# ═══════════════════════════════════════════════════════════════════════════
# Memory-aware packing scheduler
# ═══════════════════════════════════════════════════════════════════════════

def pack_models(
    sizes: Dict[str, int],
    budget_bytes: int,
    max_group: Optional[int] = None,
) -> List[List[str]]:
    """Greedy first-fit-decreasing bin-packing of models into VRAM groups.

    Args:
        sizes:        {revision: peak_bytes} footprint per model.
        budget_bytes: VRAM budget a single group may not exceed.
        max_group:    Optional cap on how many models share one group
                      (e.g. 2 reproduces the issue's "two at a time").

    Returns:
        List of groups; each group is a list of revision strings.

    Behaviour:
        * Largest model anchors the first group (FFD).
        * A model whose footprint alone exceeds the budget gets its own group.
        * budget_bytes <= 0 degrades to one model per group (safe fallback).
        * Deterministic for a given input.
    """
    sorted_revs = sorted(sizes, key=lambda r: sizes[r], reverse=True)

    # Non-positive budget => cannot co-resident anything safely.
    if budget_bytes <= 0:
        return [[rev] for rev in sorted_revs]

    groups: List[List[str]] = []
    current: List[str] = []
    current_cost = 0

    for rev in sorted_revs:
        cost = sizes[rev]
        fits = (current_cost + cost) <= budget_bytes
        within_cap = (max_group is None) or (len(current) < max_group)

        # Close the current group when the next model would overflow either
        # the byte budget or the explicit member cap.
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
    """Return True iff a stored gpu_profile.json is reusable for this run.

    A profile is reusable only when the recorded hardware fingerprint and
    batch size match the current run AND it covers every required revision.
    A profile that covers MORE models than required is still fresh.
    """
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
    """Hardware fingerprint for the active CUDA device, or None on CPU/MPS."""
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
    """Load gpu_profile.json and return its 'models' block when still fresh."""
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
    """Return {rev: {base_model, peak_bytes}} for packing, or None on CPU/MPS.

    Loads from the on-disk registry when it is fresh AND covers every required
    revision; re-profiles otherwise.
    """
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
            # Infinite cost => this model always becomes its own group.
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
    """Load all models in a group, stream test_images once, return {rev: rdf}.

    Each rdf carries FILE, PAGE, CLASS-1…CLASS-N, SCORE-1…SCORE-N.

    Includes a 2-batch overflow guard: after the first _PROFILING_BATCHES
    batches, if live peak memory exceeds the safety threshold the largest
    model is dropped from the group and deferred to sequential fallback.
    """
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

    # Each model has its own resolution / processor, so per-model dataloaders
    # are required; the win is avoiding load/unload churn, not decode sharing.
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

                # FIX: build predictions in the SAME shape infer_dataloader
                # produces so dataframe_results reads (idx, score) correctly.
                # The previous code stored bare top-k indices, which made
                # dataframe_results interpret an index as the score.
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

        # Overflow guard after the first profiling-window batches.
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

    # Build per-model rdfs.  Dropped models produce no usable predictions.
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
    """One model at a time, full data pass each.  Returns {rev: rdf}."""
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
    """Outer-merge per-model frames into the wide CLASS-1-{rev} layout.

    Column order follows `revision_best_models` (canonical) regardless of the
    insertion order of `rdf_by_rev` (groups may finish out of order).  Every
    non-key column is suffixed with `-{rev}`, reproducing the historical
    inline merge byte-for-byte so averaging.py's `^CLASS-(\\d+)-(.+)$`
    parsing keeps working.
    """
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
# In-memory probability averaging  (final ARUP-style output)
# ═══════════════════════════════════════════════════════════════════════════

def average_rdfs(
    all_rdfs: Dict[str, pd.DataFrame],
    top_N: int,
    revision_best_models: Optional[dict] = None,
) -> pd.DataFrame:
    """Average per-model softmax probabilities and re-rank to Top-N.

    The result matches ARUP_averaged.csv's column set:
        FILE, PAGE, V1.3 … V5.3, CLASS-1, SCORE-1, … CLASS-N, SCORE-N
    where V-columns are each model's top-1 vote (in canonical revision order)
    and CLASS-/SCORE- columns are the probability-averaged ensemble.
    """
    long_dfs = []
    num_models = len(all_rdfs)

    for rdf in all_rdfs.values():
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

        # A Top-1 rdf has no SCORE column (utils.py drops SCORE-1); treat the
        # single vote as score 1.0 so it still contributes to the average.
        if not any(c.startswith('SCORE-') for c in df_subset.columns):
            melted = (
                df_subset.rename(columns={'CLASS-1': 'CLASS'})
                .assign(SCORE=1.0)
                .dropna(subset=['CLASS'])
            )[['FILE', 'PAGE', 'CLASS', 'SCORE']]
        else:
            melted = pd.wide_to_long(
                df_subset, stubnames=['CLASS', 'SCORE'],
                i=['FILE', 'PAGE'], j='rank', sep='-', suffix=r'\d+',
            ).reset_index().dropna(subset=['CLASS'])

        melted = melted.groupby(['FILE', 'PAGE', 'CLASS'], as_index=False)['SCORE'].max()
        long_dfs.append(melted)

    if not long_dfs:
        empty_cols = (
            ["FILE", "PAGE"]
            + [f"CLASS-{i}" for i in range(1, top_N + 1)]
            + [f"SCORE-{i}" for i in range(1, top_N + 1)]
        )
        return pd.DataFrame(columns=empty_cols)

    combined = pd.concat(long_dfs, ignore_index=True)

    grouped = combined.groupby(['FILE', 'PAGE', 'CLASS'])['SCORE'].sum().reset_index()
    grouped['AVG_SCORE'] = (grouped['SCORE'] / num_models).clip(upper=1.0)

    grouped.sort_values(['FILE', 'PAGE', 'AVG_SCORE'], ascending=[True, True, False], inplace=True)
    grouped['rank'] = grouped.groupby(['FILE', 'PAGE']).cumcount() + 1
    top_n_df = grouped[grouped['rank'] <= top_N].copy()

    pivot = top_n_df.pivot_table(
        index=['FILE', 'PAGE'], columns='rank', values=['CLASS', 'AVG_SCORE'], aggfunc='first',
    )

    flat = pd.DataFrame(index=pivot.index)
    max_rank = int(top_n_df['rank'].max()) if not top_n_df.empty else 0
    for r in range(1, max_rank + 1):
        flat[f'CLASS-{r}'] = pivot.get(('CLASS', r), pd.NA)
        flat[f'SCORE-{r}'] = pivot.get(('AVG_SCORE', r), pd.NA)

    result = flat.reset_index()
    result = result.replace({0: "", 0.0: ""})

    # Mirror averaging.py: when a SCORE-i slot is blank (was zero), clear its
    # paired CLASS-i label too, so a class never appears with an empty score
    # (matches ARUP_averaged.csv, where both cells are empty/NaN together).
    for i in range(2, top_N + 1):
        score_col, class_col = f"SCORE-{i}", f"CLASS-{i}"
        if score_col in result.columns and class_col in result.columns:
            result.loc[result[score_col] == "", class_col] = ""

    # ── per-model top-1 vote columns (V1.3 … V5.3) in canonical order ──────
    order = (
        [r for r in revision_best_models if r in all_rdfs]
        if revision_best_models else list(all_rdfs.keys())
    )
    vote_cols: List[str] = []
    for rev in order:
        rdf = all_rdfs[rev]
        if 'CLASS-1' not in rdf.columns:
            continue
        col = _vote_col_name(rev)
        vcol = (
            rdf[['FILE', 'PAGE', 'CLASS-1']]
            .drop_duplicates(['FILE', 'PAGE'])
            .rename(columns={'CLASS-1': col})
        )
        result = result.merge(vcol, on=['FILE', 'PAGE'], how='left')
        vote_cols.append(col)

    avg_cols = [c for c in result.columns if c not in (['FILE', 'PAGE'] + vote_cols)]
    result = result[['FILE', 'PAGE'] + vote_cols + avg_cols]
    result.sort_values(['FILE', 'PAGE'], ascending=[True, True], inplace=True)
    return result


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
    """Run all best models and write the wide + averaged result CSVs.

    Returns the path to the averaged CSV, or "" when average_best is False
    (the wide BEST_{n}_models_TOP-1.csv is still written in that case).
    """
    out_tables = Path(output_dir) / "tables"
    out_tables.mkdir(parents=True, exist_ok=True)

    n_models = len(revision_best_models)
    all_rdfs: Dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # Choose execution engine
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Wide per-model vote file  (always)  — averaging.py-compatible
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Optional per-model Top-N intermediates
    # ------------------------------------------------------------------
    if save_intermediates:
        for rev, rdf in all_rdfs.items():
            path = out_tables / f"{time_stamp}_{rev.replace('.', '')}_TOP-{top_N}.csv"
            rdf.to_csv(path, index=False)
            print(f"[parallel_best] Saved intermediate model CSV → {path}")

    # ------------------------------------------------------------------
    # Averaged final result (ARUP-style columns)
    # ------------------------------------------------------------------
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