from typing import Dict, List, Optional

import pandas as pd


def _vote_col_name(rev: str) -> str:
    """Map a revision string to its ARUP-style vote column header."""
    return (rev[0].upper() + rev[1:]) if rev else rev

def average_rdfs(
    all_rdfs: Dict[str, pd.DataFrame],
    top_N: int,
    revision_best_models: Optional[dict] = None,
) -> pd.DataFrame:
    """Average per-model softmax probabilities and re-rank to Top-N.
    Used primarily by the parallel_best.py CLI engine.
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
    result = result.replace({0: ""})

    for i in range(2, top_N + 1):
        score_col, class_col = f"SCORE-{i}", f"CLASS-{i}"
        if score_col in result.columns and class_col in result.columns:
            result.loc[result[score_col] == "", class_col] = ""

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

def average_prediction_dicts(
    predictions_list: List[List[Dict[str, float]]],
    categories: List[str],
    top_n: int
) -> List[Dict[str, float]]:
    """
    Averages a list of prediction dictionaries.
    Used primarily by the FastAPI service for rapid in-memory JSON generation.
    """
    if not predictions_list:
        return []

    num_models = len(predictions_list)
    aggregated_scores = {cat: 0.0 for cat in categories}

    for preds in predictions_list:
        for item in preds:
            aggregated_scores[item['label']] += item['score']

    final_results = [
        {"label": lbl, "score": min(s / num_models, 1.0)}
        for lbl, s in aggregated_scores.items()
    ]
    final_results.sort(key=lambda x: x["score"], reverse=True)
    return final_results[:top_n]
