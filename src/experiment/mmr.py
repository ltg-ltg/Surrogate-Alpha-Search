"""Diversity-aware MMR selection on structural similarity (design 7.5, exploratory).

score_i = r_i - lam * max_{j in selected} sim(i, j), picked greedily; ties -> lowest id.
r_i: percentile rank of the arm's prediction within the candidates, in (0, 1].
sim: (1 + cos) / 2 between pool-standardized base AST feature vectors (string-only).
"""

from __future__ import annotations

import numpy as np


def unit_feature_vectors(X_pool: np.ndarray) -> np.ndarray:
    """Standardize columns over the pool, then L2-normalize rows."""
    mu = X_pool.mean(axis=0)
    sd = X_pool.std(axis=0)
    Z = (X_pool - mu) / np.where(sd > 0, sd, 1.0)
    norm = np.linalg.norm(Z, axis=1, keepdims=True)
    return Z / np.where(norm > 0, norm, 1.0)


def percentile_rank(scores: np.ndarray) -> np.ndarray:
    """Average-rank percentile in (0, 1]."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ties
    s_sorted = scores[order]
    i = 0
    n = len(scores)
    while i < n:
        j = i
        while j + 1 < n and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks / n


def mmr_select(
    cand_ids: np.ndarray, relevance: np.ndarray, U: np.ndarray, lam: float, n_pick: int
) -> np.ndarray:
    """Greedy MMR over candidates. U: unit feature vectors indexed by formula_id."""
    cand_ids = np.asarray(cand_ids)
    Uc = U[cand_ids]
    max_sim = np.zeros(len(cand_ids))
    taken = np.zeros(len(cand_ids), dtype=bool)
    picks = np.empty(n_pick, dtype=np.int64)
    # Secondary key for ties: lowest formula id. Candidates are pre-sorted by id.
    order = np.argsort(cand_ids, kind="mergesort")
    cand_ids, relevance, Uc = cand_ids[order], relevance[order], Uc[order]
    for t in range(n_pick):
        score = relevance - lam * max_sim
        score[taken] = -np.inf
        i = int(np.argmax(score))  # first max -> lowest id among ties
        taken[i] = True
        picks[t] = cand_ids[i]
        sim = (1.0 + Uc @ Uc[i]) / 2.0
        np.maximum(max_sim, sim, out=max_sim)
    return picks
