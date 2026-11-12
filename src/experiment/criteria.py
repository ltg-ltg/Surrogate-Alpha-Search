"""Direction rule and pass criteria 1-4 (design 5.3, 6.2).

Direction: the sign (+1 / -1) with the larger IS net Sharpe; the surrogate target and
criterion 1 use that directed IS net Sharpe, max(S+, S-). Criterion 4 is a greedy PnL
de-duplication in descending directed-IS-Sharpe order within an evaluation set.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Thresholds:
    is_sharpe: float
    test_sharpe: float
    max_is_turnover: float
    max_pnl_corr: float


def directed(results: pd.DataFrame) -> pd.DataFrame:
    """Per-formula direction, directed IS/Test net Sharpe and IS turnover."""
    pos = results["is_sharpe_pos"].to_numpy()
    neg = results["is_sharpe_neg"].to_numpy()
    direction = np.where(pos >= neg, 1, -1)
    test = np.where(
        direction == 1, results["test_sharpe_pos"].to_numpy(), results["test_sharpe_neg"]
    )
    return pd.DataFrame(
        {
            "formula_id": results["formula_id"].to_numpy(),
            "direction": direction,
            "is_sharpe": np.maximum(pos, neg),
            "test_sharpe": test,
            "is_turnover": results["is_turnover"].to_numpy(),
        }
    )


def pass_1to3(d: pd.DataFrame, th: Thresholds) -> np.ndarray:
    return (
        (d["is_sharpe"].to_numpy() >= th.is_sharpe)
        & (d["test_sharpe"].to_numpy() >= th.test_sharpe)
        & (d["is_turnover"].to_numpy() <= th.max_is_turnover)
    )


class PnlStore:
    """Directed daily IS net PnL, net = direction * gross - cost, by pool row index."""

    def __init__(self, gross_cost: np.ndarray, direction: np.ndarray):
        self.gross = gross_cost[0]
        self.cost = gross_cost[1]
        self.direction = direction

    def standardized(self, idx: np.ndarray) -> np.ndarray:
        net = self.direction[idx, None] * self.gross[idx].astype(np.float64) - self.cost[idx]
        net = net - net.mean(axis=1, keepdims=True)
        sd = np.sqrt((net**2).mean(axis=1, keepdims=True))
        return np.divide(net, sd, out=np.zeros_like(net), where=sd > 0)


def greedy_unique(
    idx: np.ndarray, is_sharpe: np.ndarray, pnl: PnlStore, max_corr: float
) -> np.ndarray:
    """Return a bool mask over `idx` of signals adopted by the criterion-4 greedy pass.

    `idx` are pool row indices already passing criteria 1-3. Order: directed IS Sharpe
    descending (ties by row index). A signal is adopted if its |corr| with every adopted
    signal is below max_corr.
    """
    order = np.lexsort((idx, -is_sharpe[idx]))
    adopted = np.zeros(len(idx), dtype=bool)
    if len(idx) == 0:
        return adopted
    z = pnl.standardized(idx[order])
    n_days = z.shape[1]
    kept = np.empty((len(idx), n_days))
    k = 0
    for i in range(len(order)):
        if k:
            corr = np.abs(kept[:k] @ z[i]) / n_days
            if corr.max() >= max_corr:
                continue
        kept[k] = z[i]
        k += 1
        adopted[order[i]] = True
    return adopted


def count_unique_passes(
    idx: np.ndarray, d: pd.DataFrame, th: Thresholds, pnl: PnlStore
) -> tuple[int, np.ndarray]:
    """Unique passes (criteria 1-4) within the evaluation set `idx` (pool row indices)."""
    ok = pass_1to3(d.iloc[idx], th)
    cand = idx[ok]
    adopted = greedy_unique(cand, d["is_sharpe"].to_numpy(), pnl, th.max_pnl_corr)
    return int(adopted.sum()), cand[adopted]
