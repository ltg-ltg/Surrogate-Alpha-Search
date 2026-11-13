"""Residual (orthogonalized) target from warm-up IS data only (design 7.6, exploratory).

1. References: warm-up formulas with directed IS net Sharpe >= IS threshold and IS
   turnover <= cap, in descending IS Sharpe, adopted if |corr| < max_corr with every
   adopted reference; keep the first K. No Test information is used.
2. Target: for each warm-up formula, OLS of its directed IS daily net PnL on the K
   reference series (with intercept); y_resid = annualized Sharpe of r - sum(beta * F).
"""

from __future__ import annotations

import numpy as np


def _standardize(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=1, keepdims=True)
    sd = np.sqrt((x**2).mean(axis=1, keepdims=True))
    return np.divide(x, sd, out=np.zeros_like(x), where=sd > 0)


def select_references(
    net: np.ndarray,
    is_sharpe: np.ndarray,
    is_turnover: np.ndarray,
    min_sharpe: float,
    max_turnover: float,
    max_corr: float,
    k: int,
) -> np.ndarray:
    """Row indices (into the warm-up arrays) of up to k mutually distinct references."""
    ok = np.flatnonzero((is_sharpe >= min_sharpe) & (is_turnover <= max_turnover))
    ok = ok[np.lexsort((ok, -is_sharpe[ok]))]
    z = _standardize(net[ok])
    n_days = net.shape[1]
    chosen: list[int] = []
    for pos, row in enumerate(ok):
        if len(chosen) == k:
            break
        if chosen:
            zc = _standardize(net[chosen])
            if np.abs(zc @ z[pos]).max() / n_days >= max_corr:
                continue
        chosen.append(int(row))
    return np.array(chosen, dtype=np.int64)


def residual_sharpe(net: np.ndarray, refs: np.ndarray, annualization: int) -> np.ndarray:
    """Sharpe of r - F beta (intercept kept in the residual) for every row of `net`."""
    if len(refs) == 0:
        resid = net
    else:
        F = net[refs].T  # (T, K)
        X = np.column_stack([np.ones(len(F)), F])
        coef, *_ = np.linalg.lstsq(X, net.T, rcond=None)  # (K+1, W)
        resid = net - (F @ coef[1:]).T  # removes factor exposure, keeps intercept
    mu = resid.mean(axis=1)
    sd = resid.std(axis=1, ddof=1)
    return np.divide(mu, sd, out=np.zeros_like(mu), where=sd > 1e-12) * np.sqrt(annualization)
