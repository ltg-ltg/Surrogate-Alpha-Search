"""Portfolio construction and per-period metrics (design 5.0-5.2)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.backtest import kernels as K


@dataclass(frozen=True)
class BacktestSeries:
    """Daily series indexed by return date (row t+2 holds the result of weights w(t))."""

    gross: np.ndarray  # (T,) sum_j w(t) * ret(t+2), aligned to row t+2
    cost: np.ndarray  # (T,) 5bp * sum |w(t) - w(t-1)|, aligned to row t+2
    turnover: np.ndarray  # (T,) 0.5 * sum |w(t) - w(t-1)|, aligned to row t+2
    ic: np.ndarray  # (T,) Spearman(signal(t), ret(t+2)), aligned to row t+2


def weights(signal: np.ndarray, valid: np.ndarray, winsor: tuple[float, float]) -> np.ndarray:
    """Preprocess -> demean -> sum|w| = 1. Missing / untradable names get weight 0."""
    s = np.where(valid & np.isfinite(signal), signal, np.nan)
    s = K.cs_winsorize(np.ascontiguousarray(s), winsor[0], winsor[1])
    with np.errstate(invalid="ignore"):
        count = np.isfinite(s).sum(axis=1, keepdims=True)
        mean = np.where(count > 0, np.nansum(s, axis=1, keepdims=True) / np.maximum(count, 1), 0)
        d = np.where(np.isfinite(s), s - mean, 0.0)
        gross = np.abs(d).sum(axis=1, keepdims=True)
        w = np.where(gross > 0, d / np.where(gross > 0, gross, 1.0), 0.0)
    return w


def run_series(
    signal: np.ndarray,
    valid: np.ndarray,
    returns: np.ndarray,
    delay: int,
    cost_bps: float,
    winsor: tuple[float, float],
) -> BacktestSeries:
    """Signal at close t, trade at close t+delay, earn ret(t+delay+1) (design 5.0)."""
    T = signal.shape[0]
    lag = delay + 1  # rows between signal date and return date (2 for delay 1)
    w = weights(signal, valid, winsor)
    dw = np.abs(np.diff(w, axis=0, prepend=np.zeros((1, w.shape[1])))).sum(axis=1)

    fwd = np.full_like(returns, np.nan)
    fwd[: T - lag] = returns[lag:]  # fwd[t] = ret(t+lag)
    fwd_filled = np.nan_to_num(fwd, nan=0.0)  # untradable / delisted day -> 0 return

    gross_t = (w * fwd_filled).sum(axis=1)
    sig_valid = np.where(valid & np.isfinite(signal), signal, np.nan)
    ic_t = K.spearman_rows(np.ascontiguousarray(sig_valid), np.ascontiguousarray(fwd))

    def to_return_date(a: np.ndarray) -> np.ndarray:
        out = np.full(T, np.nan)
        out[lag:] = a[: T - lag]
        return out

    return BacktestSeries(
        gross=to_return_date(gross_t),
        cost=to_return_date(cost_bps * 1e-4 * dw),
        turnover=to_return_date(0.5 * dw),
        ic=to_return_date(ic_t),
    )


def sharpe(net: np.ndarray, annualization: int) -> float:
    net = net[np.isfinite(net)]
    if net.size < 2:
        return 0.0
    sd = net.std(ddof=1)
    return float(net.mean() / sd * np.sqrt(annualization)) if sd > 0 else 0.0


def max_drawdown(net: np.ndarray) -> float:
    """Max drawdown of the cumulative (summed) daily net return, as a positive number."""
    net = np.nan_to_num(net, nan=0.0)
    cum = np.cumsum(net)
    peak = np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:]
    return float((peak - cum).max()) if cum.size else 0.0


def period_metrics(
    series: BacktestSeries, rows: slice, direction: float, annualization: int
) -> dict[str, float]:
    net = direction * series.gross[rows] - series.cost[rows]
    ic = series.ic[rows]
    return {
        "sharpe": sharpe(net, annualization),
        "turnover": float(np.nanmean(series.turnover[rows])),
        "rank_ic": float(direction * np.nanmean(ic)) if np.isfinite(ic).any() else 0.0,
        "mdd": max_drawdown(net),
    }
