"""Vectorized operator implementations on (T, N) arrays.

Rules (design 4.6 "연산자 계산 규칙"):
  - Time-series ops look backward only and need a full window of finite values.
  - ts_rank: percentile rank of today's value within the trailing w values.
  - rank: cross-sectional percentile rank; zscore: cross-sectional (x - mean) / std.
  - +-inf produced by any op is turned into NaN immediately.
"""

from __future__ import annotations

import numpy as np

from src.backtest import kernels as K
from src.data.panel import Panel
from src.generator.grammar import Node


def _clean(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    if not a.flags.writeable:
        a = a.copy()
    a[~np.isfinite(a)] = np.nan
    return a


def _shift(x: np.ndarray, w: int) -> np.ndarray:
    out = np.full_like(x, np.nan)
    out[w:] = x[:-w]
    return out


def apply_op(op: str, args: list[np.ndarray], window: int | None) -> np.ndarray:
    args = [np.ascontiguousarray(a, dtype=np.float64) for a in args]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        match op:
            case "ts_mean":
                out = K.rolling_mean(args[0], window)
            case "ts_std":
                out = K.rolling_std(args[0], window)
            case "ts_max":
                out = K.rolling_max(args[0], window)
            case "ts_min":
                out = K.rolling_min(args[0], window)
            case "ts_rank":
                out = K.rolling_rank(args[0], window)
            case "ts_delay":
                out = _shift(args[0], window)
            case "ts_delta":
                out = args[0] - _shift(args[0], window)
            case "ts_corr":
                out = K.rolling_corr(args[0], args[1], window)
            case "rank":
                out = K.cs_rank(args[0])
            case "zscore":
                out = K.cs_zscore(args[0])
            case "add":
                out = args[0] + args[1]
            case "sub":
                out = args[0] - args[1]
            case "mul":
                out = args[0] * args[1]
            case "div":
                out = args[0] / args[1]
            case "abs":
                out = np.abs(args[0])
            case "sign":
                out = np.sign(args[0])
            case _:
                raise ValueError(f"unknown operator {op!r}")
    return _clean(out)


def evaluate(node: Node, panel: Panel, cache: dict[str, np.ndarray] | None = None) -> np.ndarray:
    """Evaluate a formula into a (T, N) signal. `cache` memoizes subtrees by text."""
    cache = {} if cache is None else cache
    key = node.text
    if key in cache:
        return cache[key]
    if node.is_leaf:
        out = panel.fields[node.op]
    else:
        args = [evaluate(c, panel, cache) for c in node.children]
        out = apply_op(node.op, args, node.window)
    cache[key] = out
    return out
