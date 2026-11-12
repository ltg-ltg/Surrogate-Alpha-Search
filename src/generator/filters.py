"""Signal-level filters shared by all arms (design 4.8)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CONST_TOL = 1e-12


@dataclass(frozen=True)
class FilterResult:
    passed: bool
    reason: str  # "ok", "missing", "constant"
    valid_ratio: float
    constant_day_ratio: float


def signal_filter(
    signal: np.ndarray,
    valid: np.ndarray,
    rows: slice,
    min_valid_ratio: float,
    max_constant_day_ratio: float,
) -> FilterResult:
    """Judge a (T, N) signal on the evaluation window `rows` (2016-2025, design 3.3).

    valid_ratio: share of tradable stock-days (valid) where the signal is finite.
    constant day: fewer than two finite values, or zero cross-sectional std.
    """
    s = signal[rows]
    v = valid[rows]
    finite = np.isfinite(s) & v
    n_valid = v.sum()
    valid_ratio = float(finite.sum() / n_valid) if n_valid else 0.0

    masked = np.where(finite, s, np.nan)
    count = finite.sum(axis=1)
    with np.errstate(invalid="ignore"):
        mean = np.nansum(masked, axis=1) / np.maximum(count, 1)
        var = np.nansum((masked - mean[:, None]) ** 2, axis=1) / np.maximum(count, 1)
    constant = (count < 2) | (np.sqrt(var) <= CONST_TOL)
    constant_ratio = float(constant.mean())

    if valid_ratio < min_valid_ratio:
        return FilterResult(False, "missing", valid_ratio, constant_ratio)
    if constant_ratio > max_constant_day_ratio:
        return FilterResult(False, "constant", valid_ratio, constant_ratio)
    return FilterResult(True, "ok", valid_ratio, constant_ratio)
