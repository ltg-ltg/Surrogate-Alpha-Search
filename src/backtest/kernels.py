"""Numba kernels for (T, N) operator evaluation.

Semantics (design 4.6 "연산자 계산 규칙"):
  - Rolling kernels look backward only and return NaN unless all w values are finite.
  - ts_std uses ddof=1. Rank kernels use average ranks for ties, scaled to (0, 1] by count.
Time-series kernels parallelize over columns, cross-sectional kernels over rows.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

_NAN = np.nan


@njit(parallel=True, cache=True)
def _rolling_mean_nt(x, w):
    N, T = x.shape
    out = np.full((N, T), _NAN)
    for j in prange(N):
        for t in range(w - 1, T):
            s = 0.0
            ok = True
            for k in range(t - w + 1, t + 1):
                v = x[j, k]
                if not np.isfinite(v):
                    ok = False
                    break
                s += v
            if ok:
                out[j, t] = s / w
    return out


@njit(parallel=True, cache=True)
def _rolling_std_nt(x, w):
    N, T = x.shape
    out = np.full((N, T), _NAN)
    for j in prange(N):
        for t in range(w - 1, T):
            s = 0.0
            ok = True
            for k in range(t - w + 1, t + 1):
                v = x[j, k]
                if not np.isfinite(v):
                    ok = False
                    break
                s += v
            if not ok:
                continue
            m = s / w
            ss = 0.0
            for k in range(t - w + 1, t + 1):
                d = x[j, k] - m
                ss += d * d
            out[j, t] = np.sqrt(ss / (w - 1))
    return out


@njit(parallel=True, cache=True)
def _rolling_max_nt(x, w):
    N, T = x.shape
    out = np.full((N, T), _NAN)
    for j in prange(N):
        for t in range(w - 1, T):
            m = -np.inf
            ok = True
            for k in range(t - w + 1, t + 1):
                v = x[j, k]
                if not np.isfinite(v):
                    ok = False
                    break
                if v > m:
                    m = v
            if ok:
                out[j, t] = m
    return out


@njit(parallel=True, cache=True)
def _rolling_min_nt(x, w):
    N, T = x.shape
    out = np.full((N, T), _NAN)
    for j in prange(N):
        for t in range(w - 1, T):
            m = np.inf
            ok = True
            for k in range(t - w + 1, t + 1):
                v = x[j, k]
                if not np.isfinite(v):
                    ok = False
                    break
                if v < m:
                    m = v
            if ok:
                out[j, t] = m
    return out


@njit(parallel=True, cache=True)
def _rolling_rank_nt(x, w):
    """Percentile rank of x[t] among x[t-w+1..t] (average ties), in (0, 1]."""
    N, T = x.shape
    out = np.full((N, T), _NAN)
    for j in prange(N):
        for t in range(w - 1, T):
            cur = x[j, t]
            if not np.isfinite(cur):
                continue
            less = 0
            equal = 0
            ok = True
            for k in range(t - w + 1, t + 1):
                v = x[j, k]
                if not np.isfinite(v):
                    ok = False
                    break
                if v < cur:
                    less += 1
                elif v == cur:
                    equal += 1
            if ok:
                out[j, t] = (less + (equal + 1) / 2.0) / w
    return out


@njit(parallel=True, cache=True)
def _rolling_corr_nt(x, y, w):
    N, T = x.shape
    out = np.full((N, T), _NAN)
    for j in prange(N):
        for t in range(w - 1, T):
            sx = 0.0
            sy = 0.0
            ok = True
            for k in range(t - w + 1, t + 1):
                a = x[j, k]
                b = y[j, k]
                if not (np.isfinite(a) and np.isfinite(b)):
                    ok = False
                    break
                sx += a
                sy += b
            if not ok:
                continue
            mx = sx / w
            my = sy / w
            cxy = 0.0
            vx = 0.0
            vy = 0.0
            for k in range(t - w + 1, t + 1):
                dx = x[j, k] - mx
                dy = y[j, k] - my
                cxy += dx * dy
                vx += dx * dx
                vy += dy * dy
            den = np.sqrt(vx * vy)
            if den > 0.0:
                out[j, t] = cxy / den
    return out


@njit(parallel=True, cache=True)
def cs_rank(x):
    """Cross-sectional percentile rank per row (average ties), in (0, 1]."""
    T, N = x.shape
    out = np.full((T, N), _NAN)
    for t in prange(T):
        row = x[t]
        idx = np.empty(N, dtype=np.int64)
        n = 0
        for j in range(N):
            if np.isfinite(row[j]):
                idx[n] = j
                n += 1
        if n == 0:
            continue
        vals = np.empty(n)
        for i in range(n):
            vals[i] = row[idx[i]]
        order = np.argsort(vals, kind="mergesort")
        i = 0
        while i < n:
            k = i
            while k + 1 < n and vals[order[k + 1]] == vals[order[i]]:
                k += 1
            r = (i + k) / 2.0 + 1.0  # average 1-based rank of the tie block
            for m in range(i, k + 1):
                out[t, idx[order[m]]] = r / n
            i = k + 1
    return out


@njit(parallel=True, cache=True)
def cs_zscore(x):
    """(x - mean) / std per row over finite values (population std)."""
    T, N = x.shape
    out = np.full((T, N), _NAN)
    for t in prange(T):
        s = 0.0
        n = 0
        for j in range(N):
            v = x[t, j]
            if np.isfinite(v):
                s += v
                n += 1
        if n == 0:
            continue
        m = s / n
        ss = 0.0
        for j in range(N):
            v = x[t, j]
            if np.isfinite(v):
                ss += (v - m) * (v - m)
        sd = np.sqrt(ss / n)
        if sd <= 0.0:
            continue
        for j in range(N):
            v = x[t, j]
            if np.isfinite(v):
                out[t, j] = (v - m) / sd
    return out


# ---------------------------------------------------------------- (T, N) wrappers


def _ts(kernel, x, w):
    xt = np.ascontiguousarray(x.T)
    return np.ascontiguousarray(kernel(xt, w).T)


def rolling_mean(x, w):
    return _ts(_rolling_mean_nt, x, w)


def rolling_std(x, w):
    return _ts(_rolling_std_nt, x, w)


def rolling_max(x, w):
    return _ts(_rolling_max_nt, x, w)


def rolling_min(x, w):
    return _ts(_rolling_min_nt, x, w)


def rolling_rank(x, w):
    return _ts(_rolling_rank_nt, x, w)


def rolling_corr(x, y, w):
    xt = np.ascontiguousarray(x.T)
    yt = np.ascontiguousarray(y.T)
    return np.ascontiguousarray(_rolling_corr_nt(xt, yt, w).T)


# ---------------------------------------------------------------- backtest helpers


@njit(parallel=True, cache=True)
def cs_winsorize(x, q_lo, q_hi):
    """Clip each row to its [q_lo, q_hi] quantiles over finite values (linear interp)."""
    T, N = x.shape
    out = x.copy()
    for t in prange(T):
        vals = np.empty(N)
        n = 0
        for j in range(N):
            if np.isfinite(x[t, j]):
                vals[n] = x[t, j]
                n += 1
        if n < 2:
            continue
        s = np.sort(vals[:n])
        bounds = np.empty(2)
        for b in range(2):
            q = q_lo if b == 0 else q_hi
            pos = q * (n - 1)
            lo = int(np.floor(pos))
            hi = min(lo + 1, n - 1)
            bounds[b] = s[lo] + (s[hi] - s[lo]) * (pos - lo)
        for j in range(N):
            v = x[t, j]
            if np.isfinite(v):
                if v < bounds[0]:
                    out[t, j] = bounds[0]
                elif v > bounds[1]:
                    out[t, j] = bounds[1]
    return out


@njit(cache=True)
def _avg_rank(v):
    n = v.shape[0]
    order = np.argsort(v, kind="mergesort")
    r = np.empty(n)
    i = 0
    while i < n:
        k = i
        while k + 1 < n and v[order[k + 1]] == v[order[i]]:
            k += 1
        avg = (i + k) / 2.0 + 1.0
        for m in range(i, k + 1):
            r[order[m]] = avg
        i = k + 1
    return r


@njit(parallel=True, cache=True)
def spearman_rows(x, y):
    """Per-row Spearman correlation over columns where both x and y are finite."""
    T, N = x.shape
    out = np.full(T, _NAN)
    for t in prange(T):
        a = np.empty(N)
        b = np.empty(N)
        n = 0
        for j in range(N):
            if np.isfinite(x[t, j]) and np.isfinite(y[t, j]):
                a[n] = x[t, j]
                b[n] = y[t, j]
                n += 1
        if n < 3:
            continue
        ra = _avg_rank(a[:n])
        rb = _avg_rank(b[:n])
        ma = ra.mean()
        mb = rb.mean()
        cov = 0.0
        va = 0.0
        vb = 0.0
        for i in range(n):
            da = ra[i] - ma
            db = rb[i] - mb
            cov += da * db
            va += da * da
            vb += db * db
        if va > 0.0 and vb > 0.0:
            out[t] = cov / np.sqrt(va * vb)
    return out
