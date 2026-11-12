"""Look-ahead and correctness tests for operators and the signal filter."""

import numpy as np
import pandas as pd
import pytest

from src.backtest.operators import evaluate
from src.data.panel import build_panel
from src.generator.filters import signal_filter
from src.generator.grammar import load_grammar, parse

G = load_grammar()
T, N = 300, 12


def make_panel(seed: int = 0, perturb_after: int | None = None):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=T)
    cols = [f"S{i}" for i in range(N)]
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, (T, N)), axis=0))
    frames = {
        "close": close,
        "open": close * (1 + rng.normal(0, 0.005, (T, N))),
        "high": close * 1.01,
        "low": close * 0.99,
        "volume": rng.integers(1_000, 10_000, (T, N)).astype(float),
    }
    if perturb_after is not None:
        prng = np.random.default_rng(999)
        for k in frames:
            frames[k] = frames[k].copy()
            frames[k][perturb_after + 1 :] *= prng.uniform(0.5, 2.0, (T - perturb_after - 1, N))
    return build_panel({k: pd.DataFrame(v, index=dates, columns=cols) for k, v in frames.items()})


ALL_OPS_FORMULAS = [
    "ts_mean(close, 20)",
    "ts_std(returns, 10)",
    "ts_rank(volume, 20)",
    "ts_delta(close, 5)",
    "ts_delay(open, 10)",
    "ts_corr(close, volume, 20)",
    "ts_max(high, 60)",
    "ts_min(low, 60)",
    "rank(div(close, ts_mean(close, 20)))",
    "zscore(returns)",
    "add(rank(close), rank(volume))",
    "sub(high, low)",
    "mul(returns, volume)",
    "abs(returns)",
    "sign(ts_delta(close, 5))",
    "rank(ts_corr(rank(close), rank(volume), 10))",
]


@pytest.mark.parametrize("text", ALL_OPS_FORMULAS)
def test_no_lookahead(text):
    """Changing data after day t must not change signals up to and including t."""
    t = 200
    base = evaluate(parse(text, G), make_panel())
    perturbed = evaluate(parse(text, G), make_panel(perturb_after=t))
    np.testing.assert_array_equal(base[: t + 1], perturbed[: t + 1])
    assert not np.allclose(np.nan_to_num(base[t + 1 :]), np.nan_to_num(perturbed[t + 1 :]))


def test_operator_values():
    p = make_panel()
    close = p.fields["close"]
    np.testing.assert_allclose(
        evaluate(parse("ts_mean(close, 5)"), p)[10], close[6:11].mean(axis=0)
    )
    delta = evaluate(parse("ts_delta(close, 5)"), p)
    np.testing.assert_allclose(delta[10], close[10] - close[5])
    assert np.isnan(evaluate(parse("ts_mean(close, 20)"), p)[18]).all()  # full window
    r = evaluate(parse("rank(close)"), p)
    assert np.allclose(np.sort(r[50]), np.arange(1, N + 1) / N)


def test_division_by_zero_becomes_nan():
    p = make_panel()
    out = evaluate(parse("div(returns, sub(close, ts_delay(close, 5)))"), p)
    assert not np.isinf(out).any()


def test_signal_filter():
    valid = np.ones((100, 10), dtype=bool)
    rows = slice(0, 100)
    good = np.random.default_rng(0).normal(size=(100, 10))
    assert signal_filter(good, valid, rows, 0.7, 0.1).passed
    missing = good.copy()
    missing[:40] = np.nan
    assert signal_filter(missing, valid, rows, 0.7, 0.1).reason == "missing"
    const = good.copy()
    const[:20] = 1.0
    assert signal_filter(const, valid, rows, 0.7, 0.1).reason == "constant"


# ------------------------------------------------- kernels vs pandas reference


def _ref(op, x, y=None, w=None):
    df = pd.DataFrame(x)
    r = df.rolling(w, min_periods=w) if w else None
    if op == "ts_mean":
        return r.mean().to_numpy()
    if op == "ts_std":
        return r.std().to_numpy()
    if op == "ts_max":
        return r.max().to_numpy()
    if op == "ts_min":
        return r.min().to_numpy()
    if op == "ts_rank":
        return r.rank(pct=True).to_numpy()
    if op == "ts_corr":
        return r.corr(pd.DataFrame(y)).to_numpy()
    if op == "rank":
        return df.rank(axis=1, pct=True).to_numpy()
    if op == "zscore":
        return ((df.sub(df.mean(axis=1), axis=0)).div(df.std(axis=1, ddof=0), axis=0)).to_numpy()
    raise ValueError(op)


@pytest.mark.parametrize(
    "op,w",
    [
        ("ts_mean", 10),
        ("ts_std", 10),
        ("ts_max", 5),
        ("ts_min", 20),
        ("ts_rank", 10),
        ("ts_corr", 10),
        ("rank", None),
        ("zscore", None),
    ],
)
def test_kernels_match_pandas(op, w):
    from src.backtest.operators import apply_op

    rng = np.random.default_rng(1)
    x = rng.normal(size=(200, 15))
    y = rng.normal(size=(200, 15))
    x[rng.random(x.shape) < 0.05] = np.nan
    x[:, 3] = np.round(x[:, 3])  # ties
    args = [x, y] if op == "ts_corr" else [x]
    got = apply_op(op, args, w)
    exp = np.array(_ref(op, x, y, w), dtype=float, copy=True)
    exp[~np.isfinite(exp)] = np.nan
    np.testing.assert_allclose(got, exp, rtol=1e-7, atol=1e-9, equal_nan=True)
