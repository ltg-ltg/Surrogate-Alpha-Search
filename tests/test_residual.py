import numpy as np

from src.surrogate.residual import residual_sharpe, select_references


def test_references_are_distinct_and_is_only():
    rng = np.random.default_rng(0)
    f = rng.normal(size=500)
    net = np.stack([f + 0.1 * rng.normal(size=500) for _ in range(3)] + [rng.normal(size=500)])
    sharpe = np.array([2.0, 1.9, 1.8, 1.0])
    turnover = np.array([0.1, 0.1, 0.1, 0.1])
    refs = select_references(net, sharpe, turnover, 0.5, 0.5, 0.7, k=5)
    assert refs.tolist() == [0, 3]  # 1 and 2 are near-copies of 0
    # Turnover cap and Sharpe threshold apply.
    turnover[3] = 0.9
    assert select_references(net, sharpe, turnover, 0.5, 0.5, 0.7, k=5).tolist() == [0]


def test_residual_removes_factor_but_keeps_independent_alpha():
    rng = np.random.default_rng(1)
    T = 2000
    factor = 0.001 + 0.01 * rng.normal(size=T)
    clone = factor + 0.001 * rng.normal(size=T)
    alpha = 0.0008 + 0.01 * rng.normal(size=T)
    net = np.stack([factor, clone, alpha])
    s = residual_sharpe(net, np.array([0]), 252)
    raw = net.mean(1) / net.std(1, ddof=1) * np.sqrt(252)
    assert abs(s[0]) < 1e-6  # reference explains itself
    assert abs(s[1]) < raw[1] / 3  # clone loses most of its Sharpe
    assert abs(s[2] - raw[2]) < 0.2  # independent alpha keeps its Sharpe


def test_no_references_returns_raw_sharpe():
    rng = np.random.default_rng(2)
    net = rng.normal(0.001, 0.01, size=(3, 300))
    raw = net.mean(1) / net.std(1, ddof=1) * np.sqrt(252)
    np.testing.assert_allclose(residual_sharpe(net, np.array([], dtype=int), 252), raw)
