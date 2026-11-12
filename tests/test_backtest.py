"""Backtest mechanics (design 5.1) and OOS file isolation (CLAUDE.md rule 3)."""

from pathlib import Path

import numpy as np
import pytest

from src.backtest.metrics import max_drawdown, period_metrics, run_series, sharpe, weights

T, N = 30, 4
WINSOR = (0.0, 1.0)  # no clipping for hand-checkable numbers


def test_weights_demeaned_and_normalized():
    rng = np.random.default_rng(0)
    sig = rng.normal(size=(T, N))
    sig[3, 1] = np.nan
    valid = np.ones((T, N), dtype=bool)
    valid[5, 2] = False
    w = weights(sig, valid, WINSOR)
    np.testing.assert_allclose(np.abs(w).sum(axis=1), 1.0)
    np.testing.assert_allclose(w.sum(axis=1), 0.0, atol=1e-12)
    assert w[3, 1] == 0 and w[5, 2] == 0


def test_constant_signal_gives_zero_weights():
    w = weights(np.ones((T, N)), np.ones((T, N), dtype=bool), WINSOR)
    assert (w == 0).all()


def test_execution_lag_and_cost():
    """w(t) earns ret(t+2); cost 5bp * sum|dw| booked on the same return date."""
    sig = np.tile(np.array([1.0, 0.0, 0.0, -1.0]), (T, 1))  # w = [.5, 0, 0, -.5]
    rets = np.zeros((T, N))
    rets[10] = [0.02, 0, 0, -0.01]  # only ret on row 10 is nonzero
    s = run_series(sig, np.ones((T, N), dtype=bool), rets, delay=1, cost_bps=5.0, winsor=WINSOR)
    # Weights from row 8 earn row-10 return: 0.5*0.02 - 0.5*(-0.01) = 0.015.
    assert s.gross[10] == pytest.approx(0.015)
    assert np.nansum(s.gross) == pytest.approx(0.015)
    # Initial build-up (|dw| = 1) on row 0 is booked on row 2; nothing afterwards.
    assert s.cost[2] == pytest.approx(5e-4)
    assert s.turnover[2] == pytest.approx(0.5)
    assert np.nansum(s.cost) == pytest.approx(5e-4)
    assert np.isnan(s.gross[:2]).all()


def test_signal_cannot_see_its_own_return():
    """A signal equal to the same-day return must not earn that return (look-ahead)."""
    rng = np.random.default_rng(1)
    rets = rng.normal(0, 0.01, (500, 20))
    s = run_series(rets, np.ones_like(rets, dtype=bool), rets, 1, 0.0, WINSOR)
    assert abs(sharpe(s.gross, 252)) < 2.0  # an i.i.d. series has no edge 2 days ahead


def test_direction_flip_changes_gross_not_cost():
    rng = np.random.default_rng(2)
    sig = rng.normal(size=(200, 10))
    rets = rng.normal(0, 0.01, (200, 10))
    s = run_series(sig, np.ones_like(sig, dtype=bool), rets, 1, 5.0, (0.01, 0.99))
    rows = slice(0, 200)
    pos = period_metrics(s, rows, 1.0, 252)
    neg = period_metrics(s, rows, -1.0, 252)
    assert pos["turnover"] == neg["turnover"]
    assert pos["rank_ic"] == pytest.approx(-neg["rank_ic"])
    # With costs, flipping a losing signal does not give the mirror-image Sharpe.
    assert pos["sharpe"] + neg["sharpe"] < 0


def test_max_drawdown():
    assert max_drawdown(np.array([0.1, -0.2, 0.05, -0.1])) == pytest.approx(0.25)


def test_oos_results_referenced_only_by_engine_and_report():
    root = Path(__file__).resolve().parents[1] / "src"
    allowed = {"engine.py", "oos_report.py"}
    offenders = [
        p.relative_to(root).as_posix()
        for p in root.rglob("*.py")
        if p.name not in allowed
        and ("results_oos" in (txt := p.read_text(encoding="utf-8")) or "_oos.parquet" in txt)
    ]
    assert offenders == []
