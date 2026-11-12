import numpy as np
import pandas as pd

from src.experiment.criteria import (
    PnlStore,
    Thresholds,
    count_unique_passes,
    directed,
    greedy_unique,
)


def _results():
    return pd.DataFrame(
        {
            "formula_id": [0, 1, 2],
            "is_sharpe_pos": [1.2, -6.1, -0.4],
            "is_sharpe_neg": [-1.5, -6.7, 0.9],
            "test_sharpe_pos": [0.5, 0.0, 0.8],
            "test_sharpe_neg": [-0.6, 0.0, 0.4],
            "is_turnover": [0.1, 0.9, 0.2],
        }
    )


def test_direction_takes_better_net_side_not_abs():
    d = directed(_results())
    assert d["direction"].tolist() == [1, 1, -1]
    # Formula 1 loses in both directions: its target stays negative (no abs).
    assert d["is_sharpe"].tolist() == [1.2, -6.1, 0.9]
    assert d["test_sharpe"].tolist() == [0.5, 0.0, 0.4]


def test_greedy_unique_drops_correlated_lower_sharpe():
    rng = np.random.default_rng(0)
    base = rng.normal(size=300)
    gross = np.stack([base, base + 0.1 * rng.normal(size=300), rng.normal(size=300)])
    store = PnlStore(np.stack([gross, np.zeros_like(gross)]).astype(np.float32), np.ones(3))
    is_sharpe = np.array([1.0, 2.0, 0.5])
    adopted = greedy_unique(np.arange(3), is_sharpe, store, 0.7)
    # Row 1 has the higher Sharpe and wins; row 0 is a near-copy; row 2 is independent.
    assert adopted.tolist() == [False, True, True]


def test_greedy_uses_directed_pnl():
    rng = np.random.default_rng(1)
    a = rng.normal(size=300)
    gross = np.stack([a, -a])  # opposite gross PnL; same directed PnL after flipping row 1
    store = PnlStore(np.stack([gross, np.zeros_like(gross)]).astype(np.float32), np.array([1, -1]))
    adopted = greedy_unique(np.arange(2), np.array([1.0, 0.9]), store, 0.7)
    assert adopted.tolist() == [True, False]


def test_prefix_property_used_by_calibration():
    """Adopted set at a stricter IS threshold == loosest adopted set filtered by threshold."""
    rng = np.random.default_rng(2)
    n, t = 60, 200
    factors = rng.normal(size=(5, t))
    gross = factors[rng.integers(0, 5, n)] + 0.8 * rng.normal(size=(n, t))
    store = PnlStore(np.stack([gross, np.zeros_like(gross)]).astype(np.float32), np.ones(n))
    d = pd.DataFrame(
        {
            "formula_id": np.arange(n),
            "direction": 1,
            "is_sharpe": rng.uniform(0, 2, n),
            "test_sharpe": np.ones(n),
            "is_turnover": np.zeros(n),
        }
    )
    idx = np.arange(n)
    _, loose = count_unique_passes(idx, d, Thresholds(0.3, 0.0, 0.5, 0.7), store)
    for h in (0.6, 1.0, 1.4):
        _, strict = count_unique_passes(idx, d, Thresholds(h, 0.0, 0.5, 0.7), store)
        expected = loose[d["is_sharpe"].to_numpy()[loose] >= h]
        assert sorted(strict) == sorted(expected)
