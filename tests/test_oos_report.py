"""oos_report pure functions on synthetic data only (no OOS file is read)."""

import numpy as np
import pandas as pd

from src.analysis import oos_report as R
from src.backtest.metrics import sharpe


def test_seed_metrics_values():
    is_s = np.array([0.5, 1.0, 2.0, 0.4])
    oos = np.array([0.25, -0.5, 1.0, 0.4])
    m = R.seed_metrics(np.array([0, 1, 2]), is_s, oos)
    assert m["n"] == 3
    assert m["median_sharpe"] == 0.25
    assert m["median_decay"] == 0.5  # ratios 0.5, -0.5, 0.5
    assert np.isclose(m["share_positive"], 2 / 3)
    assert np.isclose(m["share_ge_03"], 1 / 3)
    assert np.isnan(R.seed_metrics(np.array([], int), is_s, oos)["median_sharpe"])


def test_verdict_rules():
    assert R.verdict(-0.05, 0.2, 0.1) == "non_inferior"
    assert R.verdict(-0.4, -0.15, 0.1) == "inferior"
    assert R.verdict(-0.2, 0.1, 0.1) == "inconclusive"


def test_shifted_wilcoxon_direction():
    rng = np.random.default_rng(0)
    near_zero = rng.normal(0.0, 0.02, 30)
    far_below = rng.normal(-0.5, 0.02, 30)
    assert R.shifted_wilcoxon_p(near_zero, 0.1) < 0.01
    assert R.shifted_wilcoxon_p(far_below, 0.1) > 0.99


def test_judge_family_flags_inferior_cell():
    rng = np.random.default_rng(1)
    rows = []
    for seed in range(30):
        base = rng.normal(0.5, 0.05)
        for cell, shift in (("rand", 0.0), ("good", 0.0), ("bad", -0.5)):
            rows.append(
                {
                    "cell": cell,
                    "seed": seed,
                    "median_sharpe": base + shift,
                    "median_decay": 0.5 + shift,
                    "share_positive": 0.7 + shift,
                }
            )
    out = pd.DataFrame(R.judge_family(pd.DataFrame(rows), "f", "us", ["good", "bad"], "rand"))
    v = out.set_index(["cell", "metric"])["verdict"]
    assert (v.loc["good"] == "non_inferior").all()
    assert (v.loc["bad"] == "inferior").all()
    assert np.isclose(out["ci_level"].iloc[0], 1 - 0.05 / 2)


def test_sharpe_rows_matches_metrics_sharpe():
    rng = np.random.default_rng(2)
    net = rng.normal(0.001, 0.01, (5, 300))
    net[1, ::7] = np.nan
    net[3] = 0.0
    got = R.sharpe_rows(net, 252)
    want = np.array([sharpe(r, 252) for r in net])
    assert np.allclose(got, want)


def test_block_indices_shape_and_contiguity():
    idx = R.block_indices(103, 20, 50, np.random.default_rng(3))
    assert idx.shape == (50, 103)
    assert idx.min() >= 0 and idx.max() < 103
    assert (np.diff(idx[:, :20], axis=1) == 1).all()


def test_regression_recovers_arm_effect():
    rng = np.random.default_rng(4)
    is_s = rng.uniform(0.4, 2.0, 400)
    stacks = {"rand": np.arange(0, 200), "m": np.arange(200, 400)}
    X, ids, names = R.regression_design(stacks, is_s, "rand")
    y = 0.1 + 0.3 * is_s[ids] - 0.2 * (ids >= 200)
    coef = R.ols(y, X)
    assert names == ["intercept", "is_sharpe", "m"]
    assert np.allclose(coef, [0.1, 0.3, -0.2])


def test_factor_stats_extremes():
    rng = np.random.default_rng(5)
    f = rng.normal(size=500)
    same = np.vstack([f, 2 * f, 3 * f])
    s = R.factor_stats(same)
    assert np.isclose(s["mean_abs_corr"], 1.0) and np.isclose(s["pc1_share"], 1.0)
    indep = rng.normal(size=(20, 5000))
    assert R.factor_stats(indep)["mean_abs_corr"] < 0.05


def test_dry_run_paths_never_point_at_oos_locations():
    locked, report = R.dirs("test")
    assert R.LOCKED_DIR not in (locked, *locked.parents)
    assert report != R.REPORT_DIR
