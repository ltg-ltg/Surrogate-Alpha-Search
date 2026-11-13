"""Accessor rules (CLAUDE.md rule 4): only own warm-up / selection, exact budget count."""

import numpy as np
import pandas as pd
import pytest

from src.experiment.accessor import AccessError, ResultAccessor
from src.experiment.criteria import PnlStore

N, T = 50, 30


def make(budget=15, eval_ids=None):
    rng = np.random.default_rng(0)
    d = pd.DataFrame(
        {
            "formula_id": np.arange(N),
            "direction": 1,
            "is_sharpe": rng.normal(size=N),
            "test_sharpe": rng.normal(size=N),
            "is_turnover": rng.uniform(0, 1, N),
        }
    )
    gc = rng.normal(size=(2, N, T)).astype(np.float32)
    ids = np.arange(N) if eval_ids is None else eval_ids
    return ResultAccessor(d, PnlStore(gc, np.ones(N)), ids, budget), d


def test_budget_counts_exactly():
    acc, d = make(budget=15)
    y = acc.warmup_targets([0, 1, 2, 3, 4])
    assert acc.queries == 5
    np.testing.assert_array_equal(y, d["is_sharpe"].to_numpy()[:5])
    acc.register_selection([10, 11, 12])
    assert acc.queries == 5  # registering is free; releasing is charged
    view = acc.selected_view()
    assert acc.queries == 8
    assert view.results["formula_id"].tolist() == [10, 11, 12]
    assert view.pnl.gross.shape == (3, T)


def test_only_selected_results_are_released():
    acc, _ = make()
    acc.register_selection([7, 3])
    view = acc.selected_view()
    assert set(view.results["formula_id"]) == {3, 7}
    with pytest.raises(AccessError):
        acc.selected_view()  # no second release
    with pytest.raises(AccessError):
        acc.register_selection([8])  # selection is fixed once


def test_budget_exceeded():
    acc, _ = make(budget=4)
    acc.warmup_targets([0, 1, 2])
    acc.register_selection([5, 6])
    with pytest.raises(AccessError):
        acc.selected_view()


def test_selection_cannot_overlap_warmup_or_leave_pool():
    acc, _ = make(eval_ids=np.arange(40))
    acc.warmup_targets([0, 1])
    with pytest.raises(AccessError):
        acc.register_selection([1, 5])
    acc2, _ = make(eval_ids=np.arange(40))
    with pytest.raises(AccessError):
        acc2.register_selection([45])  # outside the evaluation pool (e.g. JP filter)
    with pytest.raises(AccessError):
        acc2.selected_view()  # nothing registered


def test_warmup_drawn_once_and_no_duplicates():
    acc, _ = make()
    with pytest.raises(AccessError):
        acc.warmup_targets([1, 1])
    acc.warmup_targets([1, 2])
    with pytest.raises(AccessError):
        acc.warmup_targets([3])


def test_arms_module_never_touches_results_files():
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src"
    for name in (
        "experiment/arms.py",
        "surrogate/models.py",
        "surrogate/tune.py",
        "surrogate/heuristic.py",
    ):
        text = (src / name).read_text(encoding="utf-8")
        assert "results_" not in text and "pnl_" not in text and "read_parquet" not in text
