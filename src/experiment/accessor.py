"""Budgeted result accessor (CLAUDE.md rule 4, design 7.2).

Arms never receive the result table. Within one (seed, arm, market) run:
  - warmup_targets(ids): directed IS net Sharpe of warm-up formulas only (IS backtest).
  - register_selection(ids): the arm's ordered picks, fixed once.
  - selected_view(): IS/Test results and IS PnL of the registered picks only.
Every formula whose result is released is charged once against the budget (W + B in the
US experiment, B in the JP transfer where the warm-up comes from the US).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.experiment.criteria import PnlStore


class AccessError(RuntimeError):
    """Raised on any access outside an arm's own warm-up / selection or over budget."""


@dataclass(frozen=True)
class SelectedView:
    results: pd.DataFrame  # directed rows of the selection, in selection order
    pnl: PnlStore  # rows aligned with `results`


class ResultAccessor:
    def __init__(self, directed: pd.DataFrame, pnl: PnlStore, eval_ids: np.ndarray, budget: int):
        if not np.array_equal(directed["formula_id"].to_numpy(), np.arange(len(directed))):
            raise ValueError("results must be indexed by formula_id == row")
        self._d = directed
        self._pnl = pnl
        self._eval = np.zeros(len(directed), dtype=bool)
        self._eval[eval_ids] = True
        self.budget = budget
        self.queries = 0
        self._warmup: set[int] = set()
        self._selection: np.ndarray | None = None
        self._selection_released = False

    def _charge(self, n: int) -> None:
        if self.queries + n > self.budget:
            raise AccessError(f"budget exceeded: {self.queries} + {n} > {self.budget}")
        self.queries += n

    def _check_ids(self, ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(ids, dtype=np.int64)
        if len(np.unique(ids)) != len(ids):
            raise AccessError("duplicate formula ids")
        if ids.size and (ids.min() < 0 or ids.max() >= len(self._d) or not self._eval[ids].all()):
            raise AccessError("formula outside the evaluation pool")
        return ids

    def warmup_targets(self, ids) -> np.ndarray:
        """IS-only target for warm-up formulas (charged)."""
        ids = self._check_ids(ids)
        if self._warmup:
            raise AccessError("warm-up already drawn")
        self._charge(len(ids))
        self._warmup = set(ids.tolist())
        return self._d["is_sharpe"].to_numpy()[ids].copy()

    def register_selection(self, ids) -> None:
        ids = self._check_ids(ids)
        if self._selection is not None:
            raise AccessError("selection already registered")
        if self._warmup.intersection(ids.tolist()):
            raise AccessError("selection overlaps the warm-up")
        self._selection = ids

    def selected_view(self) -> SelectedView:
        """Results of the registered selection only (charged once)."""
        if self._selection is None:
            raise AccessError("no selection registered")
        if self._selection_released:
            raise AccessError("selection results already released")
        self._charge(len(self._selection))
        self._selection_released = True
        ids = self._selection
        res = self._d.iloc[ids].reset_index(drop=True)
        gc = np.stack([np.asarray(self._pnl.gross[ids]), np.asarray(self._pnl.cost[ids])])
        return SelectedView(res, PnlStore(gc, self._pnl.direction[ids]))
