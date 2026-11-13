"""Arms: random, component heuristic, and the four surrogate models (design 7.1).

An arm only sees warm-up features / targets (handed over by the accessor) and candidate
features; it returns a full ranking of the candidates. Batch selection: the first B of
the ranking are backtested; no retraining in between.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.surrogate.heuristic import ComponentHeuristic
from src.surrogate.models import MODEL_NAMES
from src.surrogate.tune import tune_and_fit

ARM_NAMES = ("random", "heuristic", *MODEL_NAMES)


@dataclass
class FittedArm:
    name: str
    model: Any  # fitted estimator, ComponentHeuristic, or None for random
    fit_sec: float
    info: dict


def fit_arm(
    name: str,
    X_warm: np.ndarray,
    y_warm: np.ndarray,
    groups_warm: np.ndarray,
    comps_warm: list,
    seed: int,
    scfg: dict,
) -> FittedArm:
    t0 = time.perf_counter()
    if name == "random":
        return FittedArm(name, None, 0.0, {})
    if name == "heuristic":
        h = ComponentHeuristic(m=scfg["heuristic_smoothing_m"]).fit(comps_warm, y_warm)
        return FittedArm(name, h, time.perf_counter() - t0, {})
    model, params, cv = tune_and_fit(
        name, X_warm, y_warm, groups_warm, scfg["optuna_trials"], scfg["cv_folds"], seed
    )
    return FittedArm(name, model, time.perf_counter() - t0, {"params": params, "cv_rmse": cv})


def rank_candidates(
    arm: FittedArm,
    cand_ids: np.ndarray,
    X_cand: np.ndarray | None,
    comps_cand: list | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray | None, float]:
    """Return (candidate ids in selection order, scores aligned with cand_ids, seconds)."""
    t0 = time.perf_counter()
    if arm.name == "random":
        order = rng.permutation(cand_ids)
        return order, None, time.perf_counter() - t0
    if arm.name == "heuristic":
        scores = arm.model.predict(comps_cand)
    else:
        scores = np.asarray(arm.model.predict(X_cand), dtype=np.float64)
    order = cand_ids[np.lexsort((cand_ids, -scores))]  # score desc, ties by id
    return order, scores, time.perf_counter() - t0
