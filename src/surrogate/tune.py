"""Optuna tuning inside the warm-up only (CLAUDE.md rule 5, design 6.4).

GroupKFold (groups = window-free skeleton) on the warm-up, the same trial budget for
every model, objective = mean validation RMSE. The best parameters are refit on the
whole warm-up.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import optuna
from sklearn.model_selection import GroupKFold

from src.surrogate.models import make_model, suggest_params

optuna.logging.set_verbosity(optuna.logging.WARNING)


def cv_rmse(
    name: str,
    params: dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_folds: int,
    seed: int,
) -> float:
    errs = []
    for tr, va in GroupKFold(n_splits=n_folds).split(X, y, groups):
        model = make_model(name, params, seed)
        model.fit(X[tr], y[tr])
        pred = model.predict(X[va])
        errs.append(float(np.sqrt(np.mean((pred - y[va]) ** 2))))
    return float(np.mean(errs))


def tune_and_fit(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_trials: int,
    n_folds: int,
    seed: int,
):
    """Return (fitted model on full warm-up, best params, best CV RMSE)."""
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(
        lambda t: cv_rmse(name, suggest_params(name, t), X, y, groups, n_folds, seed),
        n_trials=n_trials,
    )
    best = study.best_params
    model = make_model(name, best, seed)
    model.fit(X, y)
    return model, best, float(study.best_value)
