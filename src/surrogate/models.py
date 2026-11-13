"""Tree-based surrogate regressors and their Optuna search spaces (design 6.4, CPU only).

Every model runs single-threaded (n_jobs = 1); parallelism comes from running seeds in
separate processes, which also keeps the measured training time comparable across arms.
"""

from __future__ import annotations

from typing import Any

import optuna

MODEL_NAMES = ("lightgbm", "xgboost", "catboost", "random_forest")


def suggest_params(name: str, trial: optuna.Trial) -> dict[str, Any]:
    if name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 128, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
    if name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
    if name == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 100, 1000, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "depth": trial.suggest_int("depth", 3, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
        }
    if name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 30, log=True),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20, log=True),
            "max_features": trial.suggest_float("max_features", 0.2, 1.0),
        }
    raise ValueError(f"unknown model {name!r}")


def make_model(name: str, params: dict[str, Any], seed: int):
    if name == "lightgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            **params,
            subsample_freq=1,
            random_state=seed,
            n_jobs=1,
            deterministic=True,
            force_row_wise=True,
            verbose=-1,
        )
    if name == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(**params, tree_method="hist", random_state=seed, n_jobs=1)
    if name == "catboost":
        from catboost import CatBoostRegressor

        return CatBoostRegressor(
            **params, random_seed=seed, thread_count=1, verbose=0, allow_writing_files=False
        )
    if name == "random_forest":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(**params, random_state=seed, n_jobs=1)
    raise ValueError(f"unknown model {name!r}")
