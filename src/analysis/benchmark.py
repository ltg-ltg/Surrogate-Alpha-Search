"""Uniform-condition timing benchmark (design 8.5). Run alone, single process.

Measures, with every library limited to one thread:
  - environment (CPU, cores, RAM, OS, package versions)
  - backtest t_sim on 1,000 random formulas per market (full period and IS-only),
    and its ratio to the values recorded during the parallel full-pool backtest
  - feature extraction for the whole pool (amortized, once per pool)
  - per seed (0-2) and per arm / variant: residual target + tuning + fit, inference on
    all candidates, MMR (lambda 0.5); predictions are checked against the saved ones
Outputs under artifacts/benchmark/.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"

import ctypes  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from importlib.metadata import version  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.common.config import (  # noqa: E402
    ARTIFACTS_DIR,
    PROJECT_ROOT,
    load_experiment_config,
    setup_logging,  # noqa: E402
)
from src.common.seeds import rng_for  # noqa: E402

OUT = ARTIFACTS_DIR / "benchmark"
SEEDS = (0, 1, 2)
N_BACKTEST = 1000


def environment() -> dict:
    def ps(cmd: str) -> str:
        try:
            return subprocess.run(
                ["powershell", "-NoProfile", "-Command", cmd],
                capture_output=True,
                text=True,
                timeout=30,
            ).stdout.strip()
        except Exception:  # noqa: BLE001
            return ""

    class MemStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    ram_gb = None
    if sys.platform == "win32":
        m = MemStatus()
        m.dwLength = ctypes.sizeof(MemStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        ram_gb = round(m.ullTotalPhys / 2**30, 1)
    pkgs = [
        "numpy",
        "pandas",
        "numba",
        "scikit-learn",
        "lightgbm",
        "xgboost",
        "catboost",
        "optuna",
        "scipy",
        "pyarrow",
        "yfinance",
    ]
    return {
        "cpu": ps("(Get-CimInstance Win32_Processor).Name") or platform.processor(),
        "physical_cores": ps("(Get-CimInstance Win32_Processor).NumberOfCores"),
        "logical_cores": os.cpu_count(),
        "ram_gb": ram_gb,
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "packages": {p: version(p) for p in pkgs},
        "threads_per_library": 1,
        "processes": 1,
    }


def bench_backtest(market: str, logger) -> pd.DataFrame:
    from src.backtest import engine
    from src.backtest.metrics import period_metrics
    from src.generator.grammar import parse

    engine._init_worker(market)
    W = engine._W
    cfg = load_experiment_config()
    pool = pd.read_parquet(PROJECT_ROOT / cfg["pool"]["file"])
    rec = pd.read_parquet(engine.output_path("results_is_test", market))
    ids = np.sort(rng_for("benchmark", market).choice(len(pool), N_BACKTEST, replace=False))
    ann = W["bt"]["annualization"]
    for text in pool["formula"].iloc[ids[:5]]:  # JIT warm-up
        engine._backtest(W["panel"], parse(text))
    rows = []
    for fid in ids:
        node = parse(pool.at[fid, "formula"])
        t0 = time.perf_counter()
        engine._backtest(W["panel"], node)
        t_full = time.perf_counter() - t0
        t0 = time.perf_counter()
        _, s = engine._backtest(W["trunc"], node)
        for d in (1.0, -1.0):
            period_metrics(s, W["trunc_is_rows"], d, ann)
        t_is = time.perf_counter() - t0
        rows.append(
            {
                "market": market,
                "formula_id": int(fid),
                "t_full": t_full,
                "t_is": t_is,
                "rec_full": float(rec.at[fid, "time_full_sec"]),
                "rec_is": float(rec.at[fid, "time_is_sec"]),
            }
        )
    df = pd.DataFrame(rows)
    logger.info(
        "%s backtest: median t_full %.3fs t_is %.3fs; ratio full %.3f is %.3f",
        market,
        df.t_full.median(),
        df.t_is.median(),
        (df.t_full / df.rec_full).median(),
        (df.t_is / df.rec_is).median(),
    )
    return df


def bench_features(logger) -> dict:
    from src.features.ast_features import build_features
    from src.generator.grammar import load_grammar

    cfg = load_experiment_config()
    formulas = pd.read_parquet(PROJECT_ROOT / cfg["pool"]["file"])["formula"].tolist()
    g = load_grammar()
    out = {}
    for ext in (False, True):
        t0 = time.perf_counter()
        build_features(formulas, g, ext)
        out["ext" if ext else "base"] = time.perf_counter() - t0
    logger.info("feature extraction (whole pool): %s", out)
    return out


def bench_fits(logger) -> pd.DataFrame:
    import src.experiment.replay as R
    from src.common.seeds import derive_seed
    from src.experiment.accessor import ResultAccessor
    from src.experiment.arms import fit_arm, rank_candidates
    from src.experiment.mmr import mmr_select, percentile_rank, unit_feature_vectors

    cfg = load_experiment_config()
    ex, scfg = cfg["experiment"], cfg["surrogate"]
    W, B, lam = ex["warmup_size"], ex["budget"], cfg["diversity"]["primary_lambda"]
    feats = pd.read_parquet(ARTIFACTS_DIR / "features_v1.parquet")
    cols = json.loads((ARTIFACTS_DIR / "feature_schema_v1.json").read_text("utf-8"))["columns"]
    U = unit_feature_vectors(feats[cols].to_numpy(dtype=np.float64))
    rows = []
    for schema, arms in (
        ("base", ("heuristic", "lightgbm", "xgboost", "catboost", "random_forest")),
        ("ext", ("lightgbm", "xgboost", "catboost", "random_forest")),
    ):
        R._init_worker("us", schema)
        us = R._W["us"]
        for target in ("raw", "resid"):
            variant = schema if target == "raw" else f"{schema}_resid"
            preds_saved = pd.read_parquet(
                R.run_dir("us", variant) / "predictions.parquet",
                filters=[("seed", "in", list(SEEDS))],
            )
            for seed in SEEDS:
                warm = R._warmup_ids(seed)
                cand = np.setdiff1d(us["eval_ids"], warm)
                for arm in arms:
                    acc = ResultAccessor(us["d"], us["pnl"], us["eval_ids"], budget=W + B)
                    t0 = time.perf_counter()
                    y = acc.warmup_targets(warm)
                    if target == "resid":
                        y = R._residual_targets(acc, warm, y, us["th"])
                    t_resid = time.perf_counter() - t0
                    fitted = fit_arm(
                        arm,
                        R._W["X"][warm],
                        y,
                        R._W["groups"][warm],
                        [R._W["comps"][i] for i in warm] if R._W["comps"] else None,
                        derive_seed("model", schema, seed, arm),
                        scfg,
                    )
                    order, scores, t_pred = rank_candidates(
                        fitted,
                        cand,
                        R._W["X"][cand],
                        [R._W["comps"][i] for i in cand] if R._W["comps"] else None,
                        rng_for("random_order", "us", seed),
                    )
                    t0 = time.perf_counter()
                    mmr_select(cand, percentile_rank(scores), U, lam, B)
                    t_mmr = time.perf_counter() - t0
                    saved = preds_saved.query("seed == @seed and arm == @arm")
                    saved = saved.set_index("formula_id")["score"].reindex(cand).to_numpy()
                    diff = float(np.nanmax(np.abs(saved - scores.astype(np.float32))))
                    rows.append(
                        {
                            "schema": schema,
                            "target": target,
                            "seed": seed,
                            "arm": arm,
                            "t_target": t_resid,
                            "t_fit": fitted.fit_sec,
                            "t_predict": t_pred,
                            "t_mmr": t_mmr,
                            "max_abs_pred_diff": diff,
                        }
                    )
                    logger.info(
                        "%s/%s seed %d %s: fit %.1fs predict %.2fs mmr %.2fs pred diff %.2e",
                        schema,
                        target,
                        seed,
                        arm,
                        fitted.fit_sec,
                        t_pred,
                        t_mmr,
                        diff,
                    )
    return pd.DataFrame(rows)


def main() -> None:
    logger = setup_logging("benchmark")
    OUT.mkdir(parents=True, exist_ok=True)
    env = environment()
    (OUT / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    logger.info("environment: %s", env)
    bt = pd.concat([bench_backtest(m, logger) for m in ("us", "jp")], ignore_index=True)
    bt.to_parquet(OUT / "backtest_timing.parquet", index=False)
    feat = bench_features(logger)
    (OUT / "feature_timing.json").write_text(json.dumps(feat, indent=2), encoding="utf-8")
    fits = bench_fits(logger)
    fits.to_parquet(OUT / "fit_timing.parquet", index=False)
    logger.info("benchmark done")


if __name__ == "__main__":
    main()
