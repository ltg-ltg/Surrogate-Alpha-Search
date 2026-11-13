"""Replay experiment (design 7.2, 7.4, 9).

  --market us --schema base   30 seeds x 6 arms (H1, H3). Saves fitted arms for JP.
  --market us --schema ext    30 seeds x 4 models, extended features (ablation 7.4).
  --market jp                 US-trained arms applied to the JP pool, no JP warm-up (9).

Outputs under artifacts/replay/<market>_<schema>/:
  metrics.parquet      one row per (seed, arm): unique passes per budget, Precision@k,
                       enrichment, timings
  selections.parquet   ordered picks per (seed, arm) with criterion-1..4 adoption flag
  predictions.parquet  candidate scores per (seed, arm) for the Spearman analysis
OOS results are never read here.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from src.backtest.engine import output_path
from src.common.config import ARTIFACTS_DIR, PROJECT_ROOT, load_experiment_config, setup_logging
from src.common.seeds import derive_seed, rng_for
from src.experiment.accessor import ResultAccessor
from src.experiment.arms import ARM_NAMES, fit_arm, rank_candidates
from src.experiment.criteria import PnlStore, Thresholds, count_unique_passes, directed
from src.surrogate.models import MODEL_NAMES

_W: dict = {}


def run_dir(market: str, schema: str):
    return ARTIFACTS_DIR / "replay" / f"{market}_{schema}"


def model_path(schema: str, seed: int, arm: str):
    return ARTIFACTS_DIR / "replay" / "models" / schema / f"seed{seed:02d}_{arm}.pkl"


# ---------------------------------------------------------------- worker state


def _load_market(market: str) -> dict:
    results = pd.read_parquet(output_path("results_is_test", market))
    d = directed(results)
    pnl = PnlStore(np.load(output_path("pnl_is", market), mmap_mode="r"), d["direction"].to_numpy())
    th_json = json.loads(output_path("thresholds", market).read_text(encoding="utf-8"))
    if th_json["status"] != "ok":
        raise RuntimeError(f"{market} thresholds not calibrated")
    return {
        "d": d,
        "pnl": pnl,
        "eval_ids": np.flatnonzero(results["filter_pass"].to_numpy()),
        "time_full": results["time_full_sec"].to_numpy(),
        "time_is": results["time_is_sec"].to_numpy(),
        "th": Thresholds(**th_json["thresholds"]),
        "baseline": th_json["random_baseline_unique"],
    }


def _init_worker(market: str, schema: str) -> None:
    from src.generator.grammar import load_grammar
    from src.surrogate.heuristic import components

    cfg = load_experiment_config()
    suffix = "" if schema == "base" else "_ext"
    feats = pd.read_parquet(ARTIFACTS_DIR / f"features_v1{suffix}.parquet")
    schema_cols = json.loads(
        (ARTIFACTS_DIR / f"feature_schema_v1{suffix}.json").read_text(encoding="utf-8")
    )["columns"]
    pool = pd.read_parquet(PROJECT_ROOT / cfg["pool"]["file"])
    assert np.array_equal(feats["formula_id"].to_numpy(), np.arange(len(pool)))
    g = load_grammar()
    _W.update(
        cfg=cfg,
        X=feats[schema_cols].to_numpy(dtype=np.float64),
        groups=pool["skeleton"].to_numpy(),
        comps=[components(t, g) for t in pool["formula"]] if schema == "base" else None,
        us=_load_market("us"),
        jp=_load_market("jp") if market == "jp" else None,
    )


# ---------------------------------------------------------------- evaluation


def evaluate_selection(view, th: Thresholds, budgets: list[int], k: int) -> dict:
    """Unique passes (criteria 1-4) for each prefix of the arm's ordered selection."""
    local = view.results
    out = {}
    adopted_b = None
    for b in sorted({*budgets, k}):
        n, adopted = count_unique_passes(np.arange(min(b, len(local))), local, th, view.pnl)
        out[f"unique_at_{b}"] = n
        if b == max(budgets):
            adopted_b = adopted
    out["precision_at_k"] = out[f"unique_at_{k}"] / k
    flags = np.zeros(len(local), dtype=bool)
    flags[adopted_b] = True
    return out, flags


def _warmup_ids(seed: int) -> np.ndarray:
    us = _W["us"]
    rng = rng_for("warmup", seed)
    return np.sort(
        rng.choice(us["eval_ids"], _W["cfg"]["experiment"]["warmup_size"], replace=False)
    )


def _run_seed(
    args: tuple[int, str, str],
) -> tuple[list[dict], list[pd.DataFrame], list[pd.DataFrame]]:
    seed, market, schema = args
    cfg = _W["cfg"]
    ex, scfg = cfg["experiment"], cfg["surrogate"]
    W, B, k = ex["warmup_size"], ex["budget"], ex["precision_k"]
    us = _W["us"]
    mk = _W[market]
    arms = ARM_NAMES if schema == "base" else MODEL_NAMES
    warm = _warmup_ids(seed)

    if market == "us":
        cand_ids = np.setdiff1d(us["eval_ids"], warm)
    else:
        cand_ids = mk["eval_ids"]  # US warm-up formulas are not excluded (design 9.3)

    rows, sels, preds = [], [], []
    for arm_name in arms:
        if market == "us":
            acc = ResultAccessor(us["d"], us["pnl"], us["eval_ids"], budget=W + B)
            y_warm = acc.warmup_targets(warm)
            fitted = fit_arm(
                arm_name,
                _W["X"][warm],
                y_warm,
                _W["groups"][warm],
                [_W["comps"][i] for i in warm] if _W["comps"] else None,
                derive_seed("model", schema, seed, arm_name),
                scfg,
            )
            path = model_path(schema, seed, arm_name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(pickle.dumps(fitted))
            backtest_sec = float(us["time_is"][warm].sum())
        else:
            acc = ResultAccessor(mk["d"], mk["pnl"], mk["eval_ids"], budget=B)
            fitted = pickle.loads(model_path(schema, seed, arm_name).read_bytes())
            backtest_sec = 0.0

        order, scores, predict_sec = rank_candidates(
            fitted,
            cand_ids,
            _W["X"][cand_ids],
            [_W["comps"][i] for i in cand_ids] if _W["comps"] else None,
            rng_for("random_order", market, seed),
        )
        picks = order[:B]
        acc.register_selection(picks)
        view = acc.selected_view()
        metrics, adopted = evaluate_selection(view, mk["th"], ex["discovery_budgets"], k)
        backtest_sec += float(mk["time_full"][picks].sum())
        fit_sec = fitted.fit_sec if market == "us" else 0.0
        rows.append(
            {
                "seed": seed,
                "arm": arm_name,
                **metrics,
                "enrichment": metrics[f"unique_at_{B}"] / mk["baseline"][str(B)],
                "n_pass_1to3": int(
                    (
                        (view.results["is_sharpe"] >= mk["th"].is_sharpe)
                        & (view.results["test_sharpe"] >= mk["th"].test_sharpe)
                        & (view.results["is_turnover"] <= mk["th"].max_is_turnover)
                    ).sum()
                ),
                "queries": acc.queries,
                "time_backtest_sec": backtest_sec,
                "time_fit_sec": fit_sec,
                "time_predict_sec": predict_sec,
                "time_total_sec": backtest_sec + fit_sec + predict_sec,
                "cv_rmse": fitted.info.get("cv_rmse", np.nan),
                "params": json.dumps(fitted.info.get("params", {})),
            }
        )
        sels.append(
            pd.DataFrame(
                {
                    "seed": seed,
                    "arm": arm_name,
                    "rank": np.arange(B),
                    "formula_id": picks,
                    "adopted": adopted,
                }
            )
        )
        if scores is not None:
            preds.append(
                pd.DataFrame(
                    {
                        "seed": seed,
                        "arm": arm_name,
                        "formula_id": cand_ids,
                        "score": scores.astype(np.float32),
                    }
                )
            )
    return rows, sels, preds


def run(market: str, schema: str, workers: int, seeds: list[int]) -> None:
    logger = setup_logging(f"replay_{market}_{schema}")
    if market == "jp" and schema != "base":
        raise ValueError("JP transfer uses the base schema only (design 7.4)")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    t0 = time.perf_counter()
    rows, sels, preds = [], [], []
    with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(market, schema)) as ex:
        for r, s, p in ex.map(_run_seed, [(sd, market, schema) for sd in seeds]):
            rows += r
            sels += s
            preds += p
            logger.info(
                "seed %d done (%.0fs): %s",
                r[0]["seed"],
                time.perf_counter() - t0,
                {x["arm"]: x["unique_at_2000"] for x in r},
            )
    out = run_dir(market, schema)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out / "metrics.parquet", index=False)
    pd.concat(sels).to_parquet(out / "selections.parquet", index=False)
    if preds:
        pd.concat(preds).to_parquet(out / "predictions.parquet", index=False)
    logger.info(
        "replay %s/%s: %d seeds in %.0fs -> %s",
        market,
        schema,
        len(seeds),
        time.perf_counter() - t0,
        out,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market", required=True, choices=["us", "jp"])
    parser.add_argument("--schema", default="base", choices=["base", "ext"])
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    args = parser.parse_args()
    n = load_experiment_config()["experiment"]["n_seeds"]
    run(args.market, args.schema, args.workers, args.seeds if args.seeds else list(range(n)))


if __name__ == "__main__":
    main()
