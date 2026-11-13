"""Post-hoc exploratory replay: diversity-aware MMR selection (design 7.5).

Re-ranks the predictions saved by `replay` (no retraining) with MMR for each lambda,
plus a structure-stratified random control. Same seeds, warm-ups and budgets as 7.2.

Output: artifacts/replay/<market>_mmr/ (base schema) or us_ext_mmr/ (--schema ext:
models only, primary lambda, similarity still from base features).
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from src.common.config import ARTIFACTS_DIR, load_experiment_config, setup_logging
from src.common.seeds import rng_for
from src.experiment.accessor import ResultAccessor
from src.experiment.mmr import mmr_select, percentile_rank, unit_feature_vectors
from src.experiment.replay import _load_market, evaluate_selection, run_dir

_W: dict = {}


def _init_worker(market: str, schema: str, target: str) -> None:
    src = schema if target == "raw" else f"{schema}_resid"
    cfg = load_experiment_config()
    feats = pd.read_parquet(ARTIFACTS_DIR / "features_v1.parquet")
    cols = json.loads((ARTIFACTS_DIR / "feature_schema_v1.json").read_text(encoding="utf-8"))[
        "columns"
    ]
    assert np.array_equal(feats["formula_id"].to_numpy(), np.arange(len(feats)))
    _W.update(
        cfg=cfg,
        U=unit_feature_vectors(feats[cols].to_numpy(dtype=np.float64)),
        us=_load_market("us"),
        mk=_load_market(market),
        schema=schema,
        # Similarity always uses the base features; only predictions follow the schema.
        base_metrics=pd.read_parquet(run_dir(market, src) / "metrics.parquet"),
        target=target,
        pred_path=run_dir(market, src) / "predictions.parquet",
    )


def _warmup_ids(seed: int) -> np.ndarray:
    rng = rng_for("warmup", seed)
    return np.sort(
        rng.choice(_W["us"]["eval_ids"], _W["cfg"]["experiment"]["warmup_size"], replace=False)
    )


def _run_seed(args: tuple[int, str]) -> tuple[list[dict], list[pd.DataFrame]]:
    seed, market = args
    cfg = _W["cfg"]
    ex, dv = cfg["experiment"], cfg["diversity"]
    W, B, k = ex["warmup_size"], ex["budget"], ex["precision_k"]
    us, mk = _W["us"], _W["mk"]
    warm = _warmup_ids(seed)
    preds = pd.read_parquet(_W["pred_path"], filters=[("seed", "=", seed)])
    base = _W["base_metrics"].query("seed == @seed").set_index("arm")

    if _W["schema"] == "ext":  # 7.5 addendum: models only, primary lambda only
        arms = [a for a in dv["arms"] if a not in ("heuristic", "struct_random")]
        lambdas = [dv["primary_lambda"]]
    else:
        arms, lambdas = dv["arms"], dv["lambdas"]
    if _W["target"] == "resid":  # struct_random has no target: reuse the raw run
        arms = [a for a in arms if a != "struct_random"]
    rows, sels = [], []
    for arm in arms:
        if arm == "struct_random":
            cand_ids = np.setdiff1d(us["eval_ids"], warm) if market == "us" else mk["eval_ids"]
            rel = rng_for("struct_random", market, seed).uniform(size=len(cand_ids))
            fit_sec = 0.0
        else:
            p = preds[preds["arm"] == arm]
            cand_ids = p["formula_id"].to_numpy()
            rel = percentile_rank(p["score"].to_numpy(dtype=np.float64))
            fit_sec = float(base.at[arm, "time_fit_sec"])
        for lam in lambdas:
            if market == "us":
                acc = ResultAccessor(us["d"], us["pnl"], us["eval_ids"], budget=W + B)
                acc.warmup_targets(warm)  # same budget accounting as 7.2 (all arms)
                backtest_sec = float(us["time_is"][warm].sum())
            else:
                acc = ResultAccessor(mk["d"], mk["pnl"], mk["eval_ids"], budget=B)
                backtest_sec = 0.0
            t0 = time.perf_counter()
            picks = mmr_select(cand_ids, rel, _W["U"], lam, B)
            select_sec = time.perf_counter() - t0
            acc.register_selection(picks)
            view = acc.selected_view()
            metrics, adopted = evaluate_selection(view, mk["th"], ex["discovery_budgets"], k)
            backtest_sec += float(mk["time_full"][picks].sum())
            rows.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "lam": lam,
                    "primary": lam == dv["primary_lambda"],
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
                    "time_fit_sec": fit_sec if market == "us" else 0.0,
                    "time_select_sec": select_sec,
                }
            )
            sels.append(
                pd.DataFrame(
                    {
                        "seed": seed,
                        "arm": arm,
                        "lam": lam,
                        "rank": np.arange(B),
                        "formula_id": picks,
                        "adopted": adopted,
                    }
                )
            )
    return rows, sels


def run(market: str, schema: str, workers: int, seeds: list[int], target: str = "raw") -> None:
    logger = setup_logging(f"replay_mmr_{market}_{schema}")
    t0 = time.perf_counter()
    rows, sels = [], []
    with ProcessPoolExecutor(
        workers, initializer=_init_worker, initargs=(market, schema, target)
    ) as ex:
        for r, s in ex.map(_run_seed, [(sd, market) for sd in seeds]):
            rows += r
            sels += s
            primary = {x["arm"]: x["unique_at_2000"] for x in r if x["primary"]}
            logger.info("seed %d done (%.0fs): %s", r[0]["seed"], time.perf_counter() - t0, primary)
    out = run_dir(
        market, ("mmr" if schema == "base" else "ext_mmr") + ("" if target == "raw" else "_resid")
    )
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out / "metrics.parquet", index=False)
    pd.concat(sels).to_parquet(out / "selections.parquet", index=False)
    logger.info(
        "mmr replay %s: %d seeds in %.0fs -> %s", market, len(seeds), time.perf_counter() - t0, out
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market", required=True, choices=["us", "jp"])
    parser.add_argument("--schema", default="base", choices=["base", "ext"])
    parser.add_argument("--target", default="raw", choices=["raw", "resid"])
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    args = parser.parse_args()
    n = load_experiment_config()["experiment"]["n_seeds"]
    seeds = args.seeds if args.seeds else list(range(n))
    run(args.market, args.schema, args.workers, seeds, args.target)


if __name__ == "__main__":
    main()
