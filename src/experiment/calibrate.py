"""IS-threshold calibration by random-arm simulation (design 5.3), IS/Test only.

Procedure (fixed before seeing any arm or OOS result):
  1. Scan the directed IS Sharpe threshold over [0.3, 1.5] step 0.05. In each of 200
     simulations remove a random warm-up of W formulas, randomly order the rest and take
     the first B. Pick the highest threshold whose mean unique-pass count (criteria 1-4
     within the B picks) is >= 22.
  2. If even 0.3 falls short, relax the Test threshold 0.3 -> 0.0 and repeat.
  3. If still short, stop and report the bottleneck. Generation rules and the 0.7
     correlation cap are never changed.
Then, at the chosen thresholds, the same simulations give the random baselines: mean
unique passes at B (enrichment denominator), at the top k, and at each discovery budget.

Greedy de-duplication within a set runs once on the loosest threshold: candidates are
processed in descending IS Sharpe, so a stricter threshold's adopted set is a prefix.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from src.backtest.engine import output_path
from src.common.config import load_experiment_config, setup_logging
from src.common.seeds import rng_for
from src.experiment.criteria import (
    PnlStore,
    Thresholds,
    count_unique_passes,
    directed,
    greedy_unique,
    pass_1to3,
)


def random_orders(market: str, eval_idx: np.ndarray, n_sims: int, warmup: int) -> list[np.ndarray]:
    """Per simulation: the eval set minus a random warm-up, in random selection order."""
    orders = []
    for s in range(n_sims):
        rng = rng_for("calibration", market, s)
        perm = rng.permutation(eval_idx)
        orders.append(perm[warmup:])  # first `warmup` = warm-up; rest in random order
    return orders


def unique_by_threshold(
    sel: np.ndarray, d: pd.DataFrame, pnl: PnlStore, grid: np.ndarray, test_thr: float, pc: dict
) -> np.ndarray:
    """Unique-pass counts within `sel` for every IS threshold in `grid` (prefix property)."""
    loosest = Thresholds(float(grid.min()), test_thr, pc["max_is_turnover"], pc["max_pnl_corr"])
    cand = sel[pass_1to3(d.iloc[sel], loosest)]
    is_sharpe = d["is_sharpe"].to_numpy()
    adopted = cand[greedy_unique(cand, is_sharpe, pnl, pc["max_pnl_corr"])]
    a_sharpe = is_sharpe[adopted]
    return np.array([(a_sharpe >= h).sum() for h in grid])


def calibrate(market: str) -> dict:
    cfg = load_experiment_config()
    pc, ex = cfg["pass_criteria"], cfg["experiment"]
    cal = pc["calibration"]
    logger = setup_logging(f"calibrate_{market}")
    t0 = time.perf_counter()

    results = pd.read_parquet(output_path("results_is_test", market))
    d = directed(results)
    pnl = PnlStore(np.load(output_path("pnl_is", market), mmap_mode="r"), d["direction"].to_numpy())
    eval_idx = np.flatnonzero(results["filter_pass"].to_numpy())  # JP: market filter
    lo, hi = pc["is_sharpe"]["search_range"]
    step = pc["is_sharpe"]["search_step"]
    grid = np.round(np.arange(lo, hi + step / 2, step), 4)
    B, k = ex["budget"], ex["precision_k"]
    orders = random_orders(market, eval_idx, cal["n_simulations"], ex["warmup_size"])
    need = cal["min_random_unique_mean"]

    out: dict = {"market": market, "n_eval": int(len(eval_idx)), "rule": cal, "steps": []}
    chosen = None
    for step_no, test_thr in ((1, pc["test_sharpe"]), (2, pc["test_sharpe_fallback"])):
        counts = np.stack([unique_by_threshold(o[:B], d, pnl, grid, test_thr, pc) for o in orders])
        mean = counts.mean(axis=0)
        table = {
            f"{h:.2f}": {
                "mean": round(float(m), 2),
                "min": int(c.min()),
                "p5": float(np.percentile(c, 5)),
            }
            for h, m, c in zip(grid, mean, counts.T, strict=True)
        }
        out["steps"].append({"step": step_no, "test_sharpe": test_thr, "random_unique": table})
        logger.info(
            "step %d (test >= %.2f): mean unique at %.2f = %.1f",
            step_no,
            test_thr,
            grid[0],
            mean[0],
        )
        ok = np.flatnonzero(mean >= need)
        if len(ok):
            chosen = (step_no, test_thr, float(grid[ok.max()]))
            break

    if chosen is None:
        out["status"] = "stopped: random-arm unique passes below target even after Test fallback"
    else:
        step_no, test_thr, is_thr = chosen
        th = Thresholds(is_thr, test_thr, pc["max_is_turnover"], pc["max_pnl_corr"])
        baselines: dict[str, float] = {}
        for b in sorted({*ex["discovery_budgets"], B, k}):
            vals = [count_unique_passes(o[:b], d, th, pnl)[0] for o in orders]
            baselines[str(b)] = float(np.mean(vals))
        n_pool_unique, _ = count_unique_passes(eval_idx, d, th, pnl)
        out.update(
            status="ok",
            decided_at_step=step_no,
            thresholds={
                "is_sharpe": is_thr,
                "test_sharpe": test_thr,
                "max_is_turnover": pc["max_is_turnover"],
                "max_pnl_corr": pc["max_pnl_corr"],
            },
            random_baseline_unique=baselines,  # by evaluation-set size
            enrichment_denominator=baselines[str(B)] / B,
            pool_base_rate=n_pool_unique / len(eval_idx),  # reference only
            pool_unique_passes=int(n_pool_unique),
            pool_pass_1to3_rate=float(pass_1to3(d.iloc[eval_idx], th).mean()),
        )
    out["elapsed_sec"] = round(time.perf_counter() - t0, 1)

    path = output_path("thresholds", market)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    logger.info("calibration %s: %s", market, {k_: v for k_, v in out.items() if k_ != "steps"})
    if out["status"] != "ok":
        raise SystemExit(f"calibration stopped ({out['status']}); see {path}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market", required=True, choices=["us", "jp"])
    args = parser.parse_args()
    calibrate(args.market)


if __name__ == "__main__":
    main()
