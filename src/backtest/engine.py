"""Full-pool backtest (design 5, 7.2): every formula, both directions, every period.

Outputs ({market} = us | jp):
  results_{market}_is_test.parquet  IS/Test metrics for direction +1 and -1, timings,
                                    and (JP) the market-specific signal filter verdict
  results_{market}_oos.parquet      OOS metrics only. Read exclusively by oos_report.py.
  pnl_{market}_is.npy               float32 (2, n_formulas, n_is_days): [gross, cost]
                                    daily IS series for the PnL-correlation criterion
Direction is not decided here: metrics are stored for both signs so the direction rule
(src/experiment/criteria.py) is applied in one place.
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from src.common.config import PROJECT_ROOT, load_experiment_config, setup_logging

DIRECTIONS = {"pos": 1.0, "neg": -1.0}
PERIODS = ("is", "test", "oos")
_W: dict = {}


def output_path(key: str, market: str):
    return PROJECT_ROOT / load_experiment_config()["outputs"][key].format(market=market)


def _init_worker(market: str) -> None:
    import numba

    from src.data.panel import load_panel

    numba.set_num_threads(1)
    cfg = load_experiment_config()
    panel = load_panel(market)
    periods = cfg["periods"]
    # Panel truncated two trading days after IS end: what a warm-up (IS-only) run needs.
    is_end_row = panel.rows(periods["is"][0], periods["is"][1]).stop
    trunc = panel.truncated(is_end_row + cfg["backtest"]["delay_days"] + 1)
    _W.update(
        panel=panel,
        trunc=trunc,
        rows={p: panel.rows(*periods[p]) for p in PERIODS},
        trunc_is_rows=trunc.rows(*periods["is"]),
        filter_rows=panel.rows(periods["is"][0], periods["oos"][1]),
        bt=cfg["backtest"],
        fcfg=cfg["pool"]["signal_filter"],
    )


def _backtest(panel, node):
    from src.backtest.metrics import run_series
    from src.backtest.operators import evaluate

    bt = _W["bt"]
    signal = evaluate(node, panel)
    series = run_series(
        signal,
        panel.valid,
        panel.fields["returns"],
        bt["delay_days"],
        bt["cost_bps"],
        tuple(bt["winsor_quantiles"]),
    )
    return signal, series


def _run_one(item: tuple[int, str]) -> tuple[dict, np.ndarray, np.ndarray]:
    from src.backtest.metrics import period_metrics, sharpe
    from src.generator.filters import signal_filter
    from src.generator.grammar import parse

    formula_id, text = item
    node = parse(text)
    ann = _W["bt"]["annualization"]

    t0 = time.perf_counter()
    signal, series = _backtest(_W["panel"], node)
    time_full = time.perf_counter() - t0

    t0 = time.perf_counter()
    _, trunc_series = _backtest(_W["trunc"], node)
    for d in DIRECTIONS.values():
        period_metrics(trunc_series, _W["trunc_is_rows"], d, ann)
    time_is = time.perf_counter() - t0

    fres = signal_filter(
        signal,
        _W["panel"].valid,
        _W["filter_rows"],
        _W["fcfg"]["min_valid_ratio"],
        _W["fcfg"]["max_constant_day_ratio"],
    )
    row: dict = {
        "formula_id": formula_id,
        "filter_pass": fres.passed,
        "filter_reason": fres.reason,
        "time_full_sec": time_full,
        "time_is_sec": time_is,
    }
    for p in PERIODS:
        rows = _W["rows"][p]
        row[f"{p}_gross_sharpe"] = sharpe(series.gross[rows], ann)  # direction +1, no cost
        for name, d in DIRECTIONS.items():
            m = period_metrics(series, rows, d, ann)
            row[f"{p}_sharpe_{name}"] = m["sharpe"]
            row[f"{p}_rank_ic_{name}"] = m["rank_ic"]
            row[f"{p}_mdd_{name}"] = m["mdd"]
        row[f"{p}_turnover"] = m["turnover"]  # identical for both directions
    is_rows = _W["rows"]["is"]
    gross_is = np.nan_to_num(series.gross[is_rows]).astype(np.float32)
    cost_is = np.nan_to_num(series.cost[is_rows]).astype(np.float32)
    return row, gross_is, cost_is


def run(market: str, workers: int, limit: int | None) -> None:
    cfg = load_experiment_config()
    logger = setup_logging(f"backtest_{market}")
    pool = pd.read_parquet(PROJECT_ROOT / cfg["pool"]["file"])
    if limit:
        pool = pool.head(limit)
    items = list(zip(pool["formula_id"].tolist(), pool["formula"].tolist(), strict=True))
    t0 = time.perf_counter()

    rows: list[dict] = []
    gross: list[np.ndarray] = []
    cost: list[np.ndarray] = []
    with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(market,)) as ex:
        for i, (row, g, c) in enumerate(ex.map(_run_one, items, chunksize=16)):
            rows.append(row)
            gross.append(g)
            cost.append(c)
            if (i + 1) % 2000 == 0:
                logger.info("%d/%d done (%.0fs)", i + 1, len(items), time.perf_counter() - t0)

    df = pd.DataFrame(rows)
    oos_cols = [c for c in df.columns if c.startswith("oos_")]
    suffix = "" if not limit else "_smoke"
    is_test_path = output_path("results_is_test", market).with_stem(
        output_path("results_is_test", market).stem + suffix
    )
    oos_path = output_path("results_oos", market).with_stem(
        output_path("results_oos", market).stem + suffix
    )
    df.drop(columns=oos_cols).to_parquet(is_test_path, index=False)
    df[["formula_id", *oos_cols]].to_parquet(oos_path, index=False)

    from src.data.panel import load_panel

    panel = load_panel(market)
    is_dates = panel.dates[panel.rows(*cfg["periods"]["is"])]
    pnl_path = output_path("pnl_is", market).with_stem(output_path("pnl_is", market).stem + suffix)
    np.save(pnl_path, np.stack([np.stack(gross), np.stack(cost)]))
    pd.Series(is_dates.strftime("%Y-%m-%d"), name="date").to_csv(
        output_path("pnl_is_dates", market), index=False
    )

    logger.info(
        "%s: %d formulas in %.0fs; mean time full %.0f ms, IS-only %.0f ms; "
        "market filter pass %d/%d -> %s",
        market,
        len(df),
        time.perf_counter() - t0,
        df["time_full_sec"].mean() * 1e3,
        df["time_is_sec"].mean() * 1e3,
        int(df["filter_pass"].sum()),
        len(df),
        is_test_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market", required=True, choices=["us", "jp"])
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--limit", type=int, default=None, help="smoke test on first N formulas")
    args = parser.parse_args()
    run(args.market, args.workers, args.limit)


if __name__ == "__main__":
    main()
