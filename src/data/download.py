"""Download daily OHLCV (adjusted) for a market and finalize its universe.

Steps:
  1. Load (or build) the raw constituent list from the pinned Wikipedia revision.
  2. Download adjusted OHLCV via yfinance in batches with retries; cache raw panels.
  3. Drop market-holiday rows and mask zero-volume stock-days as NaN.
  4. Drop download failures, then drop duplicate share classes (same issuer),
     keeping the class with the higher mean volume over IS.
  5. Write universe/universe_<market>_v1.csv, exclusion list, daily valid counts,
     and date x ticker parquet panels under data/<market>/.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.common.config import (
    ARTIFACTS_DIR,
    DATA_DIR,
    PROJECT_ROOT,
    load_experiment_config,
    market_config,
    setup_logging,
)
from src.data.universe import load_raw_universe, meta_path

FIELDS = ["open", "high", "low", "close", "volume"]
YF_FIELDS = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}


def raw_panel_dir(market: str) -> Path:
    return DATA_DIR / market / "raw"


def panel_dir(market: str) -> Path:
    return DATA_DIR / market


def _download_batch(tickers: list[str], start: str, end_exclusive: str) -> dict[str, pd.DataFrame]:
    df = yf.download(
        tickers,
        start=start,
        end=end_exclusive,
        auto_adjust=True,
        actions=False,
        group_by="column",
        progress=False,
        threads=True,
    )
    out: dict[str, pd.DataFrame] = {}
    for field, yf_field in YF_FIELDS.items():
        if df.empty or yf_field not in df.columns.get_level_values(0):
            out[field] = pd.DataFrame(columns=tickers, dtype=float)
            continue
        panel = df[yf_field]
        if isinstance(panel, pd.Series):
            panel = panel.to_frame(tickers[0])
        out[field] = panel.reindex(columns=tickers)
    return out


def download_panels(
    tickers: list[str], start: str, end_inclusive: str, batch_size: int, max_retries: int, logger
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Return per-field date x ticker panels and the tickers that never returned data."""
    end_exclusive = (pd.Timestamp(end_inclusive) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    collected: dict[str, list[pd.DataFrame]] = {f: [] for f in FIELDS}
    pending = list(tickers)
    for attempt in range(max_retries + 1):
        if not pending:
            break
        logger.info("download attempt %d: %d tickers", attempt + 1, len(pending))
        failed: list[str] = []
        for i in range(0, len(pending), batch_size):
            batch = pending[i : i + batch_size]
            panels = _download_batch(batch, start, end_exclusive)
            ok = [t for t in batch if panels["close"][t].notna().any()]
            failed.extend(t for t in batch if t not in ok)
            for f in FIELDS:
                collected[f].append(panels[f][ok])
            time.sleep(1.0)  # be gentle with the endpoint
        pending = failed
    result = {}
    for f in FIELDS:
        frames = [p for p in collected[f] if not p.empty]
        panel = pd.concat(frames, axis=1) if frames else pd.DataFrame()
        panel.index = pd.to_datetime(panel.index).tz_localize(None).normalize()
        result[f] = panel.sort_index().astype(float)
    return result, pending


def clean_panels(
    panels: dict[str, pd.DataFrame], holiday_zero_volume_ratio: float
) -> tuple[dict[str, pd.DataFrame], dict]:
    """Remove non-trading rows (design 3.2).

    1. Market holidays: dates where at least `holiday_zero_volume_ratio` of tickers with a
       price have zero volume are dropped entirely (Yahoo emits flat rows on JP holidays).
    2. Remaining zero-volume stock-days are set to NaN in every field (not traded -> invalid).
    """
    close, volume = panels["close"], panels["volume"]
    has_price = close.notna()
    zero_vol = (volume == 0) & has_price
    ratio = zero_vol.sum(axis=1) / has_price.sum(axis=1).clip(lower=1)
    holidays = ratio.index[ratio >= holiday_zero_volume_ratio]
    keep_dates = ratio.index.difference(holidays)

    mask = zero_vol.loc[keep_dates]
    cleaned = {f: p.loc[keep_dates].mask(mask) for f, p in panels.items()}
    per_ticker = mask.sum()
    stats = {
        "holiday_dates_dropped": [d.strftime("%Y-%m-%d") for d in holidays],
        "zero_volume_cells_masked": int(mask.sum().sum()),
        "zero_volume_top_tickers": {
            t: int(n) for t, n in per_ticker[per_ticker > 0].nlargest(10).items()
        },
    }
    return cleaned, stats


def dedupe_share_classes(
    universe: pd.DataFrame, volume: pd.DataFrame, period: tuple[str, str]
) -> tuple[pd.DataFrame, list[dict]]:
    """Keep one ticker per issuer (group_id): the one with the highest mean volume in `period`."""
    mean_vol = volume.loc[period[0] : period[1]].mean()
    keep_idx = []
    excluded = []
    for _, grp in universe.groupby("group_id", sort=False):
        if len(grp) == 1:
            keep_idx.append(grp.index[0])
            continue
        vols = grp["ticker"].map(mean_vol).fillna(-1.0)
        winner = vols.idxmax()
        keep_idx.append(winner)
        for idx in grp.index.drop(winner):
            excluded.append(
                {
                    "ticker": grp.at[idx, "ticker"],
                    "reason": "duplicate_share_class",
                    "detail": f"kept {grp.at[winner, 'ticker']} "
                    f"(mean IS volume {vols[winner]:.0f} vs {vols[idx]:.0f})",
                }
            )
    return universe.loc[sorted(keep_idx)], excluded


def run(market: str, force_download: bool = False) -> None:
    logger = setup_logging(f"download_{market}")
    cfg = load_experiment_config()
    mcfg = market_config(market)
    dcfg = cfg["data"]
    t0 = time.perf_counter()

    raw = load_raw_universe(market)
    tickers = raw["ticker"].tolist()

    rdir = raw_panel_dir(market)
    cached = (
        all((rdir / f"{f}.parquet").exists() for f in FIELDS) and (rdir / "failed.json").exists()
    )
    if cached and not force_download:
        logger.info("using cached raw panels in %s", rdir)
        panels = {f: pd.read_parquet(rdir / f"{f}.parquet") for f in FIELDS}
        failed = json.loads((rdir / "failed.json").read_text(encoding="utf-8"))
    else:
        panels, failed = download_panels(
            tickers,
            dcfg["download_start"],
            dcfg["download_end"],
            dcfg["batch_size"],
            dcfg["max_retries"],
            logger,
        )
        rdir.mkdir(parents=True, exist_ok=True)
        for f in FIELDS:
            panels[f].to_parquet(rdir / f"{f}.parquet")
        (rdir / "failed.json").write_text(json.dumps(failed), encoding="utf-8")
    t_download = time.perf_counter() - t0
    logger.info("download done in %.1fs; %d failed: %s", t_download, len(failed), failed)

    panels, clean_stats = clean_panels(panels, dcfg["cleaning"]["holiday_zero_volume_ratio"])
    logger.info(
        "cleaning: dropped %d holiday dates %s; masked %d zero-volume stock-days (top: %s)",
        len(clean_stats["holiday_dates_dropped"]),
        clean_stats["holiday_dates_dropped"],
        clean_stats["zero_volume_cells_masked"],
        clean_stats["zero_volume_top_tickers"],
    )

    exclusions = [{"ticker": t, "reason": "download_failed", "detail": ""} for t in failed]
    universe = raw[~raw["ticker"].isin(failed)]
    is_period = tuple(cfg["periods"][dcfg["share_class_volume_period"]])
    universe, dup_excl = dedupe_share_classes(universe, panels["volume"], is_period)
    exclusions.extend(dup_excl)
    final = universe["ticker"].tolist()

    # Final panels restricted to the universe.
    pdir = panel_dir(market)
    pdir.mkdir(parents=True, exist_ok=True)
    for f in FIELDS:
        panels[f][final].to_parquet(pdir / f"{f}.parquet")

    # Universe file and provenance.
    out_path = PROJECT_ROOT / mcfg["universe_file"]
    universe.to_csv(out_path, index=False, encoding="utf-8")
    meta = json.loads(meta_path(market).read_text(encoding="utf-8"))
    meta["final_count"] = len(final)
    meta["excluded_count"] = len(exclusions)
    meta["cleaning"] = clean_stats
    meta_path(market).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # Records required by the design (3.2): exclusions and daily valid counts.
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(exclusions, columns=["ticker", "reason", "detail"]).to_csv(
        ARTIFACTS_DIR / f"universe_{market}_exclusions.csv", index=False, encoding="utf-8"
    )
    valid = panels["close"][final].notna().sum(axis=1).rename("n_valid")
    valid.to_csv(ARTIFACTS_DIR / f"universe_{market}_daily_counts.csv", encoding="utf-8")

    logger.info(
        "universe %s: raw %d -> final %d (excluded %d); dates %s..%s (%d days); "
        "daily valid min/median/max = %d/%d/%d; total %.1fs",
        market,
        len(raw),
        len(final),
        len(exclusions),
        valid.index.min().date(),
        valid.index.max().date(),
        len(valid),
        valid.min(),
        int(valid.median()),
        valid.max(),
        time.perf_counter() - t0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market", required=True, choices=["us", "jp"])
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()
    run(args.market, force_download=args.force_download)


if __name__ == "__main__":
    main()
