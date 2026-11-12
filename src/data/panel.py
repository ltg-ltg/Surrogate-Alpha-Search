"""Load cleaned date x ticker panels as aligned numpy arrays."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.common.config import DATA_DIR

PRICE_FIELDS = ("open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class Panel:
    dates: pd.DatetimeIndex
    tickers: tuple[str, ...]
    fields: dict[str, np.ndarray]  # each (T, N) float64; includes "returns"
    valid: np.ndarray  # (T, N) bool: close present and volume > 0 (design 3.2)

    def rows(self, start: str, end: str) -> slice:
        """Row slice covering [start, end] inclusive."""
        lo = self.dates.searchsorted(pd.Timestamp(start), side="left")
        hi = self.dates.searchsorted(pd.Timestamp(end), side="right")
        return slice(int(lo), int(hi))


def build_panel(frames: dict[str, pd.DataFrame]) -> Panel:
    close = frames["close"]
    dates, tickers = close.index, close.columns
    fields = {
        f: frames[f].reindex(index=dates, columns=tickers).to_numpy(dtype=np.float64)
        for f in PRICE_FIELDS
    }
    c = fields["close"]
    ret = np.full_like(c, np.nan)
    ret[1:] = c[1:] / c[:-1] - 1.0  # uses close(t) and close(t-1) only
    fields["returns"] = ret
    for arr in fields.values():
        arr.setflags(write=False)
    valid = np.isfinite(c) & (np.nan_to_num(fields["volume"]) > 0)
    return Panel(pd.DatetimeIndex(dates), tuple(tickers), fields, valid)


def load_panel(market: str) -> Panel:
    frames = {f: pd.read_parquet(DATA_DIR / market / f"{f}.parquet") for f in PRICE_FIELDS}
    return build_panel(frames)
