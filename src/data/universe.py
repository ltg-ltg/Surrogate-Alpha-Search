"""Build the raw constituent list from a pinned Wikipedia revision.

The raw list (before download-failure removal and share-class dedup) is saved as
universe/universe_<market>_v1_raw.csv with provenance in universe_<market>_v1_meta.json.
The final universe_<market>_v1.csv is written by src.data.download after downloading.
"""

from __future__ import annotations

import argparse
import io
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests

from src.common.config import UNIVERSE_DIR, market_config, setup_logging

USER_AGENT = "surrogate-alpha-search/0.1 (academic research script)"
RAW_COLUMNS = ["symbol", "ticker", "name", "sector", "group_id"]
JP_CODE_RE = re.compile(r"^\d{3}[0-9A-Z]$")


def raw_universe_path(market: str) -> Path:
    return UNIVERSE_DIR / f"universe_{market}_v1_raw.csv"


def meta_path(market: str) -> Path:
    return UNIVERSE_DIR / f"universe_{market}_v1_meta.json"


def fetch_revision(host: str, title: str, as_of: str) -> dict:
    """Return the last revision of `title` at or before the end of `as_of` (UTC)."""
    params = {
        "action": "query",
        "format": "json",
        "formatversion": 2,
        "prop": "revisions",
        "titles": title,
        "rvlimit": 1,
        "rvdir": "older",
        "rvstart": f"{as_of}T23:59:59Z",
        "rvprop": "ids|timestamp",
    }
    resp = requests.get(
        f"https://{host}/w/api.php", params=params, headers={"User-Agent": USER_AGENT}, timeout=60
    )
    resp.raise_for_status()
    page = resp.json()["query"]["pages"][0]
    rev = page["revisions"][0]
    return {"revid": rev["revid"], "timestamp": rev["timestamp"], "title": page["title"]}


def fetch_revision_html(host: str, revid: int) -> str:
    params = {
        "action": "parse",
        "format": "json",
        "formatversion": 2,
        "oldid": revid,
        "prop": "text",
    }
    resp = requests.get(
        f"https://{host}/w/api.php", params=params, headers={"User-Agent": USER_AGENT}, timeout=60
    )
    resp.raise_for_status()
    return resp.json()["parse"]["text"]


def us_to_yf_ticker(symbol: str) -> str:
    """Wikipedia/S&P symbols use '.', Yahoo uses '-' (BRK.B -> BRK-B)."""
    return symbol.strip().upper().replace(".", "-")


def jp_to_yf_ticker(code: str, suffix: str) -> str:
    code = code.strip().upper()
    if not JP_CODE_RE.match(code):
        raise ValueError(f"unexpected JP security code {code!r}")
    return f"{code}{suffix}"


def parse_us(html: str) -> pd.DataFrame:
    tables = pd.read_html(io.StringIO(html))
    table = next(t for t in tables if {"Symbol", "CIK"}.issubset(t.columns))
    df = pd.DataFrame(
        {
            "symbol": table["Symbol"].astype(str).str.strip(),
            "name": table["Security"].astype(str).str.strip(),
            "sector": table["GICS Sector"].astype(str).str.strip(),
            # Same CIK = same issuer -> candidate duplicate share classes.
            "group_id": table["CIK"].astype(int).astype(str),
        }
    )
    df["ticker"] = df["symbol"].map(us_to_yf_ticker)
    return df[RAW_COLUMNS]


def parse_jp(html: str, suffix: str) -> pd.DataFrame:
    """Constituent list is split into per-sector tables with a 証券コード column."""
    tables = pd.read_html(io.StringIO(html), converters={"証券コード": str})
    # Sector headings precede each table; recover them in document order.
    headings = re.findall(r"<h3[^>]*>(.*?)</h3>", html, flags=re.S)
    sector_names = [
        re.sub(r"<[^>]+>", "", h).strip()
        for h in headings
        if re.search(r"（\d+銘柄）", re.sub(r"<[^>]+>", "", h))
    ]
    comp_tables = [t for t in tables if "証券コード" in t.columns]
    if len(sector_names) != len(comp_tables):
        sector_names = [""] * len(comp_tables)
    frames = []
    for sector, t in zip(sector_names, comp_tables, strict=True):
        frames.append(
            pd.DataFrame(
                {
                    "symbol": t["証券コード"].astype(str).str.strip(),
                    "name": t["銘柄"].astype(str).str.strip(),
                    "sector": re.sub(r"（\d+銘柄）", "", sector).strip(),
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)
    df["group_id"] = df["symbol"]
    df["ticker"] = df["symbol"].map(lambda c: jp_to_yf_ticker(c, suffix))
    return df[RAW_COLUMNS]


def build_raw_universe(market: str) -> pd.DataFrame:
    logger = setup_logging(f"universe_{market}")
    mcfg = market_config(market)
    host = mcfg["source"]["wiki_host"]
    title = mcfg["source"]["wiki_title"]
    rev = fetch_revision(host, title, mcfg["snapshot_date"])
    logger.info("using %s revision %s (%s)", title, rev["revid"], rev["timestamp"])
    html = fetch_revision_html(host, rev["revid"])

    df = parse_us(html) if market == "us" else parse_jp(html, mcfg["ticker_suffix"])

    dup_tickers = df[df["ticker"].duplicated()]["ticker"].tolist()
    if dup_tickers:
        raise ValueError(f"duplicate tickers in source table: {dup_tickers}")
    lo, hi = mcfg["expected_count"]
    count_ok = lo <= len(df) <= hi
    if not count_ok:
        raise ValueError(f"{market}: {len(df)} constituents, expected within [{lo}, {hi}]")

    UNIVERSE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw_universe_path(market), index=False, encoding="utf-8")
    meta = {
        "market": market,
        "snapshot_date": mcfg["snapshot_date"],
        "source_url": f"https://{host}/w/index.php?oldid={rev['revid']}",
        "page_title": rev["title"],
        "revision_id": rev["revid"],
        "revision_timestamp": rev["timestamp"],
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "raw_count": len(df),
        "validation": {
            "expected_count_range": [lo, hi],
            "count_ok": count_ok,
            "official_crosscheck": "pending manual check",
        },
    }
    meta_path(market).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("raw universe: %d constituents -> %s", len(df), raw_universe_path(market))
    return df


def load_raw_universe(market: str, rebuild: bool = False) -> pd.DataFrame:
    path = raw_universe_path(market)
    if rebuild or not path.exists():
        return build_raw_universe(market)
    return pd.read_csv(path, dtype=str, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market", required=True, choices=["us", "jp"])
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    load_raw_universe(args.market, rebuild=args.rebuild)


if __name__ == "__main__":
    main()
