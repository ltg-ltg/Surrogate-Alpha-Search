"""Config loading and project paths."""

from __future__ import annotations

import logging
import sys
from functools import cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
LOG_DIR = ARTIFACTS_DIR / "logs"
UNIVERSE_DIR = PROJECT_ROOT / "universe"


@cache
def load_experiment_config() -> dict[str, Any]:
    with open(CONFIG_DIR / "experiment.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


@cache
def load_grammar_config() -> dict[str, Any]:
    with open(CONFIG_DIR / "grammar.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def market_config(market: str) -> dict[str, Any]:
    markets = load_experiment_config()["markets"]
    if market not in markets:
        raise ValueError(f"unknown market {market!r}; expected one of {sorted(markets)}")
    return markets[market]


def setup_logging(name: str) -> logging.Logger:
    """Log to console and to artifacts/logs/<name>.log."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    file_handler = logging.FileHandler(LOG_DIR / f"{name}.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    # Windows consoles default to a legacy code page; avoid mangling Japanese names.
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.addHandler(console)
    logger.propagate = False
    return logger
