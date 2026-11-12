"""Derive every random stream from experiment.yaml's master seed."""

from __future__ import annotations

import zlib

import numpy as np

from src.common.config import load_experiment_config


def derive_seed(*names: str | int) -> int:
    """Stable 32-bit seed for a named stream, e.g. derive_seed("generator", 3)."""
    master = load_experiment_config()["master_seed"]
    keys = [master] + [zlib.crc32(str(n).encode("utf-8")) for n in names]
    return int(np.random.SeedSequence(keys).generate_state(1)[0])


def rng_for(*names: str | int) -> np.random.Generator:
    return np.random.default_rng(derive_seed(*names))
