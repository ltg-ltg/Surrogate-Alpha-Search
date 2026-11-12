"""Generate the formula pool (design 4.6-4.9) -> artifacts/pool_v1.parquet.

Depth 2 (~363 valid formulas) is enumerated exhaustively and every filter-passing
formula is kept; the rest of the pool is split equally across depths 3, 4, 5.

Sampling (per sampled depth stratum D):
  - The root is always an operator; a node at depth d >= 2 becomes a field leaf with
    probability d / D (so depth D is a forced leaf).
  - Operator category ~ {ts: 0.4, arith: 0.4, cs: 0.2}, uniform within category;
    fields and windows uniform (windows restricted by min_window).
  - A child that breaks a local rule (forbidden nesting, unit rule, identical operands)
    is re-sampled as a subtree; after too many retries the parent is re-sampled.
  - Trees whose depth != D, or that break the node / lookback limits, are rejected.
Then: canonicalize -> dedup by hash -> signal filter on US data (2016-2025).
Candidates are accepted strictly in generation order, so the pool does not depend on
the number of worker processes.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from src.common.config import ARTIFACTS_DIR, PROJECT_ROOT, load_experiment_config, setup_logging
from src.common.seeds import rng_for
from src.generator.canonical import canonicalize, formula_hash, skeleton
from src.generator.grammar import (
    Grammar,
    GrammarError,
    Node,
    check_local,
    cumulative_lookback,
    depth,
    load_grammar,
    node_count,
    validate,
)

MAX_SUBTREE_RETRIES = 20
MAX_DUPLICATE_STREAK = 100_000


def enumerate_depth(g: Grammar, target_depth: int) -> list[Node]:
    """Every valid canonical formula of exactly `target_depth`, sorted by text.

    Only used for shallow depths (2), where the space has a few hundred formulas.
    """
    if target_depth != 2:
        raise ValueError("exhaustive enumeration is only implemented for depth 2")
    leaves = [Node(f) for f in sorted(g.fields)]
    out: dict[str, Node] = {}
    for op, spec in g.ops.items():
        windows = g.allowed_windows(op) if spec["window"] else (None,)
        operand_sets = (
            [(a,) for a in leaves]
            if spec["arity"] == 1
            else [(a, b) for a in leaves for b in leaves]
        )
        for operands in operand_sets:
            for w in windows:
                node = canonicalize(Node(op, operands, w))
                try:
                    validate(node, g)
                except GrammarError:
                    continue
                out[node.text] = node
    return [out[k] for k in sorted(out)]


class _Resample(Exception):
    """Raised when a subtree cannot satisfy local rules after retries."""


class Sampler:
    def __init__(self, g: Grammar, rng: np.random.Generator):
        self.g = g
        self.rng = rng
        self.fields = sorted(g.fields)
        self.categories = sorted(g.category_probs)
        p = np.array([g.category_probs[c] for c in self.categories], dtype=float)
        self.cat_p = p / p.sum()

    def _choice(self, seq):
        return seq[int(self.rng.integers(len(seq)))]

    def _leaf(self) -> Node:
        return Node(self._choice(self.fields))

    def _node(self, d: int, target_depth: int) -> Node:
        if d >= 2 and self.rng.random() < d / target_depth:
            return self._leaf()
        cat = self.categories[int(self.rng.choice(len(self.categories), p=self.cat_p))]
        op = self._choice(self.g.ops_by_category[cat])
        spec = self.g.ops[op]
        window = self._choice(self.g.allowed_windows(op)) if spec["window"] else None
        children: list[Node] = []
        for _ in range(spec["arity"]):
            for _attempt in range(MAX_SUBTREE_RETRIES):
                child = canonicalize(self._node(d + 1, target_depth))
                if (op, child.op) in self.g.forbidden_nesting:
                    continue
                trial = canonicalize(Node(op, (*children, child), window))
                if len(trial.children) == spec["arity"]:
                    try:
                        check_local(trial, self.g)
                    except GrammarError:
                        continue
                children.append(child)
                break
            else:
                raise _Resample
        return canonicalize(Node(op, tuple(children), window))

    def sample(self, target_depth: int) -> tuple[Node | None, str]:
        """Return (canonical node, "ok") or (None, rejection reason)."""
        try:
            node = self._node(1, target_depth)
        except _Resample:
            return None, "rule_violation"
        if depth(node) != target_depth:
            return None, "depth_mismatch"
        if node_count(node) > self.g.max_nodes or cumulative_lookback(node) > self.g.max_lookback:
            return None, "size_violation"
        validate(node, self.g)  # defensive: must never fail at this point
        return node, "ok"


# ---------------------------------------------------------------- parallel filter

_WORKER: dict = {}


def _init_worker(market: str) -> None:
    import numba

    from src.data.panel import load_panel

    numba.set_num_threads(1)  # parallelism comes from processes; avoid oversubscription

    cfg = load_experiment_config()
    panel = load_panel(market)
    start = cfg["periods"]["is"][0]
    end = cfg["periods"]["oos"][1]
    _WORKER.update(panel=panel, rows=panel.rows(start, end), fcfg=cfg["pool"]["signal_filter"])


def _filter_one(text: str) -> tuple[str, float, float]:
    from src.backtest.operators import evaluate
    from src.generator.filters import signal_filter
    from src.generator.grammar import parse

    panel = _WORKER["panel"]
    sig = evaluate(parse(text), panel)
    res = signal_filter(
        sig,
        panel.valid,
        _WORKER["rows"],
        _WORKER["fcfg"]["min_valid_ratio"],
        _WORKER["fcfg"]["max_constant_day_ratio"],
    )
    return res.reason, res.valid_ratio, res.constant_day_ratio


# ---------------------------------------------------------------- pipeline


def generate_pool(
    pool_size: int, market: str, workers: int, batch: int, logger
) -> tuple[pd.DataFrame, dict]:
    g = load_grammar()
    strata = sorted([*g.exhaustive_depths, *g.sampled_depths])
    seen: set[str] = set()
    rows: list[dict] = []
    stats: dict[int, Counter] = {d: Counter() for d in strata}
    quotas: dict[int, int] = {}
    t0 = time.perf_counter()

    def accept(ex, cands: list[Node], d: int, quota: int | None) -> None:
        results = list(ex.map(_filter_one, [c.text for c in cands], chunksize=8))
        for node, (reason, vr, cr) in zip(cands, results, strict=True):
            if quota is not None and stats[d]["accepted"] >= quota:
                stats[d]["unused_after_quota"] += 1
                continue
            if reason != "ok":
                stats[d][f"filter_{reason}"] += 1
                continue
            stats[d]["accepted"] += 1
            rows.append(
                {
                    "formula_id": len(rows),
                    "formula": node.text,
                    "hash": formula_hash(node),
                    "skeleton": skeleton(node),
                    "depth": d,
                    "node_count": node_count(node),
                    "lookback": cumulative_lookback(node),
                    "valid_ratio": vr,
                    "constant_day_ratio": cr,
                }
            )
        logger.info(
            "depth %d: accepted %d/%s (%.0fs elapsed)",
            d,
            stats[d]["accepted"],
            quota if quota is not None else "all",
            time.perf_counter() - t0,
        )

    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker, initargs=(market,)
    ) as ex:
        # 1) Shallow depths: the whole space is small, so take every valid formula.
        for d in g.exhaustive_depths:
            cands = enumerate_depth(g, d)
            stats[d]["enumerated"] = len(cands)
            for node in cands:
                seen.add(formula_hash(node))
            accept(ex, cands, d, quota=None)
            quotas[d] = stats[d]["accepted"]

        # 2) Deeper depths split the remaining pool equally.
        remaining = pool_size - len(rows)
        k = len(g.sampled_depths)
        for i, d in enumerate(g.sampled_depths):
            quotas[d] = remaining // k + (remaining % k if i == k - 1 else 0)
        for d in g.sampled_depths:
            sampler = Sampler(g, rng_for("generator", "depth", d))
            while stats[d]["accepted"] < quotas[d]:
                cands = []
                dup_streak = 0
                while len(cands) < batch:
                    node, reason = sampler.sample(d)
                    stats[d]["sampled"] += 1
                    if node is None:
                        stats[d][reason] += 1
                        continue
                    h = formula_hash(node)
                    if h in seen:
                        stats[d]["duplicate"] += 1
                        dup_streak += 1
                        if dup_streak > MAX_DUPLICATE_STREAK:
                            raise RuntimeError(f"depth {d}: formula space looks exhausted")
                        continue
                    dup_streak = 0
                    seen.add(h)
                    cands.append(node)
                accept(ex, cands, d, quotas[d])

    summary = {
        "pool_size": len(rows),
        "filter_market": market,
        "quotas": {str(d): quotas[d] for d in strata},
        "elapsed_sec": round(time.perf_counter() - t0, 1),
        "by_depth": {str(d): dict(stats[d]) for d in strata},
    }
    for d in strata:
        s = stats[d]
        s_tot = s["sampled"]
        summary["by_depth"][str(d)]["rates"] = {
            k: round(s[k] / s_tot, 4)
            for k in ("rule_violation", "depth_mismatch", "size_violation", "duplicate")
            if s_tot
        }
        filt_tot = s["accepted"] + s["filter_missing"] + s["filter_constant"]
        if filt_tot:
            summary["by_depth"][str(d)]["rates"]["filter_drop"] = round(
                (s["filter_missing"] + s["filter_constant"]) / filt_tot, 4
            )
    return pd.DataFrame(rows), summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=None, help="override pool size (smoke test)")
    parser.add_argument("--out", default=None, help="override output path (smoke test)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--batch", type=int, default=2000)
    args = parser.parse_args()

    cfg = load_experiment_config()
    logger = setup_logging("generate")
    size = args.size or cfg["pool"]["size"]
    out = PROJECT_ROOT / (args.out or cfg["pool"]["file"])

    pool, summary = generate_pool(size, "us", args.workers, args.batch, logger)
    out.parent.mkdir(parents=True, exist_ok=True)
    pool.to_parquet(out, index=False)
    stats_path = ARTIFACTS_DIR / f"{out.stem}_generation_stats.json"
    stats_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("pool of %d formulas -> %s; stats -> %s", len(pool), out, stats_path)


if __name__ == "__main__":
    main()
