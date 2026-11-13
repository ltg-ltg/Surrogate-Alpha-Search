"""Equal-total-budget robustness check (design 8.4), US only, IS/Test only.

Baseline: random 3,000 formulas from the whole evaluation pool per seed.
(a) each cell's 2,000 picks (warm-up not counted)   vs random 3,000
(b) each cell's warm-up 1,000 + 2,000 picks          vs random 3,000
Writes artifacts/analysis/equal_budget_{cells,tests}.csv.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.stats import OUT, bootstrap_ci, run_family
from src.common.config import load_experiment_config, setup_logging
from src.common.seeds import rng_for
from src.experiment.replay import _load_market, run_dir


def _warmup(eval_ids: np.ndarray, seed: int, W: int) -> np.ndarray:
    return np.sort(rng_for("warmup", seed).choice(eval_ids, W, replace=False))


def _selection_sources(lam: float) -> list[tuple[str, str, str, str]]:
    """(run dir, schema, selection, target) for every US cell."""
    out = []
    for target, sfx in (("raw", ""), ("resid", "_resid")):
        out += [
            (f"base{sfx}", "base", "top", target),
            (f"mmr{sfx}", "base", "mmr", target),
            (f"ext{sfx}", "ext", "top", target),
            (f"ext_mmr{sfx}", "ext", "mmr", target),
        ]
    return out


def main() -> None:
    logger = setup_logging("equal_budget")
    cfg = load_experiment_config()
    ex = cfg["experiment"]
    W, B, n_seeds = ex["warmup_size"], ex["budget"], ex["n_seeds"]
    lam = cfg["diversity"]["primary_lambda"]
    us = _load_market("us")
    d, pnl, th, eval_ids = us["d"], us["pnl"], us["th"], us["eval_ids"]

    from src.experiment.criteria import count_unique_passes

    cells_a: dict[str, list[int]] = {}
    cells_b: dict[str, list[int]] = {}
    random_3000 = []
    for seed in range(n_seeds):
        pick = rng_for("random_equal_budget", "us", seed).permutation(eval_ids)[: W + B]
        random_3000.append(count_unique_passes(np.sort(pick), d, th, pnl)[0])
    for src, schema, sel, target in _selection_sources(lam):
        path = run_dir("us", src) / "selections.parquet"
        if not path.exists():
            continue
        s = pd.read_parquet(path)
        if "lam" in s:
            s = s[s["lam"] == lam]
        for arm, g_arm in s.groupby("arm"):
            key = f"{arm}|{schema}|{sel}|{target}"
            a_vals, b_vals = [], []
            for seed in range(n_seeds):
                picks = g_arm[g_arm["seed"] == seed].sort_values("rank")["formula_id"].to_numpy()
                a_vals.append(count_unique_passes(np.sort(picks), d, th, pnl)[0])
                union = np.union1d(picks, _warmup(eval_ids, seed, W))
                b_vals.append(count_unique_passes(union, d, th, pnl)[0])
            cells_a[key], cells_b[key] = a_vals, b_vals
        logger.info("done %s", src)

    base = {"random3000": random_3000}
    fa = pd.DataFrame({**base, **cells_a}, index=range(n_seeds))
    fb = pd.DataFrame({**base, **cells_b}, index=range(n_seeds))
    tests = run_family("equal_budget_a", [(c, "random3000") for c in cells_a], fa, f"unique_at_{B}")
    tests += run_family(
        "equal_budget_b", [(c, "random3000") for c in cells_b], fb, f"unique_union_{W + B}"
    )
    rows = []
    for label, frame in (("a_picks_only", fa), ("b_warmup_union", fb)):
        for c in frame.columns:
            x = frame[c].to_numpy(float)
            lo, hi = bootstrap_ci(x, f"equal_budget/{label}/{c}")
            rows.append(
                {"comparison": label, "cell": c, "mean": x.mean(), "ci_low": lo, "ci_high": hi}
            )
    pd.DataFrame(rows).to_csv(OUT / "equal_budget_cells.csv", index=False)
    pd.DataFrame(tests).to_csv(OUT / "equal_budget_tests.csv", index=False)
    logger.info(
        "random3000 mean %.2f; wrote equal_budget_{cells,tests}.csv", float(np.mean(random_3000))
    )


if __name__ == "__main__":
    main()
