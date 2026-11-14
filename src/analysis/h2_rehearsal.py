"""H2 margin-resolution rehearsal on the Test period (design 8.3.1). No OOS data.

Computes the same per-seed metrics and Bonferroni bootstrap intervals as the H2 plan,
with Test net Sharpe in place of OOS, and reports each interval's half-width against
the margin. Metric 3 (share > 0) is degenerate on Test because criterion 2 already
requires Test >= threshold >= 0; an analytic binomial estimate is reported instead.
Writes artifacts/analysis/h2_rehearsal.csv.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.analysis.stats import OUT
from src.common.config import load_experiment_config, setup_logging
from src.common.seeds import rng_for
from src.experiment.replay import _load_market, run_dir

MARGINS = {"median_sharpe": 0.1, "median_decay": 0.1, "share_positive": 0.1}
N_BOOT = 10_000
PRIMARY = ("heuristic", "lightgbm", "xgboost", "catboost", "random_forest")
HEADLINE = ("lightgbm", "xgboost", "catboost")


def per_seed(sel: pd.DataFrame, d: pd.DataFrame) -> pd.DataFrame:
    is_s = d["is_sharpe"].to_numpy()
    te = d["test_sharpe"].to_numpy()
    rows = []
    for seed, g in sel[sel["adopted"]].groupby("seed"):
        ids = g["formula_id"].to_numpy()
        rows.append(
            {
                "seed": seed,
                "n": len(ids),
                "median_sharpe": float(np.median(te[ids])),
                "median_decay": float(np.median(te[ids] / is_s[ids])),
                "share_positive": float((te[ids] > 0).mean()),
            }
        )
    return pd.DataFrame(rows).set_index("seed")


def ci(x: np.ndarray, level: float, name: str) -> tuple[float, float]:
    idx = rng_for("h2_rehearsal", name).integers(0, len(x), (N_BOOT, len(x)))
    m = x[idx].mean(axis=1)
    a = (1 - level) / 2
    return float(np.quantile(m, a)), float(np.quantile(m, 1 - a))


def main() -> None:
    logger = setup_logging("h2_rehearsal")
    lam = load_experiment_config()["diversity"]["primary_lambda"]
    rows = []
    for market in ("us", "jp"):
        M = _load_market(market)
        base = pd.read_parquet(run_dir(market, "base") / "selections.parquet")
        rnd = per_seed(base[base["arm"] == "random"], M["d"])
        head = pd.read_parquet(run_dir(market, "ext_mmr_resid") / "selections.parquet")
        head = head[head["lam"] == lam]
        groups = {
            "primary": {a: base[base["arm"] == a] for a in PRIMARY},
            "headline": {a: head[head["arm"] == a] for a in HEADLINE},
        }
        for family, cells in groups.items():
            m = len(cells)
            level = 1 - 0.05 / m
            for arm, sel in cells.items():
                ps = per_seed(sel, M["d"])
                common = ps.index.intersection(rnd.index)
                for metric, margin in MARGINS.items():
                    delta = (ps.loc[common, metric] - rnd.loc[common, metric]).to_numpy()
                    lo, hi = ci(delta, level, f"{market}/{family}/{arm}/{metric}")
                    row = {
                        "market": market,
                        "family": family,
                        "arm": arm,
                        "metric": metric,
                        "n_seeds": len(common),
                        "mean_adopted": float(ps["n"].mean()),
                        "level": level,
                        "delta_mean": float(delta.mean()),
                        "ci_low": lo,
                        "ci_high": hi,
                        "half_width": (hi - lo) / 2,
                        "margin": margin,
                    }
                    if metric == "share_positive":
                        # degenerate on Test; binomial estimate at p = 0.5 instead
                        n_a, n_r = ps["n"].mean(), rnd["n"].mean()
                        se = np.sqrt(0.25 / n_a + 0.25 / n_r) / np.sqrt(len(common))
                        z = float(norm.ppf((1 + level) / 2))
                        row["half_width"] = float(z * se)
                        row["note"] = "binomial estimate (Test share degenerate)"
                    row["resolvable"] = row["half_width"] < margin
                    rows.append(row)
        logger.info("done %s", market)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "h2_rehearsal.csv", index=False)
    logger.info("wrote h2_rehearsal.csv")


if __name__ == "__main__":
    main()
