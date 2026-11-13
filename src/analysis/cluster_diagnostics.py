"""Post-hoc cluster-concentration diagnostic (design 8.6), US only, IS/Test only.

Descriptive statistics only (no tests): for every US cell's B picks per seed,
  - criteria 1-3 passes, unique passes (1-4), collapse = 1 - unique / passes_1to3
  - skeleton concentration: distinct groups, effective groups (1 / sum share^2), top-10 share
  - IS daily net PnL redundancy: mean |corr| and share of pairs with |corr| >= max_corr,
    over all picks and over criteria 1-3 passers only
Writes artifacts/analysis/cluster_diagnostics{,_summary}.csv.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.analysis.stats import OUT, bootstrap_ci  # noqa: E402
from src.common.config import PROJECT_ROOT, load_experiment_config, setup_logging  # noqa: E402
from src.experiment.criteria import count_unique_passes, pass_1to3  # noqa: E402
from src.experiment.replay import _load_market, run_dir  # noqa: E402

# (run dir, schema, selection, target)
SOURCES = [
    (f"{name}{sfx}", schema, sel, target)
    for target, sfx in (("raw", ""), ("resid", "_resid"))
    for name, schema, sel in (
        ("base", "base", "top"),
        ("mmr", "base", "mmr"),
        ("ext", "ext", "top"),
        ("ext_mmr", "ext", "mmr"),
    )
]


def _pair_stats(z: np.ndarray, max_corr: float) -> tuple[float, float]:
    n = len(z)
    if n < 2:
        return float("nan"), float("nan")
    c = np.abs(z @ z.T) / z.shape[1]
    iu = np.triu_indices(n, 1)
    v = c[iu]
    return float(v.mean()), float((v >= max_corr).mean())


def diagnose(picks: np.ndarray, us: dict, groups: np.ndarray) -> dict:
    d, pnl, th = us["d"], us["pnl"], us["th"]
    ok = pass_1to3(d.iloc[picks], th)
    n13 = int(ok.sum())
    n_unique = count_unique_passes(np.sort(picks), d, th, pnl)[0]
    _, counts = np.unique(groups[picks], return_counts=True)
    share = counts / counts.sum()
    z = pnl.standardized(picks)
    mean_all, hi_all = _pair_stats(z, th.max_pnl_corr)
    mean_13, hi_13 = _pair_stats(z[ok], th.max_pnl_corr)
    return {
        "pass_1to3": n13,
        "unique": n_unique,
        "collapse": 1 - n_unique / n13 if n13 else float("nan"),
        "n_groups": len(counts),
        "eff_groups": float(1 / (share**2).sum()),
        "top10_share": float(np.sort(share)[::-1][:10].sum()),
        "mean_abs_corr": mean_all,
        "high_corr_pairs": hi_all,
        "mean_abs_corr_1to3": mean_13,
        "high_corr_pairs_1to3": hi_13,
    }


def main() -> None:
    logger = setup_logging("cluster_diagnostics")
    cfg = load_experiment_config()
    lam = cfg["diversity"]["primary_lambda"]
    us = _load_market("us")
    pool = pd.read_parquet(PROJECT_ROOT / cfg["pool"]["file"])
    groups = pd.factorize(pool["skeleton"])[0]

    rows = []
    for src, schema, sel, target in SOURCES:
        path = run_dir("us", src) / "selections.parquet"
        if not path.exists():
            continue
        s = pd.read_parquet(path)
        if "lam" in s:
            s = s[s["lam"] == lam]
        for (arm, seed), g in s.groupby(["arm", "seed"]):
            picks = g.sort_values("rank")["formula_id"].to_numpy()
            rows.append(
                {
                    "cell": f"{arm}|{schema}|{sel}|{target}",
                    "arm": arm,
                    "schema": schema,
                    "selection": sel,
                    "target": target,
                    "seed": int(seed),
                    **diagnose(picks, us, groups),
                }
            )
        logger.info("done %s", src)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "cluster_diagnostics.csv", index=False)
    metrics = [
        c for c in df.columns if c not in ("cell", "arm", "schema", "selection", "target", "seed")
    ]
    summary = []
    for cell, g in df.groupby("cell", sort=False):
        row = {"cell": cell, "n_seeds": len(g)}
        for m in metrics:
            x = g[m].dropna().to_numpy(float)
            lo, hi = bootstrap_ci(x, f"cluster/{cell}/{m}") if len(x) else (np.nan, np.nan)
            row[m] = x.mean() if len(x) else np.nan
            row[f"{m}_ci_low"], row[f"{m}_ci_high"] = lo, hi
        summary.append(row)
    pd.DataFrame(summary).to_csv(OUT / "cluster_diagnostics_summary.csv", index=False)
    logger.info("wrote cluster_diagnostics{,_summary}.csv (%d cells)", len(summary))


if __name__ == "__main__":
    main()
