"""Core metrics, paired tests and bootstrap CIs (design 8.1-8.2, 7.4, 7.5, 9.5).

Reads replay outputs and IS/Test results only (never OOS). Writes
artifacts/analysis/{summary_tables.csv, tests.csv, spearman.parquet}.

Families (Holm within each family, two-sided Wilcoxon signed-rank, 30 paired seeds):
  h1_<market>        each model and the heuristic vs random; each model vs heuristic
  h3_<market>        model pairs (unique passes)
  ablation_<metric>  base vs extended schema per model (US)       [design 7.4]
  mmr_<market>       lambda 0.5 MMR vs top-B / random / structure random  [7.5, exploratory]
  resid_<market>     residual-target cells vs random and vs raw-target twin  [7.6, exploratory]
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from src.backtest.engine import output_path
from src.common.config import ARTIFACTS_DIR, load_experiment_config, setup_logging
from src.common.seeds import rng_for
from src.experiment.criteria import directed
from src.experiment.replay import run_dir
from src.surrogate.models import MODEL_NAMES

OUT = ARTIFACTS_DIR / "analysis"
N_BOOT = 10_000


def bootstrap_ci(x: np.ndarray, name: str) -> tuple[float, float]:
    rng = rng_for("bootstrap", name)
    idx = rng.integers(0, len(x), (N_BOOT, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def paired_test(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Return (mean difference a - b, two-sided Wilcoxon p)."""
    d = a - b
    if np.allclose(d, 0):
        return 0.0, 1.0
    return float(d.mean()), float(wilcoxon(a, b, zero_method="wilcox").pvalue)


def spearman_table(market: str, schema: str) -> pd.DataFrame:
    """Spearman(prediction, directed IS net Sharpe) over all candidates (analysis only)."""
    d = directed(pd.read_parquet(output_path("results_is_test", market)))
    y = d["is_sharpe"].to_numpy()
    preds = pd.read_parquet(run_dir(market, schema) / "predictions.parquet")
    rows = []
    for (seed, arm), g in preds.groupby(["seed", "arm"]):
        rho = spearmanr(g["score"].to_numpy(), y[g["formula_id"].to_numpy()]).statistic
        rows.append({"seed": seed, "arm": arm, "spearman": float(rho)})
    return pd.DataFrame(rows)


def summarize(m: pd.DataFrame, label: str, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for arm, g in m.groupby("arm", sort=False):
        for met in metrics:
            if met not in g or g[met].isna().all():
                continue
            x = g.sort_values("seed")[met].to_numpy(dtype=float)
            lo, hi = bootstrap_ci(x, f"{label}/{arm}/{met}")
            rows.append(
                {
                    "experiment": label,
                    "arm": arm,
                    "metric": met,
                    "mean": x.mean(),
                    "ci_low": lo,
                    "ci_high": hi,
                }
            )
    return pd.DataFrame(rows)


def pivot(m: pd.DataFrame, metric: str) -> pd.DataFrame:
    return m.pivot(index="seed", columns="arm", values=metric).sort_index()


def run_family(name: str, pairs: list[tuple[str, str]], p: pd.DataFrame, metric: str) -> list:
    out = []
    for a, b in pairs:
        diff, pv = paired_test(p[a].to_numpy(float), p[b].to_numpy(float))
        wins = int((p[a] > p[b]).sum())
        out.append(
            {
                "family": name,
                "metric": metric,
                "a": a,
                "b": b,
                "mean_diff": diff,
                "a_wins": wins,
                "ties": int((p[a] == p[b]).sum()),
                "p": pv,
            }
        )
    if out:
        adj = holm(np.array([r["p"] for r in out]))
        for r, q in zip(out, adj, strict=True):
            r["p_holm"] = float(q)
    return out


def _cells(market: str, lam: float, B: int, data: dict) -> dict[str, pd.Series]:
    """unique_at_B per seed for every (arm, schema, selection, target) cell of one market."""
    met = f"unique_at_{B}"
    cells: dict[str, pd.Series] = {}

    def add(df: pd.DataFrame, schema: str, sel: str, target: str) -> None:
        for arm, g in df.groupby("arm"):
            cells[f"{arm}|{schema}|{sel}|{target}"] = g.set_index("seed")[met].sort_index()

    schemas = ("base", "ext")  # JP ext added as a replication test (design 7.6)
    for schema in schemas:
        for target in ("raw", "resid"):
            sfx = "" if target == "raw" else "_resid"
            top = run_dir(market, schema + sfx) / "metrics.parquet"
            mmr = run_dir(market, ("mmr" if schema == "base" else "ext_mmr") + sfx)
            if schema == "base" and target == "raw":
                add(data[(market, "base")], schema, "top", target)
            elif top.exists():
                add(pd.read_parquet(top), schema, "top", target)
            if (mmr / "metrics.parquet").exists():
                m = pd.read_parquet(mmr / "metrics.parquet")
                add(m[m["lam"] == lam], schema, "mmr", target)
    return cells


def residual_section(data: dict, B: int, lam: float) -> tuple[list, list]:
    tables, tests = [], []
    for market in ("us", "jp"):
        cells = _cells(market, lam, B, data)
        frame = pd.DataFrame(cells)
        resid_keys = [c for c in frame.columns if c.endswith("|resid")]
        pairs = [(c, "random|base|top|raw") for c in resid_keys]
        pairs += [
            (c, c.replace("|resid", "|raw"))
            for c in resid_keys
            if c.replace("|resid", "|raw") in frame
        ]
        tests += run_family(f"resid_{market}", pairs, frame, f"unique_at_{B}")
        for c in resid_keys:
            x = frame[c].to_numpy(float)
            lo, hi = bootstrap_ci(x, f"resid/{market}/{c}")
            tables.append(
                {
                    "experiment": f"{market}_resid",
                    "arm": c,
                    "metric": f"unique_at_{B}",
                    "mean": x.mean(),
                    "ci_low": lo,
                    "ci_high": hi,
                }
            )
    return [pd.DataFrame(tables)], tests


def main() -> None:
    logger = setup_logging("stats")
    cfg = load_experiment_config()
    B = cfg["experiment"]["budget"]
    OUT.mkdir(parents=True, exist_ok=True)
    core = [
        f"unique_at_{B}",
        "precision_at_k",
        "enrichment",
        "n_pass_1to3",
        "spearman",
        "time_total_sec",
        "time_fit_sec",
        "time_backtest_sec",
    ]
    core += [f"unique_at_{b}" for b in cfg["experiment"]["discovery_budgets"] if b != B]
    tables, tests, sp_all = [], [], []

    data = {}
    for market, schema in (("us", "base"), ("jp", "base"), ("us", "ext")):
        m = pd.read_parquet(run_dir(market, schema) / "metrics.parquet")
        sp = spearman_table(market, schema)
        sp["market"], sp["schema"] = market, schema
        sp_all.append(sp)
        m = m.merge(sp, on=["seed", "arm"], how="left")
        data[(market, schema)] = m
        tables.append(summarize(m, f"{market}_{schema}", core))

    # H1 / H3 per market on the base schema.
    for market in ("us", "jp"):
        p = pivot(data[(market, "base")], f"unique_at_{B}")
        pairs = [(a, "random") for a in (*MODEL_NAMES, "heuristic")]
        pairs += [(a, "heuristic") for a in MODEL_NAMES]
        tests += run_family(f"h1_{market}", pairs, p, f"unique_at_{B}")
        tests += run_family(
            f"h3_{market}", list(itertools.combinations(MODEL_NAMES, 2)), p, f"unique_at_{B}"
        )

    # Ablation (US): base vs ext per model, one Holm family per metric.
    for met in (f"unique_at_{B}", "precision_at_k", "spearman", "n_pass_1to3"):
        base = pivot(data[("us", "base")], met)
        ext = pivot(data[("us", "ext")], met)
        joined = pd.concat(
            {**{f"{a}": base[a] for a in MODEL_NAMES}, **{f"{a}_ext": ext[a] for a in MODEL_NAMES}},
            axis=1,
        )
        tests += run_family(f"ablation_{met}", [(f"{a}_ext", a) for a in MODEL_NAMES], joined, met)

    # MMR (exploratory, design 7.5).
    for market in ("us", "jp"):
        mm = pd.read_parquet(run_dir(market, "mmr") / "metrics.parquet")
        tables.append(summarize(mm[mm["primary"]], f"{market}_mmr_lam0.5", core))
        for lam in sorted(mm["lam"].unique()):
            if lam != cfg["diversity"]["primary_lambda"]:
                tables.append(
                    summarize(
                        mm[mm["lam"] == lam],
                        f"{market}_mmr_lam{lam}",
                        [f"unique_at_{B}", "n_pass_1to3"],
                    )
                )
        prim = pivot(mm[mm["primary"]], f"unique_at_{B}")
        base = pivot(data[(market, "base")], f"unique_at_{B}")
        joined = pd.concat(
            {**{f"{a}_mmr": prim[a] for a in prim.columns}, **{a: base[a] for a in base.columns}},
            axis=1,
        )
        arms = [a for a in ("heuristic", *MODEL_NAMES)]
        pairs = [(f"{a}_mmr", a) for a in arms]
        pairs += [(f"{a}_mmr", "random") for a in arms]
        pairs += [(f"{a}_mmr", "struct_random_mmr") for a in arms]
        tests += run_family(f"mmr_{market}", pairs, joined, f"unique_at_{B}")
    ext_mmr = pd.read_parquet(run_dir("us", "ext_mmr") / "metrics.parquet")
    tables.append(
        summarize(ext_mmr, "us_ext_mmr_lam0.5", [f"unique_at_{B}", "n_pass_1to3", "precision_at_k"])
    )

    # Residual target (exploratory, design 7.6). Skipped until its runs exist.
    if (run_dir("us", "base_resid") / "metrics.parquet").exists():
        r_tables, r_tests = residual_section(data, B, cfg["diversity"]["primary_lambda"])
        tables += r_tables
        tests += r_tests

    summary = pd.concat(tables, ignore_index=True)
    test_df = pd.DataFrame(tests)
    summary.to_csv(OUT / "summary_tables.csv", index=False)
    test_df.to_csv(OUT / "tests.csv", index=False)
    pd.concat(sp_all).to_parquet(OUT / "spearman.parquet", index=False)
    logger.info("wrote %d summary rows and %d tests to %s", len(summary), len(test_df), OUT)


if __name__ == "__main__":
    main()
