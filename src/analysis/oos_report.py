"""H2: the single OOS run (design 8.3, 8.3.1). Run ONLY on the user's explicit request.

This is the only module allowed to read results_{market}_oos.parquet and the locked
OOS daily PnL it writes under artifacts/oos_locked/ (CLAUDE.md rule 3).

Steps per market:
  1. re-backtest every criterion-1..3 passer of the evaluation pool (a superset of every
     cell's adopted signals) and save directed OOS daily gross / cost, dates and per-year
     net Sharpe; check the recomputed OOS net Sharpe against results_*_oos.parquet
  2. per-seed metrics of every cell's adopted signals; non-inferiority decisions vs the
     random arm with Bonferroni bootstrap intervals and a margin-shifted Wilcoxon (Holm)
  3. auxiliary: day-block bootstrap on pooled unique signals, IS-level regression,
     common-factor statistics, per-year metrics, IS-OOS Spearman, figures
Usage: python -m src.analysis.oos_report --confirm-single-oos-run [--workers 8]
       python -m src.analysis.oos_report --dry-run-test-period   (Test stands in for OOS;
       reads no OOS file, writes only under artifacts/analysis/oos_dryrun_test_period/)
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from src.analysis.stats import holm
from src.backtest.engine import output_path
from src.common.config import (
    ARTIFACTS_DIR,
    PROJECT_ROOT,
    load_experiment_config,
    setup_logging,
)
from src.common.seeds import rng_for

LOCKED_DIR = ARTIFACTS_DIR / "oos_locked"
REPORT_DIR = ARTIFACTS_DIR / "analysis" / "oos"
DONE_MARKER = LOCKED_DIR / "RUN_COMPLETE.json"
DRY_DIR = ARTIFACTS_DIR / "analysis" / "oos_dryrun_test_period"


def dirs(period: str):
    """(locked PnL dir, report dir). The dry run never touches the OOS locations."""
    if period == "oos":
        return LOCKED_DIR, REPORT_DIR
    return DRY_DIR / "pnl", DRY_DIR


METRICS = ("median_sharpe", "median_decay", "share_positive")
MARGINS = {"median_sharpe": 0.1, "median_decay": 0.1, "share_positive": 0.1}
AUX_METRICS = ("share_ge_03",)
PRIMARY_ARMS = ("heuristic", "lightgbm", "xgboost", "catboost", "random_forest")
HEADLINE_ARMS = ("lightgbm", "xgboost", "catboost")
N_BOOT_SEED = 10_000
N_BOOT_BLOCK = 2_000
BLOCK_DAYS = 20
ALPHA = 0.05

_W: dict = {}


# ------------------------------------------------------------------ pure metric functions


def seed_metrics(ids: np.ndarray, is_s: np.ndarray, oos_s: np.ndarray) -> dict:
    """Metrics of one adopted set (pool-row ids). IS Sharpe is > 0 by criterion 1."""
    if len(ids) == 0:
        return {"n": 0, **{m: np.nan for m in (*METRICS, *AUX_METRICS)}}
    o, i = oos_s[ids], is_s[ids]
    return {
        "n": len(ids),
        "median_sharpe": float(np.median(o)),
        "median_decay": float(np.median(o / i)),
        "share_positive": float((o > 0).mean()),
        "share_ge_03": float((o >= 0.3).mean()),
    }


def bootstrap_mean_ci(x: np.ndarray, level: float, name: str) -> tuple[float, float]:
    idx = rng_for("oos_seed_boot", name).integers(0, len(x), (N_BOOT_SEED, len(x)))
    means = x[idx].mean(axis=1)
    a = (1 - level) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def verdict(lo: float, hi: float, margin: float) -> str:
    if lo > -margin:
        return "non_inferior"
    if hi < -margin:
        return "inferior"
    return "inconclusive"


def shifted_wilcoxon_p(delta: np.ndarray, margin: float) -> float:
    """One-sided p for H0: delta <= -margin (i.e. test delta + margin > 0)."""
    z = delta + margin
    if np.allclose(z, 0):
        return 1.0
    return float(wilcoxon(z, zero_method="wilcox", alternative="greater").pvalue)


def judge_family(
    table: pd.DataFrame, family: str, market: str, cells: list[str], random_cell: str
) -> list[dict]:
    """Non-inferiority decisions for `cells` vs `random_cell` (design 8.3.1)."""
    m = len(cells)
    level = 1 - ALPHA / m
    rnd = table[table["cell"] == random_cell].set_index("seed")
    out = []
    for metric in METRICS:
        margin = MARGINS[metric]
        rows = []
        for cell in cells:
            c = table[table["cell"] == cell].set_index("seed")
            common = c.index.intersection(rnd.index)
            delta = (c.loc[common, metric] - rnd.loc[common, metric]).to_numpy(float)
            ok = np.isfinite(delta)
            delta = delta[ok]
            lo, hi = bootstrap_mean_ci(delta, level, f"{market}/{family}/{cell}/{metric}")
            rows.append(
                {
                    "market": market,
                    "family": family,
                    "cell": cell,
                    "metric": metric,
                    "margin": margin,
                    "n_seeds": int(ok.sum()),
                    "seeds_excluded": int(len(common) - ok.sum()),
                    "cell_mean": float(c.loc[common, metric].mean()),
                    "random_mean": float(rnd.loc[common, metric].mean()),
                    "delta_mean": float(delta.mean()),
                    "ci_level": level,
                    "ci_low": lo,
                    "ci_high": hi,
                    "verdict": verdict(lo, hi, margin),
                    "wilcoxon_shift_p": shifted_wilcoxon_p(delta, margin),
                }
            )
        adj = holm(np.array([r["wilcoxon_shift_p"] for r in rows]))
        for r, a in zip(rows, adj, strict=True):
            r["wilcoxon_shift_p_holm"] = float(a)
        out += rows
    return out


def sharpe_rows(net: np.ndarray, ann: int) -> np.ndarray:
    """Row-wise Sharpe ignoring non-finite days (same rule as metrics.sharpe)."""
    ok = np.isfinite(net)
    n = ok.sum(axis=1)
    x = np.where(ok, net, 0.0)
    mean = x.sum(axis=1) / np.maximum(n, 1)
    var = (np.where(ok, net - mean[:, None], 0.0) ** 2).sum(axis=1) / np.maximum(n - 1, 1)
    sd = np.sqrt(var)
    out = np.zeros(len(net))
    good = (n >= 2) & (sd > 0)
    out[good] = mean[good] / sd[good] * np.sqrt(ann)
    return out


def block_indices(n_days: int, block: int, n_boot: int, rng) -> np.ndarray:
    """Moving-block bootstrap day indices, shape (n_boot, n_days)."""
    n_blocks = -(-n_days // block)
    starts = rng.integers(0, n_days - block + 1, (n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)
    return idx[:, :n_days]


def pooled_metric(ids: np.ndarray, is_s: np.ndarray, oos_s: np.ndarray) -> dict:
    return seed_metrics(ids, is_s, oos_s)


def ols(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(X, y, rcond=None)[0]


def regression_design(
    stacks: dict[str, np.ndarray], is_s: np.ndarray, random_cell: str
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Rows = (cell, unique signal); columns = [1, IS Sharpe, dummies for non-random cells]."""
    cells = [c for c in stacks if c != random_cell]
    ids, cols = [], []
    for c, u in stacks.items():
        ids.append(u)
        d = np.zeros((len(u), len(cells)))
        if c != random_cell:
            d[:, cells.index(c)] = 1.0
        cols.append(d)
    ids_all = np.concatenate(ids)
    X = np.column_stack([np.ones(len(ids_all)), is_s[ids_all], np.vstack(cols)])
    return X, ids_all, ["intercept", "is_sharpe", *cells]


def factor_stats(net: np.ndarray) -> dict:
    """Mean |corr| off-diagonal and first-PC variance share of daily OOS net PnL."""
    if len(net) < 2:
        return {"mean_abs_corr": np.nan, "pc1_share": np.nan}
    x = np.nan_to_num(net)
    x = x - x.mean(axis=1, keepdims=True)
    sd = np.sqrt((x**2).mean(axis=1, keepdims=True))
    z = np.divide(x, sd, out=np.zeros_like(x), where=sd > 0)
    c = z @ z.T / z.shape[1]
    n = len(c)
    off = np.abs(c[~np.eye(n, dtype=bool)])
    ev = np.linalg.eigvalsh(c)
    return {"mean_abs_corr": float(off.mean()), "pc1_share": float(ev[-1] / ev.sum())}


# ------------------------------------------------------------------ OOS regeneration


def _init_worker(market: str) -> None:
    from src.backtest import engine

    engine._init_worker(market)
    _W["engine"] = engine


def _regen_one(item: tuple[int, str, float, str]) -> tuple[int, np.ndarray, np.ndarray, float]:
    from src.backtest.metrics import sharpe
    from src.generator.grammar import parse

    fid, text, direction, period = item
    E = _W["engine"]._W
    _, series = _W["engine"]._backtest(E["panel"], parse(text))
    rows = E["rows"][period]
    gross, cost = series.gross[rows], series.cost[rows]
    s = sharpe(direction * gross - cost, E["bt"]["annualization"])
    return fid, gross.astype(np.float32), cost.astype(np.float32), s


def regenerate(
    market: str, ids: np.ndarray, direction: np.ndarray, workers: int, logger, period: str
):
    from src.data.panel import load_panel

    cfg = load_experiment_config()
    pool = pd.read_parquet(PROJECT_ROOT / cfg["pool"]["file"])
    panel = load_panel(market)
    dates = panel.dates[panel.rows(*cfg["periods"][period])]
    items = [(int(i), pool.at[int(i), "formula"], float(direction[i]), period) for i in ids]
    gross = np.empty((len(ids), len(dates)), np.float32)
    cost = np.empty_like(gross)
    s64 = np.empty(len(ids))
    pos = {int(i): k for k, i in enumerate(ids)}
    t0 = time.perf_counter()
    with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(market,)) as ex:
        for n, (fid, g, c, s) in enumerate(ex.map(_regen_one, items, chunksize=16)):
            k = pos[fid]
            gross[k], cost[k], s64[k] = g, c, s
            if (n + 1) % 2000 == 0:
                logger.info(
                    "%s regen %d/%d (%.0fs)", market, n + 1, len(ids), time.perf_counter() - t0
                )
    return gross, cost, s64, pd.DatetimeIndex(dates)


# ------------------------------------------------------------------ cells


def collect_cells(market: str, M: dict, lam: float) -> tuple[dict, dict]:
    """{cell: {seed: adopted ids}} and {cell: group} for every cell of the plan."""
    from src.analysis.equal_budget import _warmup
    from src.experiment.criteria import count_unique_passes
    from src.experiment.replay import run_dir

    cfg = load_experiment_config()
    W, B, n_seeds = (cfg["experiment"][k] for k in ("warmup_size", "budget", "n_seeds"))
    cells: dict[str, dict[int, np.ndarray]] = {}
    group: dict[str, str] = {}
    dirs = ["base", "mmr", "ext", "ext_mmr"]
    dirs += [f"{x}_resid" for x in dirs]
    for src in dirs:
        path = run_dir(market, src) / "selections.parquet"
        if not path.exists():
            continue
        s = pd.read_parquet(path)
        if "lam" in s:
            s = s[s["lam"] == lam]
        for arm, g in s.groupby("arm"):
            cell = f"{src}|{arm}"
            sel = g[g["adopted"]]
            cells[cell] = {
                int(seed): np.sort(sel[sel["seed"] == seed]["formula_id"].to_numpy())
                for seed in range(n_seeds)
            }
            if src == "base":
                group[cell] = "primary_random" if arm == "random" else "primary"
            elif src == "ext_mmr_resid" and arm in HEADLINE_ARMS:
                group[cell] = "headline"
            else:
                group[cell] = "descriptive"
            if market == "us":  # (b) warm-up union cells (8.4), descriptive
                ucell = f"{src}|{arm}|union"
                cells[ucell] = {}
                for seed in range(n_seeds):
                    picks = g[g["seed"] == seed]["formula_id"].to_numpy()
                    union = np.union1d(picks, _warmup(M["eval_ids"], seed, W))
                    cells[ucell][seed] = np.sort(
                        count_unique_passes(union, M["d"], M["th"], M["pnl"])[1]
                    )
                group[ucell] = "descriptive"
    if market == "us":
        cells["random3000"] = {}
        for seed in range(n_seeds):
            pick = rng_for("random_equal_budget", "us", seed).permutation(M["eval_ids"])[: W + B]
            cells["random3000"][seed] = np.sort(
                count_unique_passes(np.sort(pick), M["d"], M["th"], M["pnl"])[1]
            )
        group["random3000"] = "descriptive"
    return cells, group


# ------------------------------------------------------------------ main


def analyse_market(market: str, workers: int, logger, period: str = "oos") -> dict:
    """period = "oos" for the real run; "test" for the dry run (Test stands in for OOS)."""
    from src.experiment.criteria import pass_1to3
    from src.experiment.replay import _load_market

    cfg = load_experiment_config()
    ann = cfg["backtest"]["annualization"]
    lam = cfg["diversity"]["primary_lambda"]
    M = _load_market(market)
    d = M["d"]
    is_s = d["is_sharpe"].to_numpy()
    direction = d["direction"].to_numpy().astype(float)
    ids13 = M["eval_ids"][pass_1to3(d.iloc[M["eval_ids"]], M["th"])]

    locked, report = dirs(period)
    # 1. regeneration (reused if already saved) + check against the recorded metrics
    pnl_path = locked / f"pnl_{market}_{period}.npy"
    ids_path = locked / f"pnl_{market}_{period}_ids.csv"
    dates_path = locked / f"pnl_{market}_{period}_dates.csv"
    if pnl_path.exists():
        saved_ids = pd.read_csv(ids_path)["formula_id"].to_numpy()
        if not np.array_equal(saved_ids, ids13):
            raise RuntimeError(f"{market}: saved {period} PnL ids differ from criteria 1-3 set")
        gross, cost = np.load(pnl_path)
        dates = pd.DatetimeIndex(pd.read_csv(dates_path)["date"])
        s64 = sharpe_rows(direction[ids13, None] * gross.astype(np.float64) - cost, ann)
        logger.info("%s: reusing saved %s PnL", market, period)
    else:
        gross, cost, s64, dates = regenerate(market, ids13, direction, workers, logger, period)
        locked.mkdir(parents=True, exist_ok=True)
        np.save(pnl_path, np.stack([gross, cost]))
        pd.DataFrame({"formula_id": ids13, "direction": direction[ids13]}).to_csv(
            ids_path, index=False
        )
        pd.Series(dates, name="date").to_csv(dates_path, index=False)
    if period == "oos":
        rec_df = pd.read_parquet(output_path("results_oos", market)).set_index("formula_id")
    else:
        rec_df = pd.read_parquet(output_path("results_is_test", market)).set_index("formula_id")
    rec = np.where(
        direction[ids13] > 0,
        rec_df.loc[ids13, f"{period}_sharpe_pos"].to_numpy(),
        rec_df.loc[ids13, f"{period}_sharpe_neg"].to_numpy(),
    )
    max_diff = float(np.max(np.abs(rec - s64)))
    if max_diff > 1e-4:  # float32 storage of reused PnL; fresh runs are exact
        raise RuntimeError(f"{market}: recomputed {period} Sharpe differs (max {max_diff:.3e})")
    net = direction[ids13, None] * gross.astype(np.float64) - cost
    row_of = np.full(len(d), -1)
    row_of[ids13] = np.arange(len(ids13))
    oos_s = np.full(len(d), np.nan)
    oos_s[ids13] = s64
    years = {}
    for y in sorted(set(dates.year)):
        m = dates.year == y
        years[y] = np.full(len(d), np.nan)
        years[y][ids13] = sharpe_rows(net[:, m], ann)
    pd.DataFrame(
        {
            "formula_id": ids13,
            "oos_sharpe": s64,
            **{f"sharpe_{y}": v[ids13] for y, v in years.items()},
        }
    ).to_parquet(locked / f"{period}_{market}_signal_sharpe.parquet", index=False)
    logger.info(
        "%s: %d signals regenerated, max |diff| vs recorded %.2e", market, len(ids13), max_diff
    )

    # 2. per-seed metrics and decisions
    cells, group = collect_cells(market, M, lam)
    per_seed = []
    for cell, by_seed in cells.items():
        for seed, ids in by_seed.items():
            assert (row_of[ids] >= 0).all(), "adopted signal outside criteria 1-3 set"
            per_seed.append(
                {
                    "market": market,
                    "cell": cell,
                    "group": group[cell],
                    "seed": seed,
                    **seed_metrics(ids, is_s, oos_s),
                }
            )
    table = pd.DataFrame(per_seed)
    random_cell = "base|random"
    primary = [c for c in cells if group[c] == "primary"]
    headline = [c for c in cells if group[c] == "headline"]
    decisions = judge_family(table, "primary", market, primary, random_cell)
    decisions += judge_family(table, "headline", market, headline, random_cell)

    # 3a. pooled unique signals + day-block bootstrap
    pooled = {c: np.unique(np.concatenate(list(v.values()))) for c, v in cells.items()}
    focus = [random_cell, *primary, *headline]
    rng = rng_for("oos_block_boot", market)
    idx = block_indices(len(dates), BLOCK_DAYS, N_BOOT_BLOCK, rng)
    need = np.unique(np.concatenate([pooled[c] for c in focus]))
    sub = net[row_of[need]]
    boot = {c: {m: np.empty(N_BOOT_BLOCK) for m in METRICS} for c in focus}
    stacks_p = {c: pooled[c] for c in [random_cell, *primary]}
    stacks_h = {c: pooled[c] for c in [random_cell, *headline]}
    Xp, idp, names_p = regression_design(stacks_p, is_s, random_cell)
    Xh, idh, names_h = regression_design(stacks_h, is_s, random_cell)
    coef_boot_p = np.empty((N_BOOT_BLOCK, Xp.shape[1]))
    coef_boot_h = np.empty((N_BOOT_BLOCK, Xh.shape[1]))
    oos_b = np.full(len(d), np.nan)
    for b in range(N_BOOT_BLOCK):
        oos_b[need] = sharpe_rows(sub[:, idx[b]], ann)
        for c in focus:
            mt = pooled_metric(pooled[c], is_s, oos_b)
            for m in METRICS:
                boot[c][m][b] = mt[m]
        coef_boot_p[b] = ols(oos_b[idp], Xp)
        coef_boot_h[b] = ols(oos_b[idh], Xh)
    pooled_rows = []
    for c in focus:
        pm = pooled_metric(pooled[c], is_s, oos_s)
        for m in METRICS:
            diff = boot[c][m] - boot[random_cell][m]
            pooled_rows.append(
                {
                    "market": market,
                    "cell": c,
                    "group": group[c],
                    "metric": m,
                    "n_unique": len(pooled[c]),
                    "value": pm[m],
                    "random_value": pooled_metric(pooled[random_cell], is_s, oos_s)[m],
                    "diff_ci_low": float(np.quantile(diff, 0.025)),
                    "diff_ci_high": float(np.quantile(diff, 0.975)),
                }
            )
    reg_rows = []
    for fam, X, ids_r, names, cb in (
        ("primary", Xp, idp, names_p, coef_boot_p),
        ("headline", Xh, idh, names_h, coef_boot_h),
    ):
        coef = ols(oos_s[ids_r], X)
        for k, nm in enumerate(names):
            reg_rows.append(
                {
                    "market": market,
                    "family": fam,
                    "term": nm,
                    "coef": float(coef[k]),
                    "ci_low": float(np.quantile(cb[:, k], 0.025)),
                    "ci_high": float(np.quantile(cb[:, k], 0.975)),
                    "n_rows": len(ids_r),
                }
            )

    # 3b. descriptive: every cell pooled, factor stats, per-year, Spearman
    desc_rows = []
    for c, u in pooled.items():
        row = {"market": market, "cell": c, "group": group[c], "n_unique": len(u)}
        row.update({f"pooled_{k}": v for k, v in pooled_metric(u, is_s, oos_s).items() if k != "n"})
        row.update(factor_stats(net[row_of[u]]))
        for y, v in years.items():
            row[f"median_sharpe_{y}"] = float(np.median(v[u])) if len(u) else np.nan
            row[f"share_positive_{y}"] = float((v[u] > 0).mean()) if len(u) else np.nan
        row["spearman_is_oos"] = (
            float(spearmanr(is_s[u], oos_s[u]).statistic) if len(u) > 2 else np.nan
        )
        desc_rows.append(row)
    ref = pooled_metric(ids13, is_s, oos_s)
    desc_rows.append(
        {
            "market": market,
            "cell": "pool_criteria_1to3",
            "group": "reference",
            "n_unique": len(ids13),
            **{f"pooled_{k}": v for k, v in ref.items() if k != "n"},
        }
    )

    _figures(report, market, pooled, focus, is_s, oos_s)
    return {
        "per_seed": table,
        "decisions": pd.DataFrame(decisions),
        "pooled_bootstrap": pd.DataFrame(pooled_rows),
        "regression": pd.DataFrame(reg_rows),
        "descriptive": pd.DataFrame(desc_rows),
        "check": {"n_signals": len(ids13), "max_abs_diff_vs_recorded": max_diff},
    }


def _figures(report, market: str, pooled: dict, focus: list[str], is_s, oos_s) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        2, (len(focus) + 1) // 2, figsize=(4 * ((len(focus) + 1) // 2), 8), sharex=True, sharey=True
    )
    for ax, c in zip(axes.ravel(), focus, strict=False):
        u = pooled[c]
        ax.scatter(is_s[u], oos_s[u], s=6, alpha=0.5)
        ax.axhline(0, lw=0.8, color="grey")
        ax.set_title(c, fontsize=8)
        ax.set_xlabel("IS net Sharpe")
        ax.set_ylabel("OOS net Sharpe")
    fig.tight_layout()
    fig.savefig(report / f"scatter_is_oos_{market}.png", dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm-single-oos-run", action="store_true")
    ap.add_argument("--dry-run-test-period", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if args.dry_run_test_period:
        run_all("test", args.workers, setup_logging("oos_report_dryrun"))
        return
    if not args.confirm_single_oos_run:
        raise SystemExit("refusing: pass --confirm-single-oos-run (user approval required)")
    if DONE_MARKER.exists():
        raise SystemExit(f"refusing: the single OOS run already completed ({DONE_MARKER})")
    commit = git_commit()
    LOCKED_DIR.mkdir(parents=True, exist_ok=True)
    attempt = {"started": pd.Timestamp.now().isoformat(), "commit": commit}
    log_attempt(attempt)
    try:
        checks = run_all("oos", args.workers, setup_logging("oos_report"))
    except BaseException as e:
        log_attempt({**attempt, "outcome": "failed", "error": repr(e)[:500]})
        raise
    done = {**attempt, "completed": pd.Timestamp.now().isoformat(), "checks": checks}
    log_attempt({**done, "outcome": "completed"})
    DONE_MARKER.write_text(json.dumps(done, indent=2), encoding="utf-8")


def git_commit() -> str:
    """HEAD hash; refuse unless design doc, config and src are committed (design 8.3.1)."""
    import subprocess

    def git(*a: str) -> str:
        return subprocess.run(
            ["git", *a], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()

    dirty = git("status", "--porcelain", "--", "surrogate_alpha_search_design.md", "config", "src")
    if dirty:
        raise SystemExit(f"refusing: uncommitted changes in plan or code: {dirty!r}")
    return git("rev-parse", "HEAD")


def log_attempt(entry: dict) -> None:
    """Append-only attempt log next to the locked OOS data."""
    with open(LOCKED_DIR / "run_attempts.jsonl", "a", encoding="utf-8") as f:
        print(json.dumps(entry), file=f)


def run_all(period: str, workers: int, logger) -> dict:
    _, report = dirs(period)
    report.mkdir(parents=True, exist_ok=True)
    out: dict[str, list] = {}
    checks = {}
    for market in ("us", "jp"):
        res = analyse_market(market, workers, logger, period)
        checks[market] = res.pop("check")
        for k, v in res.items():
            out.setdefault(k, []).append(v)
    for k, frames in out.items():
        pd.concat(frames, ignore_index=True).to_csv(report / f"{k}.csv", index=False)
    (report / "checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    logger.info("%s report written to %s", period, report)
    return checks


if __name__ == "__main__":
    main()
