"""Figures for the final deliverables (design 11). Reads IS/Test-side artifacts only.

The IS-vs-OOS scatter (deliverable 4) is drawn by oos_report.py, the only module allowed
to read OOS data. Outputs: artifacts/analysis/figures/*.png
  1 discovery_curves.png      unique passes vs budget, H1 arms + headline cells (US, JP)
  2 cell_comparison.png       unique passes at B for every arm x variant (US, JP)
  3 time_per_search.png       per-search wall time by stage, US (design 8.5)
  5 us_jp_sharpe.png          directed IS net Sharpe, US vs JP, same formula
  7 mmr_lambda.png            unique passes vs MMR lambda (lambda 0 = top-B)
  9 cluster_vs_unique.png     high-correlation pair share vs unique passes (8.6)
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.analysis.stats import bootstrap_ci  # noqa: E402
from src.backtest.engine import output_path  # noqa: E402
from src.common.config import ARTIFACTS_DIR, load_experiment_config, setup_logging  # noqa: E402
from src.experiment.criteria import directed  # noqa: E402
from src.experiment.replay import run_dir  # noqa: E402

OUT = ARTIFACTS_DIR / "analysis" / "figures"

# Validated categorical slots (dataviz reference palette, light mode), fixed per arm.
SURFACE = "#fcfcfb"
INK, INK_2, INK_3, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
ARM_COLOR = {
    "lightgbm": "#2a78d6",
    "xgboost": "#eb6834",
    "catboost": "#1baf7a",
    "random_forest": "#eda100",
    "heuristic": "#e87ba4",
}
RANDOM_COLOR = INK_2
ARM_LABEL = {
    "random": "Random",
    "struct_random": "Struct. random",
    "heuristic": "Heuristic",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "catboost": "CatBoost",
    "random_forest": "Random Forest",
}
MODEL_ARMS = ("heuristic", "lightgbm", "xgboost", "catboost", "random_forest")
BOOSTING = ("lightgbm", "xgboost", "catboost")
VARIANTS = [  # (run dir suffix, label)
    ("base", "base · top-B · raw"),
    ("mmr", "base · MMR · raw"),
    ("ext", "ext · top-B · raw"),
    ("ext_mmr", "ext · MMR · raw"),
    ("base_resid", "base · top-B · resid"),
    ("mmr_resid", "base · MMR · resid"),
    ("ext_resid", "ext · top-B · resid"),
    ("ext_mmr_resid", "ext · MMR · resid"),
]
MARKET_LABEL = {"us": "US (S&P 500)", "jp": "Japan (Nikkei 225)"}


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": INK_3,
            "ytick.color": INK_3,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "lines.linewidth": 2,
            "font.family": "DejaVu Sans",
        }
    )


def color_of(arm: str) -> str:
    return ARM_COLOR.get(arm, RANDOM_COLOR)


def metrics(market: str, variant: str, lam: float) -> pd.DataFrame | None:
    path = run_dir(market, variant) / "metrics.parquet"
    if not path.exists():
        return None
    m = pd.read_parquet(path)
    if "lam" in m:
        m = m[m["lam"] == lam]
    return m


def mean_ci(x: np.ndarray, name: str) -> tuple[float, float, float]:
    lo, hi = bootstrap_ci(np.asarray(x, float), name)
    return float(np.mean(x)), lo, hi


def end_labels(ax, x: float, items: list[tuple[float, str]], gap_frac: float = 0.045) -> None:
    """Direct labels at the line ends, pushed apart vertically so they never overlap."""
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * gap_frac
    items = sorted(items)
    placed: list[float] = []
    for y, _ in items:
        placed.append(max(y, placed[-1] + gap) if placed else y)
    shift = max(0.0, placed[-1] - (hi - gap / 2)) if placed else 0.0
    xr = ax.get_xlim()
    dx = (xr[1] - xr[0]) * 0.015
    for (_y, text), yp in zip(items, placed, strict=True):
        ax.text(x + dx, yp - shift, text, va="center", fontsize=7.5, color=INK_2)


# ------------------------------------------------------------------ 1 discovery curves


def discovery_curves(lam: float, budgets: list[int]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for col, market in enumerate(("us", "jp")):
        base = metrics(market, "base", lam)
        head = metrics(market, "ext_mmr_resid", lam)
        rnd = base[base["arm"] == "random"]
        for row, (frame, arms, title) in enumerate(
            (
                (base, MODEL_ARMS, "Pre-registered (H1): base · top-B · raw"),
                (head, BOOSTING, "Exploratory: ext · MMR · resid"),
            )
        ):
            ax = axes[row, col]
            ends: list[tuple[float, str]] = []
            for arm, g in [("random", rnd), *[(a, frame[frame["arm"] == a]) for a in arms]]:
                stats = [
                    mean_ci(g[f"unique_at_{b}"].to_numpy(), f"fig1/{market}/{row}/{arm}/{b}")
                    for b in budgets
                ]
                mu = [s[0] for s in stats]
                ax.fill_between(
                    budgets,
                    [s[1] for s in stats],
                    [s[2] for s in stats],
                    color=color_of(arm),
                    alpha=0.15,
                    lw=0,
                )
                ax.plot(
                    budgets,
                    mu,
                    color=color_of(arm),
                    marker="o",
                    ms=4,
                    ls="--" if arm == "random" else "-",
                    label=ARM_LABEL[arm],
                )
                ends.append((mu[-1], f"{ARM_LABEL[arm]} {mu[-1]:.1f}"))
            ax.set_title(f"{MARKET_LABEL[market]} — {title}", loc="left", fontsize=10)
            ax.set_xlim(budgets[0] - 50, budgets[-1] + 650)
            end_labels(ax, budgets[-1], ends)
            ax.set_ylabel("Unique passing signals (mean, 95% CI)")
            if row == 1:
                ax.set_xlabel("Backtest budget (selected formulas)")
            ax.legend(loc="upper left")
    fig.suptitle("Discovery curves: unique passes vs budget (30 seeds)", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "discovery_curves.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ 2 cell comparison


def cell_comparison(lam: float, B: int) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(16, 7.5), sharey=True)
    for r, market in enumerate(("us", "jp")):
        rnd = metrics(market, "base", lam)
        rnd = rnd[rnd["arm"] == "random"][f"unique_at_{B}"].to_numpy()
        r_mu, r_lo, r_hi = mean_ci(rnd, f"fig2/{market}/random")
        for c, arm in enumerate(MODEL_ARMS):
            ax = axes[r, c]
            ax.axvspan(r_lo, r_hi, color=RANDOM_COLOR, alpha=0.12, lw=0)
            ax.axvline(r_mu, color=RANDOM_COLOR, ls="--", lw=1.2)
            for i, (variant, _label) in enumerate(VARIANTS):
                m = metrics(market, variant, lam)
                if m is None or arm not in set(m["arm"]):
                    continue
                x = m[m["arm"] == arm][f"unique_at_{B}"].to_numpy()
                mu, lo, hi = mean_ci(x, f"fig2/{market}/{variant}/{arm}")
                ax.plot([lo, hi], [i, i], color=color_of(arm), lw=2, solid_capstyle="round")
                ax.plot(mu, i, "o", color=color_of(arm), ms=7, mec=SURFACE, mew=1.5)
                ax.annotate(
                    f"{mu:.1f}",
                    (hi, i),
                    xytext=(4, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=7,
                    color=INK_2,
                )
            ax.set_title(
                f"{MARKET_LABEL[market].split(' ')[0]} · {ARM_LABEL[arm]}", loc="left", fontsize=9.5
            )
            ax.set_xlim(0, 50)
            ax.grid(axis="y", visible=False)
            if r == 1:
                ax.set_xlabel(f"Unique passes at B = {B}")
    axes[0, 0].set_yticks(range(len(VARIANTS)), [v[1] for v in VARIANTS])
    axes[0, 0].invert_yaxis()
    for ax in axes[:, 0]:
        ax.set_yticks(range(len(VARIANTS)), [v[1] for v in VARIANTS])
    fig.suptitle(
        "Unique passes by arm and variant (mean, 95% CI; dashed band = random arm)",
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(OUT / "cell_comparison.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ 3 time per search


def time_per_search(W: int, B: int, lam: float) -> None:
    from src.analysis.equal_budget import _warmup
    from src.common.seeds import rng_for
    from src.experiment.replay import _load_market

    bt = pd.read_parquet(ARTIFACTS_DIR / "benchmark" / "backtest_timing.parquet")
    bt = bt[bt["market"] == "us"]
    r_full = float((bt.t_full / bt.rec_full).median())
    fit = pd.read_parquet(ARTIFACTS_DIR / "benchmark" / "fit_timing.parquet")
    fit = (
        fit[(fit.schema == "ext") & (fit.target == "resid")].groupby("arm").mean(numeric_only=True)
    )
    M = _load_market("us")
    tf = M["time_full"] * r_full
    warm_full = np.mean([tf[_warmup(M["eval_ids"], s, W)].sum() for s in range(30)])
    r3000 = np.mean(
        [
            tf[rng_for("random_equal_budget", "us", s).permutation(M["eval_ids"])[: W + B]].sum()
            for s in range(30)
        ]
    )
    base_sel = pd.read_parquet(run_dir("us", "base") / "selections.parquet")
    r2000 = (
        base_sel[base_sel.arm == "random"]
        .groupby("seed")
        .formula_id.apply(lambda x: tf[x.to_numpy()].sum())
        .mean()
    )
    sel = pd.read_parquet(run_dir("us", "ext_mmr_resid") / "selections.parquet")
    sel = sel[sel["lam"] == lam]
    cells = pd.read_csv(ARTIFACTS_DIR / "analysis" / "equal_budget_cells.csv")
    union = cells[cells.comparison == "b_warmup_union"].set_index("cell")["mean"]
    rows = [
        ("Random 2,000 (7.2)", 0, 0, 0, r2000, None),
        ("Random 3,000 (8.4)", 0, 0, 0, r3000, union["random3000"]),
    ]
    for arm in ("lightgbm", "xgboost", "catboost", "random_forest"):
        picks = (
            sel[sel.arm == arm]
            .groupby("seed")
            .formula_id.apply(lambda x: tf[x.to_numpy()].sum())
            .mean()
        )
        f = fit.loc[arm]
        rows.append(
            (
                f"{ARM_LABEL[arm]} ext·MMR·resid (b)",
                warm_full,
                f.t_target + f.t_fit + f.t_predict,
                f.t_mmr,
                picks,
                union[f"{arm}|ext|mmr|resid"],
            )
        )
    stages = ["Warm-up backtest (full)", "Fit + tune + predict", "MMR", "Selected backtest"]
    stage_colors = ["#2a78d6", "#eb6834", "#1baf7a", "#86b6ef"]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for i, row in enumerate(rows):
        left = 0.0
        for k, v in enumerate(row[1:5]):
            if v > 0:
                ax.barh(
                    i,
                    v,
                    left=left,
                    color=stage_colors[k],
                    height=0.6,
                    edgecolor=SURFACE,
                    linewidth=2,
                    label=stages[k] if i == 2 else None,
                )
            left += v
        txt = f"{left:,.0f}s"
        if row[5] is not None:
            txt += f"  ·  {left / row[5]:.0f}s per unique pass ({row[5]:.1f})"
        ax.annotate(
            txt,
            (left, i),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=INK_2,
        )
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 4800)
    ax.set_xlabel("Wall time per search, seconds (single process, single thread; corrected)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=4)
    ax.set_title(
        "Time per search, US (design 8.5); unique passes counted with warm-up (8.4 b)", loc="left"
    )
    fig.tight_layout()
    fig.savefig(OUT / "time_per_search.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ 5 US vs JP Sharpe


def us_jp_sharpe() -> None:
    from scipy.stats import spearmanr

    us = directed(pd.read_parquet(output_path("results_is_test", "us")))
    jp_res = pd.read_parquet(output_path("results_is_test", "jp"))
    jp = directed(jp_res)
    ok = jp_res["filter_pass"].to_numpy()
    x, y = us["is_sharpe"].to_numpy()[ok], jp["is_sharpe"].to_numpy()[ok]
    rho = spearmanr(x, y).statistic
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    hb = ax.hexbin(
        x,
        y,
        gridsize=70,
        bins="log",
        cmap="Blues",
        mincnt=1,
        extent=(-6, 2.5, -6, 2.5),
        linewidths=0,
    )
    ax.plot([-6, 2.5], [-6, 2.5], color=INK_3, lw=1, ls=":")
    th = {
        m: json.loads(output_path("thresholds", m).read_text("utf-8"))["thresholds"]["is_sharpe"]
        for m in ("us", "jp")
    }
    ax.axvline(th["us"], color=INK_2, lw=1, ls="--")
    ax.axhline(th["jp"], color=INK_2, lw=1, ls="--")
    ax.set_xlabel("US directed IS net Sharpe (2016–2021)")
    ax.set_ylabel("Japan directed IS net Sharpe (2016–2021)")
    ax.set_title(f"Same formula, two markets (n = {ok.sum():,}, Spearman {rho:.2f})", loc="left")
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("formulas per cell (log)", color=INK_2, fontsize=8)
    cb.outline.set_visible(False)
    ax.annotate(
        "dashed: IS pass thresholds",
        (0.02, 0.97),
        xycoords="axes fraction",
        fontsize=7.5,
        color=INK_2,
        va="top",
    )
    fig.tight_layout()
    fig.savefig(OUT / "us_jp_sharpe.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ 7 MMR lambda


def mmr_lambda(lambdas: list[float], B: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.0), sharey=True)
    xs = [0.0, *lambdas]
    for ax, market in zip(axes, ("us", "jp"), strict=True):
        base = pd.read_parquet(run_dir(market, "base") / "metrics.parquet")
        mmr = pd.read_parquet(run_dir(market, "mmr") / "metrics.parquet")
        rnd = base[base.arm == "random"][f"unique_at_{B}"].mean()
        ax.axhline(rnd, color=RANDOM_COLOR, ls="--", lw=1.2)
        ax.text(xs[0], rnd + 0.25, f"Random arm {rnd:.1f}", fontsize=7.5, color=INK_2, va="bottom")
        ends = []
        for arm in MODEL_ARMS:
            ys = [base[base.arm == arm][f"unique_at_{B}"].mean()]
            ys += [
                mmr[(mmr.arm == arm) & (mmr.lam == lam)][f"unique_at_{B}"].mean() for lam in lambdas
            ]
            ax.plot(xs, ys, color=color_of(arm), marker="o", ms=5, label=ARM_LABEL[arm])
            ends.append((ys[-1], f"{ARM_LABEL[arm]} {ys[-1]:.1f}"))
        ax.set_xticks(xs, ["top-B\n(λ=0)", *[f"λ={v:g}" for v in lambdas]])
        ax.set_xlim(-0.08, xs[-1] + 0.45)
        ax.set_ylim(8, 26)
        end_labels(ax, xs[-1], ends)
        ax.set_title(f"{MARKET_LABEL[market]} — base · raw target", loc="left", fontsize=10)
        ax.set_ylabel(f"Unique passes at B = {B} (mean of 30 seeds)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5)
    fig.suptitle("MMR diversity weight λ (exploratory, design 7.5)", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "mmr_lambda.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ 9 cluster vs unique


def cluster_vs_unique() -> None:
    s = pd.read_csv(ARTIFACTS_DIR / "analysis" / "cluster_diagnostics_summary.csv")
    parts = s["cell"].str.split("|", expand=True)
    s["arm"], s["target"] = parts[0], parts[3]
    fig, ax = plt.subplots(figsize=(7, 5))
    groups = (
        ("raw", "Raw target", "#2a78d6"),
        ("resid", "Residual target", "#eb6834"),
    )
    rnd = s[s.arm.isin(["random", "struct_random"])]
    models = s[~s.arm.isin(["random", "struct_random"])]
    for key, label, col in groups:
        g = models[models.target == key]
        ax.scatter(
            g.high_corr_pairs * 100,
            g.unique,
            s=40,
            color=col,
            label=label,
            edgecolor=SURFACE,
            linewidth=1.5,
            zorder=3,
        )
    ax.scatter(
        rnd.high_corr_pairs * 100,
        rnd.unique,
        s=60,
        marker="D",
        color=RANDOM_COLOR,
        label="Random / struct. random",
        edgecolor=SURFACE,
        linewidth=1.5,
        zorder=3,
    )
    ax.set_xlabel("Selected pairs with |IS PnL corr| ≥ 0.7, % (mean of 30 seeds)")
    ax.set_ylabel("Unique passes at B = 2,000")
    ax.set_title(
        "US: redundancy among picks vs unique passes (design 8.6, descriptive)",
        loc="left",
        fontsize=10,
    )
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "cluster_vs_unique.png", dpi=150)
    plt.close(fig)


def main() -> None:
    logger = setup_logging("plots")
    style()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_experiment_config()
    ex, dv = cfg["experiment"], cfg["diversity"]
    lam, B = dv["primary_lambda"], ex["budget"]
    budgets = sorted({200, *ex["discovery_budgets"]})
    discovery_curves(lam, budgets)
    cell_comparison(lam, B)
    time_per_search(ex["warmup_size"], B, lam)
    us_jp_sharpe()
    mmr_lambda(sorted(dv["lambdas"]), B)
    cluster_vs_unique()
    logger.info("figures written to %s", OUT)


if __name__ == "__main__":
    main()
