# Surrogate Alpha Search

A study testing whether, under the same backtest budget, a tree-based surrogate model that predicts the performance of formulaic alpha candidates in advance finds more passing signals than random selection.

- A/B tests were run in the US (S&P 500).
- The model trained in the US was applied to Japan (Nikkei 225) as is, without retraining, to check whether it transfers.
- OOS (2024–2025) was opened only once, following a plan fixed in advance.

| Document | Contents |
|---|---|
| [`surrogate_alpha_search_design.md`](surrogate_alpha_search_design.md) | Experiment design (single source of truth, including full change history) |
| [`research_results.md`](research_results.md) | Research results (H1, H3, post-hoc exploration, H2, list of figures) |
| [`CLAUDE.md`](CLAUDE.md) | Implementation rules (leakage prevention, fixed values, backtest spec) |

The linked documents are written in Korean.

## Key Results

| Question | Result |
|---|---|
| **H1** Under the same budget (B = 2,000), does the surrogate model find more unique passing signals than random? | **No.** In both the US and Japan, every model found significantly fewer than random (random 24.1 vs. models 9.5–15.6, US) |
| Cause of failure (post-hoc diagnosis) | Top-predicted selections concentrate in a few clusters of the return space. About 61% of selected pairs have IS return correlation ≥ 0.7 (random: 10.5%), so 99% are removed by deduplication |
| **H3** Differences between models | No difference among the three boosting models (LightGBM, XGBoost, CatBoost); only Random Forest underperforms |
| Feature ablation (pre-registered) | Combined features significantly increased unique passes (+3.5), but still fell short of random |
| Post-hoc exploration: residual target + extended features + MMR | Significantly beats random in the US (35.6 vs. 24.1). Same direction in Japan, but not significant after correction |
| **H2** OOS quality (pre-registered cells) | **Inconclusive** under the pre-fixed rules (2 inconclusive cells, 0 inferior cells). Median OOS Sharpe of signals chosen by tree models is 0.2–0.5 higher than random |
| H2 exploratory cell (residual target) | Inferior OOS quality in Japan. It finds more passing signals, but their OOS quality is lower than random |

Detailed figures, confidence intervals, and limitations are in [`research_results.md`](research_results.md).

## Experiment Setup Summary

- **Formula pool:** 40,000 formulas generated from a grammar. 6 fields, 16 operators, windows {5, 10, 20, 60, 120}, depth 2–5
- **Periods:** IS 2016–2021, Test 2022–2023, OOS 2024–2025. Data is downloaded from 2015; 2015 is used only as a lookback buffer
- **Backtest:** signal at day-t close → cross-sectional demeaning → execution at day t+1 close → return realized on day t+2. One-way cost 5bp
- **Pass criteria:** direction-adjusted IS net Sharpe ≥ threshold (US 0.35, Japan 0.40), same-direction Test net Sharpe ≥ 0.3 (Japan 0.0), IS turnover ≤ 50%, IS daily return correlation with existing passing signals < 0.7
- **Experiment:** warm-up W = 1,000, budget B = 2,000, 30 seeds. Tuned with 50 Optuna trials, cross-validated with GroupKFold that puts structurally identical formulas in the same group
- **Data:** free price and volume data from yfinance. Constituents taken from a pinned Wikipedia revision

## Installation

Requires Python 3.11 or later, using a project-specific virtual environment.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]" matplotlib   # Windows
# .venv/bin/python -m pip install -e ".[dev]" matplotlib     # macOS / Linux
```

Computation assumes CPU only; no GPU is used.

## Execution Order

```bash
python -m src.data.universe --market us          # Collect constituents (called automatically by download if missing)
python -m src.data.download --market us          # Download, clean, and cache prices and volumes
python -m src.data.download --market jp
python -m src.generator.generate                 # 40,000-formula pool → artifacts/pool_v1.parquet
python -m src.features.ast_features              # Extract base and extended features
python -m src.backtest.engine --market us        # Full-pool backtest (IS/Test and OOS saved separately)
python -m src.backtest.engine --market jp
python -m src.experiment.calibrate --market us   # Calibrate IS threshold via random-arm simulation
python -m src.experiment.calibrate --market jp
python -m src.experiment.replay --market us --schema base     # H1, H3 (30 seeds × 6 arms)
python -m src.experiment.replay --market us --schema ext      # Feature ablation
python -m src.experiment.replay --market jp --schema base     # Japan transfer
python -m src.experiment.replay_mmr --market us               # Post-hoc exploration: MMR
python -m src.experiment.replay --market us --schema ext --target resid   # Post-hoc exploration: residual target
python -m src.analysis.stats                     # Key metrics and tests
python -m src.analysis.equal_budget              # Equal-total-budget robustness analysis
python -m src.analysis.cluster_diagnostics       # Post-hoc cluster concentration diagnostics
python -m src.analysis.benchmark                 # Timing benchmark under uniform conditions (run standalone)
python -m src.analysis.plots                     # Figures
pytest
```

- All commands above are run with the Python in `.venv` (Windows: `.venv/Scripts/python`).
- Replay and MMR run each variant via combinations of `--schema base|ext`, `--target raw|resid`, and `--market us|jp`.
- The full set of combinations used is in Sections 7.4–7.6 of the design document.

### OOS (H2)

```bash
python -m src.analysis.oos_report --dry-run-test-period        # Trial run on the Test period (does not read OOS)
python -m src.analysis.oos_report --confirm-single-oos-run     # The single actual OOS run
```

In this study, OOS has **already been run once and is locked.** The lock is enforced by the following mechanisms.

- **Read restriction:** Only `oos_report.py` reads the OOS result files, and tests verify this.
- **Run conditions:** It refuses to run if the plan document or code has uncommitted changes, and records the commit hash it ran on.
- **Rerun prevention:** When a run finishes, `artifacts/oos_locked/RUN_COMPLETE.json` is created; if this file exists, it will not run again.

## Repository Structure

```
config/          grammar.yaml (grammar), experiment.yaml (fixed values, seeds, criteria)
universe/        Constituent snapshots (universe_{us,jp}_v1.csv)
src/common/      Paths, config, logging, seed derivation
src/data/        Constituent collection, download and cleaning, panels
src/generator/   Grammar, canonical form and hashing, signal filters, pool generation
src/features/    AST features (formula strings only)
src/backtest/    numba kernels, operators, metrics, full backtest engine
src/surrogate/   Model definitions, tuning, heuristics, residual target
src/experiment/  Pass criteria, threshold calibration, accessor, arms, replay, MMR
src/analysis/    Statistics, robustness analysis, diagnostics, benchmark, H2 rehearsal, OOS report, figures
tests/           Leakage, canonical form, determinism, grammar, accessor, OOS isolation tests
```

`data/` and `artifacts/` are excluded from git.

## Reproducibility

- All randomness is derived from the master seed (20261101) in `config/experiment.yaml`. Arms with the same seed share the same warm-up.
- Tags
  - `h2-plan-frozen`: the commit that fixed the H2 plan before opening OOS. The actual OOS run was also executed at this commit
  - `v1.0-results`: final results and figures
- Identical code does not guarantee identical results, because yfinance adjusted prices can differ depending on when the data is downloaded. The data and artifacts used in the study are not included in the repository.

## Limitations

- **Data:** Free data was used; constituents are fixed to the current membership, which introduces survivorship bias; adjusted prices may contain errors.
- **Costs:** A simple one-way cost of 5bp is assumed.
- **Scope:** Results are limited to the single pool produced by this grammar and generator. OOS covers only two years, and the 30 seeds share the same pool and period, so they are not independent.
- **Post-hoc analyses:** MMR, the residual target, the equal-total-budget analysis, and the cluster diagnostics are post-hoc analyses added after seeing the H1 results. Each had its settings fixed before running, and they are reported separately from the pre-registered results.
