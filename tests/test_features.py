import ast
import inspect

from src.features import ast_features
from src.features.ast_features import base_columns, build_features, formula_features
from src.generator.grammar import load_grammar

G = load_grammar()


def test_feature_determinism():
    text = "rank(ts_corr(rank(close), ts_mean(volume, 20), 10))"
    a = formula_features(text, G, extended=True)
    b = formula_features(text, G, extended=True)
    assert a == b
    df1 = build_features([text] * 3, G, extended=False)
    assert (df1.iloc[0] == df1.iloc[2]).all()


def test_commutative_variants_share_features():
    a = formula_features("add(rank(close), ts_rank(volume, 5))", G, extended=True)
    b = formula_features("add(ts_rank(volume, 5), rank(close))", G, extended=True)
    assert a == b


def test_feature_values():
    f = formula_features("rank(ts_mean(close, 5))", G)
    assert f["depth"] == 3 and f["node_count"] == 3
    assert f["cnt_rank"] == 1 and f["cnt_ts_mean"] == 1 and f["cnt_close"] == 1
    assert f["win_min"] == f["win_max"] == f["win_mean"] == 5 and f["n_windows"] == 1
    assert f["root_rank"] == 1 and f["cs_at_root"] == 1 and f["ts_nest_depth"] == 1
    assert abs(f["ts_ratio"] - 0.5) < 1e-12 and abs(f["cs_ratio"] - 0.5) < 1e-12
    assert list(f) == base_columns(G)


def test_extended_features_distinguish_nesting_order():
    a = formula_features("rank(ts_delta(close, 5))", G, extended=True)
    b = formula_features("ts_delta(rank(close), 5)", G, extended=True)
    base_cols = [c for c in base_columns(G) if not c.startswith("root_")]
    base_cols.remove("cs_at_root")
    assert all(a[c] == b[c] for c in base_cols if c not in ("ts_nest_depth",))
    assert a["pc_rank__ts_delta"] == 1 and b["pc_ts_delta__rank"] == 1


def test_features_never_touch_price_data():
    """Rule 2: features are computed from the formula string only."""
    tree = ast.parse(inspect.getsource(ast_features))
    imported = {(n.module or "") for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | {
        a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names
    }
    assert not any(m.startswith(("src.data", "src.backtest")) for m in imported)
