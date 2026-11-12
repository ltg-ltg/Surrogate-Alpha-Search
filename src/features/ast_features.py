"""AST features computed from the formula string only (design 6.1, 7.4).

Base schema (feature_schema_v1.json):
  size       depth, node_count, n_unique_ops, n_unique_fields
  op counts  cnt_<op> for all 16 operators
  ratios     ts_ratio, cs_ratio, arith_ratio (share of operator nodes)
  fields     cnt_<field> for all 6 fields
  windows    win_min, win_max, win_mean, n_windows (distinct window values; 0 if none)
  structure  root_<op> one-hot, cs_at_root, ts_nest_depth (max ts ops on a root-leaf path)
Extended schema (feature_schema_v1_ext.json, ablation only) adds
  pc_<parent>__<child>  parent-child pair counts (child = operator or field)
  ow_<op>__<window>     time-series operator x window pair counts
Zero-variance columns over the pool are dropped once and the schema is frozen.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter

import numpy as np
import pandas as pd

from src.common.config import ARTIFACTS_DIR, PROJECT_ROOT, load_experiment_config, setup_logging
from src.generator.canonical import canonicalize
from src.generator.grammar import Grammar, Node, depth, load_grammar, node_count, parse


def _ts_nest_depth(node: Node, g: Grammar) -> int:
    own = 1 if (not node.is_leaf and g.ops[node.op]["category"] == "ts") else 0
    return own + max((_ts_nest_depth(c, g) for c in node.children), default=0)


def base_columns(g: Grammar) -> list[str]:
    ops, fields = sorted(g.ops), sorted(g.fields)
    return (
        ["depth", "node_count", "n_unique_ops", "n_unique_fields"]
        + [f"cnt_{op}" for op in ops]
        + ["ts_ratio", "cs_ratio", "arith_ratio"]
        + [f"cnt_{f}" for f in fields]
        + ["win_min", "win_max", "win_mean", "n_windows"]
        + [f"root_{op}" for op in ops]
        + ["cs_at_root", "ts_nest_depth"]
    )


def ext_columns(g: Grammar) -> list[str]:
    ops, fields = sorted(g.ops), sorted(g.fields)
    pc = [f"pc_{p}__{c}" for p in ops for c in ops + fields]
    ow = [f"ow_{op}__{w}" for op in ops if g.ops[op]["window"] for w in g.allowed_windows(op)]
    return pc + ow


def formula_features(text: str, g: Grammar, extended: bool = False) -> dict[str, float]:
    """Deterministic feature dict for one formula (canonicalized first)."""
    node = canonicalize(parse(text, g))
    ops = [n for n in node.iter_nodes() if not n.is_leaf]
    leaves = [n for n in node.iter_nodes() if n.is_leaf]
    op_cnt = Counter(n.op for n in ops)
    field_cnt = Counter(n.op for n in leaves)
    windows = [n.window for n in ops if n.window is not None]
    cats = Counter(g.ops[n.op]["category"] for n in ops)
    n_ops = max(len(ops), 1)

    f: dict[str, float] = {
        "depth": depth(node),
        "node_count": node_count(node),
        "n_unique_ops": len(op_cnt),
        "n_unique_fields": len(field_cnt),
    }
    for op in sorted(g.ops):
        f[f"cnt_{op}"] = op_cnt.get(op, 0)
    f["ts_ratio"] = cats.get("ts", 0) / n_ops
    f["cs_ratio"] = cats.get("cs", 0) / n_ops
    f["arith_ratio"] = cats.get("arith", 0) / n_ops
    for fld in sorted(g.fields):
        f[f"cnt_{fld}"] = field_cnt.get(fld, 0)
    f["win_min"] = min(windows) if windows else 0
    f["win_max"] = max(windows) if windows else 0
    f["win_mean"] = float(np.mean(windows)) if windows else 0.0
    f["n_windows"] = len(set(windows))
    for op in sorted(g.ops):
        f[f"root_{op}"] = 1 if node.op == op else 0
    f["cs_at_root"] = 1 if g.ops[node.op]["category"] == "cs" else 0
    f["ts_nest_depth"] = _ts_nest_depth(node, g)

    if extended:
        pc = Counter(f"pc_{p.op}__{c.op}" for p, c in node.iter_edges())
        ow = Counter(f"ow_{n.op}__{n.window}" for n in ops if n.window is not None)
        for col in ext_columns(g):
            f[col] = pc.get(col, 0) + ow.get(col, 0)
    return f


def build_features(formulas: list[str], g: Grammar, extended: bool) -> pd.DataFrame:
    cols = base_columns(g) + (ext_columns(g) if extended else [])
    rows = [formula_features(t, g, extended) for t in formulas]
    return pd.DataFrame(rows, columns=cols).astype(np.float64)


def freeze_schema(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Drop zero-variance columns (one-time, after pool generation)."""
    keep = [c for c in df.columns if df[c].nunique(dropna=False) > 1]
    dropped = [c for c in df.columns if c not in keep]
    return keep, dropped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    cfg = load_experiment_config()
    logger = setup_logging("features")
    g = load_grammar()
    t0 = time.perf_counter()

    pool = pd.read_parquet(PROJECT_ROOT / cfg["pool"]["file"])
    formulas = pool["formula"].tolist()
    for extended, suffix in ((False, ""), (True, "_ext")):
        df = build_features(formulas, g, extended)
        keep, dropped = freeze_schema(df)
        schema = {
            "version": "v1" + suffix,
            "columns": keep,
            "dropped_zero_variance": dropped,
            "n_columns": len(keep),
        }
        schema_path = ARTIFACTS_DIR / f"feature_schema_v1{suffix}.json"
        if schema_path.exists():
            frozen = json.loads(schema_path.read_text(encoding="utf-8"))
            if frozen["columns"] != keep:
                raise RuntimeError(f"{schema_path.name} is frozen and differs from this pool")
        schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
        out = df[keep].copy()
        out.insert(0, "formula_id", pool["formula_id"].to_numpy())
        out.to_parquet(ARTIFACTS_DIR / f"features_v1{suffix}.parquet", index=False)
        logger.info(
            "features%s: %d columns kept, %d zero-variance dropped %s",
            suffix,
            len(keep),
            len(dropped),
            dropped if len(dropped) <= 20 else f"{dropped[:20]}...",
        )
    logger.info("done in %.1fs", time.perf_counter() - t0)


if __name__ == "__main__":
    main()
