import numpy as np
import pytest

from src.common.seeds import rng_for
from src.generator.canonical import canonicalize, formula_hash, skeleton
from src.generator.generate import Sampler
from src.generator.grammar import (
    GrammarError,
    cumulative_lookback,
    depth,
    load_grammar,
    node_count,
    parse,
    validate,
)

G = load_grammar()


# ------------------------------------------------------------- canonical form


def test_commutative_operands_share_canonical_form_and_hash():
    a = parse("add(ts_mean(close, 5), rank(volume))")
    b = parse("add(rank(volume), ts_mean(close, 5))")
    assert canonicalize(a) == canonicalize(b)
    assert formula_hash(a) == formula_hash(b)
    m1 = parse("ts_corr(volume, close, 20)")
    m2 = parse("ts_corr(close, volume, 20)")
    assert formula_hash(m1) == formula_hash(m2)


def test_non_commutative_operands_differ():
    assert formula_hash(parse("sub(open, close)")) != formula_hash(parse("sub(close, open)"))


def test_skeleton_ignores_windows():
    assert skeleton(parse("rank(ts_mean(close, 5))")) == skeleton(parse("rank(ts_mean(close, 20))"))
    a = parse("add(ts_mean(close, 120), ts_mean(close, 5))")
    b = parse("add(ts_mean(close, 5), ts_mean(close, 60))")
    assert skeleton(a) == skeleton(b)


def test_parse_roundtrip():
    text = "rank(ts_corr(close, volume, 20))"
    assert parse(text).text == text


# ------------------------------------------------------------- grammar rules


@pytest.mark.parametrize(
    "text",
    [
        "rank(rank(close))",
        "abs(abs(returns))",
        "sign(sign(returns))",
        "ts_delay(ts_delay(close, 5), 5)",
        "zscore(rank(close))",
        "rank(zscore(close))",
        "abs(rank(close))",
        "sign(abs(returns))",
        "sub(close, close)",
        "div(volume, volume)",
        "ts_corr(close, close, 20)",
        "add(close, volume)",  # unit mismatch
        "add(mul(close, volume), close)",  # mixed unit in add
        "ts_std(close, 5)",  # min window 10
        "ts_corr(close, volume, 5)",
        "close",  # depth 1
    ],
)
def test_forbidden_formulas_rejected(text):
    with pytest.raises(GrammarError):
        validate(parse(text), G)


@pytest.mark.parametrize(
    "text",
    [
        "rank(ts_mean(close, 5))",
        "div(close, ts_mean(close, 20))",
        "rank(mul(close, volume))",
        "add(rank(close), returns)",
        "ts_corr(close, volume, 10)",
    ],
)
def test_valid_formulas_accepted(text):
    validate(parse(text), G)


def test_lookback_limit():
    with pytest.raises(GrammarError):
        validate(parse("ts_mean(ts_mean(ts_mean(close, 120), 120), 20)"), G)


@pytest.mark.parametrize("target", [2, 3, 4, 5])
def test_sampler_never_emits_rule_violations(target):
    sampler = Sampler(G, rng_for("test", target))
    produced = 0
    for _ in range(3000):
        node, reason = sampler.sample(target)
        if node is None:
            continue
        produced += 1
        validate(node, G)  # raises on any violation
        assert depth(node) == target
        assert node_count(node) <= G.max_nodes
        assert cumulative_lookback(node) <= G.max_lookback
        assert canonicalize(node) == node
        for parent, child in node.iter_edges():
            assert (parent.op, child.op) not in G.forbidden_nesting
    assert produced > 100


def test_sampler_is_deterministic():
    a = [Sampler(G, rng_for("det")).sample(4) for _ in range(1)]
    s1, s2 = Sampler(G, rng_for("det")), Sampler(G, rng_for("det"))
    seq1 = [s1.sample(4)[0] for _ in range(200)]
    seq2 = [s2.sample(4)[0] for _ in range(200)]
    assert seq1 == seq2
    assert a  # smoke


def test_category_mix_roughly_matches_config():
    sampler = Sampler(G, rng_for("mix"))
    cats = {"ts": 0, "cs": 0, "arith": 0}
    for _ in range(4000):
        node, _ = sampler.sample(3)
        if node is not None:
            cats[G.ops[node.op]["category"]] += 1
    total = sum(cats.values())
    # Root ops are drawn before any rejection; rejection skews the mix a bit.
    assert cats["ts"] / total > 0.2 and cats["arith"] / total > 0.2 and cats["cs"] / total > 0.05


def test_rng_streams_independent():
    assert rng_for("a").random() != rng_for("b").random()
    assert np.isclose(rng_for("a").random(), rng_for("a").random())


def test_depth2_enumeration_is_complete_and_valid():
    from src.generator.generate import enumerate_depth

    nodes = enumerate_depth(G, 2)
    assert len(nodes) == len({n.text for n in nodes}) == 363
    for n in nodes:
        validate(n, G)
        assert depth(n) == 2
    # Every sampled depth-2 formula must be in the enumerated set.
    texts = {n.text for n in nodes}
    sampler = Sampler(G, rng_for("enum-check"))
    for _ in range(500):
        node, _ = sampler.sample(2)
        if node is not None:
            assert node.text in texts
