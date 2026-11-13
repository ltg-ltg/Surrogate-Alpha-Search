import numpy as np

from src.generator.grammar import load_grammar
from src.surrogate.heuristic import ComponentHeuristic, components

G = load_grammar()


def test_components():
    assert components("rank(ts_mean(close, 5))", G) == {
        "op:rank",
        "op:ts_mean",
        "field:close",
        "win:5",
    }


def test_smoothing_and_unseen():
    comps = [components("rank(close)", G), components("rank(volume)", G)]
    y = np.array([1.0, 3.0])
    h = ComponentHeuristic(m=5).fit(comps, y)
    g = 2.0
    assert h.scores["op:rank"] == (4.0 + 5 * g) / (2 + 5)
    assert h.scores["field:close"] == (1.0 + 5 * g) / (1 + 5)
    pred = h.predict([components("zscore(open)", G)])  # all unseen -> global mean
    assert pred[0] == g
