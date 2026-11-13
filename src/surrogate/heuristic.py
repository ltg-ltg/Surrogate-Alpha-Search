"""Component-average heuristic (design 7.1).

Components of a formula: the distinct operators, fields and windows it contains.
Component score = (n * mean_y + m * global_mean) / (n + m), where n is the number of
warm-up formulas containing the component and mean_y their mean target (m = 5, fixed).
Unseen components get the global mean. A formula's score is the mean over its components.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.generator.grammar import Grammar, parse


def components(text: str, g: Grammar) -> frozenset[str]:
    node = parse(text, g)
    out = set()
    for n in node.iter_nodes():
        if n.is_leaf:
            out.add(f"field:{n.op}")
        else:
            out.add(f"op:{n.op}")
            if n.window is not None:
                out.add(f"win:{n.window}")
    return frozenset(out)


@dataclass
class ComponentHeuristic:
    m: float
    scores: dict[str, float] | None = None
    global_mean: float = 0.0

    def fit(self, comps: list[frozenset[str]], y: np.ndarray) -> ComponentHeuristic:
        self.global_mean = float(np.mean(y))
        sums: dict[str, float] = {}
        counts: dict[str, int] = {}
        for cs, v in zip(comps, y, strict=True):
            for c in cs:
                sums[c] = sums.get(c, 0.0) + float(v)
                counts[c] = counts.get(c, 0) + 1
        self.scores = {
            c: (sums[c] + self.m * self.global_mean) / (counts[c] + self.m) for c in sums
        }
        return self

    def predict(self, comps: list[frozenset[str]]) -> np.ndarray:
        assert self.scores is not None
        return np.array(
            [np.mean([self.scores.get(c, self.global_mean) for c in cs]) for cs in comps]
        )
