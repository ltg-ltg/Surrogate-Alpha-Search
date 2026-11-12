"""Formula AST, parser, unit inference and rule validation (design 4.1-4.5)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import cached_property

from src.common.config import load_grammar_config

PRICE, VOLUME, DIMLESS, MIXED = "price", "volume", "dimensionless", "mixed"


class GrammarError(ValueError):
    """Raised when a formula violates the grammar."""


@dataclass(frozen=True)
class Node:
    """A formula node. Leaves have `op` set to a field name and no children."""

    op: str
    children: tuple[Node, ...] = ()
    window: int | None = None

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def __str__(self) -> str:  # canonical text form used everywhere
        return self.text

    @cached_property
    def text(self) -> str:
        if self.is_leaf:
            return self.op
        args = [c.text for c in self.children]
        if self.window is not None:
            args.append(str(self.window))
        return f"{self.op}({', '.join(args)})"

    def iter_nodes(self):
        yield self
        for c in self.children:
            yield from c.iter_nodes()

    def iter_edges(self):
        """Yield (parent, child) pairs."""
        for c in self.children:
            yield self, c
            yield from c.iter_edges()


@dataclass(frozen=True)
class Grammar:
    fields: dict[str, str]
    windows: tuple[int, ...]
    ops: dict[str, dict]
    min_depth: int
    max_depth: int
    max_nodes: int
    max_lookback: int
    forbidden_nesting: frozenset[tuple[str, str]]
    forbidden_self_ops: frozenset[str]
    category_probs: dict[str, float]
    exhaustive_depths: tuple[int, ...]
    sampled_depths: tuple[int, ...]
    ops_by_category: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def allowed_windows(self, op: str) -> tuple[int, ...]:
        lo = self.ops[op].get("min_window", 0)
        return tuple(w for w in self.windows if w >= lo)


def load_grammar() -> Grammar:
    cfg = load_grammar_config()
    ops = cfg["operators"]
    by_cat: dict[str, list[str]] = {}
    for name, spec in ops.items():
        by_cat.setdefault(spec["category"], []).append(name)
    return Grammar(
        fields=dict(cfg["fields"]),
        windows=tuple(cfg["windows"]),
        ops=ops,
        min_depth=cfg["size"]["min_depth"],
        max_depth=cfg["size"]["max_depth"],
        max_nodes=cfg["size"]["max_nodes"],
        max_lookback=cfg["size"]["max_cumulative_lookback"],
        forbidden_nesting=frozenset(tuple(p) for p in cfg["forbidden_nesting"]),
        forbidden_self_ops=frozenset(cfg["forbidden_self_ops"]),
        category_probs=dict(cfg["sampling"]["category_probs"]),
        exhaustive_depths=tuple(cfg["sampling"]["exhaustive_depths"]),
        sampled_depths=tuple(cfg["sampling"]["sampled_depths"]),
        ops_by_category={k: tuple(v) for k, v in by_cat.items()},
    )


# ---------------------------------------------------------------- size metrics


def depth(node: Node) -> int:
    """Nodes on the longest root-to-leaf path; a bare field has depth 1."""
    return 1 + max((depth(c) for c in node.children), default=0)


def node_count(node: Node) -> int:
    """Operators plus field leaves (window constants are not nodes)."""
    return sum(1 for _ in node.iter_nodes())


def cumulative_lookback(node: Node) -> int:
    """Max over root-to-leaf paths of the sum of windows along the path."""
    own = node.window or 0
    return own + max((cumulative_lookback(c) for c in node.children), default=0)


# ---------------------------------------------------------------- units


def infer_unit(node: Node, g: Grammar) -> str:
    if node.is_leaf:
        return g.fields[node.op]
    spec = g.ops[node.op]
    units = [infer_unit(c, g) for c in node.children]
    rule = spec["unit"]
    if rule == "same":
        return units[0]
    if rule == "dimless":
        return DIMLESS
    if rule == "same_binary":
        a, b = units
        if MIXED in (a, b):
            raise GrammarError(f"{node.op}: mixed-unit operand not allowed: {node}")
        if a != b:
            raise GrammarError(f"{node.op}: unit mismatch {a} vs {b}: {node}")
        return a
    if rule == "mixed":
        a, b = units
        if a == DIMLESS:
            return b
        if b == DIMLESS:
            return a
        if node.op == "div" and a == b and a != MIXED:
            return DIMLESS
        return MIXED
    raise GrammarError(f"unknown unit rule {rule!r}")


# ---------------------------------------------------------------- validation


def check_local(node: Node, g: Grammar) -> None:
    """Rules that only involve a node and its direct children (used during sampling)."""
    if node.is_leaf:
        if node.op not in g.fields:
            raise GrammarError(f"unknown field {node.op!r}")
        return
    if node.op not in g.ops:
        raise GrammarError(f"unknown operator {node.op!r}")
    spec = g.ops[node.op]
    if len(node.children) != spec["arity"]:
        raise GrammarError(f"{node.op}: expected {spec['arity']} operands")
    if spec["window"]:
        if node.window not in g.allowed_windows(node.op):
            raise GrammarError(f"{node.op}: window {node.window} not allowed")
    elif node.window is not None:
        raise GrammarError(f"{node.op}: takes no window")
    for c in node.children:
        if (node.op, c.op) in g.forbidden_nesting:
            raise GrammarError(f"forbidden nesting {node.op}({c.op}(...))")
    if node.op in g.forbidden_self_ops and node.children[0] == node.children[1]:
        raise GrammarError(f"{node.op} of identical operands: {node}")
    infer_unit(node, g)


def validate(node: Node, g: Grammar, canonical: bool = True) -> None:
    """Full grammar check. Self-operand checks compare canonical subtrees."""
    if canonical:
        from src.generator.canonical import canonicalize

        node = canonicalize(node)
    for n in node.iter_nodes():
        check_local(n, g)
    d = depth(node)
    if not g.min_depth <= d <= g.max_depth:
        raise GrammarError(f"depth {d} outside [{g.min_depth}, {g.max_depth}]")
    if node_count(node) > g.max_nodes:
        raise GrammarError(f"node count {node_count(node)} > {g.max_nodes}")
    if cumulative_lookback(node) > g.max_lookback:
        raise GrammarError(f"lookback {cumulative_lookback(node)} > {g.max_lookback}")


# ---------------------------------------------------------------- parser

_TOKEN = re.compile(r"\s*(?:([A-Za-z_][A-Za-z_0-9]*)|(\d+)|([(),]))")


def parse(text: str, g: Grammar | None = None) -> Node:
    g = g or load_grammar()
    tokens: list[tuple[str, str]] = []
    pos = 0
    text = text.strip()
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m or m.end() == pos:
            raise GrammarError(f"cannot tokenize at {pos}: {text!r}")
        ident, num, punct = m.groups()
        tokens.append(("id", ident) if ident else ("num", num) if num else ("p", punct))
        pos = m.end()
    i = 0

    def expect(kind: str, value: str | None = None) -> str:
        nonlocal i
        if i >= len(tokens) or tokens[i][0] != kind or (value and tokens[i][1] != value):
            raise GrammarError(f"parse error near token {i} in {text!r}")
        i += 1
        return tokens[i - 1][1]

    def node() -> Node:
        nonlocal i
        name = expect("id")
        if name in g.fields:
            return Node(name)
        if name not in g.ops:
            raise GrammarError(f"unknown name {name!r}")
        spec = g.ops[name]
        expect("p", "(")
        children = []
        for k in range(spec["arity"]):
            if k:
                expect("p", ",")
            children.append(node())
        window = None
        if spec["window"]:
            expect("p", ",")
            window = int(expect("num"))
        expect("p", ")")
        return Node(name, tuple(children), window)

    result = node()
    if i != len(tokens):
        raise GrammarError(f"trailing tokens in {text!r}")
    return result
