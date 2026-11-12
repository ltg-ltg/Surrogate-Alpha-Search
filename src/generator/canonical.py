"""Canonical form, hashing and structural groups (design 4.7, 6.3)."""

from __future__ import annotations

import hashlib

from src.common.config import load_grammar_config
from src.generator.grammar import Node


def _commutative_ops() -> frozenset[str]:
    ops = load_grammar_config()["operators"]
    return frozenset(name for name, spec in ops.items() if spec.get("commutative"))


_COMMUTATIVE = _commutative_ops()


def canonicalize(node: Node) -> Node:
    """Bottom-up: sort operands of commutative ops (add, mul, ts_corr) by text."""
    if node.is_leaf:
        return node
    children = tuple(canonicalize(c) for c in node.children)
    if node.op in _COMMUTATIVE:
        children = tuple(sorted(children, key=lambda c: c.text))
    return Node(node.op, children, node.window)


def formula_hash(node: Node) -> str:
    """Hash of the canonical text (callers should pass a canonical node)."""
    return hashlib.sha1(canonicalize(node).text.encode("utf-8")).hexdigest()[:16]


def skeleton(node: Node) -> str:
    """Canonical text with every window replaced by 'w' (GroupKFold group key).

    Commutative operands are re-sorted on their skeleton text so that formulas that
    differ only in windows always map to the same group.
    """
    if node.is_leaf:
        return node.op
    args = [skeleton(c) for c in node.children]
    if node.op in _COMMUTATIVE:
        args.sort()
    if node.window is not None:
        args.append("w")
    return f"{node.op}({', '.join(args)})"
