"""TFNE/2 normal form for the whole-system checkpoint architecture.

Authority: docs/tfne/TFNE_V2_SPEC.md (TFNE/2 LANGUAGE = SEALED). This module implements the
subset the checkpoint architecture needs -- `:=`, `{}`, `^n`, `O[k]`, `X[k]`, `.`, `N/P/G`,
`in`/`out` -- over a typed programmatic representation. It is not a parser: it accepts
normalized semantics, never source strings. Text parsing is a separate future capability.

What it produces is `NF(A)`: objects with their declarations, composite boundaries with
structural keys, ordered sequences with their adjacency rules, cross relations, frontiers, and
the generated projection set `G_0`. Canonical traversal order is typed-natural, never
declaration order, so reordering the input cannot change realization identity.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field

MECHANISM_DEFAULT = "AMPA"


def natural_key(name):
    """Typed path token: (stem, numeric suffix or index, remainder). L2 < L10; SEG.2 < SEG.10.

    Identical to scripts/tfne/v2_conformance.py natural_key, which the sealed spec's appendix
    conformance block is checked against.
    """
    m = re.fullmatch(r"([^0-9]*)([0-9]*)(.*)", str(name))
    stem, num, rest = m.group(1), m.group(2), m.group(3)
    return (stem, int(num) if num else -1, rest)


def path_key(path):
    return tuple(natural_key(p) for p in path)


@dataclass(frozen=True)
class Address:
    """A resolved TFNE address: object path plus optional layer and cell class."""
    path: tuple[str, ...]
    layer: str | None = None
    cls: str | None = None

    def __str__(self):
        return ".".join([*self.path, *(p for p in (self.layer, self.cls) if p)])


@dataclass(frozen=True)
class Projection:
    """One generated projection identity. Direction is structural, never a claim about sign."""
    source: Address
    target: Address
    mechanism: str
    relation: str          # the rule invocation that generated it, e.g. "O[fffb]#0.ff"

    def identity(self):
        return (str(self.source), str(self.target), self.mechanism)

    def as_dict(self):
        return {"source": str(self.source), "target": str(self.target),
                "mechanism": self.mechanism, "relation": self.relation}


@dataclass(frozen=True)
class Instance:
    """One instance of a definition, e.g. V1.1 := CTX[jomission_v0]."""
    path: tuple[str, ...]
    definition: str
    decl: dict             # the definition's D4 content

    def address(self, spec):
        """Resolve an interface entry like 'L4.E' against this instance."""
        parts = str(spec).split(".")
        if len(parts) != 2:
            raise ValueError(f"E_ADDRESS_UNRESOLVED: {spec!r} is not <layer>.<class>")
        layer, cls = parts
        if layer not in self.decl["layers"]:
            raise ValueError(f"E_ADDRESS_UNRESOLVED: {self.name()}.{layer}")
        if cls not in self.decl["classes"]:
            raise ValueError(f"E_ADDRESS_UNRESOLVED: {self.name()}.{layer}.{cls}")
        if self.decl["P"][layer].get(cls, 0.0) == 0.0:
            raise ValueError(f"E_ADDRESS_UNRESOLVED: {self.name()}.{layer}.{cls} is an absent population")
        return Address(self.path, layer, cls)

    def name(self):
        return ".".join(self.path)


@dataclass
class Composite:
    """`{E}`: a preserved structural boundary. Never silently flattened."""
    members: list[Instance]
    relation: str | None = None          # the X rule joining the members, or None
    label: str | None = None
    _key: str = field(default="", init=False)

    def instances(self):
        return sorted(self.members, key=lambda i: path_key(i.path))

    def serialize(self):
        """Deterministic canonical serialization of NF(E): the authority for identity."""
        return json.dumps({"members": [{"path": list(i.path), "definition": i.definition}
                                       for i in self.instances()],
                           "relation": self.relation}, sort_keys=True, separators=(",", ":"))

    def structural_key(self):
        return self.label or self.serialize()

    def receipt_hash(self):
        """A compact identity receipt. Never the ordering key -- the serialization is."""
        return hashlib.sha256(self.serialize().encode("utf-8")).hexdigest()[:16]

    def frontier(self, which):
        """in({A X B}) = in(A) u in(B); out likewise. X may override; `lat` does not."""
        out = []
        for inst in self.instances():
            for spec in inst.decl[which]:
                out.append(inst.address(spec))
        if not out:
            raise ValueError(f"E_FRONTIER_UNRESOLVED: {self.label} has no {which}")
        return out


@dataclass
class OChain:
    """An unbraced O chain normalizes to an adjacency-labelled ordered sequence.

    `A O[k] B O[j] C` -> `(A, k, B, j, C)`: each rule belongs to its own adjacency, and no
    relation is generated between non-adjacent members. That invariant is the whole point of
    the architecture receipt.
    """
    items: list[Composite]
    rules: list[str]

    def __post_init__(self):
        if len(self.rules) != len(self.items) - 1:
            raise ValueError("E_AMBIGUOUS_EXPANSION: one rule per adjacency is required")

    def adjacencies(self):
        return [(i, self.items[i], self.rules[i], self.items[i + 1]) for i in range(len(self.rules))]


def expand_x(composite, rules):
    """`X[k]` over a composite's members. Pairwise, symmetric, and never inter-level."""
    if composite.relation is None:
        return []
    rule = rules[composite.relation]
    insts = composite.instances()
    if len(insts) > 2 and not rule.get("associative", False):
        raise ValueError(f"E_X_GROUPING: {composite.relation} declares no associative policy")
    out = []
    for a in range(len(insts)):
        for b in range(a + 1, len(insts)):
            left, right = insts[a], insts[b]
            for clause in rule["clauses"]:
                src = left.address(clause["source"])
                dst = right.address(clause["target"])
                mech = clause.get("mechanism", MECHANISM_DEFAULT)
                rel = f"X[{composite.relation}]@{composite.label}"
                out.append(Projection(src, dst, mech, rel))
                if clause.get("bidirectional", False):
                    out.append(Projection(right.address(clause["source"]),
                                          left.address(clause["target"]), mech, rel))
    return out


def _pairs(left, right, policy):
    """Which (left instance, right instance) pairs an O adjacency relates."""
    li, ri = left.instances(), right.instances()
    if policy == "derived_frontier_union":
        return [(a, b) for a in li for b in ri]
    if policy == "matched_instance":
        by_index = {}
        for a in li:
            by_index.setdefault(a.path[-1], []).append(a)
        out = []
        for b in ri:
            match = by_index.get(b.path[-1])
            if match is None:
                raise ValueError(f"E_FRONTIER_UNRESOLVED: no instance matching {b.name()} on the left")
            out.extend((a, b) for a in match)
        return out
    raise ValueError(f"E_AMBIGUOUS_EXPANSION: unknown mapping policy {policy!r}")


def expand_o(index, left, rule_name, right, rules):
    """`O[k]` over one adjacency. Generates only the clauses the rule declares."""
    rule = rules[rule_name]
    out = []
    for a, b in _pairs(left, right, rule["mapping"]):
        for clause in rule["clauses"]:
            frm, to = (a, b) if clause["direction"] == "forward" else (b, a)
            src = frm.address(clause["source"])
            dst = to.address(clause["target"])
            out.append(Projection(src, dst, clause.get("mechanism", MECHANISM_DEFAULT),
                                  f"O[{rule_name}]#{index}.{clause['name']}"))
    return out


def normalize(chain, rules, external=()):
    """Produce NF(A): boundaries, adjacencies, cross relations, frontiers and G_0."""
    projections = []
    for comp in chain.items:
        projections.extend(expand_x(comp, rules))
    for index, left, rule_name, right in chain.adjacencies():
        projections.extend(expand_o(index, left, rule_name, right, rules))
    for proj in external:
        projections.append(proj)

    seen, ordered = set(), []
    for p in sorted(projections, key=lambda p: (path_key(p.source.path), natural_key(p.source.layer or ""),
                                                natural_key(p.source.cls or ""), path_key(p.target.path),
                                                natural_key(p.target.layer or ""), natural_key(p.target.cls or ""),
                                                p.mechanism)):
        if p.identity() in seen:
            raise ValueError(f"E_AMBIGUOUS_EXPANSION: duplicate projection {p.identity()}")
        seen.add(p.identity())
        ordered.append(p)

    instances = sorted((i for c in chain.items for i in c.members), key=lambda i: path_key(i.path))
    return {
     "objects": [{"path": list(i.path), "name": i.name(), "definition": i.definition} for i in instances],
     "composites": [{"label": c.label, "members": [i.name() for i in c.instances()],
                     "relation": c.relation, "serialization": c.serialize(),
                     "receipt_hash": c.receipt_hash(),
                     "in": [str(a) for a in c.frontier("in")],
                     "out": [str(a) for a in c.frontier("out")]} for c in chain.items],
     "o_adjacencies": [{"index": i, "left": left.label, "rule": rule, "right": right.label,
                        "mapping": rules[rule]["mapping"]}
                       for i, left, rule, right in chain.adjacencies()],
     "x_relations": [{"composite": c.label, "rule": c.relation,
                      "members": [i.name() for i in c.instances()]}
                     for c in chain.items if c.relation],
     "projections": [p.as_dict() for p in ordered],
     "_instances": instances,
     "_projection_objects": ordered,
    }


def non_adjacent_pairs(chain):
    """Every ordered pair of composites that an O chain must NOT relate."""
    labels = [c.label for c in chain.items]
    return [(labels[i], labels[j]) for i in range(len(labels)) for j in range(len(labels))
            if abs(i - j) > 1 and i != j]
