"""Tier 0: drop-in packet copies match their declared relation to the source of truth.

A packet ships a copy of a file this repository also runs and tests. Nothing otherwise stops
the two diverging, and the failure is silent: the packet keeps looking correct while handing
upstream a file that is not the one whose tests passed. The relation each copy claims is in
manifests/dropin_packets.json and is enforced here, including the honest case where a copy
was deliberately adapted and no byte correspondence exists.
"""

from __future__ import annotations

import ast
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "manifests" / "dropin_packets.json").read_text("utf-8"))
PACKETS = MANIFEST["packets"]
ENTRIES = [(p, f) for p in PACKETS for f in p["files"]]
IDS = [f"{p['dir'].split('/')[-1]}:{f['copy']}" for p, f in ENTRIES]


def _read(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


def _strip_docstring(src: str) -> str:
    """Drop the module docstring, which a packet rewrites for an upstream reader."""
    tree = ast.parse(src)
    doc = ast.get_docstring(tree, clean=False)
    if doc is None or not tree.body:
        return src
    first = tree.body[0]
    if not (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)):
        return src
    lines = src.splitlines(keepends=True)
    return "".join(lines[first.end_lineno:])


def test_manifest_relations_are_all_known():
    known = set(MANIFEST["relations"])
    for _p, f in ENTRIES:
        assert f["relation"] in known, f"{f['copy']}: unknown relation {f['relation']!r}"


@pytest.mark.parametrize("packet,entry", ENTRIES, ids=IDS)
def test_every_declared_copy_exists(packet, entry):
    assert (ROOT / packet["dir"] / entry["copy"]).is_file()
    if entry.get("source"):
        assert (ROOT / entry["source"]).is_file(), f"source of truth missing: {entry['source']}"


def test_every_shipped_python_file_is_declared():
    """An undeclared copy is the drift this guard exists to prevent."""
    for packet in PACKETS:
        shipped = {p.name for p in (ROOT / packet["dir"]).glob("*.py")}
        declared = {f["copy"] for f in packet["files"]}
        assert shipped == declared, (
            f"{packet['dir']}: shipped {sorted(shipped - declared)} undeclared, "
            f"declared {sorted(declared - shipped)} missing")


def check_relation(copy: str, source: str | None, entry: dict, where: str) -> list[str]:
    """Return the drift errors for one declared copy. Shared by the guard and its own test."""
    relation = entry["relation"]
    if relation == "adapted":
        # No byte correspondence is claimed. The manifest must say why and what it costs,
        # so the divergence is recorded rather than quietly tolerated.
        err = []
        if not entry.get("reason"):
            err.append(f"{where}: adapted without a reason")
        if not entry.get("consequence"):
            err.append(f"{where}: adapted without stating the cost")
        return err
    if relation == "verbatim":
        if copy != source:
            return [f"{where} has drifted from {entry['source']}. Re-copy it, or change its "
                    f"relation in manifests/dropin_packets.json."]
        return []
    if relation == "rewritten_imports":
        for upstream, local in entry["substitutions"]:
            if upstream not in copy:
                return [f"{where}: declared substitution absent: {upstream}"]
            copy = copy.replace(upstream, local)
        if _strip_docstring(copy) != _strip_docstring(source):
            return [f"{where} differs from {entry['source']} beyond its declared "
                    f"substitutions and its docstring."]
        return []
    return [f"{where}: unknown relation {relation!r}"]


@pytest.mark.parametrize("packet,entry", ENTRIES, ids=IDS)
def test_copy_matches_its_declared_relation(packet, entry):
    where = f"{packet['dir']}/{entry['copy']}"
    copy = _read(ROOT / packet["dir"] / entry["copy"])
    source = _read(ROOT / entry["source"]) if entry.get("source") else None
    assert check_relation(copy, source, entry, where) == []


@pytest.mark.parametrize("packet,entry", ENTRIES, ids=IDS)
def test_shipped_copy_imports_nothing_from_this_repository(packet, entry):
    """Whatever the relation, a file headed upstream cannot import jomission or tfne."""
    tree = ast.parse(_read(ROOT / packet["dir"] / entry["copy"]))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    bad = [n for n in names if n.split(".")[0] in ("jomission", "tfne")]
    assert bad == [], f"{packet['dir']}/{entry['copy']} imports {bad}"


@pytest.mark.parametrize("packet", PACKETS, ids=[p["dir"].split("/")[-1] for p in PACKETS])
def test_every_packet_has_integration_instructions(packet):
    assert (ROOT / packet["dir"] / "README.md").is_file()
    if packet.get("handout"):
        assert (ROOT / packet["handout"]).is_file()


def test_the_guard_catches_a_one_byte_change():
    """A guard that never fails is indistinguishable from no guard."""
    verbatim = [(p, f) for p, f in ENTRIES if f["relation"] == "verbatim"]
    assert verbatim, "no verbatim copy to exercise"
    packet, entry = verbatim[0]
    source = _read(ROOT / entry["source"])
    assert check_relation(source, source, entry, "x") == []
    assert check_relation(source + "# drift\n", source, entry, "x") != []


def test_the_guard_catches_an_unsubstituted_rewrite():
    rewritten = [(p, f) for p, f in ENTRIES if f["relation"] == "rewritten_imports"]
    assert rewritten, "no rewritten_imports copy to exercise"
    packet, entry = rewritten[0]
    copy = _read(ROOT / packet["dir"] / entry["copy"])
    source = _read(ROOT / entry["source"])
    assert check_relation(copy, source, entry, "x") == []
    # A change to the body, past the docstring and past the declared substitutions, is drift.
    assert check_relation(copy + "assert False\n", source, entry, "x") != []


def test_an_adapted_copy_must_state_its_reason_and_cost():
    bare = {"relation": "adapted", "copy": "x.py"}
    assert len(check_relation("a", None, bare, "x")) == 2
    assert check_relation("a", None, {**bare, "reason": "r", "consequence": "c"}, "x") == []
