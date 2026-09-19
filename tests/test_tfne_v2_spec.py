"""Tier 0: the permanent TFNE v2 spec matches its conformance derivation and covers the corpus.

Mechanical check of a document, not of a language: no parser is involved.
"""

import json
import re
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "docs" / "tfne" / "TFNE_V2_SPEC.md"


def _spec():
    return SPEC.read_text(encoding="utf-8")


def _flat():
    """Spec text with runs of whitespace collapsed, so line wrapping cannot hide a phrase."""
    return re.sub(r"\s+", " ", _spec())


def _mod():
    return runpy.run_path(str(ROOT / "scripts" / "tfne" / "v2_conformance.py"), run_name="tfne_v2")


def test_appendix_numbers_equal_derivation():
    block = re.search(r"```json\n(\{.*?\n\})\n```", _spec(), re.S)
    assert block, "appendix A has no JSON conformance block"
    assert json.loads(block.group(1)) == _mod()["derive"]()


def test_spec_covers_every_decision_and_case():
    text = _flat()
    required = [
        # owner decisions
        "O[k](A^n)", "SEG.2", "$L", "$R", "E_FRONTIER_UNRESOLVED", "hash(serialization(NF(E)))",
        "occurrence indices", "adjacency-labelled", "A.in", "A.out",
        # invariants
        "biology introduces definitions, not grammar",
        r"structural identity} \neq \text{instance identity} \neq \text{realization identity",
        r"(h_t,x_t;s)\mapsto(h_{t+1},y_t)",
        # corpus and probes
        *[f"| {c} |" for c in ("C1", "C2", "C3", "C4", "C5", "C6", "N1", "N2", "N3", "N4", "N5", "N6")],
        # error vocabulary
        *[f"`{e}`" for e in ("E_PROPORTION_INVALID", "E_ADDRESS_UNRESOLVED", "E_X_GROUPING",
                             "E_MECHANISM_UNRESOLVED", "E_EXCLUSION_UNKNOWN", "E_ADDITION_REDUNDANT",
                             "E_AMBIGUOUS_EXPANSION", "E_ALLOCATION_POLICY_MISSING")],
    ]
    assert [r for r in required if r not in text] == []


def test_reserved_names_complete_and_q_retired():
    text = _flat()
    assert " `Q` " not in text and "A Q B" not in text
    for name in ("`O`", "`X`", "`H`", "`h`, `s`", "`N`, `P`, `G`, `C`", "`{}`", "`[]`", "`.`",
                 "`>`, `<`, `<>`", "`≠>`, `≠<`"):
        assert name in text, name


def test_natural_key_order_and_allocation():
    m = _mod()
    assert m["derive"]()["natural_order_probe"] == ["L1", "L2", "L10", "SEG.2", "SEG.10"]
    assert m["allocate"](101, {"E": 0.8, "PV": 0.2}) == {"E": 81, "PV": 20}
    assert sum(m["allocate"](10000, {"L1": 0.02, "L2": 0.15, "L3": 0.20,
                                     "L4": 0.22, "L5": 0.23, "L6": 0.18}).values()) == 10000
    assert m["chain"]("SEG", 8, "spinal", "inter", "inter")[0]["src"] == "SEG.1.inter"
    assert len(m["chain"]("SEG", 8, "spinal", "inter", "inter")) == 7
