"""Tier 0: the VISUALIZATION_CONTRACT rule, on synthetic lineage records.

No construction and no simulation. The registry's own records are checked by
test_harness_state; these cases check that the rule bites, since a validator that only
ever returns [] is indistinguishable from one that is never called.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from jomission.harness import validate as v

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONTRACT = json.loads((ROOT / "manifests" / "visualization_contract.json").read_text("utf-8"))
STAGES = ("V0", "V1", "V2", "V3")
# A path that resolves at HEAD, standing in for a real figure in these synthetic records.
IN_GIT = "manifests/vocabulary.json"


def rec(**over):
    r = {"lineage_id": "SYNTHETIC-1", "executed": True, "commit": "HEAD"}
    r.update(over)
    return r


def test_contract_declares_four_ordered_stages():
    assert CONTRACT["order"] == list(STAGES)
    assert [CONTRACT["stages"][s]["artifact"] for s in STAGES] == [
        "jaxfne_network_hspice.png", "raster_initial_1000ms.png", "raster_final.png",
        "atlas/index.html"]
    # V0 and V1 are the two that must precede interpretation; V2 and V3 must not be.
    before = [s for s in STAGES if CONTRACT["stages"][s]["before_interpretation"]]
    assert before == ["V0", "V1"]


def test_v0_forbids_individual_neurons_and_requires_x_and_y():
    v0 = CONTRACT["stages"]["V0"]
    assert "individual neurons" in v0["must_not_contain"]
    assert any("far left" in c and "far right" in c for c in v0["required_content"])


def test_v2_reuses_v1_conventions():
    """A before/after comparison is only readable if the ordering is the same figure twice."""
    assert any("same ordering" in c for c in CONTRACT["stages"]["V2"]["required_content"])
    assert any("reused verbatim by V2" in c for c in CONTRACT["stages"]["V1"]["required_content"])


def test_field_missing_is_an_error():
    err = v.validate_visualization(rec(), CONTRACT)
    assert len(err) == 1 and "missing visualization" in err[0]


def test_each_missing_stage_is_reported_by_name():
    for stage in STAGES:
        viz = {s: IN_GIT for s in STAGES if s != stage}
        err = v.validate_visualization(rec(visualization=viz), CONTRACT)
        assert len(err) == 1, (stage, err)
        assert f"missing stage {stage}" in err[0]
        assert CONTRACT["stages"][stage]["artifact"] in err[0]


def test_a_declared_path_that_does_not_resolve_is_an_error():
    viz = {s: IN_GIT for s in STAGES}
    viz["V2"] = "results/viz/nothing/raster_final.png"
    err = v.validate_visualization(rec(visualization=viz), CONTRACT)
    assert len(err) == 1 and "V2" in err[0] and "absent at" in err[0]


def test_complete_and_resolvable_passes():
    viz = {s: IN_GIT for s in STAGES}
    assert v.validate_visualization(rec(visualization=viz), CONTRACT) == []


def test_grandfathered_lineages_are_never_reopened():
    """The contract is additive: a rule written today does not invalidate sealed evidence."""
    assert "WS-PROP-1" in CONTRACT["enforcement"]["grandfathered"]
    for lid in CONTRACT["enforcement"]["grandfathered"]:
        assert v.validate_visualization(rec(lineage_id=lid), CONTRACT) == [], lid


def test_unexecuted_lineage_is_not_yet_required_to_have_figures():
    assert v.validate_visualization(rec(executed=False), CONTRACT) == []


def test_exemption_requires_a_reason():
    tag = CONTRACT["exemption"]["shape"]["exempt"]
    ok = v.validate_visualization(rec(visualization={"exempt": tag, "reason": "no model built"}),
                                  CONTRACT)
    assert ok == []
    bare = v.validate_visualization(rec(visualization={"exempt": tag}), CONTRACT)
    assert len(bare) == 1 and "without a reason" in bare[0]
    bogus = v.validate_visualization(rec(visualization={"exempt": "BECAUSE_I_SAID_SO"}), CONTRACT)
    assert len(bogus) == 1 and "unknown" in bogus[0]


def test_validate_lineage_actually_calls_the_contract():
    """Guards against the rule existing but never being reached from the registry validator."""
    src = (ROOT / "jomission" / "harness" / "validate.py").read_text("utf-8")
    body = src.split("def validate_lineage(")[1]
    assert "validate_visualization(rec)" in body


def test_contract_is_registered_as_an_active_invariant():
    state = json.loads((ROOT / "manifests" / "current_state.json").read_text("utf-8"))
    assert CONTRACT["invariant"] in state["active_invariants"]


@pytest.mark.parametrize("stage", STAGES)
def test_every_stage_declares_purpose_and_timing(stage):
    s = CONTRACT["stages"][stage]
    for k in ("name", "artifact", "purpose", "timing", "required_content"):
        assert s.get(k), (stage, k)
