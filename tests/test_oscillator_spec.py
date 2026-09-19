"""Tier 0: the SCI-WS-OSC-1 pre-execution spec says what it claims to say.

The spec claims its duration, kernel and gate blocks are the parent's verbatim, which is what
makes "everything WS-AUTH-2 froze" checkable rather than asserted. It also declares two arms
whose only difference is the E->E cut. Both are enforced here.
"""

from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = json.loads((ROOT / "results/whole_system_oscillator_spec.json").read_text("utf-8"))
PARENT = json.loads(
    (ROOT / "results/whole_system_authority_bracket_spec.json").read_text("utf-8"))


def test_runtime_blocks_are_the_parents_verbatim():
    for block in ("duration", "kernel", "gates"):
        assert SPEC[block] == PARENT[block], f"{block} diverges from the parent spec"
    assert SPEC["frozen_full"] == PARENT["frozen"]


def test_the_arms_differ_only_in_the_cut():
    a, b = SPEC["arms"]["SINGLE_AREA"], SPEC["arms"]["EE_CUT"]
    assert a["ee_cut"] is False and b["ee_cut"] is True
    assert a["g"] == b["g"] == SPEC["shared_regime"]["g"] == 0.0
    assert set(a) == set(b)


def test_the_commands_match_the_arms_they_name():
    cmd = {r["arm"]: r["command"] for r in SPEC["runs"]}
    assert "--ee-cut" in cmd["EE_CUT"]
    assert "--ee-cut" not in cmd["SINGLE_AREA"]
    for arm, c in cmd.items():
        assert "--spec results/whole_system_oscillator_spec.json" in c
        assert "--seed 0" in c and "--stimulus on" in c


def test_every_predeclared_outcome_is_one_the_estimator_can_return():
    from jomission.harness import oscillation

    declared = set(SPEC["predeclared_outcomes"]) - {"rule", "committed"}
    m = {"periodic_mode_present": True, "period_ms": 190.0, "peak_autocorr": 0.9}
    off = {**m, "periodic_mode_present": False}
    reachable = {
        oscillation.compare(off, off)["outcome"],
        oscillation.compare(m, off)["outcome"],
        oscillation.compare(m, m)["outcome"],
        oscillation.compare(m, {**m, "period_ms": 400.0})["outcome"],
    }
    assert declared == reachable


def test_the_visualization_contract_stages_are_the_manifest_paths():
    from jomission.harness import validate as V

    contract = V.visualization_contract()
    prefix = contract["path_convention"].replace("<lineage_id>", SPEC["lineage_id"])
    for stage, path in SPEC["visualization_contract"].items():
        if stage == "rule":
            continue
        assert path == prefix.replace("<artifact>", contract["stages"][stage]["artifact"])
