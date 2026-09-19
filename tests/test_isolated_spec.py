"""Tier 0: the SCI-WS-OSC-2 pre-execution spec says what it claims to say.

Two things are worth enforcing mechanically. The runtime blocks claim to be the parent's
verbatim, which is what makes "everything SCI-WS-OSC-1 froze" checkable. And the measurement
hazard, that a flat population rate cannot decide this lineage, has to survive in the spec,
because it is the reason the interval observable is primary.
"""

from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = json.loads((ROOT / "results/whole_system_isolated_spec.json").read_text("utf-8"))
PARENT = json.loads((ROOT / "results/whole_system_oscillator_spec.json").read_text("utf-8"))


def test_runtime_blocks_are_the_parents_verbatim():
    for block in ("duration", "kernel", "gates"):
        assert SPEC[block] == PARENT[block], f"{block} diverges from the parent spec"


def test_the_arms_differ_only_in_the_reduction():
    a, b = SPEC["arms"]["ISOLATED_POP"], SPEC["arms"]["ISOLATED_1"]
    assert a["single"] is False and b["single"] is True
    assert a["g"] == b["g"] == 0.0
    assert set(a) == set(b)


def test_the_commands_match_the_arms_they_name():
    cmd = {r["arm"]: r["command"] for r in SPEC["runs"]}
    assert "--single" in cmd["ISOLATED_1"]
    assert "--single" not in cmd["ISOLATED_POP"]
    for c in cmd.values():
        assert "--spec results/whole_system_isolated_spec.json" in c
        assert "--seed 0" in c and "--stimulus off" in c


def test_the_population_rate_hazard_is_declared():
    """Without this, a flat population rate would look like absence of the mode."""
    h = SPEC["observable"]["hazard_declared_before_execution"]
    assert "UNINFORMATIVE" in h
    assert SPEC["observable"]["primary"].startswith("per-cell interspike interval")


def test_the_estimator_is_the_parents_and_is_unmodified():
    assert SPEC["estimator"]["module"] == PARENT["estimator"]["module"]
    assert SPEC["estimator"]["function"] == PARENT["estimator"]["function"]
    assert SPEC["oscillation"]["max_lag_ms"] == PARENT["oscillation"]["max_lag_ms"]
    assert SPEC["oscillation"]["n_surrogate"] == PARENT["oscillation"]["n_surrogate"]


def test_the_outcomes_are_three_and_none_is_preferred():
    declared = set(SPEC["predeclared_outcomes"]) - {"comparison_target", "committed"}
    assert declared == {"INTRINSIC_GENERATOR_CONFIRMED", "INTRINSIC_ABSENT",
                        "INTRINSIC_PRESENT_DIFFERENT_PERIOD"}
    assert "no outcome is preferred" in SPEC["predeclared_outcomes"]["committed"]


def test_the_visualization_contract_stages_are_the_manifest_paths():
    from jomission.harness import validate as V

    contract = V.visualization_contract()
    prefix = contract["path_convention"].replace("<lineage_id>", SPEC["lineage_id"])
    for stage, path in SPEC["visualization_contract"].items():
        if stage == "rule":
            continue
        assert path == prefix.replace("<artifact>", contract["stages"][stage]["artifact"])
