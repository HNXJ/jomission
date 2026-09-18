"""Tier 0: the whole-system realizer's invariants, and consistency of its recorded result.

No construction happens here -- construction is the driver's job. These tests check the
normalization TFNE owns and, when the result file exists, that its verdict follows from its own
recorded numbers rather than from an assertion made alongside them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jomission.tfne import architecture, ctx, nf, realize

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "whole_system_realization.json"
RESULT_R2 = ROOT / "results" / "whole_system_realization_r2.json"
AUDIT = ROOT / "results" / "ctx_allocation_audit.json"


def results():
    return [p for p in (RESULT, RESULT_R2) if p.exists()]


@pytest.fixture(scope="module")
def normal():
    chain, rules, external, _ = architecture.build()
    return chain, rules, nf.normalize(chain, rules, external)


def test_objects_are_the_six_declared_instances(normal):
    _, _, n = normal
    assert [o["name"] for o in n["objects"]] == ["FEF", "PFC", "V1.1", "V1.2", "V4.1", "V4.2"]


def test_only_adjacent_composites_are_related(normal):
    chain, _, n = normal
    member_of = {i.name(): c.label for c in chain.items for i in c.instances()}
    order = [c.label for c in chain.items]
    for p in n["_projection_objects"]:
        if p.source.path == ("Retina",):
            continue
        a, b = member_of[".".join(p.source.path)], member_of[".".join(p.target.path)]
        assert abs(order.index(a) - order.index(b)) <= 1, f"skip projection {p.identity()}"
    assert ("{V1^2 X[lat]}", "{FEF X[lat] PFC}") in nf.non_adjacent_pairs(chain)


def test_v1_to_v4_is_matched_instance_not_all_pairs(normal):
    _, _, n = normal
    ff = {(p.source.path, p.target.path) for p in n["_projection_objects"]
          if p.relation.startswith("O[fffb_matched]") and p.relation.endswith((".ff.L2", ".ff.L3"))}
    assert ff == {(("V1", "1"), ("V4", "1")), (("V1", "2"), ("V4", "2"))}


def test_v4_reaches_both_frontal_objects(normal):
    _, _, n = normal
    ff = {(p.source.path, p.target.path) for p in n["_projection_objects"]
          if p.relation.startswith("O[fffb]") and p.relation.endswith((".ff.L2", ".ff.L3"))}
    assert ff == {(("V4", i), (t,)) for i in ("1", "2") for t in ("FEF", "PFC")}


def test_absent_populations_are_unaddressable(normal):
    chain, _, _ = normal
    inst = chain.items[0].instances()[0]
    assert "L6.PV" in ctx.declaration()["absent_populations"]
    with pytest.raises(ValueError, match="E_ADDRESS_UNRESOLVED"):
        inst.address("L6.PV")


def test_every_declared_mechanism_has_a_tau(normal):
    _, _, n = normal
    assert {p.mechanism for p in n["_projection_objects"]} <= set(realize.MECHANISM_TAU_MS)


@pytest.mark.skipif(not RESULT.exists(), reason="realization has not been run")
@pytest.mark.parametrize("path", results(), ids=lambda p: p.stem)
def test_recorded_verdict_follows_from_recorded_numbers(path):
    rec = json.loads(path.read_text(encoding="utf-8"))
    pr = rec["projection_reconciliation"]
    consistent = (not pr["missing"] and not pr["extra"]
                  and not rec["population_reconciliation"]["problems"]
                  and not rec["mechanism_reconciliation"]["problems"]
                  and rec["adjacency_invariant"]["holds"]
                  and rec["output_interface"]["resolves_nonempty"]
                  and rec["output_interface"]["metadata_exact"]
                  and all(rec["deltas"][k] == 0 for k in ("objects", "populations", "projections", "index")))
    assert rec["verdict"] == "WHOLE_SYSTEM_REALIZATION_" + ("PASS" if consistent else "FAIL")


@pytest.mark.skipif(not RESULT.exists(), reason="realization has not been run")
@pytest.mark.parametrize("path", results(), ids=lambda p: p.stem)
def test_recorded_expected_set_matches_the_current_normalization(normal, path):
    _, _, n = normal
    rec = json.loads(path.read_text(encoding="utf-8"))
    assert rec["projection_reconciliation"]["expected"] == len(realize.expected_projections(n))
    assert len(rec["normalized"]["projections"]) == len(n["projections"])


@pytest.mark.skipif(not RESULT.exists(), reason="realization has not been run")
@pytest.mark.parametrize("path", results(), ids=lambda p: p.stem)
def test_no_simulation_artifacts_in_the_realization_record(path):
    text = path.read_text(encoding="utf-8").lower()
    for token in ("rate_hz", "spikes_per", "window_rates", "trajectory", "run_continuation"):
        assert token not in text, f"{token} implies the realizer simulated"


def test_allocation_policy_preserves_exact_cardinality():
    """R is admissible only if every layer allocates exactly its budget."""
    P = ctx.layer_cell_type_fractions()
    for lay in ctx.LAYERS:
        for total in range(0, 301):
            assert sum(ctx.allocate(total, P[lay]).values()) == total, f"{lay} at N_l={total}"
    layer = ctx.allocate(ctx.N_PER_INSTANCE, ctx.layer_population_fractions())
    assert sum(layer.values()) == ctx.N_PER_INSTANCE
    assert sum(sum(ctx.allocate(n, P[lay]).values()) for lay, n in layer.items()) == ctx.N_PER_INSTANCE


def test_allocation_policy_is_the_engines_own_function():
    """CTX[jomission_v0].R is OBSERVED_CURRENT, so it must not drift from what is realized."""
    from jaxfne._config import _counts_from_fractions as engine
    P = ctx.layer_cell_type_fractions()
    for lay in ctx.LAYERS:
        for total in (0, 1, 7, 20, 30, 40, 60, 200):
            assert ctx.allocate(total, P[lay]) == engine(total, P[lay])


@pytest.mark.skipif(not AUDIT.exists(), reason="audit has not been run")
def test_audit_verdict_follows_from_its_own_numbers():
    rec = json.loads(AUDIT.read_text(encoding="utf-8"))
    seq = rec["policies"]["engine_sequential"]
    assert seq["preserves_exact_cardinality"] == (seq["violations"] == 0)
    ok = seq["preserves_exact_cardinality"] and rec["at_architecture_size"]["sum_layers"] == rec["at_architecture_size"]["N"]
    assert rec["verdict"] == "ALLOCATION_POLICY_" + ("PASS" if ok else "FAIL")
