"""E0 verdict rule: known positive, each margin at its boundary, undefined and incomplete are never a pass."""
import copy

import pytest

from jomission.laminar import e0_verdict as v


def _rec(f=33.8, lag=7.04, e=1.106, pv=11.37, sst=7.05, steps=v.STEPS_EXPECTED, osc=True, undefined=()):
    return {"steps": steps, "analysis": {
        "rates_Hz": {"E": e, "PV": pv, "SOM": sst}, "peak_freq_E_Hz": f, "lag_PV_SOM_ms": lag,
        "oscillates": osc, "undefined": list(undefined)}}


def _all(**kw):
    return {s: _rec(**kw) for s in v.SEEDS}


def test_reference_values_pass():
    out = v.verdict(_all())
    assert out["verdict"] == "PASS"
    assert set(out["estimands"]) == set(v.REFERENCE) and len(v.REFERENCE) == 5
    assert all(e["status"] == "IN" for e in out["estimands"].values())


@pytest.mark.parametrize("name,key", [("f_pk_Hz", "f"), ("lag_PV_SOM_ms", "lag"), ("rate_E_Hz", "e"),
                                      ("rate_PV_Hz", "pv"), ("rate_SST_Hz", "sst")])
def test_each_margin_inside_and_outside(name, key):
    ref, margin, kind = v.REFERENCE[name]
    limit = margin if kind == "abs" else margin * ref
    inside = _all(**{key: ref + 0.99 * limit})
    outside = _all(**{key: ref + 1.01 * limit})
    below = _all(**{key: ref - 1.01 * limit})
    assert v.verdict(inside)["verdict"] == "PASS"
    for bad in (outside, below):
        out = v.verdict(bad)
        assert out["verdict"] == "FAIL"
        assert out["estimands"][name]["status"] == "OUT"
        assert [n for n, e in out["estimands"].items() if e["status"] == "OUT"] == [name], "only the perturbed estimand fails"


def test_mean_over_seeds_not_each_seed():
    recs = _all()
    ref, margin, _ = v.REFERENCE["f_pk_Hz"]
    recs[1] = _rec(f=ref + 1.5 * margin)  # one seed outside, mean inside
    recs[2] = _rec(f=ref - 1.5 * margin)
    assert v.verdict(recs)["verdict"] == "PASS"


def test_no_oscillation_is_undefined_and_fails():
    recs = _all()
    recs[3] = _rec(osc=False)
    out = v.verdict(recs)
    assert out["verdict"] == "FAIL" and out["estimands"]["f_pk_Hz"]["status"] == "UNDEFINED"


def test_undefined_population_fails():
    recs = _all()
    recs[4] = _rec(undefined=["SOM"])
    assert v.verdict(recs)["verdict"] == "FAIL"


def test_missing_short_and_extra_seed_are_incomplete():
    recs = _all()
    del recs[5]
    assert v.verdict(recs) == {**v.verdict(recs), "verdict": "INCOMPLETE", "missing_seeds": [5]}
    recs = _all()
    recs[2] = _rec(steps=v.STEPS_EXPECTED - 1)
    assert v.verdict(recs)["short_seeds"] == [2] and v.verdict(recs)["verdict"] == "INCOMPLETE"
    recs = _all()
    recs[0] = _rec()
    assert v.verdict(recs)["verdict"] == "INCOMPLETE"


def test_input_not_mutated():
    recs = _all()
    before = copy.deepcopy(recs)
    v.verdict(recs)
    assert recs == before


def test_stored_result_equals_recomputed_verdict():
    import json
    from pathlib import Path

    d = Path(__file__).resolve().parents[1] / "results/laminar_e0"
    res = json.loads((d / "e0_result.json").read_text(encoding="utf-8"))
    seeds = {n: json.loads((d / f"e0_seed{n}.json").read_text(encoding="utf-8")) for n in v.SEEDS}
    assert all(s["jaxfne_version"] == "0.5.9" and s["p"] == 0.07 and s["T_s"] == 12.0 for s in seeds.values())
    again = v.verdict(seeds)
    assert res["verdict"] == again["verdict"] and res["estimands"] == again["estimands"]


def test_references_equal_the_seal_table():
    text = open("docs/laminar_motif/E0_SEAL.md", encoding="utf-8").read()
    for token in ("33.8 Hz", "1 Hz", "7.04 ms", "0.34 ms", "1.106 Hz", "6.1 %", "11.37 Hz", "1.2 %", "7.05 Hz", "8.2 %"):
        assert token in text, token
