"""f-I engine equivalence tests (criteria: docs/receipts/fi_engine_equivalence_criteria.md).

Asserts the oracle self-check (G0), that the engine lanes are uncoupled single neurons, that the
negative control detects, and that the recorded receipt reproduces from a live run. It does not
assert equivalence: an outcome NOT_EQUIVALENT is a result, not a test failure.
Writes nothing; the receipt is written only by scripts/fi_engine_equivalence.py.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fi_engine_equivalence as fe  # noqa: E402


def oracle_spike_steps(a, b, c, d, I, v0, u0, n_steps, dt_ms):
    """O2: scalar float64 Izhikevich (2003) loop, written here from the equations.

    Explicit Euler, dt_ms; spike when v >= 30 after the update; reset v = c, u = u + d.
    Returns the list of spike step indices. Not imported from O1.
    """
    v, u = float(v0), float(u0)
    steps = []
    for t in range(n_steps):
        dv = 0.04 * v * v + 5.0 * v + 140.0 - u + I
        du = a * (b * v - u)
        v = v + dt_ms * dv
        u = u + dt_ms * du
        if v >= 30.0:
            steps.append(t)
            v = c
            u = u + d
    return steps


def test_G0_oracle_self_check():
    checked, bad = 0, []
    for spec in fe.SETS:
        p = fe.params_of(spec)
        o1, _ = fe.o1_counts(p, fe.EXTRA)
        assert o1.sum() > 0, f"fixture must spike: {spec['id']}"
        for idx in fe.G0_IDX:
            n2 = len(oracle_spike_steps(p["a"], p["b"], p["c"], p["d"], p["tonic"] + float(fe.EXTRA[idx]),
                                        p["c"], p["b"] * p["c"], fe.N_STEPS, fe.DT_MS))
            checked += 1
            if n2 != o1[idx]:
                bad.append((spec["id"], float(fe.EXTRA[idx]), int(o1[idx]), n2))
    assert checked == len(fe.SETS) * len(fe.G0_IDX) == 66
    assert not bad, f"G0 mismatches: {bad}"


def test_engine_lanes_are_uncoupled_single_neurons():
    p = fe.params_of(fe.SETS[0])
    lanes_c, lanes_s = fe.engine_run(p, fe.EXTRA)
    idx = [0, 40, 80]
    assert lanes_c[idx].sum() > 0, "fixture must spike"
    for i in idx:
        c1, s1 = fe.engine_run(p, fe.EXTRA[i:i + 1])
        assert c1[0] == lanes_c[i], (i, c1[0], lanes_c[i])
        assert np.array_equal(s1[0], lanes_s[i]), i


def test_negative_control_detects():
    p = fe.params_of(fe.SETS[0])
    e_c, _ = fe.engine_run(p, fe.EXTRA)
    assert e_c.sum() > 0, "fixture must spike"
    neg, _ = fe.o1_counts(dict(p, d=p["d"] + 0.5), fe.EXTRA)
    stats = fe.diff_stats(neg, e_c, fe.EXTRA)
    assert stats["mismatch_points"] > 0, "negative control detects nothing: comparison is vacuous"


def test_receipt_reproduces_and_verdict_is_consistent():
    rec = json.loads(fe.RECEIPT.read_text(encoding="utf-8"))
    live = fe.build_receipt(oracle_spike_steps)
    live.pop("provenance")
    rec_np = {k: v for k, v in rec.items() if k != "provenance"}
    assert json.loads(json.dumps(live)) == rec_np
    t = rec["totals"]
    assert rec["verdict"] == t["verdict"] == fe.decide(
        t["G0_mismatch_points"], t["negative_control"]["mismatch_points"], t["G1_mismatch_points"])
    assert t["G1_points"] == 6 * 81 == 486
    assert t["negative_control"]["mismatch_points"] > 0


def test_receipt_provenance_names_loaded_modules():
    rec = json.loads(fe.RECEIPT.read_text(encoding="utf-8"))
    pv = rec["provenance"]
    assert pv["jaxfne_version_metadata"] == "0.5.8"
    assert "site-packages" in pv["jaxfne_file"]
    assert Path(pv["fi_single_neuron_file"]).resolve() == (ROOT / "scripts" / "fi_single_neuron.py").resolve()
    assert rec["engine"]["entry_point"] == "jaxfne.emitters.simulate_eig_izhikevich"
