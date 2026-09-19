"""SCI-V21-PV-M1: matched-state transplant. Which single typed state block switches the PV fate?

Spec (committed and pushed before the procedure runs): results/v21_pv_m1_spec.json.

Five arms from one base and one future stochastic stream (PRNGKey(11), step 0):

    A_CTRL  h_A unmodified                      must stay low-PV
    B_CTRL  h_B unmodified                      must stay active-PV (this is run C of the parent)
    A_v     h_A with v   <- v_B
    A_u     h_A with u   <- u_B
    A_syn   h_A with syn_state[edges into PV] <- the same entries of h_B

M0 closed the candidate set to v, u and syn_state: w, H, prev_spikes and b are bitwise identical
between the two states, so no transplant of them could change anything. Magnitude is not causal
authority, so u is tested despite differing least in PV. Every constructed state is verified
bitwise -- the named block moved, nothing else did -- before any fate is interpreted.

Tier 3: steps the engine and writes results/v21_pv_m1.json.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "results" / "v21_pv_m1_spec.json"
OUT = ROOT / "results" / "v21_pv_m1.json"
PARENT = ROOT / "results" / "v21_pv_predecessor.json"

LAMBDA_C = 0.65625
N_CHUNK = 12
SEED = 11
LATE_WINDOWS = 3
TOL_ABS_HZ, TOL_REL = 1.0, 0.10
REPRO_TOL_HZ = 0.5
A_HASH = "83ce0d0c3fc44a71"
B_HASH = "8704423170e6ae32"
SEALED_LOW_PV_HZ = 0.7066666666666666
SEALED_ACTIVE_PV_HZ = 14.425000000000002
CLASSES = ("E", "PV", "SST", "VIP")
ARMS = ("A_CTRL", "B_CTRL", "A_v", "A_u", "A_syn")


def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def band(reference):
    return max(TOL_ABS_HZ, TOL_REL * abs(reference))


def matches(rate, reference):
    return bool(abs(rate - reference) <= band(reference))


def leaves_equal(x, y):
    """Bitwise comparison of two DynamicState instances, leaf by leaf."""
    m0 = _load("tests/test_v21_pv_m0.py", "_v21_m0_cmp")
    a, b = m0.leaves(x), m0.leaves(y)
    out = {}
    for k in sorted(set(a) | set(b)):
        u, v = a.get(k), b.get(k)
        if u is None and v is None:
            out[k] = True
        elif u is None or v is None:
            out[k] = False
        else:
            out[k] = bool(np.array_equal(np.asarray(u), np.asarray(v)))
    return out


def test_v21_pv_m1():
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    parent = json.loads(PARENT.read_text(encoding="utf-8"))
    m0 = _load("tests/test_v21_pv_m0.py", "_v21_m0")
    pred = _load("tests/test_v21_pv_predecessor.py", "_v21_pred")
    sim = pred._Sim()
    jnp = sim.jnp
    cls = np.asarray(sim.cls)

    a0, b0 = sim.fresh_state(), sim.lambda1_end_state()
    rec = {"spec": "results/v21_pv_m1_spec.json", "lambda": LAMBDA_C,
           "regeneration": {"A_hash": m0.v_hash(a0.dynamic), "B_hash": m0.v_hash(b0.dynamic),
                            "A_expected": A_HASH, "B_expected": B_HASH}}
    rec["regeneration"]["hashes_match"] = bool(rec["regeneration"]["A_hash"] == A_HASH
                                               and rec["regeneration"]["B_hash"] == B_HASH)
    assert rec["regeneration"]["hashes_match"], "regenerated states do not match the sealed v-hashes"

    # every arm carries the same future stochastic stream
    fresh_key = sim.jax.random.PRNGKey(SEED)
    base_a = a0._replace(prng_key=fresh_key, step_index=0)
    base_b = b0._replace(prng_key=fresh_key, step_index=0)

    pv_edges = np.flatnonzero(cls[sim.post] == "PV")
    syn_a = np.asarray(a0.dynamic.syn_state)
    syn_b = np.asarray(b0.dynamic.syn_state)
    syn_mixed = syn_a.copy()
    syn_mixed[pv_edges] = syn_b[pv_edges]
    rec["syn_arm_scope"] = {"n_edges_total": int(syn_a.size), "n_edges_into_PV": int(pv_edges.size),
                            "fraction": float(pv_edges.size / syn_a.size),
                            "presynaptic_classes": sorted({str(c) for c in cls[sim.pre[pv_edges]]})}

    states = {
     "A_CTRL": base_a,
     "B_CTRL": base_b,
     "A_v": base_a._replace(dynamic=a0.dynamic._replace(v=b0.dynamic.v)),
     "A_u": base_a._replace(dynamic=a0.dynamic._replace(u=b0.dynamic.u)),
     "A_syn": base_a._replace(dynamic=a0.dynamic._replace(syn_state=jnp.asarray(syn_mixed))),
    }

    # transplant verification: the named block moved, nothing else did
    moved = {"A_v": "v", "A_u": "u", "A_syn": "syn_state"}
    ver = {}
    for name, block in moved.items():
        eq_a = leaves_equal(states[name].dynamic, a0.dynamic)
        eq_b = leaves_equal(states[name].dynamic, b0.dynamic)
        others_held = all(v for k, v in eq_a.items() if k != block)
        if block == "syn_state":
            s = np.asarray(states[name].dynamic.syn_state)
            block_ok = bool(np.array_equal(s[pv_edges], syn_b[pv_edges]))
            complement = np.setdiff1d(np.arange(s.size), pv_edges, assume_unique=False)
            block_ok = block_ok and bool(np.array_equal(s[complement], syn_a[complement]))
        else:
            block_ok = bool(eq_b[block]) and not bool(eq_a[block])
        ver[name] = {"block": block, "block_took_donor_value": block_ok,
                     "every_other_leaf_equals_A": bool(others_held),
                     "leaves_differing_from_A": sorted(k for k, v in eq_a.items() if not v)}
    rec["transplant_verification"] = ver
    if not all(v["block_took_donor_value"] and v["every_other_leaf_equals_A"] for v in ver.values()):
        rec["verdict"] = "M1_TRANSPLANT_" + "INVALID"
        rec["reason"] = "a constructed state moved something other than the named block"
        OUT.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8", newline="\n")
        raise AssertionError(rec["reason"])

    rec["arms"] = {}
    for name in ARMS:
        r = sim.run(states[name])
        rates = [{c: w["rates"][c] for c in CLASSES} for w in r["summaries"]]
        late = float(np.mean([w["PV"] for w in rates[-LATE_WINDOWS:]]))
        rec["arms"][name] = {"window_rates_hz": rates, "late_pv_hz": late,
                             "late_other_classes_hz": {c: float(np.mean([w[c] for w in rates[-LATE_WINDOWS:]]))
                                                       for c in CLASSES if c != "PV"}}
        del r

    # controls, before any transplant fate is interpreted
    a_windows = parent["runs"]["A_collapse"]["window_rates_hz"]
    got = rec["arms"]["A_CTRL"]["window_rates_hz"]
    a_dev = max(abs(got[k][c] - a_windows[k][c]) for k in range(len(a_windows)) for c in CLASSES)
    c_late = parent["P1_fate"]["late_pv_hz"]["C_control"]
    rec["controls"] = {
     "A_CTRL_reproduces_sealed_run_A": {"max_abs_diff_hz": float(a_dev), "within_tol": bool(a_dev <= REPRO_TOL_HZ)},
     "A_CTRL_is_low_pv": matches(rec["arms"]["A_CTRL"]["late_pv_hz"], SEALED_LOW_PV_HZ),
     "B_CTRL_is_active_pv": matches(rec["arms"]["B_CTRL"]["late_pv_hz"], SEALED_ACTIVE_PV_HZ),
     "B_CTRL_sealed_reference_run_C_hz": c_late}
    controls_ok = (rec["controls"]["A_CTRL_reproduces_sealed_run_A"]["within_tol"]
                   and rec["controls"]["A_CTRL_is_low_pv"] and rec["controls"]["B_CTRL_is_active_pv"])
    if not controls_ok:
        rec["verdict"] = "M1_CONTROL_" + "FAILED"
        rec["reason"] = "a control did not reproduce its sealed fate; no transplant fate is interpreted"
        OUT.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8", newline="\n")
        raise AssertionError(rec["reason"])

    flips = []
    for name in ("A_v", "A_u", "A_syn"):
        late = rec["arms"][name]["late_pv_hz"]
        active, low = matches(late, SEALED_ACTIVE_PV_HZ), matches(late, SEALED_LOW_PV_HZ)
        rec["arms"][name]["fate"] = "active" if active else ("low" if low else "neither")
        rec["arms"][name]["classification"] = "CAUSAL_SUFFICIENT" if active else "NOT_SUFFICIENT"
        if active:
            flips.append(name)
    rec["flips"] = flips

    if len(flips) == 1:
        label = "SINGLE_BLOCK_SUFFICIENT_" + "PASS"
        reason = f"{moved[flips[0]]} is sufficient: {flips[0]} reaches {rec['arms'][flips[0]]['late_pv_hz']:.2f} Hz"
    elif len(flips) > 1:
        label = "MULTIPLE_BLOCKS_SUFFICIENT_" + "PASS"
        reason = ("more than one block is independently sufficient: "
                  + ", ".join(f"{moved[f]} ({rec['arms'][f]['late_pv_hz']:.2f} Hz)" for f in flips)
                  + ". They are not ranked")
    else:
        label = "NO_SINGLE_BLOCK_SUFFICIENT_" + "FAIL"
        reason = ("no single transplanted block switches the PV fate; no conjunction search follows automatically")
    rec["verdict"], rec["reason"] = label, reason
    rec["forbidden_respected"] = spec["forbidden"]
    OUT.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(label, reason)
    for name in ARMS:
        print(f"  {name}: late PV {rec['arms'][name]['late_pv_hz']:.3f} Hz")


def test_spec_constants_match():
    c = json.loads(SPEC.read_text(encoding="utf-8"))["procedure_constants"]
    assert (LAMBDA_C, N_CHUNK, SEED, LATE_WINDOWS, TOL_ABS_HZ, TOL_REL, A_HASH, B_HASH,
            SEALED_LOW_PV_HZ, SEALED_ACTIVE_PV_HZ) == (
        c["lambda"], c["n_chunk"], c["engine_seed"], c["late_windows"], c["tolerance_abs_hz"],
        c["tolerance_rel"], c["a_hash"], c["b_hash"], c["sealed_low_pv_hz"], c["sealed_active_pv_hz"])
    pred = _load("tests/test_v21_pv_predecessor.py", "_v21_pred_c")
    assert (pred.N_CHUNK, pred.LAMBDA_C, pred.SEED) == (N_CHUNK, LAMBDA_C, SEED)
