"""SCI-V21-PV-M2: pairwise transplant. Does the PV fate depend on compatibility among state blocks?

Spec (committed and pushed before the procedure runs): results/v21_pv_m2_spec.json.

M1 showed that none of v, u or the PV-incoming synaptic state is independently sufficient, and
that each alone drives PV *below* the untouched base. This battery tests the three missing pairs
from the same base, the same donor and the same future stochastic stream:

    A_CTRL  h_A unmodified                          must stay low-PV
    B_CTRL  h_B unmodified                          must stay active-PV
    A_vu    (v, u)                <- (v_B, u_B)
    A_vs    (v, syn_PV-incoming)  <- (v_B, syn_B)
    A_us    (u, syn_PV-incoming)  <- (u_B, syn_B)

The triple is not run: it approaches reconstruction of B and would say less about the minimal
interaction. The controls are rerun rather than reused, which also proves the pair-construction
path builds the same base state.

Tier 3: steps the engine and writes results/v21_pv_m2.json.
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "results" / "v21_pv_m2_spec.json"
OUT = ROOT / "results" / "v21_pv_m2.json"
PARENT = ROOT / "results" / "v21_pv_predecessor.json"

LATE_WINDOWS = 3
REPRO_TOL_HZ = 0.5
CLASSES = ("E", "PV", "SST", "VIP")
ARMS = ("A_CTRL", "B_CTRL", "A_vu", "A_vs", "A_us")
PAIRS = {"A_vu": ("v", "u"), "A_vs": ("v", "syn_state"), "A_us": ("u", "syn_state")}


def _m1():
    import importlib.util
    spec = importlib.util.spec_from_file_location("_v21_m1", ROOT / "tests" / "test_v21_pv_m1.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_v21_pv_m2():
    m1 = _m1()
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    parent = json.loads(PARENT.read_text(encoding="utf-8"))
    m0 = m1._load("tests/test_v21_pv_m0.py", "_v21_m0_b")
    pred = m1._load("tests/test_v21_pv_predecessor.py", "_v21_pred_b")
    sim = pred._Sim()
    jnp = sim.jnp
    cls = np.asarray(sim.cls)

    a0, b0 = sim.fresh_state(), sim.lambda1_end_state()
    rec = {"spec": "results/v21_pv_m2_spec.json", "lambda": m1.LAMBDA_C,
           "corrects_nothing": "additive to SCI-V21-PV-M1; that result stands",
           "regeneration": {"A_hash": m0.v_hash(a0.dynamic), "B_hash": m0.v_hash(b0.dynamic),
                            "A_expected": m1.A_HASH, "B_expected": m1.B_HASH}}
    rec["regeneration"]["hashes_match"] = bool(rec["regeneration"]["A_hash"] == m1.A_HASH
                                               and rec["regeneration"]["B_hash"] == m1.B_HASH)
    assert rec["regeneration"]["hashes_match"], "regenerated states do not match the sealed v-hashes"

    fresh_key = sim.jax.random.PRNGKey(m1.SEED)
    base_a = a0._replace(prng_key=fresh_key, step_index=0)
    base_b = b0._replace(prng_key=fresh_key, step_index=0)

    pv_edges = np.flatnonzero(cls[sim.post] == "PV")
    syn_a, syn_b = np.asarray(a0.dynamic.syn_state), np.asarray(b0.dynamic.syn_state)
    syn_mixed = syn_a.copy()
    syn_mixed[pv_edges] = syn_b[pv_edges]
    syn_donor = jnp.asarray(syn_mixed)
    rec["syn_arm_scope"] = {"n_edges_total": int(syn_a.size), "n_edges_into_PV": int(pv_edges.size),
                            "fraction": float(pv_edges.size / syn_a.size),
                            "presynaptic_classes": sorted({str(c) for c in cls[sim.pre[pv_edges]]}),
                            "limit": spec["scope_limit"]}

    states = {
     "A_CTRL": base_a,
     "B_CTRL": base_b,
     "A_vu": base_a._replace(dynamic=a0.dynamic._replace(v=b0.dynamic.v, u=b0.dynamic.u)),
     "A_vs": base_a._replace(dynamic=a0.dynamic._replace(v=b0.dynamic.v, syn_state=syn_donor)),
     "A_us": base_a._replace(dynamic=a0.dynamic._replace(u=b0.dynamic.u, syn_state=syn_donor)),
    }

    ver = {}
    for name, blocks in PAIRS.items():
        eq_a = m1.leaves_equal(states[name].dynamic, a0.dynamic)
        eq_b = m1.leaves_equal(states[name].dynamic, b0.dynamic)
        moved_ok = True
        for block in blocks:
            if block == "syn_state":
                s = np.asarray(states[name].dynamic.syn_state)
                complement = np.setdiff1d(np.arange(s.size), pv_edges)
                moved_ok = moved_ok and bool(np.array_equal(s[pv_edges], syn_b[pv_edges])) \
                    and bool(np.array_equal(s[complement], syn_a[complement]))
            else:
                moved_ok = moved_ok and bool(eq_b[block]) and not bool(eq_a[block])
        ver[name] = {"blocks": list(blocks), "blocks_took_donor_values": bool(moved_ok),
                     "every_other_leaf_equals_A": bool(all(v for k, v in eq_a.items() if k not in blocks)),
                     "leaves_differing_from_A": sorted(k for k, v in eq_a.items() if not v)}
    rec["transplant_verification"] = ver
    if not all(v["blocks_took_donor_values"] and v["every_other_leaf_equals_A"] for v in ver.values()):
        rec["verdict"] = "M2_TRANSPLANT_" + "INVALID"
        rec["reason"] = "a constructed state moved something other than its two named blocks"
        OUT.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8", newline="\n")
        raise AssertionError(rec["reason"])

    rec["arms"] = {}
    for name in ARMS:
        r = sim.run(states[name])
        rates = [{c: w["rates"][c] for c in CLASSES} for w in r["summaries"]]
        rec["arms"][name] = {
         "window_rates_hz": rates,
         "late_pv_hz": float(np.mean([w["PV"] for w in rates[-LATE_WINDOWS:]])),
         "late_other_classes_hz": {c: float(np.mean([w[c] for w in rates[-LATE_WINDOWS:]]))
                                   for c in CLASSES if c != "PV"}}
        del r

    a_windows = parent["runs"]["A_collapse"]["window_rates_hz"]
    got = rec["arms"]["A_CTRL"]["window_rates_hz"]
    a_dev = max(abs(got[k][c] - a_windows[k][c]) for k in range(len(a_windows)) for c in CLASSES)
    rec["controls"] = {
     "A_CTRL_reproduces_sealed_run_A": {"max_abs_diff_hz": float(a_dev), "within_tol": bool(a_dev <= REPRO_TOL_HZ)},
     "A_CTRL_is_low_pv": m1.matches(rec["arms"]["A_CTRL"]["late_pv_hz"], m1.SEALED_LOW_PV_HZ),
     "B_CTRL_is_active_pv": m1.matches(rec["arms"]["B_CTRL"]["late_pv_hz"], m1.SEALED_ACTIVE_PV_HZ),
     "m1_reference_hz": {"A_CTRL": 0.7066666666666666, "B_CTRL": 15.11}}
    if not (rec["controls"]["A_CTRL_reproduces_sealed_run_A"]["within_tol"]
            and rec["controls"]["A_CTRL_is_low_pv"] and rec["controls"]["B_CTRL_is_active_pv"]):
        rec["verdict"] = "M2_CONTROL_" + "FAILED"
        rec["reason"] = "a control did not reproduce its sealed fate; no transplant fate is interpreted"
        OUT.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8", newline="\n")
        raise AssertionError(rec["reason"])

    flips = []
    for name in PAIRS:
        late = rec["arms"][name]["late_pv_hz"]
        active = m1.matches(late, m1.SEALED_ACTIVE_PV_HZ)
        low = m1.matches(late, m1.SEALED_LOW_PV_HZ)
        rec["arms"][name]["fate"] = "active" if active else ("low" if low else "neither")
        rec["arms"][name]["classification"] = "PAIR_SUFFICIENT" if active else "NOT_SUFFICIENT"
        if active:
            flips.append(name)
    rec["flips"] = flips

    if len(flips) == 1:
        label = "MINIMAL_PAIR_" + "IDENTIFIED"
        reason = (f"{' and '.join(PAIRS[flips[0]])} together are sufficient while neither is alone: "
                  f"{flips[0]} reaches {rec['arms'][flips[0]]['late_pv_hz']:.2f} Hz")
    elif len(flips) > 1:
        label = "MULTIPLE_PAIR_" + "SUFFICIENT"
        reason = ("more than one pair is sufficient: "
                  + ", ".join(f"{f} ({rec['arms'][f]['late_pv_hz']:.2f} Hz)" for f in flips) + ". They are not ranked")
    else:
        label = "NO_PAIR_" + "SUFFICIENT"
        reason = ("no pair switches the PV fate. Scope: this covers v, u and the PV-incoming synaptic entries only, "
                  "not the global syn_state, of which 90 percent was left at the base value")
    rec["verdict"], rec["reason"] = label, reason
    rec["forbidden_respected"] = spec["forbidden"]
    OUT.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(label, reason)
    for name in ARMS:
        print(f"  {name}: late PV {rec['arms'][name]['late_pv_hz']:.3f} Hz")


def test_spec_constants_match():
    m1 = _m1()
    c = json.loads(SPEC.read_text(encoding="utf-8"))["procedure_constants"]
    assert (m1.LAMBDA_C, m1.N_CHUNK, m1.SEED, LATE_WINDOWS, m1.TOL_ABS_HZ, m1.TOL_REL, m1.A_HASH,
            m1.B_HASH, m1.SEALED_LOW_PV_HZ, m1.SEALED_ACTIVE_PV_HZ) == (
        c["lambda"], c["n_chunk"], c["engine_seed"], c["late_windows"], c["tolerance_abs_hz"],
        c["tolerance_rel"], c["a_hash"], c["b_hash"], c["sealed_low_pv_hz"], c["sealed_active_pv_hz"])
    assert set(PAIRS) | {"A_CTRL", "B_CTRL"} == set(json.loads(SPEC.read_text(encoding="utf-8"))["arms"])
