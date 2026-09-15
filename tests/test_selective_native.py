"""S6 selective native qualification: transfer of the inverse design to native.

Per predeclared s_E in {50,54,57,61,65} (results/selective_bracket.json):
(a) E-out shape on the all-E star (landed Phase-2 knobs);
(b) inhibitory-origin invariance vs sealed B2/B3 rows (stringent allclose +
    recorded bitwise flag);
(c) common E-out modulation across EE/E->PV/E->SST (no target-class gains);
(d) no-clip margin (descriptor/kernel);
(e) rule-off identity; (f) continuation/determinism/JIT.

Writes results/selective_shape_s{TAG}.json. NO geometry in S6.
"""

import jax
import jax.numpy as jnp
import numpy as np

import jaxfne as jtfne
from jomission.qualification import hdp_attractor as ha
from jomission.qualification import ei_geometry as G
from jomission.qualification.cmin import initial_state
from jomission.qualification.hdp_rule_sat_ee import PASS_BANDS
from jomission.qualification.hdp_rule_sel_eout import (
    S_BRACKET,
    W_BASE_I,
    ensure_registered_scale,
    scaled_params,
    scaled_rule_name,
    scaled_w_base_E,
    scaled_w_bounds,
    scale_tag,
)

DT_MS = 0.1
SEED = 11
N_POST = 50
DRIVE_S = 6.0
POST_S = 2.0
MID_S = 4.0

SHAPE_AMPS = {"rest": 0.0, "transition": 5.0, "active": 10.0, "above": 16.0}


def _fresh_state(model, rule_name, seed=SEED):
    dyn = jtfne.dynamic_state_from_model(
        model, hdp_params={"hdp_rule": rule_name})
    return jtfne.ContinuationState(
        dynamic=dyn, prng_key=jax.random.PRNGKey(int(seed)),
        step_index=0, delay_state=None)


def _compile(model, rule_name, rule_params):
    return jtfne.compile_step_fn(
        model, dt_ms=DT_MS, kernel="hdp", hdp_rule=rule_name,
        hdp_rule_params=dict(rule_params), record_weight_trace=True)


def _run(model, step_fn, st, sched):
    st1, out = jtfne.run_continuation(step_fn, st, sched)
    jax.block_until_ready(out[0])
    return st1, out


def _zero_tonic(model):
    e = model.params["emitter"]
    return model.with_emitter_parameters(
        drive_per_neuron=jnp.zeros_like(e.drive))


def _drive(n, total_steps, drive_steps, amp, dtype, idx=0):
    sched = jnp.zeros((total_steps, n), dtype=dtype)
    return sched.at[:drive_steps, idx].set(float(amp))


def edge_current_mean(sp_drv, w_tr, tau_ms=2.0, dt_ms=0.1):
    decay = float(np.exp(-dt_ms / tau_ms))
    syn = np.empty(sp_drv.shape[0])
    acc = 0.0
    for t in range(sp_drv.shape[0]):
        acc = acc * decay + sp_drv[t]
        syn[t] = acc
    return float((w_tr * syn[:, None]).mean())


def run_shape_probe(s, amp):
    rule_name = ensure_registered_scale(s)
    rp = scaled_params(s)
    model, _ = ha.calibration_star(n_post=N_POST, seed=0,
                                   w_base=scaled_w_base_E(s), tau_ms=2.0)
    model = _zero_tonic(model)
    n = N_POST + 1
    dtype = model.params["emitter"].v0.dtype
    drv_steps = int(DRIVE_S / (DT_MS / 1000))
    post_steps = int(POST_S / (DT_MS / 1000))
    mid_steps = int(MID_S / (DT_MS / 1000))
    sched = _drive(n, drv_steps + post_steps, drv_steps, amp, dtype)
    step_fn, _ = _compile(model, rule_name, rp)
    st = _fresh_state(model, rule_name, SEED)
    st_mid, _ = _run(model, step_fn, st, sched[:mid_steps])
    st_end, out = _run(model, step_fn, st_mid, sched[mid_steps:drv_steps])
    _, out_post = _run(model, step_fn, st_end, sched[drv_steps:])
    sp = np.asarray(out[1], dtype=float)
    w_tr = np.asarray(out[4])
    w_post_tr = np.asarray(out_post[4])
    return {
        "amp": float(amp), "rate": float(sp[-30000:, 0].mean() * 10000.0),
        "w_mid": float(w_tr[:10000].mean()), "w": float(w_tr[-30000:].mean()),
        "w_std": float(w_tr[-30000:].std()), "w_max": float(w_tr[-30000:].max()),
        "w_post": float(w_post_tr[-10000:].mean()),
        "H_idle": float(np.abs(np.asarray(st_end.dynamic.H) - 1.0).max()),
        "spikes": sp, "w_tr": w_tr[-30000:],
        "model": model, "step_fn": step_fn,
    }


def check_shape(s):
    import json
    assert s in S_BRACKET
    rp = scaled_params(s)
    w_hi_E = rp["w_hi_E"]
    landed = {k: run_shape_probe(s, a) for k, a in SHAPE_AMPS.items()}
    res = {}
    for k, r in landed.items():
        res[k] = {kk: vv for kk, vv in r.items()
                  if kk not in ("spikes", "w_tr", "model", "step_fn")}
        assert abs(r["w"] - r["w_mid"]) <= 0.001 * s, (k, r["w"], r["w_mid"])
        assert r["H_idle"] == 0.0, (k, r["H_idle"])
    for k in ("rest", "transition", "active", "above"):
        landed[k]["I_exact"] = edge_current_mean(
            landed[k]["spikes"][-30000:, 0], landed[k]["w_tr"])
        res[k]["I_exact"] = landed[k]["I_exact"]
    res["below"] = dict(res["rest"])
    res["below"]["substituted"] = True
    res["below"]["substitution_cause"] = "sealed Phase-2 substitution (F-I jump; driver rule-independent)"
    order = ["rest", "below", "transition", "active", "above"]
    Is = [res[n]["I_exact"] for n in order]
    assert res["rest"]["w"] <= PASS_BANDS["w_rest_max"] * s
    assert res["transition"]["w"] / max(res["below"]["w"], 1e-12) >= PASS_BANDS["w_trans_over_below_min"]
    assert PASS_BANDS["w_active_lo"] * s <= res["active"]["w"] <= PASS_BANDS["w_active_hi"] * s
    assert res["above"]["w"] <= PASS_BANDS["w_above_hi"] * s
    assert res["above"]["w"] / max(res["active"]["w"], 1e-12) <= PASS_BANDS["w_above_over_active_max"]
    assert all(b >= a for a, b in zip(Is, Is[1:])), Is
    assert Is[3] >= PASS_BANDS["I_active_min"] * s, Is
    # Causal on/off at matched active drive.
    r = landed["active"]
    step_fn, _ = _compile(r["model"], ensure_registered_scale(s), {**rp, "k_w": 0.0})
    _drv = int(DRIVE_S / (DT_MS / 1000))
    sched = _drive(N_POST + 1, _drv, _drv, SHAPE_AMPS["active"],
                   r["model"].params["emitter"].v0.dtype)
    _, out = _run(r["model"], step_fn, _fresh_state(r["model"], ensure_registered_scale(s), SEED), sched)
    null_sp = np.asarray(out[1], dtype=float)[:, 0]
    I_off = edge_current_mean(null_sp[-30000:], np.full((30000, N_POST), scaled_w_base_E(s)))
    res["I_off_active"] = I_off
    assert Is[3] / max(I_off, 1e-12) >= PASS_BANDS["I_on_over_off_min"]
    assert res["active"]["w_post"] <= PASS_BANDS["w_post_max"] * s
    assert max(res[n]["w_max"] for n in order) <= w_hi_E * 1.05
    # Predicted linear transfer vs s=1 sealed reference.
    ref = json.load(open("results/native_transformation.json"))
    for k in ("transition", "active", "above"):
        assert abs(res[k]["w"] / max(ref[k]["w"], 1e-12) - s) / s <= 0.05, (k,)
        assert abs(res[k]["I_exact"] / max(ref[k]["I_exact"], 1e-12) - s) / s <= 0.10, (k,)
    return res


def check_inhibitory_invariance(s):
    """PV/SST-driver motifs under the selective rule vs sealed B2/B3 rows.

    Inhibitory edges must behave as under v0 (same form, same s=1 params,
    untouched aux/H path). Cross-process correction (measured 2026-09-15):
    active-motif w/I window-means vary up to ~1e-4 abs across processes
    (threaded float32 reduction order over 60k steps; spike counts exact,
    timing phases vary). The no-leakage claim needs 1e-3, not 1e-9:
    leakage would be ~50x, not 1e-3. Bitwise flag recorded as observed.
    """
    import json
    rule_name = ensure_registered_scale(s)
    rp = scaled_params(s)
    out = {}
    for driver, amp, key in (("PV", 5.0, "b2"), ("SST", 3.0, "b3")):
        model, drv, tgt = G.build_star(driver_class=driver,
                                       n_targets={"E": 20}, seed=0)
        model = _zero_tonic(model)
        n = int(model.params["emitter"].v0.shape[0])
        dtype = model.params["emitter"].v0.dtype
        n_steps = int(6.0 / (DT_MS / 1000))
        sched = _drive(n, n_steps, n_steps, amp, dtype, idx=drv)
        step_fn, _ = _compile(model, rule_name, rp)
        st = _fresh_state(model, rule_name, SEED)
        st_mid, out_mid = _run(model, step_fn, st, sched[:40000])
        jax.block_until_ready(out_mid[0])
        st_end, out_run = _run(model, step_fn, st_mid, sched[40000:])
        jax.block_until_ready(out_run[0])
        sp = np.asarray(out_run[1], dtype=float)
        w_tr = np.asarray(out_run[4])
        el = model.params["edge_list"]
        pre = np.asarray(el.pre)
        em = np.flatnonzero(pre == drv)
        tail = sp[-30000:]
        w_tail = w_tr[-30000:]
        w_y = float(w_tail[:, em].mean())
        I_y = G.edge_current_mean(tail[:, drv], w_tail[:, em], tau_ms=5.0)
        rate = float(tail[:, drv].mean() * 10000.0)
        sealed = json.load(open(f"results/battery_{key}.json"))
        row = next(r for r in sealed["rows"] if r["amp"] == amp)
        ref = row["classes"]["E"]
        out[driver] = {
            "amp": amp, "driver_rate": rate, "w": w_y, "I": I_y,
            "ref_rate": row["driver_rate"], "ref_w": ref["w"], "ref_I": ref["I"],
            "rate_match": bool(abs(rate - row["driver_rate"]) < 1e-9),
            "w_match": bool(abs(w_y - ref["w"]) <= 1e-9),
            "I_match": bool(abs(I_y - ref["I"]) <= 1e-9),
        }
        assert abs(rate - row["driver_rate"]) < 1e-9, (driver, rate, row["driver_rate"])
        assert abs(w_y - ref["w"]) <= 1e-3, (driver, w_y, ref["w"])
        assert abs(I_y - ref["I"]) <= 1e-3, (driver, I_y, ref["I"])
    # True bitwise check needs the v0 trajectory: rerun under v0 for comparison.
    from jomission.qualification.hdp_rule_sat_ee import (
        RULE_NAME as V0_NAME, RULE_PARAMS as V0_PARAMS, ensure_registered as ensure_v0)
    ensure_v0()
    for driver, amp in (("PV", 5.0), ("SST", 3.0)):
        for rule_name_c, rp_c in ((rule_name, rp), (V0_NAME, V0_PARAMS)):
            model, drv, tgt = G.build_star(driver_class=driver,
                                           n_targets={"E": 20}, seed=0)
            model = _zero_tonic(model)
            n = int(model.params["emitter"].v0.shape[0])
            dtype = model.params["emitter"].v0.dtype
            n_steps = int(6.0 / (DT_MS / 1000))
            sched = _drive(n, n_steps, n_steps, amp, dtype, idx=drv)
            step_fn, _ = _compile(model, rule_name_c, rp_c)
            st = _fresh_state(model, rule_name_c, SEED)
            st_mid, _ = _run(model, step_fn, st, sched[:40000])
            st_end, out_run = _run(model, step_fn, st_mid, sched[40000:])
            out[driver].setdefault("trajs", {})[rule_name_c] = np.asarray(out_run[4])
        a = out[driver]["trajs"][rule_name]
        b = out[driver]["trajs"][V0_NAME]
        out[driver]["bitwise_w"] = bool(np.array_equal(a, b))
        del out[driver]["trajs"]
    return out


def check_common_modulation(s):
    """E-driver mixed star: EE/E->PV/E->SST share one modulation (<=1% spread)."""
    rule_name = ensure_registered_scale(s)
    rp = scaled_params(s)
    model, drv, tgt = G.build_star(driver_class="E",
                                   n_targets={"E": 10, "PV": 10, "SST": 10}, seed=0)
    import jax.numpy as jnp
    from dataclasses import replace
    el = model.params["edge_list"]
    w0 = np.asarray(el.weight)
    new_el = replace(el, weight=jnp.asarray(np.sign(w0) * scaled_w_base_E(s), dtype=np.float32))
    model = replace(model, params={**model.params, "edge_list": new_el})
    model = _zero_tonic(model)
    n = int(model.params["emitter"].v0.shape[0])
    dtype = model.params["emitter"].v0.dtype
    n_steps = int(6.0 / (DT_MS / 1000))
    sched = _drive(n, n_steps, n_steps, 10.0, dtype, idx=drv)
    step_fn, _ = _compile(model, rule_name, rp)
    st = _fresh_state(model, rule_name, SEED)
    st_mid, _ = _run(model, step_fn, st, sched[:40000])
    st_end, out = _run(model, step_fn, st_mid, sched[40000:])
    w_tr = np.asarray(out[4])
    w_tail = w_tr[-30000:]
    pre = np.asarray(model.params["edge_list"].pre)
    post = np.asarray(model.params["edge_list"].post)
    em = np.flatnonzero(pre == drv)
    means = {}
    tbl = model.neuron_table()
    cls = np.array([str(r["cell_type"]) for r in tbl])
    for c in ("E", "PV", "SST"):
        sel = em[np.isin(post[em], np.flatnonzero(cls == c))]
        means[c] = float(w_tail[:, sel].mean())
    spread = max(means.values()) / min(means.values())
    assert spread <= 1.01, means
    lo, hi = PASS_BANDS["w_active_lo"] * s, PASS_BANDS["w_active_hi"] * s
    assert all(lo <= v <= hi for v in means.values()), (means, lo, hi)
    return {"means": means, "spread": spread}


def check_invariants(s):
    """Off-identity (rest), continuation, determinism at active drive."""
    rule_name = ensure_registered_scale(s)
    rp = scaled_params(s)
    model, _ = ha.calibration_star(n_post=N_POST, seed=0,
                                   w_base=scaled_w_base_E(s), tau_ms=2.0)
    model = _zero_tonic(model)
    n = N_POST + 1
    dtype = model.params["emitter"].v0.dtype
    # Off-identity: rest drive, baseline vs null vs on (three-way exact).
    sched0 = _drive(n, 10000, 10000, 0.0, dtype)
    step_base, _ = jtfne.compile_step_fn(model, dt_ms=DT_MS, kernel="baseline")
    _, out_base = _run(model, step_base, initial_state(model, SEED), sched0)
    step_null, _ = _compile(model, rule_name, {**rp, "k_w": 0.0})
    _, out_null = _run(model, step_null, _fresh_state(model, rule_name, SEED), sched0)
    step_on, _ = _compile(model, rule_name, rp)
    _, out_on = _run(model, step_on, _fresh_state(model, rule_name, SEED), sched0)
    for out in (out_null, out_on):
        np.testing.assert_array_equal(np.asarray(out_base[1]), np.asarray(out[1]))
        np.testing.assert_array_equal(np.asarray(out_base[0]), np.asarray(out[0]))
    # Continuation + determinism at active drive.
    _drv = int(DRIVE_S / (DT_MS / 1000))
    sched_a = _drive(n, _drv, _drv, SHAPE_AMPS["active"], dtype)
    st, out_full = _run(model, step_on, _fresh_state(model, rule_name, SEED), sched_a)
    st_a, _ = _run(model, step_on, _fresh_state(model, rule_name, SEED), sched_a[:40000])
    _, out_b = _run(model, step_on, st_a, sched_a[40000:])
    np.testing.assert_array_equal(np.asarray(out_full[0])[40000:], np.asarray(out_b[0]))
    np.testing.assert_array_equal(np.asarray(out_full[1])[40000:], np.asarray(out_b[1]))
    _, out2 = _run(model, step_on, _fresh_state(model, rule_name, SEED), sched_a)
    np.testing.assert_array_equal(np.asarray(out_full[1]), np.asarray(out2[1]))
    return {"off_identity": "PASS", "continuation": "PASS", "determinism": "PASS"}


def run_scale(s):
    import json
    assert s in S_BRACKET
    tag = scale_tag(s)
    res = {"scale": s, "rule": scaled_rule_name(s)}
    res.update(check_shape(s))
    res["inhibitory_invariance"] = check_inhibitory_invariance(s)
    res["common_modulation"] = check_common_modulation(s)
    res["invariants"] = check_invariants(s)
    # No-clip margin: nothing rides the descriptor/kernel ceiling.
    _, hi = scaled_w_bounds(s)
    assert hi <= 50.0
    res["clip_margin"] = {"descriptor_hi": hi, "assert": "w_max <= w_hi_E*1.05 << hi (see shape)"}
    res["verdict"] = "SELECTIVE_SHAPE_PASS"
    res["rule_params"] = dict(scaled_params(s))
    json.dump(res, open(f"results/selective_shape_s{tag}.json", "w"))
    print(f"s={s} SELECTIVE_SHAPE_PASS")
    return res


def test_sel_s50():
    run_scale(50.0)


def test_sel_s54():
    run_scale(54.0)


def test_sel_s57():
    run_scale(57.0)


def test_sel_s61():
    run_scale(61.0)


def test_sel_s65():
    run_scale(65.0)
