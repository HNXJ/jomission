"""HDP efficacy-scale lineage: shape requalification + E/I geometry per scale.

One delta only: ``w_eff(H;s) = s*w_eff(H;1)`` via scaled rule instances
(see jomission/qualification/hdp_scale.py). Frozen: rule form, H-dynamics,
shape params, cells, inhibition, tonic, homeostatic config, GEO_BANDS,
probe/battery amps, F curves (rule-independent), K (structural).

Run ascending (``-k s2``, then ``s4p5``, ...); stop at the first PASS /
PATHOLOGICAL per the predeclared stop rule. Each scale writes::

    results/hdp_scale_shape_s{TAG}.json
    results/hdp_scale_battery_{b1,b2,b3}_s{TAG}.json
    results/hdp_scale_geometry_s{TAG}.json
"""

import jax
import jax.numpy as jnp
import numpy as np

import jaxfne as jtfne
from jomission.qualification import hdp_attractor as ha
from jomission.qualification import ei_geometry as G
from jomission.qualification.ei_geometry import GEO_BANDS
from jomission.qualification.hdp_rule_sat_ee import PASS_BANDS
from jomission.qualification.hdp_scale import (
    SCALE_BRACKET,
    ensure_registered_scale,
    scaled_params,
    scaled_rule_name,
    scaled_w_base,
    scaled_w_bounds,
    scale_tag,
)

DT_MS = 0.1
SEED = 11
N_POST = 50
DRIVE_S = 6.0
POST_S = 2.0
MID_S = 4.0

# Landed Phase-2 knobs (measured-rate probes; below-band uses the sealed
# rest substitution, driver F-I jump is rule-independent by construction).
SHAPE_AMPS = {"rest": 0.0, "transition": 5.0, "active": 10.0, "above": 16.0}

B1_AMPS = [0.0, 5.0, 7.0, 10.0, 13.0, 16.0]
B2_AMPS = [0.0, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 5.0, 6.0]
B3_AMPS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
N_TRGT = {"E": 10, "PV": 10, "SST": 10, "VIP": 10}


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


def run_shape_probe(s, amp):
    """All-E star shape probe at one landed knob under scale s."""
    rule_name = ensure_registered_scale(s)
    rp = scaled_params(s)
    wb = scaled_w_base(s)
    model, _ = ha.calibration_star(n_post=N_POST, seed=0, w_base=wb, tau_ms=2.0)
    model = _zero_tonic(model)
    n = N_POST + 1
    dtype = model.params["emitter"].v0.dtype
    drv_steps = int(DRIVE_S / (DT_MS / 1000))
    post_steps = int(POST_S / (DT_MS / 1000))
    mid_steps = int(MID_S / (DT_MS / 1000))
    sched = jnp.zeros((drv_steps + post_steps, n), dtype=dtype).at[:drv_steps, 0].set(float(amp))
    step_fn, _ = _compile(model, rule_name, rp)
    st = _fresh_state(model, rule_name, SEED)
    st_mid, _ = _run(model, step_fn, st, sched[:mid_steps])
    st_end, out = _run(model, step_fn, st_mid, sched[mid_steps:drv_steps])
    st_post, out_post = _run(model, step_fn, st_end, sched[drv_steps:])
    V = np.asarray(out[0])
    sp = np.asarray(out[1], dtype=float)
    w_tr = np.asarray(out[4])
    w_post_tr = np.asarray(out_post[4])
    tail = sp[-30000:, 0]
    return {
        "amp": float(amp), "rate": float(tail.mean() * 10000.0),
        "w_mid": float(w_tr[:10000].mean()), "w": float(w_tr[-30000:].mean()),
        "w_std": float(w_tr[-30000:].std()), "w_max": float(w_tr[-30000:].max()),
        "w_post": float(w_post_tr[-10000:].mean()),
        "H_idle": float(np.abs(np.asarray(st_end.dynamic.H) - 1.0).max()),
        "aux_end": float(np.asarray(st_end.dynamic.aux).mean()),
        "spikes": sp, "w_tr": w_tr[-30000:],
    }


def edge_current_mean(sp_drv, w_tr, tau_ms=2.0, dt_ms=0.1):
    decay = float(np.exp(-dt_ms / tau_ms))
    syn = np.empty(sp_drv.shape[0])
    acc = 0.0
    for t in range(sp_drv.shape[0]):
        acc = acc * decay + sp_drv[t]
        syn[t] = acc
    return float((w_tr * syn[:, None]).mean())


def check_shape(s):
    """Shape requalification at scale s; returns sealed result dict."""
    import json
    assert s in SCALE_BRACKET
    tag = scale_tag(s)
    rule_name = ensure_registered_scale(s)
    rp = scaled_params(s)
    wb = scaled_w_base(s)
    w_lo, w_hi = rp["w_lo"], rp["w_hi"]
    landed = {k: run_shape_probe(s, a) for k, a in SHAPE_AMPS.items()}
    res = {}
    for k, r in landed.items():
        res[k] = {kk: vv for kk, vv in r.items() if kk not in ("spikes", "w_tr")}
        assert abs(r["w"] - r["w_mid"]) <= 0.001 * s, (k, r["w"], r["w_mid"])
        assert r["H_idle"] == 0.0, (k, r["H_idle"])
    for k in ("rest", "transition", "active", "above"):
        landed[k]["I_exact"] = edge_current_mean(
            landed[k]["spikes"][-30000:, 0], landed[k]["w_tr"])
        res[k]["I_exact"] = landed[k]["I_exact"]
    # Below-band sealed substitution (rest represents the sub-separatrix regime).
    res["below"] = dict(res["rest"])
    res["below"]["substituted"] = True
    res["below"]["substitution_cause"] = (
        "sealed Phase-2 substitution: F-I jump 0->7Hz; driver rule-independent")
    order = ["rest", "below", "transition", "active", "above"]
    Is = [res[n]["I_exact"] for n in order]
    # Scaled absolute bands; ratios invariant.
    assert res["rest"]["w"] <= PASS_BANDS["w_rest_max"] * s
    assert res["transition"]["w"] / max(res["below"]["w"], 1e-12) >= PASS_BANDS["w_trans_over_below_min"]
    assert PASS_BANDS["w_active_lo"] * s <= res["active"]["w"] <= PASS_BANDS["w_active_hi"] * s
    assert res["above"]["w"] <= PASS_BANDS["w_above_hi"] * s
    assert res["above"]["w"] / max(res["active"]["w"], 1e-12) <= PASS_BANDS["w_above_over_active_max"]
    assert all(b >= a for a, b in zip(Is, Is[1:])), Is
    assert Is[3] >= PASS_BANDS["I_active_min"] * s, Is
    # Causal on/off at matched active drive (null rule, scaled base).
    model, _ = ha.calibration_star(n_post=N_POST, seed=0, w_base=wb, tau_ms=2.0)
    model = _zero_tonic(model)
    n = N_POST + 1
    dtype = model.params["emitter"].v0.dtype
    step_fn, _ = _compile(model, rule_name, {**rp, "k_w": 0.0})
    sched = jnp.zeros((int(DRIVE_S / (DT_MS / 1000)), n), dtype=dtype).at[:, 0].set(
        SHAPE_AMPS["active"])
    _, out = _run(model, step_fn, _fresh_state(model, rule_name, SEED), sched)
    null_sp = np.asarray(out[1], dtype=float)[:, 0]
    I_off = edge_current_mean(null_sp[-30000:], np.full((30000, N_POST), wb))
    res["I_off_active"] = I_off
    assert Is[3] / max(I_off, 1e-12) >= PASS_BANDS["I_on_over_off_min"], (Is[3], I_off)
    assert res["active"]["w_post"] <= PASS_BANDS["w_post_max"] * s
    assert max(res[n]["w_max"] for n in order) <= w_hi * 1.05
    # Realized w/I scaling vs s=1 sealed reference (linear prediction).
    ref = json.load(open("results/native_transformation.json"))
    for k in ("transition", "active", "above"):
        assert abs(res[k]["w"] / max(ref[k]["w"], 1e-12) - s) / s <= 0.05, (k, res[k]["w"], ref[k]["w"])
        assert abs(res[k]["I_exact"] / max(ref[k]["I_exact"], 1e-12) - s) / s <= 0.10, (k, res[k]["I_exact"], ref[k]["I_exact"])
    res["verdict"] = "SCALE_SHAPE_PASS"
    res["scale"] = s
    res["rule"] = rule_name
    res["rule_params"] = dict(rp)
    res["w_bounds"] = list(scaled_w_bounds(s))
    json.dump(res, open(f"results/hdp_scale_shape_s{tag}.json", "w"))
    print(f"s={s} SCALE_SHAPE_PASS")
    return res


def run_geo_level(s, model, drv, tgt, amp, tau_all, rule_name, rp, dur_s=6.0, seed=SEED):
    n = int(model.params["emitter"].v0.shape[0])
    dtype = model.params["emitter"].v0.dtype
    n_steps = int(dur_s / (DT_MS / 1000))
    sched = jnp.zeros((n_steps, n), dtype=dtype).at[:, drv].set(float(amp))
    step_fn, _ = _compile(model, rule_name, rp)
    st = _fresh_state(model, rule_name, seed)
    st_mid, out_mid = _run(model, step_fn, st, sched[:40000])
    jax.block_until_ready(out_mid[0])
    st_end, out = _run(model, step_fn, st_mid, sched[40000:])
    jax.block_until_ready(out[0])
    V = np.asarray(out[0])
    sp = np.asarray(out[1], dtype=float)
    w_tr = np.asarray(out[4])
    w_tr_mid = np.asarray(out_mid[4])
    el = model.params["edge_list"]
    pre = np.asarray(el.pre)
    tail = sp[-30000:]
    w_tail = w_tr[-30000:]
    res = {"amp": float(amp),
           "driver_rate": float(tail[:, drv].mean() * 10000.0),
           "H_idle": float(np.abs(np.asarray(st_end.dynamic.H) - 1.0).max())}
    worst = 0.0
    out_c = {}
    em = np.flatnonzero(pre == drv)
    for c, idx in tgt.items():
        post = np.asarray(el.post)
        sel = em[np.isin(post[em], np.asarray(idx))]
        r_y = tail[:, np.asarray(idx)].mean() * 10000.0
        w_y = w_tail[:, sel].mean()
        w_y_mid = w_tr_mid[-10000:, sel].mean()
        worst = max(worst, abs(float(w_y) - float(w_y_mid)))
        I_y = G.edge_current_mean(tail[:, drv], w_tail[:, sel], tau_ms=tau_all)
        out_c[c] = {"rate": float(r_y), "w": float(w_y),
                    "w_max": float(w_tail[:, sel].max()), "I": float(I_y),
                    "n_edges": int(sel.shape[0])}
    res["settle"] = float(worst)
    res["classes"] = out_c
    assert res["H_idle"] == 0.0
    assert res["settle"] <= 0.002 * s, (amp, res["settle"])
    assert np.all(np.isfinite(V))
    return res


def run_geo_battery(s, driver_class, amps, n_targets, seed=0):
    rule_name = ensure_registered_scale(s)
    rp = scaled_params(s)
    wb = scaled_w_base(s)
    model, drv, tgt = G.build_star(driver_class=driver_class,
                                   n_targets=n_targets, seed=seed)
    # Scaled probe-plant base (sign by receptor, as in the sealed assay).
    from dataclasses import replace
    el = model.params["edge_list"]
    w0 = np.asarray(el.weight)
    new_el = replace(el, weight=jnp.asarray(np.sign(w0) * wb, dtype=np.float32))
    model = replace(model, params={**model.params, "edge_list": new_el})
    tau_all = 2.0 if driver_class == "E" else 5.0
    rows = [run_geo_level(s, model, drv, tgt, a, tau_all, rule_name, rp) for a in amps]
    return {"driver_class": driver_class, "scale": s, "rows": rows}


def solve_geometry(s, tag):
    """Fresh K + sealed F + scaled batteries -> operating points + verdict."""
    import json
    from scipy.optimize import brentq
    from jomission.qualification.cmin import build_cmin

    b1 = json.load(open(f"results/hdp_scale_battery_b1_s{tag}.json"))
    b2 = json.load(open(f"results/hdp_scale_battery_b2_s{tag}.json"))
    b3 = json.load(open(f"results/hdp_scale_battery_b3_s{tag}.json"))
    F_sealed = json.load(open("results/battery_b1.json"))["F"]
    m = build_cmin()
    el = m.params["edge_list"]
    tbl = m.neuron_table()
    cls = np.array([str(r["cell_type"]) for r in tbl])
    pre = np.asarray(el.pre)
    post = np.asarray(el.post)
    K = {}
    for cpost in ("E", "PV", "SST"):
        for cpre in ("E", "PV", "SST"):
            sel = (cls[post] == cpost) & (cls[pre] == cpre)
            K[(cpre, cpost)] = float(sel.sum() / max((cls == cpost).sum(), 1))
    F = {c: (np.array(v["I"]), np.array(v["r"])) for c, v in F_sealed.items()}

    def curve(rows, key):
        xs = np.array([r["driver_rate"] for r in rows])
        ys = np.array([r["classes"][key]["I"] if key in r["classes"]
                       else r["classes"]["E"]["I"] for r in rows])
        o = np.argsort(xs)
        return xs[o], ys[o]

    x_ee, i_ee = curve(b1["rows"], "E")
    x_epv, i_epv = curve(b1["rows"], "PV")
    x_esst, i_esst = curve(b1["rows"], "SST")
    x_pve, i_pve = curve(b2["rows"], "E")
    x_sste, i_sste = curve(b3["rows"], "E")

    def F_of(c, I):
        Ii, rr = F[c]
        return float(np.interp(max(I, 0.0), Ii, rr))

    def residual(rE):
        iEE = float(np.interp(rE, x_ee, i_ee))
        iEPV = float(np.interp(rE, x_epv, i_epv))
        iESST = float(np.interp(rE, x_esst, i_esst))
        rPV = F_of("PV", K[("E", "PV")] * iEPV)
        rSST = F_of("SST", K[("E", "SST")] * iESST)
        iPVE = float(np.interp(min(rPV, x_pve.max()), x_pve, i_pve))
        iSSTE = float(np.interp(min(rSST, x_sste.max()), x_sste, i_sste))
        Ie = K[("E", "E")] * iEE + K[("PV", "E")] * iPVE + K[("SST", "E")] * iSSTE
        return F_of("E", Ie) - rE, rPV, rSST, Ie

    grid = np.arange(0.0, 40.5, 0.5)
    vals = [residual(x) for x in grid]
    R = np.array([v[0] for v in vals])
    roots = []
    for a, b, ra, rb in zip(grid[:-1], grid[1:], R[:-1], R[1:]):
        if ra == 0.0:
            roots.append(float(a))
        elif ra * rb < 0.0:
            roots.append(float(brentq(lambda x: residual(x)[0], a, b)))
    roots = sorted(set([round(x, 3) for x in roots]))
    detail = []
    for rt in roots:
        rr, rpv, rsst, Ie = residual(rt)
        detail.append({"rE": rt, "rPV": rpv, "rSST": rsst, "I_E": Ie})
    cortical = [d for d in detail
                if GEO_BANDS["rE_lo"] <= d["rE"] <= GEO_BANDS["rE_hi"]
                and d["rPV"] >= GEO_BANDS["rPV_min"]
                and d["rSST"] >= GEO_BANDS["rSST_min"]
                and max(d["rE"], d["rPV"], d["rSST"]) < GEO_BANDS["sat_max"]]
    # Pathology flags: saturation-adjacent or clip-riding efficacies.
    _, hi = scaled_w_bounds(s)
    clip_ride = False
    for b in (b1, b2, b3):
        for r in b["rows"]:
            for c, d in r["classes"].items():
                if abs(d["w"]) >= 0.99 * hi:
                    clip_ride = True
    vip_recruit = bool(max(r["classes"]["VIP"]["rate"] for r in b1["rows"]) >= 1.0)
    out = {"scale": s, "K": {f"{a}->{b}": v for (a, b), v in K.items()},
           "roots": detail, "cortical": cortical,
           "clip_ride": clip_ride, "vip_recruited": vip_recruit,
           "F_source": "sealed results/battery_b1.json (rule-independent emitter property)",
           "HDP_state": {"H": 1.0, "note": "H idle exact; w per motif measured in scaled batteries"},
           "scope": "PV<->SST cross-inhibition omitted (no edges measured); VIP excluded from map; domain limited to measured rates (no extrapolation)"}
    if cortical and (clip_ride or max(max(d["rE"], d["rPV"], d["rSST"]) for d in cortical) >= GEO_BANDS["sat_max"]):
        out["verdict"] = "SCALE_PATHOLOGICAL"
    elif cortical:
        out["verdict"] = "EI_GEOMETRY_PASS"
    elif len(detail) == 1 and detail[0]["rE"] < 1.0:
        out["verdict"] = "SCALE_FAIL"
        out["boundary"] = ("silent root only at s=%.1f: max total EE drive %.3f units vs E rheobase 4.0" % (
            s, max(K[("E", "E")] * float(np.interp(x, x_ee, i_ee)) for x in grid)))
    else:
        out["verdict"] = "SCALE_UNRESOLVED"
    json.dump(out, open(f"results/hdp_scale_geometry_s{tag}.json", "w"))
    print(f"s={s} roots:", detail, "verdict:", out["verdict"])
    assert isinstance(out["roots"], list) and len(out["roots"]) >= 1
    assert out["verdict"] in ("EI_GEOMETRY_PASS", "SCALE_PATHOLOGICAL",
                              "SCALE_FAIL", "SCALE_UNRESOLVED")
    return out


def run_scale(s):
    import json
    assert s in SCALE_BRACKET
    tag = scale_tag(s)
    check_shape(s)
    b1 = run_geo_battery(s, "E", B1_AMPS, N_TRGT)
    rates = [r["driver_rate"] for r in b1["rows"]]
    assert len([x for x in rates if x > 0.5]) >= 4 and max(rates) >= 30.0, rates
    b1["vip_silent"] = bool(max(r["classes"]["VIP"]["rate"] for r in b1["rows"]) < 1.0)
    json.dump(b1, open(f"results/hdp_scale_battery_b1_s{tag}.json", "w"))
    b2 = run_geo_battery(s, "PV", B2_AMPS, {"E": 20})
    assert len([x for x in [r["driver_rate"] for r in b2["rows"]] if x > 0.5]) >= 3
    json.dump(b2, open(f"results/hdp_scale_battery_b2_s{tag}.json", "w"))
    b3 = run_geo_battery(s, "SST", B3_AMPS, {"E": 20})
    assert len([x for x in [r["driver_rate"] for r in b3["rows"]] if x > 0.5]) >= 3
    json.dump(b3, open(f"results/hdp_scale_battery_b3_s{tag}.json", "w"))
    return solve_geometry(s, tag)


def test_scale_s2():
    run_scale(2.0)


def test_scale_s4p5():
    run_scale(4.5)


def test_scale_s9():
    run_scale(9.0)


def test_scale_s18():
    run_scale(18.0)


def test_scale_s36():
    run_scale(36.0)
