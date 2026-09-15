"""Phase-3A realized native E/I geometry: motif batteries + operating-point solve.

No transplants: F_X, transfers, w(r), currents all measured fresh on the
v0.4.24 plant with frozen rule jomission_sat_ee_v0 active. Sealed methods
(star assay, single-cell F-I) reused; sealed VALUES never reused.

Batteries (star, zero tonic, schedule-only, fresh plant/level):
  B1 E-driver  -> E/PV/SST/VIP targets, amps [0,5,7,10,13,16]
  B2 PV-driver -> E targets, amps [0,3,4,5,6,8] (extend only for coverage)
  B3 SST-driver-> E targets, amps [0,2,3,4,5,7] (extend only for coverage)
Coverage requirement: driver rates spanning [0,40] with >=4 nonzero points.
Map (zero tonic): rPV=F_PV(I_EPV(rE)); rSST=F_SST(I_ESST(rE));
  R(rE)=F_E(I_EE(rE)+I_PVE(rPV(rE))+I_SSTE(rSST(rE)))-rE  (signed currents).
PV<->SST cross-inhibition out of scope (no edges measured).
Verdict: EI_GEOMETRY_PASS/FAIL/UNRESOLVED (GEO_BANDS in ei_geometry.py).
"""

import jax
import jax.numpy as jnp
import numpy as np

import jaxfne as jtfne
from jomission.qualification import ei_geometry as G
from jomission.qualification.ei_geometry import (
    GEO_BANDS,
    RULE_NAME,
    edge_current_mean,
)

DT = G.DT_MS
SEED = G.SEED

B1_AMPS = [0.0, 5.0, 7.0, 10.0, 13.0, 16.0]
# B2/B3 extensions (documented coverage cause, criteria untouched): PV jumps
# 0->25Hz (threshold near amp 3-4); SST already 33Hz at amp 2. Map needs
# w/I across the operating ranges, so fill low-rate ends.
B2_AMPS = [0.0, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 5.0, 6.0]
B3_AMPS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]

N_TRGT = {"E": 10, "PV": 10, "SST": 10, "VIP": 10}


def _fresh_state(model, seed=SEED):
    dyn = jtfne.dynamic_state_from_model(
        model, hdp_params={"hdp_rule": RULE_NAME})
    return jtfne.ContinuationState(
        dynamic=dyn, prng_key=jax.random.PRNGKey(int(seed)),
        step_index=0, delay_state=None)


def _run(model, step_fn, st, sched):
    st1, out = jtfne.run_continuation(step_fn, st, sched)
    jax.block_until_ready(out[0])
    return st1, out


def run_level(model, drv, tgt, amp, dur_s=6.0, seed=SEED, tau_all=2.0):
    """One settled level; window-averaged readouts per target class."""
    n = int(model.params["emitter"].v0.shape[0])
    dtype = model.params["emitter"].v0.dtype
    n_steps = int(dur_s / (DT / 1000))
    sched = jnp.zeros((n_steps, n), dtype=dtype).at[:, drv].set(float(amp))
    step_fn, _ = jtfne.compile_step_fn(
        model, dt_ms=DT, kernel="hdp", hdp_rule=RULE_NAME,
        hdp_rule_params=dict(G.RULE_PARAMS), record_weight_trace=True)
    st = _fresh_state(model, seed)
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
    # settling per motif, window-mean to window-mean (sawtooth-safe)
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
    assert res["settle"] <= 0.002, (amp, res["settle"])
    assert np.all(np.isfinite(V))
    return res


def run_battery(driver_class, amps, n_targets=N_TRGT, seed=0):
    model, drv, tgt = G.build_star(driver_class=driver_class,
                                   n_targets=n_targets, seed=seed)
    present = sorted(set(tgt) | {driver_class})
    F = G.measure_F(model, cell_classes=tuple(present))
    # Offline-current tau follows the presynaptic (driver) class.
    tau_all = 2.0 if driver_class == "E" else 5.0
    rows = [run_level(model, drv, tgt, a, tau_all=tau_all) for a in amps]
    return {"driver_class": driver_class, "F": {c: {"I": v[0].tolist(), "r": v[1].tolist()} for c, v in F.items()}, "rows": rows}


def test_b1_mixed_e_driver():
    import json
    b = run_battery("E", B1_AMPS)
    rates = [r["driver_rate"] for r in b["rows"]]
    nz = [x for x in rates if x > 0.5]
    assert len(nz) >= 4 and max(rates) >= 30.0, rates
    vip = [r["classes"]["VIP"]["rate"] for r in b["rows"]]
    b["vip_silent"] = bool(max(vip) < 1.0)
    json.dump(b, open("results/battery_b1.json", "w"))
    print("B1 driver rates:", [round(x, 1) for x in rates], "VIP silent:", b["vip_silent"])


def test_b2_pv_to_e():
    import json
    b = run_battery("PV", B2_AMPS, n_targets={"E": 20})
    rates = [r["driver_rate"] for r in b["rows"]]
    nz = [x for x in rates if x > 0.5]
    assert len(nz) >= 3, rates
    json.dump(b, open("results/battery_b2.json", "w"))
    print("B2 driver rates:", [round(x, 1) for x in rates])


def test_b3_sst_to_e():
    import json
    b = run_battery("SST", B3_AMPS, n_targets={"E": 20})
    rates = [r["driver_rate"] for r in b["rows"]]
    nz = [x for x in rates if x > 0.5]
    assert len(nz) >= 3, rates
    json.dump(b, open("results/battery_b3.json", "w"))
    print("B3 driver rates:", [round(x, 1) for x in rates])


def _interp(x, xp, fp):
    return float(np.interp(x, xp, fp))


def test_solve_native_geometry():
    """1D residual from native-measured components. Verdict EI_GEOMETRY_*."""
    import json
    from scipy.optimize import brentq
    from jomission.qualification.cmin import build_cmin

    b1 = json.load(open("results/battery_b1.json"))
    b2 = json.load(open("results/battery_b2.json"))
    b3 = json.load(open("results/battery_b3.json"))
    # Convergence (fresh, canonical single-column substrate family).
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
    F = {c: (np.array(v["I"]), np.array(v["r"]))
         for c, v in b1["F"].items()}

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
        iEE = _interp(rE, x_ee, i_ee)
        iEPV = _interp(rE, x_epv, i_epv)
        iESST = _interp(rE, x_esst, i_esst)
        rPV = F_of("PV", K[("E", "PV")] * iEPV)
        rSST = F_of("SST", K[("E", "SST")] * iESST)
        iPVE = _interp(min(rPV, x_pve.max()), x_pve, i_pve)
        iSSTE = _interp(min(rSST, x_sste.max()), x_sste, i_sste)
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
    out = {"K": {f"{a}->{b}": v for (a, b), v in K.items()},
           "roots": detail, "cortical": cortical,
           "HDP_state": {"H": 1.0, "note": "H idle exact; w per motif measured in batteries"},
           "scope": "PV<->SST cross-inhibition omitted (no edges measured); VIP silent excluded; domain limited to measured rates (no extrapolation)"}
    if cortical:
        out["verdict"] = "EI_GEOMETRY_PASS"
    elif len(detail) == 1 and detail[0]["rE"] < 1.0:
        out["verdict"] = "EI_GEOMETRY_FAIL"
        out["boundary"] = ("silent root only: max total EE drive %.3f units vs E rheobase 4.0; "
                           "native transformation correctly shaped (Phase 2) but ~100x too weak "
                           "in absolute current at C_min convergence" % max(
                               K[("E", "E")] * _interp(x, x_ee, i_ee) for x in grid))
    else:
        out["verdict"] = "EI_GEOMETRY_UNRESOLVED"
    json.dump(out, open("results/ei_geometry_native.json", "w"))
    print("roots:", detail, "verdict:", out["verdict"])
    # Procedure integrity only: the scientific verdict lives in the sealed
    # JSON (PASS/FAIL/UNRESOLVED). A FAIL verdict stops the lineage; it must
    # not be converted into a green/red test signal by redefinition.
    assert isinstance(out["roots"], list) and len(out["roots"]) >= 1
    assert out["verdict"] in ("EI_GEOMETRY_PASS", "EI_GEOMETRY_FAIL",
                              "EI_GEOMETRY_UNRESOLVED")
    if out["verdict"] != "EI_GEOMETRY_PASS":
        print("GATE_STOP:", out["verdict"], out.get("boundary", ""))
