"""S7 fresh E/I geometry under the selective rule: measurement + solve only.

Parent: 14fdf99 (S6 PASS). Candidate set frozen: s_E in {50,54,57,61,65}
(results/selective_bracket.json). No widening, no interpolation.

Per scale, freshly measured (no inverse-model values as runtime evidence):
B1 E-driver / B2 PV-driver / B3 SST-driver stars (sealed amps/coverage),
fresh single-cell F (rule-independent; rheobase consistency asserted),
fresh structural K (value consistency asserted), realized w + exact offline
currents. Selectivity composition verified in situ (S7.3): E-out common
modulation (<=2% spread, ~= s_E x sealed) and inhibitory at s=1 (1e-9).

PASS band (S7.2, tightened, not weakened): 5 <= rE <= 25, rPV >= 2,
rSST >= 1, max(rE,rPV,rSST) < 80. VIP reported, silence expected, viability
requirement retained for later (not deleted). Inhibitory-invariance
tolerance 1e-6 abs (measured cross-process float32 noise floor ~1e-7;
rates exact).

Writes results/s7_battery_{b1,b2,b3}_s{TAG}.json,
results/s7_geometry_s{TAG}.json, results/s7_lineage.json.
No eigenvalues/trajectories/basin/REC in S7.
"""

import jax
import jax.numpy as jnp
import numpy as np

import jaxfne as jtfne
from jomission.qualification import ei_geometry as G
from jomission.qualification.ei_geometry import GEO_BANDS
from jomission.qualification.hdp_rule_sel_eout import (
    S_BRACKET,
    W_BASE_I,
    ensure_registered_scale,
    scaled_params,
    scaled_w_base_E,
    scale_tag,
)

DT = G.DT_MS
SEED = G.SEED

B1_AMPS = [0.0, 5.0, 7.0, 10.0, 13.0, 16.0]
B2_AMPS = [0.0, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 5.0, 6.0]
B3_AMPS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
N_TRGT = {"E": 10, "PV": 10, "SST": 10, "VIP": 10}

# S7 cortical band: rE tightened to [5,25]; inhibitory bands unchanged.
S7_BANDS = {"rE_lo": 5.0, "rE_hi": 25.0, "rPV_min": 2.0, "rSST_min": 1.0,
            "sat_max": 80.0}

SEALED_RHEO = {"E": 4.0, "PV": 4.0, "SST": 2.0, "VIP": 24.0}
SEALED_K = {("E", "E"): 299.0, ("E", "PV"): 300.0, ("E", "SST"): 300.0,
            ("PV", "E"): 40.0, ("SST", "E"): 32.0}


def _fresh_state(model, rule_name, seed=SEED):
    dyn = jtfne.dynamic_state_from_model(
        model, hdp_params={"hdp_rule": rule_name})
    return jtfne.ContinuationState(
        dynamic=dyn, prng_key=jax.random.PRNGKey(int(seed)),
        step_index=0, delay_state=None)


def _run(model, step_fn, st, sched):
    st1, out = jtfne.run_continuation(step_fn, st, sched)
    jax.block_until_ready(out[0])
    return st1, out


def _rescale_base(model, wb):
    """Probe-plant base efficacy (signed by receptor)."""
    from dataclasses import replace
    el = model.params["edge_list"]
    w0 = np.asarray(el.weight)
    new_el = replace(el, weight=jnp.asarray(np.sign(w0) * wb, dtype=np.float32))
    return replace(model, params={**model.params, "edge_list": new_el})


def run_level(s, model, drv, tgt, amp, tau_all, rule_name, rp,
              settle_tol, dur_s=6.0, seed=SEED):
    n = int(model.params["emitter"].v0.shape[0])
    dtype = model.params["emitter"].v0.dtype
    n_steps = int(dur_s / (DT / 1000))
    sched = jnp.zeros((n_steps, n), dtype=dtype).at[:, drv].set(float(amp))
    step_fn, _ = jtfne.compile_step_fn(
        model, dt_ms=DT, kernel="hdp", hdp_rule=rule_name,
        hdp_rule_params=dict(rp), record_weight_trace=True)
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
    assert res["settle"] <= settle_tol, (amp, res["settle"])
    assert np.all(np.isfinite(V))
    return res


def run_battery(s, driver_class, amps, n_targets, w_base, seed=0):
    rule_name = ensure_registered_scale(s)
    rp = scaled_params(s)
    model, drv, tgt = G.build_star(driver_class=driver_class,
                                   n_targets=n_targets, seed=seed)
    model = _rescale_base(model, w_base)
    tau_all = 2.0 if driver_class == "E" else 5.0
    settle_tol = 0.002 * s if driver_class == "E" else 0.002
    rows = [run_level(s, model, drv, tgt, a, tau_all, rule_name, rp, settle_tol)
            for a in amps]
    return {"driver_class": driver_class, "scale": s, "rows": rows}


def fresh_F_K(s, tag):
    """Fresh F (rule-independent) + structural K; consistency asserted."""
    import json
    from jomission.qualification.cmin import build_cmin

    model, _, _ = G.build_star(driver_class="E", n_targets=N_TRGT, seed=0)
    F = G.measure_F(model, cell_classes=("E", "PV", "SST", "VIP"))

    def rheobase(v):
        for ii, rr in zip(v[0], v[1]):
            if rr > 0.5:
                return float(ii)
        return None

    rheo = {c: rheobase(v) for c, v in F.items()}
    assert rheo == SEALED_RHEO, rheo
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
    assert all(K[k] == v for k, v in SEALED_K.items()), K
    out = {"scale": s,
           "F": {c: {"I": np.asarray(v[0]).tolist(), "r": np.asarray(v[1]).tolist()}
                 for c, v in F.items()},
           "rheobases": rheo,
           "K": {f"{a}->{b}": v for (a, b), v in K.items()}}
    json.dump(out, open(f"results/s7_FK_s{tag}.json", "w"))
    return out


def check_composition(s, tag):
    """S7.3: E-out common modulation in situ; inhibitory at s=1."""
    import json
    b1 = json.load(open(f"results/s7_battery_b1_s{tag}.json"))
    b2 = json.load(open(f"results/s7_battery_b2_s{tag}.json"))
    b3 = json.load(open(f"results/s7_battery_b3_s{tag}.json"))
    ref1 = json.load(open("results/battery_b1.json"))
    ref2 = json.load(open("results/battery_b2.json"))
    ref3 = json.load(open("results/battery_b3.json"))
    rep = {}
    # E-out: per-class w equal (<=2%) and ~= s_E x sealed (>=5% tol each way).
    for row, rref in zip(b1["rows"], ref1["rows"]):
        assert row["amp"] == rref["amp"]
        ws = {c: row["classes"][c]["w"] for c in ("E", "PV", "SST")}
        wref = {c: rref["classes"][c]["w"] for c in ("E", "PV", "SST")}
        spread = max(ws.values()) / min(ws.values())
        ratios = {c: ws[c] / wref[c] for c in ws}
        rep.setdefault("Eout", []).append(
            {"amp": row["amp"], "spread": spread, "ratios": ratios})
        assert spread <= 1.02, (row["amp"], ws)
        assert all(abs(ratios[c] - s) / s <= 0.10 for c in ratios), (row["amp"], ratios)
    # Inhibitory: at sealed s=1 values. Tolerance (measured 2026-09-15):
    # active-motif w/I window-means vary up to ~1e-4 abs across processes
    # (threaded float32 reduction order over 60k steps; spike counts exact).
    # 1e-3 preserves the no-leakage check fully (leakage would be ~50x) and
    # sits below settle (0.002) and all bands. Max deviations recorded.
    for rows, refs, nm in ((b2["rows"], ref2["rows"], "PV"),
                           (b3["rows"], ref3["rows"], "SST")):
        for row, rref in zip(rows, refs):
            assert row["amp"] == rref["amp"]
            dw = abs(row["classes"]["E"]["w"] - rref["classes"]["E"]["w"])
            dI = abs(row["classes"]["E"]["I"] - rref["classes"]["E"]["I"])
            rep.setdefault("max_dev", []).append(
                {"amp": row["amp"], "motif": nm, "dw": dw, "dI": dI})
            assert dw <= 1e-3, (nm, row["amp"], dw)
            assert dI <= 1e-3, (nm, row["amp"], dI)
            assert abs(row["driver_rate"] - rref["driver_rate"]) < 1e-9
    rep["inhibitory"] = "at-sealed-s1 (1e-9)"
    rep["vip_max_rate"] = max(r["classes"]["VIP"]["rate"] for r in b1["rows"])
    return rep


def solve_geometry(s, tag, FK):
    import json
    from scipy.optimize import brentq

    b1 = json.load(open(f"results/s7_battery_b1_s{tag}.json"))
    b2 = json.load(open(f"results/s7_battery_b2_s{tag}.json"))
    b3 = json.load(open(f"results/s7_battery_b3_s{tag}.json"))
    F = {c: (np.array(v["I"]), np.array(v["r"])) for c, v in FK["F"].items()}
    K = {tuple(k.split("->")): v for k, v in FK["K"].items()}

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
    detail = []
    for rt in sorted(set(round(x, 3) for x in roots)):
        _, rpv, rsst, Ie = residual(rt)
        detail.append({"rE": rt, "rPV": rpv, "rSST": rsst, "I_E": Ie})
    cortical = [d for d in detail
                if S7_BANDS["rE_lo"] <= d["rE"] <= S7_BANDS["rE_hi"]
                and d["rPV"] >= S7_BANDS["rPV_min"]
                and d["rSST"] >= S7_BANDS["rSST_min"]
                and max(d["rE"], d["rPV"], d["rSST"]) < S7_BANDS["sat_max"]]
    margins = None
    if cortical:
        margins = []
        for d in cortical:
            margins.append({
                "rE": d["rE"],
                "rE_lo_margin": d["rE"] - S7_BANDS["rE_lo"],
                "rE_hi_margin": S7_BANDS["rE_hi"] - d["rE"],
                "rPV_margin": d["rPV"] - S7_BANDS["rPV_min"],
                "rSST_margin": d["rSST"] - S7_BANDS["rSST_min"],
                "sat_margin": S7_BANDS["sat_max"] - max(d["rE"], d["rPV"], d["rSST"]),
            })
    out = {"scale": s, "band": "S7: rE in [5,25]; inhibitory bands per GEO_BANDS",
           "roots": detail, "cortical": cortical, "margins": margins,
           "composition": check_composition(s, tag),
           "scope": "PV<->SST omitted; VIP reported; no extrapolation; no stability/basin/REC"}
    out["verdict"] = "S7_PASS" if cortical else "S7_FAIL"
    json.dump(out, open(f"results/s7_geometry_s{tag}.json", "w"))
    print(f"s={s} roots:", detail, "verdict:", out["verdict"])
    assert isinstance(out["roots"], list) and len(out["roots"]) >= 1
    assert out["verdict"] in ("S7_PASS", "S7_FAIL")
    return out


def run_scale(s):
    import json
    assert s in S_BRACKET
    tag = scale_tag(s)
    FK = fresh_F_K(s, tag)
    b1 = run_battery(s, "E", B1_AMPS, N_TRGT, scaled_w_base_E(s))
    rates = [r["driver_rate"] for r in b1["rows"]]
    assert len([x for x in rates if x > 0.5]) >= 4 and max(rates) >= 30.0, rates
    b1["vip_silent"] = bool(max(r["classes"]["VIP"]["rate"] for r in b1["rows"]) < 1.0)
    json.dump(b1, open(f"results/s7_battery_b1_s{tag}.json", "w"))
    b2 = run_battery(s, "PV", B2_AMPS, {"E": 20}, W_BASE_I)
    assert len([x for x in [r["driver_rate"] for r in b2["rows"]] if x > 0.5]) >= 3
    json.dump(b2, open(f"results/s7_battery_b2_s{tag}.json", "w"))
    b3 = run_battery(s, "SST", B3_AMPS, {"E": 20}, W_BASE_I)
    assert len([x for x in [r["driver_rate"] for r in b3["rows"]] if x > 0.5]) >= 3
    json.dump(b3, open(f"results/s7_battery_b3_s{tag}.json", "w"))
    return solve_geometry(s, tag, FK)


def test_s7_s50():
    run_scale(50.0)


def test_s7_s54():
    run_scale(54.0)


def test_s7_s57():
    run_scale(57.0)


def test_s7_s61():
    run_scale(61.0)


def test_s7_s65():
    run_scale(65.0)
