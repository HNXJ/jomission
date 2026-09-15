"""S9 native basin battery: construction, release, adjudication, verdict.

Writes results/s9_init_s{TAG}_{INIT}.json, results/s9_candidate_s{TAG}.json,
results/s9_lineage.json. Stops before REC intervention.
"""

import jax
import jax.numpy as jnp
import numpy as np

import jaxfne as jtfne
from jomission.qualification import basin as B
from jomission.qualification.basin import CAPTURE
from jomission.qualification.cmin import build_cmin
from jomission.qualification.hdp_rule_sel_eout import (
    S_BRACKET,
    ensure_registered_scale,
    scaled_params,
    scaled_w_bounds,
    scale_tag,
)

DT = B.DT_MS
SEED = B.SEED
N_CHUNK = int(B.RELEASE_S / B.CHUNK_S)
STEPS_CHUNK = int(B.CHUNK_S / (DT / 1000))


def _fresh_state(model, rule_name, seed=SEED):
    dyn = jtfne.dynamic_state_from_model(
        model, hdp_params={"hdp_rule": rule_name})
    return jtfne.ContinuationState(
        dynamic=dyn, prng_key=jax.random.PRNGKey(int(seed)),
        step_index=0, delay_state=None)


def build_plant(s):
    """C-min column, zero tonic, selective rule; returns (model, step_fn)."""
    rule_name = ensure_registered_scale(s)
    model = build_cmin()
    e = model.params["emitter"]
    model = model.with_emitter_parameters(
        drive_per_neuron=jnp.zeros_like(e.drive))
    step_fn, _ = jtfne.compile_step_fn(
        model, dt_ms=DT, kernel="hdp", hdp_rule=rule_name,
        hdp_rule_params=dict(scaled_params(s)), record_weight_trace=False)
    return model, step_fn, rule_name


def construct_state(model, rule_name, s, init):
    """Rest-consistent construction (P0 neural rest; P7 rest + FP aux/w).

    Active-state V construction is abandoned (unphysical V-u pairing):
    P1/P3/P4/P5 use driven-preparation snapshots instead.
    """
    assert init in ("P0", "P7"), init
    fp = B.fp_tables(s)
    _, hi = scaled_w_bounds(s)
    tbl = model.neuron_table()
    cls = np.array([str(r["cell_type"]) for r in tbl])
    n = len(cls)
    el = model.params["edge_list"]
    pre = np.asarray(el.pre)
    post = np.asarray(el.post)
    pre_cls = cls[pre]
    post_cls = cls[post]
    e = model.params["emitter"]
    a = np.asarray(e.a, dtype=np.float64)
    bb = np.asarray(e.b, dtype=np.float64)
    cc = np.asarray(e.c, dtype=np.float64)
    rpre = {"E": fp["r"]["E"], "PV": fp["r"]["PV"], "SST": fp["r"]["SST"],
            "VIP": 0.0}

    def ptag_edge(pe, po):
        if pe == "E" and po == "E":
            return "EE"
        if pe == "E" and po == "PV":
            return "EPV"
        if pe == "E" and po == "SST":
            return "ESST"
        if pe == "E" and po == "VIP":
            return "EVIP"
        if pe == "PV":
            return "PVE"
        if pe == "SST":
            return "SSTE"
        return "VX"

    tags = np.array([ptag_edge(pe, po) for pe, po in zip(pre_cls, post_cls)])
    w_fp = np.zeros_like(pre, dtype=np.float64)
    for t in ("EE", "EPV", "ESST", "EVIP"):
        w_fp[tags == t] = fp["wpath"][t]
    for t in ("PVE", "SSTE"):
        w_fp[tags == t] = -abs(fp["wpath"][t])
    vx = tags == "VX"
    w_fp[vx] = np.asarray(el.weight)[vx].astype(np.float64)  # untouched motifs
    rpre = {"E": fp["r"]["E"], "PV": fp["r"]["PV"], "SST": fp["r"]["SST"],
            "VIP": 0.0}
    aux_fp = np.array([rpre.get(pe, 0.0) * B.TAU_ACT / 1000.0 for pe in pre_cls])
    H_fp = np.ones(n)
    u_rest = bb * cc
    V_rest = cc.copy()

    w = w_fp.copy()
    aux = aux_fp.copy()
    H = H_fp.copy()
    u = u_rest.copy()
    V = V_rest.copy()
    if init == "P0":
        V, u, H = V_rest.copy(), u_rest.copy(), np.ones(n)
        aux = np.zeros_like(aux)
        w = np.asarray(el.weight).astype(np.float64)
    elif init == "P7":
        V, u = V_rest.copy(), u_rest.copy()
    else:
        raise AssertionError(init)
    st0 = _fresh_state(model, rule_name, SEED)
    d = st0.dynamic
    dyn = d._replace(
        v=jnp.asarray(V, dtype=d.v.dtype),
        u=jnp.asarray(u, dtype=d.u.dtype),
        H=jnp.asarray(H, dtype=d.H.dtype),
        aux=jnp.asarray(aux, dtype=d.aux.dtype),
        w=jnp.asarray(w, dtype=d.w.dtype),
        syn_state=jnp.zeros_like(d.syn_state),
        prev_spikes=jnp.zeros_like(d.prev_spikes),
    )
    # Model edge weights track construction (kernel reads carry; belt+suspenders).
    from dataclasses import replace
    new_el = replace(el, weight=jnp.asarray(np.sign(w) * np.abs(w), dtype=np.asarray(el.weight).dtype))
    model2 = replace(model, params={**model.params, "edge_list": new_el})
    return model2, dyn


def run_release(model, step_fn, rule_name, dyn):
    """20 s zero-drive release in 5 s chunks; summaries + final carry."""
    n = int(model.params["emitter"].v0.shape[0])
    dtype = model.params["emitter"].v0.dtype
    tbl = model.neuron_table()
    cls = np.array([str(r["cell_type"]) for r in tbl])
    st = jtfne.ContinuationState(dynamic=dyn, prng_key=jax.random.PRNGKey(SEED),
                                 step_index=0, delay_state=None)
    segs = []
    snaps = []
    last = None
    for _ in range(N_CHUNK):
        sched = jnp.zeros((STEPS_CHUNK, n), dtype=dtype)
        st, out = jtfne.run_continuation(step_fn, st, sched)
        jax.block_until_ready(out[0])
        sp = np.asarray(out[1], dtype=float)
        rates = {c: float(sp[:, cls == c].mean() * 10000.0)
                 for c in ("E", "PV", "SST", "VIP")}
        segs.append(rates)
        dd = st.dynamic
        el = model.params["edge_list"]
        pre = np.asarray(el.pre)
        wv = np.asarray(dd.w, dtype=float)
        snaps.append({"w_Epre": float(wv[cls[pre] == "E"].mean()),
                      "w_Ipre": float(np.abs(wv[cls[pre] != "E"]).mean()),
                      "H": float(np.asarray(dd.H, dtype=float).mean()),
                      "aux": float(np.asarray(dd.aux, dtype=float).mean())})
        last = (sp[:, cls == "E"], dd)
    return segs, snaps, last


def adjudicate(segs, snaps, last):
    """Capture/collapse/other per the predeclared criterion."""
    sp_last, dd_last = last
    # NOTE: chunk is 5 s = 50000 steps; halves below are 2.5 s windows used
    # only for the synchrony guard. Rates/drift come from chunk summaries.
    tail5 = sp_last[-25000:]
    cls_rates = segs[-1]
    rE5 = cls_rates["E"]
    rEprev = segs[-2]["E"]
    drift = abs(rE5 - rEprev) / max(rE5, 1.0)
    wdrift = abs(snaps[-1]["w_Epre"] - snaps[-2]["w_Epre"]) / max(abs(snaps[-2]["w_Epre"]), 1e-12)
    mx = max(cls_rates[c] for c in ("E", "PV", "SST", "VIP"))
    # Synchrony guard on last chunk: steps with >50% E co-spiking.
    sync = float((tail5.mean(axis=1) > 0.5).mean())
    base = {"rE_last": rE5, "rE_prev": rEprev, "drift": drift,
            "w_drift": wdrift, "rates": cls_rates, "max_rate": mx,
            "sync": sync, "H": snaps[-1]["H"]}
    if rE5 < 1.0:
        return {"label": "COLLAPSED", **base}
    ok = (CAPTURE["rE_lo"] <= rE5 <= CAPTURE["rE_hi"]
          and drift <= CAPTURE["drift_max"] and wdrift <= CAPTURE["w_drift_max"]
          and cls_rates["PV"] >= CAPTURE["rPV_min"]
          and cls_rates["SST"] >= CAPTURE["rSST_min"]
          and mx < CAPTURE["sat_max"] and sync <= CAPTURE["sync_max"])
    return {"label": "CAPTURED" if ok else "OTHER", **base}


def prep_run(model, step_fn, rule_name, s, e_amp, pv_amp=0.0):
    """Driven preparation: uniform E schedule (+PV for IBIAS), 8 s settle."""
    n = int(model.params["emitter"].v0.shape[0])
    dtype = model.params["emitter"].v0.dtype
    tbl = model.neuron_table()
    cls = np.array([str(r["cell_type"]) for r in tbl])
    n_steps = int(B.PREP_S / (DT / 1000))
    sched = jnp.zeros((n_steps, n), dtype=dtype)
    sched = sched.at[:, cls == "E"].set(float(e_amp))
    if pv_amp:
        sched = sched.at[:, cls == "PV"].set(float(pv_amp))
    st = _fresh_state(model, rule_name, SEED)
    st_end, out = jtfne.run_continuation(step_fn, st, sched)
    jax.block_until_ready(out[0])
    sp = np.asarray(out[1], dtype=float)
    q1 = float(sp[:20000][:, cls == "E"].mean() * 10000.0)
    q2 = float(sp[20000:40000][:, cls == "E"].mean() * 10000.0)
    q4 = float(sp[60000:][:, cls == "E"].mean() * 10000.0)
    drift = abs(q4 - q2) / max(q4, 1e-9)
    vip = float(sp[60000:][:, cls == "VIP"].mean() * 10000.0)
    return {"st": st_end, "rate": q4, "drift": drift, "vip": vip,
            "settled": bool(drift <= B.PREP_SETTLE_TOL)}


def prep_grid(model, step_fn, rule_name, s, pv_amp=0.0):
    """Shared E-drive grid (one execution; P1/P3/P4 select from it).

    Optimization with identical selection semantics: the prep simulations
    depend only on (model, amps, pv_amp) -- not on the init target -- so
    one grid execution serves all E-grid inits. Deterministic in-process;
    selection (nearest settled Hz-scale rate) is immune to the ~1e-4
    cross-process floor (observed rates are 0 or 5000).
    """
    grid = []
    for a in list(B.PREP_AMPS):
        r = prep_run(model, step_fn, rule_name, s, a, pv_amp)
        grid.append({"amp": a, "rate": r["rate"], "drift": r["drift"],
                     "vip": r["vip"], "settled": r["settled"],
                     "state": r["st"]})
    return grid


def select_prep(model, step_fn, rule_name, s, target, pv_amp=0.0,
                grid=None):
    """Nearest settled rate to target from the grid (+bounded extension).

    Returns (best_or_None, scan_table). The full scan table is sealed in
    every init artifact (usable or not): prep steering evidence.
    """
    scan = []
    cands = []
    base = list(B.PREP_AMPS) if grid is None else []
    if grid is not None:
        for g in grid:
            scan.append({k: g[k] for k in ("amp", "rate", "drift", "vip", "settled")})
            if g["settled"]:
                cands.append(g)
    for amp in base + [None]:
        if amp is None:
            if cands and min(abs(c["rate"] - target) for c in cands) / target <= B.PREP_USABLE_TOL:
                break
            ext = list(B.PREP_AMPS_EXT)
        else:
            ext = [amp]
        for a in ext:
            r = prep_run(model, step_fn, rule_name, s, a, pv_amp)
            scan.append({"amp": a, "rate": r["rate"], "drift": r["drift"],
                         "vip": r["vip"], "settled": r["settled"]})
            if r["settled"]:
                cands.append({"amp": a, **{k: v for k, v in r.items() if k != "st"},
                              "state": r["st"]})
    if not cands:
        return None, scan
    best = min(cands, key=lambda c: abs(c["rate"] - target))
    if abs(best["rate"] - target) / target > B.PREP_USABLE_TOL:
        return None, scan
    return best, scan


def run_init(s, init, plant=None, egrid=None, p5grid=None):
    import json
    assert s in S_BRACKET and init in B.INITS
    if plant is None:
        model, step_fn, rule_name = build_plant(s)
    else:
        model, step_fn, rule_name = plant
    tag = scale_tag(s)
    if init in ("P0", "P7"):
        model2, dyn = construct_state(model, rule_name, s, init)
        segs, snaps, last = run_release(model2, step_fn, rule_name, dyn)
        prep = None
        scan = None
    else:
        fp = B.fp_tables(s)
        target = {"P1": fp["r"]["E"], "P3": 0.5 * fp["r"]["E"],
                  "P4": 1.5 * fp["r"]["E"],
                  "P5": fp["r"]["E"]}[init]
        pv_amp = B.PREP_IBIAS_PV_AMP if init == "P5" else 0.0
        grid = p5grid if init == "P5" else egrid
        best, scan = select_prep(model, step_fn, rule_name, s, target,
                                 pv_amp, grid=grid)
        if best is None:
            adj = {"scale": s, "init": init, "label": "UNUSABLE",
                   "detail": "no settled prep within 30% of target",
                   "prep_scan": scan}
            json.dump(adj, open(f"results/s9_init_s{tag}_{init}.json", "w"),
                      indent=2)
            print(f"s={s} {init}: UNUSABLE")
            return adj
        # Sync model weights to the prep-end carry (kernel reads carry;
        # belt+suspenders against param/carry ambiguity).
        from dataclasses import replace
        el = model.params["edge_list"]
        wv = np.asarray(best["state"].dynamic.w)
        new_el = replace(el, weight=jnp.asarray(
            wv.astype(np.float32)))
        model2 = replace(model, params={**model.params, "edge_list": new_el})
        segs, snaps, last = run_release(model2, step_fn, rule_name,
                                        best["state"].dynamic)
        prep = {k: v for k, v in best.items() if k != "state"}
    adj = adjudicate(segs, snaps, last)
    adj["scale"] = s
    adj["init"] = init
    adj["segments"] = segs
    adj["prep"] = prep
    adj["prep_scan"] = scan
    json.dump(adj, open(f"results/s9_init_s{tag}_{init}.json", "w"),
              indent=2)
    print(f"s={s} {init}: {adj['label']} rE={adj.get('rE_last', float('nan')):.2f}")
    return adj


def run_candidate(s):
    """One plant build, one shared E-grid + one P5-grid per candidate."""
    model, step_fn, rule_name = build_plant(s)
    plant = (model, step_fn, rule_name)
    egrid = prep_grid(model, step_fn, rule_name, s, pv_amp=0.0)
    p5grid = prep_grid(model, step_fn, rule_name, s,
                       pv_amp=B.PREP_IBIAS_PV_AMP)
    return [run_init(s, i, plant=plant, egrid=egrid, p5grid=p5grid)
            for i in B.INITS]


def test_s9_s50():
    run_candidate(50.0)


def test_s9_s54():
    run_candidate(54.0)


def test_s9_s57():
    run_candidate(57.0)


def test_s9_s61():
    run_candidate(61.0)


def test_s9_verdict():
    """Basin verdict: >=2 distinct finite inits captured to same regime."""
    import json
    cand = {}
    for s in (50.0, 54.0, 57.0, 61.0):
        adjs = [json.load(open(f"results/s9_init_s{scale_tag(s)}_{i}.json"))
                for i in B.INITS]
        caps = [a for a in adjs if a["label"] == "CAPTURED" and a["init"] != "P0"]
        regs = sorted(c["rE_last"] for c in caps)
        same = (len(regs) >= 2 and (max(regs) - min(regs)) / max(min(regs), 1e-9)
                <= CAPTURE["regime_tol"])
        cand[s] = {"inits": {a["init"]: (a["label"], round(a["rE_last"], 2))
                             for a in adjs},
                   "captured": regs, "same_regime": bool(same),
                   "pass": bool(same)}
    passing = [s for s, c in cand.items() if c["pass"]]
    near_collapse = [s for s, c in cand.items()
                     if c["inits"].get("P1", ("", 0))[0] == "COLLAPSED"]
    lin = {"parent": "23c56d5+ac4422e", "candidates": cand,
           "passing_subset": passing,
           "closure_transfer_failures": near_collapse,
           "selection": "none performed",
           "verdict": ("ACTIVE_BASIN_PASS" if passing
                       else "ACTIVE_BASIN_FAIL"),
           "next_authorized_action": ("STOP before REC intervention"
                                      if passing else
                                      "STOP; localize S8-closure vs native discrepancy")}
    json.dump(lin, open("results/s9_lineage.json", "w"), indent=2)
    print("S9 verdict:", lin["verdict"], passing)
    assert lin["verdict"] in ("ACTIVE_BASIN_PASS", "ACTIVE_BASIN_FAIL",
                              "ACTIVE_BASIN_UNRESOLVED")
