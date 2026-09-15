"""B0/B1 saved-trajectory visitation + B2 bisection/ladder + B3 verdict.

Writes results/b_min_d.json (B1), results/b_bisect_s{TAG}.json,
results/b_ladder_s{TAG}_{T}.json, results/b_lineage.json.
"""

import jax
import jax.numpy as jnp
import numpy as np

import jaxfne as jtfne
from jomission.qualification import basin as B
from jomission.qualification import visitation as V
from jomission.qualification.cmin import build_cmin
from jomission.qualification.hdp_rule_sel_eout import (
    ensure_registered_scale,
    scaled_params,
    scale_tag,
)

DT = 0.1
SEED = 11
CHUNK_1S = int(1.0 / (DT / 1000))


def _plant(s):
    rule_name = ensure_registered_scale(s)
    model = build_cmin()
    e = model.params["emitter"]
    model = model.with_emitter_parameters(
        drive_per_neuron=jnp.zeros_like(e.drive))
    step_fn, _ = jtfne.compile_step_fn(
        model, dt_ms=DT, kernel="hdp", hdp_rule=rule_name,
        hdp_rule_params=dict(scaled_params(s)), record_weight_trace=False)
    return model, step_fn, rule_name


def _fresh(model, rule_name):
    dyn = jtfne.dynamic_state_from_model(
        model, hdp_params={"hdp_rule": rule_name})
    return jtfne.ContinuationState(
        dynamic=dyn, prng_key=jax.random.PRNGKey(SEED),
        step_index=0, delay_state=None)


def _summarize(model, sp, dd):
    tbl = model.neuron_table()
    cls = np.array([str(r["cell_type"]) for r in tbl])
    el = model.params["edge_list"]
    pre = np.asarray(el.pre)
    wv = np.asarray(dd.w, dtype=float)
    return {"rE": float(sp[:, cls == "E"].mean() * 10000.0),
            "rPV": float(sp[:, cls == "PV"].mean() * 10000.0),
            "rSST": float(sp[:, cls == "SST"].mean() * 10000.0),
            "wE": float(wv[cls[pre] == "E"].mean()),
            "aux": float(np.asarray(dd.aux, dtype=float).mean()),
            "H": float(np.asarray(dd.H, dtype=float).mean())}


def test_b0_b1_saved():
    """B0/B1: min-d over sealed S9 release chunks + prep endpoints."""
    import json
    out = {}
    for s in V.CANDIDATES:
        tag = scale_tag(s)
        ref = V.fp_ref(s)
        rec = {"releases": {}, "prep_endpoints": []}
        for i in ("P0", "P7"):
            d = json.load(open(f"results/s9_init_s{tag}_{i}.json"))
            ds = []
            for k, seg in enumerate(d["segments"]):
                obs = {"rE": seg["E"], "rPV": seg["PV"], "rSST": seg["SST"]}
                ds.append(V.dist(obs, ref))
            rec["releases"][i] = {"min_d": min(ds), "chunks": ds,
                                  "label": d["label"]}
        for i in ("P1", "P3", "P4", "P5"):
            d = json.load(open(f"results/s9_init_s{tag}_{i}.json"))
            for r in d["prep_scan"]:
                obs = {"rE": r["rate"]}
                rec["prep_endpoints"].append(
                    {"init": i, "amp": r["amp"],
                     "d_rate": V.dist(obs, ref)})
        # Approach detection uses full-state distances only: rate-only
        # endpoint norms are single-term artifacts (a silent endpoint scores
        # |0-rE*|/40 < NEAR_D without approaching anything). Endpoints stand
        # as jump evidence, not visitation evidence.
        rec["min_d_saved"] = min(v["min_d"] for v in rec["releases"].values())
        out[str(s)] = rec
    json.dump(out, open("results/b_min_d.json", "w"), indent=2)
    print("B1 min-d:", {k: round(v["min_d_saved"], 3) for k, v in out.items()})
    assert isinstance(out, dict)


def run_prep_tracked(model, step_fn, rule_name, s, amp, pv_amp=0.0, dur_s=8.0):
    """Prep with 1 s-chunk coarse trajectory (rates + carry means)."""
    n = int(model.params["emitter"].v0.shape[0])
    dtype = model.params["emitter"].v0.dtype
    tbl = model.neuron_table()
    cls = np.array([str(r["cell_type"]) for r in tbl])
    n_steps = int(dur_s / (DT / 1000))
    sched = jnp.zeros((n_steps, n), dtype=dtype)
    sched = sched.at[:, cls == "E"].set(float(amp))
    if pv_amp:
        sched = sched.at[:, cls == "PV"].set(float(pv_amp))
    st = _fresh(model, rule_name)
    traj = []
    for k in range(int(n_steps / CHUNK_1S)):
        st, out = jtfne.run_continuation(
            step_fn, st, sched[k * CHUNK_1S:(k + 1) * CHUNK_1S])
        jax.block_until_ready(out[0])
        traj.append(_summarize(model, np.asarray(out[1], dtype=float),
                               st.dynamic))
    return traj, st


def test_b2_bisect_ladder():
    """B2: L1 bisection amps + one L2 midpoint + duration ladder."""
    import json
    for s in V.CANDIDATES:
        tag = scale_tag(s)
        model, step_fn, rule_name = _plant(s)
        ref = V.fp_ref(s, model=model)
        # L1: reuse sealed {2.0, 4.0} endpoints, run {2.5, 3.0, 3.5}.
        sealed = json.load(open(f"results/s9_init_s{tag}_P1.json"))["prep_scan"]
        grid = {r["amp"]: r["rate"] for r in sealed}
        trajs, ends = {}, {}
        for a in V.L1_AMPS:
            traj, st = run_prep_tracked(model, step_fn, rule_name, s, a)
            trajs[str(a)] = traj
            ends[a] = float(traj[-1]["rE"])
            grid[a] = ends[a]
        lo = max([a for a, r in grid.items() if r < 1.0], default=None)
        hi = min([a for a, r in grid.items() if r >= 1.0], default=None)
        mid = None
        if lo is not None and hi is not None:
            mid = (lo + hi) / 2.0
            traj, st = run_prep_tracked(model, step_fn, rule_name, s, mid)
            trajs[str(mid)] = traj
            ends[mid] = float(traj[-1]["rE"])
        # Linger: min-d and fraction of 1 s-chunks with d < NEAR_D.
        ds = []
        for a, traj in trajs.items():
            for t in traj:
                ds.append(V.dist(t, ref))
        linger = float(np.mean([d < V.NEAR_D for d in ds])) if ds else 0.0
        out = {"scale": s, "grid": grid, "flip": [lo, hi, mid],
               "min_d_bisect": min(ds) if ds else None, "linger": linger,
               "n_chunks": len(ds)}
        json.dump(out, open(f"results/b_bisect_s{tag}.json", "w"), indent=2)
        print(f"s={s} flip={out['flip']} min_d={out['min_d_bisect']:.3f} "
              f"linger={linger:.3f}", flush=True)
        # Duration ladder at first-sync amp.
        sync_amps = sorted(a for a, r in grid.items() if r >= 1.0)
        assert sync_amps, "no sync amp found"
        a_hi = sync_amps[0]
        for T in V.LADDER_T:
            traj, st = run_prep_tracked(model, step_fn, rule_name, s, a_hi,
                                        dur_s=T)
            n = int(model.params["emitter"].v0.shape[0])
            dtype = model.params["emitter"].v0.dtype
            segs = []
            for _ in range(int(B.RELEASE_S / B.CHUNK_S)):
                st, out = jtfne.run_continuation(
                    step_fn, st, jnp.zeros((int(B.CHUNK_S / (DT / 1000)), n),
                                           dtype=dtype))
                jax.block_until_ready(out[0])
                segs.append(_summarize(
                    model, np.asarray(out[1], dtype=float), st.dynamic))
            rE5 = segs[-1]["rE"]
            rEprev = segs[-2]["rE"]
            drift = abs(rE5 - rEprev) / max(rE5, 1.0)
            label = ("COLLAPSED" if rE5 < 1.0
                     else "RELEASED_ACTIVE" if rE5 >= B.CAPTURE["rE_lo"] and
                     drift <= B.CAPTURE["drift_max"] and
                     segs[-1]["rPV"] >= B.CAPTURE["rPV_min"] and
                     segs[-1]["rSST"] >= B.CAPTURE["rSST_min"] and
                     max(segs[-1][c] for c in ("rE", "rPV", "rSST")) <
                     B.CAPTURE["sat_max"] else "OTHER")
            ds2 = [V.dist(t, ref) for t in traj + segs]
            lad = {"scale": s, "amp": a_hi, "T": T, "label": label,
                   "rE_release": rE5, "drift": drift,
                   "min_d": min(ds2), "rates": segs[-1]}
            json.dump(lad, open(f"results/b_ladder_s{tag}_{T}.json", "w"),
                      indent=2)
            print(f"  T={T}: {label} rE={rE5:.2f} min_d={min(ds2):.3f}",
                  flush=True)
    assert True


def test_b3_verdict():
    """B3: lineage-global verdict from approach/capture evidence."""
    import json
    per, approach, captured = {}, False, []
    for s in V.CANDIDATES:
        tag = scale_tag(s)
        bi = json.load(open(f"results/b_bisect_s{tag}.json"))
        lads = [json.load(open(f"results/b_ladder_s{tag}_{T}.json"))
                for T in V.LADDER_T]
        caps = [L["T"] for L in lads if L["label"] == "RELEASED_ACTIVE"]
        mind = min([bi["min_d_bisect"]] + [L["min_d"] for L in lads])
        per[str(s)] = {"min_d": mind, "linger": bi["linger"],
                       "captured_T": caps,
                       "labels": [L["label"] for L in lads]}
        if caps or mind < V.NEAR_D:
            approach = True
        captured += [(s, T) for T in caps]
    if captured or approach:
        verdict = "BASIN_TINY_OR_INACCESSIBLE"
        detail = ("approach/capture evidence: captured=%s; min-d per "
                  "candidate=%s" % (captured, {k: round(v["min_d"], 3)
                                               for k, v in per.items()}))
    elif all(v["min_d"] >= V.NEAR_D for v in per.values()):
        verdict = "COARSE_FIXED_POINT_OFF_NATIVE_MANIFOLD"
        detail = ("all boundary trajectories bypass the FP neighborhood "
                  "(min-d>=0.25) with direct silence<->sync jumps")
    else:
        verdict = "UNRESOLVED"
        detail = "observables cannot make the distinction"
    lin = {"parent": "S9 ACTIVE_BASIN_FAIL", "per_candidate": per,
           "verdict": verdict, "detail": detail,
           "next_authorized_action":
               ("basin enlargement design" if verdict == "BASIN_TINY_OR_INACCESSIBLE"
                else "reconstruct operating model from native geometry" if verdict ==
                "COARSE_FIXED_POINT_OFF_NATIVE_MANIFOLD" else "STOP")}
    json.dump(lin, open("results/b_lineage.json", "w"), indent=2)
    print("B3 verdict:", verdict, "|", detail)
    assert verdict in ("BASIN_TINY_OR_INACCESSIBLE",
                       "COARSE_FIXED_POINT_OFF_NATIVE_MANIFOLD", "UNRESOLVED")
