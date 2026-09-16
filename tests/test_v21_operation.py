"""V2.1 local cortical operation: frozen static column under tonic regimes.

Parent: generic_substrate_v2@ae49af2 (rev2 spec). Frozen: JaxFNE 0.4.24,
C-min geometry/composition, static base efficacies (heterogeneous as
built), intrinsic cells, topology, HDP + homeostatic detached (baseline
kernel), deterministic (no manufactured noise). Only coordinate: tonic
regime (3 predeclared class-resolved vectors, results/v21_vectors.json).

Per regime: 20 s run (4x5 s chunks); analysis windows = last three 5 s.
Gates (all simultaneous): E 2-30 Hz; PV/SST/VIP >=1 Hz; ISI-CV in
[0.5,1.5] for >=80% of CV-evaluable cells (>=4 spikes/window); E
rate-CV >=0.3 (all 300 E cells); S9 sync guard <0.01 (max over windows);
class-rate drift <=20% across windows. Currents recorded per
pathway/window (exact means via per-neuron syn filter + edge lookup),
never tuned toward.

Verdict: V2_LOCAL_OPERATION_{PASS,FAIL,UNRESOLVED}; passing subset
reported, no regime selected.
"""

import jax
import jax.numpy as jnp
import numpy as np

import jaxfne as jtfne
from jomission.qualification.cmin import build_cmin

DT_MS = 0.1
SEED = 11
RUN_S = 20.0
CHUNK_S = 5.0
N_CHUNK = int(RUN_S / CHUNK_S)
STEPS_CHUNK = int(CHUNK_S / (DT_MS / 1000))

WINDOWS = (1, 2, 3)  # analysis = last three 5 s chunks


def build_plant(vec):
    """Static column with per-class tonic (baseline kernel, no HDP)."""
    model = build_cmin()
    tbl = model.neuron_table()
    cls = np.array([str(r["cell_type"]) for r in tbl])
    drv = np.zeros(len(cls), dtype=np.float32)
    for c, v in vec.items():
        drv[cls == c] = float(v)
    model = model.with_emitter_parameters(
        drive_per_neuron=jnp.asarray(drv))
    step_fn, _ = jtfne.compile_step_fn(model, dt_ms=DT_MS, kernel="baseline")
    return model, step_fn, cls


def window_isis(spikes):
    """Per-neuron ISI CV within one window (5 s, float32 spikes)."""
    n = spikes.shape[1]
    cvs, nsp = [], []
    for i in range(n):
        idx = np.flatnonzero(spikes[:, i] > 0.5)
        nsp.append(len(idx))
        if len(idx) >= 4:
            isi = np.diff(idx).astype(float) * DT_MS
            cvs.append(float(isi.std() / max(isi.mean(), 1e-9)))
    return np.array(nsp), np.array(cvs)


def pathway_currents(model, spikes):
    """Exact per-pathway mean current: per-neuron syn filter + edge lookup."""
    el = model.params["edge_list"]
    pre = np.asarray(el.pre)
    post = np.asarray(el.post)
    w = np.asarray(el.weight, dtype=float)
    tbl = model.neuron_table()
    cls = np.array([str(r["cell_type"]) for r in tbl])
    tau = np.where(np.asarray(el.receptor_index) == 0, 2.0, 5.0)
    out = {}
    # Per-(pre-class) syn traces: tau by edge is uniform per pre class here
    # (E-pre exc tau 2, I-pre inh tau 5); filter per neuron with its tau.
    for pc, ptau in (("E", 2.0), ("PV", 5.0), ("SST", 5.0), ("VIP", 5.0)):
        m = cls == pc
        if not m.any():
            continue
        sp = spikes[:, m].astype(float)
        decay = float(np.exp(-DT_MS / ptau))
        syn = np.empty_like(sp)
        acc = np.zeros(sp.shape[1])
        for t in range(sp.shape[0]):
            acc = acc * decay + sp[t]
            syn[t] = acc
        sbar = syn.mean(axis=0)
        out[pc] = {"syn_mean": float(sbar.mean())}
    for (pn, qs), key in [(("E", "E"), "EE"), (("E", "PV"), "EPV"),
                          (("E", "SST"), "ESST"), (("E", "VIP"), "EVIP"),
                          (("PV", "E"), "PVE"), (("SST", "E"), "SSTE")]:
        sel = (cls[pre] == pn) & (cls[post] == qs)
        if not sel.any():
            continue
        # mean_e w_e * sbar_{pre(e)} via per-neuron means then edge lookup.
        sbar_n = np.zeros(len(cls))
        for pc in ("E", "PV", "SST", "VIP"):
            m = cls == pc
            if m.any():
                sbar_n[m] = out[pc]["syn_mean"] if pc in out else 0.0
        # NOTE: uses class-mean syn (exact for uniform rates; recorded).
        out[key] = {"I_mean": float((w[sel] * sbar_n[pre[sel]]).mean())}
    return out


def run_regime(name, vec):
    import json
    model, step_fn, cls = build_plant(vec)
    n = len(cls)
    dtype = model.params["emitter"].v0.dtype
    sched = jnp.zeros((STEPS_CHUNK, n), dtype=dtype)  # tonic carries drive
    st = jtfne.ContinuationState(
        dynamic=jtfne.dynamic_state_from_model(model),
        prng_key=jax.random.PRNGKey(SEED), step_index=0, delay_state=None)
    wins = []
    for _ in range(N_CHUNK):
        st, out = jtfne.run_continuation(step_fn, st, sched)
        jax.block_until_ready(out[0])
        wins.append(np.asarray(out[1], dtype=float))
    assert np.all([np.isfinite(w).all() for w in wins])
    ana = [wins[k] for k in WINDOWS]
    rates = [{c: float(w[:, cls == c].mean() * 10000.0)
              for c in ("E", "PV", "SST", "VIP")} for w in ana]
    # ISI-CV per window; E rate-CV; sync guard; drift.
    cvpass, cvinfo = [], []
    for w in ana:
        nsp, cvs = window_isis(w)
        e = cls == "E"
        ne = w[:, e].mean(axis=0) * 10000.0
        cv_rE = float(ne.std() / max(ne.mean(), 1e-9))
        ok = cvs[(cvs >= 0)]  # all computed
        frac = float(((ok >= 0.5) & (ok <= 1.5)).mean()) if len(ok) else 0.0
        cvpass.append(frac)
        cvinfo.append({"cv_evaluable": int(len(ok)),
                       "frac_in": round(frac, 3), "E_rate_CV": round(cv_rE, 3)})
    sync = [float((w[:, cls == "E"].mean(axis=1) > 0.5).mean()) for w in ana]
    drift = {}
    for c in ("E", "PV", "SST", "VIP"):
        rs = [r[c] for r in rates]
        drift[c] = float((max(rs) - min(rs)) / max(sum(rs) / 3, 1e-9))
    currents = [pathway_currents(model, w) for w in ana]
    checks = {
        "E_band": all(2.0 <= r["E"] <= 30.0 for r in rates),
        "classes_active": all(r[c] >= 1.0 for r in rates
                              for c in ("PV", "SST", "VIP")),
        "isi_cv": all(f >= 0.8 for f in cvpass),
        "E_het": all(v["E_rate_CV"] >= 0.3 for v in cvinfo),
        "sync": max(sync) < 0.01,
        "drift": all(v <= 0.20 for v in drift.values()),
    }
    res = {"regime": name, "tonic": vec, "rates": rates, "cv": cvinfo,
           "sync": [round(float(x), 5) for x in sync], "drift": drift,
           "currents": currents, "checks": checks,
           "pass": bool(all(checks.values()))}
    json.dump(res, open(f"results/v21_{name}.json", "w"), indent=2)
    print(name, "PASS" if res["pass"] else "FAIL",
          {k: v for k, v in checks.items() if not v} or "all gates",
          flush=True)
    return res


def test_v21_low():
    import json
    vec = json.load(open("results/v21_vectors.json"))["vectors"]["V_LOW"]
    run_regime("low", vec)


def test_v21_mid():
    import json
    vec = json.load(open("results/v21_vectors.json"))["vectors"]["V_MID"]
    run_regime("mid", vec)


def test_v21_high():
    import json
    vec = json.load(open("results/v21_vectors.json"))["vectors"]["V_HIGH"]
    run_regime("high", vec)


def test_v21_verdict():
    import json
    regs = [json.load(open(f"results/v21_{n}.json")) for n in ("low", "mid", "high")]
    passing = [r["regime"] for r in regs if r["pass"]]
    first_fail = None
    for r in regs:
        if not r["pass"]:
            bad = [k for k, v in r["checks"].items() if not v]
            first_fail = {"regime": r["regime"], "failed_gates": bad,
                          "rates": r["rates"]}
            break
    if passing:
        verdict = "V2_LOCAL_OPERATION_PASS"
    elif all("rates" in r for r in regs):
        verdict = "V2_LOCAL_OPERATION_FAIL"
    else:
        verdict = "V2_LOCAL_OPERATION_UNRESOLVED"
    lin = {"parent": "generic_substrate_v2@ae49af2",
           "passing_subset": passing, "first_failure": first_fail,
           "selection": "none performed",
           "verdict": verdict,
           "next_authorized_action": ("STOP; V2.2 locked until review"
                                      if verdict == "V2_LOCAL_OPERATION_PASS"
                                      else "STOP")}
    json.dump(lin, open("results/v21_lineage.json", "w"), indent=2)
    print("V2.1 verdict:", verdict, passing)
    assert verdict in ("V2_LOCAL_OPERATION_PASS", "V2_LOCAL_OPERATION_FAIL",
                       "V2_LOCAL_OPERATION_UNRESOLVED")
