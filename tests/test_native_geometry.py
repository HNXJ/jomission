"""N1 native trajectory dataset: boundary runs with full-state reduction.

Writes results/n_traj_s{TAG}_{lo,mid,hi}.json. Raw spikes/V discarded
after deterministic online reduction (reduction code = provenance).
"""

import jax
import jax.numpy as jnp
import numpy as np

import jaxfne as jtfne
from jomission.qualification import nativegeo as NG
from jomission.qualification.cmin import build_cmin
from jomission.qualification.hdp_rule_sel_eout import (
    ensure_registered_scale,
    scaled_params,
)

DT = NG.DT_MS
SEED = NG.SEED
N_RUN = int(NG.PREP_S / (DT / 1000))
N_CHUNK = 16  # 500 ms chunks: carry snapshots resolve aux/w (tau 20-55 ms)
STEPS_CHUNK = N_RUN // N_CHUNK


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


def run_trajectory(s, amp):
    """8 s uniform-E prep; retains spikes+V per 2 s chunk for reduction."""
    model, step_fn, rule_name = _plant(s)
    n = int(model.params["emitter"].v0.shape[0])
    dtype = model.params["emitter"].v0.dtype
    tbl = model.neuron_table()
    cls = np.array([str(r["cell_type"]) for r in tbl])
    n_steps = N_RUN
    sched = jnp.zeros((n_steps, n), dtype=dtype)
    sched = sched.at[:, cls == "E"].set(float(amp))
    dyn = jtfne.dynamic_state_from_model(
        model, hdp_params={"hdp_rule": rule_name})
    st = jtfne.ContinuationState(
        dynamic=dyn, prng_key=jax.random.PRNGKey(SEED),
        step_index=0, delay_state=None)
    sp_all, v_all, snaps = [], [], []
    for k in range(N_CHUNK):
        st, out = jtfne.run_continuation(
            step_fn, st, sched[k * STEPS_CHUNK:(k + 1) * STEPS_CHUNK])
        jax.block_until_ready(out[0])
        sp_all.append(np.asarray(out[1], dtype=np.float32))
        v_all.append(np.asarray(out[0], dtype=np.float32))
        dd = st.dynamic
        el = model.params["edge_list"]
        pre = np.asarray(el.pre)
        post = np.asarray(el.post)
        wv = np.asarray(dd.w, dtype=float)
        auxv = np.asarray(dd.aux, dtype=float)
        uv = np.asarray(dd.u, dtype=float)
        vvb = np.asarray(dd.v, dtype=float)
        per_path = {}
        for t, m in (("EE", (cls[pre] == "E") & (cls[post] == "E")),
                     ("EPV", (cls[pre] == "E") & (cls[post] == "PV")),
                     ("ESST", (cls[pre] == "E") & (cls[post] == "SST")),
                     ("EVIP", (cls[pre] == "E") & (cls[post] == "VIP")),
                     ("PVE", cls[pre] == "PV"),
                     ("SSTE", cls[pre] == "SST")):
            per_path[t] = {"w": float(wv[m].mean()),
                           "aux": float(auxv[m].mean())}
        snaps.append({"t_s": round((k + 1) * 0.5, 2), "per_path": per_path,
                      "H": float(np.asarray(dd.H, dtype=float).mean()),
                      "u": {c: float(uv[cls == c].mean())
                            for c in ("E", "PV", "SST", "VIP")},
                      "Vbar": {c: float(vvb[cls == c].mean())
                               for c in ("E", "PV", "SST", "VIP")}})
    sp = np.concatenate(sp_all, axis=0)
    vv = np.concatenate(v_all, axis=0)
    del sp_all, v_all
    return model, cls, sp, vv, snaps


def reduce_run(s, amp, name):
    import json
    model, cls, sp, vv, snaps = run_trajectory(s, amp)
    n2 = NG.BIN_2MS
    nb = sp.shape[0] // n2
    rates, sync = {}, []
    for c in ("E", "PV", "SST", "VIP"):
        m = cls == c
        binned = sp[:, m].reshape(nb, n2, -1).mean(axis=(1, 2)) * 10000.0
        rates[c] = [round(float(x), 3) for x in binned]
    ebin = sp[:, cls == "E"].reshape(nb, n2, -1).mean(axis=2)
    sync = [round(float(x), 4) for x in ebin.mean(axis=1)]
    n1 = NG.BIN_10MS
    nb1 = sp.shape[0] // n1
    means = {}
    el = model.params["edge_list"]
    pre = np.asarray(el.pre)
    post = np.asarray(el.post)
    for c in ("E", "PV", "SST", "VIP"):
        m = cls == c
        means[f"V_{c}"] = [round(float(x), 3) for x in
                           vv[:, m].reshape(nb1, n1, -1).mean(axis=(1, 2))]
    # Constructed pathway currents at 10 ms bins (exact kernel on binned rates).
    r10 = {}
    for c in ("E", "PV", "SST"):
        m = cls == c
        r10[c] = sp[:, m].reshape(nb1, n1, -1).mean(axis=(1, 2)) * 10000.0 / 1000.0
    Ich = {}
    for t, pc, tc, tau in (("EE", "E", "E", 2.0), ("EPV", "E", "PV", 2.0),
                           ("ESST", "E", "SST", 2.0), ("PVE", "PV", "E", 5.0),
                           ("SSTE", "SST", "E", 5.0)):
        r = r10[pc]
        dt = 0.01
        decay = float(np.exp(-dt / (tau / 1000.0)))
        syn = np.empty_like(r)
        acc = 0.0
        for k in range(len(r)):
            acc = acc * decay + r[k] * dt
            syn[k] = acc
        K = {"EE": 299.0, "EPV": 300.0, "ESST": 300.0, "PVE": 40.0,
             "SSTE": 32.0}[t]
        w = np.array([sn["per_path"][t]["w"] for sn in snaps])
        w2 = np.repeat(w, nb1 // len(w))
        Ich[t] = [round(float(x), 5) for x in K * w2 * syn]
    er = np.array(rates["E"])
    hit = np.flatnonzero(er > NG.COMMIT_R)
    if len(hit):
        tc = int(hit[0]) * n2
    else:
        tc = None
    if tc is None:
        lo, hi = int(5.0 / (DT / 1000)), int(7.0 / (DT / 1000))
    else:
        lo, hi = max(tc - int(NG.WIN_S / (DT / 1000)), 0), min(
            tc + int(NG.WIN_S / (DT / 1000)), sp.shape[0])
    ex = {}
    for c in ("E", "PV", "SST"):
        i = int(np.flatnonzero(cls == c)[0])
        ex[c] = [round(float(x), 2) for x in vv[lo:hi, i]]
    out = {"scale": s, "amp": amp, "name": name,
           "t_commit_s": (tc * DT / 1000.0) if tc is not None else None,
           "outcome": ("SYNC" if (tc is not None) else
                       ("SILENT" if er[-500:].mean() < 1.0 else "OTHER")),
           "rates_2ms": rates, "sync_2ms": sync, "means_10ms": means,
           "Ixy_10ms": Ich, "V_win": ex,
           "win_s": [lo * DT / 1000.0, hi * DT / 1000.0],
           "snaps_500ms": snaps}
    json.dump(out, open(f"results/n_traj_s{NG.scale_tag(s)}_{name}.json", "w"))
    print(f"s={s} {name}(amp={amp}): {out['outcome']} "
          f"t_commit={out['t_commit_s']}", flush=True)
    return out


def test_n1_dataset():
    for s in NG.CANDIDATES:
        lo, hi, mid = NG.flip_amps(s)
        for name, amp in (("lo", lo), ("mid", mid), ("hi", hi)):
            reduce_run(s, amp, name)
    assert True
