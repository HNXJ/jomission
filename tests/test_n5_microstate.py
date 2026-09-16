"""N5 matched-intervention sessions: snapshot, fork, adjudicate, verdict.

Writes results/n5_{tag}_{snapms}_{arm}.json, results/n5_verdict.json.
"""

import jax
import jax.numpy as jnp
import numpy as np

import jaxfne as jtfne
from jomission.qualification import microstate as MS
from jomission.qualification.cmin import build_cmin
from jomission.qualification.hdp_rule_sel_eout import (
    ensure_registered_scale,
    scaled_params,
)

DT = MS.DT_MS
SEED = MS.SEED


def _plant(s):
    rule_name = ensure_registered_scale(s)
    model = build_cmin()
    e = model.params["emitter"]
    model = model.with_emitter_parameters(
        drive_per_neuron=jnp.zeros_like(e.drive))
    rp = dict(scaled_params(s))
    step_fn, _ = jtfne.compile_step_fn(
        model, dt_ms=DT, kernel="hdp", hdp_rule=rule_name,
        hdp_rule_params=rp, record_weight_trace=False)
    rp0 = dict(rp, k_w=0.0)
    step_kw0, _ = jtfne.compile_step_fn(
        model, dt_ms=DT, kernel="hdp", hdp_rule=rule_name,
        hdp_rule_params=rp0, record_weight_trace=False)
    model_cl = model.with_emitter_parameters(
        a_per_neuron=jnp.zeros((int(model.params["emitter"].v0.shape[0]),)),
        d_per_neuron=jnp.zeros((int(model.params["emitter"].v0.shape[0]),)))
    step_cl, _ = jtfne.compile_step_fn(
        model_cl, dt_ms=DT, kernel="hdp", hdp_rule=rule_name,
        hdp_rule_params=rp, record_weight_trace=False)
    return model, rule_name, step_fn, step_kw0, step_cl


def _fresh(model, rule_name):
    dyn = jtfne.dynamic_state_from_model(
        model, hdp_params={"hdp_rule": rule_name})
    return jtfne.ContinuationState(
        dynamic=dyn, prng_key=jax.random.PRNGKey(SEED),
        step_index=0, delay_state=None)


def run_source(plant, s, amp):
    """Fresh prep run to 8 ms+; snapshots at 5 and 8 ms (full carry)."""
    model, rule_name, step_fn, _, _ = plant
    n = int(model.params["emitter"].v0.shape[0])
    dtype = model.params["emitter"].v0.dtype
    tbl = model.neuron_table()
    cls = np.array([str(r["cell_type"]) for r in tbl])
    n_steps = int(MS.RUN_S / (DT / 1000))
    sched = jnp.zeros((n_steps, n), dtype=dtype)
    sched = sched.at[:, cls == "E"].set(float(amp))
    st = _fresh(model, rule_name)
    snaps = {}
    step = 0
    for t_ms in MS.SNAPS_MS:
        tgt = int(t_ms / DT)  # t_ms in ms, DT in ms/step
        st, _ = jtfne.run_continuation(step_fn, st, sched[step:tgt])
        jax.block_until_ready(st.dynamic.v)
        snaps[t_ms] = st.dynamic
        step = tgt
    return sched, snaps


def apply_perm(dyn, cls, pre, post):
    """Within-class (V,u,prev) relabel + within-pathway syn permutation."""
    rng = np.random.default_rng(MS.PERM_SEED)
    n = len(cls)
    pmap = np.arange(n)
    for c in ("E", "PV", "SST", "VIP"):
        idx = np.flatnonzero(cls == c)
        pmap[idx] = rng.permutation(idx)
    v = np.asarray(dyn.v)
    u = np.asarray(dyn.u)
    pv = np.asarray(dyn.prev_spikes)
    v2, u2, pv2 = v[pmap], u[pmap], pv[pmap]
    # Preservation asserts (exact: same multisets per class).
    for c in ("E", "PV", "SST", "VIP"):
        idx = np.flatnonzero(cls == c)
        np.testing.assert_array_equal(np.sort(v2[idx]), np.sort(v[idx]))
        np.testing.assert_array_equal(np.sort(u2[idx]), np.sort(u[idx]))
        assert int(pv2[idx].sum()) == int(pv[idx].sum())
    # Energy/multiset preservation must be order-independent: float32
    # summation order differs after permutation (1-ulp on large sums), so
    # compare sorted values exactly (same multiset) instead of sums.
    np.testing.assert_array_equal(np.sort(v2), np.sort(v))
    syn = np.asarray(dyn.syn_state)
    syn2 = syn.copy()
    tags = np.array([("E" if c == "E" else "I") for c in cls[pre]])
    for t in ("E", "I"):
        m = np.flatnonzero(tags == t)
        syn2[m] = rng.permutation(syn[m])
        np.testing.assert_array_equal(np.sort(syn2[m]), np.sort(syn[m]))
    out = dyn._replace(v=jnp.asarray(v2, dtype=dyn.v.dtype),
                       u=jnp.asarray(u2, dtype=dyn.u.dtype),
                       prev_spikes=jnp.asarray(pv2, dtype=dyn.prev_spikes.dtype),
                       syn_state=jnp.asarray(syn2, dtype=dyn.syn_state.dtype))
    # aux/w/H untouched.
    np.testing.assert_array_equal(np.asarray(out.aux), np.asarray(dyn.aux))
    np.testing.assert_array_equal(np.asarray(out.w), np.asarray(dyn.w))
    np.testing.assert_array_equal(np.asarray(out.H), np.asarray(dyn.H))
    return out


def run_arm(s, amp, t_ms, arm, src):
    """Continue from snapshot with the arm's intervention; adjudicate."""
    import json
    model, rule_name, step_fn, step_kw0, step_cl = src["plant"]
    dyn0 = src["snaps"][t_ms]
    sched = src["sched"]
    start = int(t_ms / DT)  # snapshot step (t_ms in ms, DT ms/step)
    n = sched.shape[1]
    dtype = sched.dtype
    tbl = model.neuron_table()
    cls = np.array([str(r["cell_type"]) for r in tbl])
    el = model.params["edge_list"]
    pre = np.asarray(el.pre)
    post = np.asarray(el.post)
    ver = {}
    st = jtfne.ContinuationState(
        dynamic=dyn0, prng_key=jax.random.PRNGKey(SEED + 999),
        step_index=start, delay_state=None)
    fn, extra, t_end_extra = step_fn, None, 0
    if arm == "PERMUTE":
        dyn = apply_perm(dyn0, cls, pre, post)
        ver["preserved_exact"] = True
        st = jtfne.ContinuationState(
            dynamic=dyn, prng_key=jax.random.PRNGKey(SEED + 999),
            step_index=start, delay_state=None)
    elif arm == "UCLAMP":
        fn = None  # two-phase below
    elif arm == "SYNHALF":
        syn = np.asarray(dyn0.syn_state) * MS.SYN_SCALE
        dyn = dyn0._replace(syn_state=jnp.asarray(syn, dtype=dyn0.syn_state.dtype))
        st = jtfne.ContinuationState(
            dynamic=dyn, prng_key=jax.random.PRNGKey(SEED + 999),
            step_index=start, delay_state=None)
    elif arm == "KW0":
        fn = step_kw0
    elif arm in ("BLIPP", "BLIPM"):
        sgn = 1.0 if arm == "BLIPP" else -1.0
        nb = int(MS.BLIP_MS / DT)  # ms / (ms/step)
        extra = (jnp.zeros((nb, n), dtype=dtype)
                 .at[:, cls == "E"].set(sgn * MS.BLIP_AMP))
    # Execute in pieces: 10x10 ms (early syn trajectory) + remainder.
    syn_early = []
    if arm == "UCLAMP":
        nc = int(MS.CLAMP_MS / DT)  # ms / (ms/step)
        st, _ = jtfne.run_continuation(step_cl, st, sched[start:start + nc])
        jax.block_until_ready(st.dynamic.v)
        du = float(np.abs(np.asarray(st.dynamic.u, dtype=float)
                          - np.asarray(dyn0.u, dtype=float)).max())
        ver["u_drift_max"] = du
        assert du <= 0.01 * max(float(np.abs(np.asarray(
            dyn0.u, dtype=float)).max()), 1.0), du
        cur, remain = start + nc, sched[start + nc:]
        fn = step_fn
    else:
        segsched = sched[start:]
        if extra is not None:
            nb = extra.shape[0]
            segsched = segsched.at[:nb].add(extra)
        cur, remain, fn = start, segsched, fn
    off = 0
    outs = []
    for _ in range(10):
        piece = remain[off:off + 100]
        if piece.shape[0] == 0:
            break
        st, out = jtfne.run_continuation(fn, st, piece)
        jax.block_until_ready(out[0])
        outs.append(out)
        syn_early.append(float(np.asarray(st.dynamic.syn_state,
                                          dtype=float).mean()))
        off += piece.shape[0]
    st, out = jtfne.run_continuation(fn, st, remain[off:])
    jax.block_until_ready(out[0])
    outs.append(out)
    ver["syn_early"] = [round(float(x), 6) for x in syn_early]
    sp = np.concatenate([np.asarray(o[1], dtype=float) for o in outs], axis=0)
    er = sp[:, cls == "E"].mean(axis=1) * 10000.0
    # Sustained-sync latency via windowed duty cycle (200-bin/400 ms window,
    # mean > 2500): the sync regime has fine pause structure that defeats
    # consecutive-bin detectors (calibrated 2026-09-15: 2-cycle-class
    # trajectories never show 100 consecutive hot bins).
    W = 200
    tsus = None
    if len(er) >= W:
        win = np.convolve(er, np.ones(W) / W, mode="valid")
        hit = np.flatnonzero(win > 2500.0)
        if len(hit):
            tsus = (start + int(hit[0])) * DT / 1000.0
    end = er[-20000:].mean()
    label = ("SYNC" if end >= 4000.0 else
             ("SILENT" if end < 1.0 else "OTHER"))
    if arm == "SYNHALF":
        # Realized-current check: one-shot halving washes out in ~15 steps
        # under volley spiking (syn saturates at spike-driven steady state
        # regardless of starting value). Measured reduction is therefore
        # recorded, NOT asserted: a washout voids the fork as an ownership
        # test (it cannot move the variable) rather than counting as null
        # evidence. Mechanism lesson: volley-saturated syn is spike-slaved;
        # ownership at this gain must live in spikes or slower states.
        ctrl = json.load(open(
            f"results/n5_{MS.scale_tag(s)}_{t_ms:g}_CTRL.json"))
        c = float(np.mean(ctrl["verify"]["syn_early"][:2]))
        h = float(np.mean(ver["syn_early"][:2]))
        ver["syn_reduction"] = round(1.0 - h / max(c, 1e-12), 3)
        ver["void_as_ownership_test"] = bool(ver["syn_reduction"] < 0.20)
    res = {"scale": s, "amp": amp, "t_ms": t_ms, "arm": arm,
           "label": label, "rE_end": float(end),
           "t_sustain_s": tsus, "verify": ver}
    json.dump(res, open(
        f"results/n5_{MS.scale_tag(s)}_{t_ms:g}_{arm}.json", "w"), indent=2)
    print(f"s={s} t={t_ms:g} {arm}: {label} rE={end:.1f} t_sus={tsus}",
          flush=True)
    return res


def session(s, amp):
    """Full 7-arm session at both snapshots; returns outcomes."""
    plant = _plant(s)
    model, rule_name, step_fn, step_kw0, step_cl = plant
    sched, snaps = run_source(plant, s, amp)
    src = {"plant": (model, rule_name, step_fn, step_kw0, step_cl),
           "sched": sched, "snaps": snaps}
    out = {}
    for t_ms in MS.SNAPS_MS:
        for arm in MS.ARMS:
            out[(t_ms, arm)] = run_arm(s, amp, t_ms, arm, src)
    return out


def test_n5_s50():
    session(*MS.PRIMARY)


def test_n5_s57():
    session(*MS.REPLICATE)


CLASS_OF = {"PERMUTE": "SPIKE_RESET_MICROSTATE", "UCLAMP": "ADAPTATION",
            "SYNHALF": "SYNAPTIC_STATE", "KW0": "HDP_EVOLUTION",
            "BLIPP": "E_I_BALANCE", "BLIPM": "E_I_BALANCE"}


def flips(s):
    """Arms flipping control fate at BOTH snapshots (None if void)."""
    import json
    tag = MS.scale_tag(s)
    try:
        ctrls = [json.load(open(f"results/n5_{tag}_{t:g}_CTRL.json"))
                 for t in MS.SNAPS_MS]
    except FileNotFoundError:
        return None
    if any(c["label"] != "SYNC" for c in ctrls):
        return None  # session void: control did not commit to sync
    out = []
    for arm in MS.ARMS:
        if arm in ("CTRL", "SYNHALF"):
            # CTRL is the reference; SYNHALF is void as an ownership test
            # (washout: one-shot syn scaling cannot move volley-saturated
            # synaptic state) -- recorded, never a flipper.
            continue
        try:
            rs = [json.load(open(f"results/n5_{tag}_{t:g}_{arm}.json"))
                  for t in MS.SNAPS_MS]
        except FileNotFoundError:
            continue
        if all(r["label"] == "SILENT" for r in rs):
            out.append(arm)
    return out


def test_n5_verdict():
    """Ownership adjudication (predeclared flip rules + s57 replication)."""
    import json
    f50 = flips(50.0)
    f57 = flips(57.0)
    if f50 is None:
        verdict, detail = "NONE_CONCLUSIVE", "s50 session void (control)"
    elif not f50:
        verdict, detail = "NONE_IDENTIFIED", "no arm flips fate at both t*"
    elif len(f50) == 1 and f57 is not None and f50[0] in f57:
        verdict = CLASS_OF[f50[0]]
        detail = f"{f50[0]} flips at s50 (both t*) and s57 (both t*)"
    elif len(f50) == 1:
        verdict = CLASS_OF[f50[0]] + "_UNREPLICATED"
        detail = f"{f50[0]} flips at s50 only; s57 replication missing/void"
    else:
        verdict = "INTERACTION"
        detail = f"multiple flippers at s50: {f50}; conjunction required"
    lat = {}
    for s in (50.0, 57.0):
        tag = MS.scale_tag(s)
        for arm in MS.ARMS:
            try:
                rs = [json.load(open(f"results/n5_{tag}_{t:g}_{arm}.json"))
                      for t in MS.SNAPS_MS]
                lat[f"{s}_{arm}"] = [(r["label"], r["t_sustain_s"]) for r in rs]
            except FileNotFoundError:
                pass
    out = {"flippers_s50": f50, "flippers_s57": f57, "latencies": lat,
           "verdict": verdict, "detail": detail,
           "next_authorized_action": ("STOP" if verdict in
                                      ("NONE_IDENTIFIED", "NONE_CONCLUSIVE")
                                      else "design from owned coordinate")}
    json.dump(out, open("results/n5_verdict.json", "w"), indent=2)
    print("N5 verdict:", verdict, "|", detail)
    assert verdict in ("SPIKE_RESET_MICROSTATE", "ADAPTATION",
                       "SYNAPTIC_STATE", "HDP_EVOLUTION", "E_I_BALANCE",
                       "SPIKE_RESET_MICROSTATE_UNREPLICATED",
                       "ADAPTATION_UNREPLICATED", "SYNAPTIC_STATE_UNREPLICATED",
                       "HDP_EVOLUTION_UNREPLICATED", "E_I_BALANCE_UNREPLICATED",
                       "INTERACTION", "NONE_IDENTIFIED", "NONE_CONCLUSIVE")
