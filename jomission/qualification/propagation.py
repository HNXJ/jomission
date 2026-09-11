"""Inter-area propagation battery (Gen-2 independent qualification).

Frozen 4-area plant (V1/V4/FEF/PFC, tonic-supported, HDP OFF). Ordinary
perturbation transmission measured area by area; never inferred from
connectivity. FF and FB separately. No omission/exposure/HDP/recurrence
changes. Endpoint: FF_FUNCTIONAL | FB_FUNCTIONAL | STRUCTURAL_ONLY | MIXED.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import jaxfne as jtfne

FF_PATH = (("V1", "V4"), ("V4", "FEF"), ("FEF", "PFC"))
FB_PATH = (("PFC", "FEF"), ("FEF", "V4"), ("V4", "V1"))


def area_index(model):
    """area -> neuron indices; layer/cell lookup tables."""
    tbl = model.neuron_table()
    areas = np.array([str(r["area"]) for r in tbl])
    return ({a: np.flatnonzero(areas == a) for a in ("V1", "V4", "FEF", "PFC")},
            tbl)


def pulse_run(model, target_idx, amp=8.0, pre_ms=1000.0, pulse_ms=50.0,
              post_ms=1000.0, dt_ms=0.1, seed=0):
    """Tonic baseline, brief pulse on targets, record. Returns spikes + info."""
    from jomission.qualification.cmin import initial_state

    nN = int(model.params["emitter"].n_neurons)
    dtype = model.params["emitter"].v0.dtype
    step_fn, _ = jtfne.compile_step_fn(model, dt_ms=float(dt_ms), kernel="baseline",
                                       record_weight_trace=False)
    state = initial_state(model, seed)
    n_pre = int(round(pre_ms / dt_ms))
    n_p = int(round(pulse_ms / dt_ms))
    n_post = int(round(post_ms / dt_ms))
    base = np.zeros(nN)
    kick = np.zeros(nN)
    kick[np.asarray(target_idx)] = float(amp)
    drive = jnp.asarray(np.concatenate([np.tile(base, (n_pre, 1)),
                                        np.tile(kick, (n_p, 1)),
                                        np.tile(base, (n_post, 1))]),
                        dtype=dtype)
    state, out = jtfne.run_continuation(step_fn, state, drive)
    jax.block_until_ready(out[0])
    return np.asarray(out[1], dtype=float), {"n_pre": n_pre, "n_p": n_p,
                                             "dt_ms": float(dt_ms)}


def window_rates(spikes, idx, windows, dt_ms=0.1):
    """Mean Hz per window (list of (start_ms, end_ms) post-onset offsets)."""
    out = []
    for (a, b) in windows:
        i0, i1 = int(round(a / dt_ms)), int(round(b / dt_ms))
        seg = spikes[i0:i1][:, np.asarray(idx)]
        out.append(float(seg.mean() * (1000.0 / dt_ms)) if i1 > i0 else 0.0)
    return out


def cut_pathway(model, src, tgt):
    """Zero all src->tgt directed edges. Returns (model, n_cut)."""
    from dataclasses import replace

    from jaxfne.emitters import EdgeList

    tbl = model.neuron_table()
    areas = np.array([str(r["area"]) for r in tbl])
    el = model.params["edge_list"]
    pre = np.asarray(el.pre, dtype=np.int64)
    post = np.asarray(el.post, dtype=np.int64)
    w = np.asarray(el.weight, dtype=float)
    cut = (areas[pre] == str(src)) & (areas[post] == str(tgt))
    w2 = w.copy()
    w2[cut] = 0.0
    kwargs = dict(pre=el.pre, post=el.post,
                  weight=jnp.asarray(w2, dtype=el.weight.dtype),
                  receptor_index=el.receptor_index, tau_ms=el.tau_ms,
                  source_calibration_status=el.source_calibration_status)
    if getattr(el, "delay_steps", None) is not None:
        kwargs["delay_steps"] = el.delay_steps
    return replace(model, params={**model.params, "edge_list": EdgeList(**kwargs)}), int(cut.sum())


def pathway_response(model, src_area, tgt_area, drive_layer="L4", drive_class="E",
                     amp=8.0, seed=0):
    """Stimulate src layer/class; measure tgt response + matched cut effect.

    Returns dict with baseline/pulse/post rates per area, response deltas,
    cut counts, and post-cut downstream response.
    """
    areas, tbl = area_index(model)
    cls = np.array([str(r["cell_type"]) for r in tbl])
    lyr = np.array([str(r["layer"]) for r in tbl])
    tgt_idx = np.flatnonzero((np.array([str(r["area"]) for r in tbl]) == src_area)
                             & (lyr == drive_layer) & (cls == drive_class))
    sp, info = pulse_run(model, tgt_idx, amp=amp, seed=seed)
    n_pre, dt = info["n_pre"], info["dt_ms"]
    base = {a: float(sp[n_pre - 5000:n_pre][:, areas[a]].mean() * (1000.0 / dt))
            for a in areas}
    resp = {a: float(sp[n_pre:n_pre + 5000][:, areas[a]].mean() * (1000.0 / dt)) - base[a]
            for a in areas}
    mc, ncut = cut_pathway(model, src_area, tgt_area)
    spc, _ = pulse_run(mc, tgt_idx, amp=amp, seed=seed)
    respc = {a: float(spc[n_pre:n_pre + 5000][:, areas[a]].mean() * (1000.0 / dt)) - base[a]
             for a in areas}
    return {"src": src_area, "tgt": tgt_area, "base": base, "delta": resp,
            "delta_cut": respc, "n_cut": ncut, "n_driven": int(len(tgt_idx))}


def scale_pathway(model, src, tgt, mult):
    """Multiply src->tgt edge weights (bounded inter-area efficacy repair).

    Connectivity, targeting, and all else fixed. Returns (model, n_scaled).
    """
    from dataclasses import replace

    from jaxfne.emitters import EdgeList

    tbl = model.neuron_table()
    areas = np.array([str(r["area"]) for r in tbl])
    el = model.params["edge_list"]
    pre = np.asarray(el.pre, dtype=np.int64)
    post = np.asarray(el.post, dtype=np.int64)
    w = np.asarray(el.weight, dtype=float)
    sel = (areas[pre] == str(src)) & (areas[post] == str(tgt))
    w2 = w.copy()
    w2[sel] = w[sel] * float(mult)
    kwargs = dict(pre=el.pre, post=el.post,
                  weight=jnp.asarray(w2, dtype=el.weight.dtype),
                  receptor_index=el.receptor_index, tau_ms=el.tau_ms,
                  source_calibration_status=el.source_calibration_status)
    if getattr(el, "delay_steps", None) is not None:
        kwargs["delay_steps"] = el.delay_steps
    return replace(model, params={**model.params, "edge_list": EdgeList(**kwargs)}), int(sel.sum())


def pure_pulse_drive(n_steps, n_neurons, target_idx, amp, n_pre, n_post, dtype):
    """Additive perturbation drive with GUARANTEED zero tonic content.

    Baseline segments are exactly 0; stimulus adds amp on targets only.
    Emitter tonic stays invariant automatically (schedule adds to it).
    Any schedule built by tiling emitter drive fails the companion test.
    """
    base = np.zeros((n_pre, n_neurons))
    stim = np.zeros((n_steps, n_neurons))
    stim[:, np.asarray(target_idx)] = float(amp)
    post = np.zeros((n_post, n_neurons))
    return jnp.asarray(np.concatenate([base, stim, post], axis=0), dtype=dtype)


def add_convergence(model, src, tgt, factor, seed=0):
    """Add FF edges onto the SAME target set (convergence dial).

    factor x more presynaptic sources per target, sampled from src area
    (excluding existing pres), same weight scale (median of src->tgt
    weights), receptor/tau/delay matched to src->tgt edges. Topology of
    everything else fixed. Returns (model, n_added).
    """
    from dataclasses import replace

    import numpy as _np

    from jaxfne.emitters import EdgeList

    rng = _np.random.default_rng(int(seed))
    tbl = model.neuron_table()
    areas = _np.array([str(r["area"]) for r in tbl])
    el = model.params["edge_list"]
    pre = _np.asarray(el.pre, dtype=_np.int64)
    post = _np.asarray(el.post, dtype=_np.int64)
    w = _np.asarray(el.weight, dtype=float)
    ri = _np.asarray(el.receptor_index, dtype=_np.int64)
    tau = _np.asarray(el.tau_ms, dtype=float)
    sel = (areas[pre] == str(src)) & (areas[post] == str(tgt))
    tgts = _np.unique(post[sel])
    src_pool = _np.flatnonzero(areas == str(src))
    existing = set(zip(pre[sel].tolist(), post[sel].tolist()))
    wmed = float(_np.median(w[sel]))
    rmode = int(_np.argmax(_np.bincount(ri[sel]))) if sel.sum() else 0
    tmode = float(_np.median(tau[sel])) if sel.sum() else 2.0
    delay = getattr(el, "delay_steps", None)
    dmode = int(_np.median(_np.asarray(delay)[sel])) if delay is not None else None
    n_add = int(round((float(factor) - 1.0) * sel.sum()))
    new_pre, new_post = [], []
    have = set()
    tries = 0
    while len(new_pre) < n_add and tries < 10 * max(n_add, 1):
        tries += 1
        p = int(src_pool[int(rng.integers(len(src_pool)))])
        q = int(tgts[int(rng.integers(len(tgts)))])
        if p == q or (p, q) in existing or (p, q) in have:
            continue
        have.add((p, q))
        new_pre.append(p)
        new_post.append(q)
    pre2 = _np.concatenate([pre, _np.array(new_pre, dtype=_np.int64)])
    post2 = _np.concatenate([post, _np.array(new_post, dtype=_np.int64)])
    w2 = _np.concatenate([w, _np.full(len(new_pre), wmed)])
    ri2 = _np.concatenate([ri, _np.full(len(new_pre), rmode, dtype=_np.int64)])
    tau2 = _np.concatenate([tau, _np.full(len(new_pre), tmode)])
    kwargs = dict(pre=jnp.asarray(pre2), post=jnp.asarray(post2),
                  weight=jnp.asarray(w2, dtype=el.weight.dtype),
                  receptor_index=jnp.asarray(ri2, dtype=el.receptor_index.dtype),
                  tau_ms=jnp.asarray(tau2, dtype=el.tau_ms.dtype),
                  source_calibration_status=el.source_calibration_status)
    if delay is not None:
        darr = _np.asarray(delay)
        kwargs["delay_steps"] = jnp.asarray(
            _np.concatenate([darr, _np.full(len(new_pre), dmode if dmode is not None else 0,
                                            dtype=_np.int64)]))
    return replace(model, params={**model.params, "edge_list": EdgeList(**kwargs)}), len(new_pre)
