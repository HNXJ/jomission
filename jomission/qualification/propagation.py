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
