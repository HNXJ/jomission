"""Phase-3A realized native E/I geometry: measurement module.

No transplants: every component is measured fresh on the v0.4.24 plant with
the frozen rule ``jomission_sat_ee_v0`` (see hdp_rule_sat_ee.py) active.
Old transfer values, PV-recruitment repairs, and Hill-law geometry are NOT
reused; sealed methods (star transfer assay, single-cell F-I) are.

Batteries (star topology, zero tonic, schedule-only assay, fresh plant/run):
  B1 mixed E-driver  -> E/PV/SST/VIP targets (E->Y transfer + w + current)
  B2 PV-driver       -> E targets (PV->E feedback + w + current)
  B3 SST-driver      -> E targets (SST->E feedback + w + current)

Reduced map (zero tonic, star-measured components only):
  rPV  = F_PV(S_EPV * w_EPV(rE) * rE)
  rSST = F_SST(S_ESST * w_ESST(rE) * rE)
  R(rE) = F_E(S_EE*w_EE(rE)*rE - S_PVE*w_PVE(rPV)*rPV
              - S_SSTE*w_SSTE(rSST)*rSST) - rE
roots of R = operating points. PV<->SST cross-inhibition omitted by scope
(no edges measured); recorded, not assumed zero.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import jaxfne as jtfne
from jomission.qualification.hdp_rule_sat_ee import (
    RULE_NAME,
    RULE_PARAMS,
    ensure_registered,
)

DT_MS = 0.1
SEED = 11
W_LO = RULE_PARAMS["w_lo"]

# Predeclared geometry acceptance (frozen before execution).
GEO_BANDS = {
    "rE_lo": 5.0,
    "rE_hi": 40.0,
    "rPV_min": 2.0,
    "rSST_min": 1.0,
    "sat_max": 80.0,
}


def _fresh_state(model, seed=SEED):
    dyn = jtfne.dynamic_state_from_model(
        model, hdp_params={"hdp_rule": RULE_NAME})
    return jtfne.ContinuationState(
        dynamic=dyn, prng_key=jax.random.PRNGKey(int(seed)),
        step_index=0, delay_state=None)


def _compile(model, rule_params=RULE_PARAMS):
    return jtfne.compile_step_fn(
        model, dt_ms=DT_MS, kernel="hdp", hdp_rule=RULE_NAME,
        hdp_rule_params=dict(rule_params), record_weight_trace=True)


def build_star(driver_class="E", n_targets=None, seed=0):
    """Star column: one driver of driver_class, typed targets, zero tonic.

    n_targets maps class -> count. Edges driver->targets, w_base=w_lo,
    tau by receptor sign (2.0 exc / 5.0 inh). Returns (model, driver_idx,
    target_idx_by_class).
    """
    from dataclasses import replace

    from jaxfne.emitters import EdgeList

    ensure_registered()
    n_targets = dict(n_targets or {"E": 10, "PV": 10, "SST": 10, "VIP": 10})
    classes = [driver_class]
    for c, k in n_targets.items():
        classes += [c] * int(k)
    n = len(classes)
    frac = {c: classes.count(c) / n for c in set(classes)}
    cfg = jtfne.build_laminar_column(
        "GEO", n=n, layers=["L4"], cell_type_fractions=frac,
        ei_profile="flat", geometry="laminar", edge_seed=int(seed))
    cfg = (cfg.runtime(seed=int(seed), duration_ms=1000.0, dt_ms=DT_MS, dtype="float32")
           .set_emitter("izhikevich", "cortical_eig")
           .probes(["spikes", "V_m", "source"], n_contacts=4))
    model = jtfne.construct(cfg)
    tbl = model.neuron_table()
    got = [str(r["cell_type"]) for r in tbl]
    drv = next(i for i, c in enumerate(got) if c == driver_class)
    tgt = {}
    for i, c in enumerate(got):
        if i == drv:
            continue
        tgt.setdefault(c, []).append(i)
    tgt_idx = np.array([i for i in range(n) if i != drv], dtype=np.int64)
    pre = np.full(tgt_idx.shape, drv, dtype=np.int64)
    # Receptor follows the PRESYNAPTIC (driver) neurotransmitter, not the
    # postsynaptic class: E-driver -> all excitatory (ri=0, tau 2.0);
    # PV/SST/VIP-driver -> all inhibitory (ri=1, tau 5.0).
    drv_exc = (driver_class == "E")
    ri = np.zeros(tgt_idx.shape, dtype=np.int32) if drv_exc else np.ones(tgt_idx.shape, dtype=np.int32)
    tau = (np.where(ri == 0, 2.0, 5.0)).astype(np.float32)
    el0 = model.params["edge_list"]
    new_el = EdgeList(
        pre=jnp.asarray(pre), post=jnp.asarray(tgt_idx),
        weight=jnp.asarray(np.full(tgt_idx.shape, W_LO, dtype=np.float32)
                           * np.where(ri == 0, 1.0, -1.0)),
        receptor_index=jnp.asarray(ri),
        tau_ms=jnp.asarray(tau),
        source_calibration_status=el0.source_calibration_status)
    model = replace(model, params={**model.params, "edge_list": new_el})
    e = model.params["emitter"]
    model = model.with_emitter_parameters(
        drive_per_neuron=jnp.zeros_like(e.drive))
    return model, int(drv), {c: np.array(v) for c, v in tgt.items()}


def measure_F(model, cell_classes=("E", "PV", "SST", "VIP"),
              currents=None, dur_ms=2000.0):
    """Fresh single-cell F-I per class (numpy Euler, current plant params)."""
    currents = list(currents if currents is not None else np.arange(0, 41, 2.0))
    e = model.params["emitter"]
    tbl = model.neuron_table()
    got = np.array([str(r["cell_type"]) for r in tbl])
    out = {}
    for c in cell_classes:
        i = int(np.flatnonzero(got == c)[0])
        a, b, cc, d = (float(np.asarray(e.a)[i]), float(np.asarray(e.b)[i]),
                       float(np.asarray(e.c)[i]), float(np.asarray(e.d)[i]))
        curve = []
        for I in currents:
            v, u = float(cc), float(b) * float(cc)
            n_steps = int(round(dur_ms / DT_MS))
            skip = n_steps // 2
            nsp2 = 0
            for t in range(n_steps):
                dv = 0.04 * v * v + 5.0 * v + 140.0 - u + float(I)
                du = a * (b * v - u)
                v += DT_MS * dv
                u += DT_MS * du
                if v >= 30.0:
                    if t >= skip:
                        nsp2 += 1
                    v = float(cc)
                    u += float(d)
            curve.append(nsp2 / (dur_ms / 1000.0 / 2.0))
        out[c] = (np.array(currents), np.array(curve))
    return out


def edge_current_mean(sp_pre, w_tr, tau_ms=2.0, dt_ms=0.1):
    """Exact offline current I_e(t) = w_e(t)*syn_e(t). Mean over time+edges.

    Same exponential kernel math as the engine (deterministic given recorded
    spikes and recorded w, signed weights preserved).
    """
    decay = float(np.exp(-dt_ms / tau_ms))
    syn = np.empty(sp_pre.shape[0])
    acc = 0.0
    for t in range(sp_pre.shape[0]):
        acc = acc * decay + sp_pre[t]
        syn[t] = acc
    return float((w_tr * syn[:, None]).mean())


def run_level(model, drv, amp, dur_s=6.0, seed=SEED, rule_params=RULE_PARAMS):
    """One settled level: 4s + 2s segments; window-averaged readouts."""
    ensure_registered()
    n = int(model.params["emitter"].v0.shape[0])
    dtype = model.params["emitter"].v0.dtype
    n_drive = int(dur_s / (DT_MS / 1000))
    sched = jnp.zeros((n_drive, n), dtype=dtype).at[:, drv].set(float(amp))
    step_fn, _ = _compile(model, rule_params)
    st = _fresh_state(model, seed)
    st_mid, _ = jtfne.run_continuation(step_fn, st, sched[:40000])
    jax.block_until_ready(st_mid.dynamic.v)
    st_end, out = jtfne.run_continuation(step_fn, st_mid, sched[40000:])
    jax.block_until_ready(out[0])
    return st_end, out
