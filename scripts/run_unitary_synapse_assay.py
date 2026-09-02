"""Experiment A: Unitary Synapse Assay for Jomission / JaxFNE Cortical Motifs.

Injects exactly one presynaptic spike into an otherwise controlled postsynaptic neuron.
Directly measures:
  I_peak, Q = integral I_syn dt, Delta V_peak, t_peak, decay
across representative connections:
  E -> E, E -> PV, E -> SST, E -> VIP, PV -> E, SST -> E, VIP -> E.
"""

import os
import json
import numpy as np

from jomission.network.builder import build_jomission_model


def simulate_single_unitary_response(
    a: float,
    b: float,
    c: float,
    d: float,
    w: float,
    tau_ms: float,
    i_inj: float = 0.0,
    dur_ms: float = 200.0,
    dt_ms: float = 0.1,
    spike_time_ms: float = 50.0,
):
    n_steps = int(dur_ms / dt_ms)
    spike_step = int(spike_time_ms / dt_ms)
    decay = np.exp(-dt_ms / max(tau_ms, 1e-6))

    # Initialize at fixed point or resting state
    v = -65.0
    u = b * v

    # Burn-in 100 ms with i_inj to reach exact holding potential
    burn_in_steps = int(100.0 / dt_ms)
    for _ in range(burn_in_steps):
        v_next = v + dt_ms * (0.04 * v**2 + 5.0 * v + 140.0 - u + i_inj)
        u_next = u + dt_ms * a * (b * v - u)
        v = v_next
        u = u_next

    v_baseline = float(v)
    syn_state = 0.0

    v_trace = np.zeros(n_steps)
    i_syn_trace = np.zeros(n_steps)

    for step in range(n_steps):
        # Apply incoming spike at spike_step
        if step == spike_step:
            syn_state += 1.0

        i_syn = w * syn_state
        i_total = i_inj + i_syn

        v_next = v + dt_ms * (0.04 * v**2 + 5.0 * v + 140.0 - u + i_total)
        u_next = u + dt_ms * a * (b * v - u)

        v_trace[step] = v_next
        i_syn_trace[step] = i_syn

        v = v_next
        u = u_next
        syn_state = syn_state * decay

    # Analyze response in the post-spike window
    post_spike_v = v_trace[spike_step:]
    post_spike_i = i_syn_trace[spike_step:]
    t_rel = np.arange(len(post_spike_v)) * dt_ms

    i_peak = float(np.max(post_spike_i)) if w >= 0 else float(np.min(post_spike_i))
    q_charge = float(np.sum(post_spike_i) * dt_ms)

    delta_v = post_spike_v - v_baseline
    if w >= 0:
        peak_idx = int(np.argmax(delta_v))
        delta_v_peak = float(delta_v[peak_idx])
    else:
        peak_idx = int(np.argmin(delta_v))
        delta_v_peak = float(delta_v[peak_idx])

    t_peak = float(t_rel[peak_idx])

    # Compute half-decay time from peak
    half_val = delta_v_peak / 2.0
    decay_idx = np.where(np.abs(delta_v[peak_idx:]) <= np.abs(half_val))[0]
    t_half_decay = float(decay_idx[0] * dt_ms) if len(decay_idx) > 0 else float("nan")

    return {
        "v_baseline_mV": v_baseline,
        "i_peak": i_peak,
        "q_charge_current_ms": q_charge,
        "delta_v_peak_mV": delta_v_peak,
        "t_peak_ms": t_peak,
        "t_half_decay_ms": t_half_decay,
    }


def run_unitary_synapse_assay():
    print("=== STARTING EXPERIMENT A: UNITARY SYNAPSE ASSAY ===")
    model = build_jomission_model(n_per_area=100, seed=0)
    tbl = model.neuron_table()
    em = model.params['emitter']
    el = model.params['edge_list']

    pre = np.asarray(el.pre)
    post = np.asarray(el.post)
    w = np.asarray(el.weight)
    tau = np.asarray(el.tau_ms)

    # Class cell parameters
    class_params = {}
    for ct in ("E", "PV", "SST", "VIP"):
        idx = np.array([i for i, r in enumerate(tbl) if r['cell_type'] == ct])
        class_params[ct] = {
            "a": float(em.a[idx].mean()),
            "b": float(em.b[idx].mean()),
            "c": float(em.c[idx].mean()),
            "d": float(em.d[idx].mean()),
            "drive": float(em.drive[idx].mean()),
        }

    motifs = [
        ("E", "E"),
        ("E", "PV"),
        ("E", "SST"),
        ("E", "VIP"),
        ("PV", "E"),
        ("SST", "E"),
        ("VIP", "E"),
    ]

    results = {}

    print("\n--- 1. UNITARY POSTSYNAPTIC RESPONSES AT REST (I_inj = 0) ---")
    print(f"{'Motif':9s} | {'Weight w':10s} | {'tau (ms)':8s} | {'I_peak':9s} | {'Q (I*ms)':10s} | {'dV_peak (mV)':13s} | {'t_peak (ms)':11s} | {'t_half (ms)':11s}")
    print("-" * 92)

    for src, tgt in motifs:
        idx = [ei for ei in range(len(pre)) if tbl[pre[ei]]['cell_type'] == src and tbl[post[ei]]['cell_type'] == tgt and tbl[pre[ei]]['area'] == tbl[post[ei]]['area']]
        mean_w = float(w[idx].mean())
        mean_tau = float(tau[idx].mean())

        p = class_params[tgt]
        res_rest = simulate_single_unitary_response(
            a=p['a'], b=p['b'], c=p['c'], d=p['d'],
            w=mean_w, tau_ms=mean_tau, i_inj=0.0,
        )

        res_op = simulate_single_unitary_response(
            a=p['a'], b=p['b'], c=p['c'], d=p['d'],
            w=mean_w, tau_ms=mean_tau, i_inj=p['drive'],
        )

        results[f"{src}->{tgt}"] = {
            "mean_weight": mean_w,
            "tau_ms": mean_tau,
            "at_rest": res_rest,
            "at_operating_point": res_op,
        }

        print(f"{src:3s} -> {tgt:3s} | {mean_w:10.6f} | {mean_tau:8.2f} | {res_rest['i_peak']:9.6f} | {res_rest['q_charge_current_ms']:10.6f} | {res_rest['delta_v_peak_mV']:+13.4f} | {res_rest['t_peak_ms']:11.2f} | {res_rest['t_half_decay_ms']:11.2f}")

    print("\n--- 2. UNITARY POSTSYNAPTIC RESPONSES AT OPERATING DRIVE (I_inj = I_tonic) ---")
    print(f"{'Motif':9s} | {'Weight w':10s} | {'V_hold':9s} | {'I_peak':9s} | {'Q (I*ms)':10s} | {'dV_peak (mV)':13s} | {'t_peak (ms)':11s} | {'t_half (ms)':11s}")
    print("-" * 92)

    for src, tgt in motifs:
        res_op = results[f"{src}->{tgt}"]["at_operating_point"]
        mean_w = results[f"{src}->{tgt}"]["mean_weight"]
        v_hold = res_op["v_baseline_mV"]
        print(f"{src:3s} -> {tgt:3s} | {mean_w:10.6f} | {v_hold:9.2f} | {res_op['i_peak']:9.6f} | {res_op['q_charge_current_ms']:10.6f} | {res_op['delta_v_peak_mV']:+13.4f} | {res_op['t_peak_ms']:11.2f} | {res_op['t_half_decay_ms']:11.2f}")

    os.makedirs("results", exist_ok=True)
    out_file = "results/unitary_synapse_assay_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved complete unitary synapse receipts to {out_file}")


if __name__ == "__main__":
    run_unitary_synapse_assay()
