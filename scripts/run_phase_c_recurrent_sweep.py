"""Phase C: Generic Recurrent-Gain System-Identification Sweep.

Evaluates scaling recurrent weights by g_R in {1, 2, 4, 8, 16, 32}.
Characterizes transition map:
  (R_E, R_I) -> (r, CV_ISI, Fano, rho, stability)
where R_E = <I_E> / <I_ext>, R_I = <|I_I|> / <I_ext>.

Also measures executed unitary E->E and PV->E responses at each scale.
Stops immediately upon non-finite voltages or runaway dynamics.
"""

import os
import json
import numpy as np
import jax
import jax.numpy as jnp
from dataclasses import replace

import jaxfne as jtfne
from jaxfne._signals import _make_poisson_drive
from jaxfne._pipeline import compile_step_fn, continuation_state_from_model, run_continuation
from jomission.network.builder import build_jomission_model


def compute_fano_factor(spikes, dt_ms=0.1, bin_ms=50.0):
    n_steps, n_neurons = spikes.shape
    bin_steps = int(bin_ms / dt_ms)
    n_bins = n_steps // bin_steps
    binned = np.zeros((n_neurons, n_bins), dtype=np.float32)
    for b in range(n_bins):
        binned[:, b] = spikes[b * bin_steps : (b + 1) * bin_steps, :].sum(axis=0)

    active_neurons = np.where(binned.mean(axis=1) > 0.05)[0]
    if len(active_neurons) == 0:
        return 0.0, 0.0

    fanos = []
    for ni in active_neurons:
        m = np.mean(binned[ni])
        v = np.var(binned[ni])
        if m > 1e-6:
            fanos.append(float(v / m))

    return float(np.mean(fanos)) if fanos else 0.0, float(np.median(fanos)) if fanos else 0.0


def compute_binned_rho(spikes, dt_ms=0.1, bin_ms=10.0, max_pairs=300, rng_seed=0):
    n_steps, n_neurons = spikes.shape
    bin_steps = int(bin_ms / dt_ms)
    n_bins = n_steps // bin_steps
    binned = np.zeros((n_neurons, n_bins), dtype=np.float32)
    for b in range(n_bins):
        binned[:, b] = spikes[b * bin_steps : (b + 1) * bin_steps, :].sum(axis=0)

    active_neurons = np.where(binned.std(axis=1) > 1e-6)[0]
    if len(active_neurons) < 2:
        return 0.0

    rng = np.random.RandomState(rng_seed)
    pairs = []
    for i_idx in range(len(active_neurons)):
        for j_idx in range(i_idx + 1, len(active_neurons)):
            pairs.append((active_neurons[i_idx], active_neurons[j_idx]))

    if len(pairs) > max_pairs:
        sampled_indices = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[idx] for idx in sampled_indices]

    corrs = []
    for ni, nj in pairs:
        c = np.corrcoef(binned[ni], binned[nj])[0, 1]
        if np.isfinite(c):
            corrs.append(float(c))

    return float(np.mean(corrs)) if corrs else 0.0


def compute_cv_isi(spikes, dt_ms=0.1):
    n_steps, n_neurons = spikes.shape
    cvs = []
    for ni in range(n_neurons):
        spike_times = np.where(spikes[:, ni] > 0.5)[0] * dt_ms
        if len(spike_times) >= 3:
            isis = np.diff(spike_times)
            mean_isi = np.mean(isis)
            std_isi = np.std(isis)
            if mean_isi > 1e-6:
                cvs.append(float(std_isi / mean_isi))
    return float(np.mean(cvs)) if cvs else 0.0


def simulate_unitary(a, b, c, d, w, tau_ms, i_inj, dur_ms=100.0, dt_ms=0.1, spike_t=20.0):
    n_steps = int(dur_ms / dt_ms)
    spike_step = int(spike_t / dt_ms)
    decay = np.exp(-dt_ms / max(tau_ms, 1e-6))

    v = -65.0
    u = b * v
    # settle 50 ms
    for _ in range(int(50.0 / dt_ms)):
        vn = v + dt_ms * (0.04 * v**2 + 5.0 * v + 140.0 - u + i_inj)
        un = u + dt_ms * a * (b * v - u)
        v, u = vn, un

    v_base = float(v)
    syn = 0.0
    v_trace = np.zeros(n_steps)
    for s in range(n_steps):
        if s == spike_step:
            syn += 1.0
        i_syn = w * syn
        vn = v + dt_ms * (0.04 * v**2 + 5.0 * v + 140.0 - u + i_inj + i_syn)
        un = u + dt_ms * a * (b * v - u)
        v_trace[s] = vn
        v, u = vn, un
        syn *= decay

    dv = v_trace[spike_step:] - v_base
    if w >= 0:
        return float(np.max(dv))
    else:
        return float(np.min(dv))


def run_phase_c_recurrent_sweep():
    print("=== STARTING PHASE C: GENERIC RECURRENT-GAIN SYSTEM-ID SWEEP ===")
    dt_ms = 0.1
    dur_ms = 2000.0
    n_steps = int(dur_ms / dt_ms)
    g_r_values = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]

    # Base canonical model
    base_model = build_jomission_model(n_per_area=100, seed=0)
    tbl = base_model.neuron_table()
    em = base_model.params['emitter']
    el = base_model.params['edge_list']

    pre_base = np.asarray(el.pre)
    post_base = np.asarray(el.post)
    w_base = np.asarray(el.weight)
    tau_base = np.asarray(el.tau_ms)
    rec_idx_base = np.asarray(el.receptor_index)

    # Recurrent edge mask
    area_labels = [r['area'] for r in tbl]
    class_labels = np.array([r['cell_type'] for r in tbl])
    rec_edge_mask = np.array([area_labels[pre_base[i]] == area_labels[post_base[i]] for i in range(len(pre_base))])

    # Class parameter averages for unitary simulations
    e_idx = np.where(class_labels == 'E')[0]
    pv_idx = np.where(class_labels == 'PV')[0]
    sst_idx = np.where(class_labels == 'SST')[0]
    vip_idx = np.where(class_labels == 'VIP')[0]

    e_params = {
        'a': float(np.asarray(em.a)[e_idx].mean()),
        'b': float(np.asarray(em.b)[e_idx].mean()),
        'c': float(np.asarray(em.c)[e_idx].mean()),
        'd': float(np.asarray(em.d)[e_idx].mean()),
        'drive': float(np.asarray(em.drive)[e_idx].mean()),
    }

    # Mean weights for E->E and PV->E
    ee_edges = [ei for ei in range(len(pre_base)) if class_labels[pre_base[ei]] == 'E' and class_labels[post_base[ei]] == 'E' and rec_edge_mask[ei]]
    pve_edges = [ei for ei in range(len(pre_base)) if class_labels[pre_base[ei]] == 'PV' and class_labels[post_base[ei]] == 'E' and rec_edge_mask[ei]]

    w_ee_base = float(w_base[ee_edges].mean())
    tau_ee = float(tau_base[ee_edges].mean())
    w_pve_base = float(w_base[pve_edges].mean())
    tau_pve = float(tau_base[pve_edges].mean())

    poisson_arr = _make_poisson_drive(n_steps=n_steps, n_neurons=400, rate_hz=2000.0, amplitude=2.0, dt_ms=dt_ms, seed=0 + 7919, target='all')
    i_ext_mean = float(np.asarray(em.drive).mean() + poisson_arr.mean())

    sweep_results = {}

    print("\n" + "=" * 115)
    print(f"{'g_R':5s} | {'R_E (%)':8s} | {'R_I (%)':8s} | {'r_pop':7s} | {'r_E':6s} | {'r_PV':6s} | {'r_SST':6s} | {'r_VIP':6s} | {'CV_ISI':7s} | {'Fano':6s} | {'rho_late':8s} | {'uEPSP (mV)':10s} | {'uIPSP (mV)':10s}")
    print("=" * 115)

    for g_r in g_r_values:
        # Scale recurrent weights
        scaled_w = w_base.copy()
        scaled_w[rec_edge_mask] = w_base[rec_edge_mask] * g_r

        new_el = replace(el, weight=jnp.asarray(scaled_w, dtype=el.weight.dtype))
        new_params = dict(base_model.params)
        new_params['edge_list'] = new_el
        model_scaled = replace(base_model, params=new_params)

        # Unitary responses
        uepsp = simulate_unitary(e_params['a'], e_params['b'], e_params['c'], e_params['d'], w_ee_base * g_r, tau_ee, e_params['drive'])
        uipsp = simulate_unitary(e_params['a'], e_params['b'], e_params['c'], e_params['d'], w_pve_base * g_r, tau_pve, e_params['drive'])

        # Simulation
        step_fn, _ = compile_step_fn(model_scaled, dt_ms=dt_ms, kernel='baseline', record_edge_current=True, record_current_trace=True)
        init_state = continuation_state_from_model(model_scaled, seed=0)
        state, outs = run_continuation(step_fn, init_state, jnp.asarray(poisson_arr, dtype=jnp.float32))

        v_trace = np.asarray(outs[0])
        spikes = np.asarray(outs[1])
        ec_trace = np.asarray(outs[5])
        cur_trace = np.asarray(outs[6])

        # Check finite
        if not (np.all(np.isfinite(v_trace)) and np.all(np.isfinite(spikes))):
            print(f"{g_r:5.1f} | RUNAWAY / NON-FINITE DYNAMICS DETECTED! STOPPING SWEEP.")
            sweep_results[g_r] = {"status": "NON_FINITE_RUNAWAY"}
            break

        # Check saturation
        rates_per_neuron = spikes.mean(axis=0) * (1000.0 / dt_ms)
        pop_rate = float(rates_per_neuron.mean())
        if pop_rate > 200.0:
            print(f"{g_r:5.1f} | SEVERE POPULATION HYPERACTIVATION (>200 Hz). STOPPING SWEEP.")
            sweep_results[g_r] = {"status": "HYPERACTIVATION_SATURATION", "pop_rate": pop_rate}
            break

        # Currents
        mean_ec = ec_trace.mean(axis=0)
        exc_edges = np.where(rec_idx_base == 0)[0]
        inh_edges = np.where(rec_idx_base != 0)[0]

        i_e_soma = np.bincount(post_base[exc_edges], weights=mean_ec[exc_edges], minlength=400)
        i_i_soma = np.bincount(post_base[inh_edges], weights=mean_ec[inh_edges], minlength=400)

        mu_ie = float(i_e_soma.mean())
        mu_ii = float(np.abs(i_i_soma.mean()))

        r_e_ratio = (mu_ie / i_ext_mean) * 100.0
        r_i_ratio = (mu_ii / i_ext_mean) * 100.0

        # Class rates
        r_e = float(rates_per_neuron[e_idx].mean())
        r_pv = float(rates_per_neuron[pv_idx].mean())
        r_sst = float(rates_per_neuron[sst_idx].mean())
        r_vip = float(rates_per_neuron[vip_idx].mean())

        # Steady-state window 500-2000 ms
        late_spikes = spikes[int(500.0 / dt_ms) :, :]
        cv_isi = compute_cv_isi(late_spikes, dt_ms=dt_ms)
        fano_mean, fano_med = compute_fano_factor(late_spikes, dt_ms=dt_ms)
        rho_late = compute_binned_rho(late_spikes, dt_ms=dt_ms)

        sweep_results[g_r] = {
            "g_R": g_r,
            "R_E_percent": r_e_ratio,
            "R_I_percent": r_i_ratio,
            "mu_I_E": mu_ie,
            "mu_abs_I_I": mu_ii,
            "pop_rate": pop_rate,
            "rate_E": r_e,
            "rate_PV": r_pv,
            "rate_SST": r_sst,
            "rate_VIP": r_vip,
            "cv_isi": cv_isi,
            "fano_mean": fano_mean,
            "fano_median": fano_med,
            "rho_late": rho_late,
            "unitary_epsp_mV": uepsp,
            "unitary_ipsp_mV": uipsp,
        }

        print(f"{g_r:5.1f} | {r_e_ratio:7.2f}% | {r_i_ratio:7.2f}% | {pop_rate:7.2f} | {r_e:6.2f} | {r_pv:6.2f} | {r_sst:6.2f} | {r_vip:6.2f} | {cv_isi:7.4f} | {fano_med:6.2f} | {rho_late:+8.4f} | {uepsp:+10.4f} | {uipsp:+10.4f}")

    os.makedirs("results", exist_ok=True)
    out_file = "results/phase_c_recurrent_gain_sweep_results.json"
    with open(out_file, "w") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"\nSaved complete Phase C sweep receipts to {out_file}")


if __name__ == "__main__":
    run_phase_c_recurrent_sweep()
