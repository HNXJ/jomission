"""Experiment B: Connectivity-Normalization Assay across K_in in {10, 25, 50, 100}.

Tests candidate normalization laws:
  w \propto 1, w \propto K^{-1/2}, w \propto K^{-1}
Measures:
  \mu(I_E), \mu(|I_I|), Var(I_syn), r, CV, \rho.
Determines which scaling law preserves recurrent-input statistics when topology is sparsified.
"""

import os
import json
import numpy as np
import jax
import jax.numpy as jnp
from dataclasses import replace

import jaxfne as jtfne
from jaxfne import Simulation, RuntimeConfig
from jaxfne._signals import _make_poisson_drive
from jaxfne._pipeline import compile_step_fn, continuation_state_from_model, run_continuation
from jomission.network.builder import build_jomission_model, _apply_spatial_locality


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


def build_model_with_k_and_scaling(k_in: int, law_name: str, k_ref: int = 25, seed: int = 0):
    # 1. Build network configuration with target max_in_degree = k_in
    from jomission.network.builder import build_jomission_network
    cfg = build_jomission_network(n_per_area=100, seed=seed)
    cfg = cfg.connectivity(within_area="spatial_local", within_gain=0.35, spatial_sigma=0.08, max_in_degree=k_in)
    model = jtfne.construct(cfg)
    model = _apply_spatial_locality(model, spatial_sigma=0.08, max_in_degree=k_in, seed=seed)

    # 2. Compute scaling factor relative to k_ref = 25
    if law_name == "unscaled (w ~ 1)":
        scale_factor = 1.0
    elif law_name == "sqrt (w ~ K^-1/2)":
        scale_factor = np.sqrt(float(k_ref) / float(k_in))
    elif law_name == "linear (w ~ K^-1)":
        scale_factor = float(k_ref) / float(k_in)
    else:
        raise ValueError(f"Unknown law: {law_name}")

    # Scale recurrent weights only (keep FF/FB unchanged)
    el = model.params['edge_list']
    pre = np.asarray(el.pre)
    post = np.asarray(el.post)
    weights = np.asarray(el.weight).copy()
    tbl = model.neuron_table()

    recurrent_mask = np.array([tbl[pre[i]]['area'] == tbl[post[i]]['area'] for i in range(len(pre))])
    weights[recurrent_mask] = weights[recurrent_mask] * scale_factor

    new_el = replace(el, weight=jnp.asarray(weights, dtype=el.weight.dtype))
    new_params = dict(model.params)
    new_params['edge_list'] = new_el
    return replace(model, params=new_params), scale_factor


def run_connectivity_normalization_assay():
    print("=== STARTING EXPERIMENT B: CONNECTIVITY-NORMALIZATION ASSAY ===")
    dt_ms = 0.1
    dur_ms = 2000.0
    n_steps = int(dur_ms / dt_ms)
    k_ref = 25

    k_values = [10, 25, 50, 100]
    laws = ["unscaled (w ~ 1)", "sqrt (w ~ K^-1/2)", "linear (w ~ K^-1)"]

    poisson_arr = _make_poisson_drive(n_steps=n_steps, n_neurons=400, rate_hz=2000.0, amplitude=2.0, dt_ms=dt_ms, seed=0 + 7919, target='all')

    results = {}

    print("\n" + "=" * 95)
    print(f"{'K_in':5s} | {'Normalization Law':20s} | {'Scale':6s} | {'mu(I_E)':8s} | {'mu(|I_I|)':8s} | {'Var(I_syn)':10s} | {'Rate (Hz)':9s} | {'CV_ISI':7s} | {'rho':7s}")
    print("=" * 95)

    for k_in in k_values:
        results[k_in] = {}
        for law in laws:
            # If k_in == k_ref, all laws produce scale = 1.0, but we evaluate to verify numerical consistency
            model_test, scale_factor = build_model_with_k_and_scaling(k_in=k_in, law_name=law, k_ref=k_ref, seed=0)
            el = model_test.params['edge_list']
            rec_arr = np.asarray(el.receptor_index)
            post_arr = np.asarray(el.post)

            # Compile step function and run
            step_fn, _ = compile_step_fn(model_test, dt_ms=dt_ms, kernel='baseline', record_edge_current=True, record_current_trace=True)
            init_state = continuation_state_from_model(model_test, seed=0)
            state, outs = run_continuation(step_fn, init_state, jnp.asarray(poisson_arr, dtype=jnp.float32))

            v_trace = np.asarray(outs[0])
            spikes = np.asarray(outs[1])
            ec_trace = np.asarray(outs[5])
            cur_trace = np.asarray(outs[6])

            # Measure quantities
            mean_rates = spikes.mean(axis=0) * (1000.0 / dt_ms)
            pop_rate = float(mean_rates.mean())

            # Steady-state late window (500-2000ms) for CV and rho
            spikes_late = spikes[int(500.0 / dt_ms) :, :]
            cv_isi = compute_cv_isi(spikes_late, dt_ms=dt_ms)
            rho_val = compute_binned_rho(spikes_late, dt_ms=dt_ms)

            # Decompose currents
            mean_ec = ec_trace.mean(axis=0)
            exc_edges = np.where(rec_arr == 0)[0]
            inh_edges = np.where(rec_arr != 0)[0]

            # Inbound synaptic current sum per neuron over time
            # Compute total synaptic trace per neuron: I_syn(t, i) = sum_{e in post==i} ec(t, e)
            # Sample 20 representative neurons to compute temporal variance
            sample_neurons = np.arange(0, 400, 20)
            var_syn_list = []
            for ni in sample_neurons:
                in_edges = np.where(post_arr == ni)[0]
                if len(in_edges) > 0:
                    i_syn_t = ec_trace[:, in_edges].sum(axis=1)
                    var_syn_list.append(float(np.var(i_syn_t)))
            var_syn = float(np.mean(var_syn_list)) if var_syn_list else 0.0

            # Mean I_E and I_I across neurons
            i_e_per_post = np.bincount(post_arr[exc_edges], weights=mean_ec[exc_edges], minlength=400)
            i_i_per_post = np.bincount(post_arr[inh_edges], weights=mean_ec[inh_edges], minlength=400)

            mu_ie = float(i_e_per_post.mean())
            mu_ii = float(np.abs(i_i_per_post.mean()))

            results[k_in][law] = {
                "scale_factor": float(scale_factor),
                "mu_I_E": mu_ie,
                "mu_abs_I_I": mu_ii,
                "var_I_syn": var_syn,
                "firing_rate_hz": pop_rate,
                "cv_isi": cv_isi,
                "rho_late": rho_val,
            }

            print(f"{k_in:5d} | {law:20s} | {scale_factor:6.3f} | {mu_ie:8.4f} | {mu_ii:8.4f} | {var_syn:10.6f} | {pop_rate:9.2f} | {cv_isi:7.4f} | {rho_val:+7.4f}")

    # Summary analysis
    os.makedirs("results", exist_ok=True)
    out_file = "results/connectivity_normalization_assay_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved complete normalization receipts to {out_file}")


if __name__ == "__main__":
    run_connectivity_normalization_assay()
