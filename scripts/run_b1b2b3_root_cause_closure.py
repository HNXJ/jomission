import os
import json
import numpy as np
import jax
import jax.numpy as jnp

import jaxfne as jtfne
from jaxfne import Simulation, RuntimeConfig
from jaxfne._signals import _make_poisson_drive
from jaxfne._pipeline import compile_step_fn, continuation_state_from_model, run_continuation
from jomission.network.builder import build_jomission_model, simulation_with_background_poisson


def simulate_isolated_izhikevich(a, b, c, d, I, dur_ms=2000.0, dt_ms=0.1):
    n_steps = int(dur_ms / dt_ms)
    v = -65.0
    u = b * v
    spikes = 0
    settle_steps = int(500.0 / dt_ms)
    for step in range(n_steps):
        v_next = v + dt_ms * (0.04 * v**2 + 5.0 * v + 140.0 - u + I)
        u_next = u + dt_ms * a * (b * v - u)
        if v_next >= 30.0:
            if step >= settle_steps:
                spikes += 1
            v = c
            u = u_next + d
        else:
            v = v_next
            u = u_next
    return float(spikes / ((dur_ms - 500.0) / 1000.0))


def compute_isolated_rheobase(a, b, c, d):
    i_vals = np.arange(0.0, 10.05, 0.05)
    rates = np.array([simulate_isolated_izhikevich(a, b, c, d, I) for I in i_vals])
    firing = np.where(rates >= 1.0)[0]
    i_rh = float(i_vals[firing[0]]) if len(firing) > 0 else float("nan")
    return i_rh, i_vals, rates


def compute_binned_rho(spikes, dt_ms=0.1, bin_ms=10.0, max_pairs=500, rng_seed=0):
    n_steps, n_neurons = spikes.shape
    bin_steps = int(bin_ms / dt_ms)
    n_bins = n_steps // bin_steps
    binned = np.zeros((n_neurons, n_bins), dtype=np.float32)
    for b in range(n_bins):
        binned[:, b] = spikes[b * bin_steps : (b + 1) * bin_steps, :].sum(axis=0)

    active_neurons = np.where(binned.std(axis=1) > 1e-6)[0]
    if len(active_neurons) < 2:
        return 0.0, 0.0, 0

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

    mean_rho = float(np.mean(corrs)) if corrs else 0.0
    median_rho = float(np.median(corrs)) if corrs else 0.0
    return mean_rho, median_rho, len(corrs)


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
    return float(np.mean(cvs)) if cvs else 0.0, float(np.median(cvs)) if cvs else 0.0, len(cvs)


def run_b1b2b3_root_cause_closure():
    print("=== STARTING B1-B3 ROOT-CAUSE DIAGNOSTIC CLOSURE ===")
    dt_ms = 0.1
    dur_ms = 2000.0
    n_steps = int(dur_ms / dt_ms)

    model = build_jomission_model(n_per_area=100, seed=0, dt_ms=dt_ms)
    tbl = model.neuron_table()
    em = model.params['emitter']
    el = model.params['edge_list']

    cell_types = [r['cell_type'] for r in tbl]
    ct_indices = {ct: np.array([i for i, r in enumerate(tbl) if r['cell_type'] == ct]) for ct in ('E', 'PV', 'SST', 'VIP')}

    # =========================================================================
    # PART 1: PHASE / RANDOM-INITIAL-STATE TEST FOR RHO
    # =========================================================================
    print("\n" + "=" * 60)
    print("PART 1: PHASE / RANDOM-INITIAL-STATE TEST FOR RHO")
    print("=" * 60)

    # 1. Arm A: Default Canonical Initialization
    print("Running Arm A: Default Canonical Initialization...")
    sim_canon = simulation_with_background_poisson(model.cfg, duration_ms=dur_ms, dt_ms=dt_ms, seed=0)
    poisson_arr = _make_poisson_drive(n_steps=n_steps, n_neurons=400, rate_hz=2000.0, amplitude=2.0, dt_ms=dt_ms, seed=0 + 7919, target='all')

    step_fn, _ = compile_step_fn(model, dt_ms=dt_ms, kernel='baseline', record_edge_current=True, record_current_trace=True)
    init_state_canon = continuation_state_from_model(model, seed=0)

    state_canon, outs_canon = run_continuation(step_fn, init_state_canon, jnp.asarray(poisson_arr, dtype=jnp.float32))
    v_canon = np.asarray(outs_canon[0])
    spikes_canon = np.asarray(outs_canon[1])
    ec_canon = np.asarray(outs_canon[5])
    cur_canon = np.asarray(outs_canon[6])

    # 2. Arm B: Subthreshold Phase Jitter (v in [-65, -50] mV)
    print("Running Arm B: Subthreshold Phase Jitter...")
    rng = np.random.RandomState(42)
    init_state_jitter = continuation_state_from_model(model, seed=0)
    v_jitter = rng.uniform(-65.0, -50.0, size=400).astype(np.float32)
    b_vals = np.asarray(em.b)
    u_jitter = (b_vals * v_jitter).astype(np.float32)
    dyn_jitter = init_state_jitter.dynamic._replace(v=jnp.asarray(v_jitter), u=jnp.asarray(u_jitter))
    init_state_jitter = init_state_jitter._replace(dynamic=dyn_jitter)

    state_jitter, outs_jitter = run_continuation(step_fn, init_state_jitter, jnp.asarray(poisson_arr, dtype=jnp.float32))
    spikes_jitter = np.asarray(outs_jitter[1])

    # 3. Arm C: Full Cycle Dispersion (v in [c, 25.0] mV)
    print("Running Arm C: Full Dynamic Cycle Dispersion...")
    init_state_disp = continuation_state_from_model(model, seed=0)
    c_vals = np.asarray(em.c)
    v_disp = rng.uniform(c_vals, 25.0).astype(np.float32)
    u_disp = (b_vals * v_disp).astype(np.float32)
    dyn_disp = init_state_disp.dynamic._replace(v=jnp.asarray(v_disp), u=jnp.asarray(u_disp))
    init_state_disp = init_state_disp._replace(dynamic=dyn_disp)

    state_disp, outs_disp = run_continuation(step_fn, init_state_disp, jnp.asarray(poisson_arr, dtype=jnp.float32))
    spikes_disp = np.asarray(outs_disp[1])

    # Evaluate correlation and CV across arms and time windows
    arms = {
        "Arm A (Canonical)": spikes_canon,
        "Arm B (Subthreshold Jitter)": spikes_jitter,
        "Arm C (Full Dynamic Dispersion)": spikes_disp,
    }

    phase_results = {}
    print("\n--- PHASE RANDOMIZATION RESULTS ACROSS TIME WINDOWS ---")
    print(f"{'Condition':32s} | {'Time Window':14s} | {'Mean rho':9s} | {'Median rho':10s} | {'Mean CV_ISI':11s} | {'Rate (Hz)':9s}")
    print("-" * 95)

    for arm_name, s_arr in arms.items():
        phase_results[arm_name] = {}
        for w_name, (t_start, t_end) in [
            ("Full (0-2000ms)", (0, 2000)),
            ("Early (0-500ms)", (0, 500)),
            ("Late (500-2000ms)", (500, 2000)),
        ]:
            s_sub = s_arr[int(t_start / dt_ms) : int(t_end / dt_ms), :]
            m_rho, med_rho, n_pairs = compute_binned_rho(s_sub, dt_ms=dt_ms, bin_ms=10.0)
            m_cv, med_cv, n_cv = compute_cv_isi(s_sub, dt_ms=dt_ms)
            pop_rate = float(s_sub.mean() * (1000.0 / dt_ms))

            phase_results[arm_name][w_name] = {
                "mean_rho": m_rho,
                "median_rho": med_rho,
                "mean_cv_isi": m_cv,
                "median_cv_isi": med_cv,
                "rate_hz": pop_rate,
                "n_active_pairs": n_pairs,
            }
            print(f"{arm_name:32s} | {w_name:14s} | {m_rho:9.4f} | {med_rho:10.4f} | {m_cv:11.4f} | {pop_rate:9.2f}")

    # =========================================================================
    # PART 2: QUANTITATIVE OPERATING-POINT MAP (I_executed vs f_I^isolated)
    # =========================================================================
    print("\n" + "=" * 60)
    print("PART 2: QUANTITATIVE OPERATING-POINT MAP (I_executed vs f_I^isolated)")
    print("=" * 60)

    pre_arr = np.asarray(el.pre)
    post_arr = np.asarray(el.post)
    receptor_arr = np.asarray(el.receptor_index)

    post_to_e_edges = {i: [] for i in range(400)}
    post_to_i_edges = {i: [] for i in range(400)}

    for edge_i in range(len(pre_arr)):
        p_node = int(pre_arr[edge_i])
        t_node = int(post_arr[edge_i])
        rec = int(receptor_arr[edge_i])
        if rec == 0:
            post_to_e_edges[t_node].append(edge_i)
        else:
            post_to_i_edges[t_node].append(edge_i)

    mean_ec = ec_canon.mean(axis=0)
    mean_cur_total = cur_canon.mean(axis=0)
    observed_rates = spikes_canon.mean(axis=0) * (1000.0 / dt_ms)

    tonic_drive = np.asarray(em.drive)
    poisson_mean = np.asarray(poisson_arr).mean(axis=0)

    operating_map = {}
    print(f"{'Class':5s} | {'N':3s} | {'I_native':8s} | {'I_E':8s} | {'I_I':8s} | {'I_ext':8s} | {'I_rh':6s} | {'I-I_rh':7s} | {'r_obs':7s} | {'r_pred':7s} | {'Silence%':8s}")
    print("-" * 105)

    for ct in ('E', 'PV', 'SST', 'VIP'):
        idx = ct_indices[ct]
        n_c = len(idx)

        mean_a = float(em.a[idx].mean())
        mean_b = float(em.b[idx].mean())
        mean_c = float(em.c[idx].mean())
        mean_d = float(em.d[idx].mean())
        i_rh, _, _ = compute_isolated_rheobase(mean_a, mean_b, mean_c, mean_d)

        c_total = mean_cur_total[idx]
        c_e = np.array([mean_ec[post_to_e_edges[i]].sum() if post_to_e_edges[i] else 0.0 for i in idx])
        c_i = np.array([mean_ec[post_to_i_edges[i]].sum() if post_to_i_edges[i] else 0.0 for i in idx])
        c_ext = tonic_drive[idx] + poisson_mean[idx]
        r_obs = observed_rates[idx]

        mean_native_curr = float(c_total.mean())
        r_pred = simulate_isolated_izhikevich(mean_a, mean_b, mean_c, mean_d, mean_native_curr)
        silence_pct = float(np.mean(r_obs < 0.5) * 100.0)
        dist_rh = mean_native_curr - i_rh

        operating_map[ct] = {
            "n_neurons": n_c,
            "mean_i_native": mean_native_curr,
            "std_i_native": float(c_total.std()),
            "mean_i_e": float(c_e.mean()),
            "mean_i_i": float(c_i.mean()),
            "mean_i_ext": float(c_ext.mean()),
            "i_rheobase": i_rh,
            "distance_to_rheobase": dist_rh,
            "observed_rate_mean": float(r_obs.mean()),
            "observed_rate_std": float(r_obs.std()),
            "isolated_predicted_rate": r_pred,
            "silence_percent": silence_pct,
            "hyper_percent": float(np.mean(r_obs > 60.0) * 100.0),
        }

        print(f"{ct:5s} | {n_c:3d} | {mean_native_curr:8.4f} | {c_e.mean():8.4f} | {c_i.mean():8.4f} | {c_ext.mean():8.4f} | {i_rh:6.2f} | {dist_rh:+7.3f} | {r_obs.mean():7.2f} | {r_pred:7.2f} | {silence_pct:7.1f}%")

    full_output = {
        "phase_randomization_test": phase_results,
        "operating_point_map": operating_map,
    }

    os.makedirs('results', exist_ok=True)
    with open('results/b1b2b3_root_cause_closure_results.json', 'w') as f:
        json.dump(full_output, f, indent=2)

    print("\nSaved full receipts to results/b1b2b3_root_cause_closure_results.json")


if __name__ == '__main__':
    run_b1b2b3_root_cause_closure()
