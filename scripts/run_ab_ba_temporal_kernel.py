import os
import json
import numpy as np
import jax
import jax.numpy as jnp
from dataclasses import replace

import jaxfne as jtfne
from jaxfne import Simulation, RuntimeConfig
import jaxfne.hdp_network as hdp
from jaxfne._signals import StimulusSchedule
from jomission.network.builder import build_jomission_model
from jomission.network.rf import RFConfig, RFOperator
from jomission.dynamics.plasticity_trajectory import EdgePartition


def build_two_event_schedule(rf_op, order="AB", delta_t_ms=50.0, n_reps=4, dur_ms=531.0, base_amp=5.0):
    drv_a = rf_op.drive_for_stimulus("stimulus_A")
    drv_b = rf_op.drive_for_stimulus("stimulus_B")

    thresh_a = 0.2 * np.max(drv_a)
    thresh_b = 0.2 * np.max(drv_b)
    act_a = np.where(drv_a >= thresh_a)[0]
    act_b = np.where(drv_b >= thresh_b)[0]

    first_stim = "stimulus_A" if order == "AB" else "stimulus_B"
    second_stim = "stimulus_B" if order == "AB" else "stimulus_A"
    first_act = act_a if order == "AB" else act_b
    first_drv = drv_a if order == "AB" else drv_b
    second_act = act_b if order == "AB" else act_a
    second_drv = drv_b if order == "AB" else drv_a

    events = []
    pre_stim = 500.0
    post_stim = 500.0
    pair_dur = pre_stim + dur_ms + delta_t_ms + dur_ms + post_stim

    for rep in range(n_reps):
        t_base = rep * pair_dur
        onset1 = t_base + pre_stim
        onset2 = onset1 + dur_ms + delta_t_ms

        for u in first_act:
            norm_val = float(first_drv[u] / np.max(first_drv))
            events.append({
                "label": f"r{rep}_{first_stim}_u{u}",
                "onset_ms": float(onset1),
                "duration_ms": float(dur_ms),
                "amplitude": float(base_amp * norm_val),
                "is_drive_event": True,
                "target_indices": [int(u)],
            })
        for u in second_act:
            norm_val = float(second_drv[u] / np.max(second_drv))
            events.append({
                "label": f"r{rep}_{second_stim}_u{u}",
                "onset_ms": float(onset2),
                "duration_ms": float(dur_ms),
                "amplitude": float(base_amp * norm_val),
                "is_drive_event": True,
                "target_indices": [int(u)],
            })

    total_duration_ms = float(n_reps * pair_dur)
    sched = StimulusSchedule(events=tuple(events), n_neurons=400)
    return sched, total_duration_ms


def run_temporal_kernel_assay():
    print("=== STARTING AB/BA TEMPORAL MEMORY KERNEL ASSAY ===")
    model = build_jomission_model(n_per_area=100, seed=0)
    part = EdgePartition.from_model(model)
    w0 = np.asarray(model.params['edge_list'].weight)
    rf_op = RFOperator(RFConfig(), model)

    hp = dict(hdp.v1_pfc_aaab_hdp_params())
    runtime = RuntimeConfig(recurrent_backend='edge_list', enable_hdp=True, hdp_params=hp)

    delta_t_list = [50.0, 100.0, 250.0, 500.0, 1000.0]
    n_reps = 4
    results = {}

    print(f"Testing delta_T values: {delta_t_list} ms with N_reps={n_reps} cycles per arm.")

    for dt_val in delta_t_list:
        print(f"\n--- Testing delta_T = {dt_val:4.0f} ms ---")
        sched_ab, total_ms_ab = build_two_event_schedule(rf_op, "AB", delta_t_ms=dt_val, n_reps=n_reps)
        sched_ba, total_ms_ba = build_two_event_schedule(rf_op, "BA", delta_t_ms=dt_val, n_reps=n_reps)

        # Verify input matching
        arr_ab = sched_ab.to_array(n_steps=int(total_ms_ab / 0.1), dt_ms=0.1)
        arr_ba = sched_ba.to_array(n_steps=int(total_ms_ba / 0.1), dt_ms=0.1)
        assert np.isclose(np.sum(arr_ab), np.sum(arr_ba)), f"Energy mismatch at dt={dt_val}"
        assert np.allclose(arr_ab.sum(axis=0), arr_ba.sum(axis=0)), f"Per-neuron mismatch at dt={dt_val}"

        # Arm AB
        sim_ab = Simulation(duration_ms=total_ms_ab, dt_ms=0.1, seed=0, runtime=runtime)
        _, state_ab = jtfne.simulate(model, sim_ab, paradigm=sched_ab, return_state=True)
        w_ab = np.asarray(state_ab.dynamic.w)

        # Arm BA
        sim_ba = Simulation(duration_ms=total_ms_ba, dt_ms=0.1, seed=0, runtime=runtime)
        _, state_ba = jtfne.simulate(model, sim_ba, paradigm=sched_ba, return_state=True)
        w_ba = np.asarray(state_ba.dynamic.w)

        delta_ab = w_ab - w0
        delta_ba = w_ba - w0
        diff = delta_ab - delta_ba

        norm_ab = float(np.linalg.norm(delta_ab))
        norm_ba = float(np.linalg.norm(delta_ba))
        norm_w0 = float(np.linalg.norm(w0))
        norm_diff = float(np.linalg.norm(diff))

        d_order = float(norm_diff / (norm_ab + 1e-12))
        cos_sim = float(np.dot(delta_ab, delta_ba) / (norm_ab * norm_ba)) if (norm_ab > 1e-12 and norm_ba > 1e-12) else 1.0
        corr = float(np.corrcoef(delta_ab, delta_ba)[0, 1]) if (np.std(delta_ab) > 1e-12 and np.std(delta_ba) > 1e-12) else 1.0

        # Hierarchical projection breakdown
        proj_arr = np.array(part.projection_types)
        by_proj = {}
        for pt in ("recurrent", "FF", "FB"):
            idx = np.where(proj_arr == pt)[0]
            da_sub = delta_ab[idx]
            db_sub = delta_ba[idx]
            n_a = float(np.linalg.norm(da_sub))
            n_b = float(np.linalg.norm(db_sub))
            n_diff = float(np.linalg.norm(da_sub - db_sub))
            c_sub = float(np.dot(da_sub, db_sub) / (n_a * n_b)) if (n_a > 1e-12 and n_b > 1e-12) else 1.0
            d_ord_sub = float(n_diff / (n_a + 1e-12))
            by_proj[pt] = {
                "d_order": d_ord_sub,
                "cosine_similarity": c_sub,
                "norm_diff": n_diff,
            }

        print(f"  Delta_T = {dt_val:4.0f} ms | D_order = {d_order:8.6f} ({d_order*100:6.3f}%) | Cos = {cos_sim:8.6f} | Corr = {corr:8.6f}")
        print(f"    Recurrent D_order: {by_proj['recurrent']['d_order']*100:6.3f}% | FF: {by_proj['FF']['d_order']*100:6.3f}% | FB: {by_proj['FB']['d_order']*100:6.3f}%")

        results[str(int(dt_val))] = {
            "delta_t_ms": dt_val,
            "total_ms": total_ms_ab,
            "d_order": d_order,
            "cosine_similarity": cos_sim,
            "correlation": corr,
            "norm_delta_ab": norm_ab,
            "norm_delta_ba": norm_ba,
            "norm_diff": norm_diff,
            "by_projection": by_proj,
        }

    os.makedirs('results', exist_ok=True)
    with open('results/ab_ba_temporal_kernel_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n=== SUMMARY OF EMPIRICAL TEMPORAL MEMORY KERNEL D_order(Delta_T) ===")
    print("Delta_T (ms) | D_order (%) | Cosine Similarity | Correlation | Recurrent D_ord (%) | FF D_ord (%) | FB D_ord (%)")
    print("----------------------------------------------------------------------------------------------------------")
    for dt_val in delta_t_list:
        r = results[str(int(dt_val))]
        rec = r["by_projection"]["recurrent"]["d_order"] * 100
        ff = r["by_projection"]["FF"]["d_order"] * 100
        fb = r["by_projection"]["FB"]["d_order"] * 100
        print(f"{dt_val:12.0f} | {r['d_order']*100:10.4f}% | {r['cosine_similarity']:17.6f} | {r['correlation']:11.6f} | {rec:18.4f}% | {ff:11.4f}% | {fb:11.4f}%")

    print("\nSaved full receipts to results/ab_ba_temporal_kernel_results.json")


if __name__ == '__main__':
    run_temporal_kernel_assay()
