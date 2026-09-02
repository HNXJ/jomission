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
from jomission.dynamics.plasticity_trajectory import EdgePartition, compute_subset_metrics


def run_sequence_specificity_control():
    print("=== STARTING MATCHED SEQUENCE-SPECIFICITY CONTROL ASSAY ===")
    model = build_jomission_model(n_per_area=100, seed=0)
    part = EdgePartition.from_model(model)
    w0 = np.asarray(model.params['edge_list'].weight)

    rf_op = RFOperator(RFConfig(), model)
    sched_aaab = rf_op.to_stimulus_schedule('AAAB', base_amplitude=5.0)

    # Build exact matched order-scrambled schedule (BAAA)
    events_aaab = [dict(ev) for ev in sched_aaab.events]
    events_baaa = []
    for ev in events_aaab:
        new_ev = dict(ev)
        lbl = new_ev.get('label', '')
        if lbl.startswith('p1_'):
            new_ev['onset_ms'] = 3093.0
            new_ev['label'] = lbl.replace('p1_', 'p4_')
        elif lbl.startswith('p4_'):
            new_ev['onset_ms'] = 0.0
            new_ev['label'] = lbl.replace('p4_', 'p1_')
        events_baaa.append(new_ev)

    sched_baaa = StimulusSchedule(events=tuple(events_baaa), n_neurons=400)

    # Verify matching
    arr_a = sched_aaab.to_array(n_steps=40000, dt_ms=0.1)
    arr_b = sched_baaa.to_array(n_steps=40000, dt_ms=0.1)
    assert np.isclose(np.sum(arr_a), np.sum(arr_b)), "Total drive mismatch"
    assert np.allclose(arr_a.sum(axis=0), arr_b.sum(axis=0)), "Per-neuron drive mismatch"
    print("Pre-flight: Total drive energy and per-neuron integrals match exactly.")

    hp = dict(hdp.v1_pfc_aaab_hdp_params())
    runtime = RuntimeConfig(recurrent_backend='edge_list', enable_hdp=True, hdp_params=hp)

    # Simulate 20 seconds (4 full trials of 4.655s each + continuation)
    # Using 4 trials of 4655 ms = 18620 ms
    trial_dur_ms = 4655.0
    n_trials = 4
    total_ms = trial_dur_ms * n_trials  # 18.62 s

    print(f"Simulating {n_trials} complete trials ({total_ms/1000.0:.2f} s) per arm...")

    # Arm 1: AAAB
    print("Running Arm 1 (AAAB structured)...")
    st_a = None
    for tr in range(n_trials):
        sim = Simulation(duration_ms=trial_dur_ms, dt_ms=0.1, seed=0, runtime=runtime)
        _, st_a = jtfne.simulate(model, sim, paradigm=sched_aaab, continuation=st_a, return_state=True)
    w_aaab = np.asarray(st_a.dynamic.w)

    # Arm 2: BAAA (Scrambled)
    print("Running Arm 2 (BAAA order-scrambled matched)...")
    st_b = None
    for tr in range(n_trials):
        sim = Simulation(duration_ms=trial_dur_ms, dt_ms=0.1, seed=0, runtime=runtime)
        _, st_b = jtfne.simulate(model, sim, paradigm=sched_baaa, continuation=st_b, return_state=True)
    w_scrambled = np.asarray(st_b.dynamic.w)

    # Compute differential trajectory metrics
    delta_a = w_aaab - w0
    delta_b = w_scrambled - w0
    diff_ab = delta_a - delta_b

    norm_a = float(np.linalg.norm(delta_a))
    norm_b = float(np.linalg.norm(delta_b))
    norm_w0 = float(np.linalg.norm(w0))
    norm_diff = float(np.linalg.norm(diff_ab))

    cos_sim = float(np.dot(delta_a, delta_b) / (norm_a * norm_b)) if (norm_a > 1e-12 and norm_b > 1e-12) else 1.0
    corr = float(np.corrcoef(delta_a, delta_b)[0, 1]) if (np.std(delta_a) > 1e-12 and np.std(delta_b) > 1e-12) else 1.0
    diff_disp = float(norm_diff / norm_w0)

    print("\n=== GLOBAL SEQUENCE-SPECIFICITY METRICS ===")
    print(f"||Delta w_AAAB||_2 / ||w0||_2:      {norm_a / norm_w0:.6f}")
    print(f"||Delta w_scrambled||_2 / ||w0||_2: {norm_b / norm_w0:.6f}")
    print(f"Cosine Similarity cos(Delta_A, Delta_B): {cos_sim:.6f}")
    print(f"Correlation corr(Delta_A, Delta_B):     {corr:.6f}")
    print(f"Differential Displacement D_diff:        {diff_disp:.6f}")
    print(f"Relative Divergence ||Delta_A - Delta_B|| / ||Delta_A||: {norm_diff / norm_a:.6f}")

    # Hierarchical projection breakdown
    proj_arr = np.array(part.projection_types)
    proj_breakdown = {}
    for pt in ("recurrent", "FF", "FB"):
        idx = np.where(proj_arr == pt)[0]
        da_sub = delta_a[idx]
        db_sub = delta_b[idx]
        w0_sub = w0[idx]
        n_sub_a = float(np.linalg.norm(da_sub))
        n_sub_b = float(np.linalg.norm(db_sub))
        n_sub_w0 = float(np.linalg.norm(w0_sub))
        n_sub_diff = float(np.linalg.norm(da_sub - db_sub))
        c_sub = float(np.dot(da_sub, db_sub) / (n_sub_a * n_sub_b)) if (n_sub_a > 1e-12 and n_sub_b > 1e-12) else 1.0
        d_sub = float(n_sub_diff / n_sub_w0) if n_sub_w0 > 1e-12 else 0.0

        proj_breakdown[pt] = {
            "n_edges": len(idx),
            "d2_aaab": float(n_sub_a / n_sub_w0),
            "d2_scrambled": float(n_sub_b / n_sub_w0),
            "cosine_similarity": c_sub,
            "d_diff": d_sub,
            "relative_divergence": float(n_sub_diff / n_sub_a) if n_sub_a > 1e-12 else 0.0,
        }

    print("\n=== HIERARCHICAL PROJECTION BREAKDOWN ===")
    for pt, m in proj_breakdown.items():
        print(f"  {pt:10s} (N={m['n_edges']:5d}): cos={m['cosine_similarity']:.6f}, D_diff={m['d_diff']:.6f}, Rel_Divergence={m['relative_divergence']:.4%}")

    # Area pairs breakdown
    area_arr = np.array(part.area_pairs)
    area_breakdown = {}
    for ap in sorted(set(part.area_pairs)):
        idx = np.where(area_arr == ap)[0]
        da_sub = delta_a[idx]
        db_sub = delta_b[idx]
        w0_sub = w0[idx]
        n_sub_a = float(np.linalg.norm(da_sub))
        n_sub_b = float(np.linalg.norm(db_sub))
        n_sub_w0 = float(np.linalg.norm(w0_sub))
        n_sub_diff = float(np.linalg.norm(da_sub - db_sub))
        c_sub = float(np.dot(da_sub, db_sub) / (n_sub_a * n_sub_b)) if (n_sub_a > 1e-12 and n_sub_b > 1e-12) else 1.0
        d_sub = float(n_sub_diff / n_sub_w0) if n_sub_w0 > 1e-12 else 0.0

        area_breakdown[ap] = {
            "n_edges": len(idx),
            "d2_aaab": float(n_sub_a / n_sub_w0),
            "d2_scrambled": float(n_sub_b / n_sub_w0),
            "cosine_similarity": c_sub,
            "d_diff": d_sub,
            "relative_divergence": float(n_sub_diff / n_sub_a) if n_sub_a > 1e-12 else 0.0,
        }

    # Class pairs breakdown
    pair_c_arr = np.array(part.class_pairs)
    class_pair_breakdown = {}
    for cp in sorted(set(part.class_pairs)):
        idx = np.where(pair_c_arr == cp)[0]
        da_sub = delta_a[idx]
        db_sub = delta_b[idx]
        w0_sub = w0[idx]
        n_sub_a = float(np.linalg.norm(da_sub))
        n_sub_b = float(np.linalg.norm(db_sub))
        n_sub_w0 = float(np.linalg.norm(w0_sub))
        n_sub_diff = float(np.linalg.norm(da_sub - db_sub))
        c_sub = float(np.dot(da_sub, db_sub) / (n_sub_a * n_sub_b)) if (n_sub_a > 1e-12 and n_sub_b > 1e-12) else 1.0
        d_sub = float(n_sub_diff / n_sub_w0) if n_sub_w0 > 1e-12 else 0.0

        class_pair_breakdown[cp] = {
            "n_edges": len(idx),
            "d2_aaab": float(n_sub_a / n_sub_w0),
            "d2_scrambled": float(n_sub_b / n_sub_w0),
            "cosine_similarity": c_sub,
            "d_diff": d_sub,
            "relative_divergence": float(n_sub_diff / n_sub_a) if n_sub_a > 1e-12 else 0.0,
        }

    output = {
        "n_trials": n_trials,
        "total_duration_ms": total_ms,
        "global": {
            "d2_aaab": float(norm_a / norm_w0),
            "d2_scrambled": float(norm_b / norm_w0),
            "cosine_similarity": cos_sim,
            "correlation": corr,
            "d_diff": diff_disp,
            "relative_divergence": float(norm_diff / norm_a),
        },
        "by_projection_type": proj_breakdown,
        "by_area_pair": area_breakdown,
        "by_class_pair": class_pair_breakdown,
    }

    os.makedirs('results', exist_ok=True)
    with open('results/sequence_specificity_control_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    np.savez('results/sequence_specificity_weights.npz', w0=w0, w_aaab=w_aaab, w_scrambled=w_scrambled)
    print("\nSaved receipts to results/sequence_specificity_control_results.json and results/sequence_specificity_weights.npz")


if __name__ == '__main__':
    run_sequence_specificity_control()
