import os
import json
import numpy as np
import jax
import jax.numpy as jnp
from dataclasses import replace

import jaxfne as jtfne
from jaxfne import Simulation, RuntimeConfig
import jaxfne.hdp_network as hdp
from jomission.network.builder import build_jomission_model
from jomission.paradigm.spec import JOMISSION_PARADIGM, condition_to_stimulus_schedule
from jomission.dynamics.plasticity_trajectory import (
    EdgePartition,
    compute_subset_metrics,
    summarize_plasticity_trajectory,
)


def compute_concentration(w0, wt):
    delta = np.abs(wt - w0)
    total_l1 = float(np.sum(delta))
    total_l2_sq = float(np.sum(delta ** 2))
    idx_sort = np.argsort(-delta)
    n = len(delta)

    res = {}
    for pct in [1, 5, 10]:
        k = int(np.ceil(pct / 100.0 * n))
        l1_part = float(np.sum(delta[idx_sort[:k]]))
        l2_part = float(np.sum(delta[idx_sort[:k]] ** 2))
        res[f"top_{pct}pct"] = {
            "k_edges": k,
            "fraction_l1": (l1_part / total_l1) if total_l1 > 1e-12 else 0.0,
            "fraction_l2_sq": (l2_part / total_l2_sq) if total_l2_sq > 1e-12 else 0.0,
        }
    return res


def compute_direction_metrics(w0, wt, exc_mask, eps=1e-4):
    mag0 = np.abs(w0)
    magt = np.abs(wt)
    delta_mag = magt - mag0
    signed_delta = wt - w0

    # Excitatory
    e_dmag = delta_mag[exc_mask]
    e_sdel = signed_delta[exc_mask]
    # Inhibitory
    i_dmag = delta_mag[~exc_mask]
    i_sdel = signed_delta[~exc_mask]

    return {
        "excitatory": {
            "n_edges": int(np.sum(exc_mask)),
            "mean_delta_mag": float(np.mean(e_dmag)),
            "mean_signed_delta": float(np.mean(e_sdel)),
            "frac_strengthened": float(np.mean(e_dmag > eps)),
            "frac_weakened": float(np.mean(e_dmag < -eps)),
            "frac_unchanged": float(np.mean(np.abs(e_dmag) <= eps)),
        },
        "inhibitory": {
            "n_edges": int(np.sum(~exc_mask)),
            "mean_delta_mag": float(np.mean(i_dmag)),  # magnitude strengthening
            "mean_signed_delta": float(np.mean(i_sdel)),  # hyperpolarizing shift
            "frac_strengthened_inhibition": float(np.mean(i_dmag > eps)),
            "frac_weakened_inhibition": float(np.mean(i_dmag < -eps)),
            "frac_unchanged": float(np.mean(np.abs(i_dmag) <= eps)),
        },
    }


def compute_sparse_responders(r0, rt, neuron_table):
    shift = rt - r0
    abs_shift = np.abs(shift)
    n_total = len(shift)

    res = {}
    for thresh in [1.0, 2.0, 5.0]:
        mask = abs_shift >= thresh
        active_ids = np.where(mask)[0].tolist()
        count = int(np.sum(mask))

        # Breakdowns
        by_class = {}
        for c in ("E", "PV", "SST", "VIP"):
            c_idx = [i for i, r in enumerate(neuron_table) if r["cell_type"] == c]
            by_class[c] = int(np.sum(mask[c_idx]))

        by_area = {}
        for a in ("V1", "V4", "FEF", "PFC"):
            a_idx = [i for i, r in enumerate(neuron_table) if r["area"] == a]
            by_area[a] = int(np.sum(mask[a_idx]))

        by_layer = {}
        for l in ("L1", "L2/3", "L4", "L5", "L6"):
            l_idx = [i for i, r in enumerate(neuron_table) if r["layer"] == l]
            by_layer[l] = int(np.sum(mask[l_idx]))

        res[f"ge_{int(thresh)}Hz"] = {
            "count": count,
            "fraction": float(count / n_total),
            "neuron_ids": active_ids,
            "by_class": by_class,
            "by_area": by_area,
            "by_layer": by_layer,
        }
    return res


def run_100s_extension():
    print("=== STARTING 100-S PLASTICITY EXTENSION EXPERIMENT ===")
    model = build_jomission_model(n_per_area=100, seed=0)
    part = EdgePartition.from_model(model)
    w0 = np.asarray(model.params['edge_list'].weight)
    exc_mask = (part.receptor_index == 0)

    hp = dict(hdp.v1_pfc_aaab_hdp_params())
    runtime = RuntimeConfig(recurrent_backend='edge_list', enable_hdp=True, hdp_params=hp)
    cond = [c for c in JOMISSION_PARADIGM.conditions if c.name == 'AAAB'][0]
    sched = condition_to_stimulus_schedule(cond, n_neurons=400, drive_amplitude=5.0)

    # Checkpoints: 0, 1, 2, 5, 10, 20, 30, 60, 100 s
    checkpoints = [0, 1, 2, 5, 10, 20, 30, 60, 100]
    durations = [1.0, 1.0, 3.0, 5.0, 10.0, 10.0, 30.0, 40.0]

    weights = {0: w0}
    state = None
    curr_time = 0.0

    print(f"Initial state (t=0s): n_edges={len(w0)}")

    for dur in durations:
        sim = Simulation(duration_ms=dur * 1000.0, dt_ms=0.1, seed=0, runtime=runtime)
        sig, state = jtfne.simulate(model, sim, paradigm=sched, continuation=state, return_state=True)
        curr_time += dur
        wt = np.asarray(state.dynamic.w)
        weights[int(curr_time)] = wt
        m_glob = compute_subset_metrics(w0, wt, np.arange(len(w0)))
        print(f"t={curr_time:5.1f}s: gain={m_glob['gain']:6.4f}, d2={m_glob['d2_displacement']:6.4f}, corr={m_glob['correlation']:6.4f}, delta_w={m_glob['delta_w']:+.6f}")

    # Process all checkpoints
    summaries = {}
    delta_age = {}
    prev_w = w0

    for t_sec in checkpoints:
        wt = weights[t_sec]
        sm = summarize_plasticity_trajectory(w0, wt, part)

        # Delta age (step-to-step displacement)
        if t_sec == 0:
            d_age = 0.0
        else:
            diff = wt - prev_w
            d_age = float(np.linalg.norm(diff) / np.linalg.norm(prev_w))
        delta_age[t_sec] = d_age
        prev_w = wt

        # Weight change concentration
        sm["concentration"] = compute_concentration(w0, wt)

        # Plasticity direction
        sm["direction"] = compute_direction_metrics(w0, wt, exc_mask)

        # Class pair and projection type breakdown
        sm["by_class_pair_and_proj"] = {}
        pair_c_arr = np.array(part.class_pairs)
        proj_arr = np.array(part.projection_types)
        for cp in sorted(set(part.class_pairs)):
            sm["by_class_pair_and_proj"][cp] = {}
            for pt in ("recurrent", "FF", "FB"):
                idx = np.where((pair_c_arr == cp) & (proj_arr == pt))[0]
                if len(idx) > 0:
                    sm["by_class_pair_and_proj"][cp][pt] = compute_subset_metrics(w0, wt, idx)

        summaries[t_sec] = sm

    os.makedirs('results', exist_ok=True)
    np.savez('results/plasticity_100s_weights.npz', **{f"w_{k}": v for k, v in weights.items()})

    print("\n=== RUNNING FROZEN FUNCTIONAL PROBES AT t=0, 10, 30, 60, 100 s ===")
    sched_weak = condition_to_stimulus_schedule(cond, n_neurons=400, drive_amplitude=1.0)
    sched_ord = condition_to_stimulus_schedule(cond, n_neurons=400, drive_amplitude=3.0)
    tbl = model.neuron_table()
    probe_results = {}
    runtime_probe = RuntimeConfig(recurrent_backend='edge_list')

    probe_eval_ages = [0, 10, 30, 60, 100]

    for t_eval in probe_eval_ages:
        wt_eval = weights[t_eval]
        m_eval = build_jomission_model(n_per_area=100, seed=0)
        edges = m_eval.params['edge_list']
        m_eval.params['edge_list'] = replace(edges, weight=jnp.asarray(wt_eval))

        # Weak probe
        sim_w = Simulation(duration_ms=500.0, dt_ms=0.1, seed=1234, runtime=runtime_probe)
        sig_w = jtfne.simulate(m_eval, sim_w, paradigm=sched_weak)
        rates_w = np.asarray(sig_w.spikes).mean(axis=0) * 10000.0

        # Ordinary probe
        sim_o = Simulation(duration_ms=500.0, dt_ms=0.1, seed=1234, runtime=runtime_probe)
        sig_o = jtfne.simulate(m_eval, sim_o, paradigm=sched_ord)
        rates_o = np.asarray(sig_o.spikes).mean(axis=0) * 10000.0

        probe_results[t_eval] = {
            "weak_rates": rates_w.tolist(),
            "ord_rates": rates_o.tolist(),
            "weak_mean_rate": float(rates_w.mean()),
            "ord_mean_rate": float(rates_o.mean()),
            "weak_by_class": {c: float(rates_w[np.array([i for i, r in enumerate(tbl) if r['cell_type'] == c])].mean()) for c in ("E", "PV", "SST", "VIP")},
            "ord_by_class": {c: float(rates_o[np.array([i for i, r in enumerate(tbl) if r['cell_type'] == c])].mean()) for c in ("E", "PV", "SST", "VIP")},
            "weak_by_area": {a: float(rates_w[np.array([i for i, r in enumerate(tbl) if r['area'] == a])].mean()) for a in ("V1", "V4", "FEF", "PFC")},
            "ord_by_area": {a: float(rates_o[np.array([i for i, r in enumerate(tbl) if r['area'] == a])].mean()) for a in ("V1", "V4", "FEF", "PFC")},
        }

    # Relative functional comparisons against t=0
    for p_name in ("weak", "ord"):
        r0 = np.array(probe_results[0][f"{p_name}_rates"])
        for t_eval in [10, 30, 60, 100]:
            rt = np.array(probe_results[t_eval][f"{p_name}_rates"])
            shift = rt - r0
            probe_results[t_eval][f"{p_name}_vs_0"] = {
                "relative_mean_shift": float((rt.mean() - r0.mean()) / r0.mean()),
                "correlation": float(np.corrcoef(r0, rt)[0, 1]),
                "max_abs_shift": float(np.max(np.abs(shift))),
                "mean_abs_shift": float(np.mean(np.abs(shift))),
                "shift_quantiles": {
                    "p10": float(np.percentile(shift, 10)),
                    "p50": float(np.percentile(shift, 50)),
                    "p90": float(np.percentile(shift, 90)),
                },
                "sparse_responders": compute_sparse_responders(r0, rt, tbl),
            }

    full_output = {
        "checkpoints": checkpoints,
        "delta_age": delta_age,
        "summaries": summaries,
        "probe_results": probe_results,
    }

    with open('results/plasticity_100s_extension_results.json', 'w') as f:
        json.dump(full_output, f, indent=2)

    print("\n=== RESULTS SUMMARY ACROSS 100-S CHECKPOINTS ===")
    print("Age (s) | Global Gain | Rec Gain | FF Gain | FB Gain | Global D2 | Rec D2 | FF D2  | FB D2  | Delta Age")
    print("-------------------------------------------------------------------------------------------------------")
    for t_sec in checkpoints:
        g = summaries[t_sec]["global"]
        rec = summaries[t_sec]["by_projection_type"]["recurrent"]
        ff = summaries[t_sec]["by_projection_type"]["FF"]
        fb = summaries[t_sec]["by_projection_type"]["FB"]
        da = delta_age[t_sec]
        print(f"{t_sec:7d} | {g['gain']:11.4f} | {rec['gain']:8.4f} | {ff['gain']:7.4f} | {fb['gain']:7.4f} | {g['d2_displacement']:9.4f} | {rec['d2_displacement']:6.4f} | {ff['d2_displacement']:6.4f} | {fb['d2_displacement']:6.4f} | {da:9.6f}")

    print("\nSaved full receipts to results/plasticity_100s_extension_results.json and results/plasticity_100s_weights.npz")


if __name__ == '__main__':
    run_100s_extension()
