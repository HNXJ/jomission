import jax, jax.numpy as jnp, numpy as np
import jaxfne as jtfne
from jaxfne import Simulation, RuntimeConfig
import jaxfne.hdp_network as hdp
from jomission.network.builder import build_jomission_model
from jomission.paradigm.spec import JOMISSION_PARADIGM, condition_to_stimulus_schedule
from jomission.dynamics.plasticity_trajectory import EdgePartition, compute_subset_metrics, summarize_plasticity_trajectory
from jaxfne._signals import StimulusSchedule

model = build_jomission_model(n_per_area=100, seed=0)
part = EdgePartition.from_model(model)
w0 = np.asarray(model.params['edge_list'].weight)

hp = dict(hdp.v1_pfc_aaab_hdp_params())
runtime = RuntimeConfig(recurrent_backend='edge_list', enable_hdp=True, hdp_params=hp)

cond = [c for c in JOMISSION_PARADIGM.conditions if c.name == 'AAAB'][0]
sched = condition_to_stimulus_schedule(cond, n_neurons=400, drive_amplitude=5.0)

state = None
weights = [w0]
times = [0.0]

print('>>> ACQUISITION PHASE (0-10s with drive) <<<')
for chunk in range(10):
    sim = Simulation(duration_ms=1000.0, dt_ms=0.1, seed=0, runtime=runtime)
    sig, state = jtfne.simulate(model, sim, paradigm=sched, continuation=state, return_state=True)
    wt = np.asarray(state.dynamic.w)
    weights.append(wt)
    t_sec = (chunk + 1) * 1.0
    times.append(t_sec)
    m = compute_subset_metrics(w0, wt, np.arange(len(w0)))
    print(f"t={t_sec:.1f}s: gain={m['gain']:.4f}, d2={m['d2_displacement']:.4f}, corr={m['correlation']:.4f}, delta_w={m['delta_w']:.6f}")

print('>>> RETENTION PHASE (10-20s without drive) <<<')
null_sched = StimulusSchedule(events=(), n_neurons=400)

for chunk in range(10):
    sim = Simulation(duration_ms=1000.0, dt_ms=0.1, seed=0, runtime=runtime)
    sig, state = jtfne.simulate(model, sim, paradigm=null_sched, continuation=state, return_state=True)
    wt = np.asarray(state.dynamic.w)
    weights.append(wt)
    t_sec = 10.0 + (chunk + 1) * 1.0
    times.append(t_sec)
    m = compute_subset_metrics(w0, wt, np.arange(len(w0)))
    print(f"t={t_sec:.1f}s: gain={m['gain']:.4f}, d2={m['d2_displacement']:.4f}, corr={m['correlation']:.4f}, delta_w={m['delta_w']:.6f}")

sum_10 = summarize_plasticity_trajectory(w0, weights[10], part)
sum_20 = summarize_plasticity_trajectory(w0, weights[20], part)

print('\n=== HIERARCHICAL PROJECTION BREAKDOWN AT t=10s (ACQUISITION PEAK) ===')
for pt, sm in sum_10['by_projection_type'].items():
    print(f"  {pt} (N={sm['n_edges']}): gain={sm['gain']:.4f}, d2={sm['d2_displacement']:.4f}, corr={sm['correlation']:.4f}, delta_w={sm['delta_w']:.6f}")

print('\n=== CLASS SOURCE BREAKDOWN AT t=10s ===')
for sc, sm in sum_10['by_source_class'].items():
    print(f"  Source {sc} (N={sm['n_edges']}): gain={sm['gain']:.4f}, d2={sm['d2_displacement']:.4f}, delta_w={sm['delta_w']:.6f}")

print('\n=== CLASS TARGET BREAKDOWN AT t=10s ===')
for tc, sm in sum_10['by_target_class'].items():
    print(f"  Target {tc} (N={sm['n_edges']}): gain={sm['gain']:.4f}, d2={sm['d2_displacement']:.4f}, delta_w={sm['delta_w']:.6f}")

print('\n=== HIERARCHICAL PROJECTION BREAKDOWN AT t=20s (AFTER RETENTION) ===')
for pt, sm in sum_20['by_projection_type'].items():
    print(f"  {pt} (N={sm['n_edges']}): gain={sm['gain']:.4f}, d2={sm['d2_displacement']:.4f}, corr={sm['correlation']:.4f}, delta_w={sm['delta_w']:.6f}")

import os
os.makedirs('results', exist_ok=True)
np.savez('results/hdp_system_id_weights.npz', w0=w0, w10=weights[10], w20=weights[20], all_weights=np.array(weights), times=np.array(times))
print('Saved full weights history to results/hdp_system_id_weights.npz')
