import numpy as np
import pytest
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


def test_plasticity_partition_exhaustive():
    model = build_jomission_model(n_per_area=100, seed=0)
    part = EdgePartition.from_model(model)
    assert part.n_edges == 10666

    pt_counts = {pt: sum(1 for p in part.projection_types if p == pt) for pt in ('recurrent', 'FF', 'FB')}
    assert sum(pt_counts.values()) == part.n_edges
    assert pt_counts['recurrent'] == 10076
    assert pt_counts['FF'] == 287
    assert pt_counts['FB'] == 303

    assert len(set(part.area_pairs)) == 10
    assert len(set(part.class_pairs)) == 16


def test_plasticity_aggregate_reconstruction():
    model = build_jomission_model(n_per_area=100, seed=0)
    part = EdgePartition.from_model(model)
    w0 = np.asarray(model.params['edge_list'].weight)

    rng = np.random.default_rng(123)
    wt = w0 + rng.normal(0, 0.01, size=w0.shape)

    summary = summarize_plasticity_trajectory(w0, wt, part)

    n_rec = summary['by_projection_type']['recurrent']['n_edges']
    n_ff = summary['by_projection_type']['FF']['n_edges']
    n_fb = summary['by_projection_type']['FB']['n_edges']
    assert n_rec + n_ff + n_fb == summary['global']['n_edges'] == part.n_edges

    n_src_sum = sum(summary['by_source_class'][c]['n_edges'] for c in ('E', 'PV', 'SST', 'VIP'))
    assert n_src_sum == part.n_edges

    n_tgt_sum = sum(summary['by_target_class'][c]['n_edges'] for c in ('E', 'PV', 'SST', 'VIP'))
    assert n_tgt_sum == part.n_edges


def test_plasticity_checkpoint_continuation_exact():
    model = build_jomission_model(n_per_area=100, seed=0)
    hp = dict(hdp.v1_pfc_aaab_hdp_params())
    runtime = RuntimeConfig(recurrent_backend='edge_list', enable_hdp=True, hdp_params=hp)

    cond = [c for c in JOMISSION_PARADIGM.conditions if c.name == 'AAAB'][0]
    sched = condition_to_stimulus_schedule(cond, n_neurons=400, drive_amplitude=5.0)

    sim_full = Simulation(duration_ms=400.0, dt_ms=0.1, seed=0, runtime=runtime)
    sig_full, state_full = jtfne.simulate(model, sim_full, paradigm=sched, return_state=True)

    sim_1 = Simulation(duration_ms=200.0, dt_ms=0.1, seed=0, runtime=runtime)
    sig_1, state_1 = jtfne.simulate(model, sim_1, paradigm=sched, return_state=True)

    sim_2 = Simulation(duration_ms=200.0, dt_ms=0.1, seed=0, runtime=runtime)
    sig_2, state_2 = jtfne.simulate(model, sim_2, paradigm=sched, continuation=state_1, return_state=True)

    w_full = np.asarray(state_full.dynamic.w)
    w_2 = np.asarray(state_2.dynamic.w)

    np.testing.assert_array_equal(w_2, w_full)


def test_plasticity_recording_invariance():
    model = build_jomission_model(n_per_area=100, seed=0)
    cond = [c for c in JOMISSION_PARADIGM.conditions if c.name == 'AAAB'][0]
    sched = condition_to_stimulus_schedule(cond, n_neurons=400, drive_amplitude=5.0)

    hp_plain = dict(hdp.v1_pfc_aaab_hdp_params())
    hp_plain['record_weight_trace'] = False

    hp_diag = dict(hp_plain)
    hp_diag['record_weight_trace'] = True

    sim_plain = Simulation(duration_ms=200.0, dt_ms=0.1, seed=0, runtime=RuntimeConfig(recurrent_backend='edge_list', enable_hdp=True, hdp_params=hp_plain))
    sig_plain, st_plain = jtfne.simulate(model, sim_plain, paradigm=sched, return_state=True)

    sim_diag = Simulation(duration_ms=200.0, dt_ms=0.1, seed=0, runtime=RuntimeConfig(recurrent_backend='edge_list', enable_hdp=True, hdp_params=hp_diag))
    sig_diag, st_diag = jtfne.simulate(model, sim_diag, paradigm=sched, return_state=True)

    np.testing.assert_array_equal(np.asarray(st_plain.dynamic.w), np.asarray(st_diag.dynamic.w))
    np.testing.assert_array_equal(np.asarray(sig_plain.spikes), np.asarray(sig_diag.spikes))
    np.testing.assert_array_equal(np.asarray(sig_plain.V_m), np.asarray(sig_diag.V_m))
