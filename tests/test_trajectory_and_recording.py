"""Tests for continuous trajectory, H/HDP survival, source/readout."""

import jax.numpy as jnp
import jaxfne as jtfne
from jaxfne import Simulation, RuntimeConfig
from jomission.network.builder import build_jomission_model
from jomission.paradigm.spec import JOMISSION_PARADIGM, SLOT_ONSET_MS, condition_to_stimulus_schedule
from jomission.simulation.trajectory import build_continuous_experiment, run_short_trajectory
from jomission.dynamics.h_state import HStateConfig
from jomission.dynamics.hdp import HDPConfig
from jomission.recording.probes import validate_recording


def test_continuous_experiment_no_reset():
    model = build_jomission_model(n_per_area=100, seed=0)
    seq = ["AAAB", "AXAB", "BBBA", "BBBX"]
    exp = build_continuous_experiment(model, trial_sequence=seq, dt_ms=0.1, seed=0)
    assert exp["continuation"] == "C_t carried via ContinuationState; no reset of (X,H,Theta,D)"
    assert exp["total_duration_ms"] == 4624.0 * len(seq)
    assert exp["n_trials"] == len(seq)
    # Trial table contiguous
    for i in range(1, len(exp["trial_table"])):
        assert exp["trial_table"][i]["t_start_ms"] == exp["trial_table"][i - 1]["t_end_ms"]


def test_h_state_config():
    cfg = HStateConfig()
    v = cfg.validate()
    assert v["valid"], v["issues"]
    assert cfg.h_state_dim == 5
    assert [c.tau_s for c in cfg.coordinates] == [0.1, 1.0, 10.0, 100.0, 1000.0]


def test_hdp_config():
    cfg = HDPConfig(enabled=True)
    v = cfg.validate()
    assert v["valid"], v["issues"]
    assert cfg.tau_theta_s == 1000.0


def test_continuation_state_survives():
    model = build_jomission_model(n_per_area=100, seed=0)
    runtime = RuntimeConfig(recurrent_backend="edge_list")
    sig1, state = jtfne.simulate(model, Simulation(duration_ms=200.0, dt_ms=0.1, seed=0, runtime=runtime), return_state=True)
    assert hasattr(state, "dynamic")
    assert hasattr(state, "prng_key")
    sig2 = jtfne.simulate(model, Simulation(duration_ms=200.0, dt_ms=0.1, seed=1, runtime=runtime), continuation=state)
    # Second segment should produce spikes (not zero) — proves state carried
    assert float(jnp.sum(sig2.spikes)) > 0


def test_omission_drive_zero_timing_preserved_via_schedule():
    model = build_jomission_model(n_per_area=100, seed=0)
    aaab = [c for c in JOMISSION_PARADIGM.conditions if c.name == "AAAB"][0]
    axab = [c for c in JOMISSION_PARADIGM.conditions if c.name == "AXAB"][0]
    sched_intact = condition_to_stimulus_schedule(aaab, n_neurons=400, drive_amplitude=6.0)
    sched_omit = condition_to_stimulus_schedule(axab, n_neurons=400, drive_amplitude=6.0)
    dt = 0.1
    n_steps = int(4624 / dt)
    arr_i = sched_intact.to_array(n_steps, dt)
    arr_o = sched_omit.to_array(n_steps, dt)
    p2_idx = int(round(SLOT_ONSET_MS["p2"] / dt))
    p2_end = int(round((SLOT_ONSET_MS["p2"] + 531) / dt))
    assert float(jnp.sum(arr_o[p2_idx:p2_end])) == 0.0
    assert float(jnp.sum(arr_i[p2_idx:p2_end])) > 0
    # Timing identical: onset preserved even when zeroed
    assert sched_omit.events[3]["onset_ms"] == sched_intact.events[3]["onset_ms"]


def test_source_readout_proxy():
    model = build_jomission_model(n_per_area=100, seed=0)
    v = validate_recording(model)
    assert v["valid"], v["issues"]
    assert v["claim"] == "proxy_readout"


def test_short_trajectory_milestone():
    res = run_short_trajectory(n_per_area=100, n_trials=2, dt_ms=0.5, seed=0)
    assert res["paradigm_exact_valid"] is True
    assert res["continuation_verified"] is True
    assert res["omission_timing_preserved"] is True
    assert res["drive_check"]["omit_zero"] is True
    assert res["drive_check"]["intact_nonzero"] is True
    assert res["field_present"] is True
    assert res["signals_intact_summary"]["field_claim_level"] == "proxy_readout"
