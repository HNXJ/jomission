"""Acceptance tests T01-T12 for dms_task (memory_dms_spec_v1.md section 5).

Run from this directory: python3 -m pytest -q test_dms_task.py
T11 and T12 use the explicit-memory reference controller (B2, b2_memory_controller.py).
"""

import numpy as np
import pytest

from b2_memory_controller import ExplicitMemoryController
from dms_task import (
    CUE, DELAY, END, GO, IDLE, IDX_CUE_VALID, IDX_GO, IDX_S, N_OBS,
    DMSEnv, Event, TaskConfig, oracle_action, run, schedule,
)

ZERO = lambda env: np.zeros(2)  # noqa: E731


def test_T01_exact_phase_transitions_at_step_boundaries():
    # onset 2.0 s = step 100; cue 25; delay 100; go 250. Target unreachable: deadline closes the trial.
    cfg = TaskConfig(events=(Event(2.0, 100.0, 100.0),))
    env = run(DMSEnv(cfg), 500, ZERO)
    phases = env.log_arrays()["phase"].tolist()
    expected = [IDLE] * 100 + [CUE] * 25 + [DELAY] * 100 + [GO] * 250 + [END] + [IDLE] * 24
    assert len(phases) == 500
    assert phases == expected
    assert env.results[0]["t_end"] == 475 and env.results[0]["success"] is False


def test_T01_rejects_off_grid_onset():
    with pytest.raises(ValueError):
        schedule(TaskConfig(events=(Event(2.005, 0.0, 0.0),)))


def test_T01_rejects_overlapping_trials():
    # first trial ends no later than 2.0 + 7.5 = 9.5 s; onset 5.0 s overlaps it
    with pytest.raises(ValueError):
        schedule(TaskConfig(events=(Event(2.0, 0.0, 0.0), Event(5.0, 0.0, 0.0))))


def test_T02_cue_visible_only_during_cue():
    cfg = TaskConfig(events=(Event(2.0, 3.0, 1.0),))
    env = run(DMSEnv(cfg), 480, ZERO)
    obs = env.log_arrays()["obs"]
    assert obs.shape == (480, N_OBS)
    cue = slice(100, 125)
    assert np.all(obs[cue, IDX_CUE_VALID] == 1.0)
    assert np.allclose(obs[cue, IDX_S], [3.0, 1.0])  # p(s) = 0, so s = q - 0
    rest = np.r_[0:100, 125:480]
    assert np.all(obs[rest, IDX_CUE_VALID] == 0.0)
    assert np.all(obs[rest, IDX_S] == 0.0)


def test_T03_no_leakage_of_hidden_target_after_cue():
    # Physics does not depend on the target, so same actions give the same trajectory.
    # Two targets, both unreachable, both run to the deadline.
    rng = np.random.default_rng(1)
    acts = rng.normal(size=(480, 2))
    pol = lambda env: acts[env.k]  # noqa: E731
    a = run(DMSEnv(TaskConfig(events=(Event(2.0, 100.0, 100.0),))), 480, pol)
    b = run(DMSEnv(TaskConfig(events=(Event(2.0, -40.0, 7.0),))), 480, pol)
    oa, ob = a.log_arrays()["obs"], b.log_arrays()["obs"]
    post = np.r_[125:480]  # DELAY, GO and END/IDLE after the cue
    assert np.array_equal(oa[post], ob[post])
    # known positive: the two targets differ during the cue, so the comparison can detect leakage
    assert not np.allclose(oa[100:125], ob[100:125])


def test_T04_motor_gated_outside_go():
    cfg = TaskConfig(events=(Event(2.0, 100.0, 100.0),))
    rng = np.random.default_rng(2)
    acts = rng.normal(size=(480, 2)) * 10.0
    zero = run(DMSEnv(cfg), 480, ZERO).log_arrays()
    rnd = run(DMSEnv(cfg), 480, lambda env: acts[env.k]).log_arrays()
    outside = np.r_[0:225, 475:480]
    assert np.all(rnd["F"][outside] == 0.0)
    # before GO the trajectories must be identical; after GO, momentum from GO forces legitimately differs
    assert np.array_equal(rnd["p"][0:225], zero["p"][0:225])
    # known positive: inside GO the same actions do move the boat
    assert np.abs(rnd["F"][225:475]).sum() > 0
    assert not np.allclose(rnd["p"][300], zero["p"][300])


def test_T05_state_continuity_across_phases():
    # The integrator must run every step, with no reset at CUE->DELAY->GO->END boundaries.
    cfg = TaskConfig(events=(Event(2.0, 100.0, 100.0),), max_force=10.0)
    rng = np.random.default_rng(3)
    acts = rng.normal(size=(480, 2))
    lg = run(DMSEnv(cfg), 480, lambda env: acts[env.k]).log_arrays()
    dt, beta = cfg.dt, cfg.damping
    V, F, P = lg["v"], lg["F"], lg["p"]
    assert np.allclose(V[1:], V[:-1] + dt * (-beta * V[:-1] + F[:-1]), atol=1e-12)
    assert np.allclose(P[1:], P[:-1] + 0.5 * dt * (V[:-1] + V[1:]), atol=1e-12)
    # known positive: the boat moves through the cue and delay (velocity is carried, not reset)
    assert np.any(np.abs(V[125]) > 0) or np.any(np.abs(V[225]) > 0) or np.any(np.abs(V[100:]) > 0)


def test_T06_deadline_inclusive_and_strict():
    tgt = Event(2.0, 1.5, 0.0)
    probe = run(DMSEnv(TaskConfig(events=(tgt,), go_s=30.0, max_force=1.0)), 1800, oracle_action)
    res = probe.results[0]
    assert res["success"] is True
    kc = res["t_end"]
    L = kc - 225
    assert L >= 1
    exact = run(DMSEnv(TaskConfig(events=(tgt,), go_s=L * 0.02, max_force=1.0)), 1800, oracle_action)
    short = run(DMSEnv(TaskConfig(events=(tgt,), go_s=(L - 1) * 0.02, max_force=1.0)), 1800, oracle_action)
    assert exact.results[0]["success"] is True and exact.results[0]["t_end"] == kc
    assert short.results[0]["success"] is False and short.results[0]["t_end"] == kc - 1


def test_T07_feasibility_bound_flags_impossible_and_oracle_reaches_feasible():
    fm, gam, go = 1.0, 0.5, 5.0
    # bound = go_s * sqrt(2) fm / beta. The |v| term is 0: v = 0 at GO onset (no force before GO).
    bound = go * np.sqrt(2) * fm / gam
    near = run(DMSEnv(TaskConfig(events=(Event(2.0, 3.0, 0.0),), max_force=fm)), 480, oracle_action).results[0]
    far = run(DMSEnv(TaskConfig(events=(Event(2.0, 30.0, 0.0),), max_force=fm)), 480, oracle_action).results[0]
    assert near["feasible"] is True and abs(near["bound"] - bound) < 1e-9
    assert near["success"] is True  # oracle demonstrates reachability on a feasible target
    assert far["feasible"] is False  # impossible target flagged
    assert far["success"] is False
    # undefined when no force bound is set
    unb = run(DMSEnv(TaskConfig(events=(Event(2.0, 3.0, 0.0),))), 480, ZERO).results[0]
    assert unb["feasible"] is None


def test_T07b_saturated_diagonal_reach_exceeds_euclidean_bound_and_is_within_sqrt2_bound():
    # Componentwise saturation F = (fm, fm) from rest for go_s: reach ~ sqrt(2) * 16.03 = 22.7 at go_s = 10.
    # Known positive: the Euclidean bound fm/beta * go_s = 20 is violated, so it is not a valid bound.
    fm, gam, go = 1.0, 0.5, 10.0
    env = DMSEnv(TaskConfig(events=(Event(2.0, 50.0, 50.0),), go_s=go, max_force=fm))
    tr = env.trials[0]
    while env.k < tr.t3:
        env.step(np.array([fm, fm]))  # gated to zero outside GO; target far, so the deadline closes the trial
    res = env.results[0]
    reach = float(np.linalg.norm(env.p))  # GO onset at rest from p = 0
    assert res["success"] is False and res["t_end"] == tr.t3
    assert reach > go * fm / gam, "fixture must exceed the Euclidean bound"
    assert res["bound"] >= reach, "feasibility bound must dominate componentwise-clipped reach"
    assert abs(res["bound"] - go * np.sqrt(2) * fm / gam) < 1e-9


def test_T07c_high_go_onset_speed_uses_velocity_term_of_bound():
    # Test-only state injection: v = (20, 0) during CUE (motor gated, so no force). Decay over the 0.5 s cue
    # leaves |v| ~ 15.6 at GO onset, above sqrt(2) fm / beta = 2.83. The |v| branch must be active and dominate
    # zero-force reach. Known positive: a bound without the |v| term (28.28) is below reach (~31).
    fm, gam, go = 1.0, 0.5, 10.0
    env = DMSEnv(TaskConfig(events=(Event(0.0, 50.0, 50.0),), cue_s=0.5, delay_s=0.0, go_s=go, max_force=fm))
    env.v = np.array([20.0, 0.0])
    tr = env.trials[0]
    while env.k < tr.t2:
        env.step(np.zeros(2))
    p_onset = env.p.copy()
    v_onset = float(np.linalg.norm(env.v))
    while env.k < tr.t3:
        env.step(np.zeros(2))
    res = env.results[0]
    reach = float(np.linalg.norm(env.p - p_onset))
    assert v_onset > np.sqrt(2) * fm / gam, "fixture must put |v| above the sqrt(2) fm / beta branch"
    assert reach > go * np.sqrt(2) * fm / gam, "fixture must exceed the no-velocity-term bound"
    assert res["success"] is False and res["t_end"] == tr.t3
    assert abs(res["bound"] - go * v_onset) < 1e-9, "bound must use the GO-onset velocity term"
    assert res["bound"] >= reach


def test_T08_deterministic_logs():
    cfg = TaskConfig(events=(Event(2.0, 1.0, 0.5), Event(14.0, -30.0, 5.0)), max_force=1.0)
    a = run(DMSEnv(cfg), 1080, oracle_action)
    b = run(DMSEnv(cfg), 1080, oracle_action)
    la, lb = a.log_arrays(), b.log_arrays()
    for key in la:
        assert np.array_equal(la[key], lb[key]), key
    assert a.results == b.results


def test_T09_baseline_plant_matches_legacy_sailor_step():
    jaxfne_paradigm = pytest.importorskip(
        "jaxfne.paradigm",
        reason="jaxfne not importable here; T09 NOT RUN (plant equality against sailor_step unverified)",
    )
    import jax.numpy as jnp  # noqa: WPS433

    if not hasattr(jaxfne_paradigm, "SailorConfig"):
        pytest.skip("installed jaxfne lacks the sailor API (SailorConfig); T09 NOT RUN")

    cfg_legacy = jaxfne_paradigm.SailorConfig(n_dim=2, dt=0.02, damping=0.5, target=jnp.zeros(2))
    state = jaxfne_paradigm.sailor_reset(cfg_legacy, __import__("jax").random.PRNGKey(0),
                                         p0=jnp.array([1.0, -2.0]), v0=jnp.array([0.3, 0.1]))
    env = DMSEnv(TaskConfig(events=(), dt=0.02, damping=0.5))
    env.p = np.array([1.0, -2.0])
    env.v = np.array([0.3, 0.1])
    rng = np.random.default_rng(4)
    for _ in range(50):
        a = rng.normal(size=2)
        # DMS plant applies gating only; force the phase to GO to compare the raw plant
        env._phase = GO
        env.step(a)
        state, _ = jaxfne_paradigm.sailor_step(cfg_legacy, state, jnp.asarray(a))
        assert np.allclose(env.p, np.asarray(state.p), atol=1e-5)
        assert np.allclose(env.v, np.asarray(state.v), atol=1e-5)


def test_T10_no_leakage_between_trials():
    ev1 = Event(2.0, 1.0, 0.0)     # reachable: closes by arrival
    ev2 = Event(14.0, -30.0, 5.0)  # onset at step 700, after trial 1's deadline (475)
    cfg = TaskConfig(events=(ev1, ev2), max_force=1.0)
    env = run(DMSEnv(cfg), 1080, oracle_action)
    lg = env.log_arrays()
    assert env.results[0]["success"] is True and env.results[0]["t_end"] < 475
    assert env.results[1]["q"] == (-30.0, 5.0)
    # trial 1 closed on arrival with min distance in (0, radius]; that value must not carry into trial 2
    assert env.results[1]["min_dist"] > cfg.radius
    # trial 2 cue is q2 - p at its own onset, not any value from trial 1
    p_on = lg["p"][700]
    assert np.allclose(lg["obs"][700:725, IDX_S], np.array([-30.0, 5.0]) - p_on)
    # trial 2 target channels are zero after its cue
    assert np.all(lg["obs"][725:1075, IDX_S] == 0.0)


TARGETS = [(3.0, 0.0), (0.0, 3.0), (-2.5, 2.0)]  # from origin; feasible at max_force=1 (bound 10)


def _single(q, delay_s=2.0):
    return TaskConfig(events=(Event(2.0, q[0], q[1]),), delay_s=delay_s, max_force=1.0)


def test_T11_cue_dependence_positive_control():
    rate = {}
    for mode in ("intact", "absent", "shuffled"):
        ok = 0
        for i, q in enumerate(TARGETS):
            other = TARGETS[(i + 1) % len(TARGETS)]  # fixed wrong target from TARGETS
            ctrl = ExplicitMemoryController(mode, shuffled_targets=[other])
            env = run(DMSEnv(_single(q)), 480, lambda e: ctrl.act(e.observe()))
            ok += env.results[0]["success"]
        rate[mode] = ok / len(TARGETS)
    # known positive: the intact cue solves every feasible trial
    assert rate["intact"] == 1.0
    assert rate["absent"] < rate["intact"]
    assert rate["shuffled"] < rate["intact"]


def test_T12_known_memory_oracle_across_delays():
    q = TARGETS[0]
    for delay in (0.0, 1.0, 2.0, 4.0):
        env = DMSEnv(_single(q, delay))
        ctrl = ExplicitMemoryController("intact")
        held = []
        for _ in range(700):  # 4 s delay ends GO at step 575
            held.append(None if ctrl.q_hold is None else ctrl.q_hold.copy())
            env.step(ctrl.act(env.observe()))
        tr = env.trials[0]
        # the register holds the target from the first CUE step through GO
        assert all(h is not None and np.allclose(h, q, atol=1e-9) for h in held[tr.onset + 1:tr.t3]), delay
        assert env.results[0]["success"] is True, delay
    # known positive: without the cue the same controller fails the same trial
    env = run(DMSEnv(_single(q, 2.0)), 480, lambda e: ExplicitMemoryController("absent").act(e.observe()))
    assert env.results[0]["success"] is False


def test_T12_register_exact_at_moving_onset():
    # trial 1 moves the boat; trial 2 cues from a non-origin position. A register that ignores p_onset fails here.
    q1, q2 = (3.0, 0.0), (2.0, 1.0)
    cfg = TaskConfig(events=(Event(2.0, q1[0], q1[1]), Event(14.0, q2[0], q2[1])), max_force=1.0)
    env = DMSEnv(cfg)
    ctrl = ExplicitMemoryController("intact")
    held = []
    for _ in range(1080):
        held.append(None if ctrl.q_hold is None else ctrl.q_hold.copy())
        env.step(ctrl.act(env.observe()))
    tr = env.trials[1]
    p_on = env.log_arrays()["p"][tr.onset]
    assert np.linalg.norm(p_on) > 0.5  # precondition: the boat is off the origin at trial 2 onset
    assert np.allclose(held[tr.onset + 1], q2, atol=1e-9)
    assert all(np.allclose(h, q2, atol=1e-9) for h in held[tr.onset + 1:tr.t3])
    assert env.results[1]["success"] is True
