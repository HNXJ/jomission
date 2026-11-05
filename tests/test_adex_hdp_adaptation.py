"""Item 9 (a)-(e) of docs/PLAN_LAMINAR_MOTIF.md: AdEx adaptation through HDP.

Each test targets one behaviour of `jomission.dynamics.adex_hdp` on JaxFNE's
registered-rule Izhikevich kernel. No tolerance here is a scientific threshold;
the single-neuron-vs-AdEx trace tolerance (item 9f) is UNSET and not tested.
Tier 1 (short JAX, no results writes).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxfne._hdp_registrable_kernel import simulate_edge_recurrent_izhikevich_hdp_registered
from jaxfne.emitters import EdgeList, IzhikevichParams

from jomission.dynamics import adex_hdp

RULE = adex_hdp.register()
HUGE_TAU = 1e12


def _params(drive=(0.0, 0.0)):
    n = 2
    return IzhikevichParams(
        a=jnp.full((n,), 0.02), b=jnp.full((n,), 0.2),
        c=jnp.full((n,), -65.0), d=jnp.full((n,), 8.0),
        drive=jnp.array(drive, dtype=jnp.float32), sign=jnp.ones((n,)),
        W=jnp.zeros((n, n)), v0=jnp.full((n,), -65.0), u0=jnp.full((n,), -13.0),
        source_scale=jnp.ones((n,)), labels=("E", "E"), layer_labels=("L4", "L4"),
        source_calibration_status="x",
    )


def _edges(weight=0.2):
    return EdgeList(
        pre=jnp.array([0], dtype=jnp.int32), post=jnp.array([1], dtype=jnp.int32),
        weight=jnp.array([weight], dtype=jnp.float32),
        receptor_index=jnp.array([0], dtype=jnp.int32),
        tau_ms=jnp.array([2.0], dtype=jnp.float32), source_calibration_status="x",
    )


def _run(n_steps, dt, spike_steps=(), rule_params=None, drive=(0.0, 0.0), init_state=None, weight=0.2):
    forced = np.zeros((n_steps, 2), dtype=np.float32)
    for s in spike_steps:
        forced[s, 0] = 1.0
    out = simulate_edge_recurrent_izhikevich_hdp_registered(
        _params(drive), _edges(weight), n_steps, dt, jax.random.PRNGKey(0),
        hdp_rule=RULE, hdp_rule_params=rule_params, forced_spikes=jnp.asarray(forced),
        init_state=init_state,
    )
    return out


def _bias(diag):
    return np.asarray(diag["b_trace"])


@pytest.mark.parametrize("dt", [0.1, 0.5])
def test_a_spike_jump_size_sign_and_linearity(dt):
    p = {"a_s": 0.0, "b": 5.0, "tau_w_ms": HUGE_TAU, "E_L": -65.0}
    _, _, _, d1 = _run(3, dt, spike_steps=(0,), rule_params=p)
    _, _, _, d2 = _run(3, dt, spike_steps=(0, 1), rule_params=p)
    assert np.asarray(d1["b_final"])[0] == pytest.approx(-5.0, abs=1e-5)  # sign and size, any dt
    assert np.asarray(d2["b_final"])[0] == pytest.approx(-10.0, abs=1e-5)  # one jump per spike
    assert np.asarray(d1["b_final"])[1] == 0.0  # non-spiking neuron untouched


@pytest.mark.parametrize("dt", [0.1, 0.5])
def test_b_decay_matches_tau_w(dt):
    tau, b, n = 50.0, 5.0, 301
    p = {"a_s": 0.0, "b": b, "tau_w_ms": tau, "E_L": -65.0}
    _, _, _, d = _run(n, dt, spike_steps=(0,), rule_params=p)
    bias = _bias(d)[:, 0]
    # exact discrete: jump at step 0, then n-1 decay steps
    assert bias[-1] == pytest.approx(-b * (1.0 - dt / tau) ** (n - 1), rel=1e-4)
    # continuous limit within the Euler error of this dt
    assert bias[-1] == pytest.approx(-b * np.exp(-(n - 1) * dt / tau), rel=0.02 if dt > 0.2 else 0.006)
    # decay is monotone toward 0 from below
    assert np.all(np.diff(bias[1:]) >= -1e-9) and bias[-1] < 0


def test_c_subthreshold_coupling_sign():
    from jaxfne._hdp_family_kernel import apply_registered_rule_update
    from jaxfne.hdp_rule import get_hdp_rule

    desc, step = get_hdp_rule(RULE)
    n = 2

    def one(v):
        _, _, aux, b_next = apply_registered_rule_update(
            descriptor=desc, rule_step=step, H=jnp.ones((n,), jnp.float32),
            aux=jnp.zeros((n,), jnp.float32), bias=jnp.zeros((n,), jnp.float32),
            v=jnp.full((n,), v, jnp.float32), u=jnp.zeros((n,), jnp.float32),
            spikes=jnp.zeros((n,), jnp.float32), prev_spikes=jnp.zeros((n,), jnp.float32),
            syn_state=jnp.zeros((1,), jnp.float32), w=jnp.array([3.0], jnp.float32),
            pre=jnp.array([0], jnp.int32), post=jnp.array([1], jnp.int32), exc_mask=None,
            dt=jnp.asarray(1.0, jnp.float32), n_neurons=n,
            rule_params={"a_s": 2.0, "b": 0.0, "tau_w_ms": 500.0, "E_L": -65.0},
            key=jax.random.PRNGKey(0), pre_sp=None, rule_name=RULE,
        )
        return np.asarray(b_next)

    np.testing.assert_allclose(one(-50.0), [-0.06, -0.06], atol=1e-6)  # v > E_L: w grows, bias negative
    np.testing.assert_allclose(one(-80.0), [0.06, 0.06], atol=1e-6)  # v < E_L: w negative, bias positive
    np.testing.assert_allclose(one(-65.0), [0.0, 0.0], atol=1e-8)  # v = E_L: no coupling


def test_d_bias_enters_as_negative_current_equal_to_drive():
    n_steps, dt = 200, 0.5
    inert = {"a_s": 0.0, "b": 0.0, "tau_w_ms": HUGE_TAU, "E_L": -65.0}
    v_drive, *_ = _run(n_steps, dt, rule_params=inert, drive=(-10.0, 0.0))
    p = _params()
    init = {
        "v": p.v0, "u": p.u0, "prev_spikes": jnp.zeros((2,)), "syn_state": jnp.zeros((1,)),
        "b_final": jnp.array([-10.0, 0.0]), "aux_final": jnp.array([-10.0, 0.0]),
    }
    v_bias, *_ = _run(n_steps, dt, rule_params=inert, init_state=init)
    v_none, *_ = _run(n_steps, dt, rule_params=inert)
    np.testing.assert_allclose(np.asarray(v_bias)[:, 0], np.asarray(v_drive)[:, 0], atol=1e-4)
    assert np.asarray(v_bias)[-1, 0] < np.asarray(v_none)[-1, 0] - 1.0  # inhibitory, not neutral


def test_e_no_unintended_plasticity_and_bias_recovers():
    n_steps, dt, tau = 2000, 0.5, 50.0
    # a_s = 0: with subthreshold coupling the bias settles at -a_s (v_rest - E_L), not at 0
    p = {"a_s": 0.0, "b": 5.0, "tau_w_ms": tau, "E_L": -65.0}
    _, _, _, d = _run(n_steps, dt, spike_steps=(0, 10, 20), rule_params=p, weight=0.2)
    assert np.asarray(d["w_final"])[0] == pytest.approx(0.2, abs=0.0)  # edge weight unchanged
    assert np.all(np.asarray(d["H_final"]) == 1.0)  # H untouched
    bias = _bias(d)
    assert np.abs(bias).max() < adex_hdp.B_BOUNDS[1]  # nowhere near the clip
    assert abs(bias[-1, 0]) < 1e-3 * abs(bias).max()  # recovers after activity ends (20 tau)
    # mirror equals the kernel bias (no divergence through clipping)
    np.testing.assert_allclose(np.asarray(d["aux_trace"]).reshape(bias.shape), bias, atol=1e-5)
    # fresh per run: the same call repeated gives the identical result, no carry-over
    _, _, _, d2 = _run(n_steps, dt, spike_steps=(0, 10, 20), rule_params=p, weight=0.2)
    np.testing.assert_array_equal(_bias(d2), bias)


def test_e_per_type_off_switch():
    p = {"a_s": jnp.array([0.0, 0.0]), "b": jnp.array([5.0, 0.0]), "tau_w_ms": 50.0, "E_L": -65.0}
    _, _, _, d = _run(50, 0.5, spike_steps=(0,), rule_params=p)
    assert np.asarray(d["b_final"])[0] < 0 and np.asarray(d["b_final"])[1] == 0.0
    p_off = dict(p, b=jnp.array([0.0, 0.0]))
    _, _, _, d0 = _run(50, 0.5, spike_steps=(0,), rule_params=p_off)
    assert np.all(np.asarray(d0["b_final"]) == 0.0)  # b = 0: adaptation off, nothing moves
