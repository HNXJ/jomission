"""Activity-balance H rule: direct tests of the step function (N=3, dt=0.1 ms)."""

import jax
import jax.numpy as jnp
import numpy as np

from jaxfne.hdp_rule import (
    HDPRuleContext,
    get_hdp_rule,
    list_registered_hdp_rules,
)

from jomission.dynamics import activity_balance as ab
from jomission.dynamics.activity_balance import (
    DEFAULT_PARAMS,
    RULE_NAME,
    _activity_balance_step,
)

N = 3
DT = 0.1


def _ctx(H, r, spikes=None, params=None):
    H = jnp.asarray(H, dtype=jnp.float32)
    r = jnp.asarray(r, dtype=jnp.float32)
    if spikes is None:
        spikes = jnp.zeros((N,), dtype=jnp.float32)
    spikes = jnp.asarray(spikes, dtype=jnp.float32)
    zeros = jnp.zeros((N,), dtype=jnp.float32)
    return HDPRuleContext(
        H=H,
        aux=r,
        v=zeros,
        u=zeros,
        spikes=spikes,
        prev_spikes=zeros,
        syn_state=zeros,
        w=jnp.zeros((1,), dtype=jnp.float32),
        pre=jnp.zeros((1,), dtype=jnp.int32),
        post=jnp.zeros((1,), dtype=jnp.int32),
        exc_mask=jnp.ones((N,), dtype=bool),
        dt=jnp.asarray(DT, dtype=jnp.float32),
        n_neurons=N,
        rule_params=dict(DEFAULT_PARAMS if params is None else params),
        key=jax.random.PRNGKey(0),
    )


def test_baseline_rate_22hz_fixed_point_at_H_0p4():
    upd = _activity_balance_step(_ctx(H=[0.4] * N, r=[22.0] * N))
    np.testing.assert_allclose(np.asarray(upd.dH), 0.0, atol=1e-9)


def test_silent_neuron_fixed_point_at_H_1p05():
    upd = _activity_balance_step(_ctx(H=[1.05] * N, r=[0.0] * N))
    np.testing.assert_allclose(np.asarray(upd.dH), 0.0, atol=1e-9)


def test_inside_band_at_baseline_is_static():
    upd = _activity_balance_step(_ctx(H=[1.0] * N, r=[1.0, 5.0, 10.0]))
    np.testing.assert_allclose(np.asarray(upd.dH), 0.0, atol=1e-12)
    np.testing.assert_array_equal(np.asarray(upd.d_theta["drive_bias"]), 0.0)


def test_hyperactive_suppressed_and_hypoactive_driven():
    hyper = _activity_balance_step(_ctx(H=[1.0] * N, r=[20.0] * N))
    assert np.all(np.asarray(hyper.dH) < 0.0)
    assert np.all(np.asarray(hyper.d_theta["drive_bias"]) < 0.0)

    hypo = _activity_balance_step(_ctx(H=[1.0] * N, r=[0.5] * N))
    assert np.all(np.asarray(hypo.dH) > 0.0)
    assert np.all(np.asarray(hypo.d_theta["drive_bias"]) > 0.0)


def test_drive_bias_rate_is_kappa_I_times_dH_exactly():
    upd = _activity_balance_step(_ctx(H=[0.2, 1.0, 1.7], r=[0.0, 4.0, 37.0]))
    kappa_I = DEFAULT_PARAMS["kappa_I"]
    expected = kappa_I * np.asarray(upd.dH)
    np.testing.assert_array_equal(np.asarray(upd.d_theta["drive_bias"]), expected)


def test_rate_trace_derivative_matches_definition():
    spikes = jnp.asarray([1.0, 0.0, 1.0], dtype=jnp.float32)
    r = np.array([0.0, 5.0, 12.0], dtype=np.float32)
    upd = _activity_balance_step(_ctx(H=[1.0] * N, r=r, spikes=spikes))
    r_inst = np.asarray(spikes) * 1000.0 / DT
    expected = (r_inst - r) / DEFAULT_PARAMS["tau_r"]
    np.testing.assert_allclose(np.asarray(upd.d_aux), expected, rtol=1e-6, atol=1e-9)


def test_register_is_idempotent_and_rule_is_listed():
    ab.register()
    ab.register()
    assert RULE_NAME in list_registered_hdp_rules()
    descriptor, step_fn = get_hdp_rule(RULE_NAME)
    assert step_fn is _activity_balance_step
    assert descriptor.theta_targets == ("drive_bias",)


def test_negative_control_zero_kappa_gives_pure_decay_to_one():
    params = dict(DEFAULT_PARAMS, kappa=0.0)
    H = np.array([0.4, 1.0, 1.7], dtype=np.float32)
    expected = (1.0 - H) / DEFAULT_PARAMS["tau_H"]
    for r in ([0.0] * N, [5.0] * N, [22.0] * N):
        upd = _activity_balance_step(_ctx(H=H, r=r, params=params))
        np.testing.assert_allclose(np.asarray(upd.dH), expected, rtol=1e-6, atol=1e-12)
