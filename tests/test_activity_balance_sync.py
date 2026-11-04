"""Activity-balance + synchrony rule: direct tests of the step function (N=3, dt=0.1 ms)."""

import dataclasses

import jax.numpy as jnp
import numpy as np

from jaxfne.hdp_rule import list_registered_hdp_rules

from jomission.dynamics import activity_balance as ab
from jomission.dynamics import activity_balance_sync as sy
from tests.test_activity_balance import _ctx


def _sctx(H, r, rf, spikes=None, params=None):
    c = _ctx(H, r, spikes=spikes, params=sy.DEFAULT_PARAMS if params is None else params)
    aux = jnp.stack([jnp.asarray(r, jnp.float32), jnp.asarray(rf, jnp.float32)], axis=1)
    return dataclasses.replace(c, aux=aux)


def test_registered_and_idempotent():
    sy.register()
    sy.register()
    assert sy.RULE_NAME in list_registered_hdp_rules()


def test_no_burst_equals_balance_only():
    # P_f == P_s: sigma = 0, so dH equals the balance rule exactly (negative control)
    r = [2.0, 5.0, 20.0]
    out = sy._step(_sctx([1.0, 1.0, 1.0], r, r))
    ref = ab._activity_balance_step(_ctx([1.0, 1.0, 1.0], r))
    assert np.array_equal(np.asarray(out.dH), np.asarray(ref.dH))
    assert np.array_equal(np.asarray(out.d_aux[:, 0]), np.asarray(ref.d_aux))


def test_burst_pushes_H_down_in_band():
    # P_f = 3 P_s, all rates in band: balance part is 0 at H=1, sync part negative
    out = sy._step(_sctx([1.0, 1.0, 1.0], [5.0, 5.0, 5.0], [15.0, 15.0, 15.0]))
    assert np.all(np.asarray(out.dH) < 0)
    assert np.all(np.asarray(out.d_theta["drive_bias"]) < 0)


def test_weight_is_share_and_mean_one():
    r = jnp.asarray([5.0, 5.0, 5.0])
    rf = jnp.asarray([30.0, 10.0, 5.0])
    sigma, c = sy.sync_terms(r, rf, 0.5, 1e-3)
    assert float(sigma) > 0
    assert abs(float(jnp.mean(c)) - 1.0) < 1e-6
    assert float(c[0]) > float(c[1]) > float(c[2])


def test_kappa_s_zero_is_null():
    p = dict(sy.DEFAULT_PARAMS, kappa_s=0.0)
    r, rf = [2.0, 5.0, 20.0], [50.0, 1.0, 1.0]
    out = sy._step(_sctx([1.0, 1.0, 1.0], r, rf, params=p))
    ref = ab._activity_balance_step(_ctx([1.0, 1.0, 1.0], r))
    assert np.array_equal(np.asarray(out.dH), np.asarray(ref.dH))


def test_db_is_kappaI_dH_and_fast_trace_derivative():
    spikes = [1.0, 0.0, 0.0]
    r, rf = [5.0, 5.0, 5.0], [2.0, 2.0, 2.0]
    out = sy._step(_sctx([1.0, 1.0, 1.0], r, rf, spikes=spikes))
    assert np.array_equal(np.asarray(out.d_theta["drive_bias"]),
                          np.asarray(sy.DEFAULT_PARAMS["kappa_I"] * out.dH))
    exp = (np.asarray(spikes) * 1000.0 / 0.1 - np.asarray(rf)) / sy.DEFAULT_PARAMS["tau_f"]
    np.testing.assert_allclose(np.asarray(out.d_aux[:, 1]), exp, rtol=1e-5)


def test_silent_population_no_nan():
    out = sy._step(_sctx([1.0, 1.0, 1.0], [0.0] * 3, [0.0] * 3))
    assert np.all(np.isfinite(np.asarray(out.dH)))
