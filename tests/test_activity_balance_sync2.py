"""Corrected sync rule: constant reference r_ref = r_hi."""

import numpy as np

from jomission.dynamics import activity_balance as ab
from jomission.dynamics import activity_balance_sync as sy
from jomission.dynamics import activity_balance_sync2 as s2
from jomission.dynamics import activity_balance_sync_fixref as fx
from jaxfne.hdp_rule import list_registered_hdp_rules
from tests.test_activity_balance import _ctx
from tests.test_activity_balance_sync import _sctx

R, RF = [2.0, 5.0, 8.0], [30.0, 12.0, 9.0]


def test_registered_default_reference_is_r_hi():
    s2.register()
    assert s2.RULE_NAME in list_registered_hdp_rules()
    assert s2.DEFAULT_PARAMS["p_ref"] == ab.DEFAULT_PARAMS["r_hi"] == 10.0


def test_equals_fixref_at_p_ref_10_and_sync_when_ref_is_slow_mean():
    a = s2._step(_sctx([1.0] * 3, R, RF, params=s2.DEFAULT_PARAMS))
    b = fx._step(_sctx([1.0] * 3, R, RF, params=dict(fx.DEFAULT_PARAMS, p_ref=10.0)))
    assert np.array_equal(np.asarray(a.dH), np.asarray(b.dH))
    p = dict(s2.DEFAULT_PARAMS, p_ref=float(np.mean(R)))
    c = s2._step(_sctx([1.0] * 3, R, RF, params=p))
    d = sy._step(_sctx([1.0] * 3, R, RF))
    np.testing.assert_allclose(np.asarray(c.dH), np.asarray(d.dH), rtol=1e-5)


def test_sigma_bounded_at_startup():
    # P_s = 0, P_f = 40 Hz: sync blows up (sigma ~ 4e4 at eps 1e-3); sync2 sigma = 4.0
    rf = [40.0] * 3
    p = dict(s2.DEFAULT_PARAMS, kappa=0.0)
    d2 = np.asarray(s2._step(_sctx([1.0] * 3, [0.0] * 3, rf, params=p)).dH)
    # pure sync term = -kappa_s * sigma * c / 1000 with sigma = 40/10 = 4, c = 1; restoring 0 at H = 1
    np.testing.assert_allclose(d2, -0.5 * 4.0 * 1.0 / 1000.0, rtol=1e-5)


def test_kappa_s_zero_is_balance_only():
    p = dict(s2.DEFAULT_PARAMS, kappa_s=0.0)
    out = s2._step(_sctx([1.0] * 3, R, RF, params=p))
    ref = ab._activity_balance_step(_ctx([1.0] * 3, R))
    assert np.array_equal(np.asarray(out.dH), np.asarray(ref.dH))
