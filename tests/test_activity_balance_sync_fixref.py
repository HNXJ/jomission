"""Fixed-denominator probe: equals the sync rule when p_ref = P_s, differs otherwise."""

import numpy as np

from jomission.dynamics import activity_balance_sync as sy
from jomission.dynamics import activity_balance_sync_fixref as fx
from tests.test_activity_balance_sync import _sctx

R, RF = [2.0, 5.0, 8.0], [30.0, 12.0, 9.0]


def test_equals_sync_when_pref_is_slow_mean():
    p = dict(fx.DEFAULT_PARAMS, p_ref=float(np.mean(R)))
    a = fx._step(_sctx([1.0] * 3, R, RF, params=p))
    b = sy._step(_sctx([1.0] * 3, R, RF))
    np.testing.assert_allclose(np.asarray(a.dH), np.asarray(b.dH), rtol=1e-5)


def test_differs_when_pref_differs_and_scales_sigma():
    p1 = dict(fx.DEFAULT_PARAMS, p_ref=5.0, kappa=0.0)
    p2 = dict(fx.DEFAULT_PARAMS, p_ref=10.0, kappa=0.0)
    d1 = np.asarray(fx._step(_sctx([1.0] * 3, R, RF, params=p1)).dH)
    d2 = np.asarray(fx._step(_sctx([1.0] * 3, R, RF, params=p2)).dH)
    # with kappa=0 and H=1 the restoring term is 0, so dH is pure sync term, inversely proportional to p_ref
    assert np.all(d1 < 0) and np.allclose(d1, 2.0 * d2, rtol=1e-5)


def test_low_slow_rate_does_not_blow_up():
    p = dict(fx.DEFAULT_PARAMS, p_ref=10.0)
    lo = fx._step(_sctx([1.0] * 3, [0.01] * 3, [1.0] * 3, params=p))
    assert np.all(np.isfinite(np.asarray(lo.dH))) and float(np.abs(np.asarray(lo.dH)).max()) < 1e-2
