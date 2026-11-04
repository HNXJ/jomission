"""Multi-trial exposure: one simulate call per trial, state carried between trials, all trials aggregated."""

import types

import jax.numpy as jnp
import jaxfne as jtfne

from jomission.simulation import exposure


def _fake_signal(value, n_steps=10, nan=False, metadata_scope=None):
    v = jnp.full((n_steps, 1000), value, dtype=jnp.float32)
    if nan:
        v = v.at[0, 0].set(jnp.nan)
    return types.SimpleNamespace(
        V_m=v,
        spikes=jnp.zeros((n_steps, 1000), dtype=jnp.float32),
        time_ms=jnp.arange(n_steps, dtype=jnp.float32),
        field=None,
        metadata={},
        summary=lambda: {"fake": True},
    )


def test_one_call_per_trial_and_state_carried(monkeypatch):
    calls = []
    original = jtfne.simulate

    def recording(model, sim, *args, **kwargs):
        calls.append(kwargs)
        return original(model, sim, *args, **kwargs)

    monkeypatch.setattr(jtfne, "simulate", recording)
    res = exposure.run_shortened_exposure(duration_s=10, dt_ms=0.5, seed=0, enable_hdp=False)
    assert res["n_trials"] == 2
    assert res["measurement_scope"] == "all_trials"
    assert len(calls) == res["n_trials"], [sorted(c) for c in calls]  # no discarded preliminary run
    assert calls[0].get("continuation") is None
    assert all(c.get("continuation") is not None for c in calls[1:])
    assert all(c.get("return_state") is True for c in calls)


def test_non_finite_earlier_trial_is_detected(monkeypatch):
    seen = {"n": 0}

    def fake_simulate(model, sim, *args, **kwargs):
        i = seen["n"]
        seen["n"] += 1
        sig = _fake_signal(0.0, nan=(i == 0))
        return sig, ("state", i)

    monkeypatch.setattr(jtfne, "simulate", fake_simulate)
    res = exposure.run_shortened_exposure(duration_s=10, dt_ms=0.5, seed=0, enable_hdp=False)
    assert res["n_trials"] == 2
    assert res["finite"] is False
    assert res["non_finite_trials"] == [0]
    assert res["trials"][1]["finite"] is True


def test_exact_state_handoff_between_trials(monkeypatch):
    """Each call after the first receives exactly the state object returned by the previous call."""
    returned = []
    received = []

    def fake_simulate(model, sim, *args, **kwargs):
        received.append(kwargs.get("continuation"))
        sentinel = object()
        returned.append(sentinel)
        return _fake_signal(0.0), sentinel

    monkeypatch.setattr(jtfne, "simulate", fake_simulate)
    exposure.run_shortened_exposure(duration_s=15, dt_ms=0.5, seed=0, enable_hdp=False)
    assert len(returned) >= 3
    assert received[0] is None
    for k in range(1, len(received)):
        assert received[k] is returned[k - 1]


def test_aggregation_over_differing_trials(monkeypatch):
    """Trial 0: V=1, all spiking. Trial 1: V=3, silent. Aggregates must reflect both trials."""
    seen = {"n": 0}

    def fake_simulate(model, sim, *args, **kwargs):
        i = seen["n"]
        seen["n"] += 1
        if i == 0:
            sig = _fake_signal(1.0)
            sig.spikes = jnp.ones((10, 1000), dtype=jnp.float32)
        else:
            sig = _fake_signal(3.0)
        return sig, ("state", i)

    monkeypatch.setattr(jtfne, "simulate", fake_simulate)
    res = exposure.run_shortened_exposure(duration_s=10, dt_ms=0.5, seed=0, enable_hdp=False)
    assert res["n_trials"] == 2
    assert res["finite"] is True
    assert res["v_min"] == 1.0 and res["v_max"] == 3.0 and res["v_mean"] == 2.0
    assert res["n_steps"] == 20
    # rate = mean(spikes) * 1000/dt = 1 * 2000 in trial 0, 0 in trial 1
    assert abs(res["rate_hz_mean"] - 1000.0) < 1e-3
    assert all(abs(r - 1000.0) < 1e-3 for r in res["area_rates_hz"].values())
