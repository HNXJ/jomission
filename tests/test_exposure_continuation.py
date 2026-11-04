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
