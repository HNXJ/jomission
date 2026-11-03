"""Multi-trial exposure carries state between trials: one simulate call per trial, resumed from the previous state."""

import jaxfne as jtfne

from jomission.simulation import exposure


def test_multi_trial_resumes_state_with_one_call_per_trial(monkeypatch):
    calls = []
    original = jtfne.simulate

    def recording(model, sim, *args, **kwargs):
        calls.append(kwargs)
        return original(model, sim, *args, **kwargs)

    monkeypatch.setattr(jtfne, "simulate", recording)
    res = exposure.run_shortened_exposure(duration_s=10, dt_ms=0.5, seed=0, enable_hdp=False)
    assert res["n_trials"] == 2
    # One full-length simulate, then one simulate per trial.
    assert len(calls) == 1 + res["n_trials"], [sorted(c) for c in calls]
    trial_calls = calls[1:]
    assert trial_calls[0].get("continuation") is None
    assert all(c.get("continuation") is not None for c in trial_calls[1:])
    assert all(c.get("return_state") is True for c in trial_calls)
