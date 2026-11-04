"""Shortened exposure — exercises multiple H timescales and HDP before production.

Not for scientific phenotype; for numerical/state drift detection.
"""

from __future__ import annotations

import jax.numpy as jnp
import jaxfne as jtfne
from jaxfne import Simulation, RuntimeConfig

from jomission.network.builder import build_jomission_model
from jomission.paradigm.spec import JOMISSION_PARADIGM, condition_to_stimulus_schedule
from jomission.dynamics.h_state import HStateConfig
from jomission.paradigm.epochs import FULL_TRIAL_MS


def run_shortened_exposure(
    *,
    duration_s: float = 120.0,
    dt_ms: float = 0.1,
    seed: int = 0,
    n_per_area: int = 100,
    enable_hdp: bool = True,
) -> dict:
    """Run shortened exposure (structured AAAB/BBBA) to test stability across H timescales.

    duration_s=120 covers τ 0.1,1,10,100 s partially; 300s would cover 1000s partially but is heavier.
    Returns stability metrics, H/HDP ranges, population rates.
    Multi-trial runs aggregate every trial (measurement_scope "all_trials"); finiteness and extremes
    are taken over all trials, and per-trial summaries are kept in "trials".
    """
    model = build_jomission_model(n_per_area=n_per_area, seed=seed)
    # Balanced sequence: repeat AAAB, BBBA
    n_trials = max(1, int(duration_s * 1000 / FULL_TRIAL_MS))
    seq = [("AAAB" if i % 2 == 0 else "BBBA") for i in range(n_trials)]
    # Build combined schedule by concatenating per-trial schedules with time shift
    # For efficiency, simulate as one long continuous run with repeated AAAB/BBBA stimulus
    # Simplest: simulate first trial condition repeatedly; H dynamics will still be exercised
    # Use first condition as representative drive pattern
    rep_cond = [c for c in JOMISSION_PARADIGM.conditions if c.name == "AAAB"][0]
    sched = condition_to_stimulus_schedule(rep_cond, n_neurons=400, drive_amplitude=5.0)
    # Extend schedule by repeating with time offset — use Simulation continuation loop
    runtime = RuntimeConfig(recurrent_backend="edge_list", enable_hdp=bool(enable_hdp), hdp_params={"h_state_dim": 2, "h_state_locality": "population"} if enable_hdp else {}, seed=seed) if enable_hdp else RuntimeConfig(recurrent_backend="edge_list", seed=seed)

    total_ms = float(n_trials * FULL_TRIAL_MS)
    trials: list[dict] = []
    last = None
    if n_trials == 1:
        last = jtfne.simulate(model, Simulation(duration_ms=total_ms, dt_ms=dt_ms, seed=seed, runtime=runtime), paradigm=sched)
        trials.append(_trial_metrics(last, model, dt_ms))
    else:
        # One simulation per trial, carried state between trials. No preliminary run is discarded.
        import dataclasses as _dc
        state = None
        for i, cond_name in enumerate(seq):
            cond = [c for c in JOMISSION_PARADIGM.conditions if c.name == cond_name][0]
            sched_i = condition_to_stimulus_schedule(cond, n_neurons=400, drive_amplitude=5.0)
            rt_i = _dc.replace(runtime, seed=seed + i)  # jaxfne 0.5.8: Simulation.seed must equal runtime.seed
            sim = Simulation(duration_ms=FULL_TRIAL_MS, dt_ms=dt_ms, seed=seed + i, runtime=rt_i)
            if state is None:
                last, state = jtfne.simulate(model, sim, paradigm=sched_i, return_state=True)
            else:
                last, state = jtfne.simulate(model, sim, paradigm=sched_i, continuation=state, return_state=True)
            trials.append(_trial_metrics(last, model, dt_ms))

    import numpy as np
    finite = all(t["finite"] for t in trials)
    non_finite_trials = [i for i, t in enumerate(trials) if not t["finite"]]
    area_names = sorted({a for t in trials for a in t["area_rates_hz"]})
    area_rates = {a: float(np.mean([t["area_rates_hz"][a] for t in trials])) for a in area_names}
    # H/HDP diagnostics if enabled (last trial)
    h_info = last.metadata.get("hdp") or last.metadata.get("homeostasis") or {}
    return {
        "duration_s": float(duration_s),
        "n_trials": int(n_trials),
        "total_ms": float(total_ms),
        "measurement_scope": "all_trials" if n_trials > 1 else "single_trial",
        "rate_hz_mean": float(np.mean([t["rate_hz_mean"] for t in trials])),
        "v_mean": float(np.mean([t["v_mean"] for t in trials])),
        "v_min": float(np.min([t["v_min"] for t in trials])),
        "v_max": float(np.max([t["v_max"] for t in trials])),
        "finite": finite,
        "non_finite_trials": non_finite_trials,
        "trials": trials,
        "area_rates_hz": area_rates,
        "h_info": h_info,
        "h_info_scope": "last_trial",
        "field_present": last.field is not None,
        "n_steps": int(sum(t["n_steps"] for t in trials)),
        "signals_summary": last.summary(),
        "signals_summary_scope": "last_trial",
    }


def _trial_metrics(sig, model, dt_ms) -> dict:
    """Per-trial summary; keeps only scalars and per-area rates, not the trajectories."""
    import numpy as np
    V, spikes, dt = sig.V_m, sig.spikes, float(dt_ms)
    rate_hz = float(jnp.mean(spikes) * (1000.0 / dt)) if dt else 0.0
    area = {}
    meta = model.static.get("neuron_metadata") or []
    if meta:
        spikes_np = np.asarray(spikes)
        for a in ("V1", "V4", "FEF", "PFC"):
            idxs = [r["neuron_id"] for r in meta if r["area"] == a]
            if idxs:
                area[a] = float(np.mean(spikes_np[:, idxs]) * (1000.0 / dt))
    return {
        "finite": bool(jnp.all(jnp.isfinite(V)) and jnp.all(jnp.isfinite(spikes))),
        "rate_hz_mean": rate_hz,
        "v_mean": float(jnp.mean(V)),
        "v_min": float(jnp.min(V)),
        "v_max": float(jnp.max(V)),
        "area_rates_hz": area,
        "n_steps": int(sig.time_ms.shape[0]),
    }
