"""Canonical lifecycle runner: pre-battery -> exposure -> post-battery -> recovery, one continuous trajectory.

Continuation-boundary records emitted at each phase transition with state-identity evidence.
Y_pre is the pre-exposure battery executed on the SAME trajectory before any exposure trial.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import time
from dataclasses import replace

import jax
import jax.numpy as jnp
import jaxfne as jtfne
from jaxfne import Simulation, RuntimeConfig
import jaxfne.hdp_network as hdp
from jaxfne.io import config_hash

from jomission.network.builder import build_jomission_model
from jomission.paradigm.spec import JOMISSION_PARADIGM, condition_to_stimulus_schedule

POST_CONDITIONS = [c.name for c in JOMISSION_PARADIGM.conditions]  # 12 frozen conditions
TRIAL_MS = 4624.0


def _state_identity(model, state) -> str:
    """Hash of model params + continuation state leaves — continuation identity evidence."""
    leaves = []
    for arr in jax.tree_util.tree_leaves(model.params):
        leaves.append(np.asarray(arr).tobytes())
    for arr in jax.tree_util.tree_leaves(state):
        try:
            leaves.append(np.asarray(arr).tobytes())
        except Exception:
            pass
    h = hashlib.sha256()
    for b in leaves:
        h.update(b)
    return h.hexdigest()[:16]


def _run_phase(model, runtime, state, phase_name, cond_names, *, seed_base, heartbeat_path,
               boundary_log, prev_terminal=None):
    """Run one phase continuously; return (signals_by_cond, terminal_state, terminal_record)."""
    sigs = []
    global_step = prev_terminal["global_step"] if prev_terminal else 0
    for idx, name in enumerate(cond_names):
        cond = [c for c in JOMISSION_PARADIGM.conditions if c.name == name][0]
        sched = condition_to_stimulus_schedule(cond, n_neurons=400, drive_amplitude=5.0)
        sim = Simulation(duration_ms=TRIAL_MS, dt_ms=0.1, seed=seed_base + idx, runtime=runtime)
        sig, state = jtfne.simulate(model, sim, paradigm=sched, return_state=True)
        global_step += int(sig.V_m.shape[0])
        rec = {
            "phase": phase_name, "trial_index": idx, "condition": name,
            "global_step": global_step,
            "simulated_time_ms": float(global_step * 0.1),
        }
        # compact per-trial observables (area means) for later analysis
        spikes = np.asarray(sig.spikes)
        meta = model.static.get("neuron_metadata") or []
        area_rates = {}
        for area in ("V1", "V4", "FEF", "PFC"):
            ids = [r["neuron_id"] for r in meta if r["area"] == area]
            if ids:
                area_rates[area] = float(spikes[:, ids].mean() * 10000.0)  # dt=0.1ms -> *10000
        rec["area_rates_hz"] = area_rates
        rec["rate_hz_mean"] = float(spikes.mean() * 10000.0)
        sigs.append(rec)
        with open(heartbeat_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
    terminal = {
        "phase": phase_name,
        "global_step": global_step,
        "simulated_time_ms": float(global_step * 0.1),
        "n_trials": len(cond_names),
        "state_identity": _state_identity(model, state),
    }
    boundary_log.append(terminal)
    return sigs, state, terminal


import numpy as np


def run_canonical_lifecycle(*, seed: int = 0, results_dir: str, exposure_trials: int = 260,
                            checkpoint_interval: int = 10, pre_reps: int = 8, post_reps: int = 8):
    """One continuous lifecycle. pre battery BEFORE exposure gives matched Y_pre on same trajectory."""
    rd = pathlib.Path(results_dir)
    rd.mkdir(parents=True, exist_ok=True)
    model = build_jomission_model(n_per_area=100, seed=seed)
    ch = config_hash(model.cfg)
    hp = hdp.v1_pfc_aaab_hdp_params()
    hp_hash = hashlib.sha256(json.dumps(hp, sort_keys=True).encode()).hexdigest()[:16]
    runtime = RuntimeConfig(recurrent_backend="edge_list", enable_hdp=True, hdp_params=hp)

    boundary_log: list[dict] = []
    t_start = time.time()

    # ---- PRE battery (matched Y_pre): all 12 conditions x reps, BEFORE exposure ----
    pre_names = POST_CONDITIONS * pre_reps
    pre_sigs, state, pre_terminal = _run_phase(
        model, runtime, None, "pre", pre_names,
        seed_base=seed + 1_000_000, heartbeat_path=rd / "pre_heartbeat.jsonl",
        boundary_log=boundary_log)
    n_exposed_before = pre_terminal["global_step"]

    # ---- EXPOSURE: balanced AAAB/BBBA x exposure_trials, continuous from pre terminal ----
    seq = [("AAAB" if i % 2 == 0 else "BBBA") for i in range(exposure_trials)]
    exp_sigs = []
    global_step = pre_terminal["global_step"]
    for idx, name in enumerate(seq):
        cond = [c for c in JOMISSION_PARADIGM.conditions if c.name == name][0]
        sched = condition_to_stimulus_schedule(cond, n_neurons=400, drive_amplitude=5.0)
        sim = Simulation(duration_ms=TRIAL_MS, dt_ms=0.1, seed=seed + idx, runtime=runtime)
        sig, state = jtfne.simulate(model, sim, paradigm=sched, return_state=True)
        global_step += int(sig.V_m.shape[0])
        exp_sigs.append({"trial_index": idx, "condition": name, "global_step": global_step,
                         "rate_hz_mean": float(np.asarray(sig.spikes).mean() * 10000.0)})
        if checkpoint_interval and (idx + 1) % checkpoint_interval == 0:
            jtfne.checkpoint_state(model, str(rd / f"ckpt_trial_{idx+1:04d}"))
    exp_terminal = {"phase": "exposure", "global_step": global_step,
                    "simulated_time_ms": float(global_step * 0.1),
                    "n_trials": len(seq), "state_identity": _state_identity(model, state)}
    boundary_log.append(exp_terminal)

    # ---- POST battery: same conditions/reps as pre, continuous from exposure terminal ----
    post_names = POST_CONDITIONS * post_reps
    post_sigs, state, post_terminal = _run_phase(
        model, runtime, state, "post", post_names,
        seed_base=seed + 2_000_000, heartbeat_path=rd / "post_heartbeat.jsonl",
        boundary_log=boundary_log, prev_terminal=exp_terminal)

    # ---- RECOVERY: RRRR x 6 (~27.7 s), continuous from post terminal ----
    recov_sigs, state, recov_terminal = _run_phase(
        model, runtime, state, "recovery", ["RRRR"] * 6,
        seed_base=seed + 3_000_000, heartbeat_path=rd / "recovery_heartbeat.jsonl",
        boundary_log=boundary_log, prev_terminal=post_terminal)

    result = {
        "namespace": "canonical_confirmatory",
        "config_hash": ch, "hp_hash": hp_hash, "dt_ms": 0.1, "seed": seed,
        "pre_reps": pre_reps, "post_reps": post_reps, "exposure_trials": exposure_trials,
        "boundaries": boundary_log,
        "pre_sigs": pre_sigs, "exp_sigs_sample": exp_sigs[:5], "post_sigs": post_sigs,
        "recovery_rates": [r["rate_hz_mean"] for r in recov_sigs],
        "wall_time_s": time.time() - t_start,
        "clock_note": {"p1_to_d4_ms": 4124.0, "scheduler_trial_ms": TRIAL_MS},
    }
    (rd / "lifecycle_result.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    res = run_canonical_lifecycle(seed=0, results_dir="results/canonical_lifecycle_seed0")
    print(json.dumps({k: res[k] for k in
                      ["namespace", "config_hash", "hp_hash", "boundaries", "wall_time_s"]}, indent=2))
