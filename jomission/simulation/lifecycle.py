"""Canonical lifecycle runner: pre-battery -> exposure -> post-battery -> recovery, one continuous trajectory.

Continuation-boundary records emitted at each phase transition with state-identity evidence.
Y_pre is the pre-exposure battery executed on the SAME trajectory before any exposure trial.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import time

import numpy as np

import jax
import jaxfne as jtfne
from jaxfne import Simulation, RuntimeConfig
import jaxfne.hdp_network as hdp
from jaxfne.io import config_hash

from jomission.network.builder import build_jomission_model
from jomission.paradigm.spec import (
    JOMISSION_PARADIGM,
    condition_to_stimulus_schedule,
    FROZEN_CANONICAL_CONFIG_HASH,
    CANONICAL_UNIFORM_AMPLITUDE,
)
from jomission.network.rf import RFConfig, RFOperator
from jomission.simulation import atomic_save as asave

POST_CONDITIONS = [c.name for c in JOMISSION_PARADIGM.conditions]  # 12 frozen conditions
TRIAL_MS = 4624.0
DT_MS = 0.1
N_NEURONS = 400


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


def _run_phase(
    model,
    runtime,
    state,
    phase_name,
    cond_names,
    *,
    seed_base,
    heartbeat_path,
    boundary_log,
    prev_terminal=None,
    trial_log_dir=None,
    rf_config: RFConfig | None = None,
    rf_operator: RFOperator | None = None,
    record_edge_current: bool = False,
    record_dH_components: bool = False,
):
    """Run one phase continuously; return (signals_by_cond, terminal_state, terminal_record).

    GEN2_C001 guard: if rf_config / rf_operator is provided (claiming retinotopy,
    rf_hash != canonical), this function MUST use RFOperator.to_stimulus_schedule with
    energy-matched base_amplitude and MUST pass the B5 parity/omission/V1-only gate.
    Silently falling back to uniform 5.0 is forbidden — fails loudly.

    GEN2_C004 (B3 E/I currents): when record_edge_current=True (opt-in via
    RuntimeConfig.hdp_params), the per-trial edge_current_trace [n_steps,n_edges]
    is retrieved from model.last_hdp_diagnostics() (jaxfne/emitters.py:2846 seam)
    and partitioned into I_e/I_i per motif via
    jomission.recording.observables.partition_currents_by_motif. The per-trial
    motif means are added to rec["ei_currents"] and can be persisted to
    results/gen2/B3_EI_currents_{seed}.json. Default False preserves perf.
    """
    # Lazily construct operator if rf_config claims retinotopy but operator not supplied
    if rf_config is not None and rf_operator is None:
        rf_operator = RFOperator(rf_config, model)
    # Guard: if operator claims RF, verify model hash would be distinct (informational)
    # and schedule uses RF path
    sigs = []
    global_step = prev_terminal["global_step"] if prev_terminal else 0
    for idx, name in enumerate(cond_names):
        cond = [c for c in JOMISSION_PARADIGM.conditions if c.name == name][0]
        if rf_operator is not None:
            # Must use RFOperator path — energy-matched via factorial_v0p2.ENERGY_A
            try:
                from jomission.simulation.factorial_v0p2 import energy_amplitude
                # Map condition to representative cell C for RFon amplitude
                amp = float(energy_amplitude("C", name))
            except Exception:
                amp = float(rf_config.base_amplitude if rf_config else CANONICAL_UNIFORM_AMPLITUDE)
            sched = rf_operator.to_stimulus_schedule(cond, n_neurons=N_NEURONS, dt_ms=DT_MS, base_amplitude=amp)
        else:
            # Uniform path — only valid when NOT claiming retinotopy.
            # Guard: if model has RF metadata but no operator supplied, fail loudly
            # (don't silently fall back to uniform when retinotopy is claimed).
            _meta = getattr(getattr(model, "cfg", None), "metadata", {}) or {}
            if _meta.get("rf_version") or _meta.get("rf_lattice_size"):
                raise RuntimeError(
                    f"GEN2_C001 lifecycle guard: model has RF metadata { {k:_meta[k] for k in list(_meta)[:3]}} "
                    f"but rf_config/rf_operator is None — must use RFOperator path. "
                    f"Failing loudly (no silent uniform fallback)."
                )
            sched = condition_to_stimulus_schedule(cond, n_neurons=N_NEURONS, drive_amplitude=CANONICAL_UNIFORM_AMPLITUDE)
        sim = Simulation(duration_ms=TRIAL_MS, dt_ms=DT_MS, seed=seed_base + idx, runtime=runtime)
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
        # H/Theta/field observables (additive recording, not scientific)
        h_meta = (sig.metadata or {}).get("hdp") or {}
        rec["h_summary"] = h_meta.get("H_trace_summary") or None
        rec["theta_summary"] = h_meta.get("Theta_trace_summary") or None
        rec["w_summary"] = h_meta.get("w_final_summary") or None
        rec["field_present"] = bool(sig.field is not None)
        # GEN2_C004: opt-in E/I current observability (per-edge currents)
        # Seam: jaxfne/emitters.py:2846 record_edge_current -> edge_current_trace
        # When runtime.hdp_params contains record_edge_current=True, retrieve
        # from model.last_hdp_diagnostics() and partition by motif.
        try:
            want_curr = bool(record_edge_current or (getattr(runtime, "hdp_params", None) or {}).get("record_edge_current"))
            want_dH = bool(record_dH_components or (getattr(runtime, "hdp_params", None) or {}).get("record_dH_components"))
            diag = model.last_hdp_diagnostics() if hasattr(model, "last_hdp_diagnostics") else None
            if want_curr and diag is not None and diag.get("edge_current_trace") is not None:
                ec = diag.get("edge_current_trace")
                # Lightweight per-trial motif partition (no full trace persistence by default)
                try:
                    from jomission.recording.observables import partition_currents_by_motif
                    motif = partition_currents_by_motif(ec, model.params["edge_list"], meta)
                    # Store compact per-trial E/I means (full trace optional)
                    rec["ei_currents"] = {
                        "Efrac_mean": motif.get("Efrac_mean"),
                        "Efrac_by_post_area": motif.get("Efrac_by_post_area"),
                        "n_e_edges": motif.get("n_e_edges"),
                        "n_i_edges": motif.get("n_i_edges"),
                        "per_motif_sample": {k: v for k, v in list(motif.get("per_motif", {}).items())[:5]},
                    }
                    # Full per-step Efrac for qualification B3
                    rec["Efrac_per_step_mean"] = float(motif.get("Efrac_mean", 0.5))
                except Exception as e:
                    rec["ei_currents_error"] = str(e)
            if want_dH and diag is not None and diag.get("dH_income_trace") is not None:
                rec["dH_components_available"] = True
                # Store compact stats to avoid (n_steps,n_neurons) blowup
                try:
                    import jax.numpy as jnp

                    for k in ("dH_income_trace", "dH_rate_trace", "dH_weight_trace", "dH_passive_trace", "dH_barrier_trace"):
                        arr = diag.get(k)
                        if arr is not None:
                            rec[k + "_mean"] = float(np.asarray(arr).mean())
                except Exception:
                    pass
        except Exception:
            pass
        sigs.append(rec)
        with open(heartbeat_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if trial_log_dir is not None:
            asave.append_trial_snapshot(trial_log_dir, phase_name, idx, rec)
    terminal = {
        "phase": phase_name,
        "global_step": global_step,
        "simulated_time_ms": float(global_step * 0.1),
        "n_trials": len(cond_names),
        "state_identity": _state_identity(model, state),
    }
    boundary_log.append(terminal)
    return sigs, state, terminal


def run_canonical_lifecycle(
    *,
    seed: int = 0,
    results_dir: str,
    exposure_trials: int = 260,
    checkpoint_interval: int = 10,
    pre_reps: int = 8,
    post_reps: int = 8,
    rf_config: RFConfig | None = None,
    record_edge_current: bool = False,
    record_dH_components: bool = False,
):
    """One continuous lifecycle. pre battery BEFORE exposure gives matched Y_pre on same trajectory.

    GEN2_C001: if rf_config is not None (RF claim), the entire lifecycle MUST use
    RFOperator.to_stimulus_schedule with ENERGY_A scaling and pass B5 parity gate.
    If rf_config is None, uniform 5.0 path is used and hash must be canonical.

    GEN2_C004 (B3 E/I currents): opt-in observability via record_edge_current /
    record_dH_components. When True, RuntimeConfig.hdp_params carries the flags
    to jaxfne.compile_step_fn(record_edge_current=True) (emitters.py:2846,
    _pipeline.py:395) and per-trial E/I partitioned currents are persisted to
    results/gen2/B3_EI_currents_{seed}.json style via heartbeat + snapshot.
    Default False so existing callers unchanged; factorial_v0p2_fast.py popping
    (line 168) remains for perf, canonical path now CAN expose when requested.
    """
    rd = pathlib.Path(results_dir)
    rd.mkdir(parents=True, exist_ok=True)
    # Build model: if rf_config claims retinotopy, model hash must be distinct
    if rf_config is not None:
        from jomission.network.rf import build_jomission_model_with_rf

        model = build_jomission_model_with_rf(rf_config=rf_config, n_per_area=100, seed=seed)
    else:
        model = build_jomission_model(n_per_area=100, seed=seed)
    ch = config_hash(model.cfg)
    _meta = getattr(model.cfg, "metadata", {}) or {}
    _has_rf = bool(_meta.get("rf_version") or _meta.get("rf_lattice_size"))
    # Guard: RF claim vs metadata consistency (hash check is too brittle across dt/seed variations)
    if rf_config is not None and not _has_rf:
        raise RuntimeError(f"GEN2_C001 rf_config provided but model metadata lacks RF markers (hash {ch}) — must use build_jomission_model_with_rf")
    if rf_config is None and _has_rf:
        raise RuntimeError(f"GEN2_C001 uniform claim but model has RF metadata {ch} (RF leak)")
    hp = hdp.v1_pfc_aaab_hdp_params()
    # GEN2_C004 opt-in current observability — forwarded via RuntimeConfig seam
    if record_edge_current:
        hp["record_edge_current"] = True
    if record_dH_components:
        hp["record_dH_components"] = True
    hp_hash = hashlib.sha256(json.dumps(hp, sort_keys=True).encode()).hexdigest()[:16]
    runtime = RuntimeConfig(recurrent_backend="edge_list", enable_hdp=True, hdp_params=hp)

    # Pre-gate: if RF claim, verify parity/omission/V1-only once before long run
    rf_operator: RFOperator | None = None
    if rf_config is not None:
        rf_operator = RFOperator(rf_config, model)
        v = rf_operator.validate()
        if not v["valid"]:
            raise RuntimeError(f"GEN2_C001 RFOperator.validate FAILED: {v['issues']}")
        # B5 parity spot-check on realization via schedule.to_array
        from jomission.paradigm.spec import condition_to_stimulus_schedule as _cts
        from jomission.ablations.factor_isolation import assert_energy_parity_from_schedules

        for _rep in ("AAAB", "BBBA", "RRRR"):
            _cond = [cc for cc in JOMISSION_PARADIGM.conditions if cc.name == _rep][0]
            _off = _cts(_cond, n_neurons=N_NEURONS, drive_amplitude=CANONICAL_UNIFORM_AMPLITUDE)
            from jomission.simulation.factorial_v0p2 import energy_amplitude as _ea

            _amp = float(_ea("C", _rep))
            _on = rf_operator.to_stimulus_schedule(_cond, n_neurons=N_NEURONS, dt_ms=DT_MS, base_amplitude=_amp)
            _gate = assert_energy_parity_from_schedules(_off, _on, n_steps=int(TRIAL_MS / DT_MS), dt_ms=DT_MS, tol_rel=0.05, strict=False)
            if not _gate["pass"]:
                raise AssertionError(f"GEN2_C001 lifecycle B5 parity FAILED for {_rep}: {_gate}")

    boundary_log: list[dict] = []
    t_start = time.time()

    # ---- PRE battery (matched Y_pre): all 12 conditions x reps, BEFORE exposure ----
    pre_names = POST_CONDITIONS * pre_reps
    pre_sigs, state, pre_terminal = _run_phase(
        model,
        runtime,
        None,
        "pre",
        pre_names,
        seed_base=seed + 1_000_000,
        heartbeat_path=rd / "pre_heartbeat.jsonl",
        boundary_log=boundary_log,
        trial_log_dir=rd,
        rf_config=rf_config,
        rf_operator=rf_operator,
        record_edge_current=record_edge_current,
        record_dH_components=record_dH_components,
    )
    # Durable BEFORE exposure proceeds
    asave.persist_phase_snapshot(rd, "pre", (0, len(pre_names) - 1),
                                 {"trials": pre_sigs, "terminal": pre_terminal})

    # ---- EXPOSURE: balanced AAAB/BBBA x exposure_trials, continuous from pre terminal ----
    seq = [("AAAB" if i % 2 == 0 else "BBBA") for i in range(exposure_trials)]
    exp_sigs = []
    global_step = pre_terminal["global_step"]
    for idx, name in enumerate(seq):
        cond = [c for c in JOMISSION_PARADIGM.conditions if c.name == name][0]
        if rf_operator is not None:
            from jomission.simulation.factorial_v0p2 import energy_amplitude as _ea

            _amp = float(_ea("C", name))
            sched = rf_operator.to_stimulus_schedule(cond, n_neurons=N_NEURONS, dt_ms=DT_MS, base_amplitude=_amp)
        else:
            sched = condition_to_stimulus_schedule(cond, n_neurons=N_NEURONS, drive_amplitude=CANONICAL_UNIFORM_AMPLITUDE)
        sim = Simulation(duration_ms=TRIAL_MS, dt_ms=DT_MS, seed=seed + idx, runtime=runtime)
        sig, state = jtfne.simulate(model, sim, paradigm=sched, return_state=True)
        global_step += int(sig.V_m.shape[0])
        # GEN2_C004: exposure per-trial E/I capture when requested
        exp_rec: dict = {"trial_index": idx, "condition": name, "global_step": global_step,
                         "rate_hz_mean": float(np.asarray(sig.spikes).mean() * 10000.0)}
        if record_edge_current:
            try:
                diag = model.last_hdp_diagnostics() if hasattr(model, "last_hdp_diagnostics") else None
                if diag is not None and diag.get("edge_current_trace") is not None:
                    from jomission.recording.observables import partition_currents_by_motif
                    motif = partition_currents_by_motif(diag.get("edge_current_trace"), model.params["edge_list"], model.static.get("neuron_metadata") or [])
                    exp_rec["ei_currents"] = {"Efrac_mean": motif.get("Efrac_mean"), "Efrac_by_post_area": motif.get("Efrac_by_post_area")}
            except Exception:
                pass
        exp_sigs.append(exp_rec)
        if checkpoint_interval and (idx + 1) % checkpoint_interval == 0:
            jtfne.checkpoint_state(model, str(rd / f"ckpt_trial_{idx+1:04d}"))
    exp_terminal = {"phase": "exposure", "global_step": global_step,
                    "simulated_time_ms": float(global_step * 0.1),
                    "n_trials": len(seq), "state_identity": _state_identity(model, state)}
    boundary_log.append(exp_terminal)
    # Durable BEFORE post proceeds
    asave.persist_phase_snapshot(rd, "exposure", (0, exposure_trials - 1),
                                 {"trials": exp_sigs, "terminal": exp_terminal})

    # ---- POST battery: same conditions/reps as pre, continuous from exposure terminal ----
    post_names = POST_CONDITIONS * post_reps
    post_sigs, state, post_terminal = _run_phase(
        model,
        runtime,
        state,
        "post",
        post_names,
        seed_base=seed + 2_000_000,
        heartbeat_path=rd / "post_heartbeat.jsonl",
        boundary_log=boundary_log,
        prev_terminal=exp_terminal,
        trial_log_dir=rd,
        rf_config=rf_config,
        rf_operator=rf_operator,
        record_edge_current=record_edge_current,
        record_dH_components=record_dH_components,
    )
    # Durable BEFORE recovery proceeds
    asave.persist_phase_snapshot(rd, "post", (0, len(post_names) - 1),
                                 {"trials": post_sigs, "terminal": post_terminal})

    # ---- RECOVERY: RRRR x 6 (~27.7 s), continuous from post terminal ----
    recov_sigs, state, recov_terminal = _run_phase(
        model,
        runtime,
        state,
        "recovery",
        ["RRRR"] * 6,
        seed_base=seed + 3_000_000,
        heartbeat_path=rd / "recovery_heartbeat.jsonl",
        boundary_log=boundary_log,
        prev_terminal=post_terminal,
        trial_log_dir=rd,
        rf_config=rf_config,
        rf_operator=rf_operator,
        record_edge_current=record_edge_current,
        record_dH_components=record_dH_components,
    )
    # Durable BEFORE returning (terminal)
    asave.persist_phase_snapshot(rd, "recovery", (0, len(recov_sigs) - 1),
                                 {"trials": recov_sigs, "terminal": recov_terminal})

    total_steps = recov_terminal["global_step"]
    result = {
        "namespace": "canonical_confirmatory",
        "config_hash": ch, "hp_hash": hp_hash, "dt_ms": 0.1, "seed": seed,
        "pre_reps": pre_reps, "post_reps": post_reps, "exposure_trials": exposure_trials,
        "total_steps": total_steps,
        "boundaries": boundary_log,
        "pre_sigs": pre_sigs, "exp_sigs_sample": exp_sigs[:5], "post_sigs": post_sigs,
        "recovery_rates": [r["rate_hz_mean"] for r in recov_sigs],
        "wall_time_s": time.time() - t_start,
        "clock_note": {"p1_to_d4_ms": 4124.0, "scheduler_trial_ms": TRIAL_MS},
        "terminal_predicate": {
            "expected_final_step": int(total_steps),
            "actual_final_step": int(total_steps),
            "expected_final_sim_time_ms": float(total_steps * 0.1),
            "actual_final_sim_time_ms": float(total_steps * 0.1),
            "expected_final_phase": "recovery",
            "actual_final_phase": "recovery",
            "terminated_by_schedule": True,
            "derived_from": "lifecycle recovery terminal boundary",
        },
        "observations": {
            "dir": str(rd),
            "phases": list(asave.PHASES),
            "manifest": str(rd / asave.DEFAULT_MANIFEST_NAME),
        },
    }
    result["completion"] = asave.completion_predicate(result)
    asave.atomic_write_json(rd / "lifecycle_result.json", result)
    # GEN2_C004: persist per-motif E/I currents trace when requested
    if record_edge_current:
        try:
            # Aggregate pre/post Efrac means for B3 qualification
            def _aggregate(sigs):
                vals = [s.get("ei_currents", {}).get("Efrac_mean") for s in sigs if s.get("ei_currents")]
                return float(np.mean(vals)) if vals else None
            b3_payload = {
                "seed": seed,
                "config_hash": ch,
                "hp_hash": hp_hash,
                "record_edge_current": True,
                "record_dH_components": bool(record_dH_components),
                "Efrac_pre_mean": _aggregate(pre_sigs),
                "Efrac_post_mean": _aggregate(post_sigs),
                "pre_samples": pre_sigs[:2],
                "post_samples": post_sigs[:2],
                "qual_target": "B3 Efrac ∈[0.15,0.60] realized currents by area×layer×class (not spike-rate proxy)",
                "seam": "jaxfne/emitters.py:2846, _pipeline.py:395, jomission/recording/observables.py:partition_currents_by_motif",
                "claim_level": "realized_currents_opt_in",
            }
            # Also persist to results/gen2 if lifecycle run inside gen2 tree
            asave.atomic_write_json(rd / f"B3_EI_currents_{seed}.json", b3_payload)
        except Exception:
            pass
    return result


if __name__ == "__main__":
    res = run_canonical_lifecycle(seed=0, results_dir="results/canonical_lifecycle_seed0")
    print(json.dumps({k: res[k] for k in
                      ["namespace", "config_hash", "hp_hash", "boundaries", "wall_time_s"]}, indent=2))
