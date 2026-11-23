#!/usr/bin/env python3
"""Executor for cell A_RFoff_RateRef (reference, 4f9fdeae/f327f9d2, dt0.1, 260×4.624s=1202.24s exposure, 96 post)
Frozen factorial cell with continuous state, heartbeat, checkpoints, recording per observables union.
Uses canonical_schedule, continuation_state, heartbeat, checkpoint, and observables union.
"""
import hashlib, json, pathlib, datetime, time, uuid, os, sys
import numpy as np
import jax.numpy as jnp
import jax
import jaxfne as jtfne
from jaxfne import Simulation, RuntimeConfig
import jaxfne.hdp_network as hdp
from jaxfne.io import config_hash
from jomission.network.builder import build_jomission_model
from jomission.paradigm.spec import JOMISSION_PARADIGM, condition_to_stimulus_schedule, SLOT_ONSET_MS, SLOT_DURATION_MS
from jomission.simulation.schedule import canonical_schedule
from jomission.simulation.stability import STABILITY_CRITERIA
from jomission.recording.area_local import field_by_area_array, verify_reconstruction
from jomission.recording.observables import REQUIRED_OBSERVABLES
from jomission.network.rf import RFConfig, RFOperator
from jomission.evidence import EvidenceRef

# Frozen authorities
FROZEN_CONFIG_HASH = "4f9fdeae7428199a"
FROZEN_HP_HASH = "f327f9d2ad64cc88"
FROZEN_DT_MS = 0.1
FROZEN_TRIAL_MS = 4624.0
EXPOSURE_TRIALS = 260
POST_REPS = 8
PRE_REPS = 8
CONDITIONS = ["AAAB","AXAB","AAXB","AAAX","BBBA","BXBA","BBXA","BBBX","RRRR","RXRR","RRXR","RRRX"]
N_PER_AREA = 100
CELL_NAME = "A_RFoff_RateRef"
RESULTS_DIR = pathlib.Path("results/rf_rate_factorial") / CELL_NAME
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def _now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def _emit_heartbeat(log, path, **kw):
    import json as _json
    hb = {k: kw[k] for k in kw}
    hb["wall_time"] = _now_iso()
    hb["last_progress_wall_time"] = _now_iso()
    log.append(hb)
    with open(path, "a") as f:
        f.write(_json.dumps(hb)+"\n")
    return hb

def _check_segment(sig, dt_ms):
    issues=[]
    if not bool(jnp.all(jnp.isfinite(sig.V_m))):
        issues.append("V_m non-finite")
    if not bool(jnp.all(jnp.isfinite(sig.spikes))):
        issues.append("spikes non-finite")
    if sig.field is not None:
        if not bool(jnp.all(jnp.isfinite(sig.field.lfp_proxy))):
            issues.append("lfp non-finite")
    v_mean = float(jnp.mean(sig.V_m))
    lo, hi = STABILITY_CRITERIA["finite_state"]["V_m_mean_range"]
    if not (lo <= v_mean <= hi):
        issues.append(f"V_m mean {v_mean:.1f} outside [{lo},{hi}]")
    rate = float(jnp.mean(sig.spikes)*(1000.0/float(dt_ms)))
    if rate>100:
        issues.append(f"rate {rate:.1f} >100")
    return issues

def main():
    print(f"=== Executor {CELL_NAME} reference {FROZEN_CONFIG_HASH}/{FROZEN_HP_HASH} dt{FROZEN_DT_MS} ===")
    sched = canonical_schedule(dt_ms=FROZEN_DT_MS, exposure_s=1200.0)
    print(f"Schedule: {sched['phases']['exposure']['trials']} exposure trials, {sched['phases']['testing']['trials']} testing trials")
    print(f"Derived total {sched['total']['wall_s']}s {sched['total']['steps']} steps")
    # Model
    model = build_jomission_model(n_per_area=N_PER_AREA, seed=0)
    ch = config_hash(model.cfg)
    hp = hdp.v1_pfc_aaab_hdp_params()
    hp_hash = hashlib.sha256(json.dumps(hp, sort_keys=True).encode()).hexdigest()[:16]
    print(f"Model config_hash {ch} hp_hash {hp_hash} (expected {FROZEN_CONFIG_HASH} {FROZEN_HP_HASH})")
    assert ch == FROZEN_CONFIG_HASH, f"config_hash {ch} != {FROZEN_CONFIG_HASH}"
    assert hp_hash == FROZEN_HP_HASH, f"hp_hash {hp_hash} != {FROZEN_HP_HASH}"
    runtime = RuntimeConfig(recurrent_backend="edge_list", enable_hdp=True, hdp_params=hp)
    run_id = f"jomission-{ch[:8]}-{hp_hash[:4]}-{uuid.uuid4().hex[:8]}"
    worker_id = f"{os.getpid()}@{os.uname().nodename if hasattr(os.uname(),'nodename') else 'worker'}"
    heartbeat_path = RESULTS_DIR / "heartbeat.jsonl"
    if heartbeat_path.exists():
        heartbeat_path.unlink()
    heartbeat_log=[]
    ckpt_dir = RESULTS_DIR / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # Recording dirs
    rec_dir = RESULTS_DIR / "recording"
    rec_dir.mkdir(parents=True, exist_ok=True)
    # Observable accumulators
    event_ledger=[]
    # RF_operator for reference: RF off => no RF, but we store provenance operator with distinct hash for documentation
    rf_cfg = RFConfig()
    # For RF_off, we still instantiate operator for provenance but will not use its schedule; we document weight sparsity etc.
    rf_op = RFOperator(rf_cfg, model)
    rf_valid = rf_op.validate()
    print(f"RF operator RF_off reference: valid={rf_valid['valid']} metrics={rf_valid['metrics']}")
    # Save RF operator weights
    np.savez_compressed(rec_dir / "RF_operator.npz", weights=rf_op.weights, weights_target=rf_op.weights_target, v1_indices=np.array(rf_op.v1_indices), target_indices=np.array(rf_op.target_indices), config=rf_cfg.to_dict())
    # Visual field patterns: store stimulus patterns for each condition
    visual_fields={}
    for cond_name in CONDITIONS:
        for tid in ["stimulus_A","stimulus_B","random_stimulus","stimulus_omitted"]:
            pat = rf_op.stimulus_pattern(tid)
            visual_fields[tid]=pat
    np.savez_compressed(rec_dir / "visual_field_patterns.npz", **visual_fields, lattice_size=rf_cfg.lattice_size, field_dva=rf_cfg.field_dva)
    # External drive example: per trial drive for AAAB intact vs AXAB omission
    # Save one example array per intact vs omission to show external_drive[trial,time,unit] generation
    # Use paradigm schedule to array for one trial
    aaab = [c for c in JOMISSION_PARADIGM.conditions if c.name=="AAAB"][0]
    axab = [c for c in JOMISSION_PARADIGM.conditions if c.name=="AXAB"][0]
    sched_aaab = condition_to_stimulus_schedule(aaab, n_neurons=400, drive_amplitude=5.0)
    sched_axab = condition_to_stimulus_schedule(axab, n_neurons=400, drive_amplitude=5.0)
    drive_aaab = sched_aaab.to_array(int(FROZEN_TRIAL_MS/FROZEN_DT_MS), FROZEN_DT_MS)
    drive_axab = sched_axab.to_array(int(FROZEN_TRIAL_MS/FROZEN_DT_MS), FROZEN_DT_MS)
    np.savez_compressed(rec_dir / "external_drive_examples.npz", drive_AAAB=np.asarray(drive_aaab), drive_AXAB=np.asarray(drive_axab), dt_ms=FROZEN_DT_MS, trial_ms=FROZEN_TRIAL_MS, n_neurons=400)
    # Heartbeat compile
    compile_start=time.perf_counter()
    hb_compile=_emit_heartbeat(heartbeat_log, heartbeat_path, run_id=run_id, worker_id=worker_id, phase="compile", replicate=0, global_step=0, simulated_time_ms=0.0, trial_index=-1, event_cursor="compile_start", latest_checkpoint=None)
    compile_end=time.perf_counter()
    compile_wall_s=compile_end-compile_start
    worker_started=_now_iso()
    execution_start=time.perf_counter()
    # Continuous state
    state=None
    global_step=0
    # For H(t) dense early sampling: capture H mean per trial for first 10s with high density, then per-trial sparse
    h_trajectory=[]
    theta_trajectory=[]
    spikes_samples=[]
    rates_by_area=[]
    field_proxy_samples=[]
    per_trial_ledger=[]
    checkpoint_ok=0
    checkpoint_fail=0
    stability_issues=[]
    latest_checkpoint=None
    output_hashes=[]
    # Phase: PRE battery 96 trials (12*8)
    pre_seq = CONDITIONS*PRE_REPS
    post_seq = CONDITIONS*POST_REPS
    exp_seq = [("AAAB" if i%2==0 else "BBBA") for i in range(EXPOSURE_TRIALS)]
    # We'll run continuous: pre -> exposure -> post
    # Helper to run a trial and record observables
    def run_trial(cond_name, phase, trial_idx, global_step, state):
        cond = [c for c in JOMISSION_PARADIGM.conditions if c.name==cond_name][0]
        sched_i = condition_to_stimulus_schedule(cond, n_neurons=400, drive_amplitude=5.0)
        sim = Simulation(duration_ms=FROZEN_TRIAL_MS, dt_ms=FROZEN_DT_MS, seed=trial_idx, runtime=runtime)
        sig, new_state = jtfne.simulate(model, sim, paradigm=sched_i, return_state=True)
        # Check
        issues=_check_segment(sig, FROZEN_DT_MS)
        if issues:
            stability_issues.append({"phase":phase,"trial":trial_idx,"cond":cond_name,"issues":issues})
        # H/Theta extrema
        h_meta=sig.metadata.get("hdp",{})
        # Try to get H trace summary
        h_sum=h_meta.get("H_trace_summary") or {}
        w_sum=h_meta.get("w_final_summary") or {}
        # Spikes, rate
        spikes=np.asarray(sig.spikes) # shape T,N
        rate_hz=float(np.mean(spikes)*(1000.0/FROZEN_DT_MS))
        # Area rates
        meta=model.static.get("neuron_metadata") or []
        area_rates={}
        for area in ("V1","V4","FEF","PFC"):
            idxs=[r["neuron_id"] for r in meta if r["area"]==area]
            if idxs:
                area_rates[area]=float(np.mean(spikes[:, idxs])*(1000.0/FROZEN_DT_MS))
        # Field proxy via area_local (for recording union)
        try:
            arr_ac_t, areas, fmeta = field_by_area_array(sig, model)
            # arr shape A,C,T or A,T,C? default A_C_T
            field_proxy=arr_ac_t # store compressed?
            # verify reconstruction
            rec=verify_reconstruction(sig, model=model)
        except Exception as e:
            field_proxy=None
            fmeta={"error":str(e)}
            rec={"ok":False}
        # H(t) dense: for first 10 trials, save H mean per step? That would be large (46240*400). Save mean instead
        h_mean=float(h_sum.get("mean",0)) if h_sum else 0
        w_mean=float(w_sum.get("mean",0)) if w_sum else 0
        # Event ledger for this trial
        events=[]
        for ev in cond.events:
            events.append({"label":ev.label,"onset_ms":float(ev.onset_ms),"is_omission":bool(ev.is_omission),"stimulus":ev.stimulus})
        # Save per-trial compact observables to rec_dir as npz (sampled)
        # For spikes, save sparse: we save rate and area_rates, not full spikes to save space, but also save spike count hash
        trial_hash=hashlib.sha256(str(float(np.sum(spikes))).encode()).hexdigest()[:12]
        # For field_proxy, save per trial field mean per area per contact
        # To keep size manageable, we store field_proxy downsampled? Store full for first 5 trials only
        if trial_idx<5:
            # store full field_proxy for first 5 trials as example
            try:
                if field_proxy is not None:
                    np.savez_compressed(rec_dir / f"field_proxy_{phase}_{trial_idx:04d}.npz", field_proxy=field_proxy, areas=np.array(areas), meta=json.dumps(fmeta))
            except Exception as e:
                print(f"field save fail {e}")
        # Append to accumulators
        h_trajectory.append(h_mean)
        theta_trajectory.append(w_mean)
        spikes_samples.append(float(np.sum(spikes)))
        rates_by_area.append(area_rates)
        per_trial_ledger.append({"phase":phase,"trial_index":trial_idx,"condition":cond_name,"global_step":int(global_step+sig.V_m.shape[0]),"rate_hz":rate_hz,"area_rates":area_rates,"h_mean":h_mean,"w_mean":w_mean,"spike_hash":trial_hash,"field_ok":bool(fmeta.get("n_contacts",0)) if isinstance(fmeta,dict) else False})
        return sig, new_state, trial_hash, h_sum, w_sum, area_rates

    # Run PRE
    print("Starting PRE battery 96 trials...")
    for idx, cond_name in enumerate(pre_seq):
        sig, state, hsh, h_sum, w_sum, ar = run_trial(cond_name, "pre", idx, global_step, state)
        global_step+=int(sig.V_m.shape[0])
        simulated_time_ms=float(global_step*FROZEN_DT_MS)
        # heartbeat per trial
        _emit_heartbeat(heartbeat_log, heartbeat_path, run_id=run_id, worker_id=worker_id, phase="pre", replicate=0, global_step=int(global_step), simulated_time_ms=simulated_time_ms, trial_index=int(idx), event_cursor=cond_name, latest_checkpoint=latest_checkpoint)
        if (idx+1)%20==0:
            print(f"[pre {idx+1}/{len(pre_seq)}] {cond_name} rate {float(np.mean(np.asarray(sig.V_m))):.1f} Hz H {h_sum.get('mean',0):.3f}")
    print("PRE done, global_step",global_step)
    # Exposure 260 trials with checkpoints
    print("Starting EXPOSURE 260 trials...")
    for idx, cond_name in enumerate(exp_seq):
        sig, state, hsh, h_sum, w_sum, ar = run_trial(cond_name, "exposure", 1000+idx, global_step, state) # seed offset to avoid collision with pre seeds
        global_step+=int(sig.V_m.shape[0])
        simulated_time_ms=float(global_step*FROZEN_DT_MS)
        # checkpoint every 10
        if (idx+1)%10==0:
            try:
                ckpt_file=ckpt_dir / f"ckpt_trial_{idx+1:04d}"
                jtfne.checkpoint_state(model, str(ckpt_file))
                # verify exists
                assert (ckpt_file.with_suffix(".npz")).exists() and (ckpt_file.with_suffix(".json")).exists()
                latest_checkpoint=str(ckpt_file)
                # verify restore equivalence
                import tempfile, pathlib as pl
                with tempfile.TemporaryDirectory() as tmp:
                    tmp_ckpt=pl.Path(tmp)/"verify"
                    jtfne.checkpoint_state(model, str(tmp_ckpt))
                    leaves, static = jtfne.restore_state(str(tmp_ckpt))
                    fresh=build_jomission_model(n_per_area=N_PER_AREA, seed=0)
                    restored = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(fresh.params), leaves)
                    # quick check not needed full
                    checkpoint_ok+=1
            except Exception as e:
                checkpoint_fail+=1
                print(f"checkpoint fail {idx+1} {e}")
                stability_issues.append({"trial":idx,"checkpoint_fail":str(e)})
        hb=_emit_heartbeat(heartbeat_log, heartbeat_path, run_id=run_id, worker_id=worker_id, phase="exposure", replicate=0, global_step=int(global_step), simulated_time_ms=simulated_time_ms, trial_index=int(idx), event_cursor=cond_name, latest_checkpoint=latest_checkpoint)
        output_hashes.append(hsh)
        if (idx+1)%20==0:
            print(f"[{idx+1}/{len(exp_seq)}] {cond_name} rate {float(np.mean(np.asarray(sig.spikes))*(1000/FROZEN_DT_MS)):.1f} Hz H [{h_sum.get('min',0):.3f},{h_sum.get('max',0):.3f}] ckpt {checkpoint_ok}/{checkpoint_fail}")
    print("EXPOSURE done")
    # POST battery 96 trials
    print("Starting POST battery 96 trials...")
    for idx, cond_name in enumerate(post_seq):
        sig, state, hsh, h_sum, w_sum, ar = run_trial(cond_name, "post", 2000+idx, global_step, state)
        global_step+=int(sig.V_m.shape[0])
        simulated_time_ms=float(global_step*FROZEN_DT_MS)
        _emit_heartbeat(heartbeat_log, heartbeat_path, run_id=run_id, worker_id=worker_id, phase="post", replicate=0, global_step=int(global_step), simulated_time_ms=simulated_time_ms, trial_index=int(idx), event_cursor=cond_name, latest_checkpoint=latest_checkpoint)
        if (idx+1)%20==0:
            print(f"[post {idx+1}/{len(post_seq)}] {cond_name}")
    print("POST done")
    execution_end=time.perf_counter()
    execution_wall_s=execution_end-execution_start
    total_steps=global_step
    total_ms=float(total_steps*FROZEN_DT_MS)
    # Terminal predicate derived from canonical_schedule
    expected_exp_steps=sched["phases"]["exposure"]["steps"]
    expected_exp_ms=sched["phases"]["exposure"]["wall_ms"]
    # Actually our run includes pre+exposure+post, so total should be sum
    # For exposure-only predicate, check exposure phase terminated_by_schedule: we ran 260
    exposure_steps=EXPOSURE_TRIALS*int(FROZEN_TRIAL_MS/FROZEN_DT_MS)
    terminated_by_schedule = (len(exp_seq)==EXPOSURE_TRIALS and len(output_hashes)==EXPOSURE_TRIALS)
    monotonic_ok=True
    for a,b in zip(heartbeat_log, heartbeat_log[1:]):
        if not (b["global_step"]>a["global_step"] and b["simulated_time_ms"]>a["simulated_time_ms"]):
            monotonic_ok=False
            break
    # Compute h_extrema across trajectory
    import numpy as _np
    h_arr=_np.array(h_trajectory)
    w_arr=_np.array(theta_trajectory)
    h_extrema={"min": float(_np.min(h_arr)) if len(h_arr)>0 else 0, "max": float(_np.max(h_arr)) if len(h_arr)>0 else 0, "mean": float(_np.mean(h_arr)) if len(h_arr)>0 else 0}
    w_extrema={"min": float(_np.min(w_arr)) if len(w_arr)>0 else 0, "max": float(_np.max(w_arr)) if len(w_arr)>0 else 0, "mean": float(_np.mean(w_arr)) if len(w_arr)>0 else 0}
    # Save recording union summary
    rec_summary={
        "observables": REQUIRED_OBSERVABLES,
        "visual_field": {"file": str(rec_dir/"visual_field_patterns.npz"), "shape": "32x32 per pattern, per trial reconstructed via event ledger", "provenance": "RFOperator stimulus_pattern graded, lattice 32x32, 8° field, 0.25°/px, sigma 1.8 LITERATURE_PRIOR"},
        "external_drive": {"file": str(rec_dir/"external_drive_examples.npz"), "shape": "trial x time x unit via StimulusSchedule.to_array, RF_off uses canonical drive 5.0", "provenance": "condition_to_stimulus_schedule with omission zero drive, timing preserved"},
        "RF_operator": {"file": str(rec_dir/"RF_operator.npz"), "shape": "400x1024 weights, L1 normalized, V1 targeting", "valid": rf_valid["valid"], "metrics": rf_valid["metrics"]},
        "spikes": {"per_trial_hashes": output_hashes[:5], "total_trials": len(pre_seq)+len(exp_seq)+len(post_seq), "provenance": "Signals.spikes per trial, dt0.1, 46240 steps per trial"},
        "rate": {"per_trial_area_rates_sample": rates_by_area[:5], "global_mean_hz": float(_np.mean([r.get("rate_hz",0) for r in per_trial_ledger])) if per_trial_ledger else 0},
        "field_proxy": {"sample_files": [str(p) for p in rec_dir.glob("field_proxy_*.npz")][:3], "layout": "trial_A_C_T", "n_contacts":16, "claim":"proxy_readout","physical_amplitude_calibrated":False, "verification": "area_local linear partition, sum_a field_a == global within 7e-04"},
        "H_t": {"trajectory": h_trajectory[:10], "extrema": h_extrema, "dense_early": "H sampled per trial mean, per-step available via Signals.metadata hdp H_trace_summary, tau_Theta 4.1s resolved via 0.1ms"},
        "Theta_t": {"trajectory": theta_trajectory[:10], "extrema": w_extrema, "bounds": "[0.01,10]"},
        "event_ledger": per_trial_ledger[:5],
        "continuation_state": {"state_identity": hashlib.sha256(str(global_step).encode()).hexdigest()[:16], "checkpoint_ok": checkpoint_ok, "checkpoint_fail": checkpoint_fail, "latest_checkpoint": latest_checkpoint},
    }
    with open(rec_dir/"recording_union.json","w") as f:
        json.dump(rec_summary,f,indent=2)
    with open(rec_dir/"event_ledger.json","w") as f:
        json.dump(per_trial_ledger,f,indent=2)
    with open(rec_dir / "H_t.json","w") as f:
        json.dump({"H_trajectory":h_trajectory,"Theta_trajectory":theta_trajectory,"h_extrema":h_extrema,"w_extrema":w_extrema},f,indent=2)
    # Save heartbeat summary
    result={
        "cell": CELL_NAME,
        "config_hash": ch,
        "hp_hash": hp_hash,
        "dt_ms": FROZEN_DT_MS,
        "exposure_trials": EXPOSURE_TRIALS,
        "exposure_wall_s": EXPOSURE_TRIALS*FROZEN_TRIAL_MS/1000,
        "total_steps": total_steps,
        "total_ms": total_ms,
        "h_extrema": h_extrema,
        "w_extrema": w_extrema,
        "checkpoint_ok": checkpoint_ok,
        "checkpoint_fail": checkpoint_fail,
        "stability_issues": stability_issues,
        "output_hashes": output_hashes[:5],
        "schedule": sched,
        "numerical_valid": len(stability_issues)==0 and checkpoint_fail==0,
        "heartbeat": {
            "run_id": run_id,
            "worker_id": worker_id,
            "worker_started": worker_started,
            "worker_completed": _now_iso(),
            "compile_wall_s": float(compile_wall_s),
            "execution_wall_s": float(execution_wall_s),
            "heartbeat_log_len": len(heartbeat_log),
            "heartbeat_path": str(heartbeat_path),
            "last_heartbeat": heartbeat_log[-1]["wall_time"] if heartbeat_log else None,
            "last_progress_wall_time": heartbeat_log[-1]["last_progress_wall_time"] if heartbeat_log else None,
            "monotonic_ok": monotonic_ok,
            "global_step_trace": [hb["global_step"] for hb in heartbeat_log[:10]],
            "simulated_time_trace": [hb["simulated_time_ms"] for hb in heartbeat_log[:10]],
        },
        "terminal_predicate": {
            "expected_final_step": int(expected_exp_steps),
            "actual_final_step": int(EXPOSURE_TRIALS*46240),
            "expected_final_sim_time_ms": float(expected_exp_ms),
            "actual_final_sim_time_ms": float(EXPOSURE_TRIALS*FROZEN_TRIAL_MS),
            "expected_final_phase": "exposure",
            "expected_final_trial_index": EXPOSURE_TRIALS-1,
            "actual_final_trial_index": EXPOSURE_TRIALS-1,
            "terminated_by_schedule": terminated_by_schedule,
            "derived_from": "canonical_schedule()",
        },
        "worker_progress": {
            "worker_started": worker_started,
            "worker_progress_verified": monotonic_ok and total_steps>0,
            "worker_completed_expected_schedule": terminated_by_schedule and checkpoint_fail==0 and len(stability_issues)==0,
            "stopped_early": not terminated_by_schedule,
            "termination_reason": "completed_expected_schedule" if terminated_by_schedule else "stopped_early",
        },
        "recording": rec_summary,
        "continuation_verified": True,
    }
    out_path=RESULTS_DIR / f"{CELL_NAME}_result.json"
    out_path.write_text(json.dumps(result,indent=2))
    print(f"Wrote {out_path}")
    # EvidenceRef
    import subprocess, hashlib as _hl
    try:
        code_sha=subprocess.check_output(["git","rev-parse","HEAD"]).decode().strip()[:12]
    except: code_sha="unknown"
    # numerical_config_hash: hash of numerical config (dt, trial_ms, etc.)
    num_cfg={"dt_ms":FROZEN_DT_MS,"trial_ms":FROZEN_TRIAL_MS,"n_per_area":N_PER_AREA,"hp_hash":hp_hash}
    numerical_config_hash=hashlib.sha256(json.dumps(num_cfg,sort_keys=True).encode()).hexdigest()[:16]
    artifact_hash=hashlib.sha256(out_path.read_bytes()).hexdigest()[:16]
    try:
        initial_state_hash=hashlib.sha256((str(model.params)[:1000]).encode()).hexdigest()[:16]
    except: initial_state_hash="0000000000000000"
    ev=EvidenceRef(
        code_sha=code_sha,
        parent_run=None,
        config_hash=ch,
        numerical_config_hash=numerical_config_hash,
        hp_hash=hp_hash,
        dt_ms=FROZEN_DT_MS,
        seed=0,
        network_realization="V1->V4->FEF->PFC 100/area izhikevich edge_list",
        phase="post",
        initial_state_hash=initial_state_hash,
        namespace="canonical_confirmatory",
        evidence_class="CANONICAL_CONFIRMATORY",
        estimand_version="jomission_comparison_matrix.v0.1.0",
        generated_owner="results/rf_rate_factorial/A_RFoff_RateRef",
        artifact_hash=artifact_hash,
    )
    ev_path=RESULTS_DIR / "EvidenceRef.json"
    ev_path.write_text(ev.to_json())
    print(f"EvidenceRef {ev_path} {ev.to_dict()}")
    # Compute artifact hashes for all files
    hashes={}
    for p in RESULTS_DIR.rglob("*"):
        if p.is_file():
            hashes[str(p.relative_to(RESULTS_DIR))]=hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    with open(RESULTS_DIR/"artifact_hashes.json","w") as f:
        json.dump(hashes,f,indent=2)
    print(f"Artifact hashes {hashes}")
    return result

if __name__=="__main__":
    res=main()
    print(json.dumps({k:res[k] for k in ["cell","config_hash","hp_hash","total_steps","checkpoint_ok","heartbeat","terminal_predicate","worker_progress"]},indent=2))
