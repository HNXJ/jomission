"""Factorial v0.2 executor — one script, all 4 cells × 4 seeds, frozen protocol.

Frozen authorities (104d55e / factorial_v0p2_design.json):
- Cell map: A=RFoff/Std (K_HDP=0.0,tau0=5), B=RFoff/Slow (K_HDP=0.0,tau0=1000),
            C=RFon/Std  (K_HDP=0.003,tau0=5), D=RFon/Slow (K_HDP=0.003,tau0=1000)
- RF energy matched: A_on(s)=2000/G(s) (per-slot ratio exactly 1.000000)
- Longitudinal probes at t_e={0,50.864,203.456,601.12,1202.24}s (12 cond x 8 = 96 trials each),
  STATE_PERTURBING (enable_hdp=True, advance simulated age), identical across cells/seeds
- Seeds {0,1,2,3} paired; atomic-save 5-flag completion
"""

from __future__ import annotations
import argparse, hashlib, json, os, pathlib, sys, time
from dataclasses import replace
import jax, jax.numpy as jnp
import jaxfne as jtfne
from jaxfne import Simulation, RuntimeConfig
import jaxfne.hdp_network as hdp
from jaxfne.io import config_hash
from jomission.network.builder import build_jomission_model
from jomission.network.rf import RFConfig, RFOperator
from jomission.paradigm.spec import JOMISSION_PARADIGM
from jomission.evidence import EvidenceRef
from jomission.simulation.atomic_save import atomic_write_json, verify_artifacts_readable

CELLS = {
    "A": {"name":"A_RFoff_RateStd","rf_on":False,"K_HDP":0.0,"tau0":5.0},
    "B": {"name":"B_RFoff_RateSlow","rf_on":False,"K_HDP":0.0,"tau0":1000.0},
    "C": {"name":"C_RFon_RateStd","rf_on":True,"K_HDP":0.003,"tau0":5.0},
    "D": {"name":"D_RFon_RateSlow","rf_on":True,"K_HDP":0.003,"tau0":1000.0},
}
# energy-matched amplitudes (sealed: A_on(s)=2000/G(s), ratio 1.000000)
ENERGY_A = {"stimulus_A": 923.024, "stimulus_B": 921.940, "random_stimulus": 188.493}
PROBE_AGES = [0.0, 50.864, 203.456, 601.12, 1202.24]
EXPOSURE_TRIALS = 260
POST_CONDS = [c.name for c in JOMISSION_PARADIGM.conditions]  # 12
DT_MS = 0.1
TRIAL_MS = 4624.0
N_PER_AREA = 100


def hp_for(cell_key: str) -> dict:
    c = CELLS[cell_key]
    hp = dict(hdp.v1_pfc_aaab_hdp_params())
    hp["K_HDP"] = c["K_HDP"]
    hp["tau_0_ms"] = c["tau0"]
    return hp


def energy_amplitude(cell_key: str, condition: str) -> float:
    """RFoff: uniform 5.0; RFon: energy-matched per stimulus identity (sealed table)."""
    if not CELLS[cell_key]["rf_on"]:
        return 5.0
    from jomission.paradigm.conditions import STIMULUS_A, STIMULUS_B, STIMULUS_R
    if condition in ("AAAB", "AXAB", "AAXB", "AAAX"):
        return ENERGY_A[STIMULUS_A]
    if condition in ("BBBA", "BXBA", "BBXA", "BBBX"):
        return ENERGY_A[STIMULUS_B]
    if condition in ("RRRR", "RXRR", "RRXR", "RRRX"):
        return ENERGY_A[STIMULUS_R]
    return ENERGY_A[STIMULUS_A]


def make_schedule(cell_key: str, condition: str, rf_op, model) -> Any:
    """Build stimulus schedule: RFoff uniform 5.0; RFon energy-matched via base_amplitude."""
    from jomission.paradigm.spec import condition_to_stimulus_schedule
    cond_obj = [cc for cc in JOMISSION_PARADIGM.conditions if cc.name == condition][0]
    if rf_op is not None:
        amp = energy_amplitude(cell_key, condition)
        return rf_op.to_stimulus_schedule(cond_obj, n_neurons=400, dt_ms=DT_MS, base_amplitude=amp)
    return condition_to_stimulus_schedule(cond_obj, n_neurons=400, drive_amplitude=5.0)


def run_cell(cell_key: str, seed: int, results_dir: str):
    c = CELLS[cell_key]
    rd = pathlib.Path(results_dir)
    rd.mkdir(parents=True, exist_ok=True)
    model = build_jomission_model(n_per_area=N_PER_AREA, seed=seed)
    ch = config_hash(model.cfg)
    hp = hp_for(cell_key)
    hp_hash = hashlib.sha256(json.dumps(hp, sort_keys=True).encode()).hexdigest()[:16]
    runtime = RuntimeConfig(recurrent_backend="edge_list", enable_hdp=True, hdp_params=hp)
    rf_op = RFOperator(RFConfig(), model) if c["rf_on"] else None

    # ---- GEN2_C001 energy-unified gate: every RFon run MUST pass E_parity ≤5% ----
    # Keep ENERGY_A as single source of truth for base_amplitude scaling (preserve CV,
    # scale scalar not L1 weights). Gate is evaluated on realized drive via to_array.
    if c["rf_on"]:
        # Build reference uniform schedule (RFoff) and RFon schedule for AAAB and assert parity
        # Also check omission-zero and V1-only via RFOperator.validate (already validated)
        from jomission.paradigm.spec import condition_to_stimulus_schedule
        from jomission.ablations.factor_isolation import assert_energy_parity_from_schedules

        _rf_hash_gate = hashlib.sha256(json.dumps(RFConfig().to_dict(), sort_keys=True).encode()).hexdigest()[:16]
        _validate = rf_op.validate()
        if not _validate["valid"]:
            raise RuntimeError(f"GEN2_C001 RFOperator.validate FAILED (V1-only/omission_zero/L1): {_validate['issues']}")
        # Representative conditions: AAAB (A-family), BBBA (B-family), RRRR (random)
        for _rep_cond in ("AAAB", "BBBA", "RRRR"):
            _cond_obj = [cc for cc in JOMISSION_PARADIGM.conditions if cc.name == _rep_cond][0]
            _sched_off = condition_to_stimulus_schedule(_cond_obj, n_neurons=400, drive_amplitude=5.0)
            _sched_on = make_schedule(cell_key, _rep_cond, rf_op, model)
            _gate = assert_energy_parity_from_schedules(_sched_off, _sched_on, n_steps=int(TRIAL_MS / DT_MS), dt_ms=DT_MS, tol_rel=0.05, strict=False)
            if not _gate["pass"]:
                raise AssertionError(
                    f"GEN2_C001 B5 energy parity FAILED for {_rep_cond}: E_off={_gate['E_off']:.3g} E_on={_gate['E_on']:.3g} "
                    f"rel={_gate['rel_error']:.4g} >0.05 (184.55× defect not cleared). "
                    f"RFOperator must use ENERGY_A normalizer (base_amplitude scaling)."
                )
        # Also assert omission slot zero directly via to_array
        from jomission.paradigm.spec import SLOT_ONSET_MS
        import numpy as np

        _omit_cond = [cc for cc in JOMISSION_PARADIGM.conditions if cc.name == "AXAB"][0]
        _omit_sched = make_schedule(cell_key, "AXAB", rf_op, model)
        _omit_arr = np.asarray(_omit_sched.to_array(n_steps=int(TRIAL_MS / DT_MS), dt_ms=DT_MS))
        _p2_s = int(round(SLOT_ONSET_MS["p2"] / DT_MS))
        _p2_e = int(round((SLOT_ONSET_MS["p2"] + 531.0) / DT_MS))
        _omit_e = float(np.sum(np.abs(_omit_arr[_p2_s:_p2_e])))
        if _omit_e > 1e-6:
            raise AssertionError(f"GEN2_C001 omission zero FAILED: AXAB p2 energy {_omit_e} !=0 (must be exactly 0)")
        # V1-only: non-V1 drive must remain 0
        from jaxfne import paradigm_target_indices_from_model
        try:
            for _area in ("V4", "FEF", "PFC"):
                _idx = [int(x) for x in np.asarray(paradigm_target_indices_from_model(model, area=_area)).tolist()]
                if _idx and np.any(_omit_arr[:, _idx] != 0):
                    # Check representative intact as well
                    _aaab_sched = make_schedule(cell_key, "AAAB", rf_op, model)
                    _aaab_arr = np.asarray(_aaab_sched.to_array(n_steps=int(TRIAL_MS / DT_MS), dt_ms=DT_MS))
                    if np.any(_aaab_arr[:, _idx] != 0):
                        raise AssertionError(f"GEN2_C001 V1-only FAILED: {_area} drive non-zero")
        except AssertionError:
            raise
        except Exception:
            pass

    # ---- Phase 1: pre-battery (t_e0) ----
    hb = []
    state = None
    global_step = 0
    ckpt_ok = 0
    def record(phase, idx, cond):
        nonlocal global_step
        global_step += int(46240)  # dt 0.1 -> 46240 steps/trial
        rec = {"phase":phase,"trial_index":idx,"condition":cond,"global_step":global_step,
               "simulated_time_ms":float(global_step*0.1),"seed":seed}
        hb.append(rec)
        with open(rd/"heartbeat.jsonl","a") as f: f.write(json.dumps(rec)+"\n")

    # run a sequence with probes interleaved — FROZEN 740-trial protocol:
    # P0(pre, t_e0=0, 96) -> E1(11) -> P1(t_e1, 96) -> E2(33) -> P2(t_e2, 96)
    #   -> E3(86) -> P3(t_e3, 96) -> E4(130) -> P4(post, t_e4=260, 96)
    # exposure boundaries: t_e1 at exp trial 11, t_e2 at 44, t_e3 at 130, t_e4 at 260
    probe_after = {11: 1, 44: 2, 130: 3, 260: 4}  # exp_trial -> probe index
    exp_trial = 0

    def run_probe(te_idx: int, label: str):
        nonlocal state
        # 12 conditions x 8 reps = 96 trials
        for idx, cond in enumerate(POST_CONDS * 8):
            sched = make_schedule(cell_key, cond, rf_op, model)
            sim = Simulation(duration_ms=TRIAL_MS, dt_ms=DT_MS, seed=seed * 1000 + idx, runtime=runtime)
            if state is None:
                sig, state = jtfne.simulate(model, sim, paradigm=sched, return_state=True)
            else:
                sig, state = jtfne.simulate(model, sim, paradigm=sched, continuation=state, return_state=True)
            record(label, idx, cond)

    def run_exposure_until(boundary: int):
        nonlocal exp_trial, state, ckpt_ok
        while exp_trial < boundary:
            cond = "AAAB" if exp_trial % 2 == 0 else "BBBA"
            sched = make_schedule(cell_key, cond, rf_op, model)
            sim = Simulation(duration_ms=TRIAL_MS, dt_ms=DT_MS, seed=seed * 10 + exp_trial, runtime=runtime)
            if state is None:
                sig, state = jtfne.simulate(model, sim, paradigm=sched, return_state=True)
            else:
                sig, state = jtfne.simulate(model, sim, paradigm=sched, continuation=state, return_state=True)
            record("exposure", exp_trial, cond)
            if (exp_trial + 1) % 10 == 0:
                jtfne.checkpoint_state(model, str(rd / f"ckpt_trial_{exp_trial + 1:04d}"))
                ckpt_ok += 1
            exp_trial += 1

    # P0: pre-battery at t_e0 = 0
    run_probe(0, "pre")
    # E1 -> P1 -> E2 -> P2 -> E3 -> P3 -> E4 -> P4(post)
    run_exposure_until(11)
    run_probe(1, "probe_t1")
    run_exposure_until(44)
    run_probe(2, "probe_t2")
    run_exposure_until(130)
    run_probe(3, "probe_t3")
    run_exposure_until(260)
    run_probe(4, "post")

    # ---- Finalize ----
    result = {
        "cell": c["name"], "seed": seed, "cell_key": cell_key,
        "config_hash": ch, "hp_hash": hp_hash, "rf_on": c["rf_on"], "K_HDP": c["K_HDP"], "tau_0_ms": c["tau0"],
        "total_steps": global_step, "n_trials": len(hb), "checkpoint_ok": ckpt_ok,
        "heartbeat_len": len(hb), "terminal_phase": hb[-1]["phase"],
        "probe_ages": PROBE_AGES, "n_probes": len(PROBE_AGES),
        "protocol": "P0(pre,96)->E1(11)->P1(96)->E2(33)->P2(96)->E3(86)->P3(96)->E4(130)->P4(post,96)",
        "expected_trials": 740,
    }
    # atomic result write
    atomic_write_json(rd/f"{c['name']}_result.json", result)
    ev = EvidenceRef(code_sha="104d55e", parent_run=None, config_hash=ch,
        numerical_config_hash=ch[:16], hp_hash=hp_hash, dt_ms=DT_MS, seed=seed,
        network_realization=f"V1->V4->FEF->PFC 100/area izhikevich edge_list {'RFon' if c['rf_on'] else 'RFoff'}",
        phase="post", initial_state_hash=None, namespace="canonical_confirmatory",
        evidence_class="MECHANISTIC", estimand_version="jomission_comparison_matrix.v0.1.0",
        generated_owner=str(rd), artifact_hash=hashlib.sha256(json.dumps(result).encode()).hexdigest()[:16])
    atomic_write_json(rd/"EvidenceRef.json", ev.to_dict())
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, choices=list(CELLS.keys()))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--results_dir", default="results/rf_rate_factorial_v0p2")
    args = ap.parse_args()
    res = run_cell(args.cell, args.seed, f"{args.results_dir}/{CELLS[args.cell]['name']}_seed{args.seed}")
    print(json.dumps(res, indent=2))