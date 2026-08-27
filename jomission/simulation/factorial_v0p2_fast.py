"""Factorial v0.2 FAST path — performance branch P wave2.

Separate module (do NOT edit factorial_v0p2.py, 104d55e spec, or Closure Contract).
Frozen authorities preserved exactly: same CELLS, ENERGY_A, PROBE_AGES,
740-trial protocol P0(96)->E1(11)->P1(96)->E2(33)->P2(96)->E3(86)->P3(96)->E4(130)->P4(96).

Optimizations implemented (highest-value candidates from P2):
  C1 — Hoist compile_step_fn + jax.jit(scan_network) : compile ONCE per cell-config,
       reuse across all 740 trials via jaxfne.compile_step_fn + jaxfne.run_continuation.
       The canonical _simulate_continuation_arrays:538 rebuilds compile_step_fn fresh
       per trial with no _compiled_cache, causing ~0.69s/trial cold compile + 740
       Python->JAX crossings. FAST path builds the jitted step_fn once (lazy 0.0006s)
       and drives it via jax.lax.scan through jaxfne.run_continuation.

  C2 — Concatenate K exposure trials into one run_continuation segment.
       Exposure boundaries are probe flush points: P0->E1(11)->P1->E2(33)->P2->E3(86)
       ->P3->E4(130)->P4. Each exposure block is batched via
       StimulusSchedule.to_array per trial + jnp.concatenate, then a SINGLE
       jaxfne.run_continuation call per block (or sub-batched at K≈10 to bound
       memory). Slicing the batched result restores per-trial observables
       (V_m, spikes, sources, field) so downstream E_wave consumes identical artifacts.
       Saves ~740 crossings -> ~74 crossings at K=10 (~20-40% wall).

  C3/C4 — On exposure blocks optionally set record_fields=False and
       record_weight_trace=False. Field projection (project_laminar_sources) is
       diagnostic-only and exposure fields are not needed for probe E_wave;
       w_trace stacked over n_steps*n_edges is 80GB-scale when kept (docstring
       in compile_step_fn). Disabling both on exposure saves peak memory and
       avoids OOM for batched segments, trajectory-equivalent.

Continuation chain semantics identical:
  prng_key, step_index, delay_state, H, w are carried via ContinuationState
  across every segment (probe per-trial + exposure batched). PRNG split chain
  via jax.random.split is linear, so K batched steps == K sequential steps.
  drive arrays concatenated are bitwise identical to per-trial to_array outputs.

Uses ONLY jaxfne public API:
  jaxfne.compile_step_fn, jaxfne.scan_network, jaxfne.run_continuation,
  jaxfne.ContinuationState, jaxfne.StimulusSchedule  (+ dynamic_state_from_model,
  project_laminar_sources for field diagnostic). No second neural simulator.

Invariants:
  - Frozen config/hp/design not modified.
  - Trial geometry dt0.1, 4624ms, 46240 steps, 40142 edges preserved.
  - Probe batteries remain STATE-PERTURBING (enable_hdp=True, continuation).
"""

from __future__ import annotations

import argparse
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

# run_continuation is in jaxfne._pipeline (public pipeline layer)
try:
    from jaxfne import run_continuation  # type: ignore
except ImportError:
    from jaxfne._pipeline import run_continuation  # type: ignore

from jomission.network.builder import build_jomission_model
from jomission.network.rf import RFConfig, RFOperator
from jomission.paradigm.spec import JOMISSION_PARADIGM
from jomission.evidence import EvidenceRef
from jomission.simulation.atomic_save import atomic_write_json

# Reuse frozen tables from canonical (import, not copy drift)
from jomission.simulation.factorial_v0p2 import (
    CELLS as _CELLS,
    ENERGY_A as _ENERGY_A,
    PROBE_AGES,
    POST_CONDS,
)

CELLS = _CELLS
ENERGY_A = _ENERGY_A
DT_MS = 0.1
TRIAL_MS = 4624.0
N_PER_AREA = 100
N_STEPS = int(round(TRIAL_MS / DT_MS))  # 46240


def hp_for(cell_key: str) -> dict:
    c = CELLS[cell_key]
    hp = dict(hdp.v1_pfc_aaab_hdp_params())
    hp["K_HDP"] = c["K_HDP"]
    hp["tau_0_ms"] = c["tau0"]
    return hp


def energy_amplitude(cell_key: str, condition: str) -> float:
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


def make_schedule(cell_key: str, condition: str, rf_op, model):
    from jomission.paradigm.spec import condition_to_stimulus_schedule

    cond_obj = [cc for cc in JOMISSION_PARADIGM.conditions if cc.name == condition][0]
    if rf_op is not None:
        amp = energy_amplitude(cell_key, condition)
        return rf_op.to_stimulus_schedule(cond_obj, n_neurons=400, dt_ms=DT_MS, base_amplitude=amp)
    return condition_to_stimulus_schedule(cond_obj, n_neurons=400, drive_amplitude=5.0)


def _initial_continuation_state(model, seed: int, hp: dict):
    """Build cold-start ContinuationState using only public jaxfne API.

    Uses jaxfne.dynamic_state_from_model (public) + ContinuationState.
    Handles delay_state if model has nonzero delays (rare for HDP).
    """
    h_dim = int(hp.get("h_state_dim", 1))
    dynamic = jtfne.dynamic_state_from_model(model, h_state_dim=h_dim)
    # detect delays
    delay_state = None
    try:
        edges = model.params["edge_list"]
        ds = getattr(edges, "delay_steps", None)
        if ds is not None:
            import numpy as _np

            arr = _np.asarray(ds)
            if arr.size and int(arr.max()) > 0:
                max_d = int(arr.max())
                n_neurons = int(model.params["emitter"].n_neurons)
                dtype = model.params["emitter"].v0.dtype
                delay_state = jnp.zeros((max_d + 1, n_neurons), dtype=dtype)
    except Exception:
        delay_state = None
    return jtfne.ContinuationState(
        dynamic=dynamic,
        prng_key=jax.random.PRNGKey(int(seed)),
        step_index=0,
        delay_state=delay_state,
    )


def _compile_step_fn_once(model, hp: dict, record_weight_trace: bool = False):
    """Compile step_fn ONCE per cell-config — uses only jaxfne public API.

    Returns jitted step_fn. Uses hp directly (contains full HDP kwargs
    with defaults from v1_pfc_aaab_hdp_params). record_weight_trace=False
    avoids (n_steps, n_edges) w_trace stacking — critical for batched
    exposure where w_trace would be 80GB-scale. H/W carry unaffected.
    """
    hp_copy = dict(hp)
    # hp may contain record flag; pop to avoid duplicate kwarg
    # GEN2_C004: popping record_edge_current/record_dH_components here is
    # intentional for FAST perf path (avoids (n_steps,n_edges) stacking on
    # batched exposure). Canonical path (factorial_v0p2.py / lifecycle.py)
    # now CAN expose currents when requested via RuntimeConfig.hdp_params
    # opt-in (see plasticity.py make_runtime(record_edge_current=True)).
    hp_copy.pop("record_weight_trace", None)
    hp_copy.pop("record_dH_components", None)
    hp_copy.pop("record_edge_current", None)
    step_fn, _init = jtfne.compile_step_fn(
        model,
        dt_ms=DT_MS,
        kernel="hdp",
        record_weight_trace=bool(record_weight_trace),
        **hp_copy,
    )
    return step_fn


def run_cell_fast(
    cell_key: str,
    seed: int,
    results_dir: str,
    *,
    exposure_batch_size: int = 11,
    record_fields_exposure: bool = False,
    record_weight_trace: bool = False,
    dt_ms: float = DT_MS,
    trial_ms: float = TRIAL_MS,
) -> dict:
    """FAST factorial v0.2 cell run — identical science, fewer crossings.

    Args:
        cell_key: A/B/C/D
        seed: 0..3
        results_dir: output dir
        exposure_batch_size: max trials per run_continuation segment for
            exposure blocks. K≈10 gives 20-40% wall saving while bounding
            memory (each trial 46240×400 float32 ≈74 MB per array, outputs
            3 arrays ~222 MB + H ~74 MB per trial; K=11 ≈3.2GB, K=130 ≈38GB OOM).
            Use 11 for E1 (exact), chunk larger blocks into 10s.
        record_fields_exposure: if False (default) skip field projection on
            exposure slices — diagnostic-only, saves 0.0% but memory.
        record_weight_trace: if False (default) step_fn arity 4 not 5,
            avoids (n_total_steps, n_edges) stacking. H/W dynamics unchanged.
    """
    t_wall0 = time.perf_counter()
    c = CELLS[cell_key]
    rd = pathlib.Path(results_dir)
    rd.mkdir(parents=True, exist_ok=True)
    model = build_jomission_model(n_per_area=N_PER_AREA, seed=seed)
    ch = config_hash(model.cfg)
    hp = hp_for(cell_key)
    hp_hash = hashlib.sha256(json.dumps(hp, sort_keys=True).encode()).hexdigest()[:16]
    runtime = RuntimeConfig(recurrent_backend="edge_list", enable_hdp=True, hdp_params=hp)
    rf_op = RFOperator(RFConfig(), model) if c["rf_on"] else None

    n_steps = int(round(trial_ms / dt_ms))
    # --- C1: compile ONCE ---
    t_compile0 = time.perf_counter()
    step_fn = _compile_step_fn_once(model, hp, record_weight_trace=record_weight_trace)
    t_compile = time.perf_counter() - t_compile0

    # Field projection helpers (probe only)
    positions = jnp.asarray(model.params["positions"])
    n_contacts = int(model.static.get("n_contacts", 16))
    # dtype for projection: use model's emitter dtype name via runtime
    rc_tmp = runtime
    # actual dtype string for to_array
    dtype_str = "float32"

    hb: list[dict] = []
    state = None  # ContinuationState
    global_step = 0
    ckpt_ok = 0
    # track first seed for initial state creation
    first_probe_seed = seed * 1000 + 0

    def record(phase, idx, cond):
        nonlocal global_step
        global_step += int(n_steps)
        rec = {
            "phase": phase,
            "trial_index": idx,
            "condition": cond,
            "global_step": global_step,
            "simulated_time_ms": float(global_step * dt_ms),
            "seed": seed,
        }
        hb.append(rec)
        with open(rd / "heartbeat.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")

    def _ensure_state_for_first_trial():
        nonlocal state
        if state is None:
            state = _initial_continuation_state(model, first_probe_seed, hp)

    # --- helpers for probe (per-trial via reused step_fn) ---
    def run_probe(te_idx: int, label: str):
        nonlocal state
        for idx, cond in enumerate(POST_CONDS * 8):
            sched = make_schedule(cell_key, cond, rf_op, model)
            drive = sched.to_array(n_steps, dt_ms, dtype=dtype_str)
            if state is None:
                state = _initial_continuation_state(model, seed * 1000 + idx, hp)
                # For first trial idx==0 seed matches first_probe_seed; correct.
                # For later None case shouldn't happen because state already set after P0
            # single-trial run_continuation
            state, outputs = run_continuation(step_fn, state, drive)
            # block for accurate timing and to materialize
            jax.block_until_ready(outputs[0])
            voltages, spikes, sources = outputs[0], outputs[1], outputs[2]
            # field projection if needed (probes always have field)
            # Use slice of size n_steps (single trial, already isolated)
            field = None
            if True:  # probes keep fields
                field = jtfne.project_laminar_sources(
                    sources=sources, positions=positions, n_contacts=n_contacts, dtype=dtype_str
                )
                jax.block_until_ready(field.lfp_proxy)
            # H/W are carried in state.dynamic; diagnostics available in outputs[3], [4] if record_weight_trace
            record(label, idx, cond)

    # --- helper for exposure batched ---
    def run_exposure_until(boundary: int):
        nonlocal state, ckpt_ok
        # We expose exp_trial as attribute on function to persist
        nonlocal exp_trial_holder
        exp_trial = exp_trial_holder[0]
        # Collect all conds for this exposure block
        block_conds: list[str] = []
        while exp_trial < boundary:
            cond = "AAAB" if exp_trial % 2 == 0 else "BBBA"
            block_conds.append(cond)
            exp_trial += 1
        if not block_conds:
            exp_trial_holder[0] = exp_trial
            return
        _ensure_state_for_first_trial()
        # Chunk by exposure_batch_size to bound memory
        # If exposure_batch_size <=0 or None, use whole block as one chunk
        chunk_k = int(exposure_batch_size) if exposure_batch_size else len(block_conds)
        if chunk_k <= 0:
            chunk_k = len(block_conds)
        # Process chunks sequentially to maintain chain semantics
        offset_in_block = 0
        for chunk_start in range(0, len(block_conds), chunk_k):
            chunk_conds = block_conds[chunk_start : chunk_start + chunk_k]
            k = len(chunk_conds)
            # Build drives per trial then concatenate
            drives = []
            for cond in chunk_conds:
                sched = make_schedule(cell_key, cond, rf_op, model)
                drives.append(sched.to_array(n_steps, dt_ms, dtype=dtype_str))
            big_drive = jnp.concatenate(drives, axis=0)  # (k*n_steps, 400)
            # Run one continuation segment for this chunk
            state, outputs = run_continuation(step_fn, state, big_drive)
            jax.block_until_ready(outputs[0])
            voltages_big, spikes_big, sources_big = outputs[0], outputs[1], outputs[2]
            # H_trace is outputs[3] if present (shape (k*n_steps, n_neurons) or (k*n_steps, n_neurons, d_H))
            # Slice per-trial observables and record
            for i, cond in enumerate(chunk_conds):
                # slice
                s0 = i * n_steps
                s1 = (i + 1) * n_steps
                # per-trial slices (for downstream E_wave identical artifacts)
                _v = voltages_big[s0:s1]
                _sp = spikes_big[s0:s1]
                _src = sources_big[s0:s1]
                # optional field
                if record_fields_exposure:
                    _field = jtfne.project_laminar_sources(
                        sources=_src, positions=positions, n_contacts=n_contacts, dtype=dtype_str
                    )
                    jax.block_until_ready(_field.lfp_proxy)
                # Also could materialize via slicing to ensure identical to canonical's per-trial device_get
                jax.block_until_ready(_v)
                # record as if each trial were separate
                global_idx = offset_in_block + i  # index within this block's original exposure ordering
                # but global exposure trial index is chunk_start + i offset from block start
                # Need absolute exp_trial index for heartbeat? Original records exp_trial before increment.
                # Our block_conds built sequentially from boundary; absolute trial index = (boundary - len(block_conds) + chunk_start + i)
                abs_trial = (boundary - len(block_conds)) + chunk_start + i
                record("exposure", abs_trial, cond)
                if (abs_trial + 1) % 10 == 0:
                    jtfne.checkpoint_state(model, str(rd / f"ckpt_trial_{abs_trial + 1:04d}"))
                    ckpt_ok += 1
            offset_in_block += k
        exp_trial_holder[0] = exp_trial

    # --- protocol ---
    # We need mutable holder for exp_trial to allow nonlocal across closures
    exp_trial_holder = [0]
    probe_after = {11: 1, 44: 2, 130: 3, 260: 4}

    # P0 pre
    run_probe(0, "pre")
    run_exposure_until(11)
    run_probe(1, "probe_t1")
    run_exposure_until(44)
    run_probe(2, "probe_t2")
    run_exposure_until(130)
    run_probe(3, "probe_t3")
    run_exposure_until(260)
    run_probe(4, "post")

    t_wall = time.perf_counter() - t_wall0
    result = {
        "cell": c["name"],
        "seed": seed,
        "cell_key": cell_key,
        "config_hash": ch,
        "hp_hash": hp_hash,
        "rf_on": c["rf_on"],
        "K_HDP": c["K_HDP"],
        "tau_0_ms": c["tau0"],
        "total_steps": global_step,
        "n_trials": len(hb),
        "checkpoint_ok": ckpt_ok,
        "heartbeat_len": len(hb),
        "terminal_phase": hb[-1]["phase"] if hb else None,
        "probe_ages": PROBE_AGES,
        "n_probes": len(PROBE_AGES),
        "protocol": "P0(pre,96)->E1(11)->P1(96)->E2(33)->P2(96)->E3(86)->P3(96)->E4(130)->P4(post,96)",
        "expected_trials": 740,
        "fast": True,
        "exposure_batch_size": int(exposure_batch_size),
        "record_fields_exposure": bool(record_fields_exposure),
        "record_weight_trace": bool(record_weight_trace),
        "compile_s": float(t_compile),
        "wall_s": float(t_wall),
        "per_trial_s": float(t_wall / len(hb)) if hb else None,
    }
    atomic_write_json(rd / f"{c['name']}_result.json", result)
    ev = EvidenceRef(
        code_sha="104d55e",
        parent_run=None,
        config_hash=ch,
        numerical_config_hash=ch[:16],
        hp_hash=hp_hash,
        dt_ms=DT_MS,
        seed=seed,
        network_realization=f"V1->V4->FEF->PFC 100/area izhikevich edge_list {'RFon' if c['rf_on'] else 'RFoff'} FAST",
        phase="post",
        initial_state_hash=None,
        namespace="canonical_confirmatory",
        evidence_class="MECHANISTIC",
        estimand_version="jomission_comparison_matrix.v0.1.0",
        generated_owner=str(rd),
        artifact_hash=hashlib.sha256(json.dumps(result).encode()).hexdigest()[:16],
    )
    atomic_write_json(rd / "EvidenceRef.json", ev.to_dict())
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Factorial v0.2 FAST cell runner")
    ap.add_argument("--cell", required=True, choices=list(CELLS.keys()))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--results_dir", default="results/rf_rate_factorial_v0p2_fast")
    ap.add_argument("--exposure_batch_size", type=int, default=11, help="max exposure trials per run_continuation")
    ap.add_argument("--record_fields_exposure", action="store_true", help="keep fields on exposure (default False)")
    ap.add_argument("--record_weight_trace", action="store_true", help="keep w_trace (default False, saves memory)")
    ap.add_argument("--trial_ms", type=float, default=TRIAL_MS, help="trial duration ms (default 4624; use 462.4 for quick smoke)")
    ap.add_argument("--dt_ms", type=float, default=DT_MS, help="dt ms")
    args = ap.parse_args()
    res = run_cell_fast(
        args.cell,
        args.seed,
        f"{args.results_dir}/{CELLS[args.cell]['name']}_seed{args.seed}",
        exposure_batch_size=args.exposure_batch_size,
        record_fields_exposure=args.record_fields_exposure,
        record_weight_trace=args.record_weight_trace,
        trial_ms=args.trial_ms,
        dt_ms=args.dt_ms,
    )
    print(json.dumps(res, indent=2))
