"""W4a realized-mechanism diagnostic — stage-wise S→R_L4→I_vertical→V_L23→R_L23.

Reproduces the generic W4a sensory-transfer assay (RF 32×32 graded ENERGY_A,
not omission) with record_edge_current on baseline delayed path.
No biological delta; only instrumentation.

Stages retained synchronized:
 stimulus, L4_E spikes/rate, I_edge[L4_E→L2/3_E], I_edge inhibitory,
 total E/I, L2/3_E Vm, L2/3_E spikes.
"""
import json, hashlib, pathlib
import numpy as np
import jax.numpy as jnp
import jaxfne as jtfne
from jaxfne import Simulation
from jaxfne._pipeline import compile_step_fn, run_continuation
from jomission.network.builder import build_jomission_model
from jomission.network.rf import RFConfig, RFOperator
from jomission.paradigm.spec import JOMISSION_PARADIGM
from jomission.simulation.runtime import simulation_for_model, model_requires_edge_list

# Fixed identities
SEED = 0
DT_MS = 0.1
DURATION_MS = 600.0  # single trial ~p1-p4, enough for sensory propagation
N_PER_AREA = 100
RF_SEED = 0

def build_model():
    # Use canonical build (no RF in cfg) but RFOperator will generate graded drive
    # to V1 target; this matches W4a's RF 32×32 graded ENERGY_A via RFOperator seam
    return build_jomission_model(n_per_area=N_PER_AREA, seed=SEED, dt_ms=DT_MS)

def run_stage(seed, record=True):
    model = build_model()
    el = model.params["edge_list"]
    tbl = model.neuron_table()
    areas = [r["area"] for r in tbl]
    layers = [r["layer"] for r in tbl]
    cts = [r["cell_type"] for r in tbl]
    n_neurons = 400

    # RF operator for graded sensory drive (W4a used RF 32×32 graded ENERGY_A)
    from jomission.simulation.factorial_v0p2 import energy_amplitude as _ea
    rf_cfg = RFConfig(seed=RF_SEED, tier="graded")
    op = RFOperator(rf_cfg, model)
    # Two sensory conditions: A at p1 vs B at p1 (AAAB vs BBBA) — ordinary sensory, not omission
    condA = [c for c in JOMISSION_PARADIGM.conditions if c.name=="AAAB"][0]
    condB = [c for c in JOMISSION_PARADIGM.conditions if c.name=="BBBA"][0]
    ampA = float(_ea("C", "AAAB"))
    ampB = float(_ea("C", "BBBA"))
    schedA = op.to_stimulus_schedule(condA, n_neurons=n_neurons, dt_ms=DT_MS, base_amplitude=ampA)
    schedB = op.to_stimulus_schedule(condB, n_neurons=n_neurons, dt_ms=DT_MS, base_amplitude=ampB)
    # Also test omission is NOT primary — we keep sensory A/B only

    # Pipeline with record_edge_current
    step_fn, init = compile_step_fn(model, dt_ms=DT_MS, kernel="baseline", record_edge_current=record)
    # Convert schedules to arrays for run_continuation
    def sched_to_array(sched):
        return sched.to_array(int(DURATION_MS/DT_MS), DT_MS)
    arrA = sched_to_array(schedA)
    arrB = sched_to_array(schedB)
    # Matched seeds: same init, same RNG sequence, only drive differs
    from jaxfne._pipeline import continuation_state_from_model
    initA = continuation_state_from_model(model, seed=seed)
    initB = continuation_state_from_model(model, seed=seed)
    stateA, outsA = run_continuation(step_fn, initA, jnp.asarray(arrA, dtype=jnp.float32))
    stateB, outsB = run_continuation(step_fn, initB, jnp.asarray(arrB, dtype=jnp.float32))
    # Also run recording-OFF invariance check (same drive, different flag)
    step_fn_off, init_off = compile_step_fn(model, dt_ms=DT_MS, kernel="baseline", record_edge_current=False)
    init_offA = continuation_state_from_model(model, seed=seed)
    _, outsOff = run_continuation(step_fn_off, init_offA, jnp.asarray(arrA, dtype=jnp.float32))

    # Unpack outs: (v, spikes, sources, H, w, edge_current) when record True, else 5-tuple
    # Baseline H/w are carried unchanged
    vA, spkA = np.asarray(outsA[0]), np.asarray(outsA[1])
    vB, spkB = np.asarray(outsB[0]), np.asarray(outsB[1])
    # edge_current only when record
    ecA = np.asarray(outsA[5]) if record and len(outsA)==6 else None
    ecB = np.asarray(outsB[5]) if record and len(outsB)==6 else None
    vOff = np.asarray(outsOff[0]); spkOff = np.asarray(outsOff[1])

    # Masks
    pre = np.asarray(el.pre); post = np.asarray(el.post)
    # L4_E indices (source)
    l4e_idx = [i for i,(a,l,ct) in enumerate(zip(areas,layers,cts)) if a=="V1" and l=="L4" and ct=="E"]
    l23e_idx = [i for i,(a,l,ct) in enumerate(zip(areas,layers,cts)) if a=="V1" and l=="L2/3" and ct=="E"]
    l23_idx = [i for i,(a,l) in enumerate(zip(areas,layers)) if a=="V1" and l=="L2/3"]
    # Vertical edges V1 L4_E -> V1 L2/3 (all cell types) and specifically E
    vert_mask = []; vert_e_mask = []; vert_i_mask = []
    for ei in range(len(pre)):
        pi=int(pre[ei]); qi=int(post[ei])
        if areas[pi]=="V1" and layers[pi]=="L4" and cts[pi]=="E" and areas[qi]=="V1" and layers[qi]=="L2/3":
            vert_mask.append(ei)
            if cts[qi]=="E":
                vert_e_mask.append(ei)
            else:
                vert_i_mask.append(ei)

    # Stage 1: stimulus Δ (RF drives differ; we report drive energy)
    driveA = op.drive_for_stimulus("stimulus_A")
    driveB = op.drive_for_stimulus("stimulus_B")
    # Stage 2: L4_E rate
    def rate(spk, idx):
        # mean spikes per neuron per ms → Hz, window full duration
        return float(spk[:, idx].mean() * (1000.0/DT_MS))
    # Use p1 window 0-531ms for sensory
    p1_s = int(0/DT_MS); p1_e = int(531/DT_MS)
    # Also whole trial
    rL4_A = rate(spkA[p1_s:p1_e], l4e_idx)
    rL4_B = rate(spkB[p1_s:p1_e], l4e_idx)
    # Stage 3: vertical I_edge
    def ec_stats(ec, mask):
        if ec is None or not mask:
            return {"mean": None, "delta": None, "trace": None}
        trA = ec[:, mask].sum(axis=1)  # sum over edges per time
        # need ecB similarly; caller will handle delta
        return trA
    trA_full = ec_stats(ecA, vert_mask)
    trB_full = ec_stats(ecB, vert_mask)
    trA_e = ec_stats(ecA, vert_e_mask)
    trB_e = ec_stats(ecB, vert_e_mask)
    trA_i = ec_stats(ecA, vert_i_mask)
    trB_i = ec_stats(ecB, vert_i_mask)
    # Means over p1 window
    i_vert_A = float(trA_full[p1_s:p1_e].mean()) if trA_full is not None else None
    i_vert_B = float(trB_full[p1_s:p1_e].mean()) if trB_full is not None else None
    i_e_A = float(trA_e[p1_s:p1_e].mean()) if trA_e is not None else None
    i_e_B = float(trB_e[p1_s:p1_e].mean()) if trB_e is not None else None
    i_i_A = float(trA_i[p1_s:p1_e].mean()) if trA_i is not None else None
    i_i_B = float(trB_i[p1_s:p1_e].mean()) if trB_i is not None else None

    # Stage 4: L2/3 Vm
    vL23_A = float(vA[p1_s:p1_e, l23_idx].mean())
    vL23_B = float(vB[p1_s:p1_e, l23_idx].mean())
    vL23e_A = float(vA[p1_s:p1_e, l23e_idx].mean())
    vL23e_B = float(vB[p1_s:p1_e, l23e_idx].mean())

    # Stage 5: L2/3 rate
    rL23_A = rate(spkA[p1_s:p1_e], l23_idx)
    rL23_B = rate(spkB[p1_s:p1_e], l23_idx)
    rL23e_A = rate(spkA[p1_s:p1_e], l23e_idx)
    rL23e_B = rate(spkB[p1_s:p1_e], l23e_idx)

    # Simple decoder: rate difference d and accuracy via sign
    def effect(a,b, trials=1):
        # With single trial per condition, use per-neuron rate vector for decoder approx
        # Use L4_E vector and L2/3 vector
        # For single replicate, we approximate d as (meanA-meanB)/pooled_sd across neurons in window
        return None

    # For single-seed, report delta and use per-neuron distribution for d
    def cohen_d(arrA, arrB):
        # arr are per-neuron rates
        mA=float(arrA.mean()); mB=float(arrB.mean())
        sA=float(arrA.std()); sB=float(arrB.std())
        pooled=np.sqrt((sA**2+sB**2)/2) if (sA+sB)>0 else 1.0
        return (mA-mB)/pooled if pooled>1e-9 else 0.0

    # Per-neuron rates for L4 and L2/3 in p1 window
    # Compute per-neuron mean spikes in window
    per_neuron_L4_A = spkA[p1_s:p1_e, l4e_idx].mean(axis=0)*(1000/DT_MS)
    per_neuron_L4_B = spkB[p1_s:p1_e, l4e_idx].mean(axis=0)*(1000/DT_MS)
    per_neuron_L23_A = spkA[p1_s:p1_e, l23_idx].mean(axis=0)*(1000/DT_MS)
    per_neuron_L23_B = spkB[p1_s:p1_e, l23_idx].mean(axis=0)*(1000/DT_MS)
    d_L4 = cohen_d(per_neuron_L4_A, per_neuron_L4_B)
    d_L23 = cohen_d(per_neuron_L23_A, per_neuron_L23_B)
    # Simple threshold decoder accuracy: sign of (rate - midpoint) per neuron? For single trial, use population mean
    acc_L4 = 1.0 if (rL4_A - rL4_B) > 0 else 0.0  # trivial with one replicate; will be refined with n_seeds
    acc_L23 = 1.0 if (rL23_A - rL23_B) > 0 else 0.0

    # Pathway control: vertical OFF (zero vertical edges) should make I_vert zero
    # We simulate by zeroing vert edges in copy (not a biological delta, just control)
    # Do not mutate original model; just compute expected: if vert edges zero, ec sum zero

    # Invariance: recording OFF vs ON Vm/spikes identical
    vm_inv = bool(np.allclose(vA, vOff))
    spk_inv = bool(np.allclose(spkA, spkOff))

    result = {
        "seed": seed,
        "dt_ms": DT_MS,
        "duration_ms": DURATION_MS,
        "stimulus": {"A": "AAAB (A at p1)", "B": "BBBA (B at p1)", "drive_energy_A": float(np.sum(driveA)), "drive_energy_B": float(np.sum(driveB))},
        "stages": {
            "L4_E_rate_Hz": {"A": rL4_A, "B": rL4_B, "delta": rL4_A-rL4_B, "d": d_L4, "acc": acc_L4},
            "I_vert_total_pA_proxy": {"A": i_vert_A, "B": i_vert_B, "delta": (i_vert_A - i_vert_B) if i_vert_A is not None else None},
            "I_vert_E_to_L23E": {"A": i_e_A, "B": i_e_B, "delta": (i_e_A - i_e_B) if i_e_A is not None else None, "n_edges": len(vert_e_mask)},
            "I_vert_E_to_L23I": {"A": i_i_A, "B": i_i_B, "delta": (i_i_A - i_i_B) if i_i_A is not None else None, "n_edges": len(vert_i_mask)},
            "L23_Vm_proxy": {"A": vL23_A, "B": vL23_B, "delta": vL23_A-vL23_B, "L23_E_Vm_A": vL23e_A, "L23_E_Vm_B": vL23e_B, "delta_E": vL23e_A-vL23e_B},
            "L23_rate_Hz": {"A": rL23_A, "B": rL23_B, "delta": rL23_A-rL23_B, "d": d_L23, "acc": acc_L23, "L23_E_A": rL23e_A, "L23_E_B": rL23e_B, "delta_E": rL23e_A-rL23e_B},
        },
        "invariance": {"Vm_record_vs_off": vm_inv, "spikes_record_vs_off": spk_inv},
        "masks": {"n_vert_total": len(vert_mask), "n_vert_E": len(vert_e_mask), "n_vert_I": len(vert_i_mask), "n_L4_E": len(l4e_idx), "n_L23": len(l23_idx)},
    }
    return result, model, (ecA, ecB, vA, vB, spkA, spkB)

if __name__ == "__main__":
    # Multi-seed for uncertainty (3 seeds)
    results = []
    for s in [0,1,2]:
        res, _, _ = run_stage(seed=s, record=True)
        results.append(res)
        print(f"seed {s}: L4 D {res['stages']['L4_E_rate_Hz']['delta']:.2f} d {res['stages']['L4_E_rate_Hz']['d']:.2f} | I_vert D {res['stages']['I_vert_total_pA_proxy']['delta']:.4f} | L23 Vm D {res['stages']['L23_Vm_proxy']['delta']:.4f} | L23 rate D {res['stages']['L23_rate_Hz']['delta']:.3f} d {res['stages']['L23_rate_Hz']['d']:.2f} | inv Vm {res['invariance']['Vm_record_vs_off']} spk {res['invariance']['spikes_record_vs_off']}")
    # Aggregate
    deltas = {k: [r["stages"][k]["delta"] for r in results] for k in results[0]["stages"]}
    print("=== aggregate deltas (mean±sd) ===")
    for k, vals in deltas.items():
        print(k, f"{np.mean(vals):.4f} ± {np.std(vals):.4f}")
    # Save
    out = pathlib.Path("results/w4a_realized_diagnostic.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    # Convert for json
    def safe(o):
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        return o
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=safe)
    print(f"saved to {out}")
    # Simple classification
    # Thresholds from w4a seal: acc>0.703 and |d|>0.5 for transfer
    # For this short run, we classify based on d
    avg_d_L4 = np.mean([r["stages"]["L4_E_rate_Hz"]["d"] for r in results])
    avg_delta_L4 = np.mean([r["stages"]["L4_E_rate_Hz"]["delta"] for r in results])
    avg_delta_I = np.mean([r["stages"]["I_vert_total_pA_proxy"]["delta"] for r in results])
    avg_delta_Vm = np.mean([r["stages"]["L23_Vm_proxy"]["delta"] for r in results])
    avg_d_L23 = np.mean([r["stages"]["L23_rate_Hz"]["d"] for r in results])
    print(f"avg d L4 {avg_d_L4:.2f} avg delta L4 {avg_delta_L4:.1f} (strong if |delta|>50)")
    # First failed transition based on realized magnitudes (W4a used delta>50 and d>0.5)
    if abs(avg_delta_L4) > 50 and abs(avg_delta_I) < 1.0:
        first = "A: L4 information -> vertical I_edge fails"
    elif abs(avg_delta_I) > 1.0 and abs(avg_delta_Vm) < 0.5:
        first = "B: I_edge information -> Vm fails"
    elif abs(avg_delta_Vm) > 0.5 and abs(avg_d_L23) < 0.5:
        first = "C: Vm information -> spikes fails"
    elif abs(avg_d_L23) > 0.5:
        first = "D: all local stages carry information"
    else:
        first = "UNRESOLVED"
    print("first failed transition:", first)
