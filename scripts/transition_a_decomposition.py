"""Transition-A internal decomposition — read-only, no C023.

Decomposes L4_E -> L2/3 vertical transmission into:
 source spikes -> vertically sampled -> delayed arrivals -> syn_state -> weight*syn -> target-summed.
Stratified by target class (E vs PV/SST/VIP) and delay class [20,80,120].
"""
import json, pathlib, numpy as np, jax.numpy as jnp, jax
from jomission.network.builder import build_jomission_model
from jomission.network.rf import RFConfig, RFOperator
from jomission.paradigm.spec import JOMISSION_PARADIGM
from jaxfne.emitters import simulate_edge_recurrent_izhikevich
from dataclasses import replace

DT_MS=0.1
DUR_MS=600.0
N_STEPS=int(DUR_MS/DT_MS)
P1_S=int(0/DT_MS); P1_E=int(531/DT_MS)
SEEDS=[0,1,2]

def cohen_d(a,b):
    # a,b are arrays (n_units,) per-unit means
    ma=float(np.mean(a)); mb=float(np.mean(b))
    sa=float(np.std(a)); sb=float(np.std(b))
    pooled=np.sqrt((sa*sa+sb*sb)/2) if (sa+sb)>0 else 1.0
    return (ma-mb)/pooled if pooled>1e-9 else 0.0

def run_one(seed):
    model=build_jomission_model(n_per_area=100, seed=seed, dt_ms=DT_MS)
    el=model.params["edge_list"]
    tbl=model.neuron_table()
    areas=[r["area"] for r in tbl]; layers=[r["layer"] for r in tbl]; cts=[r["cell_type"] for r in tbl]
    pre=np.asarray(el.pre); post=np.asarray(el.post)
    w=np.asarray(el.weight); delay=np.asarray(el.delay_steps)
    # RF
    from jomission.simulation.factorial_v0p2 import energy_amplitude as _ea
    rf_cfg=RFConfig(seed=0, tier="graded")
    op=RFOperator(rf_cfg, model)
    condA=[c for c in JOMISSION_PARADIGM.conditions if c.name=="AAAB"][0]
    condB=[c for c in JOMISSION_PARADIGM.conditions if c.name=="BBBA"][0]
    ampA=float(_ea("C","AAAB")); ampB=float(_ea("C","BBBA"))
    schedA=op.to_stimulus_schedule(condA, n_neurons=400, dt_ms=DT_MS, base_amplitude=ampA)
    schedB=op.to_stimulus_schedule(condB, n_neurons=400, dt_ms=DT_MS, base_amplitude=ampB)
    arrA=schedA.to_array(N_STEPS, DT_MS)
    arrB=schedB.to_array(N_STEPS, DT_MS)
    em=model.params["emitter"]
    key=jax.random.PRNGKey(seed)
    # Run both with record_edge_current True to get edge_current_trace and presyn
    V_A, S_A, srcA, diagA = simulate_edge_recurrent_izhikevich(em, el, N_STEPS, DT_MS, key, dtype="float32", drive_schedule=jnp.asarray(arrA, dtype=jnp.float32), record_edge_current=True)
    V_B, S_B, srcB, diagB = simulate_edge_recurrent_izhikevich(em, el, N_STEPS, DT_MS, key, dtype="float32", drive_schedule=jnp.asarray(arrB, dtype=jnp.float32), record_edge_current=True)
    S_A=np.asarray(S_A); S_B=np.asarray(S_B)
    ecA=np.asarray(diagA["edge_current_trace"]); ecB=np.asarray(diagB["edge_current_trace"])
    presA=np.asarray(diagA.get("presynaptic_drive_trace", np.zeros_like(ecA)))
    presB=np.asarray(diagB.get("presynaptic_drive_trace", np.zeros_like(ecB)))
    # syn_state = ec / w (avoid div0)
    synA = np.divide(ecA, w[None,:], out=np.zeros_like(ecA), where=w[None,:]!=0)
    synB = np.divide(ecB, w[None,:], out=np.zeros_like(ecB), where=w[None,:]!=0)

    # Masks
    l4e_idx=[i for i,(a,l,ct) in enumerate(zip(areas,layers,cts)) if a=="V1" and l=="L4" and ct=="E"]
    # vertical edges L4_E -> L2/3 per target class
    vert={}; vert_e=[]; vert_i={}
    for ei in range(len(pre)):
        pi=int(pre[ei]); qi=int(post[ei])
        if areas[pi]=="V1" and layers[pi]=="L4" and cts[pi]=="E" and areas[qi]=="V1" and layers[qi]=="L2/3":
            ct=cts[qi]
            vert.setdefault(ct, []).append(ei)
            vert_e.append(ei) if ct=="E" else None
            if ct!="E":
                vert_i.setdefault(ct, []).append(ei)
    # Also total vert
    vert_all = sum(vert.values(), [])
    # Source L4_E outgoing degree to L2/3
    out_deg = {i:0 for i in l4e_idx}
    for ei in vert_all:
        out_deg[int(pre[ei])]+=1
    # Informative L4_E: per-neuron rate delta in p1
    per_n_L4_A = S_A[P1_S:P1_E, l4e_idx].mean(axis=0)*(1000/DT_MS)
    per_n_L4_B = S_B[P1_S:P1_E, l4e_idx].mean(axis=0)*(1000/DT_MS)
    # informative threshold: |delta| >5 Hz or |d|>0.5 per neuron vs pooled? Use delta>10 for strong
    deltas_L4 = per_n_L4_A - per_n_L4_B
    # Use absolute delta >10 Hz as informative (conservative, given mean 379/9≈42 per neuron)
    informative = [l4e_idx[i] for i,d in enumerate(deltas_L4) if abs(d) > 10]
    informative_connected = [n for n in informative if out_deg[n] >0]
    # Coverage
    n_l4=len(l4e_idx); n_conn=sum(1 for v in out_deg.values() if v>0)
    n_inf=len(informative); n_inf_conn=len(informative_connected)
    # outgoing degree distribution
    out_vals=list(out_deg.values())
    # target coverage: which L2/3 neurons receive vertical input
    l23_idx=[i for i,(a,l) in enumerate(zip(areas,layers)) if a=="V1" and l=="L2/3"]
    l23e_idx=[i for i,(a,l,ct) in enumerate(zip(areas,layers,cts)) if a=="V1" and l=="L2/3" and ct=="E"]
    in_deg = {i:0 for i in l23_idx}
    for ei in vert_all:
        in_deg[int(post[ei])]+=1
    in_vals=list(in_deg.values())
    # weight and delay distributions for vertical
    w_vert = w[vert_all] if vert_all else np.array([])
    d_vert = delay[vert_all] if vert_all else np.array([])
    # Also per target class
    # Discriminability at each stage
    # Stage0: source spikes per L4_E (already)
    d0 = cohen_d(per_n_L4_A, per_n_L4_B)
    # Stage1: vertically sampled source spikes: only sources with out_deg>0
    sampled_idx = [i for i in l4e_idx if out_deg[i]>0]
    if sampled_idx:
        per_n_sampled_A = S_A[P1_S:P1_E, sampled_idx].mean(axis=0)*(1000/DT_MS) if len(sampled_idx)>1 else np.array([S_A[P1_S:P1_E, sampled_idx[0]].mean()*(1000/DT_MS)])
        per_n_sampled_B = S_B[P1_S:P1_E, sampled_idx].mean(axis=0)*(1000/DT_MS) if len(sampled_idx)>1 else np.array([S_B[P1_S:P1_E, sampled_idx[0]].mean()*(1000/DT_MS)])
        # handle single element
        if per_n_sampled_A.ndim==0:
            per_n_sampled_A=np.array([float(per_n_sampled_A)])
            per_n_sampled_B=np.array([float(per_n_sampled_B)])
        d1 = cohen_d(per_n_sampled_A, per_n_sampled_B)
    else:
        d1 = 0.0
    # Stage2: delayed arrivals per vertical edge: use presyn trace summed over p1 window per edge
    # presyn is (n_steps, n_edges) delayed spike (0/1)
    pres_per_edge_A = presA[P1_S:P1_E, vert_all].mean(axis=0)*(1000/DT_MS) if vert_all else np.array([0])
    pres_per_edge_B = presB[P1_S:P1_E, vert_all].mean(axis=0)*(1000/DT_MS) if vert_all else np.array([0])
    d2 = cohen_d(pres_per_edge_A, pres_per_edge_B) if vert_all else 0.0
    # Stage3: syn_state per edge
    syn_per_edge_A = synA[P1_S:P1_E, vert_all].mean(axis=0) if vert_all else np.array([0])
    syn_per_edge_B = synB[P1_S:P1_E, vert_all].mean(axis=0) if vert_all else np.array([0])
    d3 = cohen_d(syn_per_edge_A, syn_per_edge_B) if vert_all else 0.0
    # Stage4: weight*syn = edge_current per edge
    ec_per_edge_A = ecA[P1_S:P1_E, vert_all].mean(axis=0) if vert_all else np.array([0])
    ec_per_edge_B = ecB[P1_S:P1_E, vert_all].mean(axis=0) if vert_all else np.array([0])
    d4 = cohen_d(ec_per_edge_A, ec_per_edge_B) if vert_all else 0.0
    # Stage5: target-summed current per L2/3 neuron (sum over incoming vertical edges per time, then mean over window)
    # Compute per target neuron mean current
    # For each L2/3 neuron, sum ec over its incoming vert edges
    # Build per target
    target_cur_A = {}
    target_cur_B = {}
    for ct in ["E","PV","SST","VIP"]:
        idxs = [i for i,(a,l,c) in enumerate(zip(areas,layers,cts)) if a=="V1" and l=="L2/3" and c==ct]
        # edges to these targets
        e_to = {t:[] for t in idxs}
        for ei in vert.get(ct, []):
            e_to[int(post[ei])].append(ei)
        valsA=[]; valsB=[]
        for t in idxs:
            es=e_to[t]
            if es:
                valsA.append(float(ecA[P1_S:P1_E, es].sum(axis=1).mean()))
                valsB.append(float(ecB[P1_S:P1_E, es].sum(axis=1).mean()))
            else:
                valsA.append(0.0); valsB.append(0.0)
        # keep for overall L23
        target_cur_A[ct]=np.array(valsA); target_cur_B[ct]=np.array(valsB)
    # Overall L23
    all_valsA=np.concatenate([v for v in target_cur_A.values()]) if target_cur_A else np.array([0])
    all_valsB=np.concatenate([v for v in target_cur_B.values()]) if target_cur_B else np.array([0])
    d5 = cohen_d(all_valsA, all_valsB)
    # Stratify by delay
    delay_strat={}
    for dval in [20,80,120]:
        mask=[ei for ei in vert_all if int(delay[ei])==dval]
        if mask:
            ec_d_A = ecA[P1_S:P1_E, mask].mean(axis=0) if mask else np.array([0])
            ec_d_B = ecB[P1_S:P1_E, mask].mean(axis=0) if mask else np.array([0])
            delay_strat[dval]= {"n": len(mask), "d": cohen_d(ec_d_A, ec_d_B), "meanA": float(ec_d_A.mean()), "meanB": float(ec_d_B.mean())}
        else:
            delay_strat[dval]={"n":0,"d":0.0}

    return {
        "seed": seed,
        "n_l4": n_l4, "n_conn": n_conn, "frac_conn": n_conn/n_l4 if n_l4 else 0,
        "n_inf": n_inf, "n_inf_conn": n_inf_conn, "frac_inf_conn": n_inf_conn/n_inf if n_inf else 0,
        "out_deg": {"mean": float(np.mean(out_vals)) if out_vals else 0, "max": int(np.max(out_vals)) if out_vals else 0, "zeros": sum(1 for v in out_vals if v==0)},
        "in_deg": {"mean": float(np.mean(in_vals)) if in_vals else 0, "max": int(np.max(in_vals)) if in_vals else 0, "zeros": sum(1 for v in in_vals if v==0)},
        "w_vert": {"mean": float(w_vert.mean()) if len(w_vert) else 0, "std": float(w_vert.std()) if len(w_vert) else 0, "min": float(w_vert.min()) if len(w_vert) else 0, "max": float(w_vert.max()) if len(w_vert) else 0},
        "d_vert": {"counts": {int(k): int((d_vert==k).sum()) for k in [20,80,120]}},
        "discriminability": {"d0_source": d0, "d1_sampled": d1, "d2_delayed": d2, "d3_syn": d3, "d4_weighted": d4, "d5_target": d5, "delay_strat": delay_strat},
        "target_cur": {k: {"meanA": float(v.mean()), "meanB": float(target_cur_B[k].mean()), "delta": float(v.mean()-target_cur_B[k].mean())} for k,v in target_cur_A.items()},
        "raw": {"per_n_L4_A": per_n_L4_A.tolist()[:3], "per_n_L4_B": per_n_L4_B.tolist()[:3]},
        "informative_idx": informative[:5],
        "vert_all_n": len(vert_all),
    }

if __name__=="__main__":
    results=[]
    for s in SEEDS:
        r=run_one(s)
        results.append(r)
        print(f"seed {s}: n_l4 {r['n_l4']} conn {r['n_conn']} frac {r['frac_conn']:.2f} | n_inf {r['n_inf']} inf_conn {r['n_inf_conn']} frac {r['frac_inf_conn']:.2f} | out_deg mean {r['out_deg']['mean']:.2f} max {r['out_deg']['max']} zeros {r['out_deg']['zeros']} | in_deg mean {r['in_deg']['mean']:.2f} max {r['in_deg']['max']} zeros {r['in_deg']['zeros']} | w mean {r['w_vert']['mean']:.4f} | d0 {r['discriminability']['d0_source']:.2f} d1 {r['discriminability']['d1_sampled']:.2f} d2 {r['discriminability']['d2_delayed']:.2f} d3 {r['discriminability']['d3_syn']:.2f} d4 {r['discriminability']['d4_weighted']:.2f} d5 {r['discriminability']['d5_target']:.2f}")
        print(f"  delay strat: {r['discriminability']['delay_strat']}")
        print(f"  target cur E {r['target_cur']['E']} PV {r['target_cur'].get('PV', {})} SST {r['target_cur'].get('SST', {})}")
    # Aggregate
    print("=== aggregate ===")
    for k in ["d0_source","d1_sampled","d2_delayed","d3_syn","d4_weighted","d5_target"]:
        vals=[r["discriminability"][k] for r in results]
        print(k, f"{np.mean(vals):.3f} ± {np.std(vals):.3f}")
    # Classification logic
    # A1 structural if frac_inf_conn low or out_deg zeros high or in_deg zeros high
    # A2 weight if w small or d4 << d3
    # A3 delay if d2 << d1
    # A4 syn if d3 << d2
    # Check evidence
    frac_inf = np.mean([r["frac_inf_conn"] for r in results])
    d0=np.mean([r["discriminability"]["d0_source"] for r in results])
    d1=np.mean([r["discriminability"]["d1_sampled"] for r in results])
    d2=np.mean([r["discriminability"]["d2_delayed"] for r in results])
    d3=np.mean([r["discriminability"]["d3_syn"] for r in results])
    d4=np.mean([r["discriminability"]["d4_weighted"] for r in results])
    d5=np.mean([r["discriminability"]["d5_target"] for r in results])
    print(f"frac_inf_conn {frac_inf:.2f} d0 {d0:.2f} d1 {d1:.2f} d2 {d2:.2f} d3 {d3:.2f} d4 {d4:.2f} d5 {d5:.2f}")
    # Evidence
    # A1: if informative sources not connected
    # A2: if d4 much smaller than d3 (weight attenuates) or w small
    # A3: if d2 << d1 (delay decoherence)
    # A4: if d3 << d2 (syn filtering)
    # Decide
    if frac_inf < 0.5:
        cand="A1 STRUCTURAL_SAMPLING_LIMITED"
    elif d2 < d1*0.5:
        cand="A3 DELAY_COHERENCE_LIMITED"
    elif d3 < d2*0.5:
        cand="A4 SYNAPTIC_STATE_LIMITED"
    elif d4 < d3*0.5:
        cand="A2 WEIGHT_LIMITED"
    elif d5 < d4*0.5:
        cand="A5 MIXED (summation cancellation)"
    else:
        cand="A_UNRESOLVED"
    print("candidate", cand)
    # Save
    import pathlib, json
    out=pathlib.Path("results/transition_a_decomposition.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out,"w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)
    print(f"saved to {out}")
