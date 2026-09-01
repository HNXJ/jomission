"""W4a representation-preserving vertical transfer — L4 -> I_vert -> V_L23 -> R_L23.

Uses same frozen RF A/B assay (AAAB vs BBBA, graded ENERGY_A) and pipeline
record_edge_current, p1 window, LOO centroid decoder, permutation controls.
"""
import json, pathlib, numpy as np, jax, jax.numpy as jnp
from jomission.network.builder import build_jomission_model
from jomission.network.rf import RFConfig, RFOperator
from jomission.paradigm.spec import JOMISSION_PARADIGM
from jaxfne._pipeline import compile_step_fn, run_continuation, continuation_state_from_model
from dataclasses import replace

DT_MS=0.1
DUR_MS=600.0
N_STEPS=int(DUR_MS/DT_MS)
P1_S=int(0/DT_MS); P1_E=int(531/DT_MS)
N_TRIALS_PER_COND=8  # 8 A +8 B =16 total, as before

def build():
    return build_jomission_model(n_per_area=100, seed=0, dt_ms=DT_MS)

def lovo_acc(X,y):
    n=len(y)
    correct=0
    for i in range(n):
        tr=[j for j in range(n) if j!=i]
        Xtr=X[tr]; ytr=y[tr]
        c0=Xtr[ytr==0].mean(axis=0); c1=Xtr[ytr==1].mean(axis=0)
        d0=np.linalg.norm(X[i]-c0); d1=np.linalg.norm(X[i]-c1)
        pred=0 if d0<d1 else 1
        if pred==y[i]:
            correct+=1
    return correct/n

def perm_p(X,y, n_perm=200):
    acc=lovo_acc(X,y)
    null=[]
    for _ in range(n_perm):
        yp=np.random.permutation(y)
        null.append(lovo_acc(X, yp))
    null=np.array(null)
    p=float((np.sum(null>=acc)+1)/(n_perm+1))
    return acc, p, float(null.mean()), float(null.std())

def run():
    model=build()
    tbl=model.neuron_table()
    areas=[r["area"] for r in tbl]; layers=[r["layer"] for r in tbl]; cts=[r["cell_type"] for r in tbl]
    l4e_idx=[i for i,r in enumerate(tbl) if r["area"]=="V1" and r["layer"]=="L4" and r["cell_type"]=="E"]
    l23_idx=[i for i,r in enumerate(tbl) if r["area"]=="V1" and r["layer"]=="L2/3"]
    l23e_idx=[i for i,r in enumerate(tbl) if r["area"]=="V1" and r["layer"]=="L2/3" and r["cell_type"]=="E"]
    l23pv_idx=[i for i,r in enumerate(tbl) if r["area"]=="V1" and r["layer"]=="L2/3" and r["cell_type"]=="PV"]
    l23sst_idx=[i for i,r in enumerate(tbl) if r["area"]=="V1" and r["layer"]=="L2/3" and r["cell_type"]=="SST"]
    l23vip_idx=[i for i,r in enumerate(tbl) if r["area"]=="V1" and r["layer"]=="L2/3" and r["cell_type"]=="VIP"]
    el=model.params["edge_list"]
    pre=np.asarray(el.pre); post=np.asarray(el.post); w=np.asarray(el.weight)

    # RF
    from jomission.simulation.factorial_v0p2 import energy_amplitude as _ea
    rf_cfg=RFConfig(seed=0, tier="graded")
    op=RFOperator(rf_cfg, model)
    condA=[c for c in JOMISSION_PARADIGM.conditions if c.name=="AAAB"][0]
    condB=[c for c in JOMISSION_PARADIGM.conditions if c.name=="BBBA"][0]
    ampA=float(_ea("C","AAAB")); ampB=float(_ea("C","BBBA"))
    schedA=op.to_stimulus_schedule(condA, n_neurons=400, dt_ms=DT_MS, base_amplitude=ampA)
    schedB=op.to_stimulus_schedule(condB, n_neurons=400, dt_ms=DT_MS, base_amplitude=ampB)
    arrA=schedA.to_array(N_STEPS, DT_MS); arrB=schedB.to_array(N_STEPS, DT_MS)

    # Build trials: 8 per condition, different runtime seeds, same RF drive
    trials=[]
    for rep in range(N_TRIALS_PER_COND):
        for label, arr in [("A", arrA), ("B", arrB)]:
            seed = 100+rep*10 + (0 if label=="A" else 1)
            trials.append((label, arr, seed))
    # Also need input vectors: per L4_E mean drive in p1
    # Collect stage vectors
    input_vecs=[]; vm_l4_vecs=[]; spike_l4_vecs=[]
    cur_e_vecs=[]; cur_pv_vecs=[]; cur_sst_vecs=[]; cur_vip_vecs=[]; cur_all_vecs=[]
    vm_l23_vecs=[]; spike_l23_vecs=[]
    labels=[]
    # For vertical OFF control, also collect
    # Prepare vertical mask
    vert_mask=[]
    for ei in range(len(pre)):
        pi=int(pre[ei]); qi=int(post[ei])
        if areas[pi]=="V1" and layers[pi]=="L4" and cts[pi]=="E" and areas[qi]=="V1" and layers[qi]=="L2/3":
            vert_mask.append(ei)
    # For target-summed: need per L2/3 target mapping
    # Build per L2/3 neuron incoming vert edges
    # We'll compute per trial via pipeline edge_current
    for label, arr, seed in trials:
        step_fn, init = compile_step_fn(model, dt_ms=DT_MS, kernel="baseline", record_edge_current=True)
        init_s = continuation_state_from_model(model, seed=seed)
        state, outs = run_continuation(step_fn, init_s, jnp.asarray(arr, dtype=jnp.float32))
        V=np.asarray(outs[0]); S=np.asarray(outs[1]); ec=np.asarray(outs[5])  # (6000,400) etc and (6000,10666)
        # Input vector per L4_E mean drive
        inp = np.array(arr)[P1_S:P1_E, l4e_idx].mean(axis=0)
        vm_l4 = V[P1_S:P1_E, l4e_idx].mean(axis=0)
        spk_l4 = S[P1_S:P1_E, l4e_idx].mean(axis=0)*(1000/DT_MS)
        # Currents: per L2/3 target per class
        # For each L2/3 neuron, sum ec over its incoming vertical edges in p1 window mean
        def target_vec(idxs):
            vec=[]
            for t in idxs:
                es=[ei for ei in vert_mask if int(post[ei])==t]
                if es:
                    # mean over time of sum over edges
                    vec.append(float(ec[P1_S:P1_E, es].sum(axis=1).mean()))
                else:
                    vec.append(0.0)
            return np.array(vec)
        cur_e = target_vec(l23e_idx)
        cur_pv = target_vec(l23pv_idx)
        cur_sst = target_vec(l23sst_idx)
        cur_vip = target_vec(l23vip_idx)
        cur_all = target_vec(l23_idx)
        vm_l23 = V[P1_S:P1_E, l23_idx].mean(axis=0)
        spk_l23 = S[P1_S:P1_E, l23_idx].mean(axis=0)*(1000/DT_MS)
        input_vecs.append(inp); vm_l4_vecs.append(vm_l4); spike_l4_vecs.append(spk_l4)
        cur_e_vecs.append(cur_e); cur_pv_vecs.append(cur_pv); cur_sst_vecs.append(cur_sst); cur_vip_vecs.append(cur_vip); cur_all_vecs.append(cur_all)
        vm_l23_vecs.append(vm_l23); spike_l23_vecs.append(spk_l23)
        labels.append(0 if label=="A" else 1)

    input_vecs=np.array(input_vecs); vm_l4_vecs=np.array(vm_l4_vecs); spike_l4_vecs=np.array(spike_l4_vecs)
    cur_e_vecs=np.array(cur_e_vecs); cur_pv_vecs=np.array(cur_pv_vecs); cur_sst_vecs=np.array(cur_sst_vecs); cur_vip_vecs=np.array(cur_vip_vecs); cur_all_vecs=np.array(cur_all_vecs)
    vm_l23_vecs=np.array(vm_l23_vecs); spike_l23_vecs=np.array(spike_l23_vecs); labels=np.array(labels)
    print(f"trials {len(labels)} input {input_vecs.shape} l4 vm {vm_l4_vecs.shape} l4 spk {spike_l4_vecs.shape} cur_all {cur_all_vecs.shape} vm_l23 {vm_l23_vecs.shape} spk_l23 {spike_l23_vecs.shape}")

    stages={}
    for name, X in [("input",input_vecs),("L4_Vm",vm_l4_vecs),("L4_spike",spike_l4_vecs),("I_vert_E",cur_e_vecs),("I_vert_PV",cur_pv_vecs),("I_vert_SST",cur_sst_vecs),("I_vert_VIP",cur_vip_vecs),("I_vert_all",cur_all_vecs),("L23_Vm",vm_l23_vecs),("L23_spike",spike_l23_vecs)]:
        # handle empty (e.g., VIP may have 0? but we have some)
        if X.shape[1]==0:
            print(f"{name} dim 0 skip")
            continue
        acc,p,null_m,null_s = perm_p(X, labels, n_perm=200)
        # representational distance: mean pairwise Euclidean between A vs B centroids
        c0=X[labels==0].mean(axis=0); c1=X[labels==1].mean(axis=0)
        dist=float(np.linalg.norm(c0-c1))
        # sign/opponent: check per-dimension sign
        # For 9-dim, count positive vs negative deltas
        deltas=c1-c0
        pos=int((deltas>0).sum()); neg=int((deltas<0).sum())
        print(f"{name:12} dim {X.shape[1]:2} acc {acc:.3f} p {p:.3f} null {null_m:.3f}±{null_s:.3f} dist {dist:.3f} pos {pos} neg {neg}")
        stages[name]={"dim": int(X.shape[1]), "acc": float(acc), "p": float(p), "null_m": float(null_m), "null_s": float(null_s), "dist": float(dist), "pos": int(pos), "neg": int(neg)}

    # Vertical OFF control: zero vert edges and re-run one seed per condition, check cur_all acc drops
    w_off=np.array(w, copy=True); w_off[vert_mask]=0.0
    from jaxfne.emitters import EdgeList
    el_off=EdgeList(pre=el.pre, post=el.post, weight=jnp.asarray(w_off, dtype=el.weight.dtype), receptor_index=el.receptor_index, tau_ms=el.tau_ms, delay_steps=el.delay_steps, source_calibration_status=el.source_calibration_status)
    model_off = replace(model, params=dict(model.params, edge_list=el_off))
    # Run 4 trials (2 per condition) with OFF
    cur_all_off=[]
    labels_off=[]
    for rep in range(2):
        for label, arr in [("A", arrA), ("B", arrB)]:
            seed=200+rep*10 + (0 if label=="A" else 1)
            step_fn, init = compile_step_fn(model_off, dt_ms=DT_MS, kernel="baseline", record_edge_current=True)
            init_s = continuation_state_from_model(model_off, seed=seed)
            state, outs = run_continuation(step_fn, init_s, jnp.asarray(arr, dtype=jnp.float32))
            ec=np.asarray(outs[5])
            # target vec
            vec=[]
            for t in l23_idx:
                es=[ei for ei in vert_mask if int(post[ei])==t]
                if es:
                    vec.append(float(ec[P1_S:P1_E, es].sum(axis=1).mean()))
                else:
                    vec.append(0.0)
            cur_all_off.append(np.array(vec)); labels_off.append(0 if label=="A" else 1)
    cur_all_off=np.array(cur_all_off); labels_off=np.array(labels_off)
    if cur_all_off.shape[1]>0:
        acc_off,p_off,_,_=perm_p(cur_all_off, labels_off, n_perm=100)
        print(f"Vertical OFF control cur_all dim {cur_all_off.shape[1]} acc {acc_off:.3f} p {p_off:.3f} (expect chance ~0.5)")
        stages["I_vert_all_OFF"]={"acc": float(acc_off), "p": float(p_off)}
    else:
        print("OFF no dim")

    # Information retention transitions relative to validated L4 spike (1.00)
    # T1: R_L4 (1.00) -> I_target (I_vert_all)
    # T2: I_target -> V_L23
    # T3: V_L23 -> R_L23
    acc_L4=stages.get("L4_spike",{}).get("acc",0)
    acc_I=stages.get("I_vert_all",{}).get("acc",0)
    acc_V=stages.get("L23_Vm",{}).get("acc",0)
    acc_R=stages.get("L23_spike",{}).get("acc",0)
    print(f"Transitions: L4 {acc_L4:.3f} -> I {acc_I:.3f} -> V {acc_V:.3f} -> R {acc_R:.3f}")
    # Classify first failure: threshold 0.75? Use 0.70 as before
    def is_valid(acc): return acc >= 0.75  # high bar for 16 trials? 0.70 is 0.703 at n24, here n16 => 0.75 approx
    if not is_valid(acc_I):
        first="T1 VERTICAL_CURRENT_ENCODING_FAIL"
    elif not is_valid(acc_V):
        first="T2 CURRENT_TO_VM_FAIL"
    elif not is_valid(acc_R):
        first="T3 VM_TO_SPIKE_FAIL"
    elif is_valid(acc_R):
        first="T4 LOCAL_TRANSFER_VALID"
    else:
        first="T_UNRESOLVED"
    print("first representation-space failure:", first)

    # Save
    out=pathlib.Path("results/w4a_vertical_representation.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out,"w") as f:
        json.dump({"stages": stages, "first": first, "n_trials": len(labels)}, f, indent=2)
    print(f"saved to {out}")
    return stages, first

if __name__=="__main__":
    run()
