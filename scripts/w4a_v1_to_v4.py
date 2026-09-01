"""W4a V1->V4 FF inter-area representation transfer — read-only, no C023."""
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
N_TRIALS_PER_COND=8

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
    model=build_jomission_model(n_per_area=100, seed=0, dt_ms=DT_MS)
    tbl=model.neuron_table()
    areas=[r["area"] for r in tbl]; layers=[r["layer"] for r in tbl]; cts=[r["cell_type"] for r in tbl]
    el=model.params["edge_list"]
    pre=np.asarray(el.pre); post=np.asarray(el.post); w=np.asarray(el.weight); delay=np.asarray(el.delay_steps)
    # Source V1 L2/3_E, target V4 L4 per class
    src_idx=[i for i,r in enumerate(tbl) if r["area"]=="V1" and r["layer"]=="L2/3" and r["cell_type"]=="E"]
    tgt_idx=[i for i,r in enumerate(tbl) if r["area"]=="V4" and r["layer"]=="L4"]
    tgt_e=[i for i in tgt_idx if cts[i]=="E"]
    tgt_pv=[i for i in tgt_idx if cts[i]=="PV"]
    tgt_sst=[i for i in tgt_idx if cts[i]=="SST"]
    tgt_vip=[i for i in tgt_idx if cts[i]=="VIP"]
    # FF edges V1->V4
    ff_mask=[]
    for ei in range(len(pre)):
        pi=int(pre[ei]); qi=int(post[ei])
        if areas[pi]=="V1" and layers[pi]=="L2/3" and cts[pi]=="E" and areas[qi]=="V4" and layers[qi]=="L4":
            ff_mask.append(ei)
    print(f"FF V1 L2/3_E -> V4 L4 edges {len(ff_mask)} delay unique {sorted(set(delay[ff_mask].tolist())) if ff_mask else []} w mean {w[ff_mask].mean():.4f} if any")
    # Structural audit
    out_deg={i:0 for i in src_idx}
    for ei in ff_mask:
        out_deg[int(pre[ei])]+=1
    in_deg={i:0 for i in tgt_idx}
    for ei in ff_mask:
        in_deg[int(post[ei])]+=1
    print(f"structural: n_src {len(src_idx)} n_tgt {len(tgt_idx)} (E {len(tgt_e)} PV {len(tgt_pv)} SST {len(tgt_sst)} VIP {len(tgt_vip)})")
    print(f"out_deg mean {np.mean(list(out_deg.values())):.2f} max {max(out_deg.values()) if out_deg else 0} zeros {sum(1 for v in out_deg.values() if v==0)}")
    print(f"in_deg mean {np.mean(list(in_deg.values())):.2f} max {max(list(in_deg.values())) if in_deg else 0} zeros {sum(1 for v in in_deg.values() if v==0)}")
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
    trials=[]
    for rep in range(N_TRIALS_PER_COND):
        for label, arr in [("A", arrA), ("B", arrB)]:
            seed=100+rep*10 + (0 if label=="A" else 1)
            trials.append((label, arr, seed))
    # Collect stage vectors
    v1_spike_vecs=[]; cur_e_vecs=[]; cur_all_vecs=[]; vm_v4_vecs=[]; spike_v4_vecs=[]
    labels=[]
    # Also need delayed presyn, syn etc for decomposition if F1
    for label, arr, seed in trials:
        step_fn, init = compile_step_fn(model, dt_ms=DT_MS, kernel="baseline", record_edge_current=True)
        init_s = continuation_state_from_model(model, seed=seed)
        state, outs = run_continuation(step_fn, init_s, jnp.asarray(arr, dtype=jnp.float32))
        V=np.asarray(outs[0]); S=np.asarray(outs[1]); ec=np.asarray(outs[5])
        # V1 L2/3_E spike vector
        spk_v1 = S[P1_S:P1_E, src_idx].mean(axis=0)*(1000/DT_MS)
        # I_FF per V4 L4 target
        def target_vec(idxs):
            vec=[]
            for t in idxs:
                es=[ei for ei in ff_mask if int(post[ei])==t]
                if es:
                    vec.append(float(ec[P1_S:P1_E, es].sum(axis=1).mean()))
                else:
                    vec.append(0.0)
            return np.array(vec)
        cur_all = target_vec(tgt_idx)
        cur_e = target_vec(tgt_e)
        vm_v4 = V[P1_S:P1_E, tgt_idx].mean(axis=0) if tgt_idx else np.array([0])
        spk_v4 = S[P1_S:P1_E, tgt_idx].mean(axis=0)*(1000/DT_MS) if tgt_idx else np.array([0])
        v1_spike_vecs.append(spk_v1); cur_all_vecs.append(cur_all); cur_e_vecs.append(cur_e)
        vm_v4_vecs.append(vm_v4); spike_v4_vecs.append(spk_v4)
        labels.append(0 if label=="A" else 1)
    v1_spike_vecs=np.array(v1_spike_vecs); cur_all_vecs=np.array(cur_all_vecs); cur_e_vecs=np.array(cur_e_vecs)
    vm_v4_vecs=np.array(vm_v4_vecs); spike_v4_vecs=np.array(spike_v4_vecs); labels=np.array(labels)
    print(f"trials {len(labels)} v1_spike {v1_spike_vecs.shape} cur_all {cur_all_vecs.shape} vm_v4 {vm_v4_vecs.shape} spk_v4 {spike_v4_vecs.shape}")

    stages={}
    for name, X in [("V1_L23_spike", v1_spike_vecs), ("I_FF_all", cur_all_vecs), ("I_FF_E", cur_e_vecs), ("V4_L4_Vm", vm_v4_vecs), ("V4_L4_spike", spike_v4_vecs)]:
        if X.shape[1]==0:
            print(f"{name} dim 0 skip"); continue
        acc,p,null_m,null_s = perm_p(X, labels, n_perm=200)
        c0=X[labels==0].mean(axis=0); c1=X[labels==1].mean(axis=0)
        dist=float(np.linalg.norm(c0-c1))
        deltas=c1-c0; pos=int((deltas>0).sum()); neg=int((deltas<0).sum())
        print(f"{name:12} dim {X.shape[1]:2} acc {acc:.3f} p {p:.3f} null {null_m:.3f}±{null_s:.3f} dist {dist:.3f} pos {pos} neg {neg}")
        stages[name]={"dim": int(X.shape[1]), "acc": float(acc), "p": float(p), "null_m": float(null_m), "dist": float(dist), "pos": pos, "neg": neg}

    # FF_OFF control
    w_off=np.array(w, copy=True); w_off[ff_mask]=0.0
    from jaxfne.emitters import EdgeList
    el_off=EdgeList(pre=el.pre, post=el.post, weight=jnp.asarray(w_off, dtype=el.weight.dtype), receptor_index=el.receptor_index, tau_ms=el.tau_ms, delay_steps=el.delay_steps, source_calibration_status=el.source_calibration_status)
    model_off = replace(model, params=dict(model.params, edge_list=el_off))
    cur_all_off=[]
    labels_off=[]
    for rep in range(2):
        for label, arr in [("A", arrA), ("B", arrB)]:
            seed=200+rep*10 + (0 if label=="A" else 1)
            step_fn, init = compile_step_fn(model_off, dt_ms=DT_MS, kernel="baseline", record_edge_current=True)
            init_s = continuation_state_from_model(model_off, seed=seed)
            state, outs = run_continuation(step_fn, init_s, jnp.asarray(arr, dtype=jnp.float32))
            ec=np.asarray(outs[5])
            vec=[]
            for t in tgt_idx:
                es=[ei for ei in ff_mask if int(post[ei])==t]
                if es:
                    vec.append(float(ec[P1_S:P1_E, es].sum(axis=1).mean()))
                else:
                    vec.append(0.0)
            cur_all_off.append(np.array(vec)); labels_off.append(0 if label=="A" else 1)
    cur_all_off=np.array(cur_all_off); labels_off=np.array(labels_off)
    if cur_all_off.shape[1]>0:
        acc_off,p_off,_,_=perm_p(cur_all_off, labels_off, n_perm=100)
        print(f"FF_OFF control I_FF_all acc {acc_off:.3f} p {p_off:.3f} (expect chance)")
        stages["I_FF_all_OFF"]={"acc": float(acc_off), "p": float(p_off)}
    # Recording OFF invariance (one trial)
    step_fn_on, init_on = compile_step_fn(model, dt_ms=DT_MS, kernel="baseline", record_edge_current=True)
    step_fn_off2, init_off2 = compile_step_fn(model, dt_ms=DT_MS, kernel="baseline", record_edge_current=False)
    init_a = continuation_state_from_model(model, seed=0)
    init_b = continuation_state_from_model(model, seed=0)
    _, outs_on = run_continuation(step_fn_on, init_a, jnp.asarray(arrA, dtype=jnp.float32))
    _, outs_off = run_continuation(step_fn_off2, init_b, jnp.asarray(arrA, dtype=jnp.float32))
    vm_on=np.asarray(outs_on[0]); vm_off=np.asarray(outs_off[0])
    spk_on=np.asarray(outs_on[1]); spk_off=np.asarray(outs_off[1])
    print(f"Recording invariance Vm equal {np.allclose(vm_on, vm_off)} spikes equal {np.allclose(spk_on, spk_off)}")

    # Transitions relative to validated V1 L2/3 (now 1.00)
    acc_V1=stages.get("V1_L23_spike",{}).get("acc",0)
    acc_I=stages.get("I_FF_all",{}).get("acc",0)
    acc_V=stages.get("V4_L4_Vm",{}).get("acc",0)
    acc_R=stages.get("V4_L4_spike",{}).get("acc",0)
    print(f"Transitions: V1 {acc_V1:.3f} -> I {acc_I:.3f} -> V {acc_V:.3f} -> R {acc_R:.3f}")
    def is_valid(acc): return acc >= 0.75
    if not is_valid(acc_I):
        first="F1 FF_STRUCTURAL_ENCODING_FAIL"
    elif not is_valid(acc_V):
        first="F2 FF_CURRENT_TO_VM_FAIL"
    elif not is_valid(acc_R):
        first="F3 FF_VM_TO_SPIKE_FAIL"
    elif is_valid(acc_R):
        first="F4 V1_TO_V4_TRANSFER_VALID"
    else:
        first="F_UNRESOLVED"
    print("first F failure:", first)
    out=pathlib.Path("results/w4a_v1_to_v4.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out,"w") as f:
        json.dump({"stages": stages, "first": first, "structural": {"n_src": len(src_idx), "n_tgt": len(tgt_idx), "n_ff": len(ff_mask), "out_deg_mean": float(np.mean(list(out_deg.values())) if out_deg else 0), "in_deg_mean": float(np.mean(list(in_deg.values())) if in_deg else 0)}}, f, indent=2)
    print(f"saved to {out}")
    return stages, first

if __name__=="__main__":
    run()
