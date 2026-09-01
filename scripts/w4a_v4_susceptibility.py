"""V4 L4 susceptibility assay — temporary V4 L4 current offset, no C023.

Tests F2c provisional: modest depolarization of V4 L4 only should increase
V4 L4 Vm/spike representation while I_FF unchanged.
Grid δI ∈ { -0.25, 0, +0.25, +0.5, +1.0 } native units, assay-only.
"""
import json, pathlib, numpy as np, jax, jax.numpy as jnp
from jomission.network.builder import build_jomission_model
from jomission.network.rf import RFConfig, RFOperator
from jomission.paradigm.spec import JOMISSION_PARADIGM
from jaxfne._pipeline import compile_step_fn, run_continuation, continuation_state_from_model

DT_MS=0.1
DUR_MS=600.0
N_STEPS=int(DUR_MS/DT_MS)
P1_S=int(0/DT_MS); P1_E=int(531/DT_MS)
N_TRIALS_PER_COND=8
DELTAS=[-0.25, 0, 0.25, 0.5, 1.0]

def lovo_acc(X,y):
    n=len(y)
    if X.shape[1]==0:
        return 0.5
    correct=0
    for i in range(n):
        tr=[j for j in range(n) if j!=i]
        Xtr=X[tr]; ytr=y[tr]
        if len(np.unique(ytr))<2:
            continue
        c0=Xtr[ytr==0].mean(axis=0); c1=Xtr[ytr==1].mean(axis=0)
        if np.any(np.isnan(c0)) or np.any(np.isnan(c1)):
            continue
        d0=np.linalg.norm(X[i]-c0); d1=np.linalg.norm(X[i]-c1)
        pred=0 if d0<d1 else 1
        if pred==y[i]:
            correct+=1
    return correct/n

def perm_p(X,y, n_perm=100):
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
    l4_v4_idx=[i for i,r in enumerate(tbl) if r["area"]=="V4" and r["layer"]=="L4"]
    v4_l4_idx=l4_v4_idx
    v1_l23e_idx=[i for i,r in enumerate(tbl) if r["area"]=="V1" and r["layer"]=="L2/3" and r["cell_type"]=="E"]
    areas=[r["area"] for r in tbl]; layers=[r["layer"] for r in tbl]; cts=[r["cell_type"] for r in tbl]
    el=model.params["edge_list"]
    pre=np.asarray(el.pre); post=np.asarray(el.post)
    ff_mask=[ei for ei in range(len(pre)) if areas[int(pre[ei])]=="V1" and layers[int(pre[ei])]=="L2/3" and cts[int(pre[ei])]=="E" and areas[int(post[ei])]=="V4" and layers[int(post[ei])]=="L4"]
    print(f"V4 L4 n {len(v4_l4_idx)} FF {len(ff_mask)}")

    from jomission.simulation.factorial_v0p2 import energy_amplitude as _ea
    rf_cfg=RFConfig(seed=0, tier="graded")
    op=RFOperator(rf_cfg, model)
    condA=[c for c in JOMISSION_PARADIGM.conditions if c.name=="AAAB"][0]
    condB=[c for c in JOMISSION_PARADIGM.conditions if c.name=="BBBA"][0]
    ampA=float(_ea("C","AAAB")); ampB=float(_ea("C","BBBA"))
    schedA=op.to_stimulus_schedule(condA, n_neurons=400, dt_ms=DT_MS, base_amplitude=ampA)
    schedB=op.to_stimulus_schedule(condB, n_neurons=400, dt_ms=DT_MS, base_amplitude=ampB)
    arrA_base=schedA.to_array(N_STEPS, DT_MS); arrB_base=schedB.to_array(N_STEPS, DT_MS)

    results={}
    for dI in DELTAS:
        print(f"\n=== deltaI {dI:+.2f} ===")
        # Create offset arrays for V4 L4 only
        offset = np.zeros((N_STEPS, 400), dtype=np.float32)
        offset[:, v4_l4_idx] = float(dI)
        arrA = arrA_base + offset
        arrB = arrB_base + offset
        trials=[]
        for rep in range(N_TRIALS_PER_COND):
            for label, arr in [("A", arrA), ("B", arrB)]:
                seed=100+rep*10 + (0 if label=="A" else 1)
                trials.append((label, arr, seed))
        # Collect vectors
        v1_spike=[]; cur_all=[]; vm_v4=[]; spk_v4=[]; vm_mean=[]; rate_mean=[]
        labels=[]
        for label, arr, seed in trials:
            step_fn, init = compile_step_fn(model, dt_ms=DT_MS, kernel="baseline", record_edge_current=True)
            init_s = continuation_state_from_model(model, seed=seed)
            state, outs = run_continuation(step_fn, init_s, jnp.asarray(arr, dtype=jnp.float32))
            V=np.asarray(outs[0]); S=np.asarray(outs[1]); ec=np.asarray(outs[5])
            # V1 L2/3_E spike (should be unchanged across dI)
            v1_spk = S[P1_S:P1_E, v1_l23e_idx].mean(axis=0)*(1000/DT_MS) if v1_l23e_idx else np.zeros(1)
            # I_FF per V4 L4
            def target_vec(idxs):
                vec=[]
                for t in idxs:
                    es=[ei for ei in ff_mask if int(post[ei])==t]
                    if es:
                        vec.append(float(ec[P1_S:P1_E, es].sum(axis=1).mean()))
                    else:
                        vec.append(0.0)
                return np.array(vec)
            cur = target_vec(v4_l4_idx)
            vm = V[P1_S:P1_E, v4_l4_idx].mean(axis=0) if v4_l4_idx else np.zeros(1)
            spk = S[P1_S:P1_E, v4_l4_idx].mean(axis=0)*(1000/DT_MS) if v4_l4_idx else np.zeros(1)
            v1_spike.append(v1_spk); cur_all.append(cur); vm_v4.append(vm); spk_v4.append(spk)
            labels.append(0 if label=="A" else 1)
            # For mean Vm/rate
            if label=="A" and len(vm_mean)<5: # just to collect
                pass
        v1_spike=np.array(v1_spike); cur_all=np.array(cur_all); vm_v4=np.array(vm_v4); spk_v4=np.array(spk_v4); labels=np.array(labels)
        # Also overall Vm mean and rate
        # Recompute per trial Vm mean across V4 L4 and rate
        # For reporting, use mean across V4 L4 neurons and trials
        # We have vm_v4 is per neuron per trial mean; overall mean is vm_v4.mean()
        print(f" V1 spike dim {v1_spike.shape[1]} acc {lovo_acc(v1_spike, labels):.3f}")
        acc_cur,p_cur,_,_=perm_p(cur_all, labels, n_perm=50)
        acc_vm,p_vm,_,_=perm_p(vm_v4, labels, n_perm=50)
        acc_spk,p_spk,_,_=perm_p(spk_v4, labels, n_perm=50)
        print(f" I_FF dim {cur_all.shape[1]} acc {acc_cur:.3f} p {p_cur:.3f}")
        print(f" Vm dim {vm_v4.shape[1]} acc {acc_vm:.3f} p {p_vm:.3f} mean {vm_v4.mean():.1f} std {vm_v4.std():.2f}")
        print(f" Spk dim {spk_v4.shape[1]} acc {acc_spk:.3f} p {p_spk:.3f} mean rate {spk_v4.mean():.1f} Hz")
        # Generic gates: check rates 1-80, Vm finite, etc.
        rate_ok = float(spk_v4.mean()) < 80 and float(spk_v4.mean()) > 0.5  # not pathological
        vm_ok = np.all(np.isfinite(vm_v4))
        print(f" gates rate_ok {rate_ok} vm_ok {vm_ok} (mean rate {spk_v4.mean():.1f})")
        results[float(dI)]={"acc_cur": float(acc_cur), "p_cur": float(p_cur), "acc_vm": float(acc_vm), "p_vm": float(p_vm), "acc_spk": float(acc_spk), "p_spk": float(p_spk), "vm_mean": float(vm_v4.mean()), "rate_mean": float(spk_v4.mean()), "rate_ok": bool(rate_ok)}

    # Check monotonicity
    # I_FF should be unchanged (since offset only V4 L4, not V1)
    deltas = sorted(results.keys())
    acc_curs=[results[d]["acc_cur"] for d in deltas]
    acc_vms=[results[d]["acc_vm"] for d in deltas]
    acc_spks=[results[d]["acc_spk"] for d in deltas]
    print(f"\nMonotonic check: I_FF acs {acc_curs} (should be stable ~1.00)")
    print(f" Vm acs {acc_vms}")
    print(f" Spk acs {acc_spks}")
    # Classification
    # G1 if modest depolarization increases Vm/spk representation while I_FF unchanged and physiology ok
    # Check delta 0 -> 0.5 ->1.0 monotonic increase
    # Use 0 vs 0.5 vs 1.0
    acc_vm_0=results[0]["acc_vm"]; acc_vm_05=results[0.5]["acc_vm"]; acc_vm_1=results[1.0]["acc_vm"]
    acc_spk_0=results[0]["acc_spk"]; acc_spk_05=results[0.5]["acc_spk"]; acc_spk_1=results[1.0]["acc_spk"]
    # Also check -0.25 as directional control: should be <=0
    acc_vm_neg=results[-0.25]["acc_vm"]
    print(f"Vm 0 {acc_vm_0:.3f} 0.5 {acc_vm_05:.3f} 1.0 {acc_vm_1:.3f} -0.25 {acc_vm_neg:.3f}")
    # Determine G
    # G1 requires monotonic increase and I_FF stable and rate_ok
    monotonic_vm = acc_vm_05 >= acc_vm_0 and acc_vm_1 >= acc_vm_05 and acc_vm_neg <= acc_vm_0 + 0.1
    monotonic_spk = acc_spk_05 >= acc_spk_0 and acc_spk_1 >= acc_spk_05
    i_stable = all(abs(results[d]["acc_cur"] - results[0]["acc_cur"]) < 0.1 for d in [0.25,0.5,1.0])
    rate_ok_all = all(results[d]["rate_ok"] for d in [0.25,0.5,1.0])
    print(f"monotonic_vm {monotonic_vm} monotonic_spk {monotonic_spk} i_stable {i_stable} rate_ok {rate_ok_all}")
    if monotonic_vm and monotonic_spk and i_stable and rate_ok_all and acc_vm_1 > acc_vm_0 + 0.1:
        cand="G1 OPERATING_POINT_CAUSAL"
    elif not (monotonic_vm and monotonic_spk):
        cand="G2 OPERATING_POINT_NOT_CAUSAL"
    elif not rate_ok_all:
        cand="G3 EFFECTIVE_BUT_PATHOLOGICAL"
    else:
        cand="G_UNRESOLVED"
    print("classification:", cand)
    # Save
    out=pathlib.Path("results/w4a_v4_susceptibility.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out,"w") as f:
        json.dump(results, f, indent=2)
    print(f"saved to {out}")

if __name__=="__main__":
    run()
