"""Bounded FF-gain counterfactual — g_FF in {0,0.5,1,2,4}, assay-only, no C023."""
import json, pathlib, numpy as np, jax, jax.numpy as jnp
from jomission.network.builder import build_jomission_model
from jomission.network.rf import RFConfig, RFOperator
from jomission.paradigm.spec import JOMISSION_PARADIGM
from jaxfne._pipeline import compile_step_fn, run_continuation, continuation_state_from_model
from dataclasses import replace
from jaxfne.emitters import EdgeList

DT_MS=0.1
DUR_MS=600.0
N_STEPS=int(DUR_MS/DT_MS)
P1_S=int(0/DT_MS); P1_E=int(531/DT_MS)
N_TRIALS_PER_COND=8
GAINS=[0, 0.5, 1, 2, 4]

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

def perm_p(X,y, n_perm=50):
    acc=lovo_acc(X,y)
    null=[]
    for _ in range(n_perm):
        yp=np.random.permutation(y)
        null.append(lovo_acc(X, yp))
    null=np.array(null)
    p=float((np.sum(null>=acc)+1)/(n_perm+1))
    return acc, p, float(null.mean()), float(null.std())

def run():
    base_model=build_jomission_model(n_per_area=100, seed=0, dt_ms=DT_MS)
    tbl=base_model.neuron_table()
    areas=[r["area"] for r in tbl]; layers=[r["layer"] for r in tbl]; cts=[r["cell_type"] for r in tbl]
    el0=base_model.params["edge_list"]
    pre0=np.asarray(el0.pre); post0=np.asarray(el0.post)
    ff_mask=[ei for ei in range(len(pre0)) if areas[int(pre0[ei])]=="V1" and layers[int(pre0[ei])]=="L2/3" and cts[int(pre0[ei])]=="E" and areas[int(post0[ei])]=="V4" and layers[int(post0[ei])]=="L4"]
    v1_l23e_idx=[i for i,r in enumerate(tbl) if r["area"]=="V1" and r["layer"]=="L2/3" and r["cell_type"]=="E"]
    v4_l4_idx=[i for i,r in enumerate(tbl) if r["area"]=="V4" and r["layer"]=="L4"]
    v4_l4e_idx=[i for i in v4_l4_idx if tbl[i]["cell_type"]=="E"]
    print(f"FF {len(ff_mask)} V1 L2/3_E {len(v1_l23e_idx)} V4 L4 {len(v4_l4_idx)}")

    from jomission.simulation.factorial_v0p2 import energy_amplitude as _ea
    rf_cfg=RFConfig(seed=0, tier="graded")
    op=RFOperator(rf_cfg, base_model)
    condA=[c for c in JOMISSION_PARADIGM.conditions if c.name=="AAAB"][0]
    condB=[c for c in JOMISSION_PARADIGM.conditions if c.name=="BBBA"][0]
    ampA=float(_ea("C","AAAB")); ampB=float(_ea("C","BBBA"))
    schedA=op.to_stimulus_schedule(condA, n_neurons=400, dt_ms=DT_MS, base_amplitude=ampA)
    schedB=op.to_stimulus_schedule(condB, n_neurons=400, dt_ms=DT_MS, base_amplitude=ampB)
    arrA=schedA.to_array(N_STEPS, DT_MS); arrB=schedB.to_array(N_STEPS, DT_MS)

    results={}
    for g in GAINS:
        print(f"\n=== g_FF {g} ===")
        # Build model with scaled FF weights (assay-only)
        if g==1:
            model=base_model
        else:
            w=np.asarray(el0.weight)
            w_new=w.copy()
            w_new[ff_mask] = w[ff_mask] * float(g)
            el_new=EdgeList(pre=el0.pre, post=el0.post, weight=jnp.asarray(w_new, dtype=el0.weight.dtype), receptor_index=el0.receptor_index, tau_ms=el0.tau_ms, delay_steps=el0.delay_steps, source_calibration_status=el0.source_calibration_status)
            model = replace(base_model, params=dict(base_model.params, edge_list=el_new))
        # Trials
        trials=[]
        for rep in range(N_TRIALS_PER_COND):
            for label, arr in [("A", arrA), ("B", arrB)]:
                seed=100+rep*10 + (0 if label=="A" else 1)
                trials.append((label, arr, seed))
        all_V=[]; all_S=[]; all_ec=[]; labels=[]
        for label, arr, seed in trials:
            step_fn, init = compile_step_fn(model, dt_ms=DT_MS, kernel="baseline", record_edge_current=True)
            init_s = continuation_state_from_model(model, seed=seed)
            state, outs = run_continuation(step_fn, init_s, jnp.asarray(arr, dtype=jnp.float32))
            V=np.asarray(outs[0]); S=np.asarray(outs[1]); ec=np.asarray(outs[5])
            all_V.append(V); all_S.append(S); all_ec.append(ec); labels.append(0 if label=="A" else 1)
        all_V=np.array(all_V); all_S=np.array(all_S); all_ec=np.array(all_ec); labels=np.array(labels)
        # Vectors per stage in p1 window
        # V1 L2/3 spike
        v1_spk = all_S[:, P1_S:P1_E, :][:, :, v1_l23e_idx].mean(axis=1)*(1000/DT_MS) if v1_l23e_idx else np.zeros((len(labels),1))
        # I_FF per V4 L4
        n_v4=len(v4_l4_idx)
        # Build per V4 L4 incoming FF map
        per_tgt={t: [] for t in v4_l4_idx}
        for ei in ff_mask:
            per_tgt[int(post0[ei])].append(ei)
        cur_all=np.zeros((len(labels), n_v4))
        for tr in range(len(labels)):
            for j, t in enumerate(v4_l4_idx):
                es=per_tgt[t]
                if es:
                    cur_all[tr, j] = float(all_ec[tr, P1_S:P1_E, es].sum(axis=1).mean())
        vm_v4 = all_V[:, P1_S:P1_E, :][:, :, v4_l4_idx].mean(axis=1) if v4_l4_idx else np.zeros((len(labels),1))
        spk_v4 = all_S[:, P1_S:P1_E, :][:, :, v4_l4_idx].mean(axis=1)*(1000/DT_MS) if v4_l4_idx else np.zeros((len(labels),1))
        # Metrics
        # Contrast norm ||ΔI_FF||
        c0_I=cur_all[labels==0].mean(axis=0); c1_I=cur_all[labels==1].mean(axis=0)
        norm_I=float(np.linalg.norm(c1_I-c0_I))
        # Decoder
        acc_v1,p_v1,_,_=perm_p(v1_spk, labels, n_perm=50)
        acc_I,p_I,_,_=perm_p(cur_all, labels, n_perm=50)
        acc_vm,p_vm,_,_=perm_p(vm_v4, labels, n_perm=50)
        acc_spk,p_spk,_,_=perm_p(spk_v4, labels, n_perm=50)
        # Correspondence cos and lag coupling
        # ΔI vs ΔVm per V4 L4
        dI=c1_I-c0_I
        c0_Vm=vm_v4[labels==0].mean(axis=0); c1_Vm=vm_v4[labels==1].mean(axis=0)
        dVm=c1_Vm-c0_Vm
        nrmI=np.linalg.norm(dI); nrmVm=np.linalg.norm(dVm)
        cos = float(np.dot(dI, dVm)/(nrmI*nrmVm)) if nrmI>0 and nrmVm>0 else 0.0
        # Trial-level I->Vm correlation (residual)
        # Per V4 L4 neuron, correlation across trials between I and Vm (conditioned)
        # Use mean across neurons
        corrs=[]
        for j in range(n_v4):
            i_vals=cur_all[:, j]; v_vals=vm_v4[:, j]
            # residual after removing condition mean
            i_res=i_vals - np.array([i_vals[labels==0].mean() if labels[tr]==0 else i_vals[labels==1].mean() for tr in range(len(labels))])
            v_res=v_vals - np.array([v_vals[labels==0].mean() if labels[tr]==0 else v_vals[labels==1].mean() for tr in range(len(labels))])
            if np.std(i_res)>1e-9 and np.std(v_res)>1e-9:
                corrs.append(float(np.corrcoef(i_res, v_res)[0,1]))
        corr_mean=float(np.mean(corrs)) if corrs else 0.0
        # Rates
        rate_v4_mean=float(spk_v4.mean())
        vm_mean=float(vm_v4.mean())
        # Generic gates: B1/B2/B3 proxies - check rates 1-80, Vm finite, etc. For assay scale, just check V4 L4 rate
        rate_ok = 1 <= rate_v4_mean <= 80
        print(f" V1 spike acc {acc_v1:.3f} p {p_v1:.3f} | I_FF norm {norm_I:.4f} acc {acc_I:.3f} p {p_I:.3f} cos {cos:.3f} corr {corr_mean:.3f} | Vm acc {acc_vm:.3f} p {p_vm:.3f} | Spk acc {acc_spk:.3f} p {p_spk:.3f} Vm {vm_mean:.1f} rate {rate_v4_mean:.1f} {'OK' if rate_ok else 'PATHOLOGICAL'}")
        results[float(g)]={"norm_I": float(norm_I), "acc_v1": float(acc_v1), "acc_I": float(acc_I), "p_I": float(p_I), "cos": float(cos), "corr": float(corr_mean), "acc_vm": float(acc_vm), "p_vm": float(p_vm), "acc_spk": float(acc_spk), "p_spk": float(p_spk), "vm_mean": float(vm_mean), "rate_mean": float(rate_v4_mean), "rate_ok": bool(rate_ok)}

    # Check mechanistic prediction: monotonic dose-response
    gains_sorted=sorted(results.keys())
    norms=[results[g]["norm_I"] for g in gains_sorted]
    acc_vms=[results[g]["acc_vm"] for g in gains_sorted]
    acc_spks=[results[g]["acc_spk"] for g in gains_sorted]
    print(f"\nDose-response norms {norms}")
    print(f" Vm acs {[results[g]['acc_vm'] for g in gains_sorted]}")
    print(f" Spk acs {[results[g]['acc_spk'] for g in gains_sorted]}")
    # Check monotonic: norm should increase with g, and Vm/spk should increase after g=1 if causal
    # g=0 must be chance for I
    i_ok = results[0]["acc_I"] < 0.6  # expect chance at 0
    norm_monotonic = all(norms[i] <= norms[i+1] + 1e-9 for i in range(len(norms)-1))
    # Vm should increase from g=1 onwards if causal
    # Check if acc_vm at g=2,4 > at g=1
    vm_increase = results[2]["acc_vm"] > results[1]["acc_vm"] and results[4]["acc_vm"] > results[1]["acc_vm"]
    spk_increase = results[2]["acc_spk"] > results[1]["acc_spk"] and results[4]["acc_spk"] > results[1]["acc_spk"]
    print(f"i_ok (g0 chance) {i_ok} norm_monotonic {norm_monotonic} vm_increase {vm_increase} spk_increase {spk_increase}")
    # Also check cosine/corr increase
    cors=[results[g]["cos"] for g in gains_sorted]
    corrs2=[results[g]["corr"] for g in gains_sorted]
    print(f" cos {cors} corr {corrs2}")
    # Classify K
    # K1 requires more than decoder accuracy: increasing gain must increase dynamical coupling (cos/corr) and Vm/spike representation, with V1 unchanged (V1 acc stable)
    v1_stable = all(abs(results[g]["acc_v1"] - results[1]["acc_v1"]) < 0.1 for g in gains_sorted)
    # Check rate_ok for g=2,4
    rate_ok = all(results[g]["rate_ok"] for g in [2,4])
    # Check nonmonotonic: if Vm/spk goes up then down, K4
    vm_vals=[results[g]["acc_vm"] for g in gains_sorted]
    nonmono = (vm_vals[3] < vm_vals[2] and vm_vals[4] < vm_vals[2]) or (vm_vals[2] < vm_vals[1] and vm_vals[3] > vm_vals[2])
    # For our data, we saw vm 0.438 at g1, need to see at 2 and 4
    if not i_ok:
        cand="K_UNRESOLVED (g0 not chance)"
    elif not norm_monotonic:
        cand="K4 NONMONOTONIC"
    elif vm_increase and spk_increase and v1_stable and rate_ok and (results[4]["acc_vm"] > 0.75 or results[4]["acc_spk"] > 0.75):
        cand="K1 FF_GAIN_CAUSAL"
    elif not (vm_increase or spk_increase):
        cand="K2 FF_GAIN_INSUFFICIENT"
    elif not rate_ok:
        cand="K3 EFFECTIVE_BUT_PATHOLOGICAL"
    else:
        cand="K_UNRESOLVED"
    print("classification:", cand)
    out=pathlib.Path("results/w4a_ff_gain_assay.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out,"w") as f:
        json.dump(results, f, indent=2)
    print(f"saved to {out}")

if __name__=="__main__":
    run()
