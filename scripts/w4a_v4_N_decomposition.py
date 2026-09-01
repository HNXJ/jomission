"""N1-N5 decomposition for V1->V4 F2 using executed I_total/u — no C023."""
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
    model=build_jomission_model(n_per_area=100, seed=0, dt_ms=DT_MS)
    tbl=model.neuron_table()
    v4_l4_idx=[i for i,r in enumerate(tbl) if r["area"]=="V4" and r["layer"]=="L4"]
    v1_l23e_idx=[i for i,r in enumerate(tbl) if r["area"]=="V1" and r["layer"]=="L2/3" and r["cell_type"]=="E"]
    areas=[r["area"] for r in tbl]; layers=[r["layer"] for r in tbl]; cts=[r["cell_type"] for r in tbl]
    el=model.params["edge_list"]
    pre=np.asarray(el.pre); post=np.asarray(el.post); w=np.asarray(el.weight)
    ff_mask=[ei for ei in range(len(pre)) if areas[int(pre[ei])]=="V1" and layers[int(pre[ei])]=="L2/3" and cts[int(pre[ei])]=="E" and areas[int(post[ei])]=="V4" and layers[int(post[ei])]=="L4"]
    # Also local V4 recurrent (non-FF) for I_recurrent_nonFF
    local_mask=[]
    for ei in range(len(pre)):
        pi=int(pre[ei]); qi=int(post[ei])
        if areas[pi]=="V4" and areas[qi]=="V4" and qi in v4_l4_idx:
            # exclude FF (already counted, but FF is V1->V4, so not in this)
            local_mask.append(ei)
    print(f"FF {len(ff_mask)} local V4->V4 to L4 {len(local_mask)}")

    from jomission.simulation.factorial_v0p2 import energy_amplitude as _ea
    rf_cfg=RFConfig(seed=0, tier="graded")
    op=RFOperator(rf_cfg, model)
    condA=[c for c in JOMISSION_PARADIGM.conditions if c.name=="AAAB"][0]
    condB=[c for c in JOMISSION_PARADIGM.conditions if c.name=="BBBA"][0]
    ampA=float(_ea("C","AAAB")); ampB=float(_ea("C","BBBA"))
    schedA=op.to_stimulus_schedule(condA, n_neurons=400, dt_ms=DT_MS, base_amplitude=ampA)
    schedB=op.to_stimulus_schedule(condB, n_neurons=400, dt_ms=DT_MS, base_amplitude=ampB)
    arrA=schedA.to_array(N_STEPS, DT_MS); arrB=schedB.to_array(N_STEPS, DT_MS)
    em=model.params["emitter"]
    drive_tonic=np.asarray(em.drive)

    trials=[]
    for rep in range(N_TRIALS_PER_COND):
        for label, arr in [("A", arrA), ("B", arrB)]:
            seed=100+rep*10 + (0 if label=="A" else 1)
            trials.append((label, arr, seed))
    # Collect with new traces: need current and u
    all_V=[]; all_U=[]; all_ec=[]; all_cur=[]; all_S=[]; labels=[]
    for label, arr, seed in trials:
        step_fn, init = compile_step_fn(model, dt_ms=DT_MS, kernel="baseline", record_edge_current=True, record_current_trace=True, record_u_trace=True)
        init_s = continuation_state_from_model(model, seed=seed)
        state, outs = run_continuation(step_fn, init_s, jax.numpy.asarray(arr, dtype=jax.numpy.float32))
        # outs: v, spikes, sources, H, w, edge_current, current, u
        V=np.asarray(outs[0]); S=np.asarray(outs[1]); ec=np.asarray(outs[5]); cur=np.asarray(outs[6]); u=np.asarray(outs[7])
        all_V.append(V); all_S.append(S); all_ec.append(ec); all_cur.append(cur)
        # Need also U per V4 L4: u is (n_steps, n_neurons) for all neurons, need V4 L4 slice
        all_U.append(u)
        labels.append(0 if label=="A" else 1)
    all_V=np.array(all_V); all_S=np.array(all_S); all_ec=np.array(all_ec); all_cur=np.array(all_cur); all_U=np.array(all_U)
    labels=np.array(labels)
    print(f"collected {len(labels)} trials V {all_V.shape} cur {all_cur.shape} u {all_U.shape}")

    # Verify equation: I_total == drive + schedule + recurrent + noise
    # For one trial, check first step: cur[0,0,:] vs drive + schedule[0] + syn(0)
    # syn = sum edge_current per neuron
    # Compute syn per neuron per trial per time
    n_trials=len(labels)
    # Compute syn per neuron: sum over edges per post
    # Build per post edge list for V4 L4
    syn = np.zeros((n_trials, N_STEPS, 400))
    for tr in range(n_trials):
        for ei in range(len(pre)):
            q=int(post[ei])
            syn[tr, :, q] += all_ec[tr, :, ei]
    # For V4 L4 neuron 0, check equation at t=10
    tr0=0; t0=10; nid=v4_l4_idx[0]
    drive_val=float(drive_tonic[nid])
    sched_val=float(np.array(arrA if labels[tr0]==0 else arrB)[t0, nid])  # approximate, but need per trial arr
    # For tr0, label 0 => arrA
    arr_tr0 = arrA if labels[tr0]==0 else arrB
    sched_val2=float(arr_tr0[t0, nid])
    syn_val=float(syn[tr0, t0, nid])
    cur_val=float(all_cur[tr0, t0, nid])
    # Noise approx = cur - (drive + sched + syn)
    noise_est = cur_val - (drive_val + sched_val2 + syn_val)
    print(f"Equation check trial0 t{t0} V4 L4 {nid}: drive {drive_val:.3f} sched {sched_val2:.3f} syn {syn_val:.4f} cur {cur_val:.3f} noise_est {noise_est:.4f}")
    # Also check that current leads to correct dv: we can verify by one-step V update
    # Use _izhikevich_dv_du to compute expected dv
    from jaxfne.emitters import _izhikevich_dv_du
    # Need V and u at t
    V_t = all_V[tr0, t0, nid]; u_t = all_U[tr0, t0, nid]  # note all_U is u_reset per step, which is next u, but close
    # Instead use V and u from state: we have V and U traces, but U trace is u_reset after update, not before
    # For equation check, we can just verify that cur == drive+schedule+syn+noise within tolerance by reconstructing noise as above and checking that cur is consistent across trials (noise variance)
    # Check that cur variance across trials for same condition is due to noise
    # More direct: verify that I_total_recorded equals exact current_native by checking that V update matches
    # We can do tiny deterministic network perturbation test: add 1.0 to drive for one neuron and see cur increases by 1.0
    print("Equation-level consistency: perturb drive by +1.0 for V4 L4 0 and check cur delta")
    # Create perturbed model with drive+1 for one neuron
    from dataclasses import replace
    import jax.numpy as jnp
    drive_pert = np.array(drive_tonic, copy=True)
    drive_pert[nid] += 1.0
    em_pert = replace(em, drive=jax.numpy.asarray(drive_pert, dtype=em.drive.dtype))
    from jaxfne.emitters import IzhikevichParams
    # Need to create perturbed model
    model_pert = replace(model, params=dict(model.params, emitter=em_pert))
    # Run one trial with same seed and same schedule
    step_fn_p, init_p = compile_step_fn(model_pert, dt_ms=DT_MS, kernel="baseline", record_edge_current=True, record_current_trace=True, record_u_trace=True)
    init_s_p = continuation_state_from_model(model_pert, seed=100)
    state_p, outs_p = run_continuation(step_fn_p, init_s_p, jax.numpy.asarray(arrA, dtype=jax.numpy.float32))
    cur_p = np.asarray(outs_p[6])  # current is 6th after edge (5), current (6), u (7) -> actually outs[6] is current
    # Compare cur for perturbed vs original at t0 for nid
    cur_orig_t0 = float(all_cur[0, t0, nid])  # need to map all_cur[0] is trial0 with seed 100, same as perturbed seed 100
    cur_pert_t0 = float(cur_p[t0, nid])
    print(f" cur original {cur_orig_t0:.3f} perturbed {cur_pert_t0:.3f} delta {cur_pert_t0-cur_orig_t0:.3f} (expected ~1.0)")

    # Now evaluate N1-N5 for V4 L4
    # Build vectors per stage for p1 window
    # I_FF per V4 L4 (already cur_all is per V4 L4, but I_FF is only part of cur)
    # For I_FF we need target-summed FF current per V4 L4
    n_v4=len(v4_l4_idx)
    I_FF = np.zeros((n_trials, n_v4))
    I_local = np.zeros((n_trials, n_v4))
    I_total = np.zeros((n_trials, n_v4))
    for tr in range(n_trials):
        for j, t in enumerate(v4_l4_idx):
            es_ff=[ei for ei in ff_mask if int(post[ei])==t]
            es_loc=[ei for ei in local_mask if int(post[ei])==t]
            # For I_FF, use ec sum
            if es_ff:
                I_FF[tr, j] = float(all_ec[tr][P1_S:P1_E][:, es_ff].sum(axis=1).mean())
            if es_loc:
                I_local[tr, j] = float(all_ec[tr][P1_S:P1_E][:, es_loc].sum(axis=1).mean())
            # I_total per neuron: mean current over p1 window for that neuron
            I_total[tr, j] = float(all_cur[tr][P1_S:P1_E, t].mean())
    # Also need I_external (drive+schedule) per V4 L4
    I_ext = np.zeros((n_trials, n_v4))
    for tr, (label, arr, seed) in enumerate(trials):
        for j, t in enumerate(v4_l4_idx):
            I_ext[tr, j] = float(np.array(arr)[P1_S:P1_E, t].mean() + drive_tonic[t])
    # Noise approx per trial per V4 L4: I_total - (I_FF + I_local + I_ext)
    I_noise = I_total - (I_FF + I_local + I_ext)
    # Vm and spikes per V4 L4
    Vm = np.zeros((n_trials, n_v4))
    Spk = np.zeros((n_trials, n_v4))
    for tr in range(n_trials):
        Vm[tr] = all_V[tr][P1_S:P1_E][:, v4_l4_idx].mean(axis=0)
        Spk[tr] = all_S[tr][P1_S:P1_E][:, v4_l4_idx].mean(axis=0)*(1000/DT_MS)
    # Also need u per V4 L4
    U = np.zeros((n_trials, n_v4))
    for tr in range(n_trials):
        U[tr] = all_U[tr][P1_S:P1_E][:, v4_l4_idx].mean(axis=0)

    def report(name, X):
        acc,p,_,_=perm_p(X, labels, n_perm=50)
        c0=X[labels==0].mean(axis=0); c1=X[labels==1].mean(axis=0)
        dist=float(np.linalg.norm(c1-c0))
        print(f" {name:12} acc {acc:.3f} p {p:.3f} dist {dist:.4f} meanA {c0.mean():.3f} meanB {c1.mean():.3f}")
        return acc, p, dist

    print("\nStage decoders p1 window:")
    for name, X in [("I_FF", I_FF), ("I_local", I_local), ("I_ext", I_ext), ("I_total", I_total), ("Vm", Vm), ("U", U), ("Spk", Spk)]:
        report(name, X)
    acc_IFF,_,_,_ = perm_p(I_FF, labels, n_perm=50)
    acc_Itot,_,_,_ = perm_p(I_total, labels, n_perm=50)
    acc_Vm,_,_,_ = perm_p(Vm, labels, n_perm=50)
    acc_Spk,_,_,_ = perm_p(Spk, labels, n_perm=50)
    print(f"\nTransitions: I_FF {acc_IFF:.3f} -> I_total {acc_Itot:.3f} -> Vm {acc_Vm:.3f} -> Spk {acc_Spk:.3f}")
    # Also check I_noise variance vs signal
    var_noise = float(I_noise.var())
    sig_IFF = float((I_FF[labels==0].mean(axis=0) - I_FF[labels==1].mean(axis=0)).var())
    print(f" I_noise var {var_noise:.6f} I_FF sig var {sig_IFF:.6f} ratio {var_noise/sig_IFF if sig_IFF>0 else 0:.3f}")
    # Check adaptation
    # U per V4 L4 mean
    print(f" U mean A {U[labels==0].mean():.2f} B {U[labels==1].mean():.2f} delta {U[labels==0].mean()-U[labels==1].mean():.3f}")

    # Classify N1-N5
    # N1 CANCELLED_BEFORE_I_TOTAL: I_FF 1.00 but I_total <0.75
    # N2 PRESENT_IN_I_TOTAL_BUT_CANCELLED_BY_INTRINSIC: I_total 1.00 but Vm <0.75
    # N3 MASKED_BY_STOCHASTIC: I_total sig small vs noise var large
    # N4 ADAPTATION_U_DOMINATED: U carries information and Vm not
    # N5 FF semantics not effective: I_FF effect size small (already H5)
    # Use thresholds 0.75
    def is_valid(acc): return acc >= 0.75
    if not is_valid(acc_IFF):
        # This would be H5 already, but we have acc_IFF 1.00 so not
        first="N_UNRESOLVED"
    elif not is_valid(acc_Itot):
        first="N1 CANCELLED_BEFORE_I_TOTAL"
    elif is_valid(acc_Itot) and not is_valid(acc_Vm):
        # Check if U dominates: does U decode?
        acc_U,_ ,_,_=perm_p(U, labels, n_perm=50)
        if acc_U >= 0.75:
            first="N4 ADAPTATION_U_DOMINATED"
        elif var_noise > sig_IFF:
            first="N3 MASKED_BY_STOCHASTIC_VARIANCE"
        else:
            first="N2 PRESENT_IN_I_TOTAL_BUT_CANCELLED_BY_INTRINSIC_DYNAMICS"
    elif is_valid(acc_Vm) and not is_valid(acc_Spk):
        first="N2/N4 -> Vm->Spike"
    else:
        first="N5 or N_MIXED"
    print("first N failure:", first)
    # Save
    out=pathlib.Path("results/w4a_v4_N_decomposition.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out,"w") as f:
        json.dump({"I_FF_acc": float(acc_IFF), "I_total_acc": float(acc_Itot), "Vm_acc": float(acc_Vm), "Spk_acc": float(acc_Spk), "first": first}, f, indent=2)
    print(f"saved to {out}")

if __name__=="__main__":
    run()
