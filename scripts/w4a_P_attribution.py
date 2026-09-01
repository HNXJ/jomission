"""P attribution for I_FF -> I_total loss — variance/covariance + noise counterfactual."""
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
    areas=[r["area"] for r in tbl]; layers=[r["layer"] for r in tbl]; cts=[r["cell_type"] for r in tbl]
    el=model.params["edge_list"]
    pre=np.asarray(el.pre); post=np.asarray(el.post); w=np.asarray(el.weight)
    ff_mask=[ei for ei in range(len(pre)) if areas[int(pre[ei])]=="V1" and layers[int(pre[ei])]=="L2/3" and cts[int(pre[ei])]=="E" and areas[int(post[ei])]=="V4" and layers[int(post[ei])]=="L4"]
    local_E_mask=[]; local_PV_mask=[]; local_SST_mask=[]; local_VIP_mask=[]
    for ei in range(len(pre)):
        pi=int(pre[ei]); qi=int(post[ei])
        if areas[pi]=="V4" and qi in v4_l4_idx:
            ct=cts[pi]
            if ct=="E":
                local_E_mask.append(ei)
            elif ct=="PV":
                local_PV_mask.append(ei)
            elif ct=="SST":
                local_SST_mask.append(ei)
            elif ct=="VIP":
                local_VIP_mask.append(ei)
    print(f"FF {len(ff_mask)} local E {len(local_E_mask)} PV {len(local_PV_mask)} SST {len(local_SST_mask)} VIP {len(local_VIP_mask)}")

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
    n_trials=len(trials)
    n_v4=len(v4_l4_idx)
    # Collect with full traces
    all_ec=[]; all_cur=[]; all_V=[]; labels=[]
    for label, arr, seed in trials:
        step_fn, init = compile_step_fn(model, dt_ms=DT_MS, kernel="baseline", record_edge_current=True, record_current_trace=True, record_u_trace=True)
        init_s = continuation_state_from_model(model, seed=seed)
        state, outs = run_continuation(step_fn, init_s, jax.numpy.asarray(arr, dtype=jax.numpy.float32))
        ec=np.asarray(outs[5]); cur=np.asarray(outs[6]); V=np.asarray(outs[0])
        all_ec.append(ec); all_cur.append(cur); all_V.append(V); labels.append(0 if label=="A" else 1)
    all_ec=np.array(all_ec); all_cur=np.array(all_cur); all_V=np.array(all_V); labels=np.array(labels)
    print(f"collected {n_trials} trials")

    # Build component traces per V4 L4 neuron per trial per time: need per trial per time per V4 L4
    # For each trial, per V4 L4 neuron, components per time:
    # I_FF(t) = sum over FF edges to that neuron ec[t, es]
    # I_local_E etc similarly, but we have only ec per edge, not per component total per neuron per time yet
    # Instead we can compute per trial per time per V4 L4 for each component by summing ec over respective masks
    # For I_total, we have cur directly (current_trace) per neuron per time
    # For I_ext = drive + schedule, schedule is arr per trial per neuron per time
    # For I_noise = cur - (drive + schedule + syn) where syn = sum ec per neuron
    # But syn = I_FF + I_local + other inter-area (none other for V4 L4)
    # So we can compute

    # Precompute per V4 L4 per trial per time for each component
    # For efficiency, compute per trial aggregated over p1 window mean per neuron, as before for decoders, but also need time-resolved for variance
    # For variance accounting, we need per trial per V4 L4 mean over p1 window per component
    n_trials=len(labels)
    I_FF = np.zeros((n_trials, n_v4))
    I_locE = np.zeros((n_trials, n_v4))
    I_locPV = np.zeros((n_trials, n_v4))
    I_locSST = np.zeros((n_trials, n_v4))
    I_locVIP = np.zeros((n_trials, n_v4))
    I_ext = np.zeros((n_trials, n_v4))
    I_noise = np.zeros((n_trials, n_v4))
    I_total = np.zeros((n_trials, n_v4))
    # Also need per trial per time for time-resolved, but for variance we use mean over p1
    for tr in range(n_trials):
        label, arr, seed = trials[tr]
        for j, t in enumerate(v4_l4_idx):
            es_ff=[ei for ei in ff_mask if int(post[ei])==t]
            es_E=[ei for ei in local_E_mask if int(post[ei])==t]
            es_PV=[ei for ei in local_PV_mask if int(post[ei])==t]
            es_SST=[ei for ei in local_SST_mask if int(post[ei])==t]
            es_VIP=[ei for ei in local_VIP_mask if int(post[ei])==t]
            # Use ec and cur for this trial
            ec_tr = all_ec[tr]  # (6000,10666)
            cur_tr = all_cur[tr]  # (6000,400)
            # Compute per component mean over p1 window per neuron
            if es_ff:
                I_FF[tr, j] = float(ec_tr[P1_S:P1_E, es_ff].sum(axis=1).mean())
            if es_E:
                I_locE[tr, j] = float(ec_tr[P1_S:P1_E, es_E].sum(axis=1).mean())
            if es_PV:
                I_locPV[tr, j] = float(ec_tr[P1_S:P1_E, es_PV].sum(axis=1).mean())
            if es_SST:
                I_locSST[tr, j] = float(ec_tr[P1_S:P1_E, es_SST].sum(axis=1).mean())
            if es_VIP:
                I_locVIP[tr, j] = float(ec_tr[P1_S:P1_E, es_VIP].sum(axis=1).mean())
            # I_ext = drive + schedule mean
            I_ext[tr, j] = float(np.array(arr)[P1_S:P1_E, t].mean() + drive_tonic[t])
            # I_total from cur
            I_total[tr, j] = float(cur_tr[P1_S:P1_E, t].mean())
    # Verify sum reconstructs I_total within tolerance
    I_recon = I_FF + I_locE + I_locPV + I_locSST + I_locVIP + I_ext
    # I_noise is residual: I_total - recon (should be noise + other inter-area which we ignore, but for V4 L4 other inter-area is none)
    I_noise = I_total - I_recon
    # Compute reconstruction error per trial per neuron
    err = np.abs(I_total - (I_recon + I_noise))  # by definition zero
    # Actually I_noise defined as residual, so error zero. Instead check that I_noise variance is due to noise term
    # For verification, we can check that I_noise mean near 0 and variance matches expected noise scale
    print(f"Reconstruction check: I_total mean {I_total.mean():.3f} recon mean {I_recon.mean():.3f} noise mean {I_noise.mean():.3f}")
    print(f" I_FF mean {I_FF.mean():.4f} I_locE {I_locE.mean():.4f} I_locPV {I_locPV.mean():.4f} I_ext {I_ext.mean():.3f} I_noise {I_noise.mean():.4f}")

    # Component table: A/B contrast vector, norm, trial variance/covariance, correlation with I_FF, covariance contribution to I_total
    components = {
        "I_FF": I_FF,
        "I_locE": I_locE,
        "I_locPV": I_locPV,
        "I_locSST": I_locSST,
        "I_locVIP": I_locVIP,
        "I_ext": I_ext,
        "I_noise": I_noise,
        "I_total": I_total,
    }
    print("\nComponent table p1 window per V4 L4 (15 neurons, 16 trials):")
    for name, X in components.items():
        # A/B contrast
        c0=X[labels==0].mean(axis=0); c1=X[labels==1].mean(axis=0)
        delta=c1-c0
        norm=float(np.linalg.norm(delta))
        # trial variance
        var=float(X.var())
        # correlation with I_FF
        if name != "I_FF":
            # per trial per neuron flattened? Use per trial mean across V4 L4
            # For correlation, use per trial mean across neurons
            y_ff = I_FF.mean(axis=1); y = X.mean(axis=1)
            if np.std(y_ff)>1e-9 and np.std(y)>1e-9:
                r=float(np.corrcoef(y_ff, y)[0,1])
            else:
                r=0.0
        else:
            r=1.0
        # covariance contribution to I_total: Cov(I_k, I_total) / Var(I_total)
        # For each V4 L4 neuron, compute covariance across trials
        # Use mean across neurons
        covs=[]
        for j in range(n_v4):
            xk=X[:, j]; yt=I_total[:, j]
            if np.std(xk)>1e-9 and np.std(yt)>1e-9:
                cov=np.cov(xk, yt)[0,1]
                covs.append(cov)
        cov_mean=float(np.mean(covs)) if covs else 0.0
        # Also decoder
        acc,p,_,_ = perm_p(X, labels, n_perm=50)
        print(f" {name:10} norm {norm:.4f} var {var:.5f} r_with_FF {r:.3f} cov_to_total {cov_mean:.5f} acc {acc:.3f} p {p:.3f}")

    # Variance/covariance accounting: Var(I_total) = sum Var(I_k) + 2 sum Cov(I_k,I_j)
    # Compute across trials per V4 L4 neuron, then mean across neurons
    # Use per trial per neuron I_total vs components
    # For each V4 L4 neuron, compute Var(I_total) and sum of Vars and Covs
    print("\nVariance/covariance accounting per V4 L4 neuron (mean across 15):")
    var_totals=[]
    var_sums=[]
    cov_sums=[]
    for j in range(n_v4):
        # Collect per trial per component for this neuron
        comps_j = np.array([components[k][:, j] for k in ["I_FF","I_locE","I_locPV","I_locSST","I_locVIP","I_ext","I_noise"]])  # (7,16)
        total_j = I_total[:, j]  # (16,)
        var_total = float(np.var(total_j))
        var_sum = float(np.sum([np.var(comps_j[k]) for k in range(comps_j.shape[0])]))
        # Cov sum: 2* sum_{k<j} Cov(k,j)
        cov_sum=0
        for a in range(comps_j.shape[0]):
            for b in range(a+1, comps_j.shape[0]):
                cov_sum += 2*float(np.cov(comps_j[a], comps_j[b])[0,1])
        var_totals.append(var_total); var_sums.append(var_sum); cov_sums.append(cov_sum)
        # Check reconstruct: var_total should equal var_sum + cov_sum within tolerance
        # print per neuron
        # print(f"  V4 L4 {j} var_total {var_total:.5f} var_sum {var_sum:.5f} cov_sum {cov_sum:.5f} sum {var_sum+cov_sum:.5f} err {var_total-(var_sum+cov_sum):.5f}")
    print(f" mean var_total {np.mean(var_totals):.5f} var_sum {np.mean(var_sums):.5f} cov_sum {np.mean(cov_sums):.5f} recon err {np.mean(var_totals)-np.mean(var_sums)-np.mean(cov_sums):.5f}")

    # P classification
    # Use decoder acc for I_FF vs I_total
    acc_FF,_ ,_,_ = perm_p(I_FF, labels, n_perm=50)
    acc_total,_ ,_,_ = perm_p(I_total, labels, n_perm=50)
    print(f"\nP1 check: I_FF acc {acc_FF:.3f} I_total {acc_total:.3f}")
    # Determine P
    # P1: I_FF 1.00 but I_total <0.75 and var_noise dominates
    # Check var_noise vs sig
    var_noise = float(I_noise.var())
    sig_FF = float((I_FF[labels==0].mean(axis=0) - I_FF[labels==1].mean(axis=0)).var())
    # Also check correlation of PV/SST etc with I_FF (structured cancellation)
    # For P2, check if PV/SST carry opposing contrast (r negative)
    # Compute per V4 L4 PV vs FF correlation
    # Use mean across trials per neuron
    # For each V4 L4, correlation between I_FF and I_locPV across trials
    corrs_PV=[]
    for j in range(n_v4):
        a=I_FF[:, j]; b=I_locPV[:, j]
        if np.std(a)>1e-9 and np.std(b)>1e-9:
            corrs_PV.append(float(np.corrcoef(a,b)[0,1]))
    mean_corr_PV=float(np.mean(corrs_PV)) if corrs_PV else 0
    print(f" PV correlation with FF mean {mean_corr_PV:.3f} (negative suggests cancellation)")
    # Also check I_FF vs I_total correlation
    corrs_FF_total=[]
    for j in range(n_v4):
        a=I_FF[:, j]; b=I_total[:, j]
        if np.std(a)>1e-9 and np.std(b)>1e-9:
            corrs_FF_total.append(float(np.corrcoef(a,b)[0,1]))
    print(f" I_FF vs I_total per V4 L4 mean r {np.mean(corrs_FF_total):.3f}")

    # For P1, if I_total not decodable and var_noise >> sig, then P1
    if acc_FF >= 0.75 and acc_total < 0.75 and var_noise > sig_FF*2:
        cand="P1 STOCHASTIC_MASKING"
    elif mean_corr_PV < -0.3:
        cand="P2 STRUCTURED_INHIBITORY_CANCELLATION"
    else:
        # Check other
        cand="P_UNRESOLVED"
    print("candidate", cand)

    # If P1 supported, run noise counterfactual
    if cand=="P1 STOCHASTIC_MASKING":
        print("\nRunning noise counterfactual g_noise 0,0.5,1 (assay-only)")
        for g in [0, 0.5, 1]:
            noise_scale = 0.5 * float(g)
            all_ec_g=[]; all_cur_g=[]; all_V_g=[]; labels_g=[]
            for label, arr, seed in trials:
                step_fn, init = compile_step_fn(model, dt_ms=DT_MS, kernel="baseline", record_edge_current=True, record_current_trace=True, noise_scale=noise_scale)
                init_s = continuation_state_from_model(model, seed=seed)
                state, outs = run_continuation(step_fn, init_s, jax.numpy.asarray(arr, dtype=jax.numpy.float32))
                ec=np.asarray(outs[5]); cur=np.asarray(outs[6]); V=np.asarray(outs[0])
                all_ec_g.append(ec); all_cur_g.append(cur); all_V_g.append(V)
                labels_g.append(0 if label=="A" else 1)
            all_ec_g=np.array(all_ec_g); all_cur_g=np.array(all_cur_g); all_V_g=np.array(all_V_g); labels_g=np.array(labels_g)
            n_trials_g=len(labels_g); n_v4_g=len(v4_l4_idx)
            I_FF_g=np.zeros((n_trials_g, n_v4_g))
            I_total_g=np.zeros((n_trials_g, n_v4_g))
            Vm_g=np.zeros((n_trials_g, n_v4_g))
            for tr in range(n_trials_g):
                for j, t in enumerate(v4_l4_idx):
                    es_ff=[ei for ei in ff_mask if int(post[ei])==t]
                    if es_ff:
                        I_FF_g[tr, j] = float(all_ec_g[tr, P1_S:P1_E, es_ff].sum(axis=1).mean())
                    I_total_g[tr, j] = float(all_cur_g[tr, P1_S:P1_E, t].mean())
                    Vm_g[tr, j] = float(all_V_g[tr, P1_S:P1_E, t].mean())
            acc_IFF_g,_,_,_ = perm_p(I_FF_g, labels_g, n_perm=30)
            acc_Itot_g,_,_,_ = perm_p(I_total_g, labels_g, n_perm=30)
            acc_Vm_g,_,_,_ = perm_p(Vm_g, labels_g, n_perm=30)
            print(f" g {g}: I_FF acc {acc_IFF_g:.3f} I_total {acc_Itot_g:.3f} Vm {acc_Vm_g:.3f} (expect I_FF stable, I_total/Vm increase if P1 true)")
        print("If P1 genuine, I_total/Vm should increase as g->0")

    # Save
    out=pathlib.Path("results/w4a_P_attribution.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out,"w") as f:
        json.dump({"candidate": cand, "var_noise": float(var_noise), "sig_FF": float(sig_FF), "acc_FF": float(acc_FF), "acc_total": float(acc_total)}, f, indent=2)
    print(f"saved to {out}")
    print("Observability: JaxFNE now exposes I_total and edge_current, but I_tonic/schedule/noise per step still need reconstruction via drive+schedule+syn vs current; full I_net accounting now closed via da2d198, but schedule/noise still not directly exposed as separate traces — reported as partial blocker, but total is now directly recorded.")

if __name__=="__main__":
    run()
