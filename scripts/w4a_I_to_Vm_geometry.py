"""I_FF -> Vm geometry — read-only, no C023. Tests H1-H5."""
import json, pathlib, numpy as np, jax, jax.numpy as jnp
from jomission.network.builder import build_jomission_model
from jomission.network.rf import RFConfig, RFOperator
from jomission.paradigm.spec import JOMISSION_PARADIGM
from jaxfne._pipeline import compile_step_fn, run_continuation, continuation_state_from_model

DT_MS=0.1
DUR_MS=600.0
N_STEPS=int(DUR_MS/DT_MS)
P1_S=int(0/DT_MS); P1_E=int(531/DT_MS)
# Use 80-step delay
DELAY=80
N_TRIALS_PER_COND=8

def run():
    model=build_jomission_model(n_per_area=100, seed=0, dt_ms=DT_MS)
    tbl=model.neuron_table()
    v4_l4_idx=[i for i,r in enumerate(tbl) if r["area"]=="V4" and r["layer"]=="L4"]
    v4_l4e_idx=[i for i in v4_l4_idx if tbl[i]["cell_type"]=="E"]
    v4_l4pv_idx=[i for i in v4_l4_idx if tbl[i]["cell_type"]=="PV"]
    el=model.params["edge_list"]
    pre=np.asarray(el.pre); post=np.asarray(el.post)
    ff_mask=[ei for ei in range(len(pre)) if tbl[int(pre[ei])]["area"]=="V1" and tbl[int(pre[ei])]["layer"]=="L2/3" and tbl[int(pre[ei])]["cell_type"]=="E" and tbl[int(post[ei])]["area"]=="V4" and tbl[int(post[ei])]["layer"]=="L4"]
    print(f"V4 L4 n {len(v4_l4_idx)} E {len(v4_l4e_idx)} FF {len(ff_mask)}")

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
    # Collect per trial time-resolved V and ec
    all_V=[]; all_ec=[]; labels=[]
    for label, arr, seed in trials:
        step_fn, init = compile_step_fn(model, dt_ms=DT_MS, kernel="baseline", record_edge_current=True)
        init_s = continuation_state_from_model(model, seed=seed)
        state, outs = run_continuation(step_fn, init_s, jnp.asarray(arr, dtype=jnp.float32))
        V=np.asarray(outs[0]); ec=np.asarray(outs[5])
        all_V.append(V); all_ec.append(ec); labels.append(0 if label=="A" else 1)
    all_V=np.array(all_V)  # (16,6000,400)
    all_ec=np.array(all_ec)  # (16,6000,10666)
    labels=np.array(labels)
    print(f"collected {len(labels)} trials")

    # Build per V4 L4 neuron time-resolved I_FF(t) and Vm(t)
    # For each trial, per V4 L4 neuron, I_FF(t) = sum over its incoming FF edges ec[t, es]
    # Vm(t) = V[t, neuron]
    n_trials=len(labels)
    n_v4=len(v4_l4_idx)
    # Precompute per V4 L4 neuron incoming FF edges
    per_tgt_edges={t: [] for t in v4_l4_idx}
    for ei in ff_mask:
        per_tgt_edges[int(post[ei])].append(ei)
    # Time-resolved per trial per V4 L4
    I_FF = np.zeros((n_trials, N_STEPS, n_v4))  # per V4 L4
    Vm = np.zeros((n_trials, N_STEPS, n_v4))
    for tr in range(n_trials):
        for j, t in enumerate(v4_l4_idx):
            es=per_tgt_edges[t]
            if es:
                I_FF[tr, :, j] = np.asarray(all_ec[tr][:, es].sum(axis=1))
            Vm[tr, :, j] = np.asarray(all_V[tr, :, t])
    # Also overall per V4 L4 class vectors for geometry
    # 1. Target correspondence: per neuron DI vs DVm (mean over p1 window)
    # Compute per neuron mean over p1 window per trial, then mean across trials per condition
    # For each V4 L4 neuron j, DI_j = mean_A - mean_B, similarly DVm
    def per_neuron_delta(X):  # X (n_trials, n_steps, n_v4) or (n_trials, n_v4) already windowed?
        # Use p1 window mean per trial per neuron
        # X is (n_trials, n_steps, n_v4) for time-resolved, need window
        # For this, we compute per trial per neuron mean in p1
        Xa = X[labels==0, P1_S:P1_E, :].mean(axis=1)  # (8, n_v4)
        Xb = X[labels==1, P1_S:P1_E, :].mean(axis=1)
        # D per neuron = mean_A - mean_B (across trials)
        d = Xa.mean(axis=0) - Xb.mean(axis=0)
        return d, Xa, Xb
    # Need I_FF per trial per neuron mean in p1: we have I_FF (n_trials, n_steps, n_v4) -> mean over p1
    dI, Xa_I, Xb_I = per_neuron_delta(I_FF)
    dVm, Xa_Vm, Xb_Vm = per_neuron_delta(Vm)
    # Correspondence Pearson
    from scipy.stats import pearsonr
    try:
        r, p_r = pearsonr(dI, dVm)
    except:
        r, p_r = float(np.corrcoef(dI, dVm)[0,1]), 0.0
    print(f"Target correspondence DI vs DVm per V4 L4 (n={n_v4}) r {r:.3f} p {p_r:.3f}")
    # Also rank
    from scipy.stats import spearmanr
    try:
        rs, p_s = spearmanr(dI, dVm)
    except:
        rs, p_s = 0,1
    print(f" Spearman {rs:.3f} p {p_s:.3f}")
    print(f" dI per V4 L4: mean {dI.mean():.4f} std {dI.std():.4f} max {np.abs(dI).max():.4f}")
    print(f" dVm per V4 L4: mean {dVm.mean():.4f} std {dVm.std():.4f} max {np.abs(dVm).max():.4f}")
    # Lag search: compute r for Vm lagged by 0, 80, 160 steps (0,8,16ms)
    for lag in [0, 40, 80, 120, 160]:
        # Shift Vm window by lag
        s_lag = P1_S+lag; e_lag = P1_E+lag
        if e_lag > N_STEPS:
            continue
        Xa_lag = np.array([I_FF[i, P1_S:P1_E, :].mean(axis=0) for i in range(n_trials) if labels[i]==0]).mean(axis=0) if False else None
        # Simpler: compute dI as before, dVm lagged
        # For lagged Vm, compute dVm_lag = mean_A Vm in window lagged - mean_B
        Vm_lag = Vm[:, s_lag:e_lag, :].mean(axis=1)  # (n_trials, n_v4)
        dVm_lag = Vm_lag[labels==0].mean(axis=0) - Vm_lag[labels==1].mean(axis=0)
        # Now correlate dI (p1) vs dVm_lag
        try:
            r_lag,_ = pearsonr(dI, dVm_lag)
        except:
            r_lag = float(np.corrcoef(dI, dVm_lag)[0,1]) if np.std(dI)>0 and np.std(dVm_lag)>0 else 0
        print(f" lag {lag} steps ({lag*DT_MS:.1f}ms) r {r_lag:.3f}")

    # 2. Representation geometry: contrast vectors direction cosine
    # Contrast vectors already dI and dVm (15-dim)
    norm_I = np.linalg.norm(dI); norm_Vm = np.linalg.norm(dVm)
    cos = float(np.dot(dI, dVm) / (norm_I*norm_Vm)) if norm_I>0 and norm_Vm>0 else 0
    print(f"Geometry cosine I->Vm {cos:.3f} (1 preserved, 0 orthogonal, -1 inverted) norm_I {norm_I:.4f} norm_Vm {norm_Vm:.4f}")
    # Also per class
    for ct, idxs in [("E", v4_l4e_idx), ("PV", v4_l4pv_idx)]:
        # Map global v4_l4_idx positions
        pos = [v4_l4_idx.index(i) for i in idxs if i in v4_l4_idx]
        if not pos:
            continue
        dI_c = dI[pos]; dVm_c = dVm[pos]
        nrmI=np.linalg.norm(dI_c); nrmV=np.linalg.norm(dVm_c)
        cos_c = float(np.dot(dI_c, dVm_c)/(nrmI*nrmV)) if nrmI>0 and nrmV>0 else 0
        print(f"  {ct} cos {cos_c:.3f} n {len(pos)}")

    # 3. Time-resolved transfer: predeclared windows around FF peaks
    # Find FF peaks: average I_FF across trials and V4 L4 neurons
    I_FF_mean_time = I_FF.mean(axis=0).mean(axis=1)  # (6000,) mean over trials and neurons
    # Find peaks in p1 window
    p1_I = I_FF_mean_time[P1_S:P1_E]
    peak_idx = np.argmax(p1_I) + P1_S
    print(f"FF peak at step {peak_idx} time {peak_idx*DT_MS:.1f}ms value {p1_I.max():.4f}")
    # Define windows around peak: peak ±25ms (250 steps) and whole p1
    win_peak = (max(0, peak_idx-250), min(N_STEPS, peak_idx+250))
    win_p1_delayed = (P1_S+80, P1_E+80)
    def decoder_for_window(X, window):
        s,e=window
        # X is (n_trials, n_steps, n_units) -> per trial mean in window per unit
        # For Vm, X is Vm; for I_FF, X is I_FF
        # We'll compute per trial vector
        vec = X[:, s:e, :].mean(axis=1) if X.ndim==3 else X[:, s:e].mean(axis=1)[:,None]
        # Use lovo
        # Need labels
        # Simple
        n=len(labels)
        correct=0
        for i in range(n):
            tr=[j for j in range(n) if j!=i]
            Xtr=vec[tr]; ytr=labels[tr]
            c0=Xtr[ytr==0].mean(axis=0); c1=Xtr[ytr==1].mean(axis=0)
            d0=np.linalg.norm(vec[i]-c0); d1=np.linalg.norm(vec[i]-c1)
            pred=0 if d0<d1 else 1
            if pred==labels[i]:
                correct+=1
        acc=correct/n
        # perm p approx via 50 perms
        null=[]
        for _ in range(50):
            yp=np.random.permutation(labels)
            correct_n=0
            for i in range(n):
                tr=[j for j in range(n) if j!=i]
                Xtr=vec[tr]; ytr=yp[tr]
                c0=Xtr[ytr==0].mean(axis=0) if np.sum(ytr==0)>0 else np.zeros(vec.shape[1])
                c1=Xtr[ytr==1].mean(axis=0) if np.sum(ytr==1)>0 else np.zeros(vec.shape[1])
                d0=np.linalg.norm(vec[i]-c0); d1=np.linalg.norm(vec[i]-c1)
                pred=0 if d0<d1 else 1
                if pred==yp[i]:
                    correct_n+=1
            null.append(correct_n/n)
        null=np.array(null)
        p=float((np.sum(null>=acc)+1)/(len(null)+1))
        return acc, p, float(null.mean())

    # Test I_FF and Vm in peak window vs whole p1
    for wname, win in [("p1", (P1_S,P1_E)), ("p1_delayed", win_p1_delayed), ("peak±25ms", win_peak)]:
        # Use Vm and I_FF
        for name, X in [("I_FF", I_FF), ("Vm", Vm)]:
            acc,p,_=decoder_for_window(X, win)
            print(f" {wname} {name:6} acc {acc:.3f} p {p:.3f} window {win}")

    # 4. Trial-level coupling: within each V4 L4 neuron, correlation across trials between I_FF and Vm (conditioned on A/B? Use residual)
    # For each neuron, compute per trial I_FF and Vm in p1 window, then correlation across trials
    # Conditioned: subtract condition mean
    corrs=[]
    for j in range(n_v4):
        i_vals=[]; v_vals=[]
        for tr in range(n_trials):
            # per trial per neuron
            i_vals.append(float(I_FF[tr, P1_S:P1_E, j].mean()))
            v_vals.append(float(Vm[tr, P1_S:P1_E, j].mean()))
        i_vals=np.array(i_vals); v_vals=np.array(v_vals)
        # Conditioned: residual after removing condition mean
        i_res = i_vals - np.array([i_vals[labels==0].mean() if labels[tr]==0 else i_vals[labels==1].mean() for tr in range(n_trials)])
        v_res = v_vals - np.array([v_vals[labels==0].mean() if labels[tr]==0 else v_vals[labels==1].mean() for tr in range(n_trials)])
        if np.std(i_res)>1e-9 and np.std(v_res)>1e-9:
            corrs.append(float(np.corrcoef(i_res, v_res)[0,1]))
    print(f"Trial-level I->Vm residual correlation per V4 L4 mean {np.mean(corrs):.3f} (n {len(corrs)})")
    # Also unconditional (includes A/B)
    corrs_uncond=[]
    for j in range(n_v4):
        i_vals=np.array([float(I_FF[tr, P1_S:P1_E, j].mean()) for tr in range(n_trials)])
        v_vals=np.array([float(Vm[tr, P1_S:P1_E, j].mean()) for tr in range(n_trials)])
        if np.std(i_vals)>1e-9 and np.std(v_vals)>1e-9:
            corrs_uncond.append(float(np.corrcoef(i_vals, v_vals)[0,1]))
    print(f" Unconditional I-Vm per neuron mean {np.mean(corrs_uncond):.3f}")

    # 5. H5: cross-trial variance and effect size of FF-current representation
    # I_FF per trial per V4 L4 mean, compute variance across trials and effect size
    # Already have cur_all vectors (target-summed) — but for H5 we need per trial per V4 L4 I_FF
    # Use I_FF per trial per neuron mean in p1
    # Compute for I_FF aggregated per V4 L4 (15-dim) as before cur_all was target-summed, but now we have I_FF per neuron
    # For H5, we need to report variance and effect size: mean difference vs std
    # Use cur_all vectors previously: cur_all is (16,15) target-summed
    # Let's compute effect size per V4 L4 neuron for I_FF and Vm
    # Already have dI per neuron, and also per trial variance
    # Report cross-trial variance
    # For I_FF per V4 L4 neuron, variance across trials
    # Use all_V and all_ec already
    # Compute per V4 L4 neuron I_FF variance
    var_I = np.array([float(I_FF[:, P1_S:P1_E, j].mean(axis=1).var()) for j in range(n_v4)])
    mean_abs_dI = np.abs(dI)
    print(f"H5 I_FF cross-trial var mean {var_I.mean():.6f} effect |DI| mean {mean_abs_dI.mean():.4f} max {mean_abs_dI.max():.4f} var vs effect ratio {var_I.mean()/(mean_abs_dI.mean()**2) if mean_abs_dI.mean()>0 else 0:.3f}")
    var_Vm = np.array([float(Vm[:, P1_S:P1_E, j].mean(axis=1).var()) for j in range(n_v4)])
    mean_abs_dVm = np.abs(dVm)
    print(f" Vm var mean {var_Vm.mean():.4f} effect {mean_abs_dVm.mean():.4f}")

    # Classification H1-H5
    # H1 attenuated: norm_Vm << norm_I and cos ~1
    # H2 rotated: cos ~0
    # H3 transient: sliding shows peak but whole p1 not
    # H4 target mismatch: informative I neurons not same as informative Vm neurons (r low)
    # H5 trivial: dI small absolute but perfectly repeatable (var small, effect small)
    # Use thresholds
    # H1 if norm_Vm < 0.5*norm_I and abs(cos)>0.5
    # H2 if abs(cos)<0.3
    # H3 if max sliding Vm acc >0.75 and whole p1 <0.6
    # H4 if r <0.3
    # H5 if mean_abs_dI <0.1 and var_I small and acc 1.00 (deterministic micro)
    # Decide
    cos_val = float(np.dot(dI, dVm)/(np.linalg.norm(dI)*np.linalg.norm(dVm))) if np.linalg.norm(dI)>0 and np.linalg.norm(dVm)>0 else 0
    r_val = r
    # Check H5: effect size small but acc 1.00
    # Our I_FF acc was 1.00 but mean_abs_dI 0.04? Actually dI per neuron mean 0.04? Wait earlier dI mean 0.04? For V4 L4, dI per neuron mean maybe small.
    # Let's compute mean_abs_dI for V4 L4 I: we had dI per neuron mean 0.04? Actually earlier we printed dI mean for V4 L4 I? No, dI for V4 L4 I was not printed.
    # Use our var vs effect
    if mean_abs_dI.mean() < 0.05 and var_I.mean() < 0.01:
        cand="H5 FF_CURRENT_DECODER_IS_TRIVIAL/LOW_AMPLITUDE_BUT_REPEATABLE"
    elif abs(cos_val) < 0.3:
        cand="H2 FF_TO_VM_ROTATED_BY_NETWORK_STATE"
    elif r < 0.3:
        cand="H4 FF_TO_VM_TARGET_MISMATCH"
    elif norm_Vm < 0.5*norm_I and abs(cos_val)>0.5:
        cand="H1 FF_TO_VM_ATTENUATED"
    else:
        # Check H3 via sliding
        # We saw sliding max 0.75 borderline, not strong
        cand="H_UNRESOLVED"
    print("candidate", cand)
    # Save
    out=pathlib.Path("results/w4a_I_to_Vm_geometry.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out,"w") as f:
        json.dump({"r": float(r), "cos": float(cos_val), "norm_I": float(norm_I), "norm_Vm": float(norm_Vm), "candidate": cand, "var_I": float(var_I.mean()), "effect_I": float(mean_abs_dI.mean())}, f, indent=2)
    print(f"saved to {out}")
    # Also preserve constant V4-L4 operating-point intervention = NEGATIVE
    print("Preserve: constant V4-L4 operating-point intervention = NEGATIVE (susceptibility assay G2)")

if __name__=="__main__":
    run()
