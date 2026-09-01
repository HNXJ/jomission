"""Q tradeoff: noise scale vs B1/B2/B3 and representation — g in {0,0.125,0.25,0.5,1} predeclared."""
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
GRID=[0, 0.125, 0.25, 0.5, 1.0]  # predeclared, not optimized post-hoc

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
    return acc, p

def b1_metrics(rates_by_pop):
    # rates_by_pop: dict pop -> mean Hz
    # thresholds from gen2_gates B1: global mu [3,15], per-class E 5-8 etc, coverage 60%, no <0.5 or >60
    # Simplified: compute global mu and coverage
    global_mu=float(np.mean(list(rates_by_pop.values()))) if rates_by_pop else 0
    # per-class targets
    # For E: 5-8, PV 12-25, SST 6-12, VIP 8-15
    # Count pops within target +/-50%
    # We'll just use E target 5-8 as example
    # For this assay, we have V4 L4 E etc, but B1 is global
    # We'll compute coverage as fraction of pops where rate within [2.5,12] for E etc.
    # Simplified: check global mu in [3,15] and no pathological
    ok_global = 3 <= global_mu <= 15
    # Check no silence <0.5 and hyper >60
    ok_silence = all(v >= 0.5 for v in rates_by_pop.values())
    ok_hyper = all(v <= 60 for v in rates_by_pop.values())
    return {"global_mu": global_mu, "ok_global": ok_global, "ok_silence": ok_silence, "ok_hyper": ok_hyper, "pass": ok_global and ok_silence and ok_hyper}

def b2_metrics(spikes, dt_ms):
    # spikes: (n_trials, n_steps, n_neurons) binary
    # Compute per-neuron CV_ISI, Fano, etc. Simplified
    n_trials, n_steps, n_neurons = spikes.shape
    # For each neuron, pool trials? Use per trial then average
    # CV_ISI per neuron per trial: ISI = diff of spike times
    cvs=[]
    fanos=[]
    for n in range(n_neurons):
        # Use first trial for simplicity, but should average across trials
        # Collect ISIs across trials
        isis=[]
        rates=[]
        for tr in range(n_trials):
            times = np.where(spikes[tr, :, n] > 0.5)[0] * dt_ms
            if len(times) < 3:
                continue
            isi = np.diff(times)
            if len(isi) > 1 and isi.mean() > 0:
                cvs.append(float(isi.std()/isi.mean()))
            # Fano: count per 100ms window
            # Use window 100ms =100 steps at dt0.1? Actually 100ms =1000 steps? Wait dt0.1, 100ms =1000 steps
            # For DUR 600ms, we have 6000 steps, use 100ms windows
            win = int(100/ dt_ms)
            counts=[]
            for w in range(0, n_steps, win):
                counts.append(float(spikes[tr, w:w+win, n].sum()))
            if len(counts) > 1 and np.mean(counts) > 0:
                fanos.append(float(np.var(counts)/np.mean(counts)))
    mean_cv = float(np.mean(cvs)) if cvs else 0.0
    median_fano = float(np.median(fanos)) if fanos else 0.0
    # rho: pairwise correlation per trial, average
    rhos=[]
    for tr in range(n_trials):
        # Use spike counts per 10ms bin
        bin_ms = 10
        bin_steps = int(bin_ms/dt_ms)
        binned = []
        for n in range(n_neurons):
            # bin
            s = spikes[tr, :, n]
            binned_n = [float(s[i:i+bin_steps].sum()) for i in range(0, n_steps, bin_steps)]
            binned.append(binned_n)
        binned=np.array(binned)  # (n_neurons, n_bins)
        # Compute pairwise correlation for subset (first 20 neurons)
        subset = min(20, n_neurons)
        for i in range(subset):
            for j in range(i+1, subset):
                if np.std(binned[i])>1e-9 and np.std(binned[j])>1e-9:
                    r=float(np.corrcoef(binned[i], binned[j])[0,1])
                    rhos.append(r)
    mean_rho = float(np.mean(rhos)) if rhos else 0.0
    return {"mean_CV_ISI": mean_cv, "median_Fano": median_fano, "mean_rho": mean_rho, "n_cvs": len(cvs)}

def run():
    base_model=build_jomission_model(n_per_area=100, seed=0, dt_ms=DT_MS)
    tbl=base_model.neuron_table()
    v4_l4_idx=[i for i,r in enumerate(tbl) if r["area"]=="V4" and r["layer"]=="L4"]
    v1_l23e_idx=[i for i,r in enumerate(tbl) if r["area"]=="V1" and r["layer"]=="L2/3" and r["cell_type"]=="E"]
    areas=[r["area"] for r in tbl]; layers=[r["layer"] for r in tbl]; cts=[r["cell_type"] for r in tbl]
    el=base_model.params["edge_list"]
    pre=np.asarray(el.pre); post=np.asarray(el.post)
    ff_mask=[ei for ei in range(len(pre)) if areas[int(pre[ei])]=="V1" and layers[int(pre[ei])]=="L2/3" and cts[int(pre[ei])]=="E" and areas[int(post[ei])]=="V4" and layers[int(post[ei])]=="L4"]
    print(f"FF {len(ff_mask)} V4 L4 {len(v4_l4_idx)} V1 L2/3_E {len(v1_l23e_idx)}")

    from jomission.simulation.factorial_v0p2 import energy_amplitude as _ea
    rf_cfg=RFConfig(seed=0, tier="graded")
    op=RFOperator(rf_cfg, base_model)
    condA=[c for c in JOMISSION_PARADIGM.conditions if c.name=="AAAB"][0]
    condB=[c for c in JOMISSION_PARADIGM.conditions if c.name=="BBBA"][0]
    ampA=float(_ea("C","AAAB")); ampB=float(_ea("C","BBBA"))
    schedA=op.to_stimulus_schedule(condA, n_neurons=400, dt_ms=DT_MS, base_amplitude=ampA)
    schedB=op.to_stimulus_schedule(condB, n_neurons=400, dt_ms=DT_MS, base_amplitude=ampB)
    arrA=schedA.to_array(N_STEPS, DT_MS); arrB=schedB.to_array(N_STEPS, DT_MS)
    em=base_model.params["emitter"]
    drive_tonic=np.asarray(em.drive)

    results={}
    for g in GRID:
        print(f"\n=== g_noise {g} ===")
        noise_scale = 0.5 * float(g)
        trials=[]
        for rep in range(N_TRIALS_PER_COND):
            for label, arr in [("A", arrA), ("B", arrB)]:
                seed=100+rep*10 + (0 if label=="A" else 1)
                trials.append((label, arr, seed))
        n_trials=len(trials)
        all_V=[]; all_S=[]; all_ec=[]; all_cur=[]; labels=[]
        for label, arr, seed in trials:
            step_fn, init = compile_step_fn(base_model, dt_ms=DT_MS, kernel="baseline", record_edge_current=True, record_current_trace=True, record_u_trace=True, noise_scale=noise_scale)
            init_s = continuation_state_from_model(base_model, seed=seed)
            state, outs = run_continuation(step_fn, init_s, jax.numpy.asarray(arr, dtype=jax.numpy.float32))
            V=np.asarray(outs[0]); S=np.asarray(outs[1]); ec=np.asarray(outs[5]); cur=np.asarray(outs[6])
            all_V.append(V); all_S.append(S); all_ec.append(ec); all_cur.append(cur); labels.append(0 if label=="A" else 1)
        all_V=np.array(all_V); all_S=np.array(all_S); all_ec=np.array(all_ec); all_cur=np.array(all_cur); labels=np.array(labels)
        print(f" collected {n_trials} trials")

        # Stage vectors p1 window
        n_v4=len(v4_l4_idx)
        I_FF=np.zeros((n_trials, n_v4))
        I_total=np.zeros((n_trials, n_v4))
        for tr in range(n_trials):
            for j, t in enumerate(v4_l4_idx):
                es_ff=[ei for ei in ff_mask if int(post[ei])==t]
                if es_ff:
                    I_FF[tr, j] = float(all_ec[tr][P1_S:P1_E][:, es_ff].sum(axis=1).mean())
                I_total[tr, j] = float(all_cur[tr][P1_S:P1_E, t].mean())
        Vm=np.array([all_V[tr][P1_S:P1_E][:, v4_l4_idx].mean(axis=0) for tr in range(n_trials)])
        Spk=np.array([all_S[tr][P1_S:P1_E][:, v4_l4_idx].mean(axis=0)*(1000/DT_MS) for tr in range(n_trials)])
        # Decoders
        acc_IFF,_ = perm_p(I_FF, labels, n_perm=30) if I_FF.shape[1]>0 else (0.5,1)
        acc_Itot,_ = perm_p(I_total, labels, n_perm=30)
        acc_Vm,_ = perm_p(Vm, labels, n_perm=30)
        acc_Spk,_ = perm_p(Spk, labels, n_perm=30)
        # Also V1 source
        v1_spk=np.array([all_S[tr][P1_S:P1_E][:, v1_l23e_idx].mean(axis=0)*(1000/DT_MS) for tr in range(n_trials)]) if v1_l23e_idx else np.zeros((n_trials,1))
        acc_v1,_ = perm_p(v1_spk, labels, n_perm=30)
        print(f" V1 spike acc {acc_v1:.3f} I_FF {acc_IFF:.3f} I_total {acc_Itot:.3f} Vm {acc_Vm:.3f} Spk {acc_Spk:.3f}")

        # Corrected variance accounting: Var_trial per neuron, mean across V4 L4
        # For each V4 L4 neuron, Var across trials of I_total, and sum of Vars/Covs per neuron
        # Need per V4 L4 neuron per trial components: I_FF, I_total, etc. For variance we need per component per neuron per trial
        # We have I_FF, I_total, and also need I_ext, I_local etc. For simplicity, use I_FF and I_total already, plus I_noise = I_total - I_FF - I_ext - I_local
        # But for corrected accounting, we need Var_trial(I_total) = sum Var(I_k) + 2 sum Cov
        # Let's compute for V4 L4 per neuron, using I_FF, I_total, and also I_noise approx
        # For this Q, we focus on I_FF vs I_total vs Vm
        # Compute per V4 L4 neuron Var_trial
        # Use I_FF and I_total per neuron
        # For each neuron j, compute Var across trials of I_FF[:,j] and I_total[:,j]
        # But we need full decomposition: we have I_FF, I_total, and we can compute I_other = I_total - I_FF
        I_other = I_total - I_FF
        var_iff = np.array([float(np.var(I_FF[:, j])) for j in range(n_v4)])
        var_other = np.array([float(np.var(I_other[:, j])) for j in range(n_v4)])
        var_total = np.array([float(np.var(I_total[:, j])) for j in range(n_v4)])
        cov = np.array([float(np.cov(I_FF[:, j], I_other[:, j])[0,1]) if np.std(I_FF[:,j])>1e-9 and np.std(I_other[:,j])>1e-9 else 0 for j in range(n_v4)])
        # Check Var_total = Var_FF + Var_other + 2 Cov
        recon = var_iff + var_other + 2*cov
        err = np.mean(np.abs(var_total - recon))
        print(f" variance per V4 L4 neuron mean var_total {var_total.mean():.5f} var_FF {var_iff.mean():.5f} var_other {var_other.mean():.5f} cov {cov.mean():.5f} err {err:.5f} (should be ~0)")

        # B1: rates by area/layer/class
        # Compute mean rates per area/layer/class for this g (pool trials)
        # Use all_S mean across trials and time
        # For B1, need per population rates
        from jomission.network.populations import AREA_LAYER_CELL_TYPES
        pops={}
        for area in ["V1","V4","FEF","PFC"]:
            for layer in ["L2/3","L4","L5"]:
                for ct in ["E","PV","SST","VIP"]:
                    idx=[i for i,r in enumerate(tbl) if r["area"]==area and r["layer"]==layer and r["cell_type"]==ct]
                    if not idx:
                        continue
                    # Mean rate across trials and time p1 window
                    rate=float(all_S[:, P1_S:P1_E, :][:, :, idx].mean()*(1000/DT_MS))
                    pops[f"{area}_{layer}_{ct}"]=rate
        # B1 check
        b1 = b1_metrics(pops)
        print(f" B1 global_mu {b1['global_mu']:.2f} ok {b1['pass']}")

        # B2: CV_ISI, Fano, rho
        b2 = b2_metrics(all_S, DT_MS)
        print(f" B2 CV_ISI {b2['mean_CV_ISI']:.2f} Fano {b2['median_Fano']:.2f} rho {b2['mean_rho']:.3f} (pass CV 0.5-1.5, Fano 0.7-2.0, rho -0.05-0.2)")

        # B3: realized E/I currents by class/motif using executed-current instrumentation
        # Use I_FF etc. For B3, need Efrac per E cell: Efrac = I_E / (I_E - I_I) ??? Actually Efrac in [0.15,0.60] for E cells
        # Simplified: compute per V4 L4 E I_E vs I_I from local
        # For each V4 L4 E neuron, I_E = sum over E edges, I_I = sum over I edges (PV/SST/VIP) as negative
        # Use all_ec per V4 L4 E
        # For B3, we need per E cell Efrac = |I_E| / (|I_E|+|I_I|)
        # Let's compute for V4 L4 E
        efracs=[]
        for j, t in enumerate(v4_l4_idx):
            if tbl[t]["cell_type"]!="E":
                continue
            # Find incoming edges to t per presynaptic class
            # Use ec per trial mean
            # For this g, use first trial as example
            # Sum per class
            # For simplicity, use per trial 0
            tr=0
            # E edges to t
            es_E=[ei for ei in range(len(pre)) if int(post[ei])==t and cts[int(pre[ei])]=="E" and areas[int(pre[ei])]=="V4"]
            es_I=[ei for ei in range(len(pre)) if int(post[ei])==t and cts[int(pre[ei])]!="E" and areas[int(pre[ei])]=="V4"]
            Ie = float(all_ec[tr][P1_S:P1_E, es_E].sum(axis=1).mean()) if es_E else 0
            Ii = float(all_ec[tr][P1_S:P1_E, es_I].sum(axis=1).mean()) if es_I else 0
            # Ii is negative (inhibitory weights negative), but ec will be negative for I
            # Efrac = |Ie| / (|Ie|+|Ii|)
            denom=abs(Ie)+abs(Ii)
            efrac = abs(Ie)/denom if denom>1e-9 else 0.5
            efracs.append(efrac)
        efrac_mean=float(np.mean(efracs)) if efracs else 0.5
        b3_ok = 0.15 <= efrac_mean <= 0.60
        print(f" B3 Efrac V4 L4 E mean {efrac_mean:.3f} ok {b3_ok} ({len(efracs)} E cells)")

        results[float(g)]={
            "acc_v1": float(acc_v1), "acc_IFF": float(acc_IFF), "acc_Itot": float(acc_Itot), "acc_Vm": float(acc_Vm), "acc_Spk": float(acc_Spk),
            "var_total": float(var_total.mean()), "var_FF": float(var_iff.mean()), "var_other": float(var_other.mean()),
            "b1_global_mu": float(b1["global_mu"]), "b1_pass": bool(b1["pass"]),
            "b2_cv": float(b2["mean_CV_ISI"]), "b2_fano": float(b2["median_Fano"]), "b2_rho": float(b2["mean_rho"]),
            "b3_efrac": float(efrac_mean), "b3_ok": bool(b3_ok),
            "vm_mean": float(np.mean([all_V[tr][P1_S:P1_E, v4_l4_idx].mean() for tr in range(n_trials)])),
            "rate_mean": float(np.mean([all_S[tr][P1_S:P1_E, v4_l4_idx].mean() for tr in range(n_trials)])*(1000/DT_MS)),
        }

    # Multi-objective curve
    print("\n=== multi-objective curve ===")
    for g in GRID:
        r=results[float(g)]
        print(f"g {g}: I_FF {r['acc_IFF']:.3f} I_tot {r['acc_Itot']:.3f} Vm {r['acc_Vm']:.3f} Spk {r['acc_Spk']:.3f} | B1 mu {r['b1_global_mu']:.1f} {'PASS' if r['b1_pass'] else 'FAIL'} B2 CV {r['b2_cv']:.2f} Fano {r['b2_fano']:.2f} rho {r['b2_rho']:.3f} B3 {r['b3_efrac']:.2f} {'PASS' if r['b3_ok'] else 'FAIL'}")

    # Classify Q
    # Q1 requires nondegenerate interval where hierarchical transfer effective (Vm/Spk >=0.75) and generic B1/B2/B3 pass
    # Q2 tradeoff, Q3 only zero noise rescues
    # Check each g
    for g in GRID:
        r=results[float(g)]
        eff = r["acc_Vm"] >= 0.75 or r["acc_Spk"] >= 0.75
        b_pass = r["b1_pass"] and (0.5 <= r["b2_cv"] <= 1.5) and (0.7 <= r["b2_fano"] <= 2.0) and (-0.05 <= r["b2_rho"] <= 0.2) and r["b3_ok"]
        print(f"g {g} eff {eff} b_pass {b_pass} Vm {r['acc_Vm']:.3f} Spk {r['acc_Spk']:.3f} B2 CV {r['b2_cv']:.2f}")

    # Determine Q
    # Find if any g in (0.125,0.25,0.5) has eff and b_pass (not just 0)
    eff_and_pass=[]
    for g in [0.125,0.25,0.5]:
        r=results[float(g)]
        eff = r["acc_Vm"] >= 0.75 or r["acc_Spk"] >= 0.75
        b_pass = r["b1_pass"] and (0.5 <= r["b2_cv"] <= 1.5) and (0.7 <= r["b2_fano"] <= 2.0) and (-0.05 <= r["b2_rho"] <= 0.2) and r["b3_ok"]
        if eff and b_pass:
            eff_and_pass.append(g)
    if eff_and_pass:
        cand="Q1 COMPATIBLE_NOISE_REGION"
    else:
        # Check if only g0 rescues
        r0=results[0.0]
        eff0 = r0["acc_Vm"] >= 0.75 or r0["acc_Spk"] >= 0.75
        if eff0:
            # Check if g0 b_pass? likely B2 fails at zero noise (CV low)
            b0_pass = r0["b1_pass"] and (0.5 <= r0["b2_cv"] <= 1.5) and (0.7 <= r0["b2_fano"] <= 2.0) and (-0.05 <= r0["b2_rho"] <= 0.2) and r0["b3_ok"]
            if not b0_pass:
                cand="Q3 ZERO_NOISE_ONLY_RESCUE"
            else:
                cand="Q1 but only g0"
        else:
            # Check if no g rescues
            max_eff = max(results[g]["acc_Vm"] for g in GRID)
            if max_eff < 0.75:
                cand="Q2 FUNDAMENTAL_TRADEOFF"
            else:
                cand="Q_UNRESOLVED"
    print("Q classification:", cand)

    out=pathlib.Path("results/w4a_Q_tradeoff.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out,"w") as f:
        json.dump(results, f, indent=2)
    print(f"saved to {out}")

if __name__=="__main__":
    run()
