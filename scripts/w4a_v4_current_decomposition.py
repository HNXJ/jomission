"""Decompose I_FF -> I_net -> Vm for V1->V4 F2 — read-only, no C023.

Checks observability of required current components and tests F2a-d.
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
DELAY_FF_STEPS=80  # 8ms
N_TRIALS_PER_COND=8

def lovo_acc(X,y):
    n=len(y)
    if X.shape[0]!=n or X.shape[0] < 4:
        return 0.5
    correct=0
    for i in range(n):
        tr=[j for j in range(n) if j!=i]
        Xtr=X[tr]; ytr=y[tr]
        # handle empty class
        if len(np.unique(ytr))<2:
            continue
        c0=Xtr[ytr==0].mean(axis=0); c1=Xtr[ytr==1].mean(axis=0)
        # if centroids are nan (e.g., no variance), skip
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
    areas=[r["area"] for r in tbl]; layers=[r["layer"] for r in tbl]; cts=[r["cell_type"] for r in tbl]
    el=model.params["edge_list"]
    pre=np.asarray(el.pre); post=np.asarray(el.post); w=np.asarray(el.weight); delay=np.asarray(el.delay_steps)
    # Indices
    v1_l23e_idx=[i for i,r in enumerate(tbl) if r["area"]=="V1" and r["layer"]=="L2/3" and r["cell_type"]=="E"]
    v4_l4_idx=[i for i,r in enumerate(tbl) if r["area"]=="V4" and r["layer"]=="L4"]
    v4_l4e_idx=[i for i in v4_l4_idx if cts[i]=="E"]
    v4_l4pv_idx=[i for i in v4_l4_idx if cts[i]=="PV"]
    # FF mask V1 L2/3_E -> V4 L4 (all target classes)
    ff_mask=[]
    for ei in range(len(pre)):
        pi=int(pre[ei]); qi=int(post[ei])
        if areas[pi]=="V1" and layers[pi]=="L2/3" and cts[pi]=="E" and areas[qi]=="V4" and layers[qi]=="L4":
            ff_mask.append(ei)
    # Local V4 recurrent: V4 -> V4
    local_exc_mask=[]; local_pv_mask=[]; local_sst_mask=[]; local_vip_mask=[]
    for ei in range(len(pre)):
        pi=int(pre[ei]); qi=int(post[ei])
        if areas[pi]=="V4" and areas[qi]=="V4":
            ct=cts[pi]
            if ct=="E":
                local_exc_mask.append(ei)
            elif ct=="PV":
                local_pv_mask.append(ei)
            elif ct=="SST":
                local_sst_mask.append(ei)
            elif ct=="VIP":
                local_vip_mask.append(ei)
    print(f"FF {len(ff_mask)} delay {sorted(set(delay[ff_mask].tolist())) if ff_mask else []} w {w[ff_mask].mean():.4f} if any")
    print(f"Local V4 exc {len(local_exc_mask)} PV {len(local_pv_mask)} SST {len(local_sst_mask)} VIP {len(local_vip_mask)}")
    print(f"Observability: edge_current for FF/local available via record_edge_current; I_net/tonic/private/noise/u not directly exposed as traces in 0.4.18+960bf8f (only Vm, spikes, edge_current, presyn) — will report blocker for full I_net accounting")

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
    # Drive per V4 L4 neuron: external drive + tonic (emitter drive) — we can get emitter drive
    em=model.params["emitter"]
    drive_tonic=np.asarray(em.drive)  # (400,) includes tonic per neuron
    print(f"tonic drive V4 L4 mean {drive_tonic[v4_l4_idx].mean():.3f} std {drive_tonic[v4_l4_idx].std():.3f}")

    trials=[]
    for rep in range(N_TRIALS_PER_COND):
        for label, arr in [("A", arrA), ("B", arrB)]:
            seed=100+rep*10 + (0 if label=="A" else 1)
            trials.append((label, arr, seed))
    # Collect stage data
    # For each trial, we need time-resolved traces: V, spikes, edge_current
    # We'll store per trial per time per neuron for Vm/spikes, and per edge for current
    all_V=[]; all_S=[]; all_ec=[]; labels=[]
    for label, arr, seed in trials:
        step_fn, init = compile_step_fn(model, dt_ms=DT_MS, kernel="baseline", record_edge_current=True)
        init_s = continuation_state_from_model(model, seed=seed)
        state, outs = run_continuation(step_fn, init_s, jnp.asarray(arr, dtype=jnp.float32))
        V=np.asarray(outs[0]); S=np.asarray(outs[1]); ec=np.asarray(outs[5])
        all_V.append(V); all_S.append(S); all_ec.append(ec); labels.append(0 if label=="A" else 1)
    all_V=np.array(all_V)  # (16,6000,400)
    all_S=np.array(all_S)
    all_ec=np.array(all_ec)  # (16,6000,10666)
    labels=np.array(labels)
    print(f"collected {len(labels)} trials V {all_V.shape} ec {all_ec.shape}")

    # Helper to get per-trial vector for a stage and window
    def window_vec(X, window):
        # X is (n_trials, n_steps, n_units) for Vm/spikes, or (n_trials, n_steps, n_edges) for ec
        # window is (start,end) in steps
        s,e=window
        return X[:, s:e, :].mean(axis=1)  # (n_trials, n_units) or (n_trials, n_edges)

    # Define windows: whole p1, and sliding 50ms windows aligned to FF delay (80 steps =8ms)
    # p1 0-531ms, FF arrives at t+8ms, so V4 should see effect 8-539ms
    # We'll test whole p1 (0-531) and FF-shifted p1+delay (80-611) and 50ms sliding
    win_p1=(P1_S, P1_E)
    win_p1_delayed=(P1_S+DELAY_FF_STEPS, P1_E+DELAY_FF_STEPS)
    # For 6000 steps, P1_E+80 = 5390 <6000 ok
    # Sliding windows 50ms =500 steps, step 25ms =250 steps
    slide_windows=[(i, i+500) for i in range(P1_S, P1_E-500, 250)]

    # Stage helpers: per V4 L4 target class
    def target_cur_vec(ec_trials, mask, window):
        # ec_trials (n_trials, n_steps, n_edges) -> per trial per target neuron sum
        # mask is list of edge indices for FF
        # For each trial, per target neuron, sum over its incoming edges
        # Build per target
        # First, get per target edge list
        per_target={t: [] for t in v4_l4_idx}
        for ei in mask:
            per_target[int(post[ei])].append(ei)
        # For each trial, compute per target mean current in window
        vecs=[]
        for tr in range(ec_trials.shape[0]):
            row=[]
            for t in v4_l4_idx:
                es=per_target[t]
                if es:
                    row.append(float(ec_trials[tr, window[0]:window[1], es].sum(axis=1).mean()))
                else:
                    row.append(0.0)
            vecs.append(row)
        return np.array(vecs)

    # Test F2a-d
    # Compute w and delay distributions already
    # For each window, compute decoder for each stage
    stages=["I_FF_all","I_FF_E","V4_Vm","V4_spike"]
    # We'll compute for whole p1 and delayed and sliding
    results={}
    for wname, window in [("p1",win_p1), ("p1_delayed",win_p1_delayed)]:
        print(f"\nWindow {wname} {window}")
        # I_FF vectors
        # Need to recompute per window
        # Use target_cur_vec for each window
        cur_all = target_cur_vec(all_ec, ff_mask, window)
        # Also per class
        # For simplicity, use all
        # V4 Vm and spike
        vm_v4 = all_V[:, window[0]:window[1], :][:, :, v4_l4_idx].mean(axis=1)
        spk_v4 = all_S[:, window[0]:window[1], :][:, :, v4_l4_idx].mean(axis=1)*(1000/DT_MS)
        # Also V1 L2/3 spike for reference (should be 1.00)
        v1_spk = all_S[:, window[0]:window[1], :][:, :, [i for i,r in enumerate(tbl) if r["area"]=="V1" and r["layer"]=="L2/3" and r["cell_type"]=="E"]].mean(axis=1)*(1000/DT_MS) if len([i for i,r in enumerate(tbl) if r["area"]=="V1" and r["layer"]=="L2/3" and r["cell_type"]=="E"])>0 else np.zeros((16,1))
        for name, X in [("I_FF_all",cur_all), ("V4_Vm",vm_v4), ("V4_spike",spk_v4)]:
            if X.shape[1]==0:
                continue
            acc,p,null_m,null_s = perm_p(X, labels, n_perm=100)
            c0=X[labels==0].mean(axis=0); c1=X[labels==1].mean(axis=0)
            dist=float(np.linalg.norm(c0-c1))
            print(f"  {name:10} dim {X.shape[1]:2} acc {acc:.3f} p {p:.3f} dist {dist:.3f}")
            results[f"{wname}_{name}"]=acc
        # Also sliding for Vm
        # For temporal readout mismatch, check sliding windows for Vm
        if wname=="p1":
            print("  Sliding Vm windows (50ms):")
            for sw in slide_windows:
                vm_sw = all_V[:, sw[0]:sw[1], :][:, :, v4_l4_idx].mean(axis=1)
                acc_sw,p_sw,_,_=perm_p(vm_sw, labels, n_perm=50)
                print(f"    {sw} acc {acc_sw:.3f} p {p_sw:.3f}")

    # F2a: signal to background — compare I_FF variance vs I_net variance
    # We don't have I_net fully, but we can approximate I_net as I_FF + I_local + tonic
    # Use edge_current for local as well (same ec)
    local_mask = local_exc_mask + local_pv_mask + local_sst_mask + local_vip_mask
    # Compute per trial per V4 L4 neuron I_net approx: I_FF + I_local (sum over local edges) + tonic
    # For each trial, per V4 L4 neuron, I_local = sum over local edges to that neuron
    per_target_local={t: [] for t in v4_l4_idx}
    for ei in local_mask:
        q=int(post[ei])
        if q in per_target_local:
            per_target_local[q].append(ei)
    # Compute per trial I_net approx for window p1_delayed
    window=win_p1_delayed
    i_ff_per_target=[]
    i_local_per_target=[]
    for tr in range(all_ec.shape[0]):
        row_ff=[]; row_loc=[]
        for t in v4_l4_idx:
            es_ff=[ei for ei in ff_mask if int(post[ei])==t]
            es_loc=per_target_local[t]
            row_ff.append(float(all_ec[tr, window[0]:window[1], es_ff].sum(axis=1).mean()) if es_ff else 0.0)
            row_loc.append(float(all_ec[tr, window[0]:window[1], es_loc].sum(axis=1).mean()) if es_loc else 0.0)
        i_ff_per_target.append(row_ff); i_local_per_target.append(row_loc)
    i_ff_per_target=np.array(i_ff_per_target); i_local_per_target=np.array(i_local_per_target)
    # Add tonic per neuron (same across time)
    tonic_v4 = drive_tonic[v4_l4_idx][None,:]  # (1,15)
    # I_net approx = I_FF + I_local + tonic (ignore schedule, noise, private)
    i_net_approx = i_ff_per_target + i_local_per_target + tonic_v4
    # Compute variance across trials: signal variance (between A/B means) vs background variance (within)
    # For each neuron, compute A mean, B mean, and within variance
    # Signal power: (mean_A - mean_B)^2, background: pooled within variance
    def sig_bg(X, y):
        # X (n_trials, n_units)
        xa=X[y==0]; xb=X[y==1]
        mA=xa.mean(axis=0); mB=xb.mean(axis=0)
        sig=np.mean((mA-mB)**2)
        bg=np.mean(np.vstack([xa, xb]).var(axis=0))
        return float(sig), float(bg), float(sig/bg) if bg>0 else 0
    s_ff, b_ff, r_ff = sig_bg(i_ff_per_target, labels)
    s_loc, b_loc, r_loc = sig_bg(i_local_per_target, labels)
    s_net, b_net, r_net = sig_bg(i_net_approx, labels)
    print(f"F2a signal/background: I_FF sig {s_ff:.4f} bg {b_ff:.4f} ratio {r_ff:.4f}")
    print(f" I_local sig {s_loc:.4f} bg {b_loc:.4f} ratio {r_loc:.4f}")
    print(f" I_net approx sig {s_net:.4f} bg {b_net:.4f} ratio {r_net:.4f} (tonic dominates)")
    # F2b: recurrent cancellation — does I_local oppose I_FF?
    # Check correlation between I_FF and I_local per trial per neuron (should be negative if cancellation)
    # For each V4 L4 neuron, correlation across trials between I_FF and I_local
    corrs=[]
    for unit in range(len(v4_l4_idx)):
        ff_u=i_ff_per_target[:, unit]; loc_u=i_local_per_target[:, unit]
        if np.std(ff_u)>1e-9 and np.std(loc_u)>1e-9:
            corrs.append(float(np.corrcoef(ff_u, loc_u)[0,1]))
    print(f"F2b I_FF vs I_local per-unit correlation mean {np.mean(corrs):.3f} (negative suggests cancellation) n {len(corrs)}")
    # F2c: membrane gain — I_net vs Vm
    # Need Vm per V4 L4 for same window
    vm_v4_delayed = all_V[:, window[0]:window[1], :][:, :, v4_l4_idx].mean(axis=1)
    # Compute per unit delta Vm / delta I_net
    vm_A=vm_v4_delayed[labels==0].mean(axis=0); vm_B=vm_v4_delayed[labels==1].mean(axis=0)
    dVm=vm_A-vm_B
    dInet=i_net_approx[labels==0].mean(axis=0) - i_net_approx[labels==1].mean(axis=0)
    gain = np.divide(dVm, dInet, out=np.zeros_like(dVm), where=np.abs(dInet)>1e-9)
    print(f"F2c gain dVm/dI_net per V4 L4: mean {gain.mean():.3f} median {np.median(gain):.3f} (low gain suggests membrane limited)")
    print(f" Vm mean A {vm_A.mean():.1f} B {vm_B.mean():.1f} std {vm_v4_delayed.std():.3f} distance to threshold (30) {(30-vm_A.mean()):.1f}")

    # F2d already via sliding windows above

    # Classification
    # Use whole p1 results for F2a-d
    # If signal/background ratio for I_FF <0.1 -> F2a
    # If correlation negative and strong -> F2b
    # If gain low (<0.5) -> F2c
    # If sliding shows transient but whole p1 not -> F2d
    # For now, based on observed: I_FF acc 1.00 but Vm 0.438, and I_net approx still has small signal, gain low
    # Check sliding Vm: we saw sliding Vm accs printed; if any window >0.75 then F2d
    # Let's check max sliding acc
    max_slide_acc=0
    for sw in slide_windows:
        vm_sw = all_V[:, sw[0]:sw[1], :][:, :, v4_l4_idx].mean(axis=1)
        acc_sw,_ ,_,_=perm_p(vm_sw, labels, n_perm=50)
        max_slide_acc=max(max_slide_acc, acc_sw)
    print(f"max sliding Vm acc {max_slide_acc:.3f} (if >0.75 suggests temporal readout mismatch)")

    # Final classification
    # We have F2 with I 1.00, V 0.438, so failure is at I->V
    # Among F2a-d, which is supported?
    # F2a: signal/background for I_FF is? We have r_ff = sig/bg ; if ratio <1, then background dominates
    # Our r_ff computed as sig/bg, need to interpret
    # r_ff small means signal small relative to background
    # Let's decide thresholds
    if r_ff < 0.5:
        cand="F2a SIGNAL_BACKGROUND"
    elif np.mean(corrs) < -0.3:
        cand="F2b RECURRENT_CANCELLATION"
    elif abs(gain.mean()) < 0.5:
        cand="F2c MEMBRANE_GAIN"
    elif max_slide_acc >= 0.75:
        cand="F2d TEMPORAL_READOUT"
    else:
        cand="F2_UNRESOLVED"
    print("candidate", cand)
    out=pathlib.Path("results/w4a_v4_current_decomposition.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out,"w") as f:
        json.dump({"stages": {k: float(v) for k,v in [("r_ff",r_ff),("r_net",r_net),("gain",float(gain.mean())),("max_slide",float(max_slide_acc))]}, "first": cand}, f, indent=2)
    print(f"saved to {out}")
    # Also note observability blocker for full I_net (noise, private, u not exposed)
    print("Observability: JaxFNE 0.4.18+960bf8f exposes edge_current and Vm/spikes, but not tonic/private/noise per step nor u/adaptation per step as traces — full I_net accounting requires proxy reconstruction, reported as blocker for complete current accounting.")

if __name__=="__main__":
    run()
