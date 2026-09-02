"""Canonical spontaneous executed-current system identification — B1/B2/B3 root cause, no B6, no C023."""
import json, pathlib, numpy as np, jax, jax.numpy as jnp
from jomission.network.builder import build_jomission_model
from jaxfne._pipeline import compile_step_fn, run_continuation, continuation_state_from_model
from jaxfne._signals import _make_poisson_drive

DT_MS=0.1
DUR_MS=2000.0
N_STEPS=int(DUR_MS/DT_MS)
SEEDS=[0,1,2]

def run_canonical(seed, with_record=False):
    model=build_jomission_model(n_per_area=100, seed=seed, dt_ms=DT_MS)
    # Poisson background 2000Hz amp2.0 as per I_bg
    poisson = _make_poisson_drive(n_steps=N_STEPS, n_neurons=400, rate_hz=2000, amplitude=2.0, dt_ms=DT_MS, seed=seed+7919, target="all")
    poisson_np=np.asarray(poisson)
    # Use pipeline with record if needed
    if with_record:
        step_fn, init = compile_step_fn(model, dt_ms=DT_MS, kernel="baseline", record_edge_current=True, record_current_trace=True, record_u_trace=True)
        init_s=continuation_state_from_model(model, seed=seed)
        state, outs = run_continuation(step_fn, init_s, jax.numpy.asarray(poisson_np, dtype=jax.numpy.float32))
        V=np.asarray(outs[0]); S=np.asarray(outs[1]); ec=np.asarray(outs[5]); cur=np.asarray(outs[6]); U=np.asarray(outs[7])
        return model, V, S, ec, cur, U, poisson_np
    else:
        # Use jtfne.simulate for B1/B2 (without needing edge traces)
        import jaxfne as jtfne
        from jaxfne import Simulation, RuntimeConfig
        sim=Simulation(duration_ms=DUR_MS, dt_ms=DT_MS, seed=seed, runtime=RuntimeConfig(recurrent_backend="edge_list"), poisson_drive={"rate_hz":2000, "amplitude":2.0, "target":"all", "seed":seed+7919})
        sig=jtfne.simulate(model, sim)
        V=np.asarray(sig.V_m); S=np.asarray(sig.spikes)
        return model, V, S, None, None, None, poisson_np

# 1. Collect B1/B2 across 3 seeds without record (for rates)
print("=== B1/B2 canonical 3 seeds ===")
models=[]; all_S=[]
for s in SEEDS:
    m,V,S,_,_,_,_ = run_canonical(s, with_record=False)
    models.append(m); all_S.append(S)
    print(f"seed {s} global mu {S.mean()*(1000/DT_MS):.2f}")

# Use first model for structure
model0=models[0]
tbl=model0.neuron_table()
# B1 table already done in previous script, but redo quickly for canonical
# For executed-current localization, use single seed with record
print("\n=== Executed-current localization single seed 0 with Poisson ===")
model, V, S, ec, cur, U, poisson = run_canonical(0, with_record=True)
print(f"V {V.shape} ec {ec.shape} cur {cur.shape} U {U.shape}")
# Partition per class
areas=[r["area"] for r in tbl]; layers=[r["layer"] for r in tbl]; cts=[r["cell_type"] for r in tbl]
el=model.params["edge_list"]
pre=np.asarray(el.pre); post=np.asarray(el.post); w=np.asarray(el.weight)
# Helper to get per neuron I components per trial (single trial, so per time)
# For B1/B2 we need per neuron per trial, but we have single trial, so we can compute per neuron mean currents over time
# For each V4 L4 E etc, compute I components
# I_tonic per neuron
em=model.params["emitter"]
drive_tonic=np.asarray(em.drive)
# For each class, compute per neuron I_FF etc. But for canonical spontaneous, I_FF is inter-area, not relevant for B1/B2 which is global
# For PV recruitment chain, focus on PV populations, especially failing FEF/PFC
# Let's do per area/layer PV vs E

def per_pop_stats(area, layer, ct):
    idx=[i for i,r in enumerate(tbl) if r["area"]==area and r["layer"]==layer and r["cell_type"]==ct]
    if not idx:
        return None
    # Rates
    rates=np.array([float(all_S[0][:, i].mean()*(1000/DT_MS)) for i in idx])  # use first seed S
    # But need per seed average? Use all_S from earlier
    # For this diagnostic, use the single recorded trial's S
    # Compute per neuron rate in this trial
    # Use S from recorded run (single trial)
    # For executed currents, use ec and cur for this trial
    # For each neuron in idx, compute I components
    # I_recurrent per neuron = sum over incoming edges per class
    # Build per target incoming
    # For each t in idx, sum ec over its incoming edges per presynaptic class
    # Use ec (20000,10666) and cur (20000,400)
    # Compute per neuron mean over time
    # For each t, I_total = cur mean, I_tonic = drive_tonic[t], I_poisson = poisson mean, I_recurrent = I_total - I_tonic - I_poisson - noise? But noise is part of cur - (drive+poisson+syn) -> we can compute syn as sum ec
    # For simplicity, compute I_recurrent_E etc as sum ec per class
    # Use ec and cur for this trial
    # For each t, need per time, but for mean we can average
    vals={}
    for t in idx:
        es_E=[ei for ei in range(len(pre)) if int(post[ei])==t and cts[int(pre[ei])]=="E"]
        es_PV=[ei for ei in range(len(pre)) if int(post[ei])==t and cts[int(pre[ei])]=="PV"]
        es_SST=[ei for ei in range(len(pre)) if int(post[ei])==t and cts[int(pre[ei])]=="SST"]
        es_VIP=[ei for ei in range(len(pre)) if int(post[ei])==t and cts[int(pre[ei])]=="VIP"]
        # Use ec for this trial (single)
        # ec is (20000,10666)
        # Compute mean over time per edge sum
        Ie = float(ec[:, es_E].sum(axis=1).mean()) if es_E else 0
        Ipv = float(ec[:, es_PV].sum(axis=1).mean()) if es_PV else 0
        Isst = float(ec[:, es_SST].sum(axis=1).mean()) if es_SST else 0
        Ivip = float(ec[:, es_VIP].sum(axis=1).mean()) if es_VIP else 0
        Itot = float(cur[:, t].mean())
        Ipoiss = float(poisson[:, t].mean())
        Itonic = float(drive_tonic[t])
        # Vm and u
        Vm = float(V[:, t].mean())
        # U is per neuron per time
        Uv = float(U[:, t].mean())
        # Spikes
        spk = float(S[:, t].mean()*(1000/DT_MS))
        vals[t]={"Ie":Ie, "Ipv":Ipv, "Isst":Isst, "Ivip":Ivip, "Itot":Itot, "Ipoiss":Ipoiss, "Itonic":Itonic, "Vm":Vm, "U":Uv, "spk":spk, "rate":spk}
    # Aggregate per pop
    # Compute mean across neurons in pop
    mean_Ie=float(np.mean([v["Ie"] for v in vals.values()])) if vals else 0
    mean_Ipv=float(np.mean([v["Ipv"] for v in vals.values()])) if vals else 0
    mean_Itot=float(np.mean([v["Itot"] for v in vals.values()])) if vals else 0
    mean_Vm=float(np.mean([v["Vm"] for v in vals.values()])) if vals else 0
    mean_U=float(np.mean([v["U"] for v in vals.values()])) if vals else 0
    mean_spk=float(np.mean([v["spk"] for v in vals.values()])) if vals else 0
    return {"N": len(idx), "mean_Ie": mean_Ie, "mean_Ipv": mean_Ipv, "mean_Itot": mean_Itot, "mean_Vm": mean_Vm, "mean_U": mean_U, "mean_spk": mean_spk, "vals": vals}

# A. PV recruitment chain: for every PV population, especially failing FEF/PFC
print("\n=== A. PV recruitment chain ===")
for area in ["V1","V4","FEF","PFC"]:
    for layer in ["L2/3","L4","L5","L6"]:
        key=f"{area}_{layer}_PV"
        # Use per_pop_stats for PV
        # Find PV idx
        idx=[i for i,r in enumerate(tbl) if r["area"]==area and r["layer"]==layer and r["cell_type"]=="PV"]
        if not idx:
            continue
        stats=per_pop_stats(area, layer, "PV")
        # Also get E counterpart for comparison
        e_stats=per_pop_stats(area, layer, "E")
        if stats and e_stats:
            print(f"{key} N{stats['N']} E N{e_stats['N']} PV Ie {stats['mean_Ie']:.3f} (E->PV) vs E Ie {e_stats['mean_Ie']:.3f} | PV Itot {stats['mean_Itot']:.2f} Vm {stats['mean_Vm']:.1f} U {stats['mean_U']:.1f} spk {stats['mean_spk']:.1f} vs E spk {e_stats['mean_spk']:.1f}")

# B. Inhibition delivered to E
print("\n=== B. Inhibition delivered to E (B3) ===")
for area in ["V1","V4"]:
    for layer in ["L2/3","L4"]:
        # For E target in this area/layer
        e_idx=[i for i,r in enumerate(tbl) if r["area"]==area and r["layer"]==layer and r["cell_type"]=="E"]
        if not e_idx:
            continue
        # Compute per E target, incoming I per presynaptic class
        # Use first E as example, but aggregate
        # For each E target, sum per class
        per_e_vals=[]
        for t in e_idx:
            es_E=[ei for ei in range(len(pre)) if int(post[ei])==t and cts[int(pre[ei])]=="E"]
            es_PV=[ei for ei in range(len(pre)) if int(post[ei])==t and cts[int(pre[ei])]=="PV"]
            es_SST=[ei for ei in range(len(pre)) if int(post[ei])==t and cts[int(pre[ei])]=="SST"]
            es_VIP=[ei for ei in range(len(pre)) if int(post[ei])==t and cts[int(pre[ei])]=="VIP"]
            # Use ec for single trial
            Ie = float(ec[:, es_E].sum(axis=1).mean()) if es_E else 0
            Ipv = float(ec[:, es_PV].sum(axis=1).mean()) if es_PV else 0
            Isst = float(ec[:, es_SST].sum(axis=1).mean()) if es_SST else 0
            Ivip = float(ec[:, es_VIP].sum(axis=1).mean()) if es_VIP else 0
            per_e_vals.append((Ie, Ipv, Isst, Ivip))
        per_e_vals=np.array(per_e_vals)
        Ie_m=float(per_e_vals[:,0].mean()) if len(per_e_vals) else 0
        Ipv_m=float(per_e_vals[:,1].mean()) if len(per_e_vals) else 0
        Isst_m=float(per_e_vals[:,2].mean()) if len(per_e_vals) else 0
        Ivip_m=float(per_e_vals[:,3].mean()) if len(per_e_vals) else 0
        denom=abs(Ie_m)+abs(Ipv_m)+abs(Isst_m)+abs(Ivip_m)
        efrac=abs(Ie_m)/denom if denom>1e-9 else 0.5
        print(f"{area}_{layer}_E N{len(e_idx)} Efrac {efrac:.3f} Ie {Ie_m:.3f} Ipv {Ipv_m:.3f} Isst {Isst_m:.3f} Ivip {Ivip_m:.3f} {'PASS' if 0.15<=efrac<=0.60 else 'FAIL'}")

# C. VIP coverage failure
print("\n=== C. VIP ===")
for area in ["V1","V4","FEF","PFC"]:
    for layer in ["L2/3","L4","L5","L6"]:
        idx=[i for i,r in enumerate(tbl) if r["area"]==area and r["layer"]==layer and r["cell_type"]=="VIP"]
        if not idx:
            continue
        # Check input currents for VIP
        # Use per VIP stats
        stats=per_pop_stats(area, layer, "VIP")
        if stats:
            print(f"{area}_{layer}_VIP N{stats['N']} spk {stats['mean_spk']:.2f} Vm {stats['mean_Vm']:.1f} Itot {stats['mean_Itot']:.2f} Ie {stats['mean_Ie']:.3f}")

# D. B2 covariance: shared vs private
print("\n=== D. B2 covariance ===")
# Use all_S from canonical 3 seeds? For this diagnostic, use single seed's S (20000,400) and compute binned correlations
# Use binned as before for B2
# Compute shared vs private current covariance for representative E and PV neurons
# Pick 2 E neurons in V1 L4 and 2 PV in same
# Compute I_total per neuron per time (cur[:, t]) and decompose into recurrent vs private (Poisson+noise)
# For each pair, compute Cov(I_total_i, I_total_j) partitioned
# Use cur and ec
# For E neurons in V1 L4
v1_l4e_idx=[i for i,r in enumerate(tbl) if r["area"]=="V1" and r["layer"]=="L4" and r["cell_type"]=="E"]
# Pick first 2
if len(v1_l4e_idx)>=2:
    i0=v1_l4e_idx[0]; i1=v1_l4e_idx[1]
    # I_total per time per neuron
    I0=cur[:, i0]; I1=cur[:, i1]
    # Private vs recurrent: private = Poisson + noise, recurrent = syn
    # But we don't have separate private and recurrent per time per neuron as distinct traces, but we can approximate:
    # I_private = Poisson[:, i] (from poisson) + noise (cur - (drive+schedule+syn))
    # For spontaneous, schedule=0, drive is tonic, syn is sum ec per neuron
    # Compute syn per neuron per time
    syn0=np.array([float(ec[t, [ei for ei in range(len(pre)) if int(post[ei])==i0]].sum()) if any(int(post[ei])==i0 for ei in range(len(pre))) else 0 for t in range(20000)])
    # Actually need per time, but ec is (20000,10666), sum over edges to i0 per time
    # Let's compute correctly
    # Build per V4 L4? For this, use V1 L4 E
    # For each time, syn for i0 is sum over its incoming edges
    # Use ec per time
    # For i0, incoming edges
    inc0=[ei for ei in range(len(pre)) if int(post[ei])==i0]
    inc1=[ei for ei in range(len(pre)) if int(post[ei])==i1]
    syn0_t = ec[:, inc0].sum(axis=1) if inc0 else np.zeros(20000)
    syn1_t = ec[:, inc1].sum(axis=1) if inc1 else np.zeros(20000)
    # Private = Poisson + noise + tonic? Actually tonic is constant, not variable, so covariance comes from recurrent + Poisson+noise
    # Poisson per neuron per time
    pois0=poisson[:, i0]; pois1=poisson[:, i1]
    # Noise is cur - (drive+schedule+syn) where schedule=0 for spontaneous (since Poisson is separate? Actually poisson is schedule in this run? In our pipeline run, we passed Poisson as schedule (poisson_arr), so cur includes Poisson as part of schedule? Let's check: in pipeline, schedule is poisson_arr, drive is tonic, syn is recurrent, noise is 0.5*noise
    # So I_total = drive + Poisson + syn + noise
    # We have cur, drive_tonic, poisson, syn, so noise = cur - (drive+poisson+syn)
    # Compute
    drive0=float(drive_tonic[i0]); drive1=float(drive_tonic[i1])
    # For each time, compute
    I0_total=cur[:, i0]; I1_total=cur[:, i1]
    # Compute covariance
    cov_total=float(np.cov(I0_total, I1_total)[0,1])
    cov_syn=float(np.cov(syn0_t, syn1_t)[0,1]) if len(syn0_t)>1 else 0
    cov_pois=float(np.cov(pois0, pois1)[0,1])
    print(f" V1 L4 E {i0} vs {i1}: Cov total {cov_total:.4f} Cov syn {cov_syn:.4f} Cov pois {cov_pois:.4f} (pois independent, cov ~0)")
    print(f" Var total {np.var(I0_total):.4f} {np.var(I1_total):.4f} Var syn {np.var(syn0_t):.4f}")

print("\nDone")
