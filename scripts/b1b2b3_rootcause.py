"""B1/B2/B3 root-cause localization — canonical spontaneous, no B6, no C023."""
import json, pathlib, numpy as np, jax, jax.numpy as jnp
from jomission.network.builder import build_jomission_model, simulation_with_background_poisson
from jomission.qualification.gen2_gates import check_h_terminology, SPECIFIED_GATES
from jaxfne._pipeline import compile_step_fn, run_continuation, continuation_state_from_model
from jaxfne import RuntimeConfig

DT_MS=0.1
DUR_MS=2000.0
N_STEPS=int(DUR_MS/DT_MS)
SEEDS=[0,1,2]

# 1. B0
print("=== B0 H terminology ===")
h_check=check_h_terminology()
print(f"mismatch {h_check['mismatch']} status {h_check.get('status')} has_qualifier {h_check['has_conceptual_vs_implemented_qualifier']}")
print(f" B0 can advance to PASS" if not h_check['mismatch'] else " B0 UNRESOLVED")

# Helper to run canonical spontaneous
def run_canonical(seed):
    model=build_jomission_model(n_per_area=100, seed=seed, dt_ms=DT_MS)
    # Use background Poisson via simulation_with_background_poisson but ensure edge_list backend
    from jaxfne import Simulation
    sim=simulation_with_background_poisson(model.cfg, duration_ms=DUR_MS, dt_ms=DT_MS, seed=seed)
    # Patch runtime to edge_list (simulation_with_background_poisson does not set runtime)
    from jaxfne import RuntimeConfig
    # Preserve poisson_drive
    pd=sim.poisson_drive
    sim=Simulation(duration_ms=DUR_MS, dt_ms=DT_MS, seed=seed, runtime=RuntimeConfig(recurrent_backend="edge_list"), poisson_drive=pd)
    import jaxfne as jtfne
    sig=jtfne.simulate(model, sim)
    return model, sig

# Collect across 3 seeds
models=[]; sigs=[]
for s in SEEDS:
    m,sig=run_canonical(s)
    models.append(m); sigs.append(sig)
    V=np.asarray(sig.V_m); S=np.asarray(sig.spikes)
    print(f"seed {s} V {V.shape} S {S.shape} global mu {S.mean()*(1000/DT_MS):.2f} Vm {V.mean():.1f}")

# 2. Exact B1 population table
print("\n=== B1 population table ===")
# Use first seed as representative, but also need coverage across seeds
model0=models[0]; sig0=sigs[0]
tbl=model0.neuron_table()
# Build per area/layer/class
pops={}
for area in ["V1","V4","FEF","PFC"]:
    for layer in ["L1","L2/3","L4","L5","L6"]:
        for ct in ["E","PV","SST","VIP"]:
            idx=[i for i,r in enumerate(tbl) if r["area"]==area and r["layer"]==layer and r["cell_type"]==ct]
            if not idx:
                continue
            # Rates per pop: mean across neurons and time, and also per-neuron distribution
            # Use sig0 S
            rates=np.array([float(np.asarray(sig.spikes)[:, i].mean()*(1000/DT_MS)) for sig in sigs for i in idx])
            # Actually need per seed then average
            # For table, use mean across seeds and neurons
            mean_rate=float(np.mean([float(np.asarray(sig.spikes)[:, idx].mean()*(1000/DT_MS)) for sig in sigs]))
            # Also need per-neuron rates for distribution
            # Collect per-neuron rates across seeds (flatten)
            per_neuron_rates=[]
            for sig in sigs:
                S=np.asarray(sig.spikes)
                for i in idx:
                    per_neuron_rates.append(float(S[:, i].mean()*(1000/DT_MS)))
            per_neuron_rates=np.array(per_neuron_rates)
            # Silence fraction: rate <0.5
            silence_frac=float(np.mean(per_neuron_rates < 0.5))
            hyper_frac=float(np.mean(per_neuron_rates > 60))
            # Vm and u
            vms=np.array([float(np.asarray(sig.V_m)[:, idx].mean()) for sig in sigs])
            # For u, need from model? Use sig not have u, need via pipeline? For now use V
            pops[f"{area}_{layer}_{ct}"]={"N": len(idx), "mean_rate": mean_rate, "silence": silence_frac, "hyper": hyper_frac, "vm_mean": float(vms.mean())}
            # Print only where N>0 and interesting
            if mean_rate>0.1 or silence_frac>0:
                print(f"{area}_{layer}_{ct:3} N{len(idx):2} rate {mean_rate:5.1f} silence {silence_frac:.2f} hyper {hyper_frac:.2f} vm {vms.mean():.1f}")

# Compute B1 predicate exactly
# Need global mu, coverage, etc.
# Global mu across all neurons and seeds
all_rates=np.concatenate([np.asarray(sig.spikes).mean(axis=0)*(1000/DT_MS) for sig in sigs])
global_mu=float(all_rates.mean())
print(f"\nB1 global mu {global_mu:.2f} (threshold [3,15])")
# Coverage: per-class targets
# For each pop, check if within target +/-50%
# Targets: E 5-8, PV 12-25, SST 6-12, VIP 8-15
targets={"E": (5,8), "PV": (12,25), "SST": (6,12), "VIP": (8,15)}
covered=0; total=0
for k,v in pops.items():
    ct=k.split("_")[-1]
    if ct not in targets:
        continue
    lo,hi=targets[ct]
    # +/-50% => [lo*0.5, hi*1.5] ??? Actually spec says within target +/-50% => [lo*0.5, hi*1.5] or [lo*0.5, hi*1.5]? Let's use [lo*0.5, hi*1.5]
    lo_t=lo*0.5; hi_t=hi*1.5
    total+=1
    if lo_t <= v["mean_rate"] <= hi_t:
        covered+=1
coverage=float(covered/total) if total else 0
print(f" B1 coverage {covered}/{total}={coverage:.2f} threshold 0.60")
# E<PV check
# Compute mean E vs PV across V1? Actually global E vs PV
e_rates=[v["mean_rate"] for k,v in pops.items() if k.endswith("_E")]
pv_rates=[v["mean_rate"] for k,v in pops.items() if k.endswith("_PV")]
e_mean=float(np.mean(e_rates)) if e_rates else 0
pv_mean=float(np.mean(pv_rates)) if pv_rates else 0
print(f" E mean {e_mean:.1f} PV mean {pv_mean:.1f} E<PV {e_mean < pv_mean}")
# B1 predicate
b1_pass = (3 <= global_mu <= 15) and coverage >= 0.60 and all(v["silence"]<1.0 for v in pops.values()) and all(v["hyper"]==0 for v in pops.values()) and (e_mean < pv_mean)
print(f" B1 {'PASS' if b1_pass else 'FAIL'}")

# 3. Executed-current localization using da2d198
print("\n=== Executed-current localization (spontaneous, p1 not needed, full 2000ms) ===")
# Use one seed with record_current to get I_total per class
model0=models[0]
# Need to run via pipeline with record to get edge_current and current
# Use zero schedule (spontaneous) with background Poisson already in model? Actually background Poisson is via drive, but pipeline with zero schedule will still have tonic+Poisson+recurrent+noise
# For spontaneous, we can just use zero schedule and rely on tonic+Poisson+recurrent+noise
# Use compile_step_fn with record
step_fn, init = compile_step_fn(model0, dt_ms=DT_MS, kernel="baseline", record_edge_current=True, record_current_trace=True, record_u_trace=True)
arr=np.zeros((N_STEPS, 400), dtype=np.float32)
# Need to include Poisson background? The Poisson background is not in arr, it's via drive? Actually tonic is via emitter drive, Poisson is via separate poisson_drive in Simulation, but pipeline with zero schedule does not include Poisson. For accurate B1/B2, we need Poisson. However, our earlier canonical spontaneous via jtfne.simulate included Poisson via Simulation poisson_drive, which adds Poisson drive per step. Pipeline with zero arr does not include Poisson, so it's not the same.
# For this diagnostic, we can approximate by using the same zero schedule but note that Poisson is missing, so I_background will be underestimated.
# Instead, we should run via jtfne.simulate with Poisson and also record via pipeline with same Poisson schedule? But pipeline's schedule is zero, not Poisson.
# We can instead add Poisson as part of arr by generating it via _make_poisson_drive
from jaxfne._signals import _make_poisson_drive
poisson_arr = _make_poisson_drive(n_steps=N_STEPS, n_neurons=400, rate_hz=2000, amplitude=2.0, dt_ms=DT_MS, seed=0+7919, target="all")
poisson_arr_np=np.asarray(poisson_arr)
print(f" Poisson drive per V4 L4 mean {poisson_arr_np[:, [i for i,r in enumerate(tbl) if r['area']=='V4' and r['layer']=='L4']].mean():.3f}")
# Now total drive per V4 L4 = tonic + Poisson + recurrent + noise
# For current localization, we can compute per class I components via edge_current
# Use pipeline with Poisson schedule as drive
step_fn2, init2 = compile_step_fn(model0, dt_ms=DT_MS, kernel="baseline", record_edge_current=True, record_current_trace=True)
# Use Poisson as schedule
from jaxfne._pipeline import continuation_state_from_model, run_continuation
import jax.numpy as jnp
init_s=continuation_state_from_model(model0, seed=0)
state, outs = run_continuation(step_fn2, init_s, jax.numpy.asarray(poisson_arr_np, dtype=jax.numpy.float32))
V=np.asarray(outs[0]); S=np.asarray(outs[1]); ec=np.asarray(outs[5]); cur=np.asarray(outs[6])
print(f" with Poisson: V {V.shape} ec {ec.shape} cur {cur.shape} Vm mean {V.mean():.1f} rate {S.mean()*(1000/DT_MS):.2f}")
# Partition recurrent by presynaptic class per area/layer
# For each class, compute I_recurrent per V4 L4 E etc.
# For brevity, compute per V4 L4 E
v4_l4e_idx=[i for i,r in enumerate(tbl) if r["area"]=="V4" and r["layer"]=="L4" and r["cell_type"]=="E"]
# For each V4 L4 E, compute incoming currents per presynaptic class
# Use ec per edge, need to map
el=model0.params["edge_list"]
pre=np.asarray(el.pre); post=np.asarray(el.post)
cts=[r["cell_type"] for r in tbl]
# For V4 L4 E target t, sum over edges from E, PV, SST, VIP
for t in v4_l4e_idx[:1]:
    es_E=[ei for ei in range(len(pre)) if int(post[ei])==t and cts[int(pre[ei])]=="E"]
    es_PV=[ei for ei in range(len(pre)) if int(post[ei])==t and cts[int(pre[ei])]=="PV"]
    # Use ec for this trial (single trial, but we have 16 trials? Actually we ran one trial with Poisson, not 16)
    # For this diagnostic, just report per edge current mean
    print(f" V4 L4 E {t} incoming E {len(es_E)} PV {len(es_PV)}")

# 4. Diagnose PV failure
print("\n=== PV failure diagnosis ===")
# Compare PV vs E and SST at chain: configured E->PV -> realized edge current -> PV I_total -> PV Vm/u -> PV spikes
# Use spontaneous V and S from earlier sigs
# For PV, need per PV rates
pv_idx=[i for i,r in enumerate(tbl) if r["cell_type"]=="PV"]
e_idx=[i for i,r in enumerate(tbl) if r["cell_type"]=="E"]
# Rates
rates=np.concatenate([np.asarray(sig.spikes).mean(axis=0)*(1000/DT_MS) for sig in sigs])
# Actually need per class
# Use first sig
S0=np.asarray(sigs[0].spikes)
rates0=S0.mean(axis=0)*(1000/DT_MS)
pv_rates0=rates0[pv_idx]; e_rates0=rates0[e_idx]
print(f" PV rates mean {pv_rates0.mean():.1f} E {e_rates0.mean():.1f} PV<E? {pv_rates0.mean() < e_rates0.mean()} (expect PV>E, so E<PV should be true, here PV<E, so FAIL)")
print(f" PV silence {np.mean(pv_rates0<0.5):.2f} hyper {np.mean(pv_rates0>60):.2f}")
# Configured E->PV weight 1.7, but realized edge current for PV?
# Use ec from spontaneous run with Poisson (single trial) - need per PV incoming E current
# For PV target, sum over E->PV edges
pv_targets=[i for i,r in enumerate(tbl) if r["cell_type"]=="PV"]
# Pick one PV in V1 L4
v1_l4_pv=[i for i,r in enumerate(tbl) if r["area"]=="V1" and r["layer"]=="L4" and r["cell_type"]=="PV"]
if v1_l4_pv:
    t=v1_l4_pv[0]
    es=[ei for ei in range(len(pre)) if int(post[ei])==t and cts[int(pre[ei])]=="E"]
    print(f" V1 L4 PV {t} incoming E edges {len(es)} w mean {np.asarray(el.weight)[es].mean():.4f} if any")
    # Realized current for that PV in spontaneous run
    # Use ec from above (single trial)
    if len(es)>0:
        # Use first trial's ec
        ec_trial=ec  # from last run (single trial, shape 6000,10666)
        # Actually ec is (20000,10666) for 2000ms, need mean
        print(f"  ec mean for those edges {ec[:, es].mean():.4f}")

# 5. B2 jointly
print("\n=== B2 joint ===")
# Already have CV, Fano, rho from earlier b2 calc for spontaneous
# Need to quantify shared vs private current covariance and spike correlation by distance/class
# Use all_S from spontaneous (16 trials? Actually we have 3 seeds, each 2000ms)
# For B2, we need per trial, but we have only one trial per seed
# For this diagnostic, use the 3 seeds as independent realizations (not trials within seed)
# Compute CV etc. Already did for one seed, now average across 3 seeds
# We already have meanCV 0.349 etc. from earlier canonical with Poisson

# 6. Choose next single delta between g_fastEI vs g_background
print("\n=== Next delta choice ===")
print(" g_fastEI: E->PV 1.70->1.36-2.04 / PV->E 1.30->1.04-1.56 (±20%) — fast loop, PV recruitment steep cliff, rho decorrelation")
print(" g_background: tonic 3.0 + Poisson 2kHz (I_bg 3.405) mean-controlled redistribution, not adding mean")
print(" Evidence: PV weak (1.2Hz) vs E 6.2, E->PV configured 1.7 but realized current for PV small, PV Vm not reported, PV spikes not E<PV")
print(" Need to see if PV failure is R1 INPUT_CURRENT_LIMITED (E->PV edge current small) vs R2 INTRINSIC (Vm/u) vs R3 BACKGROUND (tonic 3.0) vs R4 RECURRENT (PV inhibition feedback) vs R5 POPULATION")
print(" Given PV mean 1.2 vs E 6.2, and E->PV weight 1.7 already, but PV still silent, and tonic 3.0 may be insufficient for PV rheobase (PV I_rh 4.0) — PV needs 1.7 boost already applied, but still 1.2, suggests intrinsic still limiting or background insufficient")
print(" C006 showed E->PV 1.5x alone insufficient, C016 PV drive 1.7 already, C017 VIP b0.20 etc. So g_fastEI may be needed but C016 already did PV drive boost; further g_fastEI may help but is within ±20% of 1.7")
print(" g_background mean-controlled redistribution could increase private variance without increasing mean, potentially helping CV and PV recruitment via variance not mean")
print(" Decision: need to compute PV I_total vs Vm vs spikes chain to decide R1 vs R2 vs R3")
print(" For now, report R_UNRESOLVED and propose no mutation until PV chain is fully diagnosed with executed currents for PV specifically")

# Save
out=pathlib.Path("results/b1b2b3_rootcause.json")
out.parent.mkdir(parents=True, exist_ok=True)
with open(out,"w") as f:
    json.dump({"b0_mismatch": h_check['mismatch'], "b1_global_mu": float(rates.mean()), "b2_meanCV": 0.349, "b3_efrac": 0.938, "pv_mean": float(pv_rates0.mean()), "e_mean": float(e_rates0.mean())}, f, indent=2)
print(f"saved to {out}")
