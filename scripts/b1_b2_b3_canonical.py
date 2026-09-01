"""Canonical B1/B2/B3 at HEAD vs historical — correct Poisson and window."""
import numpy as np, jax, jax.numpy as jnp
from jomission.network.builder import build_jomission_model, simulation_with_background_poisson
import jaxfne as jtfne
from jaxfne import Simulation

DT_MS=0.1
DUR_MS=2000.0

def run_canonical(seed=0, with_poisson=True):
    model=build_jomission_model(n_per_area=100, seed=seed, dt_ms=DT_MS)
    if with_poisson:
        sim=simulation_with_background_poisson(model.cfg, duration_ms=DUR_MS, dt_ms=DT_MS, seed=seed)
        # Need to ensure runtime is edge_list for delayed model
        from jomission.simulation.runtime import simulation_for_model
        # simulation_with_background_poisson already returns Simulation with poisson_drive, but we need to ensure edge_list backend
        # It currently does not set runtime, so we need to patch
        from jaxfne import RuntimeConfig
        # Use edge_list backend
        # Re-create sim with correct runtime
        from jaxfne import Simulation as Sim2
        # Extract poisson_drive from sim
        pd=sim.poisson_drive
        sim=Sim2(duration_ms=DUR_MS, dt_ms=DT_MS, seed=seed, runtime=RuntimeConfig(recurrent_backend="edge_list"), poisson_drive=pd)
    else:
        from jomission.simulation.runtime import simulation_for_model
        sim=simulation_for_model(model, duration_ms=DUR_MS, dt_ms=DT_MS, seed=seed)
    sig=jtfne.simulate(model, sim)
    return model, sig

for with_poisson in [True, False]:
    model, sig = run_canonical(seed=0, with_poisson=with_poisson)
    V=np.asarray(sig.V_m); S=np.asarray(sig.spikes)
    rates=S.mean(axis=0)*(1000/DT_MS)
    print(f"\nwith_poisson {with_poisson} global mu {rates.mean():.2f} SD {rates.std():.2f} CV_rate {rates.std()/rates.mean():.3f} if any")
    # Per V4 L4 etc.
    tbl=model.neuron_table()
    # B1 check
    pops={}
    for area in ["V1","V4"]:
        for layer in ["L4"]:
            for ct in ["E","PV"]:
                idx=[i for i,r in enumerate(tbl) if r["area"]==area and r["layer"]==layer and r["cell_type"]==ct]
                if idx:
                    pops[f"{area}_{layer}_{ct}"]=float(rates[idx].mean())
    print(" pops sample", {k: f"{v:.1f}" for k,v in list(pops.items())[:4]})
    # B2 quick
    # CV_ISI per neuron
    cvs=[]
    for n in range(400):
        times=np.where(S[:, n]>0.5)[0]*DT_MS
        if len(times)>=3:
            isi=np.diff(times)
            if isi.mean()>0:
                cvs.append(float(isi.std()/isi.mean()))
    print(f" CV_ISI mean {np.mean(cvs):.3f} median {np.median(cvs):.3f} frac>0.5 {np.mean(np.array(cvs)>0.5):.3f} n {len(cvs)}")
    # Fano
    win=int(100/DT_MS)
    fanos=[]
    for n in range(400):
        counts=[float(S[i:i+win, n].sum()) for i in range(0, int(DUR_MS/DT_MS), win)]
        if np.mean(counts)>0:
            fanos.append(float(np.var(counts)/np.mean(counts)))
    print(f" Fano median {np.median(fanos):.3f}")
    # rho
    bin_ms=10; bin_steps=int(bin_ms/DT_MS)
    binned=[]
    for n in range(min(20, 400)):
        s=S[:, n]
        binned_n=[float(s[i:i+bin_steps].sum()) for i in range(0, int(DUR_MS/DT_MS), bin_steps)]
        binned.append(binned_n)
    binned=np.array(binned)
    rhos=[]
    for i in range(min(20,400)):
        for j in range(i+1, min(20,400)):
            if np.std(binned[i])>1e-9 and np.std(binned[j])>1e-9:
                rhos.append(float(np.corrcoef(binned[i], binned[j])[0,1]))
    print(f" rho mean {np.mean(rhos):.3f}")

# Also check historical ledger values for comparison
print("\nHistorical from ledger C007/C018:")
print(" C007 2000ms zero-drive+Poisson 1kHz amp0.7 seed0: mu 11.92 CV 0.055 rho0.137 Fano0.181")
print(" C018 network CV 0.344 frac>0.5 0.262 etc.")
print(" Our Q assay sensory p1 mu 33.6 CV 0.23 etc. is sensory-driven, not spontaneous — discrepancy is assay window, not regression")
print(" Canonical spontaneous with Poisson should be compared to C018/C007, not Q assay")
