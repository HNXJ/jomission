"""W4a L4 representation reset — RF/input -> V1 L4 Vm/spikes diagnostic.

Steps:
1. Audit configured->realized input per V1 L4_E (RF overlap, drive, normalization, contrast, ENERGY parity)
2. Map input->Vm->spikes per neuron
3. Population decoder (9-dim) with repeats, CV, permutation
4. Population-size/grammar check (why n_L4_E=9)
5. M2 audit
"""
import json, pathlib, numpy as np, jax, jax.numpy as jnp
from jomission.network.builder import build_jomission_model
from jomission.network.rf import RFConfig, RFOperator
from jomission.paradigm.spec import JOMISSION_PARADIGM
from jomission.network.populations import LAYER_COUNT_FRAC_DEFAULT, V1_LAYER_CELL_TYPES
from jaxfne.emitters import simulate_edge_recurrent_izhikevich
from dataclasses import replace

DT_MS=0.1
DUR_MS=600.0
N_STEPS=int(DUR_MS/DT_MS)
P1_S=int(0/DT_MS); P1_E=int(531/DT_MS)
SEEDS=[0,1,2,3,4]

def build():
    return build_jomission_model(n_per_area=100, seed=0, dt_ms=DT_MS)

def run():
    model=build()
    tbl=model.neuron_table()
    l4e_idx=[i for i,r in enumerate(tbl) if r["area"]=="V1" and r["layer"]=="L4" and r["cell_type"]=="E"]
    print(f"configured fractions L4 {LAYER_COUNT_FRAC_DEFAULT['L4']} E {V1_LAYER_CELL_TYPES['L4']['E']} -> expected 100*0.15*0.60=9, realized {len(l4e_idx)}")
    # RF
    rf_cfg=RFConfig(seed=0, tier="graded")
    op=RFOperator(rf_cfg, model)
    # centers for L4_E
    centers={i: op.centers[i] for i in l4e_idx}
    for i in l4e_idx[:3]:
        print(f"L4_E {i} center {centers[i]}")
    # Stimulus patterns
    patA=op.stimulus_pattern("stimulus_A"); patB=op.stimulus_pattern("stimulus_B")
    driveA=op.drive_for_stimulus("stimulus_A"); driveB=op.drive_for_stimulus("stimulus_B")
    # Per L4_E drive
    print("per L4_E drive A vs B:")
    for i in l4e_idx:
        print(f"  {i}: {float(driveA[i]):.4f} vs {float(driveB[i]):.4f} delta {float(driveA[i]-driveB[i]):.4f} RF center {centers[i]}")
    # Global ENERGY parity (from factorial)
    from jomission.simulation.factorial_v0p2 import energy_amplitude as _ea
    ampA=float(_ea("C","AAAB")); ampB=float(_ea("C","BBBA"))
    print(f"energy amplitude AAAB {ampA:.1f} BBBA {ampB:.1f} ratio {ampA/ampB:.4f}")
    # Realized input contrast per neuron: drive difference normalized
    deltas_drive=np.array([float(driveA[i]-driveB[i]) for i in l4e_idx])
    print(f"drive deltas L4_E mean {deltas_drive.mean():.4f} std {deltas_drive.std():.4f} max {np.abs(deltas_drive).max():.4f}")
    # Check if global energy matching makes small population poorly discriminative:
    # Global energy is sum of drive across V1 target weights? Compute sum of driveA vs driveB across all V1
    v1_idx=[i for i,r in enumerate(tbl) if r["area"]=="V1"]
    sumA=float(driveA[v1_idx].sum()); sumB=float(driveB[v1_idx].sum())
    print(f"global V1 drive sum A {sumA:.4f} B {sumB:.4f} ratio {sumA/sumB:.4f} delta {sumA-sumB:.4f}")
    # Now per-neuron input -> Vm -> spikes mapping
    # Run direct emitter with RF schedules (graded, energy-scaled)
    condA=[c for c in JOMISSION_PARADIGM.conditions if c.name=="AAAB"][0]
    condB=[c for c in JOMISSION_PARADIGM.conditions if c.name=="BBBA"][0]
    schedA=op.to_stimulus_schedule(condA, n_neurons=400, dt_ms=DT_MS, base_amplitude=ampA)
    schedB=op.to_stimulus_schedule(condB, n_neurons=400, dt_ms=DT_MS, base_amplitude=ampB)
    arrA=schedA.to_array(N_STEPS, DT_MS); arrB=schedB.to_array(N_STEPS, DT_MS)
    em=model.params["emitter"]; el=model.params["edge_list"]
    # Use 5 seeds for population decoder repeats (runtime seeds, not RF seed)
    trials=8
    # Collect per-trial population vectors for decoder: 9-dim L4_E rates in p1 window
    # Also input vectors and Vm vectors
    input_vecs=[]; vm_vecs=[]; spike_vecs=[]; labels=[]
    # Need to run multiple matched trials with different runtime seeds but same stimulus
    for rep in range(trials):
        for label, arr in [("A", arrA), ("B", arrB)]:
            key=jax.random.PRNGKey(100+rep*10 + (0 if label=="A" else 1))
            V,S,src,diag = simulate_edge_recurrent_izhikevich(em, el, N_STEPS, DT_MS, key, dtype="float32", drive_schedule=jnp.asarray(arr, dtype=jnp.float32), record_edge_current=False)
            V=np.asarray(V); S=np.asarray(S)
            # Input vector per L4_E: mean drive in p1 window per neuron
            inp = np.array(arr)[P1_S:P1_E, l4e_idx].mean(axis=0)
            # Vm vector: mean Vm in p1 per L4_E
            vm = V[P1_S:P1_E, l4e_idx].mean(axis=0)
            # Spike vector: mean rate per L4_E
            spk = S[P1_S:P1_E, l4e_idx].mean(axis=0)*(1000/DT_MS)
            input_vecs.append(inp); vm_vecs.append(vm); spike_vecs.append(spk); labels.append(0 if label=="A" else 1)
            # Also check per-neuron mapping for first rep
            if rep==0:
                print(f"rep0 {label}: inp mean {inp.mean():.3f} Vm mean {vm.mean():.1f} spk mean {spk.mean():.1f}")
                # Per neuron input->Vm->spike correlation
                for i_idx, nid in enumerate(l4e_idx[:3]):
                    print(f"  neuron {nid}: d_inp {float(np.array(arr)[P1_S:P1_E, nid].mean()):.3f} Vm {float(vm[i_idx]):.1f} spk {float(spk[i_idx]):.1f} w RF {centers[nid]}")

    input_vecs=np.array(input_vecs); vm_vecs=np.array(vm_vecs); spike_vecs=np.array(spike_vecs); labels=np.array(labels)
    print(f"collected {len(labels)} vectors: input {input_vecs.shape}, vm {vm_vecs.shape}, spike {spike_vecs.shape}")

    # Population decoder: simple nearest centroid, leave-one-out CV
    def lovo_acc(X,y):
        # X (n_trials,9), y 0/1
        n=len(y)
        correct=0
        for i in range(n):
            train_idx=[j for j in range(n) if j!=i]
            Xtr=X[train_idx]; ytr=y[train_idx]
            # centroids
            c0=Xtr[ytr==0].mean(axis=0); c1=Xtr[ytr==1].mean(axis=0)
            # distance of held-out
            d0=np.linalg.norm(X[i]-c0); d1=np.linalg.norm(X[i]-c1)
            pred=0 if d0<d1 else 1
            if pred==y[i]:
                correct+=1
        return correct/n

    def perm_p(X,y, n_perm=200):
        acc=lovo_acc(X,y)
        # null via label permutation
        null=[]
        import random
        for _ in range(n_perm):
            yp=np.random.permutation(y)
            null.append(lovo_acc(X, yp))
        null=np.array(null)
        p=float((np.sum(null>=acc)+1)/(n_perm+1))
        return acc, p, float(null.mean()), float(null.std())

    for name, X in [("input",input_vecs),("Vm",vm_vecs),("spike",spike_vecs)]:
        acc,p, null_m, null_s = perm_p(X, labels, n_perm=200)
        print(f"{name} decoder LOO acc {acc:.3f} p {p:.3f} null {null_m:.3f}±{null_s:.3f}")

    # Check population-size: why 9? Already printed expected. Analytic RF coverage
    # 32x32 field, 9 L4_E units, spacing 3.2, sigma 1.8, A blob center (8,8) B (24,24)
    # Compute coverage: for each pixel, max weight across 9 units?
    # Approx: weight matrix for L4_E only
    w_l4e = np.array([op.weights[i] for i in l4e_idx])  # (9,1024)
    # Reshape to 32x32 and check coverage of A/B blobs
    # Blob masks: Gaussian blob at (8,8) sigma? Use stimulus_pattern
    print(f"RF coverage: w_l4e max per pixel mean {w_l4e.max(axis=0).mean():.4f} min {w_l4e.max(axis=0).min():.4f}")
    print(f"w_l4e per-neuron L1 sum {w_l4e.sum(axis=1)[:3]} (should be 1)")

    # M2 audit: typed E mixture M2 assignments for L4_E
    # Check if RS/CH/E_FS correlates with RF location or drive
    # Need to get M2 assignments: they are via emitter a/c/d clusters. We can infer via a values
    a_vals=np.asarray(em.a)[l4e_idx]
    # RS a0.02, CH a0.02? Actually RS a0.02 CH a0.02 E_FS a0.10 -> E_FS has a0.10
    # So a0.10 indicates E_FS
    efs_mask=[a>0.05 for a in a_vals]
    print(f"M2 per L4_E a values {a_vals} -> E_FS count {sum(efs_mask)}")
    # Correlate with drive delta
    for i, nid in enumerate(l4e_idx):
        print(f"  L4_E {nid} a {a_vals[i]:.3f} drive delta {float(driveA[nid]-driveB[nid]):.3f} RF {centers[nid]}")

    # Save summary
    out=pathlib.Path("results/w4a_l4_representation.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    summary={
        "n_l4_e": len(l4e_idx),
        "expected": 9,
        "frac": 0.15*0.60,
        "drive_deltas": deltas_drive.tolist(),
        "global_parity": {"sumA": sumA, "sumB": sumB, "ratio": sumA/sumB if sumB else 0},
        "w_verbose": "audit done",
    }
    with open(out,"w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved to {out}")
    # Classification
    # S1 INPUT_NOT_DISCRIMINATIVE if input vectors not decodable
    # S2 RF_SAMPLING_LIMITED if n=9 insufficient coverage
    # S3 INPUT->VM lost, S4 VM->spikes lost, S5 valid
    # Use input decoder acc as S1 test
    acc_in,_ ,_,_= perm_p(input_vecs, labels, n_perm=100)
    acc_vm,_,_,_= perm_p(vm_vecs, labels, n_perm=100)
    acc_sp,_,_,_= perm_p(spike_vecs, labels, n_perm=100)
    print(f"acc_in {acc_in:.2f} acc_vm {acc_vm:.2f} acc_sp {acc_sp:.2f}")
    if acc_in < 0.60:
        cls="S1 INPUT_NOT_DISCRIMINATIVE"
    elif acc_in >=0.60 and acc_vm <0.60:
        cls="S3 INPUT_DISCRIMINATIVE_BUT_VM_LOST"
    elif acc_vm >=0.60 and acc_sp <0.60:
        cls="S4 VM_DISCRIMINATIVE_BUT_SPIKES_LOST"
    elif acc_sp >=0.60:
        cls="S5 L4_POPULATION_REPRESENTATION_VALID"
    else:
        # Check n=9 coverage: if 32x32 field with 9 units spacing 3.2 covers only ~9* (pi*1.8^2) /1024 ≈ 0.09, maybe insufficient
        cls="S2 RF_SAMPLING_LIMITED or S_UNRESOLVED"
    print("classification:", cls)

if __name__=="__main__":
    run()
