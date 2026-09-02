"""Authoritative Paired 2-s H_ON vs H_OFF Canonical Invariance Assay.

Protocol:
- 2000 ms spontaneous duration, dt = 0.1 ms (20,000 steps).
- Poisson background: 2000 Hz, amplitude 2.0, target="all", seed = seed + 7919.
- Seeds: {0, 1, 2}.
- Matched network realization, laminar delays, initial state, RNG streams.
- H_ON uses frozen boundary stabilization parameters:
    tau_r_s = 0.3, tau_H_E_s = 4.0, tau_H_I_s = 1.0, K_H = 0.1, g_H = 0.22, r_bar_init = 8.0.

Predeclared primary equivalence tolerances:
    |mu_ON / mu_OFF - 1| < 0.05
    |CV_ON / CV_OFF - 1| < 0.10
    |rho_ON / rho_OFF - 1| < 0.10
    |SD_ON / SD_OFF - 1| < 0.10 (NO_CLAMP physiological heterogeneity)
"""

import json
import time
import numpy as np
import jax
import jax.numpy as jnp

from jomission.network.builder import build_jomission_model
import jaxfne as jtfne
from jaxfne import Simulation, RuntimeConfig

DT_MS = 0.1
DUR_MS = 2000.0
N_STEPS = int(DUR_MS / DT_MS)
SEEDS = [0, 1, 2]

# Predeclared tolerances
TOL_MU = 0.05
TOL_CV = 0.10
TOL_RHO = 0.10
TOL_SD = 0.10

HDP_FROZEN_PARAMS = {
    "enable_boundary_stabilization": True,
    "tau_r_s": 0.3,
    "tau_H_E_s": 4.0,
    "tau_H_I_s": 1.0,
    "K_H": 0.1,
    "g_H": 0.22,
    "r_bar_init": 8.0,
}


def compute_metrics(S, DT_MS, DUR_MS):
    """Compute global and temporal metrics from spike raster S (n_steps, n_neurons)."""
    rates = S.mean(axis=0) * (1000.0 / DT_MS)
    mu_rate = float(np.mean(rates))
    sd_rate = float(np.std(rates))

    # CV_ISI per neuron with >= 3 spikes
    cvs = []
    for n in range(S.shape[1]):
        times = np.where(S[:, n] > 0.5)[0] * DT_MS
        if len(times) >= 3:
            isi = np.diff(times)
            if isi.mean() > 0:
                cvs.append(float(isi.std() / isi.mean()))
    cv_mean = float(np.mean(cvs)) if cvs else 0.0
    cv_median = float(np.median(cvs)) if cvs else 0.0

    # Fano factor (100 ms windows)
    win_steps = int(100.0 / DT_MS)
    fanos = []
    for n in range(S.shape[1]):
        counts = [float(S[i : i + win_steps, n].sum()) for i in range(0, S.shape[0], win_steps)]
        if np.mean(counts) > 0:
            fanos.append(float(np.var(counts) / np.mean(counts)))
    fano_median = float(np.median(fanos)) if fanos else 0.0

    # Spike count correlation rho (10 ms bins)
    bin_steps = int(10.0 / DT_MS)
    n_sample = min(50, S.shape[1])
    binned = []
    for n in range(n_sample):
        s_arr = S[:, n]
        binned_n = [float(s_arr[i : i + bin_steps].sum()) for i in range(0, S.shape[0], bin_steps)]
        binned.append(binned_n)
    binned = np.array(binned)
    rhos = []
    for i in range(n_sample):
        for j in range(i + 1, n_sample):
            if np.std(binned[i]) > 1e-9 and np.std(binned[j]) > 1e-9:
                c = np.corrcoef(binned[i], binned[j])[0, 1]
                if np.isfinite(c):
                    rhos.append(float(c))
    rho_mean = float(np.mean(rhos)) if rhos else 0.0

    return {
        "mu_rate_hz": mu_rate,
        "sd_rate_hz": sd_rate,
        "cv_isi_mean": cv_mean,
        "cv_isi_median": cv_median,
        "fano_median": fano_median,
        "rho_mean": rho_mean,
        "rates": rates,
    }


def run_paired_assay():
    print("=" * 70)
    print("STARTING CANONICAL 2-S H_ON VS H_OFF PAIRED INVARIANCE ASSAY")
    print(f"Seeds: {SEEDS}, Duration: {DUR_MS} ms, dt: {DT_MS} ms")
    print("Predeclared Tolerances:")
    print(f"  |mu_ON/mu_OFF - 1| < {TOL_MU:.2f}")
    print(f"  |CV_ON/CV_OFF - 1| < {TOL_CV:.2f}")
    print(f"  |rho_ON/rho_OFF - 1| < {TOL_RHO:.2f}")
    print(f"  |SD_ON/SD_OFF - 1| < {TOL_SD:.2f} (NO_CLAMP Heterogeneity)")
    print("=" * 70)

    per_seed_results = {}
    h_on_diagnostics_summary = {}

    for seed in SEEDS:
        print(f"\n--- Running Seed {seed} ---")
        t0 = time.time()
        model = build_jomission_model(n_per_area=100, seed=seed, dt_ms=DT_MS)
        tbl = model.neuron_table()

        pd = {
            "rate_hz": 2000.0,
            "amplitude": 2.0,
            "target": "all",
            "seed": seed + 7919,
        }

        # 1. Run H_OFF
        print("  Simulating H_OFF...")
        rc_off = RuntimeConfig(recurrent_backend="edge_list", enable_hdp=False)
        sim_off = Simulation(duration_ms=DUR_MS, dt_ms=DT_MS, seed=seed, runtime=rc_off, poisson_drive=pd)
        sig_off = jtfne.simulate(model, sim_off)
        S_off = np.asarray(sig_off.spikes)
        V_off = np.asarray(sig_off.V_m)

        # 2. Run H_ON
        print("  Simulating H_ON...")
        rc_on = RuntimeConfig(
            recurrent_backend="edge_list",
            enable_hdp=True,
            hdp_params=HDP_FROZEN_PARAMS,
        )
        sim_on = Simulation(duration_ms=DUR_MS, dt_ms=DT_MS, seed=seed, runtime=rc_on, poisson_drive=pd)
        sig_on = jtfne.simulate(model, sim_on)
        S_on = np.asarray(sig_on.spikes)
        V_on = np.asarray(sig_on.V_m)
        h_diag = model.last_hdp_diagnostics()

        elapsed = time.time() - t0
        print(f"  Completed in {elapsed:.2f}s")

        # Check finite
        assert np.all(np.isfinite(V_off)), f"H_OFF V_m non-finite for seed {seed}"
        assert np.all(np.isfinite(V_on)), f"H_ON V_m non-finite for seed {seed}"

        # Compute metrics
        m_off = compute_metrics(S_off, DT_MS, DUR_MS)
        m_on = compute_metrics(S_on, DT_MS, DUR_MS)

        # Relative deltas
        rel_mu = (m_on["mu_rate_hz"] - m_off["mu_rate_hz"]) / m_off["mu_rate_hz"]
        rel_sd = (m_on["sd_rate_hz"] - m_off["sd_rate_hz"]) / m_off["sd_rate_hz"]
        rel_cv = (m_on["cv_isi_mean"] - m_off["cv_isi_mean"]) / m_off["cv_isi_mean"]
        rel_rho = (m_on["rho_mean"] - m_off["rho_mean"]) / m_off["rho_mean"] if abs(m_off["rho_mean"]) > 1e-6 else 0.0

        pass_mu = abs(rel_mu) < TOL_MU
        pass_sd = abs(rel_sd) < TOL_SD
        pass_cv = abs(rel_cv) < TOL_CV
        pass_rho = abs(rel_rho) < TOL_RHO

        print(f"  Seed {seed} Metrics:")
        print(f"    Rate: OFF = {m_off['mu_rate_hz']:.2f} Hz, ON = {m_on['mu_rate_hz']:.2f} Hz -> rel delta = {rel_mu:+.4f} (Pass: {pass_mu})")
        print(f"    SD (NO_CLAMP): OFF = {m_off['sd_rate_hz']:.2f}, ON = {m_on['sd_rate_hz']:.2f} -> rel delta = {rel_sd:+.4f} (Pass: {pass_sd})")
        print(f"    CV_ISI: OFF = {m_off['cv_isi_mean']:.4f}, ON = {m_on['cv_isi_mean']:.4f} -> rel delta = {rel_cv:+.4f} (Pass: {pass_cv})")
        print(f"    rho: OFF = {m_off['rho_mean']:.4f}, ON = {m_on['rho_mean']:.4f} -> rel delta = {rel_rho:+.4f} (Pass: {pass_rho})")

        # Analyze H diagnostics
        H_trace = np.asarray(h_diag["H_trace"])       # (n_steps, 400)
        r_trace = np.asarray(h_diag["r_bar_trace"])   # (n_steps, 400)
        I_trace = np.asarray(h_diag["I_H_trace"])     # (n_steps, 400)

        # Dead zone occupancy: fraction of (time, neuron) with 1.0 <= r_bar <= 19.5
        in_dead_zone = (r_trace >= 1.0) & (r_trace <= 19.5)
        frac_dead_zone_global = float(np.mean(in_dead_zone))

        # Per cell type dead-zone and H
        ct_stats = {}
        for ct in ["E", "PV", "SST", "VIP"]:
            ct_idx = [i for i, r in enumerate(tbl) if r["cell_type"] == ct]
            if ct_idx:
                sub_r = r_trace[:, ct_idx]
                sub_H = H_trace[:, ct_idx]
                sub_I = I_trace[:, ct_idx]
                ct_stats[ct] = {
                    "count": len(ct_idx),
                    "mean_rate_hz": float(np.mean(m_on["rates"][ct_idx])),
                    "dead_zone_occupancy": float(np.mean((sub_r >= 1.0) & (sub_r <= 19.5))),
                    "mean_H": float(np.mean(sub_H)),
                    "min_H": float(np.min(sub_H)),
                    "max_H": float(np.max(sub_H)),
                    "mean_I_H": float(np.mean(sub_I)),
                    "min_I_H": float(np.min(sub_I)),
                    "max_I_H": float(np.max(sub_I)),
                }

        # B1 Population coverage check (active pops in [0.1, 50] Hz)
        pop_keys = set((r["area"], r["layer"], r["cell_type"]) for r in tbl)
        b1_off_active = 0
        b1_on_active = 0
        for (a, l, ct) in pop_keys:
            pidx = [i for i, r in enumerate(tbl) if r["area"] == a and r["layer"] == l and r["cell_type"] == ct]
            if pidx:
                r_off_pop = np.mean(m_off["rates"][pidx])
                r_on_pop = np.mean(m_on["rates"][pidx])
                if 0.1 <= r_off_pop <= 50.0:
                    b1_off_active += 1
                if 0.1 <= r_on_pop <= 50.0:
                    b1_on_active += 1
        b1_coverage_off = b1_off_active / len(pop_keys)
        b1_coverage_on = b1_on_active / len(pop_keys)

        per_seed_results[str(seed)] = {
            "H_OFF": {
                "mu_rate_hz": m_off["mu_rate_hz"],
                "sd_rate_hz": m_off["sd_rate_hz"],
                "cv_isi_mean": m_off["cv_isi_mean"],
                "cv_isi_median": m_off["cv_isi_median"],
                "fano_median": m_off["fano_median"],
                "rho_mean": m_off["rho_mean"],
                "b1_coverage": b1_coverage_off,
            },
            "H_ON": {
                "mu_rate_hz": m_on["mu_rate_hz"],
                "sd_rate_hz": m_on["sd_rate_hz"],
                "cv_isi_mean": m_on["cv_isi_mean"],
                "cv_isi_median": m_on["cv_isi_median"],
                "fano_median": m_on["fano_median"],
                "rho_mean": m_on["rho_mean"],
                "b1_coverage": b1_coverage_on,
            },
            "relative_changes": {
                "rel_mu": rel_mu,
                "rel_sd": rel_sd,
                "rel_cv": rel_cv,
                "rel_rho": rel_rho,
            },
            "pass_checks": {
                "mu": pass_mu,
                "sd_heterogeneity": pass_sd,
                "cv": pass_cv,
                "rho": pass_rho,
            },
            "H_dynamics": {
                "dead_zone_occupancy_global": frac_dead_zone_global,
                "min_H_global": float(np.min(H_trace)),
                "max_H_global": float(np.max(H_trace)),
                "mean_H_global": float(np.mean(H_trace)),
                "min_I_H_global": float(np.min(I_trace)),
                "max_I_H_global": float(np.max(I_trace)),
                "mean_I_H_global": float(np.mean(I_trace)),
                "by_cell_type": ct_stats,
            },
        }

    # Aggregate across seeds
    all_pass_mu = all(v["pass_checks"]["mu"] for v in per_seed_results.values())
    all_pass_sd = all(v["pass_checks"]["sd_heterogeneity"] for v in per_seed_results.values())
    all_pass_cv = all(v["pass_checks"]["cv"] for v in per_seed_results.values())
    all_pass_rho = all(v["pass_checks"]["rho"] for v in per_seed_results.values())
    overall_invariance_pass = all_pass_mu and all_pass_sd and all_pass_cv and all_pass_rho

    receipt = {
        "assay": "2S_CANONICAL_H_ON_VS_H_OFF_INVARIANCE",
        "verdict": "PASS" if overall_invariance_pass else "FAIL",
        "predeclared_tolerances": {
            "rel_mu_tol": TOL_MU,
            "rel_sd_tol": TOL_SD,
            "rel_cv_tol": TOL_CV,
            "rel_rho_tol": TOL_RHO,
        },
        "overall_invariance_pass": overall_invariance_pass,
        "gates": {
            "rate_invariance_pass": all_pass_mu,
            "no_clamp_heterogeneity_pass": all_pass_sd,
            "cv_isi_invariance_pass": all_pass_cv,
            "rho_invariance_pass": all_pass_rho,
        },
        "per_seed": per_seed_results,
    }

    out_path = "results/h_on_h_off_invariance_2s.json"
    with open(out_path, "w") as f:
        json.dump(receipt, f, indent=2)
    print(f"\nSaved authoritative receipt to {out_path}")
    print(f"OVERALL VERDICT: {receipt['verdict']}")


if __name__ == "__main__":
    run_paired_assay()
