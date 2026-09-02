"""Canonical Paired 2-s H_ON vs H_OFF Invariance Assay — Isolated Boundary H (K_HDP = 0).

Protocol:
- Duration: 2000 ms, dt = 0.1 ms (20,000 steps).
- Poisson background: 2000 Hz, amplitude 2.0, target="all", seed = seed + 7919.
- Seeds: {0, 1, 2}.
- Matched network realization, laminar delays, initial state, RNG streams.
- Frozen boundary stabilization parameters with strict HDP isolation:
    enable_boundary_stabilization = True
    tau_r_s = 0.3
    tau_H_E_s = 4.0
    tau_H_I_s = 1.0
    K_H = 0.1
    g_H = 0.22
    r_bar_init = 8.0
    K_HDP = 0.0          # Strict isolation: no weight plasticity
    K_w_ctrl = 0.0       # w(t) = w(0) static

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

TOL_MU = 0.05
TOL_CV = 0.10
TOL_RHO = 0.10
TOL_SD = 0.10

R_L = 0.5
R_H = 20.0
DELTA = 0.5  # dead-zone margin: 1.0 to 19.5 Hz

HDP_BOUNDARY_ISOLATED_PARAMS = {
    "enable_boundary_stabilization": True,
    "tau_r_s": 0.3,
    "tau_H_E_s": 4.0,
    "tau_H_I_s": 1.0,
    "K_H": 0.1,
    "g_H": 0.22,
    "r_bar_init": 8.0,
    "K_HDP": 0.0,
    "K_w_ctrl": 0.0,
}


def compute_metrics(S, DT_MS, DUR_MS):
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


def run_isolated_boundary_assay():
    print("=" * 75)
    print("CANONICAL 2-S H_ON VS H_OFF PAIRED INVARIANCE ASSAY — ISOLATED BOUNDARY (K_HDP=0)")
    print(f"Seeds: {SEEDS}, Duration: {DUR_MS} ms, dt: {DT_MS} ms")
    print(f"HDP Isolation: K_HDP = 0.0, K_w_ctrl = 0.0 (w(t) = w(0))")
    print("Predeclared Tolerances:")
    print(f"  |mu_ON/mu_OFF - 1| < {TOL_MU:.2f}")
    print(f"  |CV_ON/CV_OFF - 1| < {TOL_CV:.2f}")
    print(f"  |rho_ON/rho_OFF - 1| < {TOL_RHO:.2f}")
    print(f"  |SD_ON/SD_OFF - 1| < {TOL_SD:.2f} (NO_CLAMP Heterogeneity)")
    print("=" * 75)

    per_seed_results = {}

    for seed in SEEDS:
        print(f"\n>>> Running Seed {seed} <<<")
        t0 = time.time()
        model = build_jomission_model(n_per_area=100, seed=seed, dt_ms=DT_MS)
        tbl = model.neuron_table()
        w0 = np.asarray(model.params["edge_list"].weight)

        pd = {
            "rate_hz": 2000.0,
            "amplitude": 2.0,
            "target": "all",
            "seed": seed + 7919,
        }

        # 1. H_OFF Simulation
        print("  Simulating H_OFF...")
        rc_off = RuntimeConfig(recurrent_backend="edge_list", enable_hdp=False)
        sim_off = Simulation(duration_ms=DUR_MS, dt_ms=DT_MS, seed=seed, runtime=rc_off, poisson_drive=pd)
        sig_off = jtfne.simulate(model, sim_off)
        S_off = np.asarray(sig_off.spikes)
        V_off = np.asarray(sig_off.V_m)

        # 2. H_ON Simulation (with K_HDP=0)
        print("  Simulating H_ON (K_HDP=0)...")
        rc_on = RuntimeConfig(
            recurrent_backend="edge_list",
            enable_hdp=True,
            hdp_params=HDP_BOUNDARY_ISOLATED_PARAMS,
        )
        sim_on = Simulation(duration_ms=DUR_MS, dt_ms=DT_MS, seed=seed, runtime=rc_on, poisson_drive=pd)
        sig_on = jtfne.simulate(model, sim_on)
        S_on = np.asarray(sig_on.spikes)
        V_on = np.asarray(sig_on.V_m)
        h_diag = model.last_hdp_diagnostics()

        elapsed = time.time() - t0
        print(f"  Completed in {elapsed:.2f}s")

        # Verify static weights
        w_fin = np.asarray(h_diag["w_final"])
        max_w_diff = float(np.max(np.abs(w_fin - w0)))
        assert max_w_diff < 1e-5, f"Weights mutated under K_HDP=0! max_diff = {max_w_diff}"
        print(f"  Weight invariance verified: max |w_fin - w0| = {max_w_diff:.2e} (strictly static)")

        # Metrics
        m_off = compute_metrics(S_off, DT_MS, DUR_MS)
        m_on = compute_metrics(S_on, DT_MS, DUR_MS)

        rel_mu = (m_on["mu_rate_hz"] - m_off["mu_rate_hz"]) / m_off["mu_rate_hz"]
        rel_sd = (m_on["sd_rate_hz"] - m_off["sd_rate_hz"]) / m_off["sd_rate_hz"]
        rel_cv = (m_on["cv_isi_mean"] - m_off["cv_isi_mean"]) / m_off["cv_isi_mean"]
        rel_rho = (m_on["rho_mean"] - m_off["rho_mean"]) / m_off["rho_mean"] if abs(m_off["rho_mean"]) > 1e-6 else 0.0

        pass_mu = abs(rel_mu) < TOL_MU
        pass_sd = abs(rel_sd) < TOL_SD
        pass_cv = abs(rel_cv) < TOL_CV
        pass_rho = abs(rel_rho) < TOL_RHO

        print(f"  Metrics (Seed {seed}):")
        print(f"    Global Rate: OFF = {m_off['mu_rate_hz']:.3f} Hz, ON = {m_on['mu_rate_hz']:.3f} Hz -> rel delta = {rel_mu:+.4f} (Pass: {pass_mu})")
        print(f"    SD (Heterogeneity): OFF = {m_off['sd_rate_hz']:.3f}, ON = {m_on['sd_rate_hz']:.3f} -> rel delta = {rel_sd:+.4f} (Pass: {pass_sd})")
        print(f"    CV_ISI: OFF = {m_off['cv_isi_mean']:.4f}, ON = {m_on['cv_isi_mean']:.4f} -> rel delta = {rel_cv:+.4f} (Pass: {pass_cv})")
        print(f"    rho: OFF = {m_off['rho_mean']:.4f}, ON = {m_on['rho_mean']:.4f} -> rel delta = {rel_rho:+.4f} (Pass: {pass_rho})")

        # Extract Traces
        H_trace = np.asarray(h_diag["H_trace"])       # (n_steps, 400)
        r_trace = np.asarray(h_diag["r_bar_trace"])   # (n_steps, 400)
        I_trace = np.asarray(h_diag["I_H_trace"])     # (n_steps, 400)

        # Rate difference by neuron
        delta_r_all = m_on["rates"] - m_off["rates"]

        # Cell-type and Population Decomposition
        # Separate whole-run (0-2000 ms) and post-transient steady state (1000-2000 ms)
        step_1s = int(1000.0 / DT_MS)
        r_ss = r_trace[step_1s:, :]
        H_ss = H_trace[step_1s:, :]
        I_ss = I_trace[step_1s:, :]

        ct_breakdown = {}
        for ct in ["E", "PV", "SST", "VIP"]:
            idx = [i for i, r in enumerate(tbl) if r["cell_type"] == ct]
            if not idx:
                continue
            r_sub = r_trace[:, idx]
            r_sub_ss = r_ss[:, idx]
            H_sub = H_trace[:, idx]
            I_sub = I_trace[:, idx]

            # Probabilities
            p_below_rL = float(np.mean(r_sub < R_L))
            p_dead_zone = float(np.mean((r_sub >= (R_L + DELTA)) & (r_sub <= (R_H - DELTA))))
            p_above_rH = float(np.mean(r_sub > R_H))

            # Post-transient steady state probabilities (1-2 s)
            p_below_rL_ss = float(np.mean(r_sub_ss < R_L))
            p_dead_zone_ss = float(np.mean((r_sub_ss >= (R_L + DELTA)) & (r_sub_ss <= (R_H - DELTA))))
            p_above_rH_ss = float(np.mean(r_sub_ss > R_H))

            # Mean rates and contributions
            r_off_ct = float(np.mean(m_off["rates"][idx]))
            r_on_ct = float(np.mean(m_on["rates"][idx]))
            delta_r_ct = r_on_ct - r_off_ct
            contrib_global = float(len(idx) / 400.0 * delta_r_ct)

            # Transient analysis: when does population mean cross r_L or r_H?
            pop_r_mean_t = np.mean(r_sub, axis=1)  # shape (n_steps,)
            cross_low = np.where(pop_r_mean_t < R_L)[0]
            cross_high = np.where(pop_r_mean_t > R_H)[0]
            t_first_low_ms = float(cross_low[0] * DT_MS) if len(cross_low) > 0 else None
            t_first_high_ms = float(cross_high[0] * DT_MS) if len(cross_high) > 0 else None

            ct_breakdown[ct] = {
                "n_neurons": len(idx),
                "r_off_hz": r_off_ct,
                "r_on_hz": r_on_ct,
                "delta_r_hz": delta_r_ct,
                "contrib_to_global_delta_r": contrib_global,
                "P_below_rL_full": p_below_rL,
                "P_dead_zone_full": p_dead_zone,
                "P_above_rH_full": p_above_rH,
                "P_below_rL_ss": p_below_rL_ss,
                "P_dead_zone_ss": p_dead_zone_ss,
                "P_above_rH_ss": p_above_rH_ss,
                "mean_H": float(np.mean(H_sub)),
                "min_H": float(np.min(H_sub)),
                "max_H": float(np.max(H_sub)),
                "mean_I_H": float(np.mean(I_sub)),
                "min_I_H": float(np.min(I_sub)),
                "max_I_H": float(np.max(I_sub)),
                "t_first_cross_low_ms": t_first_low_ms,
                "t_first_cross_high_ms": t_first_high_ms,
            }

        # Area / Layer / Class granular breakdown
        pop_keys = sorted(list(set((r["area"], r["layer"], r["cell_type"]) for r in tbl)))
        pop_breakdown = {}
        for (a, l, ct) in pop_keys:
            pidx = [i for i, r in enumerate(tbl) if r["area"] == a and r["layer"] == l and r["cell_type"] == ct]
            if not pidx:
                continue
            r_pop = r_trace[:, pidx]
            H_pop = H_trace[:, pidx]
            I_pop = I_trace[:, pidx]
            r_off_p = float(np.mean(m_off["rates"][pidx]))
            r_on_p = float(np.mean(m_on["rates"][pidx]))
            delta_r_p = r_on_p - r_off_p

            pop_breakdown[f"{a}_{l}_{ct}"] = {
                "n": len(pidx),
                "r_off": r_off_p,
                "r_on": r_on_p,
                "delta_r": delta_r_p,
                "contrib": float(len(pidx) / 400.0 * delta_r_p),
                "P_below_rL": float(np.mean(r_pop < R_L)),
                "P_dead_zone": float(np.mean((r_pop >= 1.0) & (r_pop <= 19.5))),
                "P_above_rH": float(np.mean(r_pop > R_H)),
                "mean_H": float(np.mean(H_pop)),
                "mean_I_H": float(np.mean(I_pop)),
            }

        # B1 Population coverage
        b1_off_active = sum(1 for p in pop_breakdown.values() if 0.1 <= p["r_off"] <= 50.0)
        b1_on_active = sum(1 for p in pop_breakdown.values() if 0.1 <= p["r_on"] <= 50.0)

        per_seed_results[str(seed)] = {
            "H_OFF": {
                "mu_rate_hz": m_off["mu_rate_hz"],
                "sd_rate_hz": m_off["sd_rate_hz"],
                "cv_isi_mean": m_off["cv_isi_mean"],
                "cv_isi_median": m_off["cv_isi_median"],
                "fano_median": m_off["fano_median"],
                "rho_mean": m_off["rho_mean"],
                "b1_coverage": b1_off_active / len(pop_breakdown),
            },
            "H_ON": {
                "mu_rate_hz": m_on["mu_rate_hz"],
                "sd_rate_hz": m_on["sd_rate_hz"],
                "cv_isi_mean": m_on["cv_isi_mean"],
                "cv_isi_median": m_on["cv_isi_median"],
                "fano_median": m_on["fano_median"],
                "rho_mean": m_on["rho_mean"],
                "b1_coverage": b1_on_active / len(pop_breakdown),
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
            "cell_type_decomposition": ct_breakdown,
            "population_decomposition": pop_breakdown,
            "global_H_extrema": {
                "min_H": float(np.min(H_trace)),
                "max_H": float(np.max(H_trace)),
                "mean_H": float(np.mean(H_trace)),
                "min_I_H": float(np.min(I_trace)),
                "max_I_H": float(np.max(I_trace)),
                "mean_I_H": float(np.mean(I_trace)),
            },
        }

    all_pass_mu = all(v["pass_checks"]["mu"] for v in per_seed_results.values())
    all_pass_sd = all(v["pass_checks"]["sd_heterogeneity"] for v in per_seed_results.values())
    all_pass_cv = all(v["pass_checks"]["cv"] for v in per_seed_results.values())
    all_pass_rho = all(v["pass_checks"]["rho"] for v in per_seed_results.values())
    overall_invariance_pass = all_pass_mu and all_pass_sd and all_pass_cv and all_pass_rho

    receipt = {
        "assay": "2S_CANONICAL_H_ON_VS_H_OFF_INVARIANCE_ISOLATED_BOUNDARY",
        "hdp_isolation": "K_HDP=0.0, K_w_ctrl=0.0 (w(t)=w(0))",
        "verdict": "PASS" if overall_invariance_pass else "FAIL",
        "classification": "INVARIANCE_PASS" if overall_invariance_pass else "UNIVERSAL_RATE_BOUNDARY_MODEL_MISMATCH",
        "predeclared_tolerances": {
            "rel_mu_tol": TOL_MU,
            "rel_sd_tol": TOL_SD,
            "rel_cv_tol": TOL_CV,
            "rel_rho_tol": TOL_RHO,
        },
        "gates": {
            "rate_invariance_pass": all_pass_mu,
            "no_clamp_heterogeneity_pass": all_pass_sd,
            "cv_isi_invariance_pass": all_pass_cv,
            "rho_invariance_pass": all_pass_rho,
        },
        "per_seed": per_seed_results,
    }

    out_path = "results/h_on_h_off_invariance_isolated_boundary.json"
    with open(out_path, "w") as f:
        json.dump(receipt, f, indent=2)
    print(f"\nSaved authoritative receipt to {out_path}")
    print(f"OVERALL VERDICT: {receipt['verdict']}")
    print(f"CLASSIFICATION: {receipt['classification']}")


if __name__ == "__main__":
    run_isolated_boundary_assay()
