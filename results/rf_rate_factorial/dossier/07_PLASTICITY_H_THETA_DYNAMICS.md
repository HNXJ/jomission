# 07 — PLASTICITY: H/Θ DYNAMICS & TIMESCALES (reference vs LONG rate)

_Characterization dossier. This report documents the **actual** H (resource) and Θ (weight `w`) dynamics of the Jomission simulation across the two plasticity-rate levels of the RF×Rate factorial, and the timescale audit that motivates them. Every quantity is tagged `MEASURED` (from sealed run artifacts `results/rf_rate_factorial/*/E_*_analysis.json`, `recording/H_t.json`, `recording/continuation_state.json`), `DERIVED` (computed from measured/design constants), `MODEL_ASSUMPTION` (paradigm/design choice), or `UNRESOLVED` (not measured). Sources: `manifests/rf_rate_factorial_design.json`, `manifests/four_cell_lifecycle_seal.json`, `docs/PLASTICITY_RATE_INTERVENTION_DESIGN.md`, `E_B_analysis.json`, `E_C_analysis.json`, `jomission/ablations/rf_rate_factorial.py`._

---

## 1. Declared vs actual timescale — the dead-key audit

| quantity | value | provenance | tag |
|---|---|---|---|
| declared `tau_theta_s` | 1000 s | `jomission/dynamics/hdp.py:16`, `configs/default.json:20` | MODEL_ASSUMPTION (dead) |
| consumed-by-JaxFNE? | **no** — zero consumers of bare `tau_theta_s` | `grep` across `jaxfne` site-packages (`emitters.py:2856`, `_hdp_adaptive.py:171-172` consume `controller_tau_H_s`/`controller_tau_theta_s`/`tau_0_ms`) | MEASURED |
| production HDP path | **node-local** (`h_state_locality` absent → `"node"`), via `RuntimeConfig(enable_hdp=True, hdp_params=hp)` | `full_run.py:109-112`, `lifecycle.py:95` | MEASURED |
| population controller (`tau_H_s=0.2`, `tau_Θ_s=2.0`) | **not instantiated** in production (requires `controller_B`/`m_ei_edge_mask` absent in `v1_pfc_aaab`) | `_hdp_adaptive.py:53,171-172` | MEASURED |

**Finding:** the declared 1000 s timescale is a dead key never consumed by the engine. The implemented hierarchy is `τ_X (ms) << τ_H (0.03–4 s) << τ_w (≈1–2.5 s)` — it satisfies the weak `≫ τ_X` clause but **violates the exposure-scale design** (plasticity completes in seconds, not across the 1202.24 s exposure) at the reference rate. This motivates the LONG (200×) rate level.

## 2. Effective H timescales (node path)

`tau_i = tau_0_ms · size³` (cube law); linearized `τ_eff ≈ tau_i / (K_ctrl + γ·⟨r⟩)`, `K_ctrl=0.15`, `γ⟨r⟩≈0.0025`.

| cell type | size | tau_i | τ_eff (τ_0=5.0) | tag |
|---|---|---|---|---|
| E | 5.0 | 625 ms | **4.1 s** (design reference) | DERIVED |
| PV | 1.0 | 5 ms | 0.03 s | DERIVED |
| SST/VIP | 1.5 | 16.9 ms | 0.11 s | DERIVED |

Empirical (reference rate, cell B, coarse fit): `tau_eff_s_fit ≈ 3.53 s` (H∞ 1.0621), flagged EXPLORATORY/under-determined because sampling period 4.624 s ≥ design τ 4.1 s (`<2 samples/e-fold`) — **NOT an OBSERVED design-level timescale**. Reference τ_eff,E = 4.1 s is `SPECIFIED_NOT_MEASURED_FOR_THIS_RUN` (tag DERIVED/SPECIFIED).

**Θ (`w`) reference timescale ≈ 2.5 s** [MEASURED — reference-cell weight relaxation probe]. Weight restoring `K_w_ctrl=0.001` gives `τ_w = 1/K_w_ctrl = 1000 ms = 1.0 s` (DERIVED); the observed weight relaxation toward the operating point on the reference cell is ≈2.5 s. Note: within the factorial runs Θ(w) per-trial means are constant to ~1e-6 (B) or to machine precision (C, K_HDP=0) — `Θ_effective_timescale = NO_RESOLVABLE_DYNAMICS` — so the 2.5 s reference comes from the isolated weight probe, not the run trajectories.

## 3. Reference rate (cell B: RF on, Rate standard) — H saturates

From `E_B_analysis.json` (sealed, B `SEAL_RECONSTRUCTED_from_heartbeat`):

| quantity | value | tag |
|---|---|---|
| H run mean / min / max | 1.06209 / 1.04056 / 1.06234 | MEASURED |
| H bounds `[0.1,10]` | ok | MEASURED |
| exposure_H_std | 7.14e-5 | MEASURED |
| exposure_Θ ptp | 2.31e-7 (stationary) | MEASURED |
| interpretation | H & Θ stationary <1e-4 across 1202.24 s exposure; **no exposure-scale gradient** | MEASURED/DERIVED |
| post rate phenotypes | `NOT_RECORDED` (save crash) | UNRESOLVED |

## 4. LONG rate (cell C: RF off, Rate slow, `tau_0_ms=1000`, 200×) — H drifts

From `E_C_analysis.json` (sealed, C COMPLETE):

| quantity | value | tag |
|---|---|---|
| tau_0_ms (intervention) | 1000 ms = 5.0 × 200 | MODEL_ASSUMPTION / DERIVED |
| E tau_i | 125 s (5.0→size³·1000) | DERIVED |
| **τ_eff,E design (intended)** | **833 s** (76% of asymptote at 1200 s) | SPECIFIED (DERIVED) |
| **τ_eff measured (exploratory fit, full run)** | **1092.69 s** (430 pts, 236 samples/e-fold, residual 2.56e-4) | MEASURED (EXPLORATORY) |
| τ_eff measured (exposure+post only) | 1123.76 s | MEASURED (EXPLORATORY) |
| ratio measured/design | 1.31 | DERIVED |
| startup rise τ (EXPLORATORY, undersampled) | 12.56 s | MEASURED (EXPLORATORY) |
| H run mean / min / max | 1.01390 / 1.00274 / 1.01790 | MEASURED |
| exposure H ptp / exposure→post shift | 0.005294 / −0.001807 | MEASURED |
| H bounds `[0.1,10]` | ok | MEASURED |
| H interpretation | **NOT stationary**: slow decay of ~5.3e-3 (ptp) spans the full exposure; consistent with unresolved slow relaxation, not an adaptation transient that settles | MEASURED/DERIVED |
| Θ (w) trajectory | constant 0.00323832 (K_HDP=0) | MEASURED |
| Θ vs `w_floor=0.01` | 0.00324 **< w_floor** → clamp/barrier non-binding flag | UNRESOLVED |

**Key contrast:** reference H is stationary in exposure (cell B, std 7e-5, τ≈3.5–4.1 s); LONG H is still decaying at ~1093 s (cell C), i.e. the rate-only `tau_0_ms` intervention achieved its intended effect — moving H from second-scale saturation to exposure-scale drift. This is the measurable consequence of the ×3-rejected / 200×-recommended design.

## 5. Why ×3 was rejected (frozen design, `rf_rate_factorial.py`)

| intervention | τ_eff,E | 95% time | fraction of 1200 s exposure still drifting | tag |
|---|---|---|---|---|
| ×1 (reference) | 4.1 s | 12 s | 1% | DERIVED |
| ×3 | 12.3 s | 37 s | 3% | DERIVED (rejected) |
| ×200 (**LONG**) | 833 s | ~1200 s | 100% (still ramping) | DERIVED (adopted) |

## 6. Provenance caveats & number registry

- All `E_*_analysis.json` report `declared_dense_sampling_materialized=false`: `H_t`/`H_t_dense.npz` store **per-trial means** (spacing 4624 ms), not per-step traces; the dense early-time H probe was declared but not materialized (engine returned no H_trace). Slow-component fits are well-sampled (>230 samples/e-fold) but the startup rise is under-sampled.
- Timescales are single-exponential approximations over phase-varying drive; reported EXPLORATORY, with the design value 833 s remaining the SPECIFIED reference.
- Cell B post-phase rate/field phenotypes were lost to the save crash → UNRESOLVED (see dossier 08).
- H/Θ bounds, `w_floor/ceiling`, `H_min/max`, `alpha/gamma/K_ctrl/K_w_ctrl` are unchanged by the rate intervention (fixed points & bounds preserved — rate-only knob `tau_0_ms`).
- Every number above appears in a table with evidence tag and source; per-trial-resolution H trajectories are `MEASURED`; dense sub-trial H dynamics are `UNRESOLVED`.
