# Plasticity-Rate Intervention Design — Theta/H Timescale Audit & Rate-Only Scaling

**Status: design only (no code mutated)**  
**Author: Subagent D**  
**Date: 2026-08-24**  
**Workspace: `/Users/hamednejat/workspace/computational/jomission`**  
**Frozen baseline: `config_hash 4f9fdeae7428199a`, `hp_hash f327f9d2ad64cc88`, `jaxfne 0.4.17`, `v1_pfc_aaab_hdp_params`**

---

## 1. Objective

Design a plasticity-rate intervention that **only** changes learning timescale and preserves fixed points and bounds, such that `Θ` (adaptive weights `w`, per `jomission/simulation/state_replacement.py:16`) and `H` (per-neuron resource state) evolve **across the ~1200 s exposure** (260 trials × 4624 ms = 1202.24 s, `jomission/simulation/schedule.py:26`), rather than saturating in seconds. Determine whether a ×3 multiplier is correct.

---

## 2. Audit: Declared vs Actual Timescale

### 2.1 Declared (dead key)

- `jomission/dynamics/hdp.py:16` declares `tau_theta_s = 1000.0` ("≫ τ_X")
- `jomission/dynamics/hdp.py:31` maps it to `hdp_params["tau_theta_s"]`
- `jomission/configs/default.json:20` repeats `"tau_theta_s": 1000`

**Finding: `tau_theta_s` (without `controller_` prefix) is never consumed by JaxFNE.**  
`jaxfne/emitters.py:2856` and `jaxfne/_hdp_adaptive.py:171-172` consume `controller_tau_H_s` / `controller_tau_theta_s` (population path) and `tau_0_ms` (node path). `tau_theta_s` matches neither. `HDPConfig.to_jaxfne_hdp_params()` therefore emits a dead key. Verified via `grep` across `jaxfne` site-packages — zero consumers of bare `tau_theta_s`.

### 2.2 Actual code path in production

`jomission/simulation/full_run.py:109-112` and `jomission/simulation/lifecycle.py:95` call:

```python
hp = hdp.v1_pfc_aaab_hdp_params()
runtime = RuntimeConfig(enable_hdp=True, hdp_params=hp)
```

`v1_pfc_aaab_hdp_params` (`jaxfne/hdp_network.py:225`) returns `BASE_HDP_KWARGS_DEFAULT` + `DEFAULT_HDP_V1_PFC_AAAB`:

| key | value | role |
|-----|-------|------|
| `tau_0_ms` | 5.0 | base H integration constant |
| `size_scale_by_cell_type` | E=5.0, PV=1.0, SST/VIP=1.5 | cube-law `tau_i = tau_0_ms * size³` |
| `K_ctrl` | 0.15 | linear `K_ctrl·(1-H)` restoring |
| `alpha` | 0.05 | income `alpha·I_syn` |
| `gamma` | 0.5 | spending `gamma·H·r` |
| `K_HDP` | 0.003 | weight drive gain |
| `K_w_ctrl` | 0.001 | weight magnitude restoring |
| `h_state_locality` | absent → defaults to `"node"` | no `Theta` population controller |

`h_state_locality` not set → `resolve_h_state_locality()` returns `"node"`. The **population restoring controller** (`tau_H_s=0.2 s`, `tau_theta_s=2.0 s`, `H∈R²`, `Θ∈R²`, `dH=(-e-λH)/τ_H`, `dΘ=B·H/τ_Θ`) is **not instantiated** in the production run, despite `HDPConfig(locality="population")` existing elsewhere. `exposure.py:42` does set `h_state_locality=population` for a smoke test, but that path requires `controller_B`/`m_ei_edge_mask` which `v1_pfc_aaab` does not provide — it would raise `ValueError` if actually entered. Thus production uses **node-local** dynamics only.

### 2.3 Effective timescales (node path, current preset)

**H dynamics** (`jaxfne/emitters.py:392, 622-625`):

```
tau_i · dH_i/dt = α·I_syn + β - γ·H·r - δ·W + ρ/H² - dC/dH + K_ctrl·(1-H) + barrier
tau_i = tau_0_ms · size³
H_next = clip(H + (dt/tau_i)·dH)
```

Linearized near fixed point (ignoring barrier, `ρ=0`, `δ=0`):

```
dH/dt ≈ -(K_ctrl + γ·⟨r⟩)/tau_i · (H - H*)
τ_eff ≈ tau_i / (K_ctrl + γ·⟨r⟩)
```

| cell type | size | tau_i | τ_eff (≈ tau_i/K_ctrl, γ⟨r⟩≈0.0025) |
|-----------|------|-------|--------------------------------------|
| E | 5.0 | 625 ms | **4.1 s** |
| PV | 1.0 | 5 ms | 0.03 s |
| SST/VIP | 1.5 | 16.9 ms | 0.11 s |

Empirical confirmation (chained continuation, `jtfne.simulate` + `ContinuationState.dynamic.H`, dt=0.5 ms, AAAB/BBBA alternating, seed 42):

- Single trial H trace: global min 0.778 max 1.087, final mean 1.031, E final 1.020, PV 1.067, SST 0.999 (`H_final` logging above).
- 12-trial chain H mean: 1.0317 → 1.0362; fit `H(t)=H∞+A·exp(-t/τ)` gives **τ=3.87 s, half-life 2.68 s, H∞=1.036** (pcov τ se 0.25 s). Matches E-predicted 4.1 s.
- Saturation: 95% within 3τ ≈ 11.6 s ≈ 2.5 trials. After trial 4, drift <0.0004/trial. Of 260 exposure trials, ~255 see static H.

**Θ (w) dynamics** (`jaxfne/emitters.py:656-665`):

```
dw_exc = K_HDP · (H_post-H_pre) · wmag
dw_inh = -K_HDP · (H_post-H_pre) · wmag
dw_w_ctrl = K_w_ctrl · (w_baseline - wmag)
wmag_next = clip(wmag + dt·(dw_exc/dw_inh + dw_w_ctrl), w_floor, w_ceiling)
```

`K_w_ctrl=0.001` → `τ_w = 1/K_w_ctrl = 1000 ms = 1.0 s` (first-order restoring). Empirical |w| mean drift: trial 0 0.01296 → trial 5 0.01285 (−0.8%, saturated by trial 3). Consistent with ~1 s.

**Population controller timescale (if it were active)**: defaults `τ_H_s=0.2 s`, `τ_Θ_s=2.0 s`, `λ=0.45` → `τ_H,eff=0.44 s`, `τ_Θ=2.0 s` (`jaxfne/_hdp_adaptive.py:171-172`, `population_restoring_derivatives`). Also within 0.2–2.5 s envelope.

**Conclusion**: Declared 1000 s is 250–500× larger than any instantiated timescale. Actual hierarchy `τ_X (ms) << τ_H (0.03–4 s) << τ_w (1 s)` satisfies the weak `≫τ_X` clause in `t3_authorization.json:76`, but **violates the exposure-scale design** — plasticity completes in seconds, not across 1200 s. The prompt's "likely ~0.2–2.5 s, not 1000 s" is confirmed.

---

## 3. Is ×3 Correct?

No. ×3 fails by ~65–100×.

| intervention | H_E τ_eff | w τ | time to 95% | fraction of 1200 s exposure where still drifting |
|--------------|-----------|-----|-------------|--------------------------------------------------|
| current (×1) | 4.1 s | 1.0 s | 12 s | 1% |
| **×3** | **12.3 s** | **3.0 s** (if K co-scaled) | **37 s** | **3%** |
| ×100 | 416 s | 100 s | 1248 s | 100% (if coupled) |

×3 still saturates within the first 8 trials (37 s). The remaining 252 trials (1165 s) show no gradient — exposure-length learning is invisible to T4/T5/Q8 contrasts. Required factor for H to remain visibly drifting at t=1200 s is `τ_desired / τ_current`:

- For `τ_desired = 1200 s` (63% of asymptote at end): `1200/3.87 ≈ 310×` (mean fit) or `1200/4.17 ≈ 288×` (E theory)
- For `τ_desired = 800 s` (midpoint, 78% at end): `800/3.87 ≈ 207×`
- For `τ_desired = 600 s` (95% near end): `600/3.87 ≈ 155×`
- For `w` alone (`τ_w=1 s → 800 s`): 800×

×3 corresponds to targeting 12 s, not 1200 s. It is off by two orders of magnitude.

The confusion likely stems from treating `τ_Θ=1000 s` as actual and applying a cosmetic 3× "slowdown" (1000→3000 s) rather than auditing the 2–4 s realized value.

---

## 4. Rate-Only Intervention Design

### 4.1 Invariant: fixed points & bounds must not move

We require `∂(rate)/∂(param) ≠ 0` while `∂(fixed point)/∂(param)=0` and bounds unchanged.

**H fixed point** solves (`emitters.py:601-621`):

```
0 = α·I_syn(H*) + β - γ·H*·⟨r(H*)⟩ - δ·W(H*) + ρ/H*² - dC/dH|_{H*} + K_ctrl·(1-H*)
```

- `tau_0_ms` (via `tau_i`) divides `dH/dt` but does not appear in the zero-crossing — **pure rate knob**. Validated: scaling `tau_0_ms` 10× and 100× preserves `H∞=1.036` while stretching τ linearly. At `τ_0×100`, early H mean 1.003 vs 1.031 at ×1 but asymptote identical (confirmed via same w_floor/w_ceiling, same α/γ/K_ctrl).
- `K_ctrl`, `α`, `γ`, `δ`, `ρ`, `barrier_c/d` all shift `H*` when background drive `I_syn`, `⟨r⟩`, `W` nonzero at equilibrium (they are; `⟨r⟩≈0.005`). Must **not** be touched for rate-only change.
- `H_min/H_max/barrier_eps` set bounds — untouched.

**w fixed point** solves:

```
0 = K_HDP·ΔH·w*  +  K_w_ctrl·(w_baseline - w*)    (E branch sign, `emitters.py:663-664`)
w* = K_w_ctrl·w_baseline / (K_w_ctrl - K_HDP·ΔH)   (ΔH ≡ H_post-H_pre)
```

Fixed point depends on **ratio** `r = K_HDP/K_w_ctrl`, not absolute scale. Therefore:

- Co-scaling `K_HDP' = s·K_HDP`, `K_w_ctrl' = s·K_w_ctrl` → `r'=r` → `w*` invariant, rate ∝ `s` (since `dt·K` scales). This is the only rate-only pair for weights. Verified: `τ_w = 1/K_w_ctrl` scales inversely.
- Scaling either alone → `r` changes → `w*` moves (non-rate).
- `w_floor/w_ceiling` set hard bounds — untouched.
- `H_boost_gain`, `w_floor/ceiling` etc. are scaffold, not rate.

**Bounds & locality**: `H_min=0.1, H_max=10, w_floor=0.01, w_ceiling=10, bounds_lo/hi, H_boost_gain, barrier_c/d, α/γ/δ` all frozen.

### 4.2 Recommended intervention

Primary: **slow H via `tau_0_ms`** (cleanest, single knob, guaranteed fixed-point preservation). Secondary co-scaling of `K_HDP/K_w_ctrl` optional if weight-specific slowing is needed.

#### Recommended center: **200× on `tau_0_ms` alone** (H rate only)

```python
base = hdp.v1_pfc_aaab_hdp_params()  # tau_0_ms=5.0, K_HDP=0.003, K_w_ctrl=0.001, ...
intervention = dict(base)
intervention["tau_0_ms"] = 5.0 * 200  # 1000 ms
# K_HDP, K_w_ctrl, K_ctrl, alpha, gamma, H_min/max, w_floor/ceiling UNCHANGED
```

- New `tau_i`: E 125 s, PV 1.0 s, SST/VIP 3.375 s
- New `τ_eff,E`: 125/0.15 ≈ 833 s (γ-corrected ~800 s)
- Predicted exposure trajectory: `H(t)=1+(1.036-1)·(1-exp(-t/833))` → at 1200 s, **76% of asymptote** (H≈1.027), still visibly ramping. At trial 130 (600 s): 51% (H≈1.018). Provides continuous gradient across entire exposure, detectable in T4/T5 pre/post contrasts while preserving 5-seed stability envelope (slower H is more stable, not less; verified stable at τ_0×10 and ×100 with same K_ctrl; larger τ reduces overshoot, see H std drop 0.024→0.006 at ×100).
- `w` remains fast (τ_w=1 s) so weight tracks current H each trial — overall weight phenotype inherits H's exposure timescale without additional slowing. Empirically at `τ_0×200, K unchanged`: |w| drift 0.01274→0.01292 over 6 trials (1.4% in 27 s, monotonic, not yet saturated) vs. `τ_0×100,K/100` which over-slows w to 0.01% per 6 trials.

#### Alternative tiers (choose one, not a sweep)

| tier | `tau_0_ms` | factor | H_E τ_eff | Θ (w) `τ_w` | 1200 s attainment | use when |
|------|-----------|--------|-----------|-------------|-------------------|----------|
| **A (recommended)** | 1000 ms | 200× | 833 s | 1 s (unchanged) | 76% | default: H bottleneck governs, w tracks |
| B (conservative) | 500 ms | 100× | 416 s | 1 s | 94% | if need near-complete within exposure |
| C (exact 1200 s) | 1500 ms | 300× | 1250 s | 1 s | 62% | if require still accelerating at end |
| D (coupled slow) | 1000 ms & `K_HDP=0.000015, K_w_ctrl=5e-06` | 200× & 200× co-scaled | 833 s | 200 s | H 76%, w 99.7%? w 200 s still completes in 600 s | only if weight transients must themselves be exposure-scale (doubles suppression, see data: co-scaled w drift 200× smaller) |

**Not recommended**: ×3 (15 ms, 12.5 s) — indistinguishable from current; any factor <50 leaves H saturated within first 10% of exposure.

For **population locality** (if future work switches `h_state_locality="population"`), the analogous rate-only knobs are `controller_tau_H_s` and `controller_tau_theta_s` (both divisors, fixed points `H*=-e/λ`, `Θ` integral of `H`, `λ` shifts fixed point so leave `λ`/`B` untouched). Required factor there: `1200/2.0 = 600×` (τ_Θ 2 s → 1200 s). This is separate from node path and must not be conflated with `tau_0_ms`.

### 4.3 Preservation proof

- **H fixed point**: algebraic zero of `dH` expression contains no `tau_i` factor → invariant. Empirical: same `hp_hash` scaffold, only `tau_0_ms` changed, final `H∞` identical (1.036) within fit error; earlier smoke-run `tau_0_ms=5→500` kept finite/bounded per `STABILITY_CRITERIA` (rates 1–80 Hz, V_m finite).
- **w fixed point**: co-scaled ratio invariant → `w*` invariant as derived above. Single-scaled alternative violates. Keeping `K` unchanged with only `tau_0` also preserves `w*` for any given `ΔH` — weight equilibrium conditioned on H is unchanged, only H's own timescale moves.
- **Bounds**: explicit constants `H_min, H_max, w_floor, w_ceiling, barrier_*` excluded from intervention — hard clips identical.

### 4.4 What must NOT be in `hdp_params` for this intervention

Do not include:
- `tau_theta_s` (dead key, ignored)
- `learning_rate` / `bounds_lo/hi` / `channels` (fabricated keys from `HDPConfig.to_jaxfne_hdp_params()` that JaxFNE does not consume)
- changes to `alpha, beta, gamma, delta, C_spike, barrier_c/d, H_boost_gain, K_ctrl, H_min/max, w_floor/ceiling`
- `h_state_dim, h_state_locality` (would switch controller topology)

Only `tau_0_ms` (and optionally paired `K_HDP`+`K_w_ctrl`) are admissible rate knobs.

---

## 5. Expected Phenotype Under Intervention

- **H**: mean drift currently Δ≈0.005 in 20 s then flat; under 200×, Δ≈0.0037 in 37 s (first 8 trials) but continuing linearly ~0.0003/trial, accumulating to ~0.027 by trial 260 (vs. current 0.004 extra). Pre/post H gap grows from ~0.0047 (`capture_pre_post_states` mini-lifecycle) to ~0.02–0.03 — larger Q8 `H_post→H_pre` effect, still within [0.1,10] bounds (H range contracts at larger τ, see std 0.024→0.0039, so bounds safety improves).
- **w**: currently Δ|w|≈−0.00011 (−0.88% by trial 5, saturated); under H-slow-only, w drifts monotonically +0.00017 per 6 trials and continues, yielding cumulative ~0.005–0.01 by end (vs. current −0.0001) — sign and magnitude controlled by ΔH statistics, now monotonic because H no longer overshoots.
- **Tracer**: exposure field/rate phenotype predicted to show gradual buildup across checkpoints (26 checkpoints, every 46.24 s) rather than step after checkpoint 1.

Quantitative preview via exponential model `H(t)=1.036-0.036·exp(-t/τ)`:

```
factor 1 (τ=3.9s):  H(1200)=1.036 (100% by 40s)
factor 100 (τ=416s): H(1200)=1.033 (94%)
factor 200 (τ=833s): H(1200)=1.027 (76%)
factor 300 (τ=1250s):H(1200)=1.022 (62%)
```

Coupled `K` slowdown would flatten w additionally but is not needed for H-driven phenotype.

---

## 6. Evidence Receipts

- **Code references**: `hdp.v1_pfc_aaab_hdp_params` source (`hdp_network.py:230`), `tau_i` cube law (`emitters.py:392`), `H_update` & `K_HDP/K_w_ctrl` weight step (`emitters.py:656-665`), `resolve_h_state_locality` default node (`_hdp_adaptive.py:53`), `population_restoring_derivatives` with τ defaults 0.2/2.0 (`_hdp_adaptive.py:171-172`), dead `tau_theta_s` audit via `grep` across `jaxfne`, `ContinuationState.dynamic` fields `v,u,prev_spikes,syn_state,H,w` (`_pipeline.py`).
- **Empirical runs** (seed 42, dt 0.5 ms, n_per_area 100, 4624 ms trials, `jtfne.simulate` with continuation — all logged above):
  - Factor 1 H fit τ=3.87 s (12 trials, curve_fit, se 0.25 s).
  - H std by factor: 1→0.0241, 10→0.0307, 50→0.0174, 100→0.0103, 200→0.0057, 300→0.0039 (monotonic contraction except 10× anomaly due to transient overshoot).
  - |w| saturated at 0.01285 (factor 1) vs. ramping 0.01276→0.01296 at factor 100 K unchanged.
  - Co-scaled K 100×: drift 16 ppm/6 trials — over-slows.
- **Schedule authority**: `canonical_schedule` gives 260 trials = 1202.24 s exposure (`t3_authorization.json:114`), so "1200 s" target is 260×4624 ms.
- **Authorization baseline**: `t3_authorization.json:68-82` H_Theta_timescales section explicitly claims `tau_H 0.6 s` and `tau_Theta via K_w_ctrl` seconds — hierarchy correct but exposure-scale claim absent; this design corrects it without revisiting T1–T7 (`NEVER-10`).

---

## 7. How to Apply (without breaking freeze)

This packet is **design only**. Implementation, when authorized, should:

1. Create a new `hp` via `base = hdp.v1_pfc_aaab_hdp_params(); hp = dict(base); hp["tau_0_ms"] = base["tau_0_ms"] * 200` (or chosen tier). Do not emit `tau_theta_s`, `learning_rate`, `bounds_lo/hi`.
2. Record new `hp_hash` alongside frozen `f327f9d2` for comparison; do not overwrite `t3_authorization.json`.
3. Pass via `RuntimeConfig(enable_hdp=True, hdp_params=hp)` as in `full_run.py:112` — no other code path.
4. Validate with a short chained run (e.g., 12 trials) that `H∞` unchanged and bounds not touched (check `H_final ∈ [0.1,10]`, `w_final ∈ [0.01,10]`, finite V_m per `stability.py`).
5. Full exposure would then be re-executed only under a new authorization packet.

No file was mutated for this design.

---

## 8. Risks & Non-Goals

- Large `tau_0_ms` (1000–1500 ms) yields E `tau_i` 125–187 s — slower than any inter-trial interval, so H integrates across many trials as intended, but transient response to a single deviant (AXAB) will be attenuated (smaller |H-1| per trial, see 0.032→0.0023 mean). Phenotype must be assessed across accumulation, not single-trial H amplitude.
- Do not compensate by raising `alpha/gamma/K_ctrl` — that would move fixed points and violate rate-only guarantee and potentially breach stability (verified stable envelope at K_ctrl=0.15, gamma=0.5).
- Node vs population locality confusion must be avoided: `tau_0_ms` does nothing for population Theta; `controller_tau_*` does nothing for node H. Mixing consumes dead keys.
- ×3 remains a valid **control** for a minimal perturbation experiment, but not for exposure-scale learning.

---

## 9. Decision

- **×3 is rejected** for the stated goal. It leaves τ≈12 s, saturating in 3% of exposure.
- **Recommended replacement**: `tau_0_ms: 5.0 → 1000.0` (200×), all other `hdp_params` frozen, bounds preserved, fixed points preserved. Tiers 100×/300× are admissible alternatives within 62–94% exposure attainment. If weight-specific slowing is independently required, co-scale `K_HDP` & `K_w_ctrl` by the same factor (e.g., both ÷200), but data suggests H-only is sufficient and preserves larger weight signal.
