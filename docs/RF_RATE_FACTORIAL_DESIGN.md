# 2×2 Factorial Plasticity-Rate × RF — Frozen Design

**Version:** `rf_rate_factorial.v0.1.0`  
**Status:** FROZEN — not tuned to results. Any expansion requires new version.  
**Module:** `jomission.ablations.rf_rate_factorial`  
**Manifest:** `manifests/rf_rate_factorial_design.json`  
**Tests:** `tests/test_rf_rate_factorial.py` (14 tests, all artifact-backed)  
**Audit source:** `docs/PLASTICITY_RATE_INTERVENTION_DESIGN.md`  
**Engine:** `jaxfne 0.4.17`, `config_hash 4f9fdeae7428199a`, `hp_hash canonical f327f9d2ad64cc88`  
**Date:** 2026-08-24

---

## 1. Factors

| Factor | Levels | Operational param | Values | Fixed point | Bounds |
|--------|--------|-----------------|--------|-------------|--------|
| **RF** (receptive-field plasticity) | off / on | `K_HDP` (w update gain) `j|φ(ΔH)|·m + K_w_ctrl(m0−m)` | off 0.0, on 0.003 (canonical) | RF moves `w*` (qualitative toggle, intentional) | `w∈[0.01,10]`, `H∈[0.1,10]` frozen |
| **Rate** (plasticity rate) | standard / slow | `tau_0_ms` (cube-law `τ_i=τ_0·size³`, divisor of `dH/dt`) | std 5.0 (τ_eff,E 4.1 s, saturates 12 s), slow 1000.0 (200×, τ_eff,E 833 s, 76% at 1200 s) | **Rate preserves fixed points** (τ divides `dH/dt` only, `dH=0` unchanged) and bounds | same |

Both use `enable_hdp=True` (H alive). RF controls **whether** `w` can follow H; Rate controls **how fast** H evolves. Orthogonal.

Rejected: ×3 (15 ms → τ_eff 12.3 s, 3% of exposure) per audit — off by 65–100×.

### Cells

| Cell | RF | Rate | `K_HDP` | `tau_0_ms` | `hp_hash` | `config_hash` |
|------|----|------|---------|------------|-----------|---------------|
| A | off | standard | 0.0 | 5.0 | `bb8277e7a8e0bca2` | `4f9fdeae7428199a` |
| B | off | slow | 0.0 | 1000.0 | `f72a489841810a4b` | same |
| C | on | standard | 0.003 | 5.0 | `f327f9d2ad64cc88` (canonical) | same |
| D | on | slow | 0.003 | 1000.0 | `b326f7201c59b803` | same |

All other `hdp_params` (`K_w_ctrl=0.001`, `K_ctrl=0.15`, `α=0.05`, `γ=0.5`, `H_min/max`, `w_floor/ceiling`, `barrier_*` etc.) frozen across cells. Network: 400 neurons (100/area × V1,V4,FEF,PFC), layers L1/L2/3/L4/L5/L6, cell types E/PV/SST/VIP, FF `p=0.30`, FB `p=0.20`, `dt 0.1` canonical, emitter `izhikevich`, 16 contacts.

---

## 2. Schedule & Paradigm

Canonical schedule `canonical_schedule(trial_ms=4624, dt=0.1)`:

- initialization 2 s, baseline 10 s, **exposure 260 trials =1202.24 s (≥1000 s)**, testing 96 trials =443.904 s (12 conditions ×8 reps), recovery 30 s, total 1688.144 s, 26 checkpoints every 10 trials (=46.24 s).

Paradigm `JOMISSION_PARADIGM` exact: 12 conditions (AAAB,AXAB,AAXB,AAAX,BBBA,BXBA,BBXA,BBBX,RRRR,RXRR,RRXR,RRRX), omission preserves temporal geometry (drive zeroed in 531 ms slot, timing identical), windows reindexed at expected onset `[-1000,+1000]`, baseline `[-250,-50]`, slot `[0,531]`, post `[531,1000]`. Do not pool `p2/p3/p4` before per-position test.

---

## 3. Estimands

Primary `Y = Δ_exposure = Y_omission^{post} − Y_omission^{pre}` per replicate per cell, within-replicate paired (same seed, same condition order, same RNG, same trial battery 96). Phenotypes (each separate family):

- `H_mean`, `w_mean` (ContinuationState)
- `rate_omission_effect` global & per-area (V1,PFC) omission vs intact in `[0,531]`
- `field_low_gamma` log ratio omission vs intact per area & frontal−V1 contrast (proxy_readout, 30-50 Hz, `area_local`)
- `field_broadband` 5 bands ×4 areas ×3 positions (secondary, FDR)
- `rate_intrinsic` sanity

Field is `proxy_readout` via `field_by_area_4d` linear partition, `physical_amplitude_calibrated=False`, same kernel across cells.

---

## 4. Statistics (frozen)

Model per phenotype, `n=8` seeds (FULL) or `n=4` (PILOT), balanced:

```
Y_{ijk} = μ + α·RF_i + β·Rate_j + γ·RF_i·Rate_j + ε_{ijk}
RF_i∈{0 off,1 on}, Rate_j∈{0 std,1 slow}, k=1..n, ε~N(0,σ²)
```

Contrasts (marginal means, orthogonal):

- **RF main:** `(C+D)/2 − (A+B)/2`  (α+γ/2)
- **Rate main:** `(B+D)/2 − (A+C)/2`  (β+γ/2)
- **Interaction:** `(D−C)−(B−A)`  (γ)
- Simple: `Rate|RFon=D−C`, `Rate|RFoff=B−A`, `RF|slow=D−B`, `RF|std=C−A`

Table: 2-way ANOVA (Type III marginal, balanced → II) with `F`, `p`, `partial η²`, `SS = n·contrast²` (×1 for mains, ÷4 for interaction), `MS_err = SS_resid/df_resid`, `df_resid=4n−4`. Fallback to permutation `F` (10k) if Shapiro or Levene fails. α=0.05 per contrast, `t = contrast/SE`, `SE = √(var_pooled·sum c² / n)`, `var_pooled=SS_resid/df_resid`, `95% CI = contrast ± t_{.975,df}·SE`, `Cohen d = contrast / SD_pooled` (interaction scaled ÷2).

Multiplicity: primary families **no correction** across families (each separate claim); exploratory 5×4×3 T4 grid **FDR BH** per position, secondary.

Thresholds (descriptive, not inferential; from Q8):

- rate `|Δ|>0.5 Hz` + `|d|>0.2` + `p<.05` → POSITIVE
- field `|log ratio|>0.1` + `|d|>0.2` + `p<.05` → POSITIVE
- else `p≥.05` + below thr → NEGATIVE else UNRESOLVED (underpowered)

### Distinguishability

| Pattern | RF p | Rate p | Inter p | Inference |
|---------|------|--------|---------|-----------|
| RF only | <.05 | NS | NS | RF drives Y, timescale irrelevant |
| Rate only | NS | <.05 | NS | Timescale drives Y, RF irrelevant |
| Additive | <.05 | <.05 | NS | Independent, no synergy |
| Interaction | * | * | <.05 | Non-additive — examine simples: e.g., `Rate\|RFon` sig but `Rate\|RFoff` NS ⇒ Rate needs RF to matter |

Power: with `n=8`, `SD≈1` (pilot), `α=.05`, detectable `d≈1.0` (∼1 Hz rate or 0.7 log units) at 80% power; `n=4` pilot detectable `d≈1.5` (flag UNRESOLVED if `|d|<0.2` and `p NS` and `n<8`).

Diagnostics: Shapiro-Wilk on residuals, Levene across cells, QQ; no outlier removal (>|3SD| flagged but kept; sensitivity re-run separate).

---

## 5. Completion Predicate (exact, artifact-backed)

`COMPLETE` iff all `C1–C7` hold. Negative (all `p NS`) is still `COMPLETE` if `C1–C7` hold — failures are evidence about model.

**C1 Hashes:** `config_hash==4f9f...` for all cells; `hp_hash` per cell matches table; design version `rf_rate_factorial.v0.1.0`; manifest `sha256[:16]` verifies no post-freeze mutation.

**C2 Schedule:** exposure 260 trials, testing 96 trials (12×8), `dt==0.1` FULL (or `1.0` PILOT declared), `n_seeds≥8` FULL (`≥4` PILOT, same seeds across cells), 26 checkpoints.

**C3 Testing:** 96 pre + 96 post trials per cell, same conditions, field recorded `trial_A_C_T` 16 contacts, 4 areas, spikes/V_m finite, rates `[1,80]` Hz, `H∈[0.1,10]`, `w∈[0.01,10]`, `V_m mean∈[-90,-50]`.

**C4 Continuation:** checkpoint/restart equivalence at least one cell `V_m rtol 1e-5 atol 1e-4 spikes exact`, RNG preserved within-replicate.

**C5 Artifacts:** per-replicate `Δ` arrays `npz+json` with denominators explicit, provenance `proxy_readout / physical_amplitude_calibrated=False`, owner `generated`, under `results/rf_rate_factorial/{cell}/`, manifests sealed.

**C6 Stats:** 2-way ANOVA as above, pooling rule preserved (`DO NOT pool p2/p3/p4` until per-position), FDR only for exploratory grid.

**C7 Frozen:** no mutation of frozen evidence, hashes verified via `validate_design()` and `check_completion()`.

PILOT vs FULL: pilot `dt1.0 n4` completeness does **not** imply FULL; promotion requires re-execution at `dt0.1 n8`.

---

## 6. Not tuned to results

All `α`, thresholds, bands `[4,8,14,30,50,80]`, areas `V1,V4,FEF,PFC`, windows `[-1000,+1000]/[-250,-50]/[0,531]/[531,1000]`, contrast definitions, and predicate predicates are frozen **before** execution. No p-hacking, no post hoc band/area selection, no redefining slow factor after seeing saturation. If variance unexpected, design unchanged — variance reported, next version required.

---

## 7. Implementation

```python
from jomission.ablations.rf_rate_factorial import factorial_cells, validate_design, check_completion, anova_rf_rate, FactorialANOVAInput
cells = factorial_cells(dt_ms=0.1)  # 4 cells with hp
validate_design()  # hash checks
check_completion(..., predicate="FULL")  # C1–C7

# After collecting Y per replicate per cell:
inp = FactorialANOVAInput(data={"A_RFoff_RateStd": A, ...}, seeds=[0..7], phenotype="rate_omission_effect")
res = anova_rf_rate(inp)  # anova_table, contrasts, diagnostics, interpretation
```

See module docstring for full API.

---

## 8. Evidence Receipts

- `validate_design()` — 14 tests in `tests/test_rf_rate_factorial.py` prove orthogonality, rate-only preservation, hash backing, completion predicate, distinguishability of RF-only/Rate-only/Interaction/Additive patterns, `F=t²` consistency, pooling rule, manifest hash.
- Audit `tau_0_ms` rate-only proof via `docs/PLASTICITY_RATE_INTERVENTION_DESIGN.md` (H fixed point algebra, empirical τ=3.87 s vs 833 s at 200×).
- `hp_hash` computed `sha256(json.dumps(hp, sort_keys=True))[:16]` — matches `jaxfne.hdp_network.v1_pfc_aaab_hdp_params` + overrides.
- Paradigm exact gate `paradigm_exact_gate()` preserved.
- Field `area_local` provenance (tests 10/10) and `t4_t5_analysis` frozen bands.

Seal: `manifests/rf_rate_factorial_design.json` sha `0c2972525d1d0496`.
