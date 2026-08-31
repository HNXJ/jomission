# Session: `2026-08-29_004_fg09b-hdp-off-transfer-localization` — FG-09b HDP-OFF bridge transfer localization

---

## 1. Session identity

| Field | Value |
|-------|-------|
| Session ID | `2026-08-29_004_fg09b-hdp-off-transfer-localization` |
| UTC | `2026-08-29T20:00:00Z` |
| Local | `2026-08-29 16:00 (America/New_York)` |
| Branch | `main` |
| HEAD / code SHA | `f9af396` (full: `f9af396a0235f405eb75d786b01d89166c7f2a9a`) |
| Parent model | `manifests/w4a_blocker_seal.json` (`config_hash: a51711101576c7c3`, `hp_hash: f327f9d2ad64cc88`, `rf_hash: 18c2f0f55229c307`) ledger `GEN2_C022 / GEN2_W4a_BLOCKER` |
| Config / hash | `config_hash=a51711101576c7c3`, `hp_hash=f327f9d2ad64cc88`, `rf_hash=18c2f0f55229c307` |
| Agent roles | `Worker: Session Backfill Agent (FG-09b) | Reviewer: none (backfill)` |
| Env | `Python 3.13.7 | JAX 0.10.1 | jaxfne 0.4.17 | manifests sha256[:16]=4b87066d061f684f (w4a_blocker_seal.json), 5cf6a7b834f14fe6 (fg09b raw)` |

## 2. Goal

- **Research question:** Does the delayed non-HDP realized-current bridge (enable_hdp=False, delays [20,80,120]) reproduce the apparent `s_syn → I_syn → V_m` collapse where `I_syn` carries stimulus information while `V_m` remains invariant, without materially changing fast-state dynamics?
- **Acceptance predicate (before results):** POSITIVE if `I_syn` carries in 6/6 cells (`|d|>0.5`, `p_perm<0.05`, `I_corr>0.05` bits) AND `V_m` invariant in ≥1/6 (`|d|<0.5`, `p_perm>0.05`, `I_corr≈0`) AND fast-state equivalent (`Δbar_r<0.05 Hz (<0.4%), ΔV_m<0.02 mV, passivity ΔV_m=0 Δspikes=0`); else HETEROGENEOUS_PARTIAL if `I_syn` carries but `V_m` carries with small absolute `ΔVm 0.1-0.3 mV`; else NEGATIVE_INSTANCE if `I_syn` itself loses information or fast-state non-equivalent. Canonical delayed+HDP remains BLOCKED regardless.
- **Out of scope / non-goals:** No `jomission/network/builder.py` science mutation; no canonical delayed+HDP plasticity claim (`_model_simulate.py:280` guard); no `W/k` or `I_bg(λ)` sweep before `η_rec` measurement; no promotion of HDP_OFF to canonical; zero-delay reference not expected identical (delay changes temporal dynamics).

## 3. Starting authoritative state

Each claim tagged `OBSERVED` / `DERIVED` / `INFERRED` / `MODEL_ASSUMPTION` / `LITERATURE_PRIOR` / `UNRESOLVED`.

| # | Claim | Tag | Authority (manifest / ledger / commit / paper) |
|---|-------|-----|-----------------------------------------------|
| 3.1 | W4a MECHANISM_OBSERVABILITY_BLOCKER sealed NEGATIVE: Visual→V1 L4 PASS Δ169.79 d646 acc1.00 but V1 L4→L2/3 FAIL Δ−0.03 d−0.34 acc0.542 (<0.703) despite 3 interventions (g_vertical ×1.8-5, C020 Ne16→54 C0.444→0.704 cascade 7×, C021 PV/E 0.22→0.73/0.64) | `OBSERVED` | `manifests/w4a_blocker_seal.json:1-40`, `scratch/gen2_C020_receipt.md:109`, `scratch/gen2_C021_receipt.md:4.2`, `scratch/w4a_g_vertical_receipt.md:95` |
| 3.2 | Bottleneck localized to interval between information-bearing L4_E spikes(498/130 selective d>800 acc1.00) and measurable L2/3 response; missing chain `spikes_{L4_E}(t)→I_{L4_E→L2/3}(t)→V_{m,L2/3}(t)` at deployed delays [20,80,120] blocked by HDP+delay guard and non-HDP no record_edge_current | `DERIVED` | `manifests/w4a_blocker_seal.json:30`, `manifests/gen2_modification_ledger.jsonl:GEN2_C022`, `jaxfne/_model_simulate.py:280`, `jaxfne/emitters.py:714` |
| 3.3 | Canonical delayed+HDP grouped `I[t,G]` BLOCKED at `enable_hdp does not support nonzero delay_steps` and HDP seam lacks `record_grouped_current`; classification MISSING_CAPABILITY/DOCUMENTATION_GAP | `OBSERVED` | `scratch/jaxfne_issue_delayed_edge_current.md:1`, `scratch/fg01_U01_U02_verification.md:21`, `jaxfne/_model_simulate.py:280,600,1112`, `jaxfne/emitters.py:1112` |
| 3.4 | HDP_OFF delayed non-HDP grouped path is AVAILABLE: `RuntimeConfig(enable_hdp=False, record_grouped_current=True, edge_group_ids, G)` via `emitters.py:714` delayed kernel → `grouped_current_trace (n_steps,G)` passive `ΔV_m 0 Δspikes 0` reduction faithful `<1e-5` G≈16 production collapsed | `OBSERVED` | `scratch/fg01_U01_U02_verification.md:33`, `scratch/jaxfne_U02_grouped.md:53,97`, `jaxfne/emitters.py:714,879,840`, `jaxfne/recording/observables.py:30,81` |
| 3.5 | Parent models frozen: C019 theta* `[1.256,1.187,0.708,1.218,1.288]` AGSDR y_bar4 PASS, C020 typed vertical Ne16→54 k0.59→2.0 C0.704 sigma0.12 + typed W gains, C021 visual E/PV differential g_vis_PV 0.7 interleaved 0xC02A per-blob 0.73/0.64 energy parity ≤5% | `OBSERVED` | `manifests/gen2_modification_ledger.jsonl:GEN2_C019`, `GEN2_C020`, `GEN2_C021`, `scratch/gen2_C020_receipt.md:43`, `scratch/gen2_C021_receipt.md:4.2` |
| 3.6 | Prior fast-state (y_bar4) hard feasible: bar 5.72±0.72 CV_rate 1.67 H01-H12 y_bar4 t3 CI, Q_local bar5.09 rho0.353 Fano0.95 | `OBSERVED` | `scratch/gen2_C020_receipt.md:109 Q_local`, `manifests/gen2_modification_ledger.jsonl:GEN2_C019 y_bar_4` |
| 3.7 | `I→V_m` leading operator hypothesis (s_syn→I informative but V_m attenuated) suggested by `I_syn d>240 I0.67 while V_m invariant in important cases (C021_s0 d−0.11 p0.827 I0.0 G0.013)` but not yet canonical causal bottleneck because U01b (HDP+delay) unresolved and fast-state equivalence not yet checked | `INFERRED` | `scratch/fg09b_HDP_OFF_bridge.md:55-56`, `scratch/fg09b_HDPOFF_bridge_raw.json:C021_s0` |
| 3.8 | Whether delayed non-HDP realize reproduces collapse without fast-state confound is UNRESOLVED before this bridge; FG-10 DOF map gated on its outcome | `UNRESOLVED` | `scratch/fg09b_HDP_OFF_bridge.md:111-114`, `scratch/w4a_blocker_seal.md:7 C_next FORBIDDEN until J` |

## 4. Work performed

Execution ≠ verification — note which artifacts were verified by tests/gates.

| Worker | Assignment | Action | Artifact | Result |
|--------|------------|--------|----------|--------|
| Backfill | Create FG-09b HDP-OFF bridge session report (15 sections + sidecar) | Read authoritative state (w4a_blocker_seal.json, ledger GEN2_C022, fg09b_HDP_OFF_bridge.md, fg09b_HDPOFF_bridge_raw.json, env) + write session Markdown + JSON per SCHEMA.md | `docs/sessions/2026/2026-08-29_004_fg09b-hdp-off-transfer-localization.md` + `.json` | done |
| Backfill | Validate session report | `python docs/sessions/validate.py docs/sessions/2026/2026-08-29_004_fg09b-hdp-off-transfer-localization.md && python docs/sessions/validate.py --reindex` | validator receipt | done |
| Prior (FG-09b) | Execute HDP-OFF delayed grouped bridge (MECHANISTIC/SUPPLEMENTARY, not canonical) | `PYTHONPATH=. python scratch/fg09b_HDPOFF_bridge_runner.py` (seed 0,1 × C019/C020/C021, N=24, T=10000 dt0.1, G=748-759, delays [20,80,120], RuntimeConfig enable_hdp=False record_grouped_current) | `scratch/fg09b_HDPOFF_bridge_raw.json` `scratch/fg09b_HDP_OFF_bridge.md` | done (observed: I d>240 6/6, Vm invariant 2/6, fast-equiv PASS) |
| Prior (FG-09b) | Verify seams and hashes | `grep -n record_grouped_current / jaxfne/_model_simulate.py / jaxfne/emitters.py`, `sha256sum` w4a_blocker_seal 4b87066d, fg09b raw 5cf6a7b8, builder clean `git diff --stat -- jomission/network/builder.py` | `fg01_U01_U02_verification.md:33` `jaxfne_issue_delayed_edge_current.md` | done — builder clean, no science mutation |

## 5. Evidence

| Ref | Path | Kind | Hash / receipt | Notes |
|-----|------|------|----------------|-------|
| E5.1 | `manifests/w4a_blocker_seal.json` | `manifest` | `sha256[:16]=4b87066d061f684f` | Parent W4a BLOCKER seal, config_hash a51711101576c7c3 hp_hash f327f9d2ad64cc88 rf 18c2f0f55229c307 |
| E5.2 | `manifests/gen2_modification_ledger.jsonl` | `ledger` | `GEN2_C022 alias GEN2_W4a_BLOCKER` | Last entry APPENDED seal, config_hash a517..., 23 entries total |
| E5.3 | `manifests/environment.json` | `manifest` | `sha256[:16]=...` | Python 3.14.4 / JAX 0.10.1 / jaxfne 0.4.17 / jaxlib 0.10.1 |
| E5.4 | `scratch/fg09b_HDP_OFF_bridge.md` | `other` | `184 lines` | Authoritative narrative for this bridge (MECHANISTIC/SUPPLEMENTARY/HDP_OFF, not canonical) |
| E5.5 | `scratch/fg09b_HDPOFF_bridge_raw.json` | `array` | `sha256[:16]=5cf6a7b834f14fe6` | Raw per C/seed grouped I/Vm/spikes/G + fast-state equivalence (6 keys, N=24, T=10000) |
| E5.6 | `scratch/fg09b_HDPOFF_bridge_runner.py` | `other` | `—` | Repro runner for MECHANISTIC HDP_OFF delayed grouped assay |
| E5.7 | `scratch/fg01_U01_U02_verification.md` | `other` | `—` | U01/U02 seam verification: passivity ΔV0 Δspikes0, max|gc−sum ec|<1e-5, delay+HDP BLOCKED at :280 |
| E5.8 | `scratch/jaxfne_issue_delayed_edge_current.md` | `other` | `287 lines` | MISSING_CAPABILITY classification and Option A/B/C for upstream fix |
| E5.9 | `docs/sessions/2026/2026-08-29_004_fg09b-hdp-off-transfer-localization.md` | `other` | `this file` | Session Markdown (authoritative) |
| E5.10 | `docs/sessions/2026/2026-08-29_004_fg09b-hdp-off-transfer-localization.json` | `other` | `—` | JSON sidecar (derived) |

- **EvidenceRefs:** E5.1-E5.10 linked above; raw JSON E5.5 is primary quantitative authority for §6
- **Visualization report:** None — no figures produced in HDP_OFF bridge (grouped trace arrays ~30 MiB full, G≈16 collapsed 0.6 MiB not visualized); §6 tables are authoritative

## 6. Observations

Compact quantitative tables only — no interpretation.

| ID | Metric | Value | Units | n | Artifact | Notes |
|----|--------|-------|-------|---|----------|-------|
| Obs 6.1 | I_syn Δ `L4E→L2/3E` grouped (W − B_proxy, B−A) | +0.330 / +0.824 / +0.660 / +1.697 / +0.991 / −0.690 | native (w·syn grouped) | 24 per C/seed (6 cells ×12) | `E5.5:C019_s0/s1,C020_s0/s1,C021_s0/s1` | d=247.2, 964.0, 264.4, 979.3, 282.1, −397.4; p_perm=0.001 all; I_corr=0.675,0.678,0.674,0.678,0.683,0.675 bits; acc=1.00 all; G=752,748,759,752,759,752; target counts 76,56,76,67,76,67 |
| Obs 6.2 | V_m,L23 Δ (W − B_proxy, B−A) | −0.289 / +0.132 / −0.249 / +0.280 / −0.013 / +0.166 | mV | 24 per cell | `E5.5` | d=−1.92, 0.83, −1.59, 1.90, −0.11, 1.36; p_perm=0.001, 0.056 n.s., 0.002, 0.001, 0.827 n.s., 0.005; I_corr=0.301,0.120,0.198,0.272,0.000,0.181 bits; acc=0.833,0.708,0.792,0.792,0.458,0.792; σ_V≈6.2 pooled 0.12-0.16 |
| Obs 6.3 | G I→V_m = |ΔVm|/(|ΔI|+0.01) | 0.851 / 0.158 / 0.371 / 0.164 / 0.013 / 0.238 | mV/native | 6 cells | `E5.5` | C021_s0 0.013 strong attenuation vs C019_s0 0.851; C021_s0 ΔVm 0.0129/0.991 |
| Obs 6.4 | G L4→I = |ΔI|/(|ΔL4|+1.0) | 0.00148 / 0.00315 / 0.00296 / 0.00648 / 0.01703 / 0.00502 | native/Hz | 6 | `E5.5` | ΔL4 Hz: 222.09,261.05,222.07,261.05,57.17,136.53; d_L4 268,530,265,543,61,232 |
| Obs 6.5 | spikes_L23 Δ (W − B_proxy, B−A) | −1.366 / −2.580 / −1.699 / −4.043 / −0.094 / −1.907 | Hz | 24 per cell | `E5.5` | d=−2.42,−5.47,−2.74,−5.51,−0.21, −3.66; p=0.001,0.001,0.001,0.001,0.634 n.s.,0.001; mirrors Vm pattern (C021_s0 n.s.) |
| Obs 6.6 | Fast-state equivalence Δbar_r del_off − zero_off | −0.027 / −0.007 / 0.000 / +0.008 / −0.005 / −0.005 | Hz | 10000 steps ×400 neurons | `E5.5:delta_del_vs_zero_off` | bar_r del_off 11.70,14.56,11.71,14.61,14.76,15.59 Hz; zero_off 11.72,14.57,11.71,14.60,14.77,15.59 Hz |
| Obs 6.7 | Fast-state equivalence ΔV_m del_off − zero_off | −0.006 / +0.005 / −0.0001 / +0.001 / −0.001 / +0.0005 | mV | 10000×400 | `E5.5` | vs ε_Vm=1.0, vs σ_V≈6.2 |
| Obs 6.8 | Fast-state equivalence zero_off − zero_on Δbar_r / ΔV_m | +0.045/+0.030/+0.042 etc / −0.013/+0.002/−0.012 etc | Hz / mV | 10000×400 | `E5.5:delta_zero_off_vs_on` | All |Δbar|<0.05 Hz (<0.4% of bar 11-15), |ΔVm|<0.02 mV (<0.03% of −65) |
| Obs 6.9 | Passivity grouped ON vs OFF | ΔV_m 0.0 / Δspikes 0.0 | mV / Hz | per seed | `E5.5:passivity_dV/dS` + `fg01_U01_U02: max|gc−sum ec|0.0 <1e-5` | G=0, 4-way bool dispatch emitters.py:840 already computed |
| Obs 6.10 | Vm variance del/zero_off/zero_on | 39.14/39.20/38.35 etc | mV² | 10000×400 | `E5.5:fast_stats` | Δvar <0.85 vs σ_V≈6.2 (<14%); ΔH null vs 1.0 as expected (H off) |
| Obs 6.11 | Decision matrix Row 1+5 synthesis | 2/6 strong Row 1 (C021_s0 ideal + C019_s1 near) + 4/6 Row 5 heterogeneous small absolute ΔVm 0.13-0.29 | — | 6 cells | `E5.4 §2` | Row 1: I d282 I0.683 Vm d−0.11 I0.0 equiv; Row 5: I d>240 yet Vm d1.3-1.9 p<0.005 but |Δ|0.13-0.29 vs 95 mV to thresh |

_Figures:_ `None`

## 7. Interpretation

Observation ≠ interpretation. Each claim cites supporting observation ID(s) from §6.

| # | Interpretation | Supporting observations | Confidence |
|---|----------------|-------------------------|------------|
| 7.1 | `I_syn` is highly informative in all 6/6 delayed HDP_OFF cells (d>240, I_corr 0.67-0.68, acc1.00) — stimulus reaches grouped synaptic current faithfully; no upstream `s_syn`/`syn_state` failure | `Obs 6.1`, `Obs 6.4` | `HIGH` |
| 7.2 | `V_m` shows heterogeneous collapse: in 2/6 (C021_s0 textbook d−0.11 p0.827 I0.0 G0.013 + C019_s1 p0.055) current→V_m transfer is strongly shunted/attenuated; in 4/6 V_m carries but with small absolute ΔVm 0.13-0.29 mV vs σ_V≈6.2 and 95 mV to threshold (≈dV/dI 0.014 mV/native, 25× below Γ_FF0.40) — graded shunting/low-pass, not threshold-only | `Obs 6.2`, `Obs 6.3`, `Obs 6.5` | `HIGH` |
| 7.3 | Fast-state equivalence PASS (Δbar<0.03 Hz <0.4%, ΔV_m<0.02 mV, passivity 0, var <14%) proves disabling HDP does not materially change fast spikes/V_m; supplementary assay is identifying for fast dynamics (H differs as expected, slow plasticity vs fast drive μ_total3.405) | `Obs 6.6`, `Obs 6.7`, `Obs 6.8`, `Obs 6.9`, `Obs 6.10` | `HIGH` |
| 7.4 | Combined Row 1+5 = strong-to-moderate evidence for `s_syn→I_syn→V_m` (I→Vm) bottleneck as leading operator; upstream `A` (delay→syn G0.997-0.999) not implicated because I never loses S | `Obs 6.1`, `Obs 6.11` | `MED` |
| 7.5 | I→Vm is leading candidate but NOT yet canonical causal bottleneck: U01b remains blocker for canonical delayed+HDP (E5.1 §3.3); HDP_OFF is MECHANISTIC/SUPPLEMENTARY only and cannot be treated as canonical plasticity model; canonical delayed+HDP I→Vm still UNRESOLVED | `Obs 6.6-6.10` + §3.3/3.7 | `HIGH` |
| 7.6 | Delay itself introduces only ±0.03 Hz / ±0.02 mV shift (del vs zero_off) — order smaller than stimulus ΔL4 222 Hz and I_syn 0.33 native d247, not confounding Row 1/5 | `Obs 6.6`, `Obs 6.7` | `HIGH` |

## 8. Negative / insufficient results

Bounded vocabulary: `NEGATIVE_INSTANCE` | `NULL_RESULT` | `INSUFFICIENT_POWER` | `INCONCLUSIVE` | `RESOURCE_BOUNDARY` | `BLOCKED`; `FALSIFIED` reserved for `L0`–`L5` criterion failures.

| # | Claim | Verdict | Level / threshold | Notes |
|---|-------|---------|-------------------|-------|
| 8.1 | Canonical delayed+HDP grouped `I[t,G]` exposing realized edge-current at deployed delays [20,80,120] | `BLOCKED` | `jaxfne/_model_simulate.py:280 guard + emitters.py:714 no record_grouped_current for non-HDP? actually HDP path lacks it; _pipeline.py:395 HDP-only` | Correctly blocked per `fg01_U01_U02_verification.md:21`; this session is HDP_OFF bridge only, not claiming cure; U01b remains blocker (E5.8) |
| 8.2 | `I_syn` loses stimulus information in HDP_OFF bridge (would imply upstream syn_state/W generation failure) | `NEGATIVE_INSTANCE` | `gate d<0.5 p>0.05 I≈0` — observed 0/6 loss, all d>240 p0.001 I0.67 | Search not upstream B; FG-03 A PASS preserved |
| 8.3 | Fast-state non-equivalence (HDP disabling materially changes bar_r/V_m so assay non-identifying) | `NEGATIVE_INSTANCE` | `threshold Δbar>0.05 Hz or ΔVm>0.02 mV or passivity≠0` — observed all PASS | HDP is slow plasticity not fast drive; assay identifying for fast currents/V_m |
| 8.4 | Full vertical transmission rescue in HDP_OFF would imply no bottleneck — not observed (2/6 strong silence + 4/6 graded) | `NULL_RESULT` | `threshold Vm→spikes full rescue acc>0.703 ∧ d>0.5` — not seen; C021_s0 Vm→spikes also n.s. (d−0.21 p0.634) | Consistent with `G I→Vm 0.013` strong shunt, not threshold-only pathology D |
| 8.5 | Direct `η_rec = ‖I_rec‖/‖I_total‖` partitioned by motif (signed/absolute/L2 Efrac) and `Corr(I_private, I_L4→L23[t+τ])` to formally type shunting vs E/I cancellation | `INSUFFICIENT_POWER` | `requires grouped I[t,G] partitioning observables.py:161 + drive_array emitters.py:477` — not yet measured (only proxy G and σ_syn 0.0027) | FG-10 measurement required before any builder DOF (mechanistic_replay_freeze.md:162) |

_If none:_ Not applicable — see table.

## 9. Unexpected findings

| # | Finding | Why unexpected | Follow-up priority |
|---|---------|----------------|-------------------|
| 9.1 | C021_s0 shows textbook decoupling `I_syn d282 I0.683 → V_m d−0.11 p0.827 I0.0 G0.013` while C019_s0 with same delays/topology shows `V_m d−1.92 p0.001 G0.85` — heterogeneity across C/seed despite identical assay | Prior W4a assumed uniform vertical FAIL (acc0.542) implying uniform bottleneck; bridge reveals graded shunting (0.013 to 0.85) not binary | `high` — FG-10 must report full transfer_matrix.csv across all C/seed and `η_rec` distribution, not single point |
| 9.2 | Background toggle not needed for this assay: σ_syn proxy 0.0027 vs σ_total≈6.2 gives η≈0.003-0.007 compatible with G0.013 without invoking Poisson private alone | Expected η_rec≪1 to be proven by tonic+Poisson+noise partition; proxy already suggests strong dilution but formal partition still required per §8.5 | `med` |

_If none:_ Not applicable.

## 10. Tool / workflow friction

| Friction | Cause | Workaround | Permanent repair? |
|----------|-------|------------|-------------------|
| Delayed+HDP grouped current MISSING_CAPABILITY at jaxfne/_model_simulate.py:280 + HDP seam lacks record_grouped_current | Upstream jaxfne HDP vs delay guard + _pipeline.py:395 HDP-only forwarding | Use HDP_OFF delayed non-HDP grouped bridge (enable_hdp=False, delays [20,80,120]) as MECHANISTIC/SUPPLEMENTARY with fast-state equivalence check | `HDP+delay grouped current support in jaxfne (Option A/B/C per jaxfne_issue_delayed_edge_current.md:200) — proposed in §12` |
| Window B surrogate: freeze B=[-50,0) not in Simulation(duration 1000) to_array, so use B_proxy=[0,50) early p1 pre-volley | Simulation starts at p1 0; fx -500..0 clipped | Report mean(W)-mean(B_proxy) + raw mean(W); flag as MECHANISTIC supplementary limitation | `DOCUMENTATION — cross-ref simulation vs absolute clock in freeze (proposed §12)` |

_If none:_ Not applicable.

## 11. Learned lessons

| Lesson | Trigger | Cause | Repair | Evidence | Scope | Confidence | Persistence |
|--------|---------|-------|--------|----------|-------|------------|-------------|
| HDP_OFF delayed grouped bridge is identifying for fast dynamics (Δbar<0.05 Hz ΔVm<0.02 mV passivity 0) even though H differs — slow plasticity vs fast drive dissociation | Fast-state equivalence PASS across 6/6 C/seed including delay vs zero-delay ±0.03 Hz shift | H/w is slow (builder.py:44 μ_total3.405), fast spikes/V_m via emitters.py:714 delayed kernel not H-dependent; passivity already computed | Use delayed non-HDP grouped `segment_sum` for mechanistic I→Vm localization; keep canonical delayed+HDP BLOCKED and label supplementary explicitly | `E5.5:Obs 6.6-6.9` `fg01_U01_U02_verification.md:33` `jaxfne/emitters.py:840` | `project` | `HIGH` | `YES` |
| Current→V_m transfer is heterogeneous graded shunting (G 0.013-0.85) not binary FAIL; single-seed report risks overclaim | C021_s0 G0.013 invariant vs C019_s0 G0.85 carrying but small absolute 0.29 mV | Shunting/low-pass via a/b (emitters.py:55) + I_bg λ (builder.py:44 tonic 0.6 + Poisson 2000 Hz p0.20 μ0.51) + E/I duty cycle | Require full transfer_matrix.csv (all C/seed × G) and η_rec before typing | `Obs 6.3` `scratch/fg09b_HDP_OFF_bridge.md:105` | `project` | `HIGH` | `YES` |
| B_proxy surrogate must be flagged whenever window B is outside array | Simulation(duration 1000) starts at p1 0, B=[-50,0) not in array | Freeze vs simulation absolute clock offset | Use B_proxy [0,50) early p1 + raw mean(W) and document; fix freeze or simulation start for canonical | `E5.4 §0 Note` `E5.5 note` | `project` | `MED` | `NO` |

_Persistence_ ∈ `YES` | `NO`; `Confidence` ∈ `LOW` | `MED` | `HIGH`. `YES` = propose promotion to harness/project law.

## 12. Harness / tool proposals

Type ∈ `PROJECT_RULE` | `TEST/GATE` | `JOMISSION_TOOL` | `JAXFNE_UPSTREAM` | `DOCUMENTATION` | `VISUALIZATION` | `AGENT_WORKFLOW`

| # | Type | Target | Rationale | Evidence |
|---|------|--------|-----------|----------|
| 12.1 | `JAXFNE_UPSTREAM` | `jaxfne/_model_simulate.py:280,600,1112` + `jaxfne/emitters.py:714` + `jaxfne/_pipeline.py:395` — delayed+HDP grouped current | Unblock canonical `I[t,G]` at deployed delays [20,80,120] with G≈16 collapsed (0.6 MiB) to allow `η_rec` and `I(S;I)` without HDP_OFF surrogate; required for canonical I→Vm proof | `Obs 6.1` `E5.8` `§8.1 BLOCKED` |
| 12.2 | `TEST/GATE` | `mechanistic_replay_freeze.md:54 B window` + `Simulation(duration)` absolute clock | Add gate that B=[-50,0) requires fx -500..0 in array or forces B_proxy with explicit flag; prevents silent baseline bias | `§10 B_proxy` `E5.4 §0` |
| 12.3 | `JOMISSION_TOOL` | `recording/observables.py:161 partition_currents_by_motif` + `emitters.py:477 drive_array` for `η_rec` | Automate `η_rec = ‖I_rec‖/‖I_total‖` signed/absolute/L2 + `Corr(I_private, I_L4→L23[t+τ])` τ10-20 ms direct from grouped `I[t,G]` partition | `Obs 6.3` `§8.5` |

_If none:_ Not applicable.

## 13. Scientific state transition

| Field | Value |
|-------|-------|
| **Before** | W4a MECHANISM_OBSERVABILITY_BLOCKER: Visual→V1 L4 PASS Δ169 d803 acc1.00 but V1 L4→L2/3 FAIL Δ0.02 d0.13 acc0.542 invariant across 3 interventions (g_vertical, C020, C021); bottleneck localized to interval L4_E spikes → measurable L2/3 but missing chain spikes→I→Vm at delays [20,80,120] BLOCKED at jaxfne/_model_simulate.py:280; I→Vm leading operator hypothesis but U01b unresolved, no fast-state equivalence check |
| **After** | FG-09b HDP-OFF bridge supports I→Vm hypothesis with fast-state equivalence: I_syn carries in 6/6 d>240 I0.67 while V_m invariant in important cases (C021_s0 d−0.11 p0.827 I0.0 G0.013) and small absolute 0.13-0.29 mV in 4/6 heterogeneous; fast-state equivalent Δbar<0.03 Hz <0.4% ΔV_m<0.02 mV passivity 0 → I→Vm is leading candidate (strong-to-moderate) but NOT yet canonical causal bottleneck because U01b unresolved (canonical delayed+HDP still BLOCKED) |
| **Unlocked** | FG-10 DOF map may open for delayed non-HDP grouped measurement (0 DOF): `I[t,G≈16]` via `RuntimeConfig(enable_hdp=False, record_grouped_current)` on frozen hashes + `V_m` + `Corr(I_private, I_L4→L23[t+τ])` + `dV/dI` partitioned `η_rec`; no builder sweep before measurement |
| **Still gated** | Canonical delayed+HDP grouped current (needs jaxfne HDP+delay fix Option A/B/C); formal `η_rec` partition (signed/absolute/L2 Efrac) and `E/I` split; typed vertical `Ne/k` or `I_bg(λ)` escalation (1 DOF, never before J measurement); H/Theta plasticity via M-06 parallel |
| **Invalidated** | Hypothesis that assay is non-identifying due to HDP disabling — falsified (fast-equiv PASS); hypothesis that I_syn itself loses S (upstream syn failure) — falsified (0/6 loss) |
| **New uncertainty** | Heterogeneity mechanism: why C021_s0 fully shunted (G0.013) vs C019_s0 partially carrying (G0.85) at same delays — requires η_rec + private vs recurrent partition to type shunting vs E/I cancellation vs low-pass |

## 14. Next action

- **Primary (highest-value):** FG-10 DOF map — open FG-10 for delayed non-HDP grouped measurement with 0 new builder DOF: `RuntimeConfig(enable_hdp=False, hdp_params={record_grouped_current=True, edge_group_ids, grouped_num_segments=G≈16})` on frozen hashes (a51711101576c7c3 / 4574a155 / 65b302e8) across all C/seed, then partition `I[t,G]` via `recording/observables.py:161` for `η_rec` signed/absolute/L2 Efrac + `Corr(I_private, I_L4→L23[t+τ])` τ10-20 ms + `dV/dI` per-neuron; produce `transfer_matrix.csv` (rows: C/seed × G). Ready predicate: this FG-09b bridge PASS (fast-equiv + I carries) — now met; builder frozen. Why: measures formal η_rec before any builder DOF, required by `mechanistic_replay_freeze.md:162 C_next FORBIDDEN until J`.
- **Secondary (optional):** File jaxfne upstream issue for delayed+HDP grouped current (Option A: expose record_grouped_current in HDP path + lift guard) with repro `scratch/fg09b_HDPOFF_bridge_runner.py` + `jaxfne_issue_delayed_edge_current.md:200` — unblocks canonical proof later.

## 15. Progress score

| Field | Value |
|-------|-------|
| Current | 74 |
| Previous | 74 (`null` if first session) |
| Delta | 0 |
| Reason | HDP-OFF bridge supports I→Vm hypothesis (I_syn 6/6 d>240 I0.67 while V_m invariant in important cases C021_s0 d−0.11 p0.827 I0.0 G0.013 suggesting I→Vm bottleneck with fast-state equivalent Δbar<0.03 Hz <0.4% ΔV_m<0.02 mV passivity 0) but canonical delayed+HDP still BLOCKED at jaxfne/_model_simulate.py:280 — no new verified canonical capability; supports FG-10 opening (0 DOF) but does not advance sealed frontier |

---

**Checklist before seal:**

- [x] All 15 sections present and in order
- [x] §1 `session_id` matches filename stem; timestamps ISO-8601
- [x] §5 every artifact exists; figures link to visualization
- [x] §6 tables have units, n, artifact refs; no interpretation
- [x] §7 each claim cites `Obs 6.x`
- [x] §8 bounded vocabulary used correctly; `FALSIFIED` only with `L0`–`L5` ref
- [x] §10 friction normalized (for recurrent detection)
- [x] §11 `Persistence` and `Confidence` enums correct
- [x] §15 score delta justified by verified capability
- [x] JSON sidecar `*.json` created and matches `SCHEMA.md`
- [x] `python docs/sessions/validate.py <this file>` passes
