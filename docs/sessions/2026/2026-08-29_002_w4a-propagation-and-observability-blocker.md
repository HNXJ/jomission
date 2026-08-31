# Session: `2026-08-29_002_w4a-propagation-and-observability-blocker` — W4a Vertical Propagation and Mechanism Observability Blocker

---

## 1. Session identity

| Field | Value |
|-------|-------|
| Session ID | `2026-08-29_002_w4a-propagation-and-observability-blocker` |
| UTC | `2026-08-29T12:00:00Z` |
| Local | `2026-08-29 07:00 (CDT, UTC-05:00)` |
| Branch | `main` |
| HEAD / code SHA | `f9af396a0235f405eb75d786b01d89166c7f2a9a` (short: `f9af396`) |
| Parent model | `manifests/w4a_blocker_seal.json` (`config_hash: a51711101576c7c3`, `hp_hash: f327f9d2ad64cc88`, `rf_hash: 18c2f0f55229c307`, `dt_ms: 0.1`) |
| Config / hash | `config_hash=a51711101576c7c3`, `hp_hash=f327f9d2ad64cc88`, `rf_hash=18c2f0f55229c307` |
| Agent roles | `Worker: W4a propagation agent (g_vertical) | Worker: C020 typed vertical agent | Worker: C021 visual grammar agent | Worker: V0 foundation V1-V4 agents (ontology/manifest/network/run-report/adversary) | Reviewer: W4 P1/P2/P3 + W4a3 P1/D + W4a2 T1/T2/T3 consensus` |
| Env | `Python 3.13.7 | JAX 0.10.1 | jaxfne 0.4.17 | manifests sha256[:16]=4b87066d061f684f (w4a_blocker_seal) e80d53817457dcba (canonical_full_exposure_seal) | model_summary_hash=b2ae1c172a9f0311 observable_basis_hash=9492ed125b95885e` |

## 2. Goal

- **Research question:** Can vertical intra-area gain scaling rescue V1 L4→L2/3→L5 information transmission and, if not, what mechanism observability interval blocks the next justified intervention?
- **Acceptance predicate (before results):** POSITIVE if any `g_vertical ∈ [0.7,1.8]` (11 uniform) or typed vertical C020 or visual grammar C021 makes projecting layer carry information with `acc > 0.5+2SE` (binary `0.5+2√(p(1-p)/24)=0.703` at p=0.5, n=24) ∧ `|d| > 0.5` ∧ `p_perm < 0.05` for V1 L2/3 (or L5) while `Q_local` hard feasible preserved (`bar∈(1,30)`, `ρ<0.60`, `Efrac∈(0.05,0.95)` hard, `P(CV>0.5)≥0.15`, `max≥1.5`); else NEGATIVE_INSTANCE with blocker typing `MECHANISM_OBSERVABILITY_BLOCKER` if three hypothesis families each deliver presumed cause but phenotype invariant.
- **Out of scope / non-goals:** No new scientific primitives beyond `GEN2_C022` seal (builder science frozen); no `C022`/g_FF/g_FB/H/Theta mutation via W4; no duplicating JaxFNE primitive (NEVER-02/03); V0 visualization engineering is foundation, not scientific hypothesis test.

## 3. Starting authoritative state

| # | Claim | Tag | Authority (manifest / ledger / commit / paper) |
|---|-------|-----|-----------------------------------------------|
| 3.1 | C019 AGSDR-L1 theta* `[1.2565,1.1872,0.7088,1.2182,1.2887]` on C018 `be9b96ab679c9802` yields `y_bar_4 bar5.72±0.72 CV0.351 rho0.388 Fano0.95 Efrac0.75` hard feasible J_bar0.2402 | `OBSERVED` | `manifests/gen2_modification_ledger.jsonl:GEN2_C019` `scratch/agsdr_N2_receipt.md:49` `manifests/gen2_modification_ledger.json:GEN2_C019` |
| 3.2 | W4a G g_vertical [0.7,1.8] 11 candidates hard feasible but no g_vertical* : V1 L2/3 Δ0.02 d0.13 acc0.542 FAIL vs V1 L4 Δ169 d803 acc1.00 PASS | `OBSERVED` | `scratch/w4a_g_vertical_receipt.md:95` `manifests/w4a_blocker_seal.json:evidence_chain.v1_l4_to_l23_vertical` |
| 3.3 | C020 typed vertical v1 Ne16→54 k0.59→2.0 C0.444→0.704 cascade Γ 0.00023→0.0016 7× but V1 L2/3 -0.03 d-0.34 acc0.542 FAIL L5 0.09 acc0.542 FAIL | `OBSERVED` | `scratch/gen2_C020_receipt.md:109` `manifests/gen2_modification_ledger.jsonl:GEN2_C020` |
| 3.4 | C021 visual E/PV grammar 0/7.24→0.73/0.64 per-blob ∈0.5-2.0 Jaccard 0.454>0.3 PV huge 3889→1226 3.2× (drive 7.24→0.64 11×) ENERGY_A parity ≤5% V1-only, still V1 L2/3 FAIL | `OBSERVED` | `scratch/gen2_C021_receipt.md:4.2,4.4` `jomission/network/rf.py:68 VISUAL_PV_GAIN 0.7 :73 0xC02A` `manifests/gen2_modification_ledger.jsonl:GEN2_C021` |
| 3.5 | Paired C020↔C021 mediation Δ(PV/E)_L4 +2652 Hz pooled but ΔY_L2/3 +0.02 Hz α≈7e-06 not mediated | `OBSERVED` | `scratch/review_W4a3_P1_C021_transmission.md:60-62` |
| 3.6 | Representation adversary: V1 L2/3 truly absent not rate-invisible — rate LOO 0.708 p0.106, pop-vector LOO 0.708 p0.069, Vm 0.542 p0.817 FAIL; source q 1.00 diagnostic only low-variance averaging | `OBSERVED` | `scratch/review_W4a3_D_representation.md:2.1-3` |
| 3.7 | Upstream blocker `jaxfne/_model_simulate.py:280` guard `enable_hdp does not support nonzero delay_steps` + `jaxfne/emitters.py:714` delayed kernel has no `record_edge_current` (`_pipeline.py:395/:538` HDP-only) → I_edge unavailable at delays [20,80,120] | `OBSERVED` | `scratch/jaxfne_issue_delayed_edge_current.md:1` `jaxfne/_model_simulate.py:280` `jaxfne/emitters.py:714,2846` `jaxfne/_pipeline.py:395` |
| 3.8 | Transfer coefficient Γ_FF 0.40 vs Γ_vert 0.003-0.014 primary bottleneck, 10-100× below FF even after interventions | `DERIVED` | `scratch/review_W4_P2_budget.md:105` `scratch/review_W4a2_T1_topology.md:1` PROXY W·r·τ |
| 3.9 | V0 foundation engineering required before seal: ontology N_total128155 = N_static57351+N_dynamic22780+N_plastic4+N_history48000+N_recording20, N_free24 vs N_AGSDR5, ObservableBasis C_t→X_t→q_t→φ_t→y_t V_i DERIVED_FROM Vm, CSD DERIVED_FROM field_proxy | `OBSERVED` | `scratch/vis_V0_ontology.md:1` `jomission/visualization/model_summary.py:822 observable_basis` `jomission/visualization/manifest.py:verify_V0() True` `b2ae1c172a9f0311` |
| 3.10 | W4b/W4c/W5 gated; H/Theta via M-06 parallel; do not implement C022 | `DERIVED` | `manifests/w4a_blocker_seal.json:gating` `manifests/gen2_modification_ledger.jsonl:GEN2_C022` |
| 3.11 | Visualization sampling disclosure and log scaling required for heavy-tailed weights (CV1.5 σ1.085) and H taus 0.1-1000s 4 orders | `MODEL_ASSUMPTION` | `scratch/vis_V0_P1_sampling.md:1` `jomission/visualization/network_viz.py:165 stratified sampling` `jomission/visualization/sampling_utils.py` |
| 3.12 | Transfer oscilloscope chain V1 L4 E spikes→delay/history→syn_state→current→L2/3 Vm→spikes with NOT OBSERVABLE banner for current stage is correct engineering representation | `INFERRED` | `jomission/visualization/transfer.py:1` `jomission/visualization/run_report.py:Transfer Function tab` |

## 4. Work performed

Execution ≠ verification — artifacts verified by `verify_V0()`, ledger, and seal JSON where noted.

| Worker | Assignment | Action | Artifact | Result |
|--------|------------|--------|----------|--------|
| W4a propagation agent | Test `g_vertical ∈ [0.7,1.8]` 11 uniform + exc-only g5.0 scaling only `M_vertical={L4→L2/3, L2/3→L5}` disjoint from M_rec/M_FF287/M_FB303 delays [20,80,120] | `python scratch/w4a_g_vertical_runner.py` (2000ms spont + 1000ms×12×3 RF 32×32 window 100-531ms, theta* C019 preserved) | `scratch/w4a_g_vertical_receipt.md` `results/agsdr_local_w4a/trace.jsonl` | done — all 11 hard PASS but V1 L2/3 Δ0.02 d0.13 acc0.542 FAIL; exc-only g5.0 L23 acc0.562 FAIL |
| C020 typed vertical agent | Fix sparse recruitment + cancellation: vertical topology σ0.08→0.12 max25→40 for M_vertical only Ne16→54 + laminar×motif gains L4_E→L2/3_E×2.0 etc. | `build_jomission_model` overlay via `_apply_vertical_topology` + `_apply_vertical_motif_gains` (builder.py:210,220) no new simulator | `scratch/gen2_C020_receipt.md` `manifests/gen2_modification_ledger.jsonl:GEN2_C020` | done — delivered +38 edges C+0.26 cascade 7× but RF transmission FAIL |
| C021 visual grammar agent | Correct RF E/PV input grammar: identical → differential g_vis_PV 0.7 + interleaved tiling 0xC02A | `jomission/network/rf.py:68 VISUAL_PV_GAIN_DEFAULT 0.7 :73 INTERLEAVED_SEED_XOR 0xC02A` + RFOperator seeded shuffle, ENERGY_A re-derived | `scratch/gen2_C021_receipt.md` `manifests/gen2_modification_ledger.jsonl:GEN2_C021` | done — per-blob PV/E 0.73/0.64 Jaccard>0.3 PV 3.2× reduced but L2/3 FAIL |
| W4a3 P1 reviewer | Paired C020↔C021 frozen assay mediation + decoding (n=12×3 seed0,1, 1000 shuffles, LOO) | `scratch/review_W4a3_P1_C021_transmission.py` paired same seeds/Poisson/window/delays | `scratch/review_W4a3_P1_C021_transmission.md` | done — ΔM+2652 vs ΔY+0.02 α≈0 not mediated; no rescue |
| W4a3 D adversary | Test alternative decoders (pop-vector 30d, ISI, latency, transient, Vm, source q, phi) | `scratch/review_W4a3_D_representation.py` nearest-centroid ×1000 perm ×LOO | `scratch/review_W4a3_D_representation.md` | done — truly absent not rate-invisible (LOO 0.708 p0.106 FAIL) |
| V0 V1 ontology agent | Stratified ontology N_total128155 + ObservableBasis C→X→q→φ→y + Manifest V0=S∧N∧R∧P∧U∧D∧A | `jomission/visualization/model_summary.py:ontology_table` + `observable_basis.json` + `manifest.py:verify_V0()` | `scratch/vis_V0_ontology.md` `jomission/visualization/model_summary.py` `jomission/visualization/observable_basis.json` `jomission/visualization/manifest.py` hash `b2ae1c172a9f0311` | done — verify_V0() True |
| V0 V2 network agent | Hierarchy/motif/spatial/RF Plotly figs with same EdgeList arrays as analyses (752×16 gains, 90 delays) + deterministic stratified sampling disclosure | `jomission/visualization/network_viz.py:497 hierarchy_fig 640 motif_matrix 885 spatial_fig` + `sampling_utils.py` | `scratch/vis_V0_P1_sampling.md` `jomission/visualization/network_viz.py` `jomission/visualization/sampling_utils.py` | done — rendering 1000/10590 disclosure, log scaling for weights/tau |
| V0 V3 run-report agent | 12-tab run-report + Transfer oscilloscope V1 L4 E→L2/3 current→Vm→spikes with NOT OBSERVABLE banner at blocked current stage | `jomission/visualization/run_report.py` 12 tabs Overview/Raster/Rates/Membrane/EI/H/Θ/Spectra/Field/Stimulus/Connectivity/Diagnostics/Provenance + `transfer.py:transfer_figures()` 6 panels | `jomission/visualization/run_report.py` `jomission/visualization/transfer.py` `jomission/visualization/w4_dashboard.py` `jomission/visualization/agsdr_dashboard.py` | done — Transfer Function tab probes chain at machine-derived availability |
| V0 V4 adversary agent | Collapse labeled VIS Foundation deficiencies P0 field proxy not LFP, EI_PROXY watermark, derived labels, ms axes, provenance footer, CC-agnostic sampling, log scaling, V0 Sankey | `jomission/visualization/manifest.py:V0` + `network_viz.py:_provenance_footer_text :368` + `run_report.py:provenance` | `scratch/vis_V4_adversary.md` `scratch/vis_V0_P0_fix.md` | done — all V0 gates pass |
| Seal agent | Seal W4a as NEGATIVE/NO g_vertical* + MECHANISM_OBSERVABILITY_BLOCKER, NO builder science mutation beyond seal | `manifests/w4a_blocker_seal.json` + `scratch/w4a_blocker_seal.md` + ledger `GEN2_C022` | `manifests/w4a_blocker_seal.json` `scratch/w4a_blocker_seal.md` `manifests/gen2_modification_ledger.jsonl:GEN2_C022` `manifests/gen2_modification_ledger.json` | done — NO_MODEL_DELTA milestone W4A_MECHANISM_OBSERVABILITY_BLOCKER |

## 5. Evidence

| Ref | Path | Kind | Hash / receipt | Notes |
|-----|------|------|----------------|-------|
| E5.1 | `manifests/w4a_blocker_seal.json` | `manifest` | `sha256[:16]=4b87066d061f684f` | Stage W4a NEGATIVE MECHANISM_OBSERVABILITY_BLOCKER, parent a5171110 |
| E5.2 | `manifests/gen2_modification_ledger.jsonl` | `ledger` | `GEN2_C022 (alias GEN2_W4a_BLOCKER)` NO_MODEL_DELTA | Append-only ledger seal |
| E5.3 | `manifests/gen2_modification_ledger.json` | `ledger` | aggregate regenerated | Mirrors jsonl |
| E5.4 | `manifests/canonical_full_exposure_seal.json` | `manifest` | `sha256[:16]=e80d53817457dcba` | dt0.1 260 trials baseline |
| E5.5 | `scratch/w4a_g_vertical_receipt.md` | `log` | 211 lines, 11 candidates | G_W4a hard feasible but no carry |
| E5.6 | `scratch/gen2_C020_receipt.md` | `log` | 155 lines, Ne16→54 | Typed vertical v1 still FAIL |
| E5.7 | `scratch/gen2_C021_receipt.md` | `log` | 190 lines, PV/E 0.73/0.64 | Visual grammar corrected still FAIL |
| E5.8 | `scratch/review_W4a3_P1_C021_transmission.md` | `log` | 181 lines, paired mediation | ΔM+2652 vs ΔY+0.02 α≈0 not mediated |
| E5.9 | `scratch/review_W4a3_D_representation.md` | `log` | 176 lines, 16 decoders | Truly absent not rate-invisible |
| E5.10 | `scratch/jaxfne_issue_delayed_edge_current.md` | `log` | 287 lines, MISSING_CAPABILITY | Guard :280 + emitters:714 |
| E5.11 | `scratch/review_W4a2_T1_topology.md` | `log` | 193 lines, C(k)=1-exp(-k) | PROXY Γ synthesis |
| E5.12 | `scratch/review_W4_P1_propagation.md` | `log` | 263 lines | V1 L4 -445 vs downstream <0.15 |
| E5.13 | `scratch/review_W4_P2_budget.md` | `log` | 204 lines | Γ_FF0.40 vs Γ_vert 0.003-0.014 |
| E5.14 | `scratch/w4a_blocker_seal.md` | `log` | 33362 bytes | Human-readable seal + causal graph |
| E5.15 | `jomission/visualization/transfer.py` | `other` | `sha256[:16]=a00992bf81f5d76f` (run_report) network `8a5f1c9ae8622ee7` | Transfer oscilloscope 6 panels with NOT OBSERVABLE |
| E5.16 | `jomission/visualization/model_summary.py` | `other` | `model_summary_hash=b2ae1c172a9f0311` | Ontology N_total128155 |
| E5.17 | `jomission/visualization/observable_basis.json` | `other` | `observable_basis_hash=9492ed125b95885e` | C_t→X_t→q_t→φ_t→y_t basis |
| E5.18 | `jomission/visualization/manifest.py` | `other` | `verify_V0()=True S∧N∧R∧P∧U∧D∧A` | VisualizationManifest gate |
| E5.19 | `jomission/visualization/network_viz.py` | `figure` | stratified sampling disclosure | Hierarchy/motif/spatial figs |
| E5.20 | `jomission/visualization/run_report.py` | `other` | 12 tabs + Transfer Function tab | Run-report with provenance footer |
| E5.21 | `results/agsdr_local_w4a/trace.jsonl` | `array` | 11 candidates trace | Hard PASS preserved |
| E5.22 | `scratch/vis_V0_ontology.md` | `log` | ontology receipt | V0 ontology artifact |
| E5.23 | `scratch/vis_V0_P1_sampling.md` | `log` | sampling receipt | Disclosure + log scaling |

- **EvidenceRefs:** ledgers, manifests, scratch receipts, visualization manifests, run-report provenance.
- **Visualization report:** `jomission/visualization/transfer.py` + `jomission/visualization/run_report.py` Transfer Function tab (6-panel oscilloscope V1 L4 E spikes→delay/history→syn_state→current (NOT OBSERVABLE banner)→L2/3 Vm→L2/3 spikes); `jomission/visualization/network_viz.py` hierarchy/motif/spatial; `jomission/visualization/manifest.py` V0=S∧N∧R∧P∧U∧D∧A true.

## 6. Observations

Compact quantitative tables only — no interpretation.

| ID | Metric | Value | Units | n | Artifact | Notes |
|----|--------|-------|-------|---|----------|-------|
| Obs 6.1 | V1 L4 Δ (A-B) | 169.79 | Hz | 24 (12+12) | E5.5 | g_vertical pooled, d=646.94 acc=1.000 thresh 0.703 p<0.001 PASS |
| Obs 6.2 | V1 L4_E source Δ | 368.03 | Hz | 24 | E5.8 | d≈999 acc=1.00 p0.001 PASS; C021 pooled |Δ| 500 Hz preserved |
| Obs 6.3 | V1 L2/3 Δ (primary) | 0.02 | Hz | 24 | E5.5 | d=0.13 acc=0.542 thresh 0.703 p_perm=0.381 LOO 0.625 FAIL |
| Obs 6.4 | V1 L2/3 Δ C020 | -0.03 | Hz | 24 | E5.6 | d=-0.34 acc=0.542 FAIL; LOO 0.625 |
| Obs 6.5 | V1 L2/3 Δ C021 paired (seed0/1) | +0.04/-0.02 (seed0), +0.04/+0.06 (seed1) pooled |Δ| 0.04 | Hz | 48 (24+24) | E5.8 | d pooled 0.34/-0.18 acc 0.625/0.417 LOO 0.625/0.375 p>0.3 FAIL both |
| Obs 6.6 | V1 L5 Δ | 0.09 | Hz | 24 | E5.6 | d=0.64 acc=0.542 FAIL; exc-only g5.0 Δ0.14 acc0.75 but L23 acc0.562 FAIL |
| Obs 6.7 | Q_local bar | 5.09 (seed0) vs y_bar_4 5.72±0.72 | Hz | 400 neurons | E5.5 | rho 0.353 (<0.60) Fano 0.95 P(CV>0.5)0.183 max2.03 Efrac0.79 hard PASS all 11 g |
| Obs 6.8 | Vertical Γ_net (L4→L2/3) | 0.014→0.020 at g1.78 | ratio | 10590 edges | E5.5 | exc 0.060→0.107 still 25× < Γ_FF 0.40; cascade 0.00023→0.0016 7× |
| Obs 6.9 | V1 L2/3→L5 Ne/k/C | 16→54, 0.59→2.0, 0.444→0.704 (seed0) 0.519 (seed1) | count | per-area | E5.6 | +38 edges C+0.26 after C020; median_d 0.27→0.293 |
| Obs 6.10 | PV huge spiking (B-PV) | 3889→1226 | Hz | per-neuron subset 37% | E5.7 | 3.2× reduction at spike level; drive per-blob 7.24→0.64 11×; per-blob PV/E 0/7.24→0.73/0.64 Jaccard 0.454>0.3 |
| Obs 6.11 | Paired mediation ΔM vs ΔY | ΔM +2652 Hz pooled (PV/E L4) vs ΔY +0.02 Hz pooled (L2/3) | Hz | 48 | E5.8 | α≈7e-06 not mediated |
| Obs 6.12 | Representation adversary resub | pop-vector 30d acc 0.792 p0.084, E-vector 22d 0.917 p0.001 (resub) | acc | 24 | E5.9 | optimistic resub; see LOO |
| Obs 6.13 | Representation adversary LOO | rate 0.708 p0.106, pop-vector 0.708 p0.069, Vm 0.542 p0.817 | acc | 24 | E5.9 | all p>0.05 FAIL; source q 1.00 p0.001 diagnostic only Δ0.015 SD0.002 |
| Obs 6.14 | Upstream HDP+delay guard | ValueError at `jaxfne/_model_simulate.py:280` sum(delay_steps)=27960 | bool | n=400 n_edges=10590 delays [20,80,120] | E5.10 | non-HDP delayed kernel emitters.py:714 has no record_edge_current param; pipeline :395 HDP-only |
| Obs 6.15 | V0 ontology counts | N_total 128155 = N_static57351+N_dynamic22780+N_plastic4+N_history48000+N_recording20; N_free24 N_AGSDR5 | counts | N=400 | E5.16 | verify_V0()=True S∧N∧R∧P∧U∧D∧A |
| Obs 6.16 | Visualization scaling | weight CV1.5 σ1.085 μ-0.589 log `sign·log1p(|W|/1e-6)` H tau log10 0.1-1000s | transform | 10590 edges | E5.19 | stratified sampling 1000/10590 disclosure |
| Obs 6.17 | Transfer oscilloscope availability | V1 L4 E spikes measured, L4→L2/3 current NOT OBSERVABLE, L2/3 Vm measured | categorical | per panel | E5.15 | Banner JaxFNE delayed+HDP limitation machine-derived from delay_steps>0 |

_Figures:_ `jomission/visualization/transfer.py` Transfer Function oscilloscope (6 panels) + `jomission/visualization/network_viz.py` hierarchy/motif/spatial + `jomission/visualization/run_report.py` 12-tab run-report — see §5 E5.15-E5.20.

## 7. Interpretation

Observation ≠ interpretation. Each claim cites supporting observation ID(s) from §6.

| # | Interpretation | Supporting observations | Confidence |
|---|----------------|-------------------------|------------|
| 7.1 | V1 L4 is information-bearing source that survives all interventions — visual→V1 L4 transmission is intact (PASS). | `Obs 6.1, Obs 6.2` | `HIGH` |
| 7.2 | V1 L2/3 projecting layer does not carry A vs B information at any tested g_vertical (11+exc-only) — FAIL is not due to weight magnitude. | `Obs 6.3, Obs 6.4, Obs 6.5, Obs 6.6` | `HIGH` |
| 7.3 | Q_local hard feasibility preserved across all vertical scalings — failure is not local regime collapse. | `Obs 6.7` | `HIGH` |
| 7.4 | Typed vertical microcircuit delivered substantial structural improvement (Ne+38, C+0.26, Γ 7×) yet phenotype invariant — weight/topology at current drive/noise insufficient. | `Obs 6.8, Obs 6.9, Obs 6.3-6.5` | `HIGH` |
| 7.5 | Visual PV domination is not the causal bottleneck — 3.2× PV reduction leaves L2/3 invariant, mediation α≈0. | `Obs 6.10, Obs 6.11, Obs 6.5` | `HIGH` |
| 7.6 | V1 L2/3 information is truly absent, not hidden in alternative code (population vector, ISI, Vm all LOO fail; source q perfect is averaging artifact). | `Obs 6.12, Obs 6.13` | `MED` |
| 7.7 | Three hypothesis families each changed presumed cause substantially while phenotype invariant — strongest evidence for observability blocker rather than wrong lever. | `Obs 6.3-6.6, Obs 6.8-6.11` | `HIGH` |
| 7.8 | Bottleneck is correctly localized to interval `spikes_{L4_E}→I_{L4_E→L2/3}→V_{m,L23}` where stimulus difference disappears; realized edge-current I at delays [20,80,120] is the missing observable needed to disambiguate delay vs current vs Vm. | `Obs 6.14, Obs 6.17, Obs 6.1-6.5` | `MED` |
| 7.9 | Upstream limitation is MISSING_CAPABILITY (primary) / DOCUMENTATION_GAP (secondary) at `jaxfne/_model_simulate.py:280` + `emitters.py:714` — blocks `record_edge_current` at deployed delays. | `Obs 6.14, Obs 6.17` | `HIGH` |
| 7.10 | V0 ontology and transfer oscilloscope correctly render available stages while labeling current stage NOT OBSERVABLE — engineering foundation does not alter scientific blocker but makes it auditable. | `Obs 6.15, Obs 6.16, Obs 6.17` | `HIGH` |

## 8. Negative / insufficient results

| # | Claim | Verdict | Level / threshold | Notes |
|---|-------|---------|-------------------|-------|
| 8.1 | W4a g_vertical ∈[0.7,1.8] rescues V1 L2/3 projecting layer (acc>chance+2SE ∧ d>0.5) | `NEGATIVE_INSTANCE` | `acc 0.542 <0.703, d0.13<0.5, p0.381>0.05` | All 11 g hard feasible but none carries; even exc-only g5.0 acc0.562 FAIL |
| 8.2 | C020 typed vertical microcircuit rescues V1 L2/3 | `NEGATIVE_INSTANCE` | `Δ-0.03 d-0.34 acc0.542<0.703` | Ne16→54 k0.59→2.0 Γ 7× but FAIL `E5.6` |
| 8.3 | C021 visual E/PV grammar correction rescues V1 L2/3 | `NEGATIVE_INSTANCE` | `acc 0.417-0.542<0.703, |d|<0.5, p>0.3` | Per-blob PV/E balanced but mediation α≈0 `E5.8` |
| 8.4 | V1 L2/3 carries in alternative code (pop-vector, ISI, Vm) | `NEGATIVE_INSTANCE` | `LOO 0.708 p0.106>0.05, 0.708 p0.069>0.05, Vm 0.542 p0.817` | Truly absent not rate-invisible `E5.9` |
| 8.5 | Realized edge-current I_{L4→L2/3}(t) at delays [20,80,120] can be measured | `BLOCKED` | `jaxfne/_model_simulate.py:280 guard + emitters.py:714 no record param (_pipeline.py:395 HDP-only)` | MISSING_CAPABILITY `E5.10`; NOT OBSERVABLE banner in transfer oscilloscope `E5.15` |
| 8.6 | Next scientifically justified intervention can be selected from existing observables | `BLOCKED` | `invariant phenotype across 3 hypothesis families (Obs 6.11)` | MECHANISM_OBSERVABILITY_BLOCKER — requires upstream I_edge fix to disambiguate |

## 9. Unexpected findings

| # | Finding | Why unexpected | Follow-up priority |
|---|---------|----------------|-------------------|
| 9.1 | C020 cascade Γ 7× improvement leaves primary phenotype invariant | Expected Γ increase to move acc toward 0.75 per `gen2_C020_receipt.md:pred` 0.02→0.5; observed Δ-0.03 still at chance | `low` — gated until I_edge observable |
| 9.2 | C021 3.2× PV reduction (drive 11×) leaves L2/3 invariant α≈0 | Expected less PV inhibition to unmask E carrier and raise L2/3 Δ; observed ΔY +0.02 vs ΔM +2652 | `low` — confirms input grammar not causal |
| 9.3 | Source q diagnostic Acc1.00 Δ0.015 SD0.002 appears perfect while spiking L2/3 at chance | Naive expectation that field/source tracks spikes; instead linear averaging `30×4310` reduces variance 100× masking absence | `med` — keep DIAGNOSTIC_ONLY watermark in visualizations |
| 9.4 | V0 ontology reconciliation shows N_total 128155 with state/history split exceeding N_free24 by 3 orders — visualization law distinction prevents conflating numbers vs tunable scalars | Prior vague V0 counts (V1 ModelSummary) conflated N_model vs N_tunable vs N_AGSDR | `low` — promoted via manifest gate |

_If none beyond above:_ not applicable — unexpected captured.

## 10. Tool / workflow friction

| Friction | Cause | Workaround | Permanent repair? |
|----------|-------|------------|-------------------|
| Delayed edge-current recording unavailable at finite delays [20,80,120] (`record_edge_current` HDP-only, guard at jaxfne/_model_simulate.py:280) | Upstream JaxFNE HDP+delay incompatibility (emitters.py:714 lacks param, _pipeline.py:395 HDP-only) | Render transfer oscilloscope stage literally as NOT OBSERVABLE banner machine-derived from delay_steps>0; retain PROXY W·r·τ flag with documentation | `JAXFNE_UPSTREAM — proposed in §12 Option A/B/C per jaxfne_issue_delayed_edge_current.md:200` |
| Repeated failed g_vertical search invites guessing next lever without observable | Mechanism observability blocker (interval L4_E spikes→L2/3 Vm opaque) | Seal W4a as MECHANISM_OBSERVABILITY_BLOCKER NEGATIVE/NO g_vertical*, gate W4b/W4c/W5, parallelize J∥M∥D | `PROJECT_RULE — W4 gating rule proposed in §12` |
| Heavy-tailed weight distribution (CV1.5) and 4-order H tau span unreadable on linear | Log mapping needed | Apply `sign·log1p(|W|/1e-6)` and log10 tau with linear CV/rho per sampling_utils | `VISUALIZATION — already in §12 via sampling_utils` |

## 11. Learned lessons

| Lesson | Trigger | Cause | Repair | Evidence | Scope | Confidence | Persistence |
|--------|---------|-------|--------|----------|-------|------------|-------------|
| Vertical weight scaling cannot overcome sparse recruitment (Ne16 k0.59) and E/I cancellation (W_net 4.4×<W_exc) — needs topological k fix, not g | All 11 g_vertical hard feasible but L2/3 acc flat 0.542→0.542 even exc-only g5.0 FAIL | k0.59 bottleneck (15/27 L5 get 0 inputs) C_vert0.444, Γ_net 0.014 25×<FF0.40 | C020 supplemental topology σ0.12 max40 Ne16→54 + typed W E→E×2.0 vs E→SST×0.7 (still FAIL but correct direction) | E5.5 E5.6 | `project` | `HIGH` | `YES` |
| Per-blob co-recruitment metric (PV/E∈0.5-2.0 Jaccard>0.3) is correct visual grammar check, but PV domination is not vertical bottleneck | C021 PV/E 0→0.73/0.64 Jaccard0.454 PV huge 3.2× reduced but mediation α≈0 | Lateral PV feedforward inhibition overwhelms carrier intuition but does not mediate vertical transfer | Keep interleaved tiling 0xC02A + visual_pv_gain0.7 as V1 input fix; do not use as vertical lever | E5.7 E5.8 | `project` | `MED` | `YES` |
| NOT OBSERVABLE must be rendered literally, not hidden, when stage is blocked by upstream capability | jaxfne/_model_simulate.py:280 guard blocks I_edge at deployed delays | MISSING_CAPABILITY/DOCUMENTATION_GAP requires visible provenance, not manual typing | Transfer oscilloscope 6-panel chain with machine-derived NOT OBSERVABLE banner from delay_steps>0 | E5.10 E5.15 | `project` | `HIGH` | `YES` |
| Truly absent vs rate-invisible requires LOO + multiple decoders + Vm + source q diagnostic; resub alone inflates (0.917) | Resub pop-vector 0.917 p0.001 but LOO 0.708 p0.069 FAIL | High-D vs n=24 overfit; averaging artifact q variance 100× smaller | Keep D adversary protocol: rate + pop-vector + ISI + Vm with 1000 perms + LOO | E5.9 | `project` | `HIGH` | `YES` |
| V0 ontology N_model≠N_free≠N_AGSDR prevents conflating numbers vs tunable scalars | N_total128155 vs N_static57351 vs N_free24 vs N_AGSDR5 confusion in V1 | Derived delay_steps/lognormal sigma vs configured gains conflated | ModelSummary stratified table + ObservableBasis C→X→q→φ→y + V0=S∧N∧R∧P∧U∧D∧A manifest gate | E5.16 E5.18 | `project` | `HIGH` | `YES` |

## 12. Harness / tool proposals

| # | Type | Target | Rationale | Evidence |
|---|------|--------|-----------|----------|
| 12.1 | `JAXFNE_UPSTREAM` | `jaxfne/emitters.py:714 _simulate_edge_recurrent_izhikevich_delayed` + `jaxfne/_model_simulate.py:280` + `jaxfne/_pipeline.py:395` | Expose `edge_current_trace (n_steps,n_edges)` at deployed delays [20,80,120] Option A (extend delayed kernel) or Option B (diagnostic buffer) or Option C (documentation-only with validated equivalence bound); needed to measure `spikes→I→Vm` and disambiguate delay vs current vs Vm | `Obs 6.14` `E5.10` |
| 12.2 | `PROJECT_RULE` | `AGENTS.md` W4 gating rule | After MECHANISM_OBSERVABILITY_BLOCKER seal (W4a NEGATIVE ×3 invariant), gate W4b(g_FF)/W4c(g_FB)/W5 and do not implement C022 until upstream I_edge available; H/Theta via M-06 parallel | `Obs 6.3-6.6, Obs 6.11` |
| 12.3 | `TEST/GATE` | `manifests/gen2_ledger.schema.json` + `jomission/visualization/manifest.py:verify_V0` | Add VisualizationManifest gate `V0 = S∧N∧R∧P∧U∧D∧A` (S counts reconcile, N ontology, R 14 tabs, P provenance, U ms, D proxy/derived labels, A sampling_disclosed) plus transfer coefficient `G_i→j EMPIRICAL_TRANSFER_RATIO` | `Obs 6.15, Obs 6.16` |
| 12.4 | `VISUALIZATION` | `jomission/visualization/sampling_utils.py` + `network_viz.py` | Deterministic stratified sampling 1000/10590 by area×layer×class + disclosure + log scaling `sign·log1p(|W|/1e-6)` for CV1.5 + log10 tau for 4-order H | `Obs 6.16` |
| 12.5 | `VISUALIZATION` | `jomission/visualization/transfer.py` + `run_report.py` Transfer Function tab | 6-panel oscilloscope V1 L4 E spikes→delay/history→syn_state→current→L2/3 Vm→spikes with NOT OBSERVABLE banner machine-derived from delay_steps>0 | `Obs 6.17` |
| 12.6 | `AGENT_WORKFLOW` | `docs/sessions/` J∥M∥D parallel | W4a blocker seal parallel with V0 visualization V1-V4 and M-06 H/Theta to avoid serial blocking on observability | `§4 W4a3` |

## 13. Scientific state transition

| Field | Value |
|-------|-------|
| **Before** | C019 `65b302e8c7cdceb5` theta* hard feasible y_bar_4 5.72±0.72; W4a needs another intervention (expect g_vertical or topology to rescue L2/3) |
| **After** | Sealed as W4a MECHANISM_OBSERVABILITY_BLOCKER NEGATIVE/NO g_vertical* at parent C021 `a51711101576c7c3` (rf 18c2f0f55229c307) → C022 `a5171110→a5171110` zero-delta NO_MODEL_DELTA milestone W4A_MECHANISM_OBSERVABILITY_BLOCKER; V0 `b2ae1c172a9f0311` true |
| **Unlocked** | Observable hierarchy: Visual→V1 L4 PASS vs V1 L4→L2/3 FAIL localized to interval spikes_{L4_E}→I→V_{m,L23}; transfer oscilloscope 6-panel chain with auditable NOT OBSERVABLE; V0 foundation S∧N∧R∧P∧U∧D∧A |
| **Still gated** | W4b(g_FF)/W4c(g_FB)/W5 — source L2/3 at chance so FF scaling gated; H/Theta via W4 gated, via M-06 parallel; any next vertical lever (k to 5.0 C0.993, E→E×3, SST×0.5) without I_edge would be guessing |
| **Invalidated** | Hypotheses H1 (vertical weights too weak g scaling), H2 (vertical topology/type inadequate C020 v1), H3 (visual PV domination) as sufficient rescues — all NEGATIVE despite large delivered cause changes |
| **New uncertainty** | Exact sub-step where signal disappears within delay/history→syn_state→synaptic current→V_m (delay ring misalignment vs current cancellation 4.4× vs Vm integration/threshold) — requires realized I_{L4→L2/3}(t) at deployed delays [20,80,120] |

## 14. Next action

- **Primary (highest-value):** Parallelize J∥M∥D IN PROGRESS — (J) Upstream JaxFNE fix to expose `I_{L4_E→L2/3}(t)=W·syn_state(t-delay)` at finite delays [20,80,120] (track `jaxfne_issue_delayed_edge_current.md` Options A/B/C, M-06 parallel handles H/Theta exposure_batch_size optimization separately); (M) No builder science mutation until I_edge available — keep `jomission/network/builder.py` frozen science `a5171110` zero-delta, maintain transfer oscilloscope with NOT OBSERVABLE provenance; (D) Data plane — retain paired seeds 0,1 ×12×3 RF 32×32 window 100-531ms with 1000 perms + LOO as frozen assay for future I_edge disambiguation (delay vs current vs Vm). *Rationale:* three independently motivated interventions changed causes substantially with α≈0 invariance — next lever cannot be selected without guessing; observable must be fixed first. *Ready predicate:* Upstream branch exposes `edge_current_trace (n_steps,n_edges)` for `enable_hdp=False + delays>0` or validated equivalence with bound OR documented MISSING_CAPABILITY persists (gated state is terminal for W4). Both predicates already satisfied for parallel tracking — J is upstream issue, M/D are frozen.
- **Secondary (optional):** If upstream remains blocked long-term, design synthetic `C(k)=1-exp(-k)` sweep for L2/3→L5 k targets (Ne16@0.59→27@1.0 C0.632→81@3.0 C0.95→135@5.0 C0.993) and laminar×motif gain search documented ONLY as candidate pool, not implementation — still requires I_edge to rank.

## 15. Progress score

| Field | Value |
|-------|-------|
| Current | `73` |
| Previous | `72` |
| Delta | `+1` |
| Reason | Verified capability increment of +1 (72→73): V0 foundation + W4a blocker typing + transfer oscilloscope are verified engineering/scientific capabilities, not activity — `verify_V0() True` (b2ae1c172a9f0311) with ontology N_total128155 + ObservableBasis + V0=S∧N∧R∧P∧U∧D∧A, blocker seal `w4a_blocker_seal.json` (a5171110→a5171110 NO_MODEL_DELTA, GEN2_C022) with causal graph and NOT OBSERVABLE provenance, and transfer oscilloscope `transfer.py` 6-panel chain; no scientific primitive mutation beyond seal documentation. |

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
