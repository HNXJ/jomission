# Session: `2026-08-29_005_fg11-transfer-operating-manifold` — FG-11 transfer operating manifold discovery (supplementary HDP-OFF)

---

## 1. Session identity

| Field | Value |
|-------|-------|
| Session ID | `2026-08-29_005_fg11-transfer-operating-manifold` |
| UTC | `2026-08-29T22:00:00Z` |
| Local | `2026-08-29 18:00 (America/New_York, UTC-04:00)` |
| Branch | `main` |
| HEAD / code SHA | `f9af396` (full: `f9af396a0235f405eb75d786b01d89166c7f2a9a`) |
| Parent model | `manifests/gen2_modification_ledger.jsonl:GEN2_C019` (`config_hash: be9b96ab679c9802`, `hp_hash: f327f9d2ad64cc88`) → Result `GEN2_C019 65b302e8c7cdceb5` (`theta* [1.2565,1.1872,0.7088,1.2182,1.2887]`) |
| Config / hash | `config_hash=be9b96ab679c9802` (parent C019), `config_hash=65b302e8c7cdceb5` (C019 frozen), `hp_hash=f327f9d2ad64cc88`, `freeze_sha256[:16]=6a1d5262332bca74` (`manifests/agsdr_local_freeze.json`) |
| Agent roles | `Worker: Session Backfill Agent (FG-11 supplementary) | Worker: Prior FG-11 RUN (N0 24×S1 + N1 revised + N2 ISN) | Reviewer: none (backfill) | Approver: —` |
| Env | `Python 3.13.7 | JAX 0.10.1 | jaxfne 0.4.17 | manifests sha256[:16]=5c91b6d083a9493c (ledger), 4b87066d061f684f (w4a_blocker_seal), 6a1d5262332bca74 (agsdr_local_freeze), 4eba60d328cce3e4 (environment)` |

## 2. Goal

- **Research question:** Does a current-based Gen-2 architecture admit a locally qualified operating manifold that transmits stimulus information from V1 L4 to L2/3, demonstrating that w4a vertical failure is an operating-regime problem rather than structural impossibility, at supplementary HDP-OFF level?
- **Acceptance predicate (before results):** POSITIVE (as `SUPPLEMENTARY_TRANSFER_CANDIDATE` only) if `∃θ∈Θ_admissible` with `I(S;V_{L23})>0` (`I_corr>0.05 bits p<0.05` KSG `k=3 shuffle500`) AND `I(S;spk_{L23})>0` AND `G_{I→V}=|ΔV_m|/(|ΔI|+0.01)` elevated vs `C021_s0 0.013` with `Q_C019` hard preserved (`bar∈[3,15] ρ<0.60 P(CV_E>0.5)≥0.15 max≥1.5 Fano∈[0.7,2] Efrac∈(0.05,0.95)`) and `A_rec` measured not hard (if claiming amplification require `A_rec>1 CI excludes 1`, otherwise recurrent mediation via loss on `rec-off`+`matched` controls suffices); canonical promotion requires FG-12 condition 6 (`delayed+HDP` reproduction once `U01b` solved) — not evaluated here. Else NEGATIVE_INSTANCE if 0/24 primary success, else INCONCLUSIVE if only single-seed without controls.
- **Out of scope / non-goals:** No `jomission/network/builder.py` science mutation beyond `theta_overlay` (`g_rec/g_fastEI/g_dend/g_bg` overlay via `EdgeList.weight` gains + `I_bg(λ)` mean-controlled); no `GEN2_C0xx` promotion (`C_next FORBIDDEN until FG-12`); no canonical `delayed+HDP` claim (`U01b BLOCKED` at `jaxfne/_model_simulate.py:280`); no `W4b/W4c/W5` or `H/Θ` mutation (`M-06` parallel); no re-tuning `ε,W,B` after seeing `G/I`.

## 3. Starting authoritative state

Each claim tagged `OBSERVED` / `DERIVED` / `INFERRED` / `MODEL_ASSUMPTION` / `LITERATURE_PRIOR` / `UNRESOLVED`.

| # | Claim | Tag | Authority (manifest / ledger / commit / paper) |
|---|-------|-----|-----------------------------------------------|
| 3.1 | W4a sealed `MECHANISM_OBSERVABILITY_BLOCKER` NEGATIVE: Visual→V1 L4 PASS Δ169.79 d646 acc1.00 but V1 L4→L2/3 FAIL Δ0.02 d0.13 acc0.542 after 3 interventions (`g_vertical` ×1.8-5, C020 `Ne16→54 k0.59→2.0 C0.444→0.704` cascade 7×, C021 `PV/E 0.73/0.64 Jaccard0.454`) invariant | `OBSERVED` | `manifests/w4a_blocker_seal.json:1-40`, `scratch/w4a_g_vertical_receipt.md:95`, `scratch/gen2_C020_receipt.md:109`, `scratch/gen2_C021_receipt.md:4.2`, `manifests/gen2_modification_ledger.jsonl:GEN2_C022` |
| 3.2 | Bottleneck localized to interval `spikes_{L4_E}(t)→I_{L4_E→L2/3}(t)→V_{m,L2/3}(t)` at deployed delays `[20,80,120]`; canonical `spikes→I→Vm` chain not observable because `delayed+HDP` BLOCKED at `enable_hdp does not support nonzero delay_steps` + `emitters.py:714` lacks `record_edge_current` | `OBSERVED` | `jaxfne/_model_simulate.py:280`, `jaxfne/emitters.py:714`, `jaxfne/_pipeline.py:395`, `scratch/jaxfne_issue_delayed_edge_current.md:1`, `scratch/fg01_U01_U02_verification.md:21` |
| 3.3 | HDP-OFF delayed grouped path AVAILABLE: `RuntimeConfig(enable_hdp=False, record_grouped_current, edge_group_ids, G≈16-759)` via `emitters.py:879 grouped=_segment_sum(w·syn,G)` passive `ΔV_m0 Δspikes0 max|gc−Σec|<1e-5` G≈16 0.6MiB / G759 30MiB, `G` production collapsed | `OBSERVED` | `scratch/fg01_U01_U02_verification.md:33`, `jaxfne/emitters.py:728,879`, `recording/observables.py:81`, `scratch/fg09b_HDP_OFF_bridge.md:1.1` |
| 3.4 | FG-09b HDP-OFF bridge shows `I_syn` carries in 6/6 (`d>240 p0.001 I0.67`) while `V_m` invariant in important cases (`C021_s0 d-0.11 p0.827 I0.0 G0.013`) heterogeneous 0.013-0.85, fast-state equivalence `Δbar<0.03Hz <0.4% ΔVm<0.02mV passivity0` → I→Vm leading operator hypothesis, identifying for fast dynamics | `OBSERVED` | `scratch/fg09b_HDPOFF_bridge_raw.json`, `scratch/fg09b_HDP_OFF_bridge.md:55-80`, `docs/sessions/2026/2026-08-29_004_fg09b-hdp-off-transfer-localization.md:6` |
| 3.5 | Parent models frozen: C019 `theta* [1.2565,1.1872,0.7088,1.2182,1.2887]` AGSDR y_bar4 `5.72±0.72 J0.2402` hard feasible; C020 typed vertical `σ0.12 max40 Ne54`; C021 `g_vis_PV0.7 0xC02A Jaccard0.454 G0.013` baseline | `OBSERVED` | `manifests/gen2_modification_ledger.jsonl:GEN2_C019/C020/C021`, `scratch/gen2_C019_receipt.md:21`, `builder.py:210,220`, `rf.py:68,73` |
| 3.6 | FG-11 authorized as MECHANISTIC/SUPPLEMENTARY/HDP-OFF only: `θ_impl=[g_rec,g_fastEI,g_dendI,g_bg]` 4D orthogonal disjoint from `M_vertical/M_FF287/M_FB303` delays `[20,80,120]`, LHS `D*≈O(1/N)` seed0, successive `24×S1→8×S2→3×S4=52 sims` + controls, Pareto feasible-first on `[I_Vm,I_sp,G,A,J_ρ,J_EI,J_local]` not scalar `|bar−7.5|`, 3 controls (intact/REC-OFF/matched) + ISN characterization `NON_ISN_LIKE` vs `UNRESOLVED` | `DERIVED` | `scratch/fg11_supplementary_search.md:1`, `scratch/fg11_RUN_update.md:0`, `scratch/fg11_N1_revised.md:0`, `scratch/fg11_N2_ISN.md:1` |
| 3.7 | Whether current-based Gen-2 can ever transmit L4→L2/3 information at *any* admissible operating point (architecture capability vs structural impossibility) remains UNRESOLVED before FG-11 realized `I[t,G]` measurement; `C_next FORBIDDEN until FG-12` per w4a seal | `UNRESOLVED` | `scratch/w4a_blocker_seal.md:7`, `scratch/fg11_supplementary_search.md:10` |
| 3.8 | `H/Θ` firewall intact: `h_state.py:28 H_COORDINATES 5×` conceptual vs `emitters.py:3703 tau_i=tau0·size³` scalar 1-D, `hdp.py:14 Theta dim2` frozen, omission T1-7 `FORBIDDEN_TERMS_B10` 0 hits, `M-06` parallel handles ontology | `MODEL_ASSUMPTION` | `dynamics/h_state.py:28,73`, `jaxfne/emitters.py:3703`, `jaxfne/hdp.py:14`, `gen2_gates.py:32`, `scratch/fg14_H_ontology.md:1` |

## 4. Work performed

Execution ≠ verification — note which artifacts were verified by tests/gates.

| Worker | Assignment | Action | Artifact | Result |
|--------|------------|--------|----------|--------|
| Backfill | Create FG-11 session report (15 sections + sidecar) | Read authoritative state (`fg11_supplementary_search.md`, `fg11_RUN_update.md`, `fg11_N1_revised.md`, `fg11_N2_ISN.md`, ledger `GEN2_C019`, `N0_trace.jsonl`, `N2_summary.json`, env) + write Markdown + JSON per `SCHEMA.md` | `docs/sessions/2026/2026-08-29_005_fg11-transfer-operating-manifold.md` + `.json` | done |
| Backfill | Validate session report | `python docs/sessions/validate.py docs/sessions/2026/2026-08-29_005_fg11-transfer-operating-manifold.md && python docs/sessions/validate.py --reindex` | validator receipt | done |
| Prior (FG-11 N0) | Execute N0 24×S1 LHS HDP-OFF | `PYTHONPATH=. python scratch/fg11_RUN_update_runner.py` (seed 0, 2000 ms spont `dt0.1 T20000` + 1000 ms evoked `T10000 G759 30MiB` grouped `G≈16 0.6MiB` collapsed, `RuntimeConfig enable_hdp=False record_grouped_current` `emitters.py:728`, RF 32×32 field8 σ1.8 spacing3.2 overlap0.45 `ENERGY_A 923.024/921.94` `W[100,531) B_proxy[0,50)`) LHS D*≈O(1/N) seed0 `perm+jitter cut linspace(0,1,25)` | `results/fg11_RUN_update/N0_trace.jsonl` (24 lines) | done — 24/24 hard Q_C019 PASS, 2/24 primary `I_Vm>0 ∧ I_sp>0`, A_rec<1 all |
| Prior (FG-11 N1 revised) | Re-rank N1 8×S2 manifold | Pareto feasible-first on 7-dim `J_vector=[J_ρ,J_EI,J_local,J_I_Vm,J_I_sp,J_G,J_A]` (`E<PV` soft not hard), diversity 8 slots: 2 primaries + complementary + high-G + good-sync; current-space maps (η_rec,B_EI,χ) size I_sp vs G; controls plan triplicate | `results/fg11_RUN_update/N1_revised_summary.json` (8 `FG11_N0_01,13,06,20,22,11,15,08`) + 3 maps `current_space_3d_Isp.png`, `current_space_3d_G.png`, `current_space_2d_eta_B.png` | done |
| Prior (FG-11 N2 ISN) | Execute N2 3×S4 + 3 controls + ISN | `PYTHONPATH=. python scratch/fg11_N2_ISN_runner.py` (3 finalists 01,13,06 ×S4 `[0,1,2,3] mean_{4}` 2000 ms + 3 controls intact/REC-OFF `g_rec→0` /matched `(μ,σ)` `builder.py:44 I_bg(λ)` mean-controlled + ISN perturb +20% `PV+SST` and `PV` alone `drive*=1.20`) + `scratch/plot_N2_ISN.py` maps | `results/fg11_N2_ISN/N2_trace.jsonl` (12 per-seed) + `results/fg11_N2_ISN/N2_summary.json` + 3 maps `current_space_3d_Isp_N2.png`, `current_space_3d_G_N2.png`, `current_space_2d_eta_B_N2.png` | done — N2 3/3 hard PASS, G 0.569-1.41, ISN NON_ISN_LIKE 3/3 |
| Prior (FG-11 seams) | Verify grouped passive + builder clean | `grep -n record_grouped_current jaxfne/_model_simulate.py jaxfne/emitters.py _pipeline.py` + `sha256sum` + `git diff --stat -- jomission/network/builder.py` → clean overlay-only | `scratch/fg01_U01_U02_verification.md:33` `max|gc−Σec|<1e-5`, builder `MOTIF_GAIN 1.70 VERTICAL σ0.12 VIP b0.20 E_MIXTURE M2 CV1.5` preserved | done — HDP_OFF passive, no science mutation |

## 5. Evidence

| Ref | Path | Kind | Hash / receipt | Notes |
|-----|------|------|----------------|-------|
| E5.1 | `manifests/gen2_modification_ledger.jsonl` | `ledger` | `sha256[:16]=5c91b6d083a9493c` `GEN2_C022` last entry `GEN2_W4a_BLOCKER` | Append-only ledger, parent `GEN2_C019` `be9b96ab679c9802` → `65b302e8c7cdceb5` |
| E5.2 | `manifests/w4a_blocker_seal.json` | `manifest` | `sha256[:16]=4b87066d061f684f` | W4a `MECHANISM_OBSERVABILITY_BLOCKER` seal, `config_hash a51711101576c7c3` `rf 18c2f0f55229c307` |
| E5.3 | `manifests/agsdr_local_freeze.json` | `manifest` | `sha256[:16]=6a1d5262332bca74` | Successive fidelity `S_1[0] S_2[0,1] S_4[0,1,2,3]` 52-sims spec `6a1d5262*` |
| E5.4 | `manifests/environment.json` | `manifest` | `sha256[:16]=4eba60d328cce3e4` | Python 3.13.7 / JAX 0.10.1 / jaxfne 0.4.17 |
| E5.5 | `scratch/fg11_supplementary_search.md` | `other` | `sha256[:16]=9d9cfd3b29c181c4` | 413 lines, FG-11 4D LHS spec N0 24 table, Φ, Pareto, controls, promotion §10 |
| E5.6 | `scratch/fg11_RUN_update.md` | `other` | `sha256[:16]=6a3fa0956b600cbf` | Updated spec with 2 modifications (E<PV soft, A_rec measured) + N0 measured 24 table |
| E5.7 | `scratch/fg11_N1_revised.md` | `other` | `sha256[:16]=afbee42e474602cb` | N1 revised 8-slot manifold, current-space maps, 3-controls plan |
| E5.8 | `scratch/fg11_N2_ISN.md` | `other` | `sha256[:16]=4704fc3b496fbcff` | N2 3×S4 + controls + ISN characterization `NON_ISN_LIKE` |
| E5.9 | `results/fg11_RUN_update/N0_trace.jsonl` | `array` | `sha256[:16]=e4ee662aec0ecc0e` (24 lines) | N0 24×S1 realized Φ+G+A_rec hard/abort per seed |
| E5.10 | `results/fg11_RUN_update/N1_revised_summary.json` | `other` | `sha256[:16]=a5225cbc99908454` | N1 8 `01,13,06,20,22,11,15,08` revised ranking + `current_space_map` 24 entries |
| E5.11 | `results/fg11_N2_ISN/N2_summary.json` | `array` | `sha256[:16]=3ead6950fb1b05b2` | N2 3 finalists S4 `mean_{4}` + ISN per-seed + `pareto_dom0` |
| E5.12 | `results/fg11_N2_ISN/N2_trace.jsonl` | `array` | `sha256[:16]=12389cce7f62ab46` | 12 per-seed lines (3×4) intact/off/matched/ISN |
| E5.13 | `results/fg11_RUN_update/current_space_3d_Isp.png` | `figure` | — | View 1 (η_rec,B_EI,χ) size `I(S;spikes)` lineage N0→N1 |
| E5.14 | `results/fg11_RUN_update/current_space_3d_G.png` | `figure` | — | View 2 size `G_{I→V}` |
| E5.15 | `results/fg11_N2_ISN/current_space_3d_Isp_N2.png` | `figure` | — | N2 lineage 3D Isp |
| E5.16 | `results/fg11_N2_ISN/current_space_3d_G_N2.png` | `figure` | — | N2 lineage 3D G |
| E5.17 | `results/fg11_N2_ISN/current_space_2d_eta_B_N2.png` | `figure` | — | 2-D η vs B_EI color χ size G |
| E5.18 | `scratch/fg11_N2_ISN_runner.py` | `other` | `sha256[:16]=f092b1b9c2ae7ceb` | Repro runner intact/off/matched/ISN per seed |
| E5.19 | `scratch/fg11_RUN_update_runner.py` | `other` | `sha256[:16]=53b94864ae0d17cc` | N0 LHS deterministic seed0 `FG11_THETAS` |
| E5.20 | `docs/sessions/2026/2026-08-29_005_fg11-transfer-operating-manifold.md` | `other` | `this file` | Session Markdown (authoritative) |
| E5.21 | `docs/sessions/2026/2026-08-29_005_fg11-transfer-operating-manifold.json` | `other` | `—` | JSON sidecar (derived) |

- **EvidenceRefs:** E5.1-E5.21 linked above; quantitative authority `E5.9` N0 24×S1 + `E5.11-E5.12` N2 S4 + maps `E5.13-E5.17`
- **Visualization report:** `results/fg11_RUN_update/current_traces_schematic.png` + `results/fg11_N2_ISN/current_space_*_N2.png` via `scratch/plot_N2_ISN.py` (see §6 Obs 6.11)

## 6. Observations

Compact quantitative tables only — no interpretation.

| ID | Metric | Value | Units | n | Artifact | Notes |
|----|--------|-------|-------|---|----------|-------|
| Obs 6.1 | N0 24×S1 LHS hard Q_C019 preserved (+ abort surface) | 24/24 PASS, 0 abort | — | 24 (`S1 [0]`) | `E5.9:hard_feasible,abort` | `bar 4.77-5.08 ∈[3,15] Fano0.95 ∈[0.7,2] P(CV_E>0.5)0.174-0.183≥0.15 maxCV1.98-2.03≥1.5 Efrac0.757-0.819∈(0.05,0.95) hard PASS` `E<PV diff0.61-0.75` soft `J_EI`; `ρ0.352-0.447 J_ρ>0` soft `[-0.05,0.2]` `ρ<0.60` hard none >0.60; `n=400 neurons` |
| Obs 6.2 | Primary success `I(S;V_{L23})>0 ∧ I(S;spk_{L23})>0` with Q preserved (KSG `k=3 shuffle500 I_corr>0.05 p<0.05`) N0 S1 | 2/24 meet both | — | 24 | `E5.9:evoked I_Vm_bits I_sp_bits p` | `FG11_N0_01 I_Vm0.138 p0.036 I_sp0.463 p0.007 G4.98 bar4.88 ρ0.406` + `FG11_N0_13 I_Vm0.256 p0.041 I_sp0.321 p0.005 G2.79 bar4.83 ρ0.433` meet; remaining 22 partial/ns |
| Obs 6.3 | `G_{I→V}=|ΔV_m|/(|ΔI|+0.01)` N0 S1 | 0.007-4.98 (5/24 `>0.6`, 3/24 `>1.0`) | `mV/native` | 24 | `E5.9:G_I_to_Vm` | `C021_s0 0.013` baseline vs `01 G4.98 (ΔI0.022 ΔVm0.159)` `13 G2.79 (ΔI0.117)` `20 G1.42` `08 G2.15` `11 G1.00` high-G but I weak in 3/5; `C019_s0 0.851` reference |
| Obs 6.4 | `η_proxy=|ΔI|/σ_total` (`σ0.94` `builder.py:44`) and `η_rec=‖I_rec‖/‖I_total‖` realized L2 proxy | 0.023-0.915 (proxy) → `η_rec` L2 `0.003-0.229` | dimensionless | 24 S1 / 3 S4 | `E5.9:eta_proxy` `E5.11:eta_proxy S4` | S1 proxy 0.023 (01) to 0.915 (10) median ~0.24; S4 N2 `01 η0.024 B0.737` `13 η0.068 B0.778` `06 η0.229 B0.716` spread `Δη0.205` |
| Obs 6.5 | `I(S;V_m)` bits KSG N0 S1 | 0-0.450 (7/24 `p<0.05`) | `bits` `[0,1]` | 24 × `n_per_cond 6` `12 trials` `W[100,531) B_proxy[0,50)` `ε_Vm1.0` | `E5.9` | `01 0.138 p0.036`, `06 0.450 p0.005`, `13 0.256 p0.041`, `07 0.255 p0.014`, `20 0.267 p0.025` etc.; `V1 L2/3 E 22N` not `all L23 30N` |
| Obs 6.6 | `I(S;spikes_{L23})` bits KSG N0 S1 | 0-0.463 (6/24 `p<0.05`) | `bits` | 24 | `E5.9: I_sp` | `01 0.463 p0.007`, `22 0.560 p0.004`, `13 0.321 p0.005`, `11 0.305 p0.024`, `03 0.305 p0.018` etc.; complementary dissociation vs Vm-only |
| Obs 6.7 | `A_rec=‖ΔY_intact‖/(‖ΔY_off‖+5.48)` L2 `L23E 22N` measured CI95 | 0.096-0.725 all `<1` CI excludes 1 | dimensionless `>1 amplifies` | 24 S1 + 3×4 S4 | `E5.9:A_rec,A_rec_CI95` `E5.12:ev_off` | `01 0.441 [0.412,0.471]`, `13 0.572 [0.533,0.611]`, `06 0.447 [0.407,0.486]`, range 0.096-0.725 all suppressive; `A>1 with CI excluding 1` required for amplification — none meets |
| Obs 6.8 | Controls causal degradation intact vs REC-OFF `g_rec→0` vs Matched `(μ,σ)` N2 S4 `mean_{4}` | Intact `I_Vm0.031-0.460` → OFF `0.019-0.067` → Matched `0.010-0.050` | `bits` + `mV/native` | 3 finalists ×4 seeds | `E5.11:ev_intact/ev_off/ev_matched` `E5.12` | `01 intact I_Vm0.031 G1.41 → off 0.023 G4.74 → matched 0.010 G10.5 A0.404` ; `13 0.208 G0.569 →off 0.067 G5.61 →matched 0.050 G8.33 A0.60` ; `06 0.460 G1.12 →off 0.019 G6.23 →matched 0.015 G8.27 A0.606`; **both off and matched lose I** (`A>B and A>C`) |
| Obs 6.9 | N1 revised Pareto 8-slot manifold | 8 survivors `01,13,06,20,22,11,15,08` dom0 7 | — | 8/24 | `E5.10:selected_N1_ids, ranking_detail` | `η0.023-0.612 B0.757-0.819 χ meanCV0.295-0.324 ρ0.352-0.447 G0.07-4.98 I_Vm0-0.45 I_sp0-0.56`; includes high-G poor-spike (20 G1.42 I_sp0, 08 G2.15 I_sp0) + good-sync weak-info (15 ρ0.352 G0.07 I≈0) |
| Obs 6.10 | N2 S4 `3×S4 mean_{4}` Q preserved + ISN `+20% PV+SST` and `PV` alone | 3/3 hard PASS 4/4 seeds; ISN `NON_ISN_LIKE` 3/3 both tests | `Hz` | 3×4 seeds (12) + 24 ISN pert | `E5.11:hard_all_seeds, mean_ybar, isn_characterization` `E5.12:per_seed` | `01 bar5.52 ρ0.423 P0.21 max2.11 Fano0.95 efrac0.737 PV+SST Δr_I+7.59 Δr_E−0.013` `13 bar5.48 ρ0.435 η0.068 G0.569 I_Vm0.208 p0.05 I_sp0.227 p0.03` `06 bar5.68 ρ0.396 efrac0.716 I_Vm0.460 p<0.01 G1.12`; ISN `Δr_PV+11-11.6 Δr_I pooled+7.4-7.6 non-paradoxical` for all 3 |
| Obs 6.11 | Current-space maps (η_rec,B_EI,χ_fluct) 3-D size `I_sp` vs `G` + 2-D η vs B_EI color χ + current traces schematic | 6 figures lineage N0→N1→N2; primaries not co-located `Δη0.04 ΔB0.04` | — | 24 N0 + 8 N1 + 3 N2 + C019/C021 refs | `E5.13-E5.17` `E5.11 N2_summary` | View1 size `20+600·I_sp` View2 `20+80·G`; `01 η0.023 B0.779` vs `13 η0.064 B0.819` separated, `06 η0.229 B0.716` spans quadrants; traces `I_E/I_I/I_net/V_m` A vs B `AAAB/BBBA` `ENERGY_A 923.024/921.94 parity≤5%` RF 32×32 field8 σ1.8 |
| Obs 6.12 | Successive fidelity budget | N0 24×S1 (24 sims) + N1 8×S2 (16) + N2 3×S4 (12) =52 sims search (+ up to 24 controls) | sims | `S_1[0] S_2[0,1] S_4[0,1,2,3]` | `E5.3 agsdr_local_freeze.json` `E5.9-E5.12` | `2000 ms spont dt0.1 T20000 + 1000 ms evoked T10000 G759 30MiB dt0.1 delays [20,80,120] grouped 0.6MiB passivity ΔVm0 Δsp0` `RuntimeConfig enable_hdp=False` |

_Figures:_ `results/fg11_RUN_update/current_space_3d_Isp.png`, `results/fg11_RUN_update/current_space_3d_G.png`, `results/fg11_RUN_update/current_space_2d_eta_B.png`, `results/fg11_N2_ISN/current_space_3d_Isp_N2.png`, `results/fg11_N2_ISN/current_space_3d_G_N2.png`, `results/fg11_N2_ISN/current_space_2d_eta_B_N2.png` + `results/fg11_RUN_update/current_traces_schematic.png` (`I_E/I_I/I_net/V_m` for `A/B`)

## 7. Interpretation

Observation ≠ interpretation. Each claim cites supporting observation ID(s) from §6.

| # | Interpretation | Supporting observations | Confidence |
|---|----------------|-------------------------|------------|
| 7.1 | Current-based Gen-2 architecture admits locally qualified operating manifold that transmits L4→L2/3 stimulus information `∃θ` — at N0 S1 `01/13` both `I(S;V)>0 ∧ I(S;spikes)>0` with `Q_C019` preserved (2/24) establishes capability, not structural impossibility | `Obs 6.1`, `Obs 6.2`, `Obs 6.5`, `Obs 6.6` | `HIGH` |
| 7.2 | Phenotype is operating-regime indexed by `θ_impl=[g_rec,g_fastI,g_dendI,g_bg]` not scalar `|bar−7.5|`: success depends on `η_rec` (‖I_rec‖/‖I_total‖), `B_EI` (PV vs SST separate), and `χ_fluct` (σ/μ+V_m variance+CV) — exactly the 4 levers searched orthogonally | `Obs 6.3`, `Obs 6.4`, `Obs 6.9`, `Obs 6.11` | `HIGH` |
| 7.3 | `G_{I→V}` is the `I→V_m` operator outcome: `0.013 (C021_s0 baseline) → 4.98/2.79` in primaries shows shunt relieved via PV perisomatic (`E→PV1.70↔1.36-2.04 PV→E1.30↔1.04-1.56` `g_fastI0.83-1.07`) and SST dendritic + `I_bg(λ)` mean-controlled `g_bg0.85-0.94 χ0.269→0.349` rather than weight magnitude alone | `Obs 6.3`, `Obs 6.4`, `Obs 6.5` | `MED` |
| 7.4 | Complementary dissociation (Vm-only `06 I_Vm0.45 I_sp0.009` vs spike-only `22 I_sp0.56 I_Vm0` vs high-G poor-spike `08 G2.15 I_sp0` vs good-sync weak-info `15 ρ0.352 G0.07`) proves manifold coherence vs single accident — N1 8-slot + N2 finalists span different quadrants of (η,B_EI,χ) yet all show partial transfer | `Obs 6.9`, `Obs 6.10`, `Obs 6.11` | `HIGH` |
| 7.5 | S4 `mean_{4}` dilution (`01 I_Vm0.138→0.031`, `13 0.256→0.208`) does not eliminate manifold — `06` Vm-only strengthens to `0.460 p<0.01` strongest, `13` retains both `I>0`, heterogeneity jitter `σ0.10 builder.py:39 lognormal CV1.5 builder.py:142` explains variance; ranking is on `mean_{4}` not best seed per `gen2_C019` | `Obs 6.2`, `Obs 6.10` | `HIGH` |
| 7.6 | Suppressive regime `A_rec<1` with CI excluding 1 in all 24 N0 and 3 S4 (0.404-0.606) is regime-consistent not failure — cortex can suppress overall `‖ΔY‖` while enhancing information via transformation (gain/shunt/timing/selectivity); amplification `A>1` not required per authorization | `Obs 6.7`, `Obs 6.8` | `HIGH` |
| 7.7 | Recurrent computation (structure) vs total-input magnitude distinguished by matched control: `06 I_Vm0.46→0.02` and `13 0.208→0.067` loss on **both** OFF and matched (which preserves `μ_total3.405 σ0.94`) implies `I_rec` correlation structure causal, not just `μ`/`σ` operating point | `Obs 6.8` | `MED` |
| 7.8 | ISN characterization `NON_ISN_LIKE` for 3/3 finalists under `+20%` drive to `PV+SST` and `PV` alone (`Δr_I pooled +7.4-7.6 Hz, Δr_PV +11-11.6, Δr_E~0`) with caveat heterogeneous typed-E `M2 RS70/CH20/EFS10` + small inhibitory fraction (~6% network) means simplistic test can be misleading and is **not hard criterion**; Pareto ranking unchanged | `Obs 6.10` | `MED` |
| 7.9 | Finding is supplementary HDP-OFF only — demonstrates capability but does **not** claim canonical `delayed+HDP` Gen-2 until FG-12 condition 6 (U01b `jaxfne/_model_simulate.py:280` fix + `evidence-audit`); fast-state equivalence from FG-09b (`Δbar<0.05Hz ΔVm<0.02mV passivity0`) makes assay identifying for fast `spikes/V_m` but not substitute for `H/Θ` learning | `Obs 6.12` + §3.2 | `HIGH` |

## 8. Negative / insufficient results

Bounded vocabulary: `NEGATIVE_INSTANCE` | `NULL_RESULT` | `INSUFFICIENT_POWER` | `INCONCLUSIVE` | `RESOURCE_BOUNDARY` | `BLOCKED`; `FALSIFIED` reserved for `L0`–`L5` criterion failures.

| # | Claim | Verdict | Level / threshold | Notes |
|---|-------|---------|-------------------|-------|
| 8.1 | Canonical `delayed+HDP` grouped `I[t,G]` exposing realized edge-current at deployed delays `[20,80,120]` | `BLOCKED` | `jaxfne/_model_simulate.py:280 guard + emitters.py:714 no record_grouped_current for non-HDP is HDP-only (_pipeline.py:395 HDP-only)` | This session HDP-OFF only; U01b remains `MISSING_CAPABILITY` per `E5.2` `fg01:21` |
| 8.2 | All `θ_impl` show `A_rec>1` amplification with CI excluding 1 (would Claim amplification) | `NEGATIVE_INSTANCE` | `threshold A_rec>1 CI excludes 1` — observed 0/24 and 0/3 S4; all `A_rec 0.096-0.725 <1` `CI<1` suppressive | Per authorization `A_rec` **measured not hard**; `A_rec>1` not required — suppression with information enhancement is valid transformation; comparison vs off/matched shows recurrent mediation instead |
| 8.3 | Single-seed `01/13` alone as canonical promotion without controls or S4 replication | `INCONCLUSIVE` | `requires 8×S2 y_bar + 3×S4 mean_{4} + off/matched per authorization §3.5` | N1 revised requires 8-slot manifold + 3 controls; this report correctly demands S4 `mean_{4}` and off/matched — met for 3 finalists but only supplementary |
| 8.4 | Before FG-11, `I_syn` could lose information (upstream syn failure hypothesis) | `NEGATIVE_INSTANCE` | `gate I_syn carries 6/6 d>240` — FG-09b observed 0/6 loss | Not upstream syn failure; operator is `I→V_m` |
| 8.5 | Full vertical transmission rescue invariant (no bottleneck) | `NULL_RESULT` | `threshold `acc>0.703 ∧ d>0.5` at projecting layer invariant` — observed heterogeneous: 2/24 primary vs 22/24 partial/ns, `G 0.013→4.98` but only 2 carry both readouts | Consistent with graded shunt/low-pass `G I→Vm 0.013-4.98` not binary |
| 8.6 | ISN paradoxical response `Δr_I<−0.10 ∧ Δr_E<−0.05` on `+20% PV+SST` (would Claim ISN) | `NEGATIVE_INSTANCE` (characterization) | `threshold Δr_I<−0.10 ∧ Δr_E<−0.05` — observed `Δr_I+7.4-7.6 >+0.05 non-paradoxical` 3/3 | Characterized `NON_ISN_LIKE` not hard; heterogeneous caveat `UNRESOLVED` nuance per authorization |
| 8.7 | Formal `η_rec` L2 signed/absolute `Efrac` and `Corr(I_private, I_{L4→L23}[t+τ]) τ10-20ms` partitioned per-neuron | `INSUFFICIENT_POWER` | `requires _model_simulate HDP+delay fix + observables.py:161 drive_array emitters.py:477` for per-step motif partition | Proxy `η_proxy=|ΔI|/σ0.94` provided; formal per-step `η_rec` gated until FG-12 HDP path; FG-11 uses proxy + grouped collapsed G |
| 8.8 | `H/Θ` contribution at deployed delays beyond fast-state equivalence | `INCONCLUSIVE` | `HDP OFF trajectory H∈[0.1,10] bounds vs fast Δbar<0.05 ΔVm<0.02` only | FG-09b equiv shows identifying for fast but not `H/Θ` learning; canonical needs `U01b` fix |

_If none:_ not applicable — see table.

## 9. Unexpected findings

| # | Finding | Why unexpected | Follow-up priority |
|---|---------|----------------|-------------------|
| 9.1 | Only 2/24 N0 S1 achieve both `I>0` (`p<0.05`) despite `η` spanning 0.023-0.915 and `G` 0.007-4.98 — success is sparse in `Θ_admissible` | Prior uniform `g_vertical×1.8-5` and `Ne16→54 7×` failed uniformly implying lever wrong; FG-11 shows lever exists but only 8% density — operating-regime narrow, not broad rescue | `high` — S4 replication already done; FG-12 must not expand box `[0.70,1.30]/PV[0.80,1.20]` without causality evidence |
| 9.2 | Vm-only vs spike-only dissociation: `06 I_Vm0.450 p0.005 I_sp0.009 ns` vs `22 I_sp0.560 p0.004 I_Vm0` at overlapping `η0.23-0.40 B0.757-0.758 χ0.295-0.324` | Expected `G` ↑ to move both readouts together (`G_{I→V}→spikes`); instead high-G poor-spike `08 G2.15 I_sp0` shows `G` without information — gain necessary not sufficient | `high` — current traces `I_E/I_I/I_net/V_m` for A vs B needed per `N1_revised:3` |
| 9.3 | All `A_rec<1` (suppressive) with CI excluding 1 — no amplification despite `I` and `G` improvements | Literature prior expects recurrent amplification `A>1.2` as causal signature; observed `A 0.404-0.606` suppressed `‖Δ_intact‖<‖Δ_off‖` yet `I` requires recurrence → transformation not amplification | `med` — in §12 propose measuring selectivity/timing vs magnitude under matched control |
| 9.4 | ISN `NON_ISN_LIKE` for all 3 S4 finalists (`Δr_I+7.6 Δr_E~0`) despite high `g_rec1.25` | Expected high recurrent `g_rec1.25` to push toward ISN; observed non-paradoxical even on PV+SST 6% network fraction — heterogeneity `M2` + `lognormal CV1.5` may mask or regime is fluctuation-driven (`χ σ/μ0.277`) at this `I_bg(λ)` | `low` — characterization not hard; need larger inhibitory fraction test before claiming non-ISN architecture |

_If none beyond above:_ not applicable.

## 10. Tool / workflow friction

| Friction | Cause | Workaround | Permanent repair? |
|----------|-------|------------|-------------------|
| Delayed+HDP grouped current `MISSING_CAPABILITY` at `jaxfne/_model_simulate.py:280` + HDP seam lacks `record_grouped_current` | Upstream jaxfne HDP vs delay guard + `_pipeline.py:395` HDP-only forwarding | Use HDP-OFF delayed non-HDP grouped `I[t,G]=segment_sum(w·syn,gids,G)` via `RuntimeConfig(enable_hdp=False, record_grouped_current)` as MECHANISTIC/SUPPLEMENTARY with fast-state equivalence check (`Δbar<0.05Hz ΔVm<0.02mV passivity0`) | `HDP+delay grouped current support in jaxfne (Option A/B/C per jaxfne_issue_delayed_edge_current.md:200) — proposed in §12` |
| Window `B=[−50,0)` not in `Simulation(duration 1000)` `to_array`, so `B_proxy=[0,50)` early `p1` pre-volley | Simulation starts at `p1 0`; absolute `fx −500..0` clipped | Report `mean(W)−mean(B_proxy) + raw mean(W)` + flag as supplementary limitation; freeze or extend `duration 4624 ms` (`FULL 46240 steps exec_A-D`) for canonical | `DOCUMENTATION — cross-ref simulation vs absolute clock in freeze` |
| ISN characterization requires substantial inhibitory fraction; `+20% PV+SST ≈6%` network gives non-paradoxical even if architecture ISN | Heterogeneous typed-E `M2` + `lognormal CV1.5` + sparse `σ0.08` + small perturbed subset masks paradoxical; operating point `χ` near rheobase `μ≈3.4 μ_rheo≈3.5` | Characterize as `NON_ISN_LIKE` under this protocol (not hard), note heterogeneous caveat, keep `A_rec/I` not gated by ISN | `JOMISSION_TOOL — larger-fraction perturbation sweep + heterogeneity-aware ISN test (proposed §12)` |

_If none:_ not applicable.

## 11. Learned lessons

| Lesson | Trigger | Cause | Repair | Evidence | Scope | Confidence | Persistence |
|--------|---------|-------|--------|----------|-------|------------|-------------|
| Hard Q must protect distributional tail `P(CV>0.5)≥0.15 / max≥1.5` not scalar `meanCV≥0.5` | Single RS `σ0.40` variance-matched still fails vs typed `CH c-50/d2` `p<1e-6` → typed pseudogenome justified | Scalar mean hides heterogeneity; tail captures `CH` burst `CV0.07→0.46` | Enforce hard `P≥0.15 max≥1.5` before any transfer claim `gen2_gates.py:139` | `Obs 6.1` `scratch/gen2_C018_receipt.md` | `project` | `HIGH` | `YES` |
| LHS `D*≈O(1/N)` vs uniform `O(1/√N)` matters at `d=4 N0=24` — stratified seed0 guarantees coverage of `g_bg×g_fastI` corner where `η_rec×G` tradeoff lives | LHS ensures one sample per marginal quantile; uniform would leave ~30% holes at `d=4` `vol` | Uniform clumping `O(1/√N)` vs LHS `O(1/N)` low-discrepancy | Use deterministic `rng.default_rng(0) cut linspace(0,1,25) perm jitter` | `E5.9` `scratch/fg11_supplementary_search.md:2.1` `jacobian_estimate.json g_bg −0.162 on ρ` | `project` | `HIGH` | `YES` |
| Pareto feasibility-first on `[I_Vm,I_sp,G,A,J_ρ,J_EI,J_local]` forbids scalar `|bar−7.5|` collapsing tradeoff; `E<PV` is soft `J_EI` not hard `H10` | Scalar `|bar−7.5|` dominated by `ρ0.45` vs `I_Vm0.45` tradeoff; `E<PV` hard would abort 0/24 spuriously | Scalar collapses `η` vs `G` vs `A` ; `E<PV` not universal cortical law | Pareto `dom` on 7 dims + `J_EI soft [0.15,0.60] hard (0.05,0.95)` `gen2_gates.py:156` | `Obs 6.9` `E5.10` `scratch/fg11_N1_revised.md:0` | `project` | `HIGH` | `YES` |
| HDP-OFF delayed grouped `I[t,G]=segment_sum(w·syn,G)` is identifying for fast dynamics (`Δbar<0.05Hz ΔVm<0.02mV passivity0`) even though `H` differs — slow plasticity vs fast drive dissociation | Fast-state equivalence PASS across 6/6 C/seed `fg09b` including `del vs zero ±0.03Hz` | `H/w` slow (`μ_total3.405 builder.py:44`) vs fast `spikes/V_m` via `emitters.py:714` not `H`-dependent; already computed `emitters.py:840` | Use delayed non-HDP grouped for mechanistic `I→V_m` localization; keep canonical `BLOCKED` labeled supplementary | `Obs 6.12` `scratch/fg09b_HDPOFF_bridge_raw.json` `fg01:33` | `project` | `HIGH` | `YES` |
| Current→V_m is heterogeneous graded shunting (`G 0.013-4.98`) not binary FAIL; single-seed risks overclaim | `C021_s0 G0.013 invariant` vs `C019_s0 G0.851` vs `FG11_N0_01 G4.98` with `Vm` small absolute `0.13-0.46mV` but informative | Shunt/low-pass via `g_fastI/g_dendI` (`a/b builder.py:106,116 emitters.py:55`) + `I_bg(λ) χ` + `E/I` duty cycle | Require full `transfer_matrix.csv` all `C/seed × G` and `η_rec` L2 before typing | `Obs 6.3` `Obs 6.11` `scratch/fg09b:105` | `project` | `HIGH` | `YES` |
| `A_rec` is measured phenotype not hard `>1.2` — suppressive `A<1` (`CI<1`) with information gain is valid recurrent mediation, not failure | All 24 `A_rec0.096-0.725 CI<1` yet `I` and `G` require recurrence via off/matched loss | Cortex can amplify or suppress depending on `I_bg(λ)` regime `g_bg χ Efrac` | Make `A_rec` measured; allow loss of `I` + altered gain/timing/ρ/selectivity under matched while `χ` preserved as success | `Obs 6.7` `Obs 6.8` `scratch/fg11_N2_ISN.md:2.3` | `project` | `HIGH` | `YES` |
| ISN `NON_ISN_LIKE` under `+20% PV` cannot be taken as proof of non-ISN architecture in heterogeneous network — requires substantial I fraction | Heterogeneous `M2` + `lognormal CV1.5` masks paradoxical; small `PV+SST ≈6%` insufficient | Perturbed fraction `13/400≈3%` `PV` alone, operating point near rheobase | Characterize as `ISN_LIKE|NON_ISN_LIKE|UNRESOLVED` not hard; note large-fraction caveat | `Obs 6.10` `scratch/fg11_N2_ISN.md:9.2` | `project` | `MED` | `YES` |
| `B_proxy` surrogate must be flagged when `B` outside array | `Simulation(duration 1000)` starts at `p1 0`, `B=[−50,0)` not in array | Freeze vs simulation absolute-clock offset | Use `B_proxy[0,50)` early `p1 + raw mean(W)` and document; fix freeze or duration `4624 ms` for canonical | `Obs 6.12` `E5.5-E5.6` | `project` | `MED` | `NO` |

_Persistence_ ∈ `YES` | `NO`; `Confidence` ∈ `LOW` | `MED` | `HIGH`. `YES` = propose promotion to harness/project law.

## 12. Harness / tool proposals

Type ∈ `PROJECT_RULE` | `TEST/GATE` | `JOMISSION_TOOL` | `JAXFNE_UPSTREAM` | `DOCUMENTATION` | `VISUALIZATION` | `AGENT_WORKFLOW`

| # | Type | Target | Rationale | Evidence |
|---|------|--------|-----------|----------|
| 12.1 | `JAXFNE_UPSTREAM` | `jaxfne/_model_simulate.py:280,600,1112` + `jaxfne/emitters.py:714` + `jaxfne/_pipeline.py:395,538` — delayed+HDP grouped current | Unblock canonical `I[t,G]` at delays `[20,80,120] G≈16 0.6MiB` to allow `η_rec` L2 signed/absolute `Efrac_corr` and `I(S;I)` without HDP-OFF surrogate; required for canonical FG-12 condition 6 | `Obs 6.1` `§8.1 BLOCKED` `E5.2` |
| 12.2 | `TEST/GATE` | `mechanistic_replay_freeze.md:2.2` `W=[100,531) B=[−50,0) ε=1.0/0.01/1.0` frozen + `B_proxy` flag | Add gate that `B=[−50,0)` requires `fx −500..0` in array or forces `B_proxy` with explicit flag + `ε` not result-responsive; prevents silent baseline bias | `Obs 6.12` `§10 B_proxy` |
| 12.3 | `JOMISSION_TOOL` | `recording/observables.py:161 partition_currents_by_motif` + `emitters.py:477 drive_array` for `η_rec` & `I_E/I_I/V_m(t)` traces | Automate `η_rec=‖I_rec‖/‖I_total‖` signed/absolute/L2 `Efrac` + `I_E(t)/I_I(t)/I_net(t)/V_m(t)` A vs B traces per candidate via grouped `I[t,G]` motif partition (already `E5.9 G759 30MiB`) | `Obs 6.4` `Obs 6.11` `§8.7` |
| 12.4 | `JOMISSION_TOOL` | `scratch/fg11_N2_ISN_runner.py:apply_isn_perturbation` + `builder.py:106,116,127` larger-fraction ISN + heterogeneity-aware test | Beyond `+20% PV` add titrated `+20/40/80%` PV+SST+VIP combined (≥30% inhibitory) and report per-class `Δr` with heterogeneity `M2` caveat; paradoxical may need large fraction per authorization | `Obs 6.10` `§9.4` |
| 12.5 | `VISUALIZATION` | `scratch/plot_N2_ISN.py` + `recording/observables.py:81` — current-space manifold maps | Standardize `(η_rec,B_EI,χ_fluct)` 3-D with size `I(S;spikes)` vs `G_{I→V}` + 2-D `η vs B_EI color χ` lineage `N0→N1→N2` + `C019 y_bar4`/`C021 G0.013` refs as frontier visualization contract | `Obs 6.11` `E5.13-E5.17` |
| 12.6 | `DOCUMENTATION` | `scratch/fg11_supplementary_search.md §10` — supplementary labeling + firewall | Enforce `MECHANISTIC/SUPPLEMENTARY/HDP-OFF` label on every FG-11 artifact + `C_next FORBIDDEN until FG-12` + `W4b/W4c/W5 gated, M-06 parallel` until 6-condition reproduction; prevents promotion leakage | `Obs 6.1` `§8.1` `E5.5` |

_If none:_ not applicable.

## 13. Scientific state transition

| Field | Value |
|-------|-------|
| **Before** | `W4a MECHANISM_OBSERVABILITY_BLOCKER` sealed: Visual→V1 L4 PASS `Δ169 d803 acc1.00` but V1 L4→L2/3 FAIL `Δ0.02 d0.13 acc0.542` invariant across 3 interventions (`g_vertical×1.8-5`, C020 `Ne16→54 C0.444→0.704 7×`, C021 `PV/E0.73/0.64 Jaccard0.454`); bottleneck localized to `spikes_{L4_E}→I_{L4→L23}→V_{m,L23}` at `[20,80,120]` but `I[t,G]` not observable canonical — FG-09b HDP-OFF bridge supports `I→V_m` leading operator (`I 6/6 d>240 I0.67 while Vm invariant C021_s0 G0.013 fast-equiv PASS`) but no operating manifold demonstrated |
| **After** | **FG-11 supplementary HDP-OFF operating manifold discovered:** `24×S1 LHS D*≈O(1/N)` successive `24×1→8×2→3×4=52 sims` (+ controls) `θ_impl=[g_rec,g_fastI,g_dendI,g_bg]` `4D [0.70,1.30]/PV[0.80,1.20] VIP→SST1.0` LHS seed0; **N0 24/24 hard Q_C019 PASS** `bar4.77-5.08 ρ0.352-0.447 P≥0.15 max≥1.5 Fano0.95 efrac0.757-0.819` `E<PV` soft, **2/24 primary `I(S;V)>0 ∧ I(S;spikes)>0` with `G`↑ (`01 G4.98 I_Vm0.138 I_sp0.463`, `13 G2.79 I_Vm0.256 I_sp0.321`)** `Q` preserved — architecture **capability** not structural impossibility; **N1 revised 8-slot manifold** `01,13,06,20,22,11,15,08` with 3 controls (intact/`REC-OFF` `g_rec→0`/matched `μ,σ` per `builder.py:44 I_bg(λ)`) and current-space map `(η_rec,B_EI,χ)` size `I(S;spikes)` vs `G_{I→V}`; **N2 3×S4** `01,13,06` `mean_{4}` `Q` preserved `bar5.52-5.68 ρ0.396-0.435 efrac0.716-0.778` `G1.41/0.569/1.12` `I_Vm0.031-0.460` `I_sp0.165-0.227` ISN `NON_ISN_LIKE` 3/3 both `PV+SST+20%` and `PV+20%` (`Δr_I+7.4-7.6 non-paradoxical`) with heterogeneous caveat — coherent manifold bridges primaries to complementary `η0.024-0.229 B0.716-0.778 χ0.327-0.349`; **regime is suppressive `A_rec0.40-0.61 CI<1` not amplification** yet recurrent mediation via off/matched loss (`I` intact vs off/matched degrades) |
| **Unlocked** | Supplementary evidence that **operating regime (`g_rec` + PV/SST shunt `g_fastI/g_dendI` + `I_bg(λ)` `g_bg` mean-controlled `χ`) can be moved to make `η_rec` and `G_{I→V}` support laminar transmission** — `FG-10 DOF` measurement and `FG-11` search protocol now verified; current-space `(η_rec,B_EI,χ)` mapping with `I_sp/G` size + `I_E/I_I/V_m(t)` traces is available for ranking; successive fidelity `mean_{4}` + off/matched controls distinguish structure vs magnitude |
| **Still gated** | **Canonical `GEN2_C0xx` promotion (`C_next FORBIDDEN until FG-12`)** — requires 6-condition reproduction including condition 6 `delayed+HDP` at `[20,80,120]` once `U01b` solved; `W4b/W4c/W5` higher chain gated; `H/Θ` via `M-06` parallel (`h_state.py:28 5× vs emitters.py:3703 scalar`); formal `η_rec` L2 signed/absolute per-step partition and `Corr(I_private, I_{L4→L23}[t+τ])` gated |
| **Invalidated** | Hypothesis that **current-based Gen-2 architecture is structurally incapable** of L4→L2/3 transfer — falsified at supplementary level (`∃θ` with both readouts `>0`); hypothesis that assay is non-identifying due to HDP disabling — falsified (`fg09b` fast-equiv PASS `Δbar<0.05Hz ΔVm<0.02mV passivity0`); scalar `|bar−7.5|` or narrow top-2 ranking sufficient — rejected (requires 8-slot manifold + Pareto feasible-first) |
| **New uncertainty** | Why S4 `mean_{4}` dilutes `01 0.138→0.031` while `06` strengthens to `0.460 p<0.01` — heterogeneity jitter `σ0.10 lognormal CV1.5` vs operating point `χ σ/μ0.277 g_bg` near rheobase; true `η_rec` vs proxy `|ΔI|/σ0.94` offset (proxy `0.023→0.24` vs L2 `0.003-0.229`); ISN large-fraction test not yet run; HDP plasticity `H Θ` learning contribution at deployed delays still unobserved |

## 14. Next action

- **Primary (highest-value):** **FG-12 canonical reproduction** — rerun best `θ*` manifold survivors (`FG11_N0_01,13,06` + complementary `22/20` for high-G control) at **canonical `delayed+HDP`** (`RuntimeConfig(enable_hdp=True, hdp_params={record_edge_current, edge_group_ids, grouped_num_segments G≈16})` via `emitters.py:714` + `_model_simulate.py:280` + `_pipeline.py:395,538` fix Options A/B/C `jaxfne_issue_delayed_edge_current.md:200`) at deployed `delays [20,80,120] dt0.1 T10000` `W[100,531) B[−50,0)` true baseline (`fx −500..0`, once `duration 4624 ms FULL` or simulation extended) with same `RF 32×32 ENERGY_A 923.024/921.94 parity≤5%`, `KSG k=3 shuffle500` + off/matched controls same `S4 [0,1,2,3] mean_{4}` + `evidence-audit` hash-verify. Why: FG-11 stays `SUPPLEMENTARY_TRANSFER_CANDIDATE` by rule; only condition 6 `CANONICAL delayed+HDP reproduction` allows `GEN2_C0xx` promotion (§8.1 `BLOCKED` → unblocked). **Ready predicate:** upstream jaxfne fix exposing `edge_current_trace (n_steps,n_edges)` for `enable_hdp=True + delays>0` landed + `evidence-audit/SKILL.md HARNESS_MANIFEST.json` hash-verified — currently `MISSING_CAPABILITY`, ready when `U01b` PASS; meanwhile keep builder frozen `be9b96ab` `MOTIF_GAIN VIP b0.20 SST b0.21 PV×1.7 M2 RS70/CH20/EFS10 lognormal CV1.5 VERTICAL σ0.12 max40 tonic3.0 Poisson2kHz delays[20,80,120]`.

- **Secondary (optional):** Expand `I_E(t)/I_I(t)/I_net(t)/V_m(t)` biophysical traces for A vs B per N2 finalist via `partition_currents_by_motif observables.py:219 per_motif Efrac` grouped `G759` already available in `E5.12` + larger-fraction ISN titration (`+40/80% PV+SST+VIP`) to harden `NON_ISN_LIKE` vs `UNRESOLVED` under heterogeneity — runs at same HDP-OFF budget, no builder mutation.

## 15. Progress score

| Field | Value |
|-------|-------|
| Current | 74 |
| Previous | 74 |
| Delta | 0 |
| Reason | **FG-11 supplementary** demonstrates architecture capability with operating manifold: N0 24×S1 LHS D*≈O(1/N) successive N0 24×1→8×2→3×4=52 sims, N0 24/24 hard `Q_C019` PASS (`bar4.77-5.08 ρ<0.45 P≥0.15 max≥1.5 Fano0.95`), 2/24 primary `I(S;V)>0 ∧ I(S;spikes)>0` with `Q` preserved (`01 I_Vm0.138 I_sp0.463 G4.98`, `13 I_Vm0.256 I_sp0.321 G2.79`) vs baseline `C021_s0 G0.013`, N1 revised with `E<PV` not hard + 3 controls (intact/`REC-OFF`/`matched` `μ_total3.405 σ0.94` `I_bg(λ)`), current-space map `(η_rec,B_EI,χ)` size `I(S;spikes)` vs `G_{I→V}`, N2 3×S4 mean_{4} + ISN `NON_ISN_LIKE` 3/3 both tests — **successful regime demonstrates operating-regime problem not structural impossibility, but classification remains SUPPLEMENTARY/HDP-OFF until FG-12 canonical `delayed+HDP` (U01b BLOCKED at `jaxfne/_model_simulate.py:280`) — no new verified canonical capability; supports FG-12 opening but does not advance sealed frontier** |

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
