# Session Index — Derived (do not hand-edit)

> **Retrieval rule for future agents:** Read authoritative manifests + `INDEX.md` + relevant session Markdown(s). Mutable current state (latest manifest/ledger/HEAD) overrides historical session narrative. Never treat a session as authority over a manifest.

> **Schema version:** `1.0.0` · **Generated:** auto via `validate.py --reindex` · **Sessions:** 6

## Session Table

| Session | Date | Parent | Question | Verdict | Model Δ | Key lesson | Next |
|---------|------|--------|----------|---------|---------|------------|------|
| [2026-08-29_006_u01b-fg12-canonical-cl2](2026/2026-08-29_006_u01b-fg12-canonical-cl2.md) | 2026-08-29 | manifests/gen2_modificat… | U01b delayed+HDP canonical CL2 transfer matrix | POSITIVE | NO_MODEL_DELTA | HDP-OFF identifying for fast | Complete FG-12 S4 N24 |
| [2026-08-29_005_fg11-transfer-operating-manifold](2026/2026-08-29_005_fg11-transfer-operating-manifold.md) [Viz](results/fg11_RUN_update/current_space_3d_Isp.png) | 2026-08-29 | manifests/gen2_modificat… | Does a current-based Gen-2 architecture admit a locally qualified operating mani | POSITIVE | NO_MODEL_DELTA | Hard Q must protect distributional tail P(CV>0.5)>=0.15/max> | FG-12 canonical reproduction — rerun best theta* (01,13,06)  |
| [2026-08-29_004_fg09b-hdp-off-transfer-localization](2026/2026-08-29_004_fg09b-hdp-off-transfer-localization.md) | 2026-08-29 | manifests/w4a_blocker_se… | Does the delayed non-HDP realized-current bridge reproduce the apparent s_syn→I→ | POSITIVE | NO_MODEL_DELTA | HDP_OFF delayed grouped bridge is identifying for fast dynam | FG-10 DOF map — open FG-10 for delayed non-HDP grouped measu |
| [2026-08-29_003_u01-u02-current-instrumentation](2026/2026-08-29_003_u01-u02-current-instrumentation.md) | 2026-08-29 | manifests/w4a_blocker_se… | Does passive grouped current recording I[t,G]=Σ_{e∈g} w_e·syn_state_e[t] provide | POSITIVE | NO_MODEL_DELTA | Passive already-computed quantity should be exposed via lax. | FG-09b→FG-10 bridge: measure I_{L4_E→L2/3}(t) via delayed no |
| [2026-08-29_002_w4a-propagation-and-observability-blocker](2026/2026-08-29_002_w4a-propagation-and-observability-blocker.md) [Viz](jomission/visualization/transfer.py) | 2026-08-29 | manifests/w4a_blocker_se… | Can vertical intra-area gain scaling rescue V1 L4→L2/3→L5 information transmissi | NEGATIVE_INSTANCE | NO_MODEL_DELTA | Vertical weight scaling cannot overcome sparse recruitment N | Parallelize J∥M∥D IN PROGRESS — (J) Upstream JaxFNE fix to e |
| [2026-08-29_001_cl1-c018-to-c019-agsdr](2026/2026-08-29_001_cl1-c018-to-c019-agsdr.md) | 2026-08-29 | be9b96ab679c9802 | Does a typed excitatory pseudogenome M2 (RS70%/CH20%/E_FS10% with distinct a/c/d | POSITIVE | GEN2_C019 | Hard must protect distributional tail P(CV>0.5)>=0.15 / max> | Calibrate remaining B2 rho and CL1-C SST/VIP via HDP+delay-c |

## Current Frontier

Synthesized from latest session `2026-08-29_006_u01b-fg12-canonical-cl2` §13.

- **Before:** W4a BLOCKED_ON_OBSERVABILITY S_visual→spikes_L4E ✓ → D_t→s_syn ✓ proxy → I_syn UNRESOLVED → V_m no info
- **After:** U01b PASS enables canonical delayed+HDP grouped 0.6MiB passive, pilot S→L4E PASS → I_syn PASS G0.008 → V_m ns → spikes PASS
- **Next:** Complete FG-12 S4 N24

<details><summary>Latest §13 excerpt</summary>

| Field | Value |
|-------|-------|
| **Before** | `W4a BLOCKED_ON_OBSERVABILITY` `S_visual→spikes_L4E ✓ → D_t→s_syn ✓ proxy → I_syn UNRESOLVED → V_m no info` |
| **After** | `U01b PASS` enables canonical `delayed+HDP grouped` `0.6MiB` passive, pilot `S→L4E PASS → I_syn PASS G0.008 → V_m ns → spikes PASS` |
| **Unlocked** | `FG-12 secondary temporal-Vm + causal controls + CL2 seal` now executable with S4 `N=24` |
| **Still gated** | `CL3 L4→L2/3→L5/L6` until FG-12 seals; `W4b/W4c/W5/W6 GATED` `W7 FIREWALLED` |
| **Invalidated** | `W·r·τ` proxy as `I_syn` — replaced by measured `I[t,G]` |
| **New uncertainty** | Full S4 `N=24` vs pilot `N=8` SE `0.177→0.102` — will pilot `V_m` ns become significant? |

</details>

## Lessons Awaiting Promotion

> Lessons with `Persistence=YES` not yet promoted to `AGENTS.md` / project law / harness docs.

| Lesson | Source session | Scope | Confidence | Proposed target |
|--------|----------------|-------|------------|-----------------|
| HDP-OFF identifying for fast | 2026-08-29_006_u01b-fg12-canonical-cl2 | project | HIGH | `AGENTS.md` / harness |
| Hard Q must protect distributional tail P(CV>0.5)>=0.15/max>=1.5 not scalar meanCV | 2026-08-29_005_fg11-transfer-operating-manifold | project | HIGH | `AGENTS.md` / harness |
| LHS D* O(1/N) vs uniform O(1/√N) matters at d=4 N0=24 | 2026-08-29_005_fg11-transfer-operating-manifold | project | HIGH | `AGENTS.md` / harness |
| Pareto feasibility-first on [I_Vm,I_sp,G,A,J_rho,J_EI,J_local] forbids scalar \|bar-7.5\|; E<PV soft not hard | 2026-08-29_005_fg11-transfer-operating-manifold | project | HIGH | `AGENTS.md` / harness |
| HDP-OFF delayed grouped I[t,G]=segment_sum is identifying for fast dynamics (Δbar<0.05Hz ΔVm<0.02mV passivity0) even tho | 2026-08-29_005_fg11-transfer-operating-manifold | project | HIGH | `AGENTS.md` / harness |
| Current->V_m is heterogeneous graded shunting (G 0.013-4.98) not binary FAIL | 2026-08-29_005_fg11-transfer-operating-manifold | project | HIGH | `AGENTS.md` / harness |
| A_rec is measured phenotype not hard >1.2 — suppressive A<1 CI<1 with information gain is valid recurrent mediation | 2026-08-29_005_fg11-transfer-operating-manifold | project | HIGH | `AGENTS.md` / harness |
| ISN NON_ISN_LIKE under +20% PV cannot be proof of non-ISN in heterogeneous network | 2026-08-29_005_fg11-transfer-operating-manifold | project | MED | `AGENTS.md` / harness |
| HDP_OFF delayed grouped bridge is identifying for fast dynamics (Δbar<0.05 Hz ΔVm<0.02 mV passivity 0) even though H dif | 2026-08-29_004_fg09b-hdp-off-transfer-localization | project | HIGH | `AGENTS.md` / harness |
| Current→V_m is heterogeneous graded shunting (G 0.013-0.85) not binary FAIL; single-seed risks overclaim | 2026-08-29_004_fg09b-hdp-off-transfer-localization | project | HIGH | `AGENTS.md` / harness |
| Passive already-computed quantity should be exposed via lax.scan tail stacking, not new state — edge_current=w*syn_state | 2026-08-29_003_u01-u02-current-instrumentation | project | HIGH | `AGENTS.md` / harness |
| Grouped I[t,G]=segment_sum(I[t,E], gids, G) inside kernel gives 600× memory saving and is numerically faithful within 1e | 2026-08-29_003_u01-u02-current-instrumentation | project | HIGH | `AGENTS.md` / harness |
| config_hash (topology) must stay invariant while RuntimeConfig recording flags disambiguate via _gid_hash in JIT cache k | 2026-08-29_003_u01-u02-current-instrumentation | harness | MED | `AGENTS.md` / harness |
| MECHANISM_OBSERVABILITY_BLOCKER correctly distinguishes delayed non-HDP (solved) from delayed+HDP (blocked) — do not con | 2026-08-29_003_u01-u02-current-instrumentation | project | HIGH | `AGENTS.md` / harness |
| Vertical weight scaling cannot overcome sparse recruitment Ne16 k0.59 and E/I cancellation W_net 4.4×<W_exc | 2026-08-29_002_w4a-propagation-and-observability-blocker | project | HIGH | `AGENTS.md` / harness |
| Per-blob co-recruitment PV/E∈0.5-2.0 Jaccard>0.3 is correct visual grammar but PV domination not vertical bottleneck | 2026-08-29_002_w4a-propagation-and-observability-blocker | project | MED | `AGENTS.md` / harness |
| NOT OBSERVABLE must be rendered literally when stage blocked by upstream capability | 2026-08-29_002_w4a-propagation-and-observability-blocker | project | HIGH | `AGENTS.md` / harness |
| Truly absent vs rate-invisible requires LOO + multiple decoders + Vm + source q diagnostic | 2026-08-29_002_w4a-propagation-and-observability-blocker | project | HIGH | `AGENTS.md` / harness |
| V0 ontology N_model≠N_free≠N_AGSDR prevents conflating numbers vs tunable scalars | 2026-08-29_002_w4a-propagation-and-observability-blocker | project | HIGH | `AGENTS.md` / harness |
| Hard must protect distributional tail P(CV>0.5)>=0.15 / max>=1.5, not scalar meanCV>=0.5 | 2026-08-29_001_cl1-c018-to-c019-agsdr | project | HIGH | `AGENTS.md` / harness |
| Orthogonal partition 9+2+4+1=16 disjoint prevents double-count E->PV when scaling g_rec vs g_fastEI | 2026-08-29_001_cl1-c018-to-c019-agsdr | project | HIGH | `AGENTS.md` / harness |
| Successive fidelity frozen S1[0] S2[0,1] S4[0,1,2,3] y_bar t3 CIs prevents cherry-picking max seed | 2026-08-29_001_cl1-c018-to-c019-agsdr | project | HIGH | `AGENTS.md` / harness |
| Single RS sigma0.40 variance-matched still fails vs typed CH c-50/d2 (p<1e-6) -> typed pseudogenome justified | 2026-08-29_001_cl1-c018-to-c019-agsdr | project | HIGH | `AGENTS.md` / harness |
| Mean-controlled g_background preserves mu[3.4,4.0] while redistributing private variance | 2026-08-29_001_cl1-c018-to-c019-agsdr | project | MED | `AGENTS.md` / harness |

## Recurrent Friction

> ⚠️ `HARNESS_REVIEW_REQUIRED` — friction recurred ≥2 sessions; propose harness/tool fix in next session §12.

| Friction | Occurrences | Sessions | Last workaround | Permanent repair? | Flag |
|----------|-------------|----------|-----------------|-------------------|------|
| Delayed+HDP grouped current MISSING_CAPABILITY at jaxfne/_model_simulate.py:280 + HDP seam lacks record_grouped_current | 2 | 2026-08-29_005_fg11-transfer-operating-manifold, 2026-08-29_004_fg09b-hdp-off-transfer-localization | Use HDP-OFF delayed non-HDP grouped I[t, | HDP+delay grouped current support in jax | ⚠️ HARNESS_REVIEW_REQUIRED |
| delayed+HDP guard | 1 | 2026-08-29_006_u01b-fg12-canonical-cl2 | ring buffer | JAXFNE_UPSTREAM |  |
| Window B=[-50,0) not in Simulation(duration 1000) to_array so use B_proxy=[0,50) early p1 pre-volley | 1 | 2026-08-29_005_fg11-transfer-operating-manifold | Report mean(W)-mean(B_proxy) + raw mean( | Cross-ref simulation vs absolute clock i |  |
| ISN characterization requires substantial inhibitory fraction; +20% PV+SST ~6% gives non-paradoxical even if ISN | 1 | 2026-08-29_005_fg11-transfer-operating-manifold | Characterize as NON_ISN_LIKE under this  | Larger-fraction ISN titration + heteroge |  |
| Window B surrogate: freeze B=[-50,0) not in Simulation(duration 1000) to_array, so use B_proxy=[0,50) early p1 pre-volley | 1 | 2026-08-29_004_fg09b-hdp-off-transfer-localization | Report mean(W)-mean(B_proxy) + raw mean( | Cross-ref simulation vs absolute clock i |  |
| enable_hdp does not support nonzero edge delay_steps guard at jaxfne/_model_simulate.py:280 blocks production delayed+HDP current observability | 1 | 2026-08-29_003_u01-u02-current-instrumentation | Delayed non-HDP now solved via U01/U02 ( | JAXFNE_UPSTREAM — HDP finite-delay ring  |  |
| Delayed edge-current recording unavailable at finite delays [20,80,120] (record_edge_current HDP-only, guard at jaxfne/_model_simulate.py:280) | 1 | 2026-08-29_002_w4a-propagation-and-observability-blocker | Render transfer oscilloscope stage as NO | JAXFNE_UPSTREAM Options A/B/C per jaxfne |  |
| Repeated failed g_vertical search invites guessing next lever without observable | 1 | 2026-08-29_002_w4a-propagation-and-observability-blocker | Seal W4a as MECHANISM_OBSERVABILITY_BLOC | PROJECT_RULE W4 gating rule |  |
| Heavy-tailed weight CV1.5 and 4-order H tau unreadable on linear | 1 | 2026-08-29_002_w4a-propagation-and-observability-blocker | sign·log1p(\|W\|/1e-6) and log10 tau wit | VISUALIZATION sampling_utils already imp |  |

---

_Generated by `docs/sessions/validate.py` · schema `1.0.0` · do not hand-edit — run `python docs/sessions/validate.py --reindex` to regenerate._
