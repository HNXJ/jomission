# Checklist score (v3)

Scope: the jomission project goals. Re-weighted 2026-11-04 at owner request.
50 items, 2 points each, sum 100. Scoring: 2 = met with evidence, 1 = partial, 0 = not met.
Category weights (owner, 2026-11-04): Science program 40, Documentation and reproducibility 30, Dashboard and figures 20, rest 10. "Rest" is one pool of 5 items covering engine migration, models and harness.
Earlier revisions: v2 (83.2, 2026-11-04) in docs/checklist-score-v2.md; v1 (98, migration-only) in docs/checklist-score-v1.md.

**Current score: 55/100** (v3, 2026-11-04). This is lower than v2 because the science program now carries 40 points and is largely unmet.

## By category
| Category | Score | Items |
|---|---|---|
| Science program | 13 / 40 | 20 |
| Documentation and reproducibility | 17 / 30 | 15 |
| Dashboard and figures | 18 / 20 | 10 |
| Rest (migration, models, harness) | 7 / 10 | 5 |

## Items
| # | Category | Objective | Score (of 2) | Evidence |
|---|---|---|---|---|
| 1 | Science program | tier 3 scientific batteries run under an authorized TODO | 0 | owner declined 2026-11-03; PV group decision pending; not run |
| 2 | Science program | V2 local operation gate PASS (manifests/gates/v2_local_operation.json) | 0 | status FAIL (scientific result) |
| 3 | Science program | viable operating regime found (PV in [12,25], VIP in [8,15], SST in [6,12], E < PV) | 0 | both-repaired cell: no cell viable (docs/HANDOFF_CURRENT.md; lineage_registry EI_REGIME_STILL_UNRESOLVED) |
| 4 | Science program | regime verdict recorded as one ratified label | 2 | EI_REGIME_STILL_UNRESOLVED recorded (SCI-EI-INTERACTION-1, d846eff) |
| 5 | Science program | SCI-EI-NATIVE-STATE-1 sealed verdict | 2 | current_state.json: MIXED_STATE_MECHANISM, sealed |
| 6 | Science program | SST repeated-spiking question answered | 1 | partial: SST_STATE_ADVANTAGE; network-level question not closed |
| 7 | Science program | single-neuron f-I assay (NumPy, exploratory) | 1 | docs/receipts/fi_single_neuron_v1.json; not engine-validated |
| 8 | Science program | engine-equivalence of the f-I assay | 0 | not done (OPEN, review item) |
| 9 | Science program | visualization V0 schematic for the canonical lineage | 0 | not produced (manifests/visualization_contract.json; applicability to confirm, OPEN-02) |
| 10 | Science program | visualization V1 initial 1 s raster | 0 | not produced |
| 11 | Science program | visualization V2 final raster | 0 | not produced |
| 12 | Science program | visualization V3 atlas | 0 | not produced |
| 13 | Science program | AGSDR calibration design sealed, with amendments | 2 | docs/designs/SCI-AGSDR-CALIBRATION_design.md (0726c5d); amendments 1 and 2; B10 audit amendment |
| 14 | Science program | AGSDR stage 0 executed with receipts | 1 | 24 sims, seed 0, n=400; receipts in docs/receipts/agsdr_local/ (stage0_summary.json, stage0_tau_check.json, stage0_correction_v1.json); Efrac tau defect and selection defect open (R1, R2) |
| 15 | Science program | AGSDR stage 0 selection determined under the frozen rule | 0 | NOT DETERMINED: dimension set, overflow, epsilon predicate (OPEN-23), B3 route (OPEN-19) open; H08 hard-or-informational conflict (OPEN-24) |
| 16 | Science program | AGSDR stage 1 (top 8 x seeds 0,1) | 0 | not authorized (stop_condition in manifests/todo.json) |
| 17 | Science program | AGSDR stage 2 (top 3 x seeds 0..3) | 0 | not authorized; depends on stage 1 |
| 18 | Science program | PV drive scan (matched-state) | 0 | draft only, not approved, not in repo |
| 19 | Science program | B10 objective audit clean | 2 | docs/receipts/agsdr_local/B10_objective_audit.json (runner sha 160265516fe2) |
| 20 | Science program | build and simulation speed (compile cache) | 2 | docs/receipts/compile_cache_benchmark.json (cold median 3.53 s, warm 1.05 s) |
| 21 | Documentation and reproducibility | cloud setup handout | 2 | docs/CLOUD_HANDOUT.md |
| 22 | Documentation and reproducibility | fresh-clone reproduction of the setup | 1 | run in this container only; not on a fresh cloud image (OPEN-09) |
| 23 | Documentation and reproducibility | checklist versioned with history | 2 | docs/checklist-score-v1.md, docs/checklist-score-v2.md, this file; git history |
| 24 | Documentation and reproducibility | results inventory on the dashboard | 1 | 367 results rows listed; figures not rendered (OPEN-14) |
| 25 | Documentation and reproducibility | open stack current and consistent with audits | 1 | docs/OPEN.md current at 4e72c8a; audit conflicts (H08, regime verdict, attribution) partly recorded |
| 26 | Documentation and reproducibility | handoff regenerated and current | 2 | docs/HANDOFF_CURRENT.md; tests/test_harness_state.py passes |
| 27 | Documentation and reproducibility | TODO consistent with state | 1 | validators pass; top-level null key removed (164ad74); stop_condition lists its rulings (OPEN-18, 19, 23, 24, 25); reviewer-authorization location still undefined (OPEN-25) |
| 28 | Documentation and reproducibility | every claim has a receipt | 0 | OPEN-08: four docs make claims with no receipt |
| 29 | Documentation and reproducibility | sealed-artifact policy enforced | 1 | append-only rule enforced by guards; in-place reseals recorded (OPEN-04, governance open); cp1252 reseals owner-approved (v2 item 25); results/jaxfne_058_correction_1.json registered (v2 item 27); tests/test_within_area_sparse_coverage.py (v2 item 26) |
| 30 | Documentation and reproducibility | commit and PR provenance | 1 | Claude-Session trailer on every commit; model-name trailer conflict (OPEN-21); PR #1 body stale (OPEN-22) |
| 31 | Documentation and reproducibility | gate thresholds single source | 1 | test_gates_single_source passes; AGSDR predicates are copies of runner rules (tested, not imported) |
| 32 | Documentation and reproducibility | design-to-code traceability for AGSDR | 1 | runner sha in receipts; correction record; H08 and H07/H11 mismatch with the design unresolved |
| 33 | Documentation and reproducibility | goals document ratified | 1 | docs/GOALS_PROPOSED.md exists; not ratified by owner |
| 34 | Documentation and reproducibility | receipts name generator and hash | 1 | AGSDR receipts do; older receipts not audited |
| 35 | Documentation and reproducibility | compile-cache trust documented | 1 | default directory documented; trust decision open (OPEN-17) |
| 36 | Dashboard and figures | docs/DASHBOARD.html committed, no external resources | 2 | docs/DASHBOARD.html; self-contained SVG |
| 37 | Dashboard and figures | dashboard generated deterministically by scripts/build_dashboard.py | 2 | rebuild byte-identical (checked 2026-11-04) |
| 38 | Dashboard and figures | tier 0 figure with receipt | 2 | docs/receipts/receipts.json |
| 39 | Dashboard and figures | tier 1 figure with receipt | 2 | docs/receipts/receipts.json |
| 40 | Dashboard and figures | tier 2 figure with receipt | 2 | docs/receipts/receipts.json |
| 41 | Dashboard and figures | tier 3 shown as open with its reason | 2 | receipts open status; owner declined |
| 42 | Dashboard and figures | edge-identity table on the dashboard | 2 | receipts.json edge_identity |
| 43 | Dashboard and figures | checklist score chart by category and item | 2 | inline SVG, Chromium-rendered |
| 44 | Dashboard and figures | GitHub Pages current with the head | 0 | main deploy blocked by the github-pages environment rule (OPEN-07); last live deploy not verified |
| 45 | Dashboard and figures | open items and owner decisions shown with reasons | 2 | receipts.json open_items, owner_decisions |
| 46 | Rest | jaxfne 0.5.8 pinned, installed, and adopted as source of truth | 2 | pyproject.toml:18; project_check identity 0.5.8; results/jaxfne_058_migration.json; docs/JAXFNE_058_AUDIT.md; jaxfne tracker 0 issues, 119 PRs (v2 items 5, 6) |
| 47 | Rest | canonical network identity (15 EdgeList fields, config hash, builds) | 1 | reproduced at a9fcb65 (10666 edges); config hash 05f3d334451c53d2 frozen; not re-run on head. Also: inter-column delays, cell-map identity, within-area locality (builder.py; v2 items 12-14); lite model tests/test_lite_model.py (v2 items 11, 16) |
| 48 | Rest | tiers 0, 1 and 2 pass on current head | 1 | last full runs at 860e330: tier 0 216 passed; tier 1 non-simulating 49 passed, 1 skipped; tier 2 419 passed, 3 skipped. Head 164ad74 has only targeted runs, which pass |
| 49 | Rest | sealed registry and guards pass on head | 2 | test_harness_guards and test_harness_state pass at 4e72c8a; project_check --offline --no-tests PASS at 4e72c8a. Full project_check PASS (v2: at 860e330) not re-run on head |
| 50 | Rest | CI green on head and working tree clean and pushed | 1 | CI run on current head not verified; tree clean at push |

## Notes
- Visualization contract (manifests/visualization_contract.json) requires V0 to V3 for every non-grandfathered lineage. Whether the migrated canonical lineage is exempt is undecided (items 9 to 12).
- Tier 3, the V2 gate and the PV group are science decisions, not migration progress.
- Items 14 to 17 and the review conditions below replace the earlier simulation-program table; each fact is recorded once.

## Review conditions before stage 1 (2026-11-04)
Source: external review of a50121b and audits of f23f8f4, a30ce48, 4e72c8a.

| R | Condition | Status | Record |
|---|---|---|---|
| R1 | Selection rule: the frozen rule (nondominated set on the seven J_k, epsilon 0.05 retained by ruling 1, w=1/6 tie-break, remainder eliminated) has open parts: epsilon predicate (OPEN-23); dimension set, 6 computed vs 7 named (ruling 2); overflow beyond the front and the 1/6 vs 1/7 weight (OPEN-18). Front-order fill withdrawn as automatic. | PARTIAL | OPEN-18, OPEN-23; stage0_correction_v1.json selection |
| R2 | Efrac as W·r·tau (frozen J_EI): global value recomputed for all 24 (tau 2.0 ms E-pre, 5.0 ms I-pre); no hard verdict changes. Population B3 needs per-population ratios p' = 2p/(5-3p); none retained. Hard list: runner computes 11 gates; design line 18 lists nine; manifest lists H01-H12; H07 and H11 never computed; H08 is hard in the objectives but informational in the runner, 0 of 24 inside its box. If H08 is hard, 0 of 24 are feasible | PARTIAL | OPEN-19, OPEN-24; stage0_correction_v1.json |
| R3 | Hard gates can fail: boundary cases on copies of the predicates; 24 stored verdicts reproduced; runner-path score_run on silent and saturated stub input (2 cases). Other gates not yet exercised on the runner path | PARTIAL | OPEN-20; tests/test_agsdr_stage0_correction.py |
| R4 | Sampling departure: 16 rows redrawn, count reconstructed. Background values of redrawn rows not verified. Recorded, not accepted | RECORDED | OPEN-20; stage0_correction_v1.json sampling_departure |
| R5 | CV denominator: AGS_S0_06 P(CV_E>0.5)=0.207 over 184 of 242 E cells (>=3 spikes); defined counts 131 to 187 across candidates | RECORDED | OPEN-20; stage0_correction_v1.json |
| R6 | Stage 1 requires: R1 to R3 resolved, selection determined, eight vectors and cross-seed rule frozen, viable-regime condition of the TODO procedure addressed (OPEN-25) | BLOCKED | OPEN-18, OPEN-19, OPEN-23, OPEN-24, OPEN-25 |
