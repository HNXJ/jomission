# Checklist score (v4)

Scope: the jomission project goals, as the program stands on 2026-11-04 (branch cowork/focused-ptolemy-dz7hwc; commit shas in item evidence). This file is the checklist the dashboard reads (adopted 2026-11-04).
50 items, 2 points each, sum 100. Scoring: 2 = met with a receipt or commit that was checked, 1 = partial, 0 = not met or not verifiable.
Category maxima (owner, 2026-11-04): Science program 40 (20 items), Documentation and reproducibility 30 (15), Dashboard and figures 20 (10), Rest 10 (5). The item count per category is unchanged from v3.
Earlier revisions: v3 (50, 2026-11-04) in docs/checklist-score-v3.md; v2 (83.2) in docs/checklist-score-v2.md; v1 (98, migration-only) in docs/checklist-score-v1.md.

Rule for items 48 and 49: 0 when an in-scope failure is recorded on head; 1 when the last pass is on an older commit with no recorded in-scope failure.
Review rule for this draft: every evidence entry is a path or commit sha that was checked with ls, git show or git merge-base in this session. A claim that rests only on a commit message or on the OPEN stack is scored 1 at most, and this is noted in the item.

**Current score: 60/100** (v4, 2026-11-04). Against v3 (50): Science +7, Documentation +1, Dashboard +1, Rest +1. Owner ratified items 17, 19, 20 (Q14, 2026-11-04). PV drive scan (item 18) kept at 0 until a spec exists (Q15, 2026-11-04).

## By category
| Category | Score | Items |
|---|---|---|
| Science program | 21 / 40 | 20 |
| Documentation and reproducibility | 16 / 30 | 15 |
| Dashboard and figures | 15 / 20 | 10 |
| Rest | 8 / 10 | 5 |

## Items
| # | Category | Objective | Score (of 2) | Evidence |
|---|---|---|---|---|
| 1 | Science program | Tier 3 scientific batteries run under an authorized TODO | 0 | owner declined tier 3 (a9fcb65); receipts tier 3 status OPEN in docs/receipts/receipts.json |
| 2 | Science program | V2 local operation gate PASS (manifests/gates/v2_local_operation.json) | 0 | gate V2_LOCAL_OPERATION recorded as FAIL in docs/HANDOFF_CURRENT.md (line 40) |
| 3 | Science program | Viable operating regime found (PV in [12,25], VIP in [8,15], SST in [6,12], E < PV) | 1 | found on a non-design axis: SCI-VIP-DRIVE-PROBE (spec sealed 2fcac30; docs/receipts/vip_drive_probe/summary.json) verdict VIABLE_BRANCH_ON_VIP_DRIVE_AXIS: VIP drive x2.0 and x2.5 put AGS_S0_06 and AGS_S0_20 in all bands with E < PV at seeds 0 and 1; partial because VIP drive is outside the AGSDR design (owner ruling pending, OPEN-25), n = 2 seeds, fresh start only; lineage verdict EI_REGIME_STILL_UNRESOLVED (d846eff) stands |
| 4 | Science program | Regime verdict recorded: a sealed verdict is on file (null permitted) | 2 | manifests/lineage_registry.json (EI_REGIME_STILL_UNRESOLVED recorded); owner ruling Q8 (docs/OPEN.md OPEN-30) |
| 5 | Science program | SCI-EI-NATIVE-STATE-1 sealed verdict | 2 | manifests/current_state.json (MIXED_STATE_MECHANISM); results/whole_system_ei_native_state.json, listed in manifests/sealed_artifacts.json |
| 6 | Science program | SST repeated-spiking question answered | 1 | partial: SST_STATE_ADVANTAGE in results/whole_system_ei_native_state.json; network-level question not closed (docs/HANDOFF_CURRENT.md) |
| 7 | Science program | Single-neuron f-I assay (NumPy, exploratory) | 1 | docs/receipts/fi_single_neuron_v1.json; script scripts/fi_single_neuron.py; not engine-validated |
| 8 | Science program | Engine-equivalence of the f-I assay | 0 | not done: docs/receipts/fi_single_neuron_v1_provenance.json records provenance only; no equivalence receipt in docs/receipts/ |
| 9 | Science program | Visualization V0 (schematic) and V1 (initial 1 s raster) exist for a canonical lineage | 2 | canonical lineage SCI-EI-NATIVE-STATE-1 (owner ruling Q13, 2026-11-04): V0 jaxfne_network_hspice.png and V1 raster_initial_1000ms.png present and registered in manifests/lineage_registry.json (ls, grep); file existence checked, content not inspected here (v3 items 9, 10) |
| 10 | Science program | Visualization V2 (final raster) and V3 (atlas) exist for a canonical lineage | 2 | SCI-EI-NATIVE-STATE-1: raster_final.png and atlas/index.html (with 4 atlas pages) present and registered; existence checked, content not inspected here (v3 items 11, 12) |
| 11 | Science program | AGSDR calibration design sealed, with amendments 1 to 3 | 2 | docs/designs/SCI-AGSDR-CALIBRATION_design.md (0726c5d); docs/designs/SCI-AGSDR-CALIBRATION_amendment_1.md, _2.md, _3.md (amendment 3 sealed in 43d1d71) |
| 12 | Science program | AGSDR stage 0 executed with receipts (24 candidates, seed 0) | 2 | docs/receipts/agsdr_local/stage0_summary.json (run in a50121b); its hard_pass_count 24 / best_feasible is superseded by stage0_correction_v1.json (2 of 24 feasible under H05 all-E, OPEN-26) |
| 13 | Science program | AGSDR stage 0 selection determined under the frozen rule | 0 | NOT DETERMINED: docs/receipts/agsdr_local/stage0_correction_v1.json; OPEN-18 and OPEN-23 in docs/OPEN.md; H08 status OPEN-24 |
| 14 | Science program | AGSDR stages 1 (top 8 x seeds 0,1) and 2 (top 3 x seeds 0..3) executed | 0 | blocked: docs/OPEN.md OPEN-25 (viable-regime verdict) and R6 in the review table below; no stage-1 receipt in docs/receipts/ (merged from v3 items 16 and 17) |
| 15 | Science program | B10 term scan of the stage-0 runner clean | 2 | docs/receipts/agsdr_local/B10_objective_audit.json (22-term whole-word scan, runner sha 160265516fe2); the receipt states the objective predicate is not separately implemented (v3 item 19) |
| 16 | Science program | AGSDR Efrac tau correction applied to stage-0 receipts, including B3 | 2 | docs/receipts/agsdr_local/stage0_correction_v1.json; docs/receipts/agsdr_b3_rerun/summary.json (sealed 47d055f) (v3 item 20) |
| 17 | Science program | H balance-only rule acts on the network under a sealed test (predictions H1 to H3) | 1 | spec ce55b01 before run a7e3ae8; per-run fields in docs/receipts/h_balance_test/*_seed*.json recomputed: H1, H2, H3 hold 4 of 4. n = 2 seeds per condition; A seed 0 was seen in a smoke run (spec), so 3 of 4 are independent. Runner changed between seal and run for the mech test (b4fee06 surrogate seeding, de1b423 record_weight_trace False; spikes checked identical at 2000 ms only), disclosed here. Spec-before-run ancestry checked (git merge-base --is-ancestor): 3ca43f3<87acaf5, ce55b01<a7e3ae8, 11c1a0f<dc80abe, 024d2a6<069f21d, ac62ff0<7b21e87, 7ffa304<4df8ac9. Negative results are receipted: h_active_test (87acaf5), h_sync_test (dc80abe), h_sync2_test (4df8ac9, selection failed); ledger OPEN-27. Not a calibration claim. New item |
| 18 | Science program | PV drive scan (matched-state) | 0 | no approved spec or receipt; kept, not retired (Q15) (the calibration freezes PV_DRIVE 1.7; its J_CL1C +50% test is a response measure, not a matched-state scan) (v3 item 18) |
| 19 | Science program | H balance-only rho and population Fano rise resolved (prediction H4) | 1 | partial: cause localised, not resolved. H_BALANCE_MECH (sealed 7e074b2; docs/receipts/h_balance_mech/summary.json): with H on E cells only the B rho rise falls to 27 and 33 percent of the reference rise (rho 0.064, 0.081 vs 0.188, 0.195; rate-matched 0.067), E rate cut kept; H4 still fails (rho and Fano above baseline). Earlier: known limit, not resolved: H4 fails 4 of 4 (a7e3ae8); b9f118a; synchrony term closed by owner (Q6-H; h_sync_test, h_sync2_test). New item |
| 20 | Science program | H balance rule entered in the calibrated model with its own sealed calibration | 0 | not in the calibrated model (85c138b, Q7); H inactive in sealed 0.5.8 lineages (b1611d8); rule exists as optional jomission/dynamics/activity_balance.py. New item |
| 21 | Documentation and reproducibility | Cloud setup handout | 2 | docs/CLOUD_HANDOUT.md |
| 22 | Documentation and reproducibility | Fresh-clone reproduction of the setup | 1 | run in this container only; fresh cloud image not tested (OPEN-09 in docs/OPEN.md) |
| 23 | Documentation and reproducibility | Checklist versioned with history | 2 | docs/checklist-score-v1.md; docs/checklist-score-v2.md; docs/checklist-score-v3.md; checklist-score.md (v4, repo root); git history |
| 24 | Documentation and reproducibility | Results inventory on the dashboard | 1 | inventory section of docs/DASHBOARD.html (built by scripts/build_dashboard.py); figures not rendered (OPEN-14) |
| 25 | Documentation and reproducibility | Open stack current and consistent with audits | 1 | docs/OPEN.md (updated at 7769617: OPEN-19 removed as done, OPEN-29 added); OPEN-22 (PR #1 body stale) and OPEN-28 (full-suite failures, not bisected) open |
| 26 | Documentation and reproducibility | Handoff regenerated and current | 1 | docs/HANDOFF_CURRENT.md (regenerated at 374410d); carries no R1 to R6 status (review table below), so partial |
| 27 | Documentation and reproducibility | TODO consistent with state | 1 | manifests/todo.json; validators pass after 164ad74; rulings OPEN-18 and OPEN-23 to OPEN-26 listed (OPEN-19 removed as done at 7769617) |
| 28 | Documentation and reproducibility | Every claim has a receipt | 0 | OPEN-08 in docs/OPEN.md: four docs make claims with no receipt |
| 29 | Documentation and reproducibility | Sealed-artifact policy enforced | 1 | manifests/sealed_artifacts.json lists older results/h_defect, h_history, h_lineage, h_sensitivity and v21 results, but neither the H-controller receipts nor docs/designs/H_*_spec.md (sealed by commit only); in-place reseals recorded (OPEN-04) |
| 30 | Documentation and reproducibility | Commit and PR provenance | 1 | Claude-Session trailer on 119 of 469 non-merge commits (all since 2026-11-03; counted 2026-11-05); Co-Authored-By names vary (Opus 5 119, Sonnet 5.5 33, Opus 5.5 3, Haiku 5.5 3, internal-model 2); owner ruling OPEN-21 (b8d492e) keeps attribution lines but the written rule in docs/checklist-score-v1.md item 35 is not amended; PR #1 body stale (OPEN-22) |
| 31 | Documentation and reproducibility | Gate thresholds single source | 1 | tests/test_gates_single_source.py; AGSDR predicates are copies of runner rules (tested, not imported) |
| 32 | Documentation and reproducibility | Design-to-code traceability for AGSDR | 2 | every hard gate has code: H01 to H10d in scripts/agsdr_local_sweep.py; H07, H11 in scripts/agsdr_h07h11.py (spec sealed 7249e96, receipts docs/receipts/agsdr_h07h11); H08 informational in design (amendment 3) and runner. H07 term 4 fails its reference (OPEN-29): a definition issue, not a traceability gap |
| 33 | Documentation and reproducibility | Goals document ratified | 0 | docs/GOALS_PROPOSED.md exists; not ratified by the owner |
| 34 | Documentation and reproducibility | Receipts name generator and hash | 1 | docs/receipts/h_balance_test/summary.json (rule_sha256, script_sha256, spec_sha256); docs/receipts/agsdr_local/stage0_summary.json; older receipts not audited |
| 35 | Documentation and reproducibility | Compile cache: build-time receipt and trust documented | 1 | docs/receipts/compile_cache_benchmark.json (build only); trust decision open (OPEN-17) |
| 36 | Dashboard and figures | docs/DASHBOARD.html committed, no external resources | 2 | docs/DASHBOARD.html; no script, link or http reference found in it |
| 37 | Dashboard and figures | Dashboard generated deterministically by scripts/build_dashboard.py | 2 | rebuilt from a copy of the repo on 2026-11-04: byte-identical to docs/DASHBOARD.html |
| 38 | Dashboard and figures | Tier 0 test-count chart with receipt | 1 | docs/receipts/receipts.json (counts only; not a result figure) |
| 39 | Dashboard and figures | Tier 1 test-count chart with receipt | 1 | docs/receipts/receipts.json (counts only; not a result figure) |
| 40 | Dashboard and figures | Tier 2 test-count chart with receipt | 1 | docs/receipts/receipts.json (counts only; not a result figure); chart drawn by scripts/build_dashboard.py tier_svg (v3 item 40) |
| 41 | Dashboard and figures | Tier 3 shown as open with its reason | 2 | docs/receipts/receipts.json (tier 3 status OPEN, reason recorded); a9fcb65 |
| 42 | Dashboard and figures | Edge-identity table on the dashboard | 2 | docs/receipts/receipts.json, key edge_identity; table in docs/DASHBOARD.html |
| 43 | Dashboard and figures | Checklist score chart by category and item | 2 | scripts/build_dashboard.py (score_svg, item_svg); docs/DASHBOARD.html |
| 44 | Dashboard and figures | GitHub Pages current with the head | 0 | deploy job fails before any step, inferred to be a github-pages environment branch rule (settings unreadable, 403) (OPEN-07 in docs/OPEN.md); .github/workflows/pages.yml |
| 45 | Dashboard and figures | Open items and owner decisions shown with reasons | 2 | dashboard section "Open stack" lists all docs/OPEN.md rows with status, what is open and who decides (scripts/build_dashboard.py open_stack; tests/test_dashboard_build.py::test_open_stack_rows_shown: page rows = OPEN rows); ruled rows carry their rulings; checklist-level open list and receipts owner_decisions unchanged |
| 46 | Rest | jaxfne 0.5.8 pinned, installed, adopted as source of truth | 2 | pyproject.toml line 18 (jaxfne==0.5.8); results/jaxfne_058_migration.json; docs/JAXFNE_058_AUDIT.md |
| 47 | Rest | Canonical network identity (edge list, config hash, builds) | 2 | docs/receipts/receipts.json (edge_identity; config_hash); tests/test_lite_model.py; no commit to jomission/network/builder.py after a9fcb65 (git log --ancestry-path a9fcb65..HEAD) |
| 48 | Rest | Tiers 0, 1 and 2 pass on the current head | 2 | docs/receipts/receipts.json tiers (2026-11-05): tier 0 216 passed, tier 2 500 passed 3 skipped at 2637aca; tier 1 88 passed 1 skipped (+9 node ids) at d228841; jomission/ and tests/ unchanged between them and at head. One provenance test fails only in a worktree by design and passed in the checkout. results/ clean after tier 2. Tier 3 excluded by definition (OPEN-01) |
| 49 | Rest | Sealed registry and guards pass on head | 1 | rule: 1 when the last pass is on an older commit with no in-scope failure recorded. tests/test_harness_guards.py, test_harness_state.py, test_agsdr_stage0_correction.py; the three files pass on head 70d5c37 (56 passed, tree clean after); held at 1 because OPEN-05 (weakened rf assertion) and OPEN-06 (allowlisted test) question guard integrity |
| 50 | Rest | CI green on head, working tree clean, pushed | 1 | pushed: origin/cowork/focused-ptolemy-dz7hwc = HEAD (checked at commit time); working tree clean; CI run on the head not verified (partial) |

## Review conditions before stage 1 (carried from v3; status as of 2026-11-04)
| R | Condition | Status | Record |
|---|---|---|---|
| R1 | Selection rule: epsilon additive 0.05 (Q6-AGSDR), weight 1/6 (Q4) and six J entries (Q5) ruled; open: raw vs normalised J, boundary at exactly 0.05, order dependence of peeled fronts | PARTIAL | OPEN-18, OPEN-23; docs/receipts/agsdr_local/stage0_correction_v1.json |
| R2 | Efrac tau-weighted for all 24; H05 under all-E reading gives 2 of 24 pass; H07 and H11 never computed (OPEN-29); H08 hard in objectives, informational in runner | PARTIAL | OPEN-24, OPEN-26, OPEN-29; docs/receipts/agsdr_local/stage0_correction_v1.json |
| R3 | Hard gates can fail: boundary cases on copies of the predicates; other gates not yet exercised on the runner path | PARTIAL | OPEN-20; tests/test_agsdr_stage0_correction.py |
| R4 | Sampling departure: 16 rows redrawn; background values not verified | RECORDED | OPEN-20; docs/receipts/agsdr_local/stage0_correction_v1.json |
| R5 | CV denominator: all 242 E cells; 2 of 24 pass H05 | RULED Q1 (2026-11-04) | OPEN-20, OPEN-26 |
| R6 | Stage 1 requires R1 to R3 resolved, selection determined, vectors and cross-seed rule frozen, viable-regime verdict | BLOCKED | OPEN-18, OPEN-23, OPEN-24, OPEN-25, OPEN-26 |

## Changes from v3
Numbering: items 1 to 8 keep their v3 numbers; v3 9 to 12 became 9 (V0, V1) and 10 (V2, V3); v3 13 to 15 became 11 to 13; v3 16 and 17 became 14; v3 19 and 20 became 15 and 16; items 18 and 21 to 50 keep their numbers. Items 17, 19, 20 are new (H controller); 30 and 40 keep their v3 objectives.

| Item (v4 number) | v3 | v4 | Reason |
|---|---|---|---|
| 9, 10 | 0,0,0,0 | 2, 2 | v3 checked the wrong location; V0 to V3 exist in results/viz/<lineage>; owner ruled SCI-EI-NATIVE-STATE-1 canonical (Q13), all four registered and present |
| 17 | new | 1 | H1 to H3 hold 4 of 4; n = 2, A seed 0 smoke-informed |
| 19, 20 | new | 1, 0 | H4 cause localised to interneuron control (H_BALANCE_MECH); rule not in calibrated model |
| 18 | 0 | 0 | PV drive scan kept; retirement pending owner |
| 15 | 2 | 2 | renamed to what the receipt shows (term scan) |
| 48 | 1 | 2 | tiers 0, 1, 2 pass on head (receipts.json, 2026-11-05) |
| 30, 40 | 1, 1 | 1, 1 | unchanged; evidence updated (provenance counts; H-spec ancestry moved to item 17) |
| 45 | 1 | 2 | dashboard Open stack section from docs/OPEN.md (test: rows shown = rows in file) |
| 3 | 0 | 1 | viable bands met on the VIP-drive axis (SCI-VIP-DRIVE-PROBE), no hysteresis on two axes (SCI-PV-HYSTERESIS); outside the design |
| 32 | 1 | 2 | H07, H11 implemented and run (scripts/agsdr_h07h11.py) |
| 4 | 2 | 2 | wording changed to "sealed verdict recorded"; Q8 now in docs/OPEN.md |

Totals by category, v3 to v4: Science 14 to 21; Documentation 15 to 16; Dashboard 14 to 15; Rest 7 to 8; total 50 to 60.

## Notes
- Harness diagnosis (AGENTS section 5; score below 80). Cause: unmet science objectives (items 1 to 3, 8, 13, 14, 18 to 20 at 0) and owner rulings pending (OPEN-18 to OPEN-26, OPEN-29). No harness defect is the cause. Repair: owner rulings and the science program. Recurrence check: the dashboard build fails when the category table or header total disagrees with item rows (scripts/build_dashboard.py, check_checklist_summary).
- Tier 3, the V2 gate and the PV group are science decisions, not migration progress.
- Item 15 is a term scan of the runner, not an H05 or H08 test.
- Items 9, 10: canonical lineage ruled Q13 (SCI-EI-NATIVE-STATE-1); OPEN-02 corrected. V0 is byte-identical for SCI-EI-REGIME-1 and SCI-EI-INTERACTION-1 (checker, md5).
- Item 17: H1 to H3 follow from the leaky proportional-loop algebra of the rule (docs/PROGRESS_REVIEW_2026-11-05.md section 1); passing them shows the rule is wired and acts, not a network finding.
- Item 17 is partial by design and must not be read as H being calibrated. Items 19 and 20 stay at 0 so the limits stay visible.
- Items 17, 19, 20 were defined after the H results; ratified by the owner (Q14, 2026-11-04).
- Paradigm pages (omission; global-local oddball, 11da5b2) are not scored: category item counts are fixed by the owner.
- Item 37 was checked on a copy of the repository (byte-identical rebuild).
- OPEN-21 (b8d492e) bears on item 30 (provenance), not item 28; v2 numbering caused the mix-up. Item 28 stays 0 (OPEN-08).
- Adopted 2026-11-04: v3 moved to docs/checklist-score-v3.md; scripts/build_dashboard.py header says v4; dashboard rebuilt.
- Item 18: owner ruled keep (Q15); it is not retired. No spec is drafted.
- Checker (independent, on the 53-point draft) proposed 57 for that draft: disagreements on items 7, 38, 39 (+1 each; definitions), 22 (-1) are not adopted here without evidence beyond definition reading; recorded for ruling.
