# Checklist score (v4)

Scope: the jomission project goals, as the program stands on 2026-11-04 (HEAD 7769617, branch cowork/focused-ptolemy-dz7hwc, equal to origin). Draft for owner review; not yet the checklist the dashboard reads.
50 items, 2 points each, sum 100. Scoring: 2 = met with a receipt or commit that was checked, 1 = partial, 0 = not met or not verifiable.
Category maxima (owner, 2026-11-04): Science program 40 (20 items), Documentation and reproducibility 30 (15), Dashboard and figures 20 (10), Rest 10 (5). The item count per category is unchanged from v3.
Earlier revisions: v3 (50, 2026-11-04) at checklist-score.md (repo root); v2 (83.2) in docs/checklist-score-v2.md; v1 (98, migration-only) in docs/checklist-score-v1.md.

Review rule for this draft: every evidence entry is a path or commit sha that was checked with ls, git show or git merge-base in this session. A claim that rests only on a commit message or on the OPEN stack is scored 1 at most, and this is noted in the item.

**Current score: 53/100** (v4, 2026-11-04). Net +3 against v3: Science +3, Documentation +1, Dashboard 0, Rest -1 (see "Changes from v3").

## By category
| Category | Score | Items |
|---|---|---|
| Science program | 17 / 40 | 20 |
| Documentation and reproducibility | 16 / 30 | 15 |
| Dashboard and figures | 14 / 20 | 10 |
| Rest | 6 / 10 | 5 |

## Items
| # | Category | Objective | Score (of 2) | Evidence |
|---|---|---|---|---|
| 1 | Science program | Tier 3 scientific batteries run under an authorized TODO | 0 | owner declined tier 3 (a9fcb65); receipts tier 3 status OPEN in docs/receipts/receipts.json |
| 2 | Science program | V2 local operation gate PASS (manifests/gates/v2_local_operation.json) | 0 | gate V2_LOCAL_OPERATION recorded as FAIL in docs/HANDOFF_CURRENT.md (line 40) |
| 3 | Science program | Viable operating regime found (PV in [12,25], VIP in [8,15], SST in [6,12], E < PV) | 0 | no cell viable (d846eff); manifests/lineage_registry.json, label EI_REGIME_STILL_UNRESOLVED |
| 4 | Science program | Regime verdict recorded as one ratified label (definition-dependent; see Notes) | 2 | manifests/lineage_registry.json (EI_REGIME_STILL_UNRESOLVED recorded); owner ruling Q8 that a recorded sealed verdict counts as met (0a248e2) |
| 5 | Science program | SCI-EI-NATIVE-STATE-1 sealed verdict | 2 | manifests/current_state.json (MIXED_STATE_MECHANISM); results/whole_system_ei_native_state.json, listed in manifests/sealed_artifacts.json |
| 6 | Science program | SST repeated-spiking question answered | 1 | partial: SST_STATE_ADVANTAGE in results/whole_system_ei_native_state.json; network-level question not closed (docs/HANDOFF_CURRENT.md) |
| 7 | Science program | Single-neuron f-I assay (NumPy, exploratory) | 1 | docs/receipts/fi_single_neuron_v1.json; script scripts/fi_single_neuron.py; not engine-validated |
| 8 | Science program | Engine-equivalence of the f-I assay | 0 | not done: docs/receipts/fi_single_neuron_v1_provenance.json records provenance only; no equivalence receipt in docs/receipts/ |
| 9 | Science program | Visualization V0 (schematic) and V1 (initial 1 s raster) for the canonical lineage | 0 | not produced: manifests/visualization_contract.json requires them; docs/_static/ holds no V0 or V1 file (merged from v3 items 9 and 10) |
| 10 | Science program | Visualization V2 (final raster) and V3 (atlas) for the canonical lineage | 0 | not produced: manifests/visualization_contract.json; docs/_static/ holds no V2 or V3 file (merged from v3 items 11 and 12) |
| 11 | Science program | AGSDR calibration design sealed, with amendments 1 to 3 | 2 | docs/designs/SCI-AGSDR-CALIBRATION_design.md (0726c5d); docs/designs/SCI-AGSDR-CALIBRATION_amendment_1.md, _2.md, _3.md (amendment 3 sealed in 43d1d71) |
| 12 | Science program | AGSDR stage 0 executed with receipts (24 candidates, seed 0) | 2 | docs/receipts/agsdr_local/stage0_summary.json (run in a50121b) |
| 13 | Science program | AGSDR stage 0 selection determined under the frozen rule | 0 | NOT DETERMINED: docs/receipts/agsdr_local/stage0_correction_v1.json; OPEN-18 and OPEN-23 in docs/OPEN.md; H08 status OPEN-24 |
| 14 | Science program | AGSDR stages 1 (top 8 x seeds 0,1) and 2 (top 3 x seeds 0..3) executed | 0 | blocked: docs/OPEN.md OPEN-25 (viable-regime verdict) and R6 in the review table below; no stage-1 receipt in docs/receipts/ (merged from v3 items 16 and 17) |
| 15 | Science program | B10 objective audit clean | 2 | docs/receipts/agsdr_local/B10_objective_audit.json (v3 item 19) |
| 16 | Science program | AGSDR Efrac tau correction applied to stage-0 receipts, including B3 | 2 | docs/receipts/agsdr_local/stage0_correction_v1.json; docs/receipts/agsdr_b3_rerun/summary.json (sealed 47d055f) (v3 item 20) |
| 17 | Science program | H controller acts on the network under a sealed test: balance-only rule (drive_bias), predictions H1 to H3 | 1 | docs/designs/H_BALANCE_TEST_spec.md (sealed ce55b01 before run); docs/receipts/h_balance_test/summary.json and A_normal_seed0.json; a7e3ae8. H1 to H3 hold 4 of 4 runs (n = 2 seeds per condition); H4 fails (see item 18). Startup burst result in docs/receipts/h_sync_startup/summary.json (7b21e87). New item |
| 18 | Science program | H balance-only rho and population Fano rise resolved (prediction H4) | 0 | known limit recorded, not resolved: b9f118a; docs/receipts/h_sync_test/summary.json (H9 true, H5 H6 H8 false); synchrony term closed by owner (Q6) with the limit kept. New item |
| 19 | Science program | H balance rule entered in the calibrated model, with its own sealed calibration of kappa, kappa_I, tau_r, tau_H | 0 | not in the calibrated model (85c138b, Q7); H inactive in all sealed 0.5.8 lineages (b1611d8); the rule exists as jomission/dynamics/activity_balance.py, optional and off. New item |
| 20 | Science program | H negative results recorded as results, each with a receipt and a ledger entry | 2 | docs/receipts/h_active_test/summary.json (87acaf5, no network effect); docs/receipts/h_sync_test/summary.json (dc80abe, sync term fails); docs/receipts/h_sync2_test/selection.json (4df8ac9, picked_kappa_s null, selection failed); docs/receipts/h_balance_test/summary.json (a7e3ae8); ledger in docs/OPEN.md OPEN-27. New item (replaces v3 item 18) |
| 21 | Documentation and reproducibility | Cloud setup handout | 2 | docs/CLOUD_HANDOUT.md |
| 22 | Documentation and reproducibility | Fresh-clone reproduction of the setup | 1 | run in this container only; fresh cloud image not tested (OPEN-09 in docs/OPEN.md) |
| 23 | Documentation and reproducibility | Checklist versioned with history | 2 | docs/checklist-score-v1.md; docs/checklist-score-v2.md; checklist-score.md (v3, repo root); docs/checklist-score-v4.md (this draft); git history |
| 24 | Documentation and reproducibility | Results inventory on the dashboard | 1 | inventory section of docs/DASHBOARD.html (built by scripts/build_dashboard.py); figures not rendered (OPEN-14) |
| 25 | Documentation and reproducibility | Open stack current and consistent with audits | 1 | docs/OPEN.md (updated at 7769617: OPEN-19 removed as done, OPEN-29 added); OPEN-22 (PR #1 body stale) and OPEN-28 (full-suite failures, not bisected) open |
| 26 | Documentation and reproducibility | Handoff regenerated and current | 1 | docs/HANDOFF_CURRENT.md (regenerated at 374410d); carries no R1 to R6 status (review table below), so partial |
| 27 | Documentation and reproducibility | TODO consistent with state | 1 | manifests/todo.json; validators pass after 164ad74; rulings OPEN-18 and OPEN-23 to OPEN-26 listed (OPEN-19 removed as done at 7769617) |
| 28 | Documentation and reproducibility | Every claim has a receipt | 0 | OPEN-08 in docs/OPEN.md: four docs make claims with no receipt |
| 29 | Documentation and reproducibility | Sealed-artifact policy enforced | 1 | manifests/sealed_artifacts.json lists results/h_*.json and v21 results, but not the H specs in docs/designs/H_*_spec.md (sealed by commit only, see item 30); in-place reseals recorded (OPEN-04) |
| 30 | Documentation and reproducibility | H-controller specs committed before their runs (pre-registration order, checked by git ancestry) | 2 | 3ca43f3 before 87acaf5; ce55b01 before a7e3ae8; 11c1a0f before dc80abe; 024d2a6 before 069f21d; ac62ff0 before 7b21e87; 7ffa304 before 4df8ac9 (each checked with git merge-base --is-ancestor). Relabeled from v3 item 30 (commit and PR provenance) |
| 31 | Documentation and reproducibility | Gate thresholds single source | 1 | tests/test_gates_single_source.py; AGSDR predicates are copies of runner rules (tested, not imported) |
| 32 | Documentation and reproducibility | Design-to-code traceability for AGSDR | 1 | docs/designs/SCI-AGSDR-CALIBRATION_amendment_3.md; H07 and H11 not yet computed; H08 mismatch OPEN-24 |
| 33 | Documentation and reproducibility | Goals document ratified | 0 | docs/GOALS_PROPOSED.md exists; not ratified by the owner |
| 34 | Documentation and reproducibility | Receipts name generator and hash | 1 | docs/receipts/h_balance_test/summary.json (rule_sha256, script_sha256, spec_sha256); docs/receipts/agsdr_local/stage0_summary.json; older receipts not audited |
| 35 | Documentation and reproducibility | Compile cache: build-time receipt and trust documented | 1 | docs/receipts/compile_cache_benchmark.json (build only); trust decision open (OPEN-17) |
| 36 | Dashboard and figures | docs/DASHBOARD.html committed, no external resources | 2 | docs/DASHBOARD.html; no script, link or http reference found in it |
| 37 | Dashboard and figures | Dashboard generated deterministically by scripts/build_dashboard.py | 2 | rebuilt from a copy of the repo on 2026-11-04: byte-identical to docs/DASHBOARD.html |
| 38 | Dashboard and figures | Tier 0 test-count chart with receipt | 1 | docs/receipts/receipts.json (counts only; not a result figure) |
| 39 | Dashboard and figures | Tier 1 test-count chart with receipt | 1 | docs/receipts/receipts.json (counts only; not a result figure) |
| 40 | Dashboard and figures | Paradigm pages (omission; global-local oddball) current with their sources | 1 | site-src/pages/glo.html and site-src/pages/omission.html; group Paradigms in site-src/manifest.json; commit 11da5b2. Page tests reported passing in that commit, not re-run here. Relabeled from v3 item 40 (tier 2 test-count chart); the tier 2 bar is still drawn in the dashboard tier table |
| 41 | Dashboard and figures | Tier 3 shown as open with its reason | 2 | docs/receipts/receipts.json (tier 3 status OPEN, reason recorded); a9fcb65 |
| 42 | Dashboard and figures | Edge-identity table on the dashboard | 2 | docs/receipts/receipts.json, key edge_identity; table in docs/DASHBOARD.html |
| 43 | Dashboard and figures | Checklist score chart by category and item | 2 | scripts/build_dashboard.py (score_svg, item_svg); docs/DASHBOARD.html |
| 44 | Dashboard and figures | GitHub Pages current with the head | 0 | deploy blocked by github-pages environment rule (OPEN-07 in docs/OPEN.md); .github/workflows/pages.yml |
| 45 | Dashboard and figures | Open items and owner decisions shown with reasons | 1 | docs/receipts/receipts.json keys open_items and owner_decisions; OPEN-18 to OPEN-28 are not on the dashboard |
| 46 | Rest | jaxfne 0.5.8 pinned, installed, adopted as source of truth | 2 | pyproject.toml line 18 (jaxfne==0.5.8); results/jaxfne_058_migration.json; docs/JAXFNE_058_AUDIT.md |
| 47 | Rest | Canonical network identity (edge list, config hash, builds) | 2 | docs/receipts/receipts.json (edge_identity; config_hash); tests/test_lite_model.py; no commit to jomission/network/builder.py after a9fcb65 (git log --ancestry-path a9fcb65..HEAD) |
| 48 | Rest | Tiers 0, 1 and 2 pass on the current head | 0 | tiers 0 to 2 passed at 860e330 (docs/receipts/receipts.json), not on head 7769617; full-suite baseline (worker-reported, OPEN-28, commit b3342af) has 36 failed, 17 timed out and 1 OOM; not attributed to activity_balance; not bisected |
| 49 | Rest | Sealed registry and guards pass on head | 1 | tests/test_harness_guards.py; tests/test_harness_state.py; tests/test_agsdr_stage0_correction.py; not re-run on head 7769617; OPEN-05 and OPEN-06 open against these guards |
| 50 | Rest | CI green on head, working tree clean, pushed | 1 | pushed: origin/cowork/focused-ptolemy-dz7hwc = 7769617 = HEAD; working tree clean apart from this draft; CI run on the head not verified (partial) |

## Review conditions before stage 1 (carried from v3; status as of 2026-11-04)
| R | Condition | Status | Record |
|---|---|---|---|
| R1 | Selection rule: epsilon predicate, dimension set (6 computed vs 7 named) and overflow / weight 1/6 vs 1/7 open | PARTIAL | OPEN-18, OPEN-23; docs/receipts/agsdr_local/stage0_correction_v1.json |
| R2 | Efrac tau-weighted for all 24; H05 under all-E reading gives 2 of 24 pass; H07 and H11 never computed (OPEN-29); H08 hard in objectives, informational in runner | PARTIAL | OPEN-24, OPEN-26, OPEN-29; docs/receipts/agsdr_local/stage0_correction_v1.json |
| R3 | Hard gates can fail: boundary cases on copies of the predicates; other gates not yet exercised on the runner path | PARTIAL | OPEN-20; tests/test_agsdr_stage0_correction.py |
| R4 | Sampling departure: 16 rows redrawn; background values not verified | RECORDED | OPEN-20; docs/receipts/agsdr_local/stage0_correction_v1.json |
| R5 | CV denominator: all 242 E cells; 2 of 24 pass H05 | RULED Q1 (2026-11-04) | OPEN-20, OPEN-26 |
| R6 | Stage 1 requires R1 to R3 resolved, selection determined, vectors and cross-seed rule frozen, viable-regime verdict | BLOCKED | OPEN-18, OPEN-23, OPEN-24, OPEN-25, OPEN-26 |

## Changes from v3
Numbers: items 1 to 8 keep their v3 numbers. v3 items 9 to 12 became 9 and 10; v3 13 to 15 became 11 to 13; v3 16 and 17 became 14; v3 19 and 20 became 15 and 16; v3 21 to 24 and 26 to 29 keep their numbers; v3 30 keeps its number under a new objective; v3 31 to 35 keep their numbers; v3 36 to 40 keep their numbers (40 relabeled); v3 41 to 50 keep their numbers.

| Item (v3 to v4) | v3 score | v4 score | Reason |
|---|---|---|---|
| 9 and 10 (from v3 9 to 12) | 0, 0, 0, 0 | 0, 0 | merged: V0 with V1 and V2 with V3; same status; frees two Science slots |
| 14 (from v3 16 and 17) | 0, 0 | 0 | merged: AGSDR stages 1 and 2 are both blocked (OPEN-25); frees one Science slot |
| 18 (v3 PV drive scan, retired) | 0 | removed | retired, proposed for owner confirmation: no approved spec or receipt exists, and the parameter-scan work now sits in the sealed AGSDR calibration (0726c5d, 43d1d71); slot reused for items 17 to 20 |
| 17 (new) | new | 1 | H balance-only acts on the network (H1 to H3 hold 4 of 4); partial because H4 fails and n = 2 |
| 18 (new) | new | 0 | H4 rho and Fano rise is a recorded known limit, not resolved (b9f118a) |
| 19 (new) | new | 0 | H balance rule is not in the calibrated model (85c138b) |
| 20 (new) | new | 2 | H negative results recorded as results with receipts (87acaf5, dc80abe, 4df8ac9, a7e3ae8) |
| 30 (v3 commit and PR provenance, relabeled) | 1 | 2 | relabeled to H-controller spec order, checked by git ancestry. Provenance is partly settled by the owner ruling OPEN-21 (b8d492e, keep attribution lines); the open PR #1 body issue stays under item 25 as OPEN-22 |
| 40 (v3 tier 2 test-count chart, relabeled) | 1 | 1 | relabeled to paradigm pages (omission, global-local oddball; commit 11da5b2); score unchanged |
| 48 (tiers 0 to 2 on head) | 1 | 0 | full-suite baseline on head has 36 failures, 17 timeouts and 1 OOM (OPEN-28); tiers are shown at 860e330, not on head |
| 25, 26, 27 | 1, 1, 1 | 1, 1, 1 | unchanged (considered for merge; not merged, to keep the points) |
| 47 | 2 | 2 | unchanged; checked that builder.py has no commit after a9fcb65 |
| 50 | 1 | 1 | unchanged; pushed and clean (checked at 7769617), CI not verified (partial) |

Totals by category, v3 to v4: Science 14 to 17; Documentation and reproducibility 15 to 16; Dashboard and figures 14 to 14; Rest 7 to 6; total 50 to 53.

## Notes
- Item 4 (owner ruling Q8, 0a248e2): a recorded sealed verdict counts as met. Under a positive-verdict definition item 4 would be 0, because the recorded verdict is null (EI_REGIME_STILL_UNRESOLVED). Item 5 would need its own ruling.
- Item 17 is partial by design: it measures the balance-only rule, which is not in the calibrated model. It must not be read as evidence that H is calibrated.
- Items 18 and 19 are kept at 0 so the H-controller limits stay visible on the score. The H candidate is balance-only; the synchrony term line is closed (Q6, b9f118a).
- Item 30 is scored from commit order only. The H specs are not in manifests/sealed_artifacts.json, which is why item 29 is 1.
- Item 37 was checked on a copy of the repository. The check was not run in the repository itself, so no tracked file changed.
- The dashboard reads checklist-score.md at the repo root (scripts/build_dashboard.py, ROOT / "checklist-score.md"), not this file. Adopting v4 means moving or copying it to that path and updating the dashboard header, which currently says v3.
- OPEN-21 (b8d492e) says item 28 of v3 is to be re-scored after a checker audit. This draft does not re-score that item; it is reported here as an open audit.
