# Checklist score (v2)

Scope: the jomission project goals, not only the jaxfne migration. Redefined 2026-11-03 at owner request.
50 items, 2 points each, sum 100. Scoring: 2 = met with evidence, 1 = partial, 0 = not met.
The previous 98/100 revision is kept at docs/checklist-score-v1.md. Its migration-only scope is now one category.

**Current score: 85.2/100**

## By category
| Category | Score | Items |
|---|---|---|
| Engine migration | 20 / 20 | 10 |
| Models | 16 / 16 | 8 |
| Harness and evidence | 19.6 / 20 | 10 |
| Science program | 3 / 16 | 8 |
| Dashboard and figures | 20 / 20 | 10 |
| Documentation and reproducibility | 6.6 / 8 | 4 |

## Items
| # | Category | Objective | Score (of 2) | Evidence |
|---|---|---|---|---|
| 1 | Engine migration | jaxfne pinned 0.5.8 in pyproject.toml | 2 | pyproject.toml:18 |
| 2 | Engine migration | installed engine = pin = execution authority | 2 | project_check: jaxfne identity 0.5.8 |
| 3 | Engine migration | canonical EdgeList identical to 0.4.25 baseline (15 fields) | 2 | scripts/jaxfne_edge_identity.py; reproduced at a9fcb65 (10666 edges, 15 fields, exit 0) |
| 4 | Engine migration | frozen canonical config_hash = owner-accepted value | 2 | tests/test_rf.py:37 = 05f3d334451c53d2 (owner Q1) |
| 5 | Engine migration | jaxfne refusals audited against their rules | 2 | docs/JAXFNE_058_AUDIT.md |
| 6 | Engine migration | jaxfne issue tracker checked for reported defects | 2 | HNXJ/jaxfne: 0 issues, 119 PRs (gh api, add_repo push) |
| 7 | Engine migration | tier 0 passes on head | 2 | tier 0 on 860e330: 216 passed, 0 failed |
| 8 | Engine migration | tier 1 non-simulating passes on head | 2 | tier 1 non-simulating on 860e330: 49 passed, 1 skipped, 0 failed |
| 9 | Engine migration | tier 2 passes on head | 2 | tier 2 on 860e330: 419 passed, 3 skipped, 0 failed |
| 10 | Engine migration | owner order 0.5.8 adopted as source of truth | 2 | execution_authority promoted; record results/jaxfne_058_migration.json |
| 11 | Models | canonical 400-neuron network builds on 0.5.8 | 2 | build_jomission_model; tier 2 |
| 12 | Models | within-area locality realized (within_locality + pruning) | 2 | builder.py; edge identity |
| 13 | Models | inter-column delays applied by shim (args None) | 2 | builder.py delay sites; edge identity |
| 14 | Models | cell-type map = identity default | 2 | builder.py cell-map sites |
| 15 | Models | minimal model = canonical 400-neuron build (owner-defined) | 2 | owner confirmed 2026-11-03; tier 2 |
| 16 | Models | lite model V1+V4, 200 neurons, builds and simulates finitely | 2 | build_jomission_lite_model; tests/test_lite_model.py |
| 17 | Models | field, Retina and areas declarations realized | 2 | builder.py field; scripts/whole_system_ei_native_state_arm.py |
| 18 | Models | per-trial seed consistency in exposure | 2 | exposure: one simulation per trial, all trials aggregated (290e5e6); tests/test_exposure_continuation.py: 4 tests fail on pre-repair code, pass on fix |
| 19 | Harness and evidence | project_check full PASS on head | 2 | PROJECT_CHECK_PASS at 860e330 |
| 20 | Harness and evidence | sealed registry and guards pass | 2 | test_harness_guards: 11 passed at 860e330 |
| 21 | Harness and evidence | CI project-check success on head | 1.6 | CI tier0 verified on a9fcb65 (run 126236521645); current head run pending |
| 22 | Harness and evidence | handoff regenerated; state consistent | 2 | docs/HANDOFF_CURRENT.md; project_check [ok] |
| 23 | Harness and evidence | working tree clean and pushed to designated branch | 2 | cowork/focused-ptolemy-dz7hwc |
| 24 | Harness and evidence | draft PR #1 body reflects final state | 2 | PR #1 updated after review |
| 25 | Harness and evidence | sealed v21b/v21c reads decodable (cp1252), reseals recorded | 2 | owner option A; resealed entries; allowlist owner-approved |
| 26 | Harness and evidence | sparse within-area coverage in tests | 2 | tests/test_within_area_sparse_coverage.py |
| 27 | Harness and evidence | migration record correction is additive and registered | 2 | results/jaxfne_058_correction_1.json (registered) |
| 28 | Harness and evidence | commits follow attribution rules | 2 | Claude-Session trailer on every commit; no model identifier in commit text (system rule) |
| 29 | Science program | tier 3 scientific batteries run under an authorized TODO | 0 | owner declined (2026-11-03); PV group pending decision; not run |
| 30 | Science program | V2 local operation gate (manifests/gates/v2_local_operation.json) | 0 | status FAIL (scientific result, not this migration) |
| 31 | Science program | visualization contract V0 schematic for the migrated canonical lineage | 0 | not produced; applicability to the migrated lineage to be confirmed |
| 32 | Science program | visualization contract V1 initial 1 s raster | 0 | not produced |
| 33 | Science program | visualization contract V2 final raster | 0 | not produced |
| 34 | Science program | visualization contract V3 atlas | 0 | not produced |
| 35 | Science program | SCI-EI-NATIVE-STATE-1 sealed verdict in repo | 2 | current_state.json: MIXED_STATE_MECHANISM, sealed |
| 36 | Science program | open scientific question (SST repeated spiking) answered | 1 | partial: SST_STATE_ADVANTAGE; single-neuron f-I (docs/receipts/fi_single_neuron_v1.json): VIP silent at preset drive, fires at jomission b=0.20. Network-level question not closed; AGSDR sweep blocked (SCI-AGSDR-CALIBRATION) |
| 37 | Dashboard and figures | docs/DASHBOARD.html committed, no external resources | 2 | docs/DASHBOARD.html committed (d155593); Pages publish on dev branch (2018e30); no external resources |
| 38 | Dashboard and figures | dashboard generated by scripts/build_dashboard.py (deterministic) | 2 | rebuild byte-identical |
| 39 | Dashboard and figures | tier 0 figure with receipt | 2 | docs/receipts/receipts.json |
| 40 | Dashboard and figures | tier 1 figure with receipt | 2 | docs/receipts/receipts.json |
| 41 | Dashboard and figures | tier 2 figure with receipt | 2 | docs/receipts/receipts.json |
| 42 | Dashboard and figures | tier 3 shown as open with its reason | 2 | receipts open status, owner declined |
| 43 | Dashboard and figures | edge-identity table on the dashboard (text, not a chart) | 2 | receipts.json edge_identity |
| 44 | Dashboard and figures | checklist score chart by category and item | 2 | inline SVG, rendered in Chromium |
| 45 | Dashboard and figures | owner decisions table | 2 | receipts.json owner_decisions |
| 46 | Dashboard and figures | open items shown with reasons | 2 | receipts.json open_items |
| 47 | Documentation and reproducibility | cloud setup handout | 2 | docs/CLOUD_HANDOUT.md |
| 48 | Documentation and reproducibility | fresh-clone reproduction of the setup | 1 | handout setup steps run in this container only; not on a fresh cloud image |
| 49 | Documentation and reproducibility | checklist versioned with history | 2 | docs/checklist-score-v1.md (previous revision) and this file |
| 50 | Documentation and reproducibility | dashboard covers all results/*.json and figures (inventory and figure list; not rendered) | 1.6 | docs/DASHBOARD.html: 367 results rows, 6 docs/_static figures listed; figures not visualized |

## Notes
- Visualization contract (manifests/visualization_contract.json) requires V0 to V3 for every non-grandfathered lineage. Whether the migrated canonical lineage is exempt is not decided; items 31 to 34 stay at 0 until it is.
- Tier 3, the V2 gate, and the PV group are science decisions. They are not counted as migration progress.

## Simulation program (status 2026-11-04; not yet scored)
Scoring these needs a re-weighting decision: the 50 scored items total 100 points, so adding rows means retiring some current rows. Until then they are listed here with their evidence.

| Sim | Objective | Status | Evidence |
|---|---|---|---|
| S1 | AGSDR stage 0: 24 candidates x seed 0, 2000 ms, n=400 (24 sims) | READY, not run | design 0726c5d; amendment 1 cf82168; runner b535876 (dry-run 52 sims, audit clean); TODO SCI-AGSDR-CALIBRATION OPEN, next_authorized_task |
| S2 | AGSDR stage 1: top 8 x seeds {0,1} (16 sims) | pending S1 | design step 3 |
| S3 | AGSDR stage 2: top 3 x seeds {0..3} (12 sims) | pending S2 | design step 4 |
| S4 | Single-neuron f-I (NumPy assay, exploratory) | done, not engine-validated | docs/receipts/fi_single_neuron_v1.json; provenance v1_provenance.json |
| S5 | Engine-equivalence of the f-I assay | not done | required before engine claims (OPEN, review item) |
| S6 | PV drive scan (matched-state) | draft only, not approved, not in repo | scratch draft; pre-checks from JaxFNE handoff |
| S7 | Visualization V0-V3 for the canonical lineage | not produced | manifests/visualization_contract.json |
| S8 | Build and simulation speed | done (warm build 1.05 s; first simulate compile ~1.1 s) | docs/receipts/compile_cache_benchmark.json (3 pairs) |
