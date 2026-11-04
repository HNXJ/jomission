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
| 7 | Engine migration | tier 0 passes on head | 2 | tier 0 on cfd7333: 208 passed, 0 failed |
| 8 | Engine migration | tier 1 non-simulating passes on head | 2 | tier 1 non-simulating on cfd7333: 45 passed, 1 skipped, 0 failed |
| 9 | Engine migration | tier 2 passes on head | 2 | tier 2 on cfd7333: 405 passed, 3 skipped, 0 failed; no code change since |
| 10 | Engine migration | owner order 0.5.8 adopted as source of truth | 2 | execution_authority promoted; record results/jaxfne_058_migration.json |
| 11 | Models | canonical 400-neuron network builds on 0.5.8 | 2 | build_jomission_model; tier 2 |
| 12 | Models | within-area locality realized (within_locality + pruning) | 2 | builder.py; edge identity |
| 13 | Models | inter-column delays applied by shim (args None) | 2 | builder.py delay sites; edge identity |
| 14 | Models | cell-type map = identity default | 2 | builder.py cell-map sites |
| 15 | Models | minimal model = canonical 400-neuron build (owner-defined) | 2 | owner confirmed 2026-11-03; tier 2 |
| 16 | Models | lite model V1+V4, 200 neurons, builds and simulates finitely | 2 | build_jomission_lite_model; tests/test_lite_model.py |
| 17 | Models | field, Retina and areas declarations realized | 2 | builder.py field; scripts/whole_system_ei_native_state_arm.py |
| 18 | Models | per-trial seed consistency in exposure | 2 | jomission/simulation/exposure.py; tests/test_exposure_continuation.py |
| 19 | Harness and evidence | project_check full PASS on head | 2 | project_check PASS at cfd7333 |
| 20 | Harness and evidence | sealed registry and guards pass | 2 | test_harness_guards: 21 passed after the last registry append (59c6323) |
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
| 36 | Science program | open scientific question (SST repeated spiking) answered | 1 | partially: SST_STATE_ADVANTAGE; question not closed |
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
