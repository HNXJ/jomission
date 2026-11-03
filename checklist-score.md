# Checklist score

Scope: jaxfne 0.5.8 migration, minimal and lite jomission models, and the project checks they touch.
50 items, 2 points each, sum 100. Scoring: 2 = met with evidence, 1 = partially met, 0 = not met.
Scored on 1011990 (tier 2 rerun on this commit). The checklist may be adjusted as the goals sharpen. Done means 100/100.

**Current score: 96/100**

| # | Objective | Score (of 2) | Evidence |
|---|---|---|---|
| 1 | jaxfne pinned 0.5.8 in pyproject.toml | 2 | pyproject.toml:18 |
| 2 | installed engine = pin = execution authority | 2 | project_check: jaxfne identity 0.5.8 |
| 3 | current_state execution_authority evidence = 0.5.8 record | 2 | manifests/current_state.json |
| 4 | 0.5.8 migration record has delta, tiers, unknowns, verdict | 2 | results/jaxfne_058_migration.json |
| 5 | migration record registered in sealed registry (append-only) | 2 | manifests/sealed_artifacts.json |
| 6 | edge-identity comparison script committed as receipt | 2 | scripts/jaxfne_edge_identity.py (dump/compare receipt) |
| 7 | canonical EdgeList identical to 0.4.25 baseline (15 fields) | 2 | comparison run; 10666 edges, all fields equal |
| 8 | canonical config_hash frozen constant = owner-accepted value | 2 | tests/test_rf.py:37 = 05f3d334451c53d2 |
| 9 | canonical network builds on 0.5.8 | 2 | tier 0/2 pass; n_per_area=100 build |
| 10 | within-area locality realized (within_locality + pruning) | 2 | builder.py; edge identity |
| 11 | inter-column delay args None; delay applied by shim | 2 | builder.py delay sites |
| 12 | cell-type map None = identity default | 2 | builder.py cell-map sites |
| 13 | field() uses realized proxy form (source_mode removed) | 2 | builder.py; runtime_regression |
| 14 | Retina declared in areas metadata | 2 | scripts/whole_system_ei_native_state_arm.py |
| 15 | per-trial runtime seed equals Simulation seed | 2 | jomission/simulation/exposure.py |
| 16 | tier 0 passes on final state | 2 | 208 passed (cb15a79) |
| 17 | tier 1 (non-simulating) passes | 2 | tier 1 non-simulating: 45 passed, 1 skipped, 0 failed (51b5685) |
| 18 | tier 2 passes on final state | 2 | tier 2 on 1011990: 404 passed, 3 skipped, 0 failed |
| 19 | tier 3 authorized and run | 0 | no authorizing lineage |
| 20 | sealed v21b Unicode read failures fixed | 2 | all three v21b/v21c sealed reads fixed cp1252; reseals recorded; allowlist owner-approved |
| 21 | test_runtime_regression keeps sparse/proxy coverage | 2 | tests/test_within_area_sparse_coverage.py (sparse p_connect realized) |
| 22 | test_rf frozen-hash test passes | 2 | tests/test_rf.py |
| 23 | test_area_local_field frozen-hash test passes | 2 | tests/test_area_local_field.py |
| 24 | atlas tests pass | 2 | test_atlas_suite passes with plotly in venv |
| 25 | plotly declared in pyproject | 2 | pyproject.toml dev extra: plotly>=5 |
| 26 | handoff regenerated and state test passes | 2 | docs/HANDOFF_CURRENT.md |
| 27 | sealed-artifact and call-site guards pass | 2 | test_harness_guards: 21 passed |
| 28 | project_check full PASS | 2 | PROJECT_CHECK_PASS on cb15a79 |
| 29 | scientific head ancestry check passes | 2 | project_check: main be0db96 in history |
| 30 | todo / current_state consistent | 2 | project_check: todo, current_state [ok] |
| 31 | working tree clean and pushed | 2 | cb15a79 pushed |
| 32 | draft PR #1 exists for designated branch | 2 | HNXJ/jomission#1 |
| 33 | PR #1 body reflects final state | 2 | PR #1 body rewritten to current state |
| 34 | PR checks green | 2 | CI tier0 success on ddbadbf (.github/workflows/project-check.yml) |
| 35 | commit messages follow attribution rules | 2 | no model identifier in commits |
| 36 | work on designated branch (not _temp) | 2 | cowork/focused-ptolemy-dz7hwc |
| 37 | exposure.py continuation defect fixed | 2 | exposure.py one call per trial; tests/test_exposure_continuation.py fails on old code, passes on fix |
| 38 | known defects recorded in migration record | 2 | coverage_loss, known_defects_out_of_scope |
| 39 | jaxfne refusals audited against its rules, no unreported defect | 2 | docs/JAXFNE_058_AUDIT.md (each refusal with rule and source) |
| 40 | owner Q1 (frozen hash) applied | 2 | accepted new hash |
| 41 | owner Q2 (tier 2) applied | 2 | tier 2 run and recorded |
| 42 | owner Q3 (Unicode fix) applied | 2 | owner Q3 applied: allowlist update for test_v21b_operation.py |
| 43 | owner Q4 (PR body update) applied | 2 | owner Q4 applied: PR #1 body updated |
| 44 | owner order: 0.5.8 adopted as source of truth | 2 | authority promoted |
| 45 | cloud setup handout documented | 2 | docs/CLOUD_HANDOUT.md written |
| 46 | checklist-score.md exists and is scored from evidence | 2 | this file |
| 47 | no sealed result recomputed | 2 | sealed artifacts unchanged apart from documented reseals |
| 48 | pre-existing sealed-file defect flagged to owner | 2 | reported with options |
| 49 | minimal jomission model (n_per_area=100) builds and passes tier 2 on 0.5.8 | 2 | canonical 400-neuron build; tier 2 402 passed |
| 50 | lite jomission model defined and builds on 0.5.8 | 0 | 'lite' not defined in repo; builder refuses n_per_area<100 |

## Changes in this revision
- Removed: scientific gate V2_LOCAL_OPERATION (outside this migration) and the duplicate jaxfne-defect item.
- Added: minimal model (passes); lite model (not defined in the repo, scored 0).
- Re-scored: tier 1, Unicode fix (two of three files), owner Q3, jaxfne audit.

## Next to raise the score
- Tier 3 (item 19; 2 points): needs an authorizing lineage from the owner.
- Lite model (item 50; 2 points): not defined in this repo or in jaxfne. Give a definition.
