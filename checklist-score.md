# Checklist score

Scope: jaxfne 0.5.8 migration and the project checks it touches. 50 items, 2 points each, sum 100.
Scoring: 2 = met with evidence, 1 = partially met, 0 = not met. Scored on f9df20d.
The checklist may be adjusted as the goals sharpen. Done means 100/100.

**Current score: 73/100**

| # | Objective | Score (of 2) | Evidence |
|---|---|---|---|
| 1 | jaxfne pinned 0.5.8 in pyproject.toml | 2 | pyproject.toml:18 |
| 2 | installed engine = pin = execution authority | 2 | project_check: jaxfne identity 0.5.8 |
| 3 | current_state execution_authority evidence = 0.5.8 record | 2 | manifests/current_state.json |
| 4 | 0.5.8 migration record has delta, tiers, unknowns, verdict | 2 | results/jaxfne_058_migration.json |
| 5 | migration record registered in sealed registry (append-only) | 2 | manifests/sealed_artifacts.json, f9df20d |
| 6 | edge-identity comparison script committed as receipt | 0 | script in scratchpad only; not in repo |
| 7 | canonical EdgeList identical to 0.4.25 baseline (15 fields) | 2 | comparison run; 10666 edges, all fields equal |
| 8 | canonical config_hash frozen constant = owner-accepted value | 2 | tests/test_rf.py:37 = 05f3d334451c53d2 |
| 9 | canonical network builds on 0.5.8 | 2 | tier 0/2 pass |
| 10 | within-area locality realized (within_locality + pruning) | 2 | builder.py; edge identity |
| 11 | inter-column delay args None; delay applied by shim | 2 | builder.py delay sites |
| 12 | cell-type map None = identity default | 2 | builder.py cell-map sites |
| 13 | field() uses realized proxy form (source_mode removed) | 2 | builder.py; runtime_regression |
| 14 | Retina declared in areas metadata | 2 | scripts/whole_system_ei_native_state_arm.py |
| 15 | per-trial runtime seed equals Simulation seed | 2 | jomission/simulation/exposure.py |
| 16 | tier 0 passes on final state | 2 | 208 passed |
| 17 | tier 1 (non-simulating) passes | 0 | 2 failed: UnicodeDecodeError on sealed v21b files |
| 18 | tier 2 passes on final state | 2 | 402 passed, 3 skipped, 0 failed |
| 19 | tier 3 authorized and run | 0 | no authorizing lineage |
| 20 | sealed v21b Unicode read failures fixed | 0 | blocked: sealed test files may not be edited |
| 21 | test_runtime_regression keeps sparse/proxy coverage | 1 | fixture changed; dense path still tested |
| 22 | test_rf frozen-hash test passes | 2 | tests/test_rf.py |
| 23 | test_area_local_field frozen-hash test passes | 2 | tests/test_area_local_field.py |
| 24 | atlas tests pass | 2 | test_atlas_suite passes with plotly in venv |
| 25 | plotly declared in pyproject | 0 | required by tests, not declared |
| 26 | handoff regenerated and state test passes | 2 | docs/HANDOFF_CURRENT.md |
| 27 | sealed-artifact and call-site guards pass | 2 | test_harness_guards: 11 passed |
| 28 | project_check full PASS | 2 | PROJECT_CHECK_PASS on f9df20d |
| 29 | scientific head ancestry check passes | 2 | project_check: main be0db96 in history |
| 30 | todo / current_state consistent | 2 | project_check: todo, current_state [ok] |
| 31 | working tree clean and pushed | 2 | f9df20d pushed |
| 32 | draft PR #1 exists for designated branch | 2 | HNXJ/jomission#1 |
| 33 | PR #1 body reflects final state | 0 | body still describes the old blocker |
| 34 | PR checks green | 0 | no check runs reported |
| 35 | commit messages follow attribution rules | 2 | no model identifier in commits |
| 36 | work on designated branch (not _temp) | 2 | cowork/focused-ptolemy-dz7hwc |
| 37 | exposure.py continuation defect fixed | 0 | pre-existing, out of scope, not fixed |
| 38 | known defects recorded in migration record | 2 | coverage_loss, known_defects_out_of_scope |
| 39 | jaxfne refusals audited against its rules | 1 | audited in-session; no in-repo doc |
| 40 | no unreported jaxfne defect | 1 | none found; not checked against an issue tracker |
| 41 | owner Q1 (frozen hash) applied | 2 | accepted new hash |
| 42 | owner Q2 (tier 2) applied | 2 | tier 2 run and recorded |
| 43 | owner Q3 (Unicode fix) applied | 0 | not applied: sealed files |
| 44 | owner Q4 (PR body update) applied | 0 | not yet applied |
| 45 | owner order: 0.5.8 adopted as source of truth | 2 | authority promoted |
| 46 | cloud setup handout documented | 0 | not in repo |
| 47 | checklist-score.md exists and is scored from evidence | 2 | this file |
| 48 | no sealed result recomputed | 2 | sealed artifacts unchanged |
| 49 | pre-existing sealed-file defect flagged to owner | 2 | reported with options |
| 50 | scientific gate V2_LOCAL_OPERATION status | 0 | FAIL; outside this migration |

## Next to raise the score
- Commit the edge-identity receipt script to the repo (item 6).
- Fix or re-decide the sealed Unicode reads (items 17, 20, 43); needs a correction record or unseal.
- Restore sparse/proxy coverage (item 21); plotly declaration (item 25).
- Update PR body (items 33, 44); CI checks (item 34).
- Fix exposure.py continuation (item 37); cloud handout (item 46).
- Scientific gate V2_LOCAL_OPERATION (item 50) is outside this migration.
