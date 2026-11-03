# Checklist score

Scope: jaxfne 0.5.8 migration (items 1-50, 80 points) and the dashboard (items 51-60, 20 points, owner decision 2026-11-03: dashboard is 20% of the score).
Migration items: 1.6 points each (50 x 1.6 = 80). Dashboard items: 2 points each (10 x 2 = 20). Sum 100. Scoring: 2 = met with evidence, 1 = partially met, 0 = not met.
Scored on a9fcb65 after a full review (reproduction, CI, record audit). The checklist may be adjusted as the goals sharpen. Done means 100/100.

**Current score: 97.6/100**

| # | Objective | Score (of 2) | Evidence |
|---|---|---|---|
| 1 | jaxfne pinned 0.5.8 in pyproject.toml | 1.6 | pyproject.toml:18 |
| 2 | installed engine = pin = execution authority | 1.6 | project_check: jaxfne identity 0.5.8 |
| 3 | current_state execution_authority evidence = 0.5.8 record | 1.6 | manifests/current_state.json |
| 4 | 0.5.8 migration record has delta, tiers, unknowns, verdict | 1.6 | results/jaxfne_058_migration.json (sealed) + results/jaxfne_058_correction_1.json (registered); correction supersedes stale fields |
| 5 | migration record registered in sealed registry (append-only) | 1.6 | manifests/sealed_artifacts.json |
| 6 | edge-identity comparison script committed as receipt | 1.6 | scripts/jaxfne_edge_identity.py (dump/compare receipt) |
| 7 | canonical EdgeList identical to 0.4.25 baseline (15 fields) | 1.6 | comparison run; 10666 edges, all fields equal |
| 8 | canonical config_hash frozen constant = owner-accepted value | 1.6 | tests/test_rf.py:37 = 05f3d334451c53d2 |
| 9 | canonical network builds on 0.5.8 | 1.6 | tier 0/2 pass; n_per_area=100 build |
| 10 | within-area locality realized (within_locality + pruning) | 1.6 | builder.py; edge identity |
| 11 | inter-column delay args None; delay applied by shim | 1.6 | builder.py delay sites |
| 12 | cell-type map None = identity default | 1.6 | builder.py cell-map sites |
| 13 | field() uses realized proxy form (source_mode removed) | 1.6 | builder.py; runtime_regression |
| 14 | Retina declared in areas metadata | 1.6 | scripts/whole_system_ei_native_state_arm.py |
| 15 | per-trial runtime seed equals Simulation seed | 1.6 | jomission/simulation/exposure.py |
| 16 | tier 0 passes on final state | 1.6 | tier 0: 208 passed at a9fcb65 |
| 17 | tier 1 (non-simulating) passes | 1.6 | tier 1 non-simulating: 45 passed, 1 skipped, 0 failed at a9fcb65 |
| 18 | tier 2 passes on final state | 1.6 | tier 2 on 588da38: 405 passed, 3 skipped, 0 failed |
| 19 | tier 3 authorized and run | 0.0 | owner declined tier-3 authorization (2026-11-03); blocked by owner |
| 20 | sealed v21b Unicode read failures fixed | 1.6 | all three v21b/v21c sealed reads fixed cp1252; reseals recorded; allowlist owner-approved |
| 21 | test_runtime_regression keeps sparse/proxy coverage | 1.6 | tests/test_within_area_sparse_coverage.py (sparse p_connect realized) |
| 22 | test_rf frozen-hash test passes | 1.6 | tests/test_rf.py |
| 23 | test_area_local_field frozen-hash test passes | 1.6 | tests/test_area_local_field.py |
| 24 | atlas tests pass | 1.6 | test_atlas_suite passes with plotly in venv |
| 25 | plotly declared in pyproject | 1.6 | pyproject.toml dev extra: plotly>=5 |
| 26 | handoff regenerated and state test passes | 1.6 | docs/HANDOFF_CURRENT.md |
| 27 | sealed-artifact and call-site guards pass | 1.6 | test_harness_guards: 21 passed |
| 28 | project_check full PASS | 1.6 | PROJECT_CHECK_PASS at a9fcb65 |
| 29 | scientific head ancestry check passes | 1.6 | project_check: main be0db96 in history |
| 30 | todo / current_state consistent | 1.6 | project_check: todo, current_state [ok] |
| 31 | working tree clean and pushed | 1.6 | tree clean at a9fcb65 (verified in review) |
| 32 | draft PR #1 exists for designated branch | 1.6 | HNXJ/jomission#1 |
| 33 | PR #1 body reflects final state | 1.6 | PR #1 body refreshed to a9fcb65 state (review) |
| 34 | PR checks green | 1.6 | CI tier0 success on current head (check run 126236521645) |
| 35 | commit messages follow attribution rules | 1.6 | no model identifier in commits (system rule); Claude-Session trailer on every commit; Co-Authored-By model line deliberately omitted |
| 36 | work on designated branch (not _temp) | 1.6 | cowork/focused-ptolemy-dz7hwc |
| 37 | exposure.py continuation defect fixed | 1.6 | exposure.py one call per trial; tests/test_exposure_continuation.py fails on old code, passes on fix |
| 38 | known defects recorded in migration record | 1.6 | coverage_loss, known_defects_out_of_scope |
| 39 | jaxfne refusals audited against its rules, no unreported defect | 0.8 | docs/JAXFNE_058_AUDIT.md: every refusal audited against its rule; 'no unreported defect' NOT checked against an issue tracker (scored 1) |
| 40 | owner Q1 (frozen hash) applied | 1.6 | accepted new hash |
| 41 | owner Q2 (tier 2) applied | 1.6 | tier 2 run and recorded |
| 42 | owner Q3 (Unicode fix) applied | 1.6 | owner Q3 applied: allowlist update for test_v21b_operation.py |
| 43 | owner Q4 (PR body update) applied | 1.6 | owner Q4 applied: PR #1 body updated |
| 44 | owner order: 0.5.8 adopted as source of truth | 1.6 | authority promoted |
| 45 | cloud setup handout documented | 1.6 | docs/CLOUD_HANDOUT.md written |
| 46 | checklist-score.md exists and is scored from evidence | 1.6 | this file; scores re-verified at a9fcb65 in review |
| 47 | no sealed result recomputed | 1.6 | sealed artifacts unchanged apart from documented reseals |
| 48 | pre-existing sealed-file defect flagged to owner | 1.6 | reported with options |
| 49 | minimal jomission model (canonical 400-neuron build, n_per_area=100) builds and passes tier 2 on 0.5.8 | 1.6 | owner confirmed the definition (2026-11-03); tier 2 405 passed |
| 50 | lite jomission model (V1+V4, 200 neurons) defined and builds on 0.5.8 | 1.6 | build_jomission_lite_model: V1+V4, n_per_area=100 (200 neurons); tests/test_lite_model.py |

| 51 | docs/DASHBOARD.html committed (self-contained, no external network) | 2 | docs/DASHBOARD.html committed at d155593; no external resources |
| 52 | dashboard generated by scripts/build_dashboard.py from receipts (reproducible) | 2 | scripts/build_dashboard.py; rebuild is byte-identical to the committed file |
| 53 | tier 0 figure with receipt (208 passed, commit a9fcb65) | 2 | receipts.json tier 0: 208 passed, commit a9fcb65 |
| 54 | tier 1 figure with receipt (45 passed, 1 skipped, commit a9fcb65) | 2 | receipts.json tier 1: 45 passed, 1 skipped, commit a9fcb65 |
| 55 | tier 2 figure with receipt (405 passed, 3 skipped, commit 588da38) | 2 | receipts.json tier 2: 405 passed, 3 skipped, commit 588da38 |
| 56 | tier 3 shown as open with its reason (owner declined) | 2 | receipts.json tier 3 status OPEN, reason: owner declined; shown on page |
| 57 | edge-identity figure (15 fields, 10666 edges, exit 0) with method | 2 | receipts.json edge_identity: 15 fields, 0 mismatched, 10666 edges, exit 0 |
| 58 | checklist score chart: migration vs dashboard section, and per-item bars | 2 | inline SVG: section bars and per-item bars rendered; checked in Chromium |
| 59 | owner decisions table with applied/declined status | 2 | receipts.json owner_decisions table, rendered on page |
| 60 | open items shown, including 'minimal' assumption and jaxfne unreported-defect check | 2 | receipts.json open_items rendered on page |

## Changes in this revision (review at a9fcb65)
- Re-verified at head: tier 0, tier 1, project check, CI, edge identity (reproduced from baseline d3eaa4f).
- Downgraded: item 39 (no unreported-defect check) and item 49 (minimal definition is an assumption).
- Record correction: results/jaxfne_058_correction_1.json supersedes stale fields of the sealed migration record; item 4 stays 2 on that basis.
- Evidence refreshed for items 16, 17, 18, 28, 31, 33, 34, 46.

## Status
- 97.6/100. Open: item 19 (tier 3, owner declined, 1.6 points); item 39 (jaxfne issue check, 0.8 points, blocked: this session has no GitHub API access to HNXJ/jaxfne).
