# Worker packet: verdict mapping without scientific rerun

Baseline: HNXJ/jomission main b0b7fb84fbe24fe2f296e7404015d8c21f18c748.
Plan: docs/JOINT_OBJECTIVE_PLAN_2026-11-12.md; static finding: docs/receipts/static_defect_probe_2026-11-12.md.
Status: READY FOR LOCAL STATIC/UNIT TEST; no tier-3 authorization.

Goal: make verdict selection independently testable without changing sealed V2.1 scientific meaning.
Read: AGENTS.md, Jarness rules/start.md, manifests/test_tiers.json, tests/test_v21_operation.py, jomission/harness/validate.py, results/v21_lineage.json and gate definition.
Write: a non-writing unit test plus the smallest pure verdict-selection helper only if necessary. No changes to results/, scientific thresholds, frozen tests, historical verdicts or execution authority.
Accept: synthetic inputs establish PASS when any regime passes, FAIL when all regimes have complete rates and none passes, UNRESOLVED when evidence incomplete; verify known-positive/negative/null and mutation cases. Show red baseline against planted wrong implementation, green after minimal repair; existing tier 0 and targeted tier 1 pass. Ensure helper integration leaves historical output schema and verdict semantics unchanged.
Expect: obtain actual baseline command and output first; UNKNOWN until run. Never execute tests/test_v21_operation.py or tests/test_native_geometry.py for this packet: they write scientific result files and are tier 3.
Siblings: other verdict consumers and validators; inspect without changing unrelated lineages.
Stop when: a required scientific criterion or frozen file must change, project check fails for unrelated cause, or two repair attempts fail.
Run >5 min: smoke then checkpoint if supported.
Report: baseline SHA, exact changed paths, red/green terminal receipts, mutation controls, test tier outputs, independent review, unresolved runtime checks.
No local execution has occurred in this packet's preparation.
