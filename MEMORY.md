# MEMORY.md (jomission) — compact orientation, verified lessons

Full fields (cause, evidence, scope) by id: manifests/harness/lessons.json.
Never store current state, secrets or unverified claims here.

## Lessons

- L1-DOMAIN-SKILL-NOT-LOADED: jomission/qualification/intrinsic.py (V2.1c design numerics) written without loading the biophysical-modeling skill -> jomission-gate-runner fronts every execution; its step 1 now names domain skills to load (added from this lesson)
- L2-STATE-PINNED-IN-TEST: tier-0 failed after a legitimate TODO status change (SCI-V21C-BOUNDARY OPEN_AFTER_HARNESS -> OPEN) -> test now asserts durable invariants only (items exist in a live status; LOCKED gates stay LOCKED); statuses validated against evidence by validate_current_state
- L3-STATUS-PIN-RECURRENCE: owner retired the V2 chain 2026-09-19 and tier-0 failed on test_preserved_science_todo asserting V2.2-RECURRENCE and BLIND-OMISSION are LOCKED -> the chain assertion now accepts the durable set {LOCKED, RETIRED}; a retired chain can never become executable without a new authorization
- L4-NPZFILE-CLOSE-BEFORE-TEMP-CLEANUP: WinError 32 on TemporaryDirectory cleanup in the t4_t5 and t7 artifact tests -> close the NpzFile before the temp context exits; test-harness only, no quarantine
- L5-DYNAMICSTATE-REPLACE-NOT-CONSTRUCT: five Q8 tests failed with TypeError: DynamicState.__new__() missing theta_S, aux, b after the 0.4.24 migration -> rebuild through the namedtuple's own _replace (undeclared fields preserved by construction) and derive hash/shape/verify field lists from DynamicState._fields
- L6-NO-DETACHED-RUNS: tier-2 foreground run aborted twice by the owner; the detached relaunch was killed by the harness (ChildProcess.kill on the spawning shell) -> run long batteries only in the foreground with the owner present, or split them; do not relaunch detached
- L7-PY-EDIT-REFORMATS-FILE: editing jomission/*.py produced whole-file normalization on top of the intended change, four times: t4_t5_analysis.py 293+/93-, builder.py 181+/49-, hdp_attractor.py 1045+/554-, q8_phenotype.py and run.py in 39e8c18 -> treat the first .py diff per file as suspect: confirm with git diff -w that the non-whitespace hunks are only the intended ones, then rely on the covering tests as the behavior arbiter
- L8-ENGINE-MIGRATION-PROCEDURE: engine upgrade (jaxfne 0.4.24 to 0.4.25) with sealed results computed under the old engine -> read-only delta on touched surfaces first, then install, then tier 0+1+2, then seal the migration evidence and promote authority plus the pyproject pin together; no tier-3 replay without an authorizing lineage
- L9-TIER-BEFORE-RERUN: rerunning failed or timed-out test files from a full-suite baseline -> classify every file against manifests/test_tiers.json before any rerun; tier 3 only inside an authorising TODO
- L10-GATE-THEN-COMMIT: check and commit chained in one shell line -> chain checks to commits with '&&' only; after renaming a file, grep manifests for the old path
- L11-STALE-SYSTEM-JAXFNE: `project_check.py` FAIL "jaxfne identity: installed 0.4.25, execution authority 0.5.8" and "editable install" on vwin-pc -> run every jomission command with `D:\venvs\jomission\Scripts\python.exe` (venv, system site-packages visible, editable jaxfne `v0.5.8` worktree `D:\repos\_work\jaxfne_v058` and editable jomission; rebuild: `python -m venv --system-site-packages D:\venvs\jomission`, `pip install --no-deps --no-build-isolation -e <jaxfne@v0.5.8>` then `-e <jomission>`); never pip-install into the shared Python; do not investigate 0.4.25 further
