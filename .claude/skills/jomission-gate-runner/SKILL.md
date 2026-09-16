---
name: jomission-gate-runner
description: Execute a predeclared jomission scientific gate (a battery, bracket, scan, or design numerics) without weakening its criteria. Use before running any jomission simulation or assay that yields a verdict, and when sealing its result.
---

# Gate runner

Order is fixed. A failed step is a STOP, not a workaround.

1. **Inspect authority.** `python scripts/project_check.py` passes. The task is
   `next_authorized_task` in `manifests/todo.json` or named by the reviewer.
   Read its `parent_authority`, `frozen`, `procedure`, `acceptance`,
   `stop_condition`.
2. **Verify parent.** The parent lineage's result commit is an ancestor of HEAD
   (`git merge-base --is-ancestor`), and its sealed files are unchanged
   (tier 0 `test_sealed_artifacts_unchanged`).
3. **Verify frozen variables.** Each `frozen` entry maps to a file, blob, seed,
   or constant you can name. Anything you would have to choose now is not
   frozen: add it to the spec in step 4, before running.
4. **Verify the pre-execution seal.** Bracket, rule, criteria, gate profile,
   and stop states are in a committed spec pushed before execution. Gate
   thresholds come from `manifests/gates/*.json` via `jomission.harness.gates`
   (profile `prospective_v2` for new runs). Input schedules pass
   `jomission.harness.drive.check_additive_schedule`.
5. **Execute** exactly the declared runs (tier 3 test named by the TODO), one
   process at a time. Record configured, realized, executed, and effective
   values for any new mechanism.
6. **Classify.** Evaluate gates through the harness; write `checks`, profile,
   estimator metadata, and the `verdict` label into the result
   JSON (field `gate_blob` = `git hash-object` of the gate file). A failure matching `manifests/harness/environment_failures.json` is
   ENVIRONMENT, not a verdict.
7. **Test.** Tier 0 plus the lineage's own tests. The battery test passes when
   the procedure ran; a scientific FAIL keeps pytest green.
8. **Seal.** Commit results separately from the spec. Add the lineage record
   ([references/lineage_record.md](references/lineage_record.md)), register
   sealed outputs in `manifests/sealed_artifacts.json`, set the status in
   `manifests/current_state.json` from `verdict`, update `manifests/todo.json`,
   run `python -m jomission.harness.handoff`, then `project_check`.
9. **Stop.** Report the verdict with receipts and wait for review. Do not
   open the downstream item.

Window summary layout for `gates.evaluate`:
[references/window_schema.md](references/window_schema.md).
