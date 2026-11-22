---
name: jomission-gate-runner
description: Execute a predeclared jomission scientific gate (a battery, bracket, scan, or design numerics) without weakening its criteria. Use before running any jomission simulation or assay that yields a verdict, and when sealing its result.
---

# Gate runner

The order is fixed. A failed step is a STOP. Do not work around it.

1. Inspect authority. `python scripts/project_check.py` passes. The task is `next_authorized_task` in `manifests/todo.json`, or the task the reviewer names. Read its `parent_authority`, `frozen`, `procedure`, `acceptance` and `stop_condition`. Load the domain skills the work touches. For neuron models and units, load `biophysical-modeling`. For numerics, precision and seeds, load `numerical-computing`.
2. Verify parent. The parent lineage's result commit is an ancestor of HEAD (`git merge-base --is-ancestor`). Its sealed files are unchanged (tier 0 `test_sealed_artifacts_unchanged`). If the parent is unmerged, branch from the parent result commit, merge main, and pin the parent blobs in the spec.
3. Verify frozen variables. Each `frozen` entry maps to a file, blob, seed or constant you can name. Anything you would still have to choose is not frozen. Add it to the spec in step 4 before you run.
4. Verify the pre-execution seal. The bracket, rule, criteria, gate profile and stop states are in a committed spec. The spec is pushed before execution. Gate thresholds come from `manifests/gates/*.json` through `jomission.harness.gates`, with profile `prospective_v2` for new runs. If no gate file applies, use the lineage spec's own criteria. Input schedules pass `jomission.harness.drive.check_additive_schedule`.
5. Visualize before work (VISUALIZATION_CONTRACT, `manifests/visualization_contract.json`). Render V0 `jaxfne_network_hspice.png` from the construction. Then render V1 `raster_initial_1000ms.png` of the native regime. Use `scripts/render_visualization_contract.py`. Look at both before you interpret anything. A lineage that builds no model sets `visualization` to `{"exempt": "NOT_A_MODEL_LINEAGE", "reason": "..."}` instead.
6. Execute exactly the declared runs (the tier 3 test named by the TODO), one process at a time. For any new mechanism, record the configured, realized, executed and effective values.
7. Classify. Evaluate the gates through the harness. `jomission.harness.operation.evaluate` writes `checks`, profile, `gate_blob` and the `verdict` label into the result JSON. `gate_blob` is `git hash-object` of the gate file. A failure that matches `manifests/harness/environment_failures.json` is ENVIRONMENT, not a verdict.
8. Test. Run tier 0 and the lineage's own tests. The battery test passes when the procedure ran. A scientific FAIL keeps pytest green.
9. Seal. Commit results separately from the spec. Then:
   - Add the lineage record ([references/lineage_record.md](references/lineage_record.md)).
   - Register sealed outputs in `manifests/sealed_artifacts.json`.
   - Set the status in `manifests/current_state.json` from `verdict`. The gate status is the evidence-verdict suffix, unless a `review_override` holds it. Components carry readings, not verdicts.
   - Update `manifests/todo.json`.
   - Run `python -m jomission.harness.handoff`.
   - Then run `project_check`.
10. Visualize after work. Render V2 `raster_final.png` with the same ordering and window as V1. Then build the V3 atlas with `jomission.visualization.jaxfne_suite_atlas.build_suite_atlas`. The lineage is not complete until all four exist. The record's `visualization` block names them.
11. Stop. Report the verdict with receipts. Wait for review. Do not open the downstream item. The acceptance report must list all four contract artifact paths.

The window summary layout for `gates.evaluate` is in [references/window_schema.md](references/window_schema.md).
