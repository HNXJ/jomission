# SCI-AGSDR-CALIBRATION: amendment 2 (authorization state, additive)

Date: 2026-11-04. Authority: owner unblock 2026-11-04; owner gain ruling 2026-11-04 (multiplicative).
Amends: docs/designs/SCI-AGSDR-CALIBRATION_design.md, step "Run only after the TODO item is updated to AUTHORIZED".

## Change
The authorization state is recorded as status OPEN together with next_authorized_task = SCI-AGSDR-CALIBRATION in manifests/current_state.json. "AUTHORIZED" is not a valid TODO status in manifests/todo.json (validator: statuses DONE, OPEN, OPEN_AFTER_HARNESS, BLOCKED, LOCKED, RETIRED, SUPERSEDED, VOID). The state validator requires next_authorized_task to name an OPEN item, so OPEN plus next_authorized_task is the valid form of the authorization the design describes.

## Unchanged
The design, the sampling (amendment 1), the gates, the seeds, the budget, the stop rules.

## Prerequisite artifacts (required by the design before stage 0)
- docs/receipts/agsdr_local/B10_objective_audit.json: clean (0 hits) for scripts/agsdr_local_sweep.py at the sha recorded in the receipt.
- The runner's stage-0 scoring choices are declared in its DEFINITIONS block (CV_ISI needs at least 3 spikes; Fano from 100 ms bins; rho from 400 seeded pairs; Efrac is the frozen proxy).
