# Fixation boat: sensory encoding and trial plan

Status: OWNER DESIGN; not implemented or experimentally verified.
Owner: Hamm. Scope: jomission-fixationboat. Planning branch: `_temp_solmind`.
Baseline: `jomission@dev@c4c9c6b17a45b4d925b2d72a8bb02f2515850c70`.
This is a planning record, not an authorization to alter frozen omission criteria, execute simulations, or promote to `main`.

## Goal
Learn a closed-loop neural controller that acquires fixation at the origin and maintains it. Preserve the distinction between the boat environment, sensory encoding, hierarchical neural dynamics, motor decoding, and learning. No claim of learned performance before held-out evaluation.

## Owner-specified task
- Two-dimensional boat position `(x,y)`, confined to `[-10,10] x [-10,10]` (20 by 20 map).
- Fixation target `(0,0)`; acquisition radius `r <= 1.0`, where `r = sqrt(x*x+y*y)`.
- After each completed or aborted trial, a 2000 ms intertrial interval (ITI) begins. Fixation is absent, fixation-related currents are zero, and the boat may explore while respecting map bounds.
- After ITI, fixation appears and position-dependent input currents begin immediately. The boat must first enter `r <= 1.0`; that entry starts the active trial. A subsequent exit to `r > 1.0` aborts it.
- Sensory drive travels through the proposed area hierarchy to motor neurons; the learned motor policy should reduce displacement and maintain fixation. A trial's successful completion condition and duration remain UNSET.

## Three observation alternatives, experiment order

Let `dX = x - 0`, `dY = y - 0`, and `u(z) = max(0,z)`. Currents are zero throughout ITI. During fixation availability:

| Order | Case | Sensory currents | N | Information |
|---|---|---|---:|---|
| 1 | Case 2: signed rectification | `[u(dX),u(dY),u(-dX),u(-dY)]` | 4 | Signed position recoverable |
| 2 | Case 1: absolute axes | `[abs(dX),abs(dY)]` | 2 | Both signs lost |
| 3 | Case 3: radial distance | `[sqrt(dX*dX+dY*dY)]` | 1 | Direction lost |

Rationale: establish a directional-information baseline with Case 2, then ablate sign and axis information in Cases 1 and 3. Cases 1 and 3 require action/history-dependent inference for general directional control. No inverse-distance gain is specified: these currents increase with distance from fixation. Physical current units, gain, saturation and neuron mapping are UNSET; do not assume numeric current amplitudes equal position units.

## Proposed implementation and evaluation (not yet authorized for execution)
1. Implement a separate task-state machine: ITI -> acquisition -> active trial -> completed/aborted -> ITI. Keep generic Sailor dynamics unchanged. Define how boundary contact affects velocity without permitting positions outside the map.
2. Expose an interchangeable observation encoder with the three cases and explicit fixation-availability gate. Do not change controller architecture or motor decoding between cases without a recorded comparison amendment.
3. Use the existing intended hierarchy (V1 -> V4 -> FEF; optional additional areas only when justified). Test neural feedforward activity and actual motor effects separately; anatomical intuition is a hypothesis, not a result.
4. Define learning objective before training: minimize radial error, residual velocity and motor effort, with an abort penalty. Weights, learning algorithm, acquisition timeout, active-trial duration, and termination criterion are UNSET and require scientific design approval.
5. Freeze a held-out evaluation set and matched seeds, initial positions, disturbances, training budgets and stopping rules. Metrics: acquisition probability/time, fixation duration, abort rate, radial error, motor effort. Report effect, uncertainty, n, and negative outcomes.
6. Controls: analytic signed-position controller (positive), zero-action controller (negative), shuffled sensory channels (adversarial). Ensure each fixture is non-vacuous and test sign/axis symmetries.

## Immediate checks for a future agent
- Inspect current Jarness `main` start/rules, then jomission `AGENTS.md`, `README.md`, `manifests/current_state.json`, `manifests/todo.json`, and existing fixationboat specifications before editing code.
- Reconcile this owner design with any sealed task timing or prior fixationboat contract; report consequential conflicts rather than silently replacing them.
- First implement and test the state machine and Case 2 encoder on synthetic/scalar oracles; test boundaries, exact `r=1` crossing, zero ITI current, reappearance after 2000 ms, and deterministic replay. Cases 1 and 3 follow as controlled comparisons.
- Keep the plan in `artifacts/` on `_temp_solmind`; do not edit owner-controlled goal/fact records or sealed evidence. No push to `main`.

## Planning workflow ruling
For this conversation's ongoing Sailor plan, each authorized change to this plan should be committed and pushed immediately to `HNXJ/jomission` branch `_temp_solmind`, with a meaningful commit message. This standing instruction applies to recoverable planning changes under `artifacts/` only; it does not waive safety, conflict, verification, or promotion gates. Before each update, read the remote branch HEAD and the file, preserve concurrent edits, then confirm the remote commit and report its SHA.
