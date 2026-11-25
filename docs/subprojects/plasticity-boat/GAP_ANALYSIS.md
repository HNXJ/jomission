# Plasticity Boat: scientific gap analysis and prospective contract (DRAFT, not frozen)

Status: DRAFT for owner review, 2026-11-25. Nothing here is a result. Nothing here freezes a criterion.
Source folder (paths below are relative to it): `_temp_plasticity_boat/artifacts/programme/design/plans/plasticity_boat/`
(tracked on `dev` from an earlier integration; not migrated, see section 7). Manuscript plan: `.../programme/manuscript/plasticity_boat/main.md` (draft 0, 2026-11-19).

## 1. Question

From `protocol_v0.md` item 1: a distance-only sensor cannot express direction, so a competition between plasticity rules run
with it would measure the sensor. The contribution the programme aims at is a measured statement of which sensing and
control conditions a plasticity-rule comparison needs before it measures the rule. A rigorous negative result is a
legitimate outcome and must be reported as one.

Claim ids (A1–A3) from an earlier draft were not found in any file in the repo; they are not used here.

## 2. What exists

| Item | State | Evidence (source folder) |
|---|---|---|
| 10-neuron fixture | results R-01 to R-12 recorded; R-12 complete, 20 of 20 starts | `protocol_v0.md` item 1, section 7 |
| Criteria v1 (dev-100) | frozen; R2 and R4 fail | `dev100_gate_record.md`, `HANDOFF.md` line 23; `criteria_v1_frozen.md` sha256 d57fd981… (matches `criteria_v2.md` line 3) |
| Criteria v2 (dev-100) | approved 2026-11-19; resting sweep recorded; C3 stops at step 3 | `criteria_v2.md`; `receipts/dev100_v2/resting/`; `HANDOFF.md` line 24 |
| V0 regression, named JaxFNE 0.5.9 | errors and positions max abs diff 0.0 | `receipts/dev100_v2/regression/V0_regression.json`; `V0_source_0.5.9.log` last line `V0 exact=True` |
| V0 regression, pinned 0.5.8 | fails: `jaxfne.paradigm` has no `SailorConfig` | `receipts/dev100_v2/regression/V0_pinned_0.5.8.log` line 18 |
| Eight-strategy protocol | PROPOSED, not frozen; S4 and S8 not implemented | `protocol_v0.md` section 3 (table) and its status line 3 |
| Development starts | seed 2025 (20), calibration seed 2027 (10) | `protocol_v0.md` section 5; `HANDOFF.md` line 14 |
| Held-out starts | seed 2026 (40), unused | `protocol_v0.md` section 5 |
| Tracked todo | `PB-DEV100-V2-VALIDATION` in `manifests/todo.json` | blocked on O-8 |

## 3. Gaps, in critical-path order

G1. Directional gain (O-8, owner). At G_D = 24, R4 fails at every g_s (`HANDOFF.md` line 24). The diagnostic G_D = 4 at
g_s = 3 passes R3 and R4 but is not selection-eligible (`O8_decision_packet.md` lines 29–46). The R4 pass threshold for G_D
lies in (4, 6) and is not resolved (line 44). Options: criteria v3 with a G_D grid sealed before calibration; keep v2 and
stop at C3 step 3; drop the dev-100 track. No option is chosen.

G2. Decoder (O-9, owner). The decoder sums spikes over each 5-neuron motor group, about 5x the force gain of the mean
(`O8_decision_packet.md` section 8; `HANDOFF.md` line 32). P1 is reported under the sum. Unresolved.

G3. Directional encoder. Owner chose Option A, signed-position population (`HANDOFF.md` line 36; recorded in the trigger of
`PB-DEV100-V2-VALIDATION`). Not yet built into a passing run.

G4. Positive control. Two different controls are at issue:
- 10-neuron fixture, EX-24: 20 development starts (`criteria_v1_frozen.md`, "Positive control and regression";
  `code/directional_positive_control.py`). The result file has not been located in the folder. Status unknown until read.
- 100-neuron model (`protocol_v0.md` item 2, EX-9 in `todo_stack.md`): not located as run. No competition run is valid
  until it passes.

G5. Negative controls. `protocol_v0.md` item 3: zero drive and stationary inputs keep the boat at R = 5 (F0 holds at 5.000;
the line does not name the model). The 100-neuron control runs are not located.

G6. Reproducibility of the regression. Pinned JaxFNE 0.5.8 fails the V0 run (`SailorConfig` missing). The passing run is
named "0.5.9" by file name only: the log records no engine version, and its path points to a macOS checkout with Python
3.13. Until the version and SHA are recorded in a receipt, V0 cannot be cited as reproduced.

G7. S4 and S8 are specified but not implemented (`protocol_v0.md` section 3). Each must pass the controls before comparison.

G8. Primary metric (O-6, owner). Fixation occupancy, threshold 1.0, final 20% of intervals (`protocol_v0.md` section 4,
"resolves EX-20"). Not frozen.

G9. Units. Position p, scale d0 and currents have no units in source (`O8_decision_packet.md`, section 1). A claim in metres is
not supported until units are fixed.

G10. Falsifiers are not frozen. Section 5 proposes them; the owner must approve them before any confirmatory run.

G11. Manuscript. No results exist. `main.md` is a plan (draft 0; stage numbers provisional under O-7). A manuscript-quality
score is NOT_SCORABLE until results exist.

## 4. Owner-only decisions

| ID | Decision | Blocks |
|---|---|---|
| O-8 | G_D: criteria v3 grid, keep v2, or drop the track (`HANDOFF.md` line 31) | G1, PB-DEV100-V2-VALIDATION |
| O-9 | Decoder: sum or mean over each 5-neuron motor group (`HANDOFF.md` line 32) | G2 |
| O-6 | Eight-strategy list and primary metric (`HANDOFF.md` line 33) | G7, G8, confirmatory runs |
| O-7 | 15-stage manuscript workflow, not in repo (`HANDOFF.md` line 34) | G11 |
| new | Freeze the falsifiers in section 5 | any confirmatory run |

## 5. Prospective contract (proposed; owner approval required before any confirmatory run)

Question. Does a channel-specific three-factor rule (S5) give better directional control than the baselines S1, S4, S8,
once the directional sensor and the positive control pass? Strategies as listed in `protocol_v0.md` section 3.

Competing hypotheses.
- H0: the sensor, not the rule, sets the outcome. Predicts S1, S4, S5 and S8 do not differ on the primary metric once the
  positive control passes and the sensor carries direction.
- H1: channel identity matters. Predicts S5 exceeds S4 on fixation occupancy.
- H2: the reward baseline and reward structure matter. Predicts S5 exceeds S6 (running-mean baseline) and S7 (shuffled
  distance signal).

Falsifiers (proposed; numbered for freeze).
- F-a: the positive control fails from any development start. No competition is run; the pipeline is reported broken.
- F-b: S1 and zero drive are indistinguishable from S5 on the primary metric. H0 is not rejected; the rule shows no measured effect.
- F-c: S4 exceeds S5. H1 is rejected.

Estimand and statistics (proposed). Unit of inference: the start. Paired contrast against S1. Bootstrap over starts, 10 000
resamples, seed 0, 95% percentile interval, after the effective-variation gate of `protocol_v0.md` section 5. Multiplicity:
the seven contrasts to S1 with Holm correction. Holm is proposed here; `protocol_v0.md` does not state it.

Splits. Development seed 2025 and calibration seed 2027 only, until the owner freezes the protocol. Held-out seed 2026 (40
starts) is used once, after the freeze and all controls pass.

Stopping rules. Stop the competition if the positive control fails (F-a). Stop a strategy that fails its own pre-run controls.
Report a null or negative result with its interval.

## 6. Proposed quality rubric (100 points; freeze before any outcome)

Weights are a proposal.
- Scientific validity (40): controls pass; estimand and falsifiers written before the run; competing explanations tested;
  limitations explicit.
- Evidence and reproducibility (35): each claim traces to a receipt; engine version and SHA recorded; scripts reproduce the
  receipts; negative results preserved.
- Manuscript quality (25): contribution stated; methods reconstructible; figures and tables consistent with the receipts;
  clean build.

Gates (proposed): G-sci (no central claim without a passing control), G-rep (no unrecorded engine version), G-build (clean build).
An UNKNOWN gate caps the score at 89 and a DEFECT gate at 49, as in the LANSA rubric.

A rigorous negative result can earn full quality credit. Without executed results the score is NOT_SCORABLE.

## 7. Canonical location (proposal; nothing executed)

- Narrative and contract: `docs/subprojects/plasticity-boat/` (this folder; matches `lansa/`, `oxm/`, `fixation-sailor/`).
- Code: `jomission/plasticity_boat/` after review, not under `_temp_`.
- Receipts: `docs/receipts/plasticity_boat/`, one file per run, recording engine version and SHA.
- `_temp_plasticity_boat/` stays until every file is reconciled: compare each file with its copy, confirm receipt hashes,
  then remove the original in a separate, owner-approved commit.

## 8. Next actions (agent; reversible; no runs)

1. Locate or produce the EX-24 result and record its status (G4). Closes when: a receipt names the run, its seeds and its verdict.
2. Record the engine version and SHA for the V0 regression (G6). Closes when: a receipt names both.
3. Draft S4 and S8 specifications with their controls for owner review (G7).
4. Write the falsifiers as numbered, testable statements for owner freeze (G10).

Blocked on the owner: O-8, O-9, O-6, O-7, and the falsifier freeze.
