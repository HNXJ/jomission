# f-I practical equivalence: criteria v2 (DRAFT, unsigned; no run before ratification)

Drafted 2026-11-15 under Hamm's standing instruction to develop a prospective, separately versioned practical-equivalence specification. Baseline: jomission `e741270` on `_temp_swarm_fi_equiv_20261115`. Frozen engine: jaxfne 0.5.8 (unchanged).

## Relation to v1

`fi_engine_equivalence_criteria.md` (v1, exact counts) and its receipt `fi_engine_equivalence_v1.json` stay as written: verdict NOT_EQUIVALENT (G1 12 of 486 points differ), checklist item 8 stays 0. v2 is a new declaration, not a repair of v1. It was written after v1's outcome was known, so v1's grid cannot serve as its confirmatory data (v1 informed the question, not the thresholds below, which come from the stated physical scales). v2 is scored only on the held-out grid below.

## Question and estimand

Do the NumPy f-I assay (`scripts/fi_single_neuron.py::firing_rates`, O1) and the JaxFNE 0.5.8 dense Izhikevich kernel (E) agree within the practical tolerances below on inputs v1 never used? Estimand per parameter set (same 6 sets and arm definitions as v1): spike count N in 2000 ms, spike times, rheobase, and gain.

## Held-out grid

extra = 0.125 + 0.25 k, k = 0..79 (0.125 to 19.875; 80 points; disjoint from the v1 grid 0, 0.25, ..., 20). Refined rheobase grid: step 0.01 from the O1 rheobase_extra minus 0.25 to plus 0.25 (v1 G4 used only the lower side).

## Gates (thresholds PROPOSED; Hamm rules before any run)

- P1 count: |N_E - N_O1| <= max(1, 0.01 N_O1) at every held-out point of every set. Scale: one spike in 2 s is 0.5 Hz; 1% is far below the binning of any reported rate.
- P2 rheobase: |rheobase_E - rheobase_O1| <= 0.05 on the refined grid, per set (one fifth of the v1 grid step).
- P3 gain: relative difference of the positive-grid slope (`summarize()` definition) <= 1% per set.
- P4 timing, on points with N_E = N_O1: every spike pair (matched by rank) differs by <= 10 steps (1 ms at dt 0.1 ms). Points with unequal counts are reported, not scored by P4.
- Informational, no gate: number of held-out points with exact count equality; max absolute count difference; timing drift versus spike rank; float32 sensitivity.

## Controls (must hold or the verdict is INVALID)

- Negative control (must detect): O1 with d increased by 0.5 for the E set, compared with E by the same code, must fail P1 or P2 or P3 or P4. (v1 showed this perturbation moves counts by up to 5.)
- Determinism: E run twice is bit-identical; O1 against the scalar float64 oracle O2 equals exactly at every 8th held-out point (11 points), as v1 G0.
- Positive control: O1 against itself passes P1 to P4.

## Verdicts

- PRACTICALLY_EQUIVALENT: P1 to P4 hold for all 6 sets and all controls hold.
- NOT_PRACTICALLY_EQUIVALENT: all controls hold and at least one gate fails; report the failing set, point, and sizes.
- INVALID: a control fails or E cannot run through a supported path.
- Whatever the outcome, v1's NOT_EQUIVALENT stands. Item 8 is rescored only if Hamm rules that practical equivalence is what item 8 requires.
- A change of any threshold or grid after a v2 run is a new declaration (v3).

## Dependencies and owner decisions

- Thresholds P1 to P4 and the held-out grid are scientific tolerances; they are proposals until Hamm ratifies or edits them. No v2 run is authorized before that.
- The run needs no tier-3 battery and writes no `results/*.json`; its output would be `docs/receipts/fi_engine_practical_equivalence_v2.json` (unsealed) from a new script, reusing the v1 harness.
- Open attribution (untested): the v1 mismatches may be XLA versus CPython rounding at threshold crossings; v2 does not test the cause.
