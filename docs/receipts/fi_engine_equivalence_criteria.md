# f-I engine equivalence: criteria (declared before any engine run)

Declared 2026-11-15 by the orchestrator under Hamm's chat authorization of 2026-11-15 (checklist v4 item 8; OPEN blocker 15, class "agent work"). Baseline jomission fcd8155c. Frozen engine: jaxfne 0.5.8 (`results/jaxfne_058_migration.json`; installed wheel version 0.5.8, project_check "jaxfne identity" ok). JaxFNE dev is not used. No scientific battery, no tier-3 run, no `results/*.json` write, no parameter retune.

Question: does the JaxFNE 0.5.8 dense Izhikevich kernel realize the same single-neuron spike counts as the NumPy f-I assay (`scripts/fi_single_neuron.py::firing_rates`) when both start from the same state?
Estimand: integer spike count in 2000 ms at each of 81 extra-current values, per parameter set.

## Arms

- O1: `firing_rates` as committed (explicit Euler, dt 0.1 ms, post-update threshold v >= 30, reset v=c, u=u+d, v0=c, u0=b*c).
- O2: scalar oracle, a pure-Python float64 loop written in the test from the Izhikevich (2003) equations; not imported from O1.
- E: JaxFNE 0.5.8 dense Izhikevich scan path (`jaxfne/emitters.py`, `step`, line 641 at 0.5.8): one neuron, no synapses, zero noise, constant drive = tonic + extra, dt 0.1 ms, 20000 steps, float64 (`jaxfne.enable_x64()`), v0 and u0 set explicitly to c and b*c. The worker records which entry point realized E and why it is a supported path.

## Parameter sets (6)

E, PV, SST (engine preset), VIP (engine preset), SST (jomission b 0.21), VIP (jomission b 0.20). Parameters are read from `jaxfne.emitters.IZHIKEVICH_CELL_TYPE_DEFAULTS` plus the two b overrides in `fi_single_neuron.py`. Extra grid: `fi_single_neuron.EXTRA` (81 points, 0 to 20 step 0.25).

## Gates and tolerance (fixed here)

- G0 oracle self-check: O1 equals O2 in integer spike counts at every 8th grid point (11 points) of all 6 sets. Tolerance: exact equality.
- G1 kernel equivalence (primary): E equals O1 in integer spike counts at all 6 x 81 = 486 points. Tolerance: exact equality. No ulp allowance.
- G2 default initialization (informational, not gating): E run with the engine's own v0/u0 against O1; report number of mismatching points per set, max absolute count difference, first mismatch.
- G3 float32 sensitivity (informational, not gating): E in float32 against O1; report mismatches per set and the fraction with absolute count difference at most 1.
- Negative control (must detect): O1 with d increased by 0.5 for the E set must differ from E in at least one grid point under the same comparison code as G1. If it differs nowhere the comparison is vacuous and the verdict is INVALID.

## Verdicts

- EQUIVALENT: G0 and G1 hold and the negative control detects.
- NOT_EQUIVALENT: G0 holds, the negative control detects, and G1 fails. Report mismatching points per set, max absolute count difference, first mismatch (set, extra, O1 count, E count) and the gap's size in drive units.
- INVALID: G0 fails, or the negative control detects nothing, or E could not be run through a supported path.
- A NOT_EQUIVALENT or INVALID outcome is kept as the result. Neither O1 nor E is altered to reach agreement; a change of the oracle or tolerance after seeing outcomes is a new declaration.

## Outputs (the worker writes only these)

- `tests/test_fi_engine_equivalence.py` (tier per `manifests/test_tiers.json`; the worker lists it there at the lowest tier that covers it; it must not write `results/*.json`).
- `scripts/fi_engine_equivalence.py` writing `docs/receipts/fi_engine_equivalence_v1.json` (unsealed receipt, with engine version, dtype, the entry point used, G0 to G3 numbers, verdict).
- No change to `checklist-score.md`: item 8 is rescored only by the orchestrator after an independent review.
