# f-I practical equivalence: criteria v2 (DRAFT 2, unsigned; no run before ratification)

Drafted 2026-11-15, revised the same day after Hamm's review (P3 and P4 held). Baseline: jomission `9e4a8de` on `_temp_swarm_fi_equiv_20261115`. Frozen engine: jaxfne 0.5.8 (unchanged).

## Relation to v1 and to checklist item 8

`fi_engine_equivalence_criteria.md` (v1, exact counts) and `fi_engine_equivalence_v1.json` stay as written: NOT_EQUIVALENT (G1: 12 of 486 points differ), checklist item 8 = 0. v2 is a separate scientific claim, not a replacement. It was written after v1's outcome was known, so the v1 grid is not confirmatory data for it; v2 is scored only on the held-out grid. A passing v2 is recorded as supplementary evidence. Item 8 keeps its existing acceptance definition; its score changes only by Hamm's ruling.

## Two claims, reported separately

- Claim A, practical output equivalence: the f-I curve a downstream user reads (counts, rheobase, gain) agrees within tolerances set by what those outputs are used for. Gates P1 to P3.
- Claim B, timing consistency: spike times agree, which probes implementation equivalence beyond the f-I outputs. Gate P4.
A verdict names which claim it supports. Claim A can hold while B fails or is underpowered; neither implies the other, and neither implies v1's exact-count equality.

## Question and estimand

Do the NumPy f-I assay (`scripts/fi_single_neuron.py::firing_rates`, O1) and the JaxFNE 0.5.8 dense Izhikevich kernel (E) agree on inputs v1 never used? Arms, six parameter sets, dt 0.1 ms, 2000 ms, float64, explicit v0 = c and u0 = b c as in v1. Estimands per set: count N, spike steps, rheobase, gain.

## Held-out grid

extra = 0.125 + 0.25 k, k = 0..79 (80 points, disjoint from the v1 grid). Refined rheobase grid: step 0.01 over O1 rheobase_extra +/- 0.25 (both sides; v1 G4 covered the lower side only).

## Gates (thresholds PROPOSED; Hamm rules before any run)

Justification scales. The f-I curve feeds per-class tonic calibration (rheobase and slope from `summarize()`; the BKG calibration inverts F-I toward ~10 Hz), so output tolerances are set against rate resolution: one spike in the 2000 ms window is 0.5 Hz. Timing is set against the neuron's own inter-spike interval and the estimator lag window (`LAG_WINDOW_MS = 14 ms`, `jomission/laminar/caminos_analysis.py`).

- P1 count. |N_E - N_O1| <= max(1, 0.01 N_O1) at every held-out point. Zero counts: N_O1 = 0 permits N_E <= 1 by design; the location of onset is constrained by P2. N_O1 = 0 with N_E >= 2, or the reverse, fails.
- P2 rheobase. |rheobase_E - rheobase_O1| <= 0.05 on the refined grid, per set. Rheobase is the smallest extra with count > 0; if either arm never fires on the refined grid the set fails P2 unless both do not fire.
- P3 gain. Gain G = positive-grid slope of rate against extra (`summarize()`). Pass when |G_E - G_O1| <= max(0.01 |G_O1|, 0.005) Hz per unit extra. The absolute floor 0.005 Hz/unit is 0.1 Hz over the 20-unit span, one fifth of a spike, so a near-zero slope cannot make a relative test vacuous or impossible. If G is undefined (fewer than two firing points) in either arm, P3 passes only if undefined in both and reports `undefined`.
- P4 timing (Claim B). Report four numbers per set and gate on the first two:
  - P4a first-spike latency: every point with N >= 1 in both arms differs by <= max(10 steps, 0.1 median ISI of O1) (10 steps = 1 ms).
  - P4b rank-matched spikes: at points with N_E = N_O1, every spike pair differs by <= max(10 steps, 0.1 median ISI of O1).
  - P4c coverage: fraction of firing points (N_O1 >= 1) scored by P4b. If P4c < 0.90 for any set, P4b is UNDERPOWERED for that set and Claim B is INCONCLUSIVE, not pass.
  - P4d drift (no gate): for equal-count points, max |dt| against spike rank; for unequal-count points, the count difference and first-spike latency difference. Unequal-count points are reported here because rank matching cannot score them and would hide accumulated drift.
  The 0.1 ISI term keeps the tolerance below the estimator lag window at every rate; 1 ms is a floor, not a scientific claim of sufficiency.

## Controls (each set, or the set's verdict is INVALID)

- Negative control, all six sets: O1 with d increased by 0.5, compared with E by the same code, must fail at least one of P1 to P3 for that set (and P4a/P4b where scored). The receipt records which gate detected, per set. A set where the perturbation is undetected makes that set INVALID, so the verdict is INVALID.
- Determinism: E run twice is bit-identical; O1 equals the scalar float64 oracle O2 exactly at every 8th held-out point of all six sets.
- Positive control: O1 compared with itself passes P1 to P4 for all six sets.

## Verdicts

- Claim A, PRACTICALLY_EQUIVALENT: P1 to P3 hold for all six sets and all controls hold. Otherwise NOT_PRACTICALLY_EQUIVALENT (report failing set, point, sizes) or INVALID.
- Claim B, TIMING_CONSISTENT: P4a and P4b hold for all six sets and P4c >= 0.90 for all sets. TIMING_INCONSISTENT if a scored P4 fails; INCONCLUSIVE if coverage is short.
- v1's NOT_EQUIVALENT stands whatever v2 returns. A change of any threshold or the grid after a v2 run is a new declaration (v3).

## Dependencies and owner decisions

- Thresholds and grid are scientific tolerances: proposals until Hamm ratifies. No v2 run is authorized before that.
- The run needs no tier-3 battery, writes no `results/*.json`, and its output would be `docs/receipts/fi_engine_practical_equivalence_v2.json` (unsealed), from a new script reusing the v1 harness.
- Untested: the cause of the v1 mismatches (inferred candidate: XLA versus CPython rounding at threshold crossings). v2 does not test the cause.
