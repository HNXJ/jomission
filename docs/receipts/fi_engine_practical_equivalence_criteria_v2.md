# f-I practical equivalence: criteria v2 (DRAFT 3, unsigned; no run before ratification)

Drafted 2026-11-15; draft 2 revised after Hamm's review (P3 and P4 held); draft 3 after an independent review (REVISE: P2 vacuous for three sets, P4 justification, undefined ISI, verdict gaps). Baseline: jomission `bba726a` on `_temp_swarm_fi_equiv_20261115`. Frozen engine: jaxfne 0.5.8 (unchanged).

## Relation to v1 and to checklist item 8

`fi_engine_equivalence_criteria.md` (v1, exact counts) and `fi_engine_equivalence_v1.json` stay as written: NOT_EQUIVALENT (G1: 12 of 486 points differ), checklist item 8 = 0. v2 is a separate scientific claim, not a replacement. A passing v2 is supplementary evidence; item 8 keeps its existing acceptance definition and changes only by Hamm's ruling.

Provenance of the thresholds. v2 was written after v1's outcome was known. v1 showed |count difference| <= 1 at every mismatch, so P1's tolerance of 1 is consistent with, not independent of, v1; Hamm may tighten it (the informational exact-count tally below shows how far from exact the held-out data is). The only data used to set structure beyond v1's outcomes is an O1-only probe (no E arm): O1 onset (first firing extra on a 0.01 lattice) lies between -3.1 and 16.5 for the six sets, response monotone on the lattice. It fixes the P2 floor below. No E-versus-O1 held-out comparison has been seen.

Disjoint is not independent. The held-out points are 0.125 from v1 points, and v1's mismatches cluster at extra 2 to 7. v2 therefore cannot claim the held-out data is a fresh sample of unrelated conditions, only that no point was inspected before.

## Two claims, reported separately

- Claim A, practical output equivalence: counts, rheobase, gain. Gates P1 to P3.
- Claim B, timing consistency: spike times, a property of the implementation beyond the f-I outputs. Gates P4b and P4e.
A verdict names its claim. Neither claim implies the other, nor v1's exact-count equality.

## Question and estimand

Do the NumPy f-I assay (`scripts/fi_single_neuron.py::firing_rates`, O1) and the JaxFNE 0.5.8 dense Izhikevich kernel (E) agree on inputs v1 never used? Arms, six parameter sets, dt 0.1 ms, 2000 ms (20000 steps), float64, explicit v0 = c, u0 = b c as in v1. Estimands per set: count N, spike steps, rheobase, gain.

## Grids

- Held-out grid H: extra = 0.125 + 0.25 k, k = 0..79 (odd multiples of 0.125, 80 points; disjoint from the v1 grid of multiples of 0.25).
- Rheobase lattice L: extra = 0.005 + 0.01 m, m = -500..1999 (-4.995 to 19.995). Disjoint from v1's G4 lattice (multiples of 0.01) and from H. Floor -5.0 is set from the O1-only onset scale above.

## Gates (thresholds PROPOSED; Hamm rules before any run)

Scales. The f-I curve feeds per-class tonic calibration (rheobase and slope; the BKG calibration inverts F-I toward ~10 Hz), so output tolerances are set against rate resolution: one spike in the 2000 ms window is 0.5 Hz. At N above 100 the 1% term exceeds one spike, which is intended (relative error).

- P1 count. At every point of H: |N_E - N_O1| <= max(1, 0.01 N_O1). N_O1 = 0 permits N_E <= 1 by design; N_O1 = 0 with N_E >= 2, or N_E = 0 with N_O1 >= 2, fails. Onset location is constrained by P2.
- P2 rheobase, evaluated on L. rheobase_arm = the smallest lattice extra with count > 0 in that arm (no assumption of monotonicity beyond this definition). Pass when both arms have a rheobase above the floor point (-4.995) and |difference| <= 0.05. If an arm fires at the floor point, or an arm never fires on L, or the two arms differ in either condition, P2 cannot be evaluated or fails as follows: both at the floor or both never firing, the claim is INVALID for that set (not a pass); exactly one arm in either condition, P2 fails. Rheobase is read from L, never from `summarize()` (which is bound to the v1 grid).
- P3 gain. G = least-squares slope of rate (Hz) against extra over the points of H with count > 0 (the `summarize()` definition applied to H, `numpy.polyfit` degree 1). Pass when |G_E - G_O1| <= max(0.01 |G_O1|, 0.005) Hz per unit extra. The 0.005 floor is 0.1 Hz over a nominal 20-unit span (one fifth of a spike); the fitted span varies by set (fewer firing points for VIP_engine), so G on H is not comparable with v1's G. If fewer than two firing points in either arm, G is undefined: pass only if undefined in both, recorded `undefined`.
- P4b rank-matched timing, points of H with N_E = N_O1 >= 1 (spike steps sorted ascending in both arms). tol(point) = 10 steps if N_O1 = 1; otherwise min(max(10 steps, 0.1 x ISI_med), 140 steps), where ISI_med is the median inter-spike interval of O1 at that point in steps. Pass when every rank pair differs by <= tol. The floor 10 steps (1 ms) and the cap 140 steps (14 ms) are stipulated; `LAG_WINDOW_MS = 14` (`jomission/laminar/caminos_analysis.py:32`) is the lag search half-range, not a timing resolution, and the link from this tolerance to the smoothing scale of the lag estimator is not established.
- P4e binned timing (scores unequal-count points that rank matching cannot): at every point of H with max(N_E, N_O1) >= 1, spike counts in 20 consecutive 100 ms bins differ by <= 1 per bin. The bin width is stipulated.
- P4c coverage = (points scored by P4b) / (points with N_O1 >= 1), per set. P4c < 0.90 for any set makes P4b UNDERPOWERED there; P4e still scores every firing point.
- Informational, no gate: first-spike latency difference; exact-count tally on H (points with N_E = N_O1); max |count difference|; unequal-count points listed with first-spike and count differences; float32 sensitivity.

## Controls (per set; a failed control makes the claim INVALID)

- Negative control, all six sets: O1 with d increased by 0.5 versus E, same code, must fail at least one of P1 to P3 for Claim A, and at least one of P4b or P4e (where scored) for Claim B. The receipt records the detecting gate per set. Risk stated in advance: VIP_engine has counts of order 10 and a +/-1 count tolerance, so its detection may rest on P2 or P3; if it detects nowhere the claim is INVALID, and that outcome is informative about the assay, not about the engines.
- Determinism: E run twice is bit-identical in counts and spike steps; O1 equals the scalar float64 oracle O2 in counts and spike steps at held-out points with zero-based index i mod 8 = 0 (i = 0, 8, ..., 72; 10 points) in all six sets.
- Positive control: O1 compared with itself passes P1 to P4 for all six sets.

## Verdicts (checked in this order)

1. INVALID (for a claim): any control fails, or P2 is INVALID for a set (Claim A), or E cannot run through a supported path. All sets and gates are still reported.
2. Claim A: PRACTICALLY_EQUIVALENT if P1 to P3 hold for all six sets; otherwise NOT_PRACTICALLY_EQUIVALENT (failing set, point, sizes).
3. Claim B: TIMING_INCONSISTENT if any scored P4b or P4e fails; else INCONCLUSIVE if P4c < 0.90 for any set (P4b underpowered); else TIMING_CONSISTENT.
- v1's NOT_EQUIVALENT stands whatever v2 returns. A change to any threshold, grid, or definition after a v2 run is a new declaration (v3).

## Dependencies and owner decisions

- Thresholds, bins and grids are scientific tolerances: proposals until Hamm ratifies. No v2 run is authorized before that.
- The run needs no tier-3 battery and writes no `results/*.json`; output would be `docs/receipts/fi_engine_practical_equivalence_v2.json` (unsealed) from a new script reusing the v1 harness.
- Untested: the cause of the v1 mismatches (inferred candidate: XLA versus CPython rounding at threshold crossings).
