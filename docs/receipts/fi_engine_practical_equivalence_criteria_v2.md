# f-I practical equivalence: criteria v2 (DRAFT 6, unsigned; candidate for final independent review, no run before ratification)

Drafted 2026-11-15; draft 2 after Hamm's first review; draft 3 after an independent REVISE (P2 vacuous for three sets); draft 4 after Hamm's second review (undefined gain; timing scope); draft 5 after a fresh independent REVISE (control definitions, Claim B vacuous-pass paths, verdict masking, grid wording, conventions, and the stated blind spots of P4). Parent: jomission `2003e6e` (draft 5) on `_temp_swarm_fi_equiv_20261115`; this file is draft 6. Frozen engine: jaxfne 0.5.8 (unchanged).

## Relation to v1 and to checklist item 8

`fi_engine_equivalence_criteria.md` (v1, exact counts) and `fi_engine_equivalence_v1.json` stay as written: NOT_EQUIVALENT (G1: 12 of 486 points differ), checklist item 8 = 0. v2 is a separate scientific claim, not a replacement. A passing v2 is supplementary evidence; item 8 keeps its existing acceptance definition and changes only by Hamm's ruling.

Provenance of the thresholds. v2 was written after v1's outcome was known. v1 showed |count difference| <= 1 at every mismatch, so P1's tolerance of 1 is consistent with, not independent of, v1. The only data used beyond v1's outcomes is an O1-only probe (no E arm): O1 first-firing onset on the lattice L, per set: E -1.545, PV 0.735, SST_engine -3.095, SST_jomission -0.545, VIP_engine 16.505, VIP_jomission 0.005. It fixed the floor of L at -5.0. The v2 script must print these onsets in the receipt (the probe itself is not a committed file). No E-versus-O1 held-out comparison has been seen.

Disjoint is not independent. Held-out points lie 0.125 from v1 points, and v1's mismatches cluster at extra 2 to 7; v2 claims only that no held-out point was inspected before.

## Two claims, reported separately

- Claim A, practical output equivalence: counts, rheobase, gain. Gates P1 to P3.
- Claim B, timing consistency: spike times. Gates P4b and P4e. It is weak by construction (see Blind spots).
A verdict names its claim. Neither claim implies the other, nor v1's exact-count equality.

## Question, arms, conventions

Do the NumPy f-I assay (`scripts/fi_single_neuron.py::firing_rates`, O1) and the JaxFNE 0.5.8 dense Izhikevich kernel (E) agree on inputs v1 never used? Six parameter sets, dt 0.1 ms, 20000 steps, float64, explicit v0 = c, u0 = b c as in v1.
- Spike step = the update index i in 0..19999 at which the spike is emitted, in both arms; the v2 script records each arm's native convention and the conversion. Informational: the median signed (E minus O1) rank offset per set at equal-count points, so a constant one-step convention offset (invisible below the 10-step floor) is visible.
- O1 and the scalar oracle O2 must return spike steps, not only counts (v1's O2 returned counts).
- Bins for P4e are half-open [1000 k, 1000 (k+1)) in step index, k = 0..19.
- Grid and lattice values are held and compared as integer indices (units of 0.005); P2 compares lattice indices, never floats (a float test `<= 0.05` is a coin flip at exactly 5 lattice steps). All other comparisons use float arithmetic and `<=`.
- The spike-recording O1 is a modified copy of `firing_rates` and must reproduce its counts exactly on every point of H; O2 must use O1's expression order; the E-to-O1 spike-step conversion constant is fixed and recorded before any run.

## Grids (built from integer indices in units of 0.005; disjointness asserted in the script)

- Held-out grid H: extra = 0.125 + 0.25 k, k = 0..79 (25 + 50 k in units of 0.005).
- Rheobase lattice L: extra = 0.005 + 0.01 m, m = -500..1999 (-4.995 to 19.995; 1 + 2 m units). L contains H. H and L are both disjoint from v1's EXTRA (multiples of 0.25) and G4 lattice (multiples of 0.01); checked by integer arithmetic. Floor -5.0 from the O1-only onsets above.

## Gates (thresholds PROPOSED; Hamm rules before any run)

Scales. The f-I curve feeds per-class tonic calibration (rheobase and slope; BKG calibration inverts F-I toward ~10 Hz), so output tolerances are set against rate resolution: one spike in 2000 ms is 0.5 Hz.

- P1 count. At every point of H: |N_E - N_O1| <= max(1, 0.01 N_O1). N_O1 = 0 permits N_E <= 1; N_O1 = 0 with N_E >= 2, or N_E = 0 with N_O1 >= 2, fails. Onset location is constrained by P2. Observed in O1 only: VIP_engine has N_O1 <= 1 at 78 of 80 points of H, so P1 barely binds there; P2, P3 and P4e carry that set.
- P2 rheobase, on L. An arm HAS ONSET if it fires somewhere on L and not at the floor point (-4.995); its rheobase is the smallest lattice extra with count > 0 (first-firing onset; count need not be monotone, so an isolated transient spike moves it). Both arms have onset: pass when |m_E - m_O1| <= 5 (integer lattice indices; 0.05 = 5 steps). Exactly one arm has onset: P2 fails. Neither has onset (at the floor, never firing, or one of each): the set is INVALID for Claim A. Rheobase is read from L, never from `summarize()`.
- P3 gain. G = least-squares slope (`numpy.polyfit`, degree 1) of rate (Hz) against extra over the points of H with count > 0. Pass when |G_E - G_O1| <= max(0.01 |G_O1|, 0.005) Hz per unit extra. Both arms with fewer than two firing points: NOT EVALUABLE (never a pass; Claim A is INCONCLUSIVE). Exactly one arm: P3 fails. The floor 0.005 Hz/unit is stipulated; it is 0.1 Hz over a 20-unit span but the fitted span is the firing range (about 3.25 units on H for VIP_engine, 14 firing points). Plainly: for VIP_engine and other short-span sets, a one-spike (0.5 Hz) error at the end of the span shifts G by about 0.06 Hz/unit against a tolerance of about 0.0065, far above it, so P3 there is close to an exact-count test and is stricter than P1. Hamm ratifies knowingly (owner ruling 2).
- P4b rank-matched timing, on points of H with N_E = N_O1 >= 1 (spike steps ascending in both arms). tol(point) = 10 steps if N_O1 = 1; otherwise min(max(10, 0.1 x ISI_med), 140) steps, ISI_med = `numpy.median` of the O1 inter-spike intervals at that point (a float; compare with `<=`). Pass when every rank pair differs by <= tol. The floor 10 steps (1 ms) and cap 140 (14 ms) are stipulated; `LAG_WINDOW_MS = 14` (`jomission/laminar/caminos_analysis.py:32`) is the lag search half-range, not a timing resolution.
- P4e binned timing, on every point of H with N_O1 >= 1 or N_E >= 1: counts in the 20 bins differ by <= 1 per bin.
- P4c coverage = (points scored by P4b) / (points with N_O1 >= 1), per set; denominator 0 gives UNDERPOWERED. P4c < 0.90 makes P4b UNDERPOWERED for that set.
- Informational, no gate: first-spike latency difference; exact-count tally on H; max |count difference|; unequal-count points listed with count and first-spike differences; signed rank-offset median; float32 sensitivity; n_scored per gate per set.

## Blind spots of Claim B (observed on synthetic O1-only trains in review; stated, not fixed)

- A single dropped or extra spike passes P1 (+/-1), is not scored by P4b (unequal count) and gives a bin difference of 1 at most in P4e: no gate can see it.
- P4e does not flag uniform shifts or drift up to 200 steps (20 ms) at N <= 39; it flags shifts only when more than one spike crosses a bin edge (N around 300).
- P4b passes any shift at or below tol (140 steps at N about 5, about 50 at N about 39, 10 at N about 300), including a 14 ms offset at low N.
- P4e scores unequal-count points but passes any count difference of at most 1 per bin (up to about 20 spikes over 20 bins), so it detects count differences only above that and gives no timing control; in an O1 versus O1 (d + 0.5) probe, P4e passed at points whose counts differed by 5 or more (6 to 21 points per set). For SST_engine and SST_jomission that probe had 0 P4b-scored points, so any Claim B detection there rests on P4e counts.
- P4b is stricter as N grows (max over ranks): at N about 300, iid jitter with sd 5 steps (0.5 ms) flags every seed, while at N 20 to 60 sd 20 flags tens of percent (14.5% of 200 trials in one probe). A benign sub-millisecond jitter can fail Claim B at high N.
So TIMING_CONSISTENT means only: rank-matched timing within tol on equal-count points and 100 ms-binned counts within 1, nothing about within-bin timing or about single missing spikes.

## Controls (per set; n_scored > 0 for every gate a control relies on is required and recorded; a control that scores nothing fails)

- Negative control (Claim A), all six sets: O1 with d increased by 0.5 versus E, same code, must fail at least one of P1 to P3. Risk stated in advance: VIP_engine counts are of order 10, so detection there may rest on P2 or P3.
- Negative control (Claim B), all six sets: the same comparison must fail P4b or P4e at some scored point; a set where it fails nowhere, or scores nothing, fails the control (Claim B INVALID).
- Timing tolerance-edge control (tests the P4b implementation, not the engines), on O1 against shifted O1, shift sign positive (+s steps): at points with N_O1 >= 2, a shift of floor(tol) steps passes P4b and floor(tol)+1 fails it. A point is excluded (and counted) if any spike of the floor(tol)+1 shift would exceed step 19999. At least one eligible point per set is required (recorded; the eligible count is small for VIP_engine, 2 in review probing), else the control fails and Claim B is INVALID.
- Determinism: E run twice is bit-identical in counts and spike steps; O1 equals O2 in counts and spike steps at points of H with zero-based index i mod 8 = 0 (i = 0, 8, ..., 72; 10 points) in all six sets.
- Positive control: O1 compared with itself passes P1 to P4 for all six sets.

| control | invalidates |
|---|---|
| negative (A) | Claim A |
| negative (B), tolerance-edge | Claim B |
| determinism, O2 equality, positive | both |

## Verdicts (checked in this order, per claim)

1. INVALID for a claim: a control in the table above for that claim fails, or (Claim A) P2 is INVALID on any set, or E cannot run through a supported path. All sets and gates are still reported. INVALID masks a definitive failure on another set: the failing gates are reported, but the verdict is INVALID, because an invalid assay cannot support any verdict.
2. Claim A: NOT_PRACTICALLY_EQUIVALENT if any of P1 to P3 fails on any set; else INCONCLUSIVE if P3 is not evaluable on any set; else PRACTICALLY_EQUIVALENT.
3. Claim B: TIMING_INCONSISTENT if any scored P4b or P4e fails; else INCONCLUSIVE if, on any set, P4c is below 0.90 or undefined, or n_scored(P4b) = 0, or n_scored(P4e on points with N_O1 >= 1) = 0; else TIMING_CONSISTENT (with the scope above).
- v1's NOT_EQUIVALENT stands whatever v2 returns. A change to any threshold, grid, or definition after a v2 run is a new declaration (v3).

## Owner rulings needed before any run

1. P1 tolerance of 1 (consistent with v1's observed maximum) and whether VIP_engine needs its own bound.
2. P3 floor and the span-dependence above (P3 close to exact counts for short-span sets).
3. Whether Claim B, with its blind spots, is worth running as specified, or needs a spike-train distance measure instead of P4b and P4e.
4. Thresholds P2 0.05, P3 1%, P4b floor and cap, P4e bin width: stipulated scales, none validated against a downstream consumer.

## Dependencies

- No v2 run is authorized before ratification. It needs no tier-3 battery, writes no `results/*.json`, and its output would be `docs/receipts/fi_engine_practical_equivalence_v2.json` (unsealed) from a new script reusing the v1 harness.
- Untested: the cause of v1's mismatches (inferred candidate: XLA versus CPython rounding at threshold crossings).
