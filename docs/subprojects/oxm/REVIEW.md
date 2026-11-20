# jomission-oxm (Omission fixation model, OXM): review (2026-11-20)

Scope (README): the omission paradigm on the Fixation Sailor task. A, B and R are moving-grating waves with timing and context identical to the in-vivo omission paradigm. A is the environment wave, input to L4 of V1 and V4 as it changes distance from fixation. FEF eye neurons adjust the boat by plus or minus u and v through spiking. The decision comes from the firing of two groups, left-right and up-down.

Basis: `c4c9c6b`. Facts are OBSERVED from the file named unless marked INFERRED.

## What exists

| Piece | Where | State |
|---|---|---|
| Omission paradigm timing | `jomission/paradigm/spec.py`, `conditions.py` | 12 conditions (AAAB, AXAB, ... RRRX). Omission at p2, p3 or p4, slot keeps its 531 ms. Shared with FXS |
| Omission spec draft | `docs/laminar_motif/SPEC_OPEN38_OMISSION_DRAFT_2026-11-16.md` | Specification only. Defines the omitted event, baseline (fx window, -500 to 0 ms), and windows. No readout implemented, no experiment run |
| Omission firewall | `AGENTS.md`, `SUBSTRATE_PROGRAM.md` | No omission outcome may feed the substrate construction. Blind omission must come after the substrate is frozen |
| Fixation boat task | see `docs/subprojects/fixation-sailor/REVIEW.md` | Not implemented in jomission |
| OXM code | none | No module, test, or result |

## Findings

1. **OXM has no code.** Everything in OXM depends on FXS (the boat and its controller) and on the LANSA column (the model that receives input and drives FEF). Both are unfinished.
2. **The firewall and ordering conflict is open.** `SUBSTRATE_PROGRAM.md` places blind omission after the Gen-2 freeze. The omission proposal treats it as a separate lineage. The spec draft marks this as an owner decision. Until it is ruled, OXM cannot run.
3. **Input mapping is undefined.** The README says A is the environment wave input to L4 of V1 and V4, changing with distance from fixation. The distance-to-drive function, the gain, and the wave's spatial profile are not specified in any file.
4. **The readout is undefined.** Two FEF groups decide left-right and up-down. Their neuron counts, the threshold, and the decision window are not specified.
5. **The paradigm timing is reusable.** `paradigm/spec.py` already fixes the 531 and 500 ms slots and the omitted-slot rule. OXM should import it, not restate it.
6. **Estimands need declaring before a run.** Baseline, window, and the comparison (omission vs standard) must be fixed in a sealed spec, per the firewall.

## Owner decisions needed

- Ordering: blind OXM after the substrate freeze (the spec's default), or a separate lineage now.
- Input mapping: distance-to-drive function and the spatial wave profile.
- Readout: two groups, their size, threshold, and decision window.
- Null: which shuffle or surrogate tests the decision.
