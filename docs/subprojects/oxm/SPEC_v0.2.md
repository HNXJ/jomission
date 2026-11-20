# OXM: Omission Fixation Model. Specification v0.2 (documentation only)

Status: specification. No implementation, no run, no sealed condition changed. OXM remains BLOCKED on firewall ruling D4 and on LANSA and Fixation Sailor minimal-complete (`docs/subprojects/PLAN.md`). Supersedes v0.1 (`SPEC_v0.1.md`, sections 1-6 as proposed by the owner; section 7 review there). Date: 2026-11-20.

Each statement is tagged: ESTABLISHED (in the repo, with source), APPROVED (owner ruling, this version), PROPOSED (hypothesis or control, not ruled), OPEN (unresolved, gates implementation).

## 1. Scientific objective

PROPOSED. A circuit trained on temporally structured visual motion can develop stimulus-specific predictions that produce a measurable neural response when an expected event is omitted, while the visual environment stays stationary. The response must depend on learned expectation, not on X alone. Sensory prediction, motor control and omission response are kept separate. Omission responses are not trained for; they must emerge.

## 2. Paradigm and timing

ESTABLISHED (`jomission/paradigm/spec.py`, `conditions.py`):
- Trial: fx -500 to 0 ms; then p1, d1, p2, d2, p3, d3, p4, d4. Each p slot is 531 ms; each d slot is 500 ms. Full trial 4624 ms, p1 to d4 4124 ms.
- Event onsets relative to p1: p1 0, p2 1031, p3 2062, p4 3093 ms. Onset interval 1031 ms.
- Omission X occupies its full 531 ms p slot with no wave. The clock and all other drive are unchanged.
- Conditions: A family AAAB, AXAB, AAXB, AAAX; B family BBBA, BXBA, BBXA, BBBX; R family RRRR, RXRR, RRXR, RRRX. Omission windows: p2 for AXAB, BXBA, RXRR; p3 for AAXB, BBXA, RRXR; p4 for AAAX, BBBX, RRRX.

APPROVED (S1): accept the 531 ms event and 500 ms delay, onset interval 1031 ms. The v0.1 examples at 531 ms onset intervals are withdrawn.

## 3. Stimulus identity

ESTABLISHED: `STIMULUS_R = "random_stimulus"` (`jomission/paradigm/conditions.py:13`). The R family is the random control.

APPROVED (S2): R is random A/B.
- Each R event independently samples A or B with probability 0.5.
- RRRR denotes four independent draws, not a third stimulus. No new reference stimulus is introduced.
- The realized sequence and the RNG seed are recorded for every trial, for reproducibility.

A: 45 degrees; B: 135 degrees; identical speed, contrast, amplitude and duration (v0.1 section 2, kept as PROPOSED design constraint until a stimulus file exists; OPEN, the visual parameters are not in the repo).

## 4. Sensory interface

ESTABLISHED: the README places A as the environment wave, input to L4 of V1 and V4, with drive that changes with distance from fixation. The Fixation Sailor lab (`.claude/skills/fixation-boat-jaxfne/references/lif_laminar_lab.html`) drives input by a single cue group G and a distance-dependent encoder `enc = gEnc * d / (d + d0)`. The lab has no A/B identity: the cue is one random boolean per trial.

APPROVED (S3): preserve the V1/V4 L4 distance-based encoding. Do not replace it with signed motion vectors. Signed motion vectors remain a possible future extension, not part of v0.2.
- A/B identity determines the stimulus pathway.
- Input amplitude follows the existing distance-from-fixation rule.
- Existing projection definitions and normalization are kept.

Representational limitation (must be recorded, not hidden): distance alone cannot encode direction. The lab's current input has no A/B pathway, so it does not preserve stimulus identity independently of amplitude. Under v0.2, identity must be carried by the pathway (which L4 population or which cue group receives the drive), not by amplitude. OPEN: the pathway design that does this (gate G1 below). Until it exists, any result that depends on A versus B identity is not evaluable, and this limitation is stated with it.

## 5. Experiments (renamed O0-O4)

APPROVED (S4): experiments are O0 to O4. LANSA's E0-E6 are not reused.

| Id | Experiment | Acceptance |
|---|---|---|
| O0 | Fixation without visual waves | Stable control; quantified error |
| O1 | Fixation with A/B waves | Comparable motor performance |
| O2 | Learn AAAB / BBBA | Sequence-dependent neural prediction |
| O3 | Omission AXAB / BXBA (and AAXB etc. from ESTABLISHED conditions) | Omission-locked neural response |
| O4 | Controls and generalization | Separates prediction from novelty, timing and motor effects |

PROPOSED primary estimand: baseline-corrected omission-locked population response, relative to a matched, equally timed stationary control.

## 6. Controls

- Matched non-omission event at the same time: available from ESTABLISHED conditions (e.g. AAAB at p4 vs AAAX).
- Unpredictable timing and sequence shuffling: PROPOSED.
- Stationary environment without learned expectation: PROPOSED.
- Substitution probes (omitted A replaced by B at the same slot): DEFERRED (S5). No new conditions in v0.2.

## 7. Baselines

APPROVED (S6): the stationary control is defined separately in its own sealed spec. It is not equated with the existing fx window (-500 to 0 ms, `SPEC_OPEN38_OMISSION_DRAFT`) without evidence. OPEN: the definition, and whether the fx window or a pre-omission window is used as baseline in addition.

## 8. Training order and plasticity

DEFERRED (S7): training order (fixation first, sequences second, or joint) is substrate construction. It is decided after firewall ruling D4 (`AGENTS.md`: no omission outcome may inform substrate construction). OPEN.
- PROPOSED recommendation (not ruled): fixation first, then sequences.
- HDP stays in the LANSA motif model (README). Its use here is OPEN (v0.1 S8).

## 9. Scale

OPEN (v0.1 S9). README gives 3000 neurons, 1000 per area, over V1, V4 and FEF. Not fixed in v0.2.

## 10. Reproducibility

APPROVED with R: for every trial, record the realized A/B sequence, the RNG seed, and the paradigm condition ID. A run whose realized sequence is not recorded is not evaluable.

## 11. Gates before implementation

- G0: firewall ruling D4 (ordering of blind omission versus the substrate freeze). OPEN.
- G1: pathway design that preserves A/B identity independently of amplitude, with a unit test (A and B drive different populations at equal distance). OPEN.
- G2: LANSA minimal-complete (`docs/subprojects/lansa/REVIEW.md`).
- G3: Fixation Sailor minimal-complete, with the controller chosen and its null test passing (`docs/subprojects/fixation-sailor/REVIEW.md`).
- G4: stationary control and training order sealed (S6, S7).
- G5: sealed O-spec for each experiment before its run.

## 12. Summary of rulings applied

| Item | Status in v0.2 |
|---|---|
| S1 timing | APPROVED: 531 + 500 ms, 1031 ms onset interval |
| S2 R | APPROVED: random A/B, independent draws, RRRR = four draws |
| S3 encoding | APPROVED: keep V1/V4 L4 distance encoding; identity limitation recorded (section 4) |
| S4 names | APPROVED: O0 to O4 |
| S5 substitution | DEFERRED: no new conditions |
| S6 stationary control | APPROVED as separate definition; OPEN content |
| S7 training order | DEFERRED until D4 |
| S8 plasticity/HDP | OPEN |
| S9 scale | OPEN |
| S10 no displacement by waves | ESTABLISHED invariant (v0.1), retained |
