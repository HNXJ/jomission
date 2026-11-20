# OXM: Fixation Boat to Visual Omission Model. Proposed specification v0.1

Status: proposal from the owner, filed unchanged in section 1-6 below. Not implementation-verified. Section 7 is the review against the repo, added 2026-11-20. No implementation authorized: OXM waits on LANSA and Fixation Sailor minimal-complete and on the firewall ruling (`../PLAN.md`, D4).

## 1. Scientific objective

Build a biophysical neural circuit that learns to maintain fixation while observing predictable visual motion, then investigate its neural response when an expected motion event is unexpectedly absent.

The central hypothesis is: a circuit trained on temporally structured visual motion can develop stimulus-specific predictions, producing measurable neural activity when an expected event is omitted, even though the visual environment remains stationary.

This separates three processes that would otherwise be confounded:
- Sensory prediction: what visual motion should occur next?
- Motor control: how should the boat maintain fixation?
- Omission response: what happens neurally when predicted motion fails to occur?

The model should not be explicitly trained to classify omissions. Omission responses should emerge from the learned dynamics, if the hypothesis is correct.

## 2. Environment and task

Conceptual 2D visual environment. Arrows denote motion directions, not physical forces acting on the boat. The boat has a controlled position and a fixation target at the origin. Visual waves move through the environment but do not mechanically displace the boat. Any independent motor disturbance is generated separately and is matched across experimental conditions. The absence of a visual wave must not automatically remove a mechanical force.

| Symbol | Meaning | Motion |
|---|---|---|
| A | Visual wave A | 45 degrees |
| B | Visual wave B | 135 degrees |
| R | Reference stimulus | To define |
| X | Omission | No wave; valid stationary observation |

A and B have identical speed, contrast, amplitude, and duration; only direction differs. Each event occupies a nominal 531 ms slot, provisionally matching the intended paradigm. Whether this includes an interstimulus interval must be established from the experimental timing.

## 3. Sequence learning

Example trial sequences (illustrative four-slot sequences; onsets in ms; actual timing to be confirmed):

- Regular A: A 0, A 531, A 1062, B 1593
- Regular B: B 0, B 531, B 1062, A 1593
- Reference: R 0, R 531, R 1062, R 1593
- Omission: A 0, X 531, A 1062, B 1593

There is a subtle but consequential issue with AXAB. If the training sequence is AAAB, the second A is predictable, so AXAB is a valid omission probe. But it is not sufficient to determine whether the circuit predicted A specifically, or merely predicted that some event should occur. We therefore need matched omission and substitution probes.

## 4. Neural input and architecture

Two logically distinct information streams.

- Visual input: 2D motion vector, stimulus presence, and contrast.
- Biophysical recurrent circuit: excitatory and inhibitory populations, recurrent dynamics, adaptation and plasticity.
- Motor output: corrective control.
- Neural readout: spikes, LFP proxies, population dynamics.

For the first implementation, use a signed Cartesian motion vector rather than scalar distance. A scalar magnitude loses directional information and cannot distinguish A from B when their speeds are equal.

Boat position error and velocity enter a separate proprioceptive or control stream. Explicit prediction-error units are not required initially; prediction error can be assessed through neural activity and appropriate controls.

## 5. Experimental stages and falsification

| Stage | Experiment | Acceptance evidence |
|---|---|---|
| E0 | Fixation without visual waves | Stable control and quantified error |
| E1 | Fixation with A/B waves | Comparable motor performance |
| E2 | Learn AAAB / BBBA | Sequence-dependent neural prediction |
| E3 | Introduce AXAB / BXBA | Omission-locked neural response |
| E4 | Controls and generalization | Distinguish prediction from novelty, timing and motor effects |

Primary neural estimand: baseline-corrected omission-locked population response relative to a matched, equally timed stationary-control condition. Secondary measures: stimulus-specific decoding, neural trajectory divergence, fixation error, and motor output. Critical controls: matched non-omission events, unpredictable timing, sequence shuffling, and a stationary environment without learned expectation.

A response to X alone is not evidence of predictive coding. The response must depend on learned expectation and survive relevant sensory and motor confound controls.

## 6. Scientific decisions to freeze before implementation

- Reference R: a third visual motion direction, a repeated neutral stimulus, or something else?
- Timing: does 531 ms describe wave duration, onset-to-onset interval, or the entire stimulus slot?
- Training: fixation learning first, then sequence adaptation, or joint?
- Plasticity: which mechanisms are enabled, and what neural variables may adapt?
- Omission: does X replace one complete event slot while the clock and all other conditions stay unchanged?

Owner's recommendation: train fixation first, then expose the recurrent circuit to predictable sequences while maintaining fixation. Joint learning stays a later condition.

Principal success criterion: learned visual expectation produces a reproducible, stimulus- and timing-dependent neural response to omission, without a corresponding physical disturbance. Good fixation alone is not the criterion.

This is a proposed experimental specification, not a claim that the current Fixation Boat implements these capabilities.

## 7. Review against the repo (2026-11-20)

Sources: `jomission/paradigm/spec.py`, `conditions.py` (authoritative timing); `docs/subprojects/oxm/REVIEW.md`; `docs/subprojects/fixation-sailor/REVIEW.md`; `docs/laminar_motif/SPEC_OPEN38_OMISSION_DRAFT_2026-11-16.md`; `AGENTS.md` (omission firewall).

| Id | Severity | Spec says | Repo says | Action |
|---|---|---|---|---|
| S1 | blocker | Onset-to-onset 531 ms (examples; ISI unresolved) | Paradigm: p slot 531 ms, then a 500 ms d slot. Onsets 0, 1031, 2062, 3093 ms. Full trial 4624 ms | Use the paradigm timing. Replace the example timings. Resolves the ISI question |
| S2 | blocker | R = "to define" | README and paradigm: R is random A or B (50% Bernoulli). `RRRR` conditions exist | Decide: keep R as random A/B (canonical), or add a new reference stimulus as a new condition with a sealed spec. Do not leave it open at implementation |
| S3 | major | Visual input is a signed motion vector; A and B differ only in direction | README: A is a grating wave whose input to L4 of V1 and V4 changes with distance from fixation | Reconcile: the spec's motion vector and the README's distance-dependent L4 drive are different encodings. Pick one for the first implementation |
| S4 | major | Stages E0 to E4 | LANSA lineage already uses E0 to E6 (LM-0) | Rename OXM stages to O0 to O4 to avoid collision |
| S5 | major | Substitution probes (matched omission vs substitution) are required | Paradigm has no substitution condition | Add as new conditions with their own sealed spec |
| S6 | major | Matched stationary control | Spec draft (`SPEC_OPEN38_OMISSION_DRAFT`) defines baseline as fx window, -500 to 0 ms | Define the stationary control in the sealed spec; state how it differs from the fx baseline |
| S7 | major | Fixation learned first, then sequences | Firewall: training regime is substrate construction and must be frozen before any omission run | Owner ruling needed (D4). Freeze the regime before E3 |
| S8 | minor | Plasticity mechanisms unspecified | HDP sits in the LANSA motif model (README). Plasticity Boat is a separate programme | Decide which mechanisms apply. Keep Plasticity Boat out |
| S9 | minor | No neuron count | README: 3000 neurons, 1000 per area | Fix N in the sealed spec |
| S10 | minor | Visual waves do not displace the boat | Consistent with README (FEF spiking moves the boat). Keep | Add to the spec as a stated invariant, with the matched disturbance |

Recommendation: adopt the spec as v0.2 after S1, S2, S3, S4 and S5 are resolved in its text. S1 and S2 are factual fixes from the repo; S3, S7 are owner decisions. Do not implement until D4 (firewall ordering) is ruled and both LANSA and Fixation Sailor are minimal-complete (`../PLAN.md`).
