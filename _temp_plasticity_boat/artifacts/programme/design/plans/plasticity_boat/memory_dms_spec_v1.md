# Plasticity Boat: memory-guided DMS specification v1 (proposed)

Status: proposed specification, received as pasted text on 2026-11-21. Not implemented, not validated, not owner-ratified. The existing 10-neuron LIF controller, motor decoder, plasticity rules and boat dynamics are the baseline and stay unchanged.

## Content (as received)

**Decision:** brief spatial target cue, then a target-free delay, then a timed movement window. The controller must keep target information internally and use it to navigate after the cue disappears.

**Phases**

| Phase | Target observable | Motor action | Termination |
|---|---|---|---|
| CUE | yes | disabled | cue expires |
| DELAY | no | disabled | delay expires |
| GO | no | enabled | arrival or deadline |
| END | no | disabled | record result |

Event i: e_i = (t_i, q_i, c_i, delta_i, g_i, r_i): target q_i, cue duration c_i, delay delta_i, GO deadline g_i, success radius r_i. Boundaries t1 = t_i + c_i, t2 = t1 + delta_i, t3 = t2 + g_i. Motor output is gated in CUE and DELAY; the neural state keeps evolving. Motor gating must not reset neural state.

**Observation:** o_t = (m_t, s_t, b_t). m_t phase and GO flags; s_t = q_i - p(t_i), sampled at cue onset and held through CUE, zeroed in DELAY and GO with cue-valid = false; b_t permitted proprioception defined per condition.
Invariant: no target coordinates, target distance, bearing or target-relative reward gradient after cue offset.

**Config (YAML, abridged):** cue 0.5 s, delay 2.0 s, GO 5.0 s, success radius 0.5. Policy: movement before GO disabled; target invisible after cue; early success terminates; overlapping trials forbidden; neural reset configurable; plasticity during delay configurable. Example events at onset 2.0, 12.0, 24.0 s (illustrative, not calibrated). The scheduler rejects invalid configurations rather than rescheduling.

**Success:** S_i = 1 if the boat is within r_i of q_i at some time in [t_GO, t_END]. Primary endpoint: success rate on a frozen, feasible, held-out target set. Secondary: arrival latency, terminal error, path efficiency, control effort, delay robustness, memory advantage (intact minus shuffled cue). Report n, effect sizes, intervals, seeds, frozen config.

**Acceptance tests (all unchecked):** T01 exact transitions at step boundaries; T02 target only in CUE; T03 no leakage (hidden target changed, physical state fixed, post-cue observations identical); T04 no motor movement before GO; T05 neural state continuous across CUE, DELAY, GO; T06 deadline arrival rule; T07 oracle succeeds on feasible trials, impossible trials flagged; T08 determinism; T09 legacy fixed-target mode numerically unchanged; T10 no trial-to-trial leakage; T11 shuffled or absent cues reduce a memory-dependent positive control; T12 known-memory oracle correct across delays.

**Conditions:** C0 visible target (validate navigation); C1 brief cue, no delay (validate recall); C2 brief cue, variable delay (memory). Controls: ideal memory oracle, reactive no-memory controller, shuffled cues, no cue. Freeze targets, starts, deadline and seeds across comparisons.

**Implementation order:** (1) backend: schema, scheduler, phase machine, masking, logs; (2) T01 to T10 against an oracle and the baseline; (3) connect the LIF controller without changing neuron count or plasticity; (4) C0, C1, C2 with controls; (5) 100-neuron circuit. Keep independent of O-8 (directional gain) and O-9 (decoder normalisation) until the backend is verified.

**Qualification:** technically memory-guided delayed spatial reaching, not classical DMS (no comparison stimulus). Memory capability of the existing controller is UNKNOWN; no performance claim until C0 and C1 are established. A C2 success shows memory-guided behaviour, not the synaptic mechanism.

## Review notes (added by the agent; not part of the pasted text)

| # | Gap | Where | Needed before implementation |
|---|---|---|---|
| 1 | END boundary t_END is used in the success criterion but never defined | §1, §4 | define END duration or set t_END = arrival or deadline |
| 2 | Feasibility (T07, primary endpoint) has no rule | §4, T07 | define reachable-within-g from boat speed and dynamics, with a proof or simulation |
| 3 | "Held-out target set" undefined; owner rule keeps held-out seed 2026 unused until the lock | §4 | reconcile with the owner's held-out rule |
| 4 | Success radius 0.5 and cue timings are not tied to boat speed | §3 | calibrate against the boat's speed and dynamics |
| 5 | Proprioception in b_t plus the held s_t lets the controller reconstruct the target position | §2 | accepted as the memory itself; T03 must hold physical state fixed |
| 6 | No controller in the repo is shown to have memory; T11 needs one | §5, §7 | a memory-capable positive control must exist before T11 is meaningful |

## Next step

Implementation step 1 (backend and T01 to T10) is not started. Owner decision needed: approve implementation of step 1 on this branch, and the answer to gaps 1 to 3.
