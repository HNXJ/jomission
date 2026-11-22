# jomission-fixationboat (Fixation Sailor, FXS): review (2026-11-20)

Scope (README): the task that jomission-oxm runs on. Waves A (45 degrees), B (135 degrees), or random R (A ~ B, 50% Bernoulli). A boat is controlled by deep E neurons of FEF. Question: which E or I population, or which layer, is the best controller on the V1-V4-FEF minimal model (README: 3000 neurons, 1000 per area; layers may be expanded). HDP stays in the spectrolaminar motif model.

Basis: `c4c9c6b`. Facts are OBSERVED from the file named, unless marked INFERRED.

## What exists

| Piece | Where | State |
|---|---|---|
| Interactive lab | `.claude/skills/fixation-boat-jaxfne/references/lif_laminar_lab.html` | Single-file LIF lab: areas of six layers with E and I cells, feedforward, feedback, HDP gains, motor pools on L6-E of the top area, a 2-D boat map (target at origin, R = 5). Defaults: nA 3, nPerArea 180 (so 540 neurons, not 3000) |
| Skill | `.claude/skills/fixation-boat-jaxfne/SKILL.md` | Describes the lab and how to run it |
| Sailor engine API | `_temp_plasticity_boat/` handoff | The pinned `jaxfne==0.5.8` lacks `SailorConfig`, `sailor_step`, `sailor_decode`. A JaxFNE source labelled 0.5.9 provides them. Its V0 regression is exact against the tracked receipt, but the version and source SHA are UNVERIFIED (boat HANDOFF.md, "Not run or not examined") |
| Boat code in jomission | `jomission/` | No module for the boat task. No waves A/B/R, no boat dynamics, no decoder |
| Paradigm timing | `jomission/paradigm/spec.py`, `conditions.py` | Timing for the omission paradigm (p1-d4, 4624 ms per trial). Shared with OXM |

## Findings

1. **The boat does not fixate under the lab's default wiring.** INFERRED from the code, not measured. The force is `F = f0 * (cnt0 - cnt1, cnt2 - cnt3)` over four motor pools assigned by neuron index. No pool is tied to a direction. The only distance-dependent term is the encoder, `enc = gEnc * d / (d + d0)`, which grows with distance and drives the L4 input. Nothing reduces the force as the boat moves out. Damping only sets speed, so the boat settles into a drift. The earlier note calls this a known limit.
2. **Target and scale disagree with the README.** The lab defaults to 540 neurons with 180 per area. The README asks for 3000 neurons, 1000 per area. The scale is a parameter, so this is a mismatch in the default, not a model error.
3. **The controller is unsettled.** The README asks which E or I population, or which layer, is the controller. No experiment compares them. The lab exposes one control path (L6-E of the top area).
4. **The engine pin blocks the task.** The Sailor API is absent from the pinned `jaxfne==0.5.8`. The task needs 0.5.9 or a later engine. Either the pin changes (a decision for the owner, since the provenance test requires the exact pin) or the task is built without the Sailor API.
5. **Naming.** The README says "fixationboat"; the owner now says "Fixation Sailor". The dashboard uses Fixation Sailor. The README should use one name.
6. **Plasticity Boat is a different programme.** It has its own branch history, criteria v1 and v2, and an owner rule "no merge to dev or main before independent review". Keep it out of this plan.

## Owner decisions needed

- Engine: keep `==0.5.8` and avoid the Sailor API, or move the pin to 0.5.9 (the provenance test and `current_state.json` must change together).
- Controller comparison: which populations (E, I, layers) are candidates, and the success criterion for "fixates".
- Motor pool wiring: directional pools (+x, -x, +y, -y, option 1 in the earlier reply) or an explicit restoring term (option 2).
- Scale: 3000 neurons as in the README, or the 540-neuron lab for the first run.
