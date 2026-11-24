# Layer-model inventory (Fixation Boat dashboard)

Compares the dashboard with the requirements in the v3 brief and with the layer-model reference in `.claude/skills/fixation-boat-jaxfne/SKILL.md` (the LIF laminar lab). Status: done = implemented and tested; gap = missing; conflict = the sources disagree and the owner must choose.

| # | Requirement | Source | Status | Notes |
|---|---|---|---|---|
| 1 | L1–L6 per area with E, PV, SST, VIP cells | brief | done | `layerFrac` |
| 2 | Each area is an identical clone | brief | done | `build()` |
| 3 | Within-layer connectivity (E→E, E→PV, E→SST, E→VIP, PV→E, SST→PV one way, VIP→SST) | brief | done | `routes`; VIP→SST is an assumption, not in the brief |
| 4 | Feedforward L5 E (area k) → L2 E (area k+1) | brief, reference | done | `routes` |
| 5 | Feedback L5 E (area k+1) → L6 E and L1 (area k) | brief, reference | done | the brief targets L1 VIP; the reference targets L1 (any type) |
| 6 | Haptic input into E cells of every area | brief | done | `alpha` |
| 7 | Motor readout: four pools in the top area | brief/fixture: L5 E; reference: L6 E | **conflict** | the dashboard reads out `motorLayer` (default 4 = L5, added so both can be run). The directional cue stays on L5 in both settings. Under the tested wiring L6 is unreachable from the cue (unsupported, not shown inferior; see README) |
| 8 | Input path into L4 (area 1 only) with canonical L4→L2/3→L5→L6 chain | reference | **conflict / gap** | the brief routes input to E of every area and has no L4 chain; a choice for the owner |
| 9 | Per-layer E:I ratio | reference | done | `layerFrac` gives it per layer |
| 10 | LIF, exponential synapses, refractory, noise | reference | done | `tauMembrane`, `tauSyn`, `refractory`, `noise` |
| 11 | Per-layer additive drive | (added) | done | `layerOffset`, zero by default |
| 12 | Adaptive threshold θ | reference | gap | a state-equation change; owner decision |
| 13 | HDP gains H and plastic edge gains m | reference | gap | a state-equation change; owner decision |
| 14 | Distance encoder: tonic g·e/(e+d0) on input cells | reference (`SailorTonicEncoder`) | **conflict** | the dashboard uses hap = α·H_MAX·tanh(d) into E cells; a different functional form |
| 15 | Boat integration: trapezoidal p' = p + dt(v+v')/2 after v' = v + dt(−βv+F) | reference (jaxfne boat step) | **gap (numerical)** | the dashboard updates v then x with the new v (semi-implicit); changing it moves the default numbers |
| 16 | Directional cue on motor cells | none | labelled addition | not in the frozen spec; needed for fixation in the default network |
| 17 | Known limit: the default wiring does not fixate | reference | open | the dashboard fixates only with the cue |

## Decision

The smallest component that advances the model without choosing between the sources is item 7: a `motorLayer` parameter (0–5) with default 4 (L5, the brief). The default stays as it is. L6 is not comparable under the current connectivity (see README). Items 8, 12–15 change state equations, connectivity or the encoder form; they are not implemented here and need the owner's choice.

## Open questions for the owner

1. Motor readout: L5 (brief/fixture) or L6 (reference)? Decided: L5 default; L6 unsupported under current connectivity.
2. Input: haptic into E of every area (brief), or into L4 of area 1 with the canonical chain (reference)?
3. Encoder: keep tanh(distance), or adopt the reference's g·e/(e+d0)?
4. Add threshold adaptation and HDP gains (state equations), or keep the reduced model?
5. Boat integration: keep semi-implicit, or match the trapezoidal step (changes default numbers)?
