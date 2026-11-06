# Minimal model LM-0 (draft, 2026-11-06)

Status: structure APPROVED by the owner 2026-11-06 ("Approve as drafted"), with the three decisions below recorded. Thresholds and E0 pass criteria are UNSET, so no run is authorized yet (AGENTS.md: seal before running). Every threshold is UNSET and follows the approved structure in `ITEM03_SEAL_PROPOSAL.md`. LM-0 is the minimal-scale realization of checklist items 6, 7, 11, 15, 17 and 21; it adds no checklist item.

## Hypothesis in measurable form (owner, 2026-11-06)

Three ingredients, all at the level of the column:

| Ingredient | Definition (UNSET entries need the owner) |
|---|---|
| E/I ratio `eps(d)` | `N_E / (N_PV + N_SST + N_VIP)` per depth module `d` (the manuscript's proportion term `z(E) / sum z(inhibitory)` is the observed analogue) |
| SST/PV ratio `rho(d)` | `N_SST / N_PV` per module |
| One-direction wiring | SST to PV strong, PV to SST zero, SST to SST zero (CAMINOS asymmetry, Tahvili, Vinck and di Volo) |

Claim [H]: with the wiring fixed, a profile `rho(d)` that is low superficially and high deep makes gamma relative power peak superficially and alpha/beta relative power peak deep. Source basis for the mechanism: higher SST/PV lowers the oscillation frequency (Tahvili, Vinck and di Volo; checked in the item 2 digest). The depth profile and the motif are the new claim, so each is a test with a falsifier, never a premise.

## Structure

Depth modules `S` (superficial), `M` (middle, L4 crossover), `D` (deep); three is the minimum that has a crossover. Each module is a CAMINOS-type E-PV-SST network (VIP added only after the base works). Modules couple by feedforward and feedback E-to-E and E-to-inhibitory edges; the rule set is declared in `x : A : y` form (`ITEM08_H_STATE_SPEC.md`) with `CTX[lam]` per module.

Per-module knobs: `eps`, `rho`, drive. Everything else is shared and frozen.

## Experiments (each seals its negative control)

| Id | Manipulation | Reads | Falsifier |
|---|---|---|---|
| E0 | one isolated module with the published CAMINOS parameters, run on the JaxFNE AdEx emitter | peak frequency, PV-to-SST lag, rate order | the item 11 source values not reproduced (JaxFNE's own port is the reference) |
| E1 | one isolated module, sweep `rho` at fixed `eps` and drive | peak frequency versus `rho`; band class | frequency does not fall as `rho` rises |
| E1c | E1 with PV to SST added (symmetric wiring) | same | asymmetry makes no difference (then the wiring is not an ingredient) |
| E2 | three-module stack, `rho(S) < rho(M) < rho(D)` | relative power by depth and band | gamma peak not superficial or alpha/beta peak not deep |
| E3 | E2 with the profile reversed | same | the motif does not reverse |
| E4 | E2 with a flat profile | same | a motif remains (then the profile is not the cause) |
| E5 | E2 with top-down or bottom-up baseline drive removed | same | the motif survives drive removal (owner: ablations remove it) |
| E6 | E2 with `eps(d)` varied at fixed `rho` | same | E/I ratio alone matches the effect of `rho` (then the SST/PV ratio is not specific) |

Readout (design choice, UNSET): LFP proxy per module from summed synaptic currents onto E cells; relative power `RP(d, f) = P(d, f) / max_d P(d, f)` with the manuscript's bands. Motif score definitions come from the sealed item 3 table; no number is chosen here.

## What is reused

| Piece | Source | State |
|---|---|---|
| CAMINOS reference network and AdEx conductance kernel | JaxFNE `simulate_adex` and its port in the JaxFNE repo (`artifacts/etudes/papers/caminos`) | result file says PASS (f_pk 33.80 Hz, PV-to-SST lag 7.22 ms); the replication register says mixed (lag 6.47 ms, seeds 1 to 3). The two JaxFNE records disagree; to be shown to the owner, not resolved here |
| HDP adaptation rule | `jomission/dynamics/adex_hdp.py` | tested (item 9 a to e) |
| Izhikevich-TFNE variant | item 6 parameter sets | not built |

## Decisions (owner, 2026-11-06)

1. Neuron class: AdEx first (JaxFNE AdEx emitter, the validated CAMINOS reference); the Izhikevich-TFNE column follows as the item 6 and 11 comparison.
2. Three depth modules; LFP proxy per module is the summed synaptic current onto E cells.
3. Structure of E0 to E6 approved as drafted.

## Still open (OPEN-40)

1. E0 pass criteria (UNSET). Proposal, to be approved before E0 runs: the JaxFNE reference values and margins for the same observables, stated in the sealed E0 spec, not here.
2. Source for the Disney SST-CB and CR-VIP overlap (not in the komposer repository).
3. The two JaxFNE records that disagree on the CAMINOS port (PASS versus mixed).
