# jomission-lansa: review (2026-11-20)

Scope: spectrolaminar cortex. Goal: the best-matching small cortex model per V1/V4/FEF becomes the model used in the omission paradigm. Basis: repo state at `c4c9c6b` (dev). Every number below is OBSERVED from a file named in the row unless marked.

## What exists

| Piece | Where | State |
|---|---|---|
| LANSA density data | `docs/laminar_motif/data/lansa_density_X.json` | 21 depth bins (-0.6 to 0.4 around mid-L4, negative superficial) x 4 types (PV, CB, CR, NG) x 11 areas (V1, V3, V4, MT, MST, LIP, TEO, PMD, 7A, 8A, PFC). Test pins shape and peaks: `tests/test_lansa_density.py` |
| Minimal model spec | `docs/laminar_motif/MINIMAL_MODEL_SPEC.md`, `MINIMAL_MODEL_STATUS.json` | LM-0: AdEx, three depth modules S/M/D, LFP proxy = summed synaptic current onto E. Structure E0-E6 approved; thresholds UNSET |
| E0 | `results/laminar_e0/e0_result.json`, `jomission/laminar/caminos_e0.py` | PASS. One CAMINOS module, N_E 8000, N_inh 1500, p 0.07, seeds 1-5, 12 s: f_pk 34.0 Hz, PV-to-SST lag 7.16 ms |
| E1 | `results/laminar_e1/e1_result.json`, `E1_SEAL.md` | INCONCLUSIVE. f_pk 46.0, 39.7, 34.0, 27.0 Hz at rho 0.25, 0.5, 1, 2; no oscillation at rho 4 |
| E1b | `results/laminar_e1b/e1b_result.json`, `E1B_SEAL.md` (commit 4c99e05) | Arm A (PV varies, total inhibition fixed) INCONCLUSIVE. Arm B (PV 750, SST varies) PASS. Receipts now registered as sealed at 2cb020f |
| Lite model (repo) | `jomission/network/builder.py` `build_jomission_lite_model`, `tests/test_lite_model.py` | V1 and V4 only, 200 neurons, 50 ms smoke test. Not the requested lite |
| Paper | `paper/jomission_lansa/` | Methods, results, supplement, LaTeX |

## Findings

1. **The lite model in the repo is not the lite you specified.** The repo builds V1+V4 at 200 neurons. The requested lite is V1-FEF, 3 layers, E-I, 100-neuron demo, then 1000 per area. Nothing in the repo matches it yet.
2. **The brief's SST and layer data are not measured.** LANSA did not measure SST. The owner ruling (2026-11-06) says CB is not SST, and macaque CB/PV is about 3.0 superficial and 0.45-0.6 deep, the opposite of the hypothesised motif. Any SST/PV profile is therefore a stated assumption until mouse numbers arrive (`PLAN_LANSA_E2.md` C1).
3. **The reachable frequency range is gamma to low beta.** At E0 drive, E1 gives 46 to 27 Hz. No alpha point exists in the isolated module. The deep module at rho 4 does not oscillate (INFERRED from E1; the code path that returns "not oscillating" is in `e1_verdict.py`, residual branch, and the seal does not state it: `PLAN_LANSA_E2.md` C7).
4. **E1b shows the frequency gradient is carried by SST at fixed PV, not by the PV/SST ratio alone.** Arm B PASS; Arm A INCONCLUSIVE. This is the only positive E-I result in the lineage and it is a sensitivity result, not a verdict on E1.
5. **Provenance gap, now closed for files only.** The E1b receipts were committed on a branch and are registered as sealed at that branch commit (rule in `sealed_artifacts.json`). The E1b lineage record itself does not exist. Creating one needs a visualization exemption or V0-V3, as `visualization_contract.json` requires for every new executed lineage.
6. **E1 and E1b ran on JaxFNE 0.5.9.** The project pin is `jaxfne==0.5.8` (`pyproject.toml`). A pin-or-run mismatch exists for E0, E1, E1b, and the current gate does not catch it.
7. **Durations and windows differ.** E0/E1 use 12 s with 2 s discarded. The plan's E2 is 20 s (`PLAN_LANSA_E2.md` C6).

## Open conflicts for the owner

These come from `PLAN_LANSA_E2.md` and must be ruled before E2 can be sealed:
- C1 SST/PV profile source (standing ruling: one-parameter ramp, low superficial, high deep).
- C2 two or three depth modules (LM-0 approved three).
- C3 unit counts per module (fixed CAMINOS 9500 per module, or LANSA-scaled).
- C4 deep rho (2 keeps the oscillating range of E1; 4 needs a new principal delta).
- C6 run length (20 s or 12 s).
- C7 E1 verdict path for the case "oscillating in 4 of 5 but rho 4 silent".

## Not reviewed here

The paper prose, the 16 gate-test receipts for E0/E1, and the 0.5.9 engine behaviour. Those need their own pass.
