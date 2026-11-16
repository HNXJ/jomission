# LANSA SST/PV provenance audit — 2026-11-16

Status: **read-only provenance audit, documentation checkpoint.** Changes no model, manifest, sealed result,
scientific verdict, gate or execution authority. Owner order 2026-11-16 (main-jomission-lansa, Option 1).

Audited tree: `origin/dev` @ `fcd8155`, `origin/docs/lansa-discovery-anatomy-20261116` @ `4ad28cb`, and all 11
remote branches (2026-11-16). Search: `scipy.optimize`, `optax`, `curve_fit`, `optuna`, `minimize(`, and the
commit history for rho / ratio / optimi / estimat / fit.

## Finding

No lineage in this repository estimates an SST/PV ratio. Every SST/PV value is prescribed: a fixed reference,
a sweep grid or a control arm. The only optimizers fit the readout covariance (observation model), and T1
selects kinetics and connectivity, not ratio. A parameter sweep is sensitivity analysis, not parameter inference.

## Lineage table (rho = N_SST / N_PV)

| Lineage | SST/PV | Class | Source | Evidence |
|---|---|---|---|---|
| E1, one module | log grid 0.25, 0.5, 1, 2, 4; seeds 1-3, 15 runs | PRESCRIBED (sweep) | `docs/laminar_motif/E1_SEAL.md:14-21` | `results/laminar_e1/e1_result.json` @ `8200fac`: f_pk falls with rho; verdict INCONCLUSIVE (rho 4 does not oscillate) |
| E2, 3-unit reference | S/M/D 0.25/1/2 | PRESCRIBED | `jomission/laminar/caminos_stack.py:30`; owner rulings C1/C4 2026-11-09, `PLAN_LANSA_E2.md:33,41` (chosen inside the E1 oscillating range) | `15351ed`; `paper/jomission_lansa/sections/abstract.tex:7` "imposed" |
| E2, 21-bin stack | log ramp 0.25 to 2 | PRESCRIBED | `caminos_stack.py:53` (rulings C1/C4) | `128d6c2` |
| Deep-rho sweep (`rhodX`) | rho_D 1, 2, 4, 8 | PRESCRIBED grid; exploratory, unsealed | `PLAN_LANSA_E2.md:172-180` @ `60cf497`; modifier `263dc7d` | r_proto 0.368 / 0.376 / -0.257 / 0.251, not monotone. Seeds, raw outputs, JaxFNE version: UNKNOWN |
| T8 ratio controls | rflat 0.794 all units; rrev S 2, M 1, D 0.25 | PRESCRIBED controls; sealed | `SEAL_T8.md:20-21`; modifiers `8aa14d5`, `6dc8a38` | `paper/jomission_lansa/data/t8_summary.json` @ `d49a8d0`, 3 seeds per arm: r_proto ref - control 0.154 (rflat), 0.182 (rrev), SD 0.10; r_gm control higher (rflat) / no consistent difference (rrev) |
| W6/T4 `rhosX` | per-unit scale, PV+SOM kept | PRESCRIBED perturbation | `caminos_stack.py:114`; `3c29ca0` | perturbation only |
| T1 selection | no rho axis (axes kd, sst, sstd) | ESTIMATED by grid selection, of kinetics and connectivity only | `SEAL_T1.md:12-17`; `manifests/t1_t7_*.json` | selected for similarity to Y and scored on Y: fitted agreement |
| Readout covariance C = L L^T | not SST/PV | ESTIMATED (Nelder-Mead) | `scripts/e2_readout_capacity.py`, `e2_readout_joint.py`; records C5-C7, D3-D5 | leave-one-area-out r 0.81 ± 0.05 (SD, n = 11 folds) vs 0.93 for the 10-area mean; `results.tex:442-448`: does not test the circuit |
| Later fitted SST/PV | none | UNKNOWN outside this repo; absent inside | full history, 11 branches | no optimizer, configuration or result |

## Data

- Y: `docs/laminar_motif/data/lansa_lfp_Y.json`, 21 bins x 150 Hz x 11 areas, imported `5992eb7` (sha256 `aad7f702`).
- X: `docs/laminar_motif/data/lansa_density_X.json`, PV/CB/CR/NG; sets unit sizes, never ratios. No SST or VIP.
- Training/validation split: none for the circuit; leave-one-area-out for the readout only.

## Validity

| Claim | Status |
|---|---|
| CAMINOS reproduces frequency-dependent oscillation | supported within E0/E1 scope |
| SST/PV ratio changes oscillation frequency | supported by E1; E1 verdict INCONCLUSIVE |
| LANSA estimates SST/PV ratios from spectra | UNSUPPORTED |
| E2 uses prescribed 0.25/1/2 | supported |
| The prescribed ramp explains the complete motif | not established (r_proto plateau about 0.38) |
| Independent anatomical validation | not performed |
| Held-out circuit prediction | not performed |

T8 shows that composition matters to model behaviour; it does not identify the biological SST/PV distribution.

## Missing evidence

- Raw outputs of the deep-rho sweep and the E2/T1 runs: only summaries, tables and figures are tracked; location UNKNOWN.
- `PLAN_LANSA_E2.md:30` macaque CB/PV about 3.0 superficial, 0.45-0.6 deep: no in-repo primary citation; UNKNOWN until sourced. CB is not SST (owner ruling 2026-11-06).

## Proposed open item (for the stack owner; not entered in `docs/OPEN.md`)

| ID | Status | Item |
|---|---|---|
| OPEN-(next) | proposed | SST/PV inference provenance: no repo lineage estimates SST/PV; all values are prescribed (this audit). Text that calls LANSA ratios inferred, fitted or estimated must be corrected or backed by optimizer code, configuration and result receipts. Closing check: a grep of `docs/`, `paper/`, `site-src/` for inferred/fitted/estimated SST/PV claims returns only corrected or receipted text. |
