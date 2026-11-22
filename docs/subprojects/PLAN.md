# Plan: LANSA, then Fixation Sailor, then OXM (2026-11-20)

Per-subproject reviews: `lansa/REVIEW.md`, `fixation-sailor/REVIEW.md`, `oxm/REVIEW.md`. Order: a minimal complete LANSA and a minimal complete Fixation Sailor, in parallel; OXM after both.

## Step 0: decisions to clear before any build (owner)

| ID | Decision | Blocks |
|---|---|---|
| D0 | `plasticity boat` was merged into `dev` and `main` with `f1b9337` (`Merge origin/_temp_plasticity_boat`; its second parent holds `280ebd0`). Its handoff says: no merge before independent review. Revert on `dev`, or accept and record a ruling | Every step (history) |
| D1 | Engine pin: keep `jaxfne==0.5.8` and build without the Sailor API, or pin a build of `_temp_crazy_sailor_m0m1_b` at `fdb19c9a` or later (the `v0.5.9` tag lacks the Sailor API) | FXS |
| D2 | E1 verdict path for "oscillating in 4 of 5, rho 4 silent" (C7). E1b verdict use | LANSA E2 |
| D3 | LANSA C1 to C4, C6 (SST/PV source, module count, unit count, deep rho, run length) | LANSA E2 seal |
| D4 | Firewall ordering: blind OXM after the substrate freeze, or a separate lineage | OXM |
| D5 | Name: "fixationboat" (README) or "Fixation Sailor" (owner). One name everywhere | README, dashboard |

## Track A: LANSA minimal complete

Goal: the lite model is built, simulated, and gives a laminar E-I result that passes its own verdict rule.

| Step | Output | Closing check |
|---|---|---|
| A1 | Lite spec: V1 and FEF, 3 layers, E and I, 100 neurons (demo). Seal it. Visualization V0 schematic and V1 raster per `visualization_contract.json` | sealed spec at a commit; V0 and V1 paths resolve at that commit |
| A2 | Column composition from `lansa_density_X.json` for V1 and FEF: layer boundaries from the depth bins, E/I from NG and PV+CB+CR. Written as data, not assumed | per-layer counts reproduce from the JSON with a test |
| A3 | Run the 100-neuron lite (E2 as sealed). V2 raster, PSD, layer power profile | `results/` receipt with verdict; `e2_verdict` covers every branch (D2) |
| A4 | Scale to 1000 per area. Same spec, one principal delta (N) | receipt; drive-additivity test |
| A5 | V3 atlas. Owner review of the lite result | atlas path resolves; lineage record with verdict |

Minimal complete = A5 passes. Until D3 is ruled, A1 cannot seal.

## Track B: Fixation Sailor minimal complete (parallel with A)

Goal: a boat controlled by FEF deep E neurons fixates on the origin under the A/B/R wave sequence, and the controller population is compared.

| Step | Output | Closing check |
|---|---|---|
| B1 | Engine decision (D1). If a Sailor-API build is pinned, the pin and `current_state.json` move together | provenance test passes on the chosen pin |
| B2 | Boat dynamics and decoder in `jomission/` (not in the lab HTML). Directional motor pools per option 1, or a restoring term per option 2 (owner choice) | unit test: with zero input, a displaced boat returns toward the origin (null: the lab's current wiring must fail this test) |
| B3 | Minimal model V1-V4-FEF, 3 areas at 1000 each (README), or the lab's 540 at first with the same rule | builds and simulates finitely (cf. `test_lite_model.py`) |
| B4 | Controller comparison: E vs I vs layers, on A, B and R waves. Estimand declared first | sealed spec; receipt per controller |
| B5 | V0 to V3 for the chosen controller | visualization paths resolve |

Minimal complete = B2 null test passes and B4 yields a ranked controller with the declared estimand.

## Track C: OXM (after A and B are minimal complete)

Goal: omission on the boat task, with the decision made by two FEF groups.

| Step | Output | Closing check |
|---|---|---|
| C1 | OXM spec sealed: omitted event (from `paradigm/spec.py`), baseline, windows, readout groups, threshold, decision window, null (D4 first) | sealed spec before any run |
| C2 | Input mapping: distance-to-drive for A into L4 of V1 and V4, spatial profile | unit test: drive monotone in distance; matches the spec |
| C3 | Readout: left-right and up-down groups, decision from spike counts | unit test on a known positive (forced left) and a known negative (balanced) |
| C4 | Run conditions from `paradigm/conditions.py`, omission at p2, p3, p4 | receipt; firewall check: no omission output in substrate construction |
| C5 | Null: shuffled wave order and surrogate counts. Report effect size, n, test | null distribution in receipt |

## Gates and rules that apply throughout

- Seal every spec before the run that uses it (`AGENTS.md`).
- Visualization V0 to V3 for every new executed lineage (`visualization_contract.json`). The E1b case shows this is the step most likely to block a merge; plan for it.
- Pins match the executed engine (E0, E1, E1b ran on 0.5.9; the pin says 0.5.8).
- Push temp branches after each commit; push `dev` only with gate 0 failed.
- No merge into `dev` or `main` for Plasticity Boat before its independent review (D0).

## Not in this plan

Plasticity Boat (separate programme). Paper revision (separate). Mouse SST/PV data acquisition (waits on the source).
